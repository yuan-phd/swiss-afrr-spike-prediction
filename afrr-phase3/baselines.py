"""
baselines.py — baseline models for the phase-3 task (see TASK.md).

Each function returns (quantiles (n, 21) or None, spike_prob (n,)) for the
origins in `idx` (positions into the merged dataset) and horizon h.

  persistence           all quantiles = last observed price p(T - 15 min);
                        spike_prob = 1 if p(T - 15 min) > thr_known(T) else 0
  seasonal_persistence  all quantiles = price at the same time of day as the
                        target, on the latest day observed at T:
                        p(T + h - 96 k) with the smallest k >= 1 such that
                        T + h - 96 k <= T - 15 min (h=4: k=1, row T-92;
                        h=96: k=2, row T-96); spike_prob = 1/0 vs thr_known(T)
  climatology           21 quantiles of training-period price at full-hour rows
                        with the same local hour as the target;
                        spike_prob = training spike rate
  xgb_clf               XGBoost binary classifier, isotonic calibration;
                        no price quantiles (all NaN)
  xgb_quantile          XGBoost reg:quantileerror with the 21 levels, sorted;
                        spike_prob = metrics.prob_exceed(quantiles, thr_known(T))
"""
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as c  # noqa: E402
from metrics import prob_exceed  # noqa: E402

XGB_COMMON = dict(n_estimators=4000, learning_rate=0.03, max_depth=6, min_child_weight=5,
                  subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method='hist',
                  early_stopping_rounds=100, n_jobs=8, random_state=0)


def persistence(df, X, idx, h):
    last = X['p_lag1'].to_numpy()[idx]
    q = np.repeat(last[:, None], len(c.QCOLS), axis=1)
    prob = (last > X['thr_known'].to_numpy()[idx]).astype(float)
    prob[np.isnan(X['thr_known'].to_numpy()[idx])] = np.nan
    return q, prob


def seasonal_offset(h):
    k = 1
    while h - 96 * k > -1:
        k += 1
    return 96 * k - h        # steps back from T


def seasonal_persistence(df, X, idx, h):
    back = seasonal_offset(h)
    p = df[c.TARGET].to_numpy()
    val = p[idx - back]
    q = np.repeat(val[:, None], len(c.QCOLS), axis=1)
    thr = X['thr_known'].to_numpy()[idx]
    prob = np.where(np.isnan(thr), np.nan, (val > thr).astype(float))
    return q, prob


def fit_climatology(df):
    a, b = c.period_bounds_utc('train')
    ts = df.timestamp_utc
    m = (ts >= a) & (ts < b) & (df.timestamp_local.dt.minute == 0)
    hours = df.timestamp_local[m].dt.hour
    price = df.loc[m, c.TARGET]
    table = price.groupby(hours).quantile(c.QUANTILES).unstack()
    table.columns = c.QCOLS
    return table          # index: local hour 0..23


def climatology(df, X, idx, h, table, train_rate):
    hour = X['cal_hour'].to_numpy()[idx].astype(int)
    q = table.loc[hour].to_numpy()
    return q, np.full(len(idx), train_rate)


def train_parts(df, lab, h):
    """Origin positions for fit / early-stop / calibrate parts of the training
    period. Part [lo, hi): origin >= lo and target < hi, so consecutive parts
    are separated by a purge gap of h steps. Rows whose label is NaN (90-day
    warm-up) are excluded."""
    b0, b1, b2, b3 = c.train_split_bounds()
    ts = pd.DatetimeIndex(df.timestamp_utc)
    i = np.arange(len(df) - h)
    org, tgt = ts[i], ts[i + h]
    valid = (org.minute == 0) & ~np.isnan(lab.threshold.to_numpy()[i + h])
    parts = {}
    for name, lo, hi in [('fit', b0, b1), ('es', b1, b2), ('cal', b2, b3)]:
        parts[name] = i[valid & (org >= lo) & (tgt < hi)]
    return parts


def fit_xgb_clf(X, y_spike, parts):
    Xf, Xe, Xc = (X.iloc[parts[k]] for k in ('fit', 'es', 'cal'))
    yf, ye, yc = (y_spike[parts[k]] for k in ('fit', 'es', 'cal'))
    clf = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', **XGB_COMMON)
    clf.fit(Xf, yf, eval_set=[(Xe, ye)], verbose=False)
    raw_cal = clf.predict_proba(Xc)[:, 1]
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip').fit(raw_cal, yc)
    return clf, iso


def predict_xgb_clf(model, X, idx):
    clf, iso = model
    return None, iso.predict(clf.predict_proba(X.iloc[idx])[:, 1])


def fit_xgb_quantile(X, y_price, parts):
    Xf, Xe = X.iloc[parts['fit']], X.iloc[parts['es']]
    reg = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=c.QUANTILES,
                           **XGB_COMMON)
    reg.fit(Xf, y_price[parts['fit']], eval_set=[(Xe, y_price[parts['es']])], verbose=False)
    return reg


def predict_xgb_quantile(reg, X, idx):
    q = np.sort(reg.predict(X.iloc[idx]), axis=1)
    return q, prob_exceed(q, X['thr_known'].to_numpy()[idx])
