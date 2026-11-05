"""
run_baselines.py — rebuild labels, baseline predictions and metrics.

  1. labels.py               -> data/labels.parquet, data/labels_meta.json
  2. check_leakage.py        -> asserts no feature uses data after T (with self-test)
  3. features + baselines    -> predictions/{model}_h{h}_{period}.parquet
  4. metrics.evaluate        -> results/baselines.csv (long: value + 95 % CI; column
                                covariate_known = all | known | unknown, strata for h=96)
                                results/baselines_summary.csv (wide, main metrics)
                                results/task_stats.json (origin counts, DA NaN shares,
                                training split, XGBoost iterations)

Usage (from the repo root):  python afrr-phase3/run_baselines.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import baselines as B  # noqa: E402
import check_leakage  # noqa: E402
import common as c  # noqa: E402
import features as F  # noqa: E402
import labels  # noqa: E402
import metrics as M  # noqa: E402

MODELS = ['persistence', 'seasonal_persistence', 'climatology', 'xgb_clf', 'xgb_quantile']


def main():
    t0 = time.time()
    print('[1] labels'); labels.main()
    meta = json.load(open(os.path.join(c.DATA, 'labels_meta.json')))
    train_rate = meta['train_spike_rate']

    print('\n[2] leakage check'); check_leakage.self_test(); check_leakage.main()

    df = pd.read_parquet(c.MERGED)
    lab = pd.read_parquet(c.LABELS)
    assert (pd.DatetimeIndex(lab.timestamp_utc) == pd.DatetimeIndex(df.timestamp_utc)).all()
    thr = lab.threshold.to_numpy()
    price = df[c.TARGET].to_numpy()
    ts = pd.DatetimeIndex(df.timestamp_utc)
    clim = B.fit_climatology(df)
    stats = {'train_spike_rate': train_rate, 'split_bounds_utc': [str(b) for b in c.train_split_bounds()],
             'horizons': {}}
    all_preds = []

    for h in c.HORIZONS:
        print(f'\n[3] horizon h={h}')
        X = F.build_features(df, h)
        y_spike = np.full(len(df), np.nan)
        y_spike[:len(df) - h] = (price[h:] > thr[h:]).astype(float)
        y_spike[:len(df) - h][np.isnan(thr[h:])] = np.nan
        y_price = np.full(len(df), np.nan)
        y_price[:len(df) - h] = price[h:]

        parts = B.train_parts(df, lab, h)
        hs = {'train_parts': {k: {'n': int(len(v)), 'first_origin': str(ts[v[0]]),
                                  'last_target': str(ts[v[-1] + h]),
                                  'spike_rate': float(np.nanmean(y_spike[v]))}
                              for k, v in parts.items()}}
        t = time.time()
        clf = B.fit_xgb_clf(X, y_spike, parts)
        reg = B.fit_xgb_quantile(X, y_price, parts)
        hs['xgb_clf_best_iteration'] = int(clf[0].best_iteration)
        hs['xgb_quantile_best_iteration'] = int(reg.best_iteration)
        print(f'    XGBoost fitted in {time.time() - t:.0f}s  (clf best_iteration={clf[0].best_iteration}, '
              f'quantile best_iteration={reg.best_iteration})')

        hs['periods'] = {}
        for per in c.TEST_PERIODS:
            idx = F.select_origins(df, h, per)
            da_nan = float(np.isnan(X['DA_Price_CH_tph'].to_numpy()[idx]).mean())
            hs['periods'][per] = {'n_origins': int(len(idx)), 'da_tph_nan_fraction': da_nan,
                                  'n_label_nan': int(np.isnan(thr[idx + h]).sum())}
            out = {
                'persistence': B.persistence(df, X, idx, h),
                'seasonal_persistence': B.seasonal_persistence(df, X, idx, h),
                'climatology': B.climatology(df, X, idx, h, clim, train_rate),
                'xgb_clf': B.predict_xgb_clf(clf, X, idx),
                'xgb_quantile': B.predict_xgb_quantile(reg, X, idx),
            }
            for m in MODELS:
                q, p = out[m]
                tab = c.make_pred_table(m, per, h, ts[idx], ts[idx + h], price[idx + h],
                                        thr[idx + h], q, p)
                path = c.save_pred_table(tab)
                all_preds.append(tab)
            print(f'    {per}: {len(idx):,} origins, DA(T+h) NaN share {da_nan:.4f}, saved {len(MODELS)} tables')
        # DA NaN share over training origins too
        tr = np.concatenate(list(parts.values()))
        hs['train_da_tph_nan_fraction'] = float(np.isnan(X['DA_Price_CH_tph'].to_numpy()[tr]).mean())
        stats['horizons'][str(h)] = hs

    print('\n[4] metrics (block bootstrap, 1000 resamples)')
    preds = pd.concat(all_preds, ignore_index=True)
    res = M.evaluate(preds, train_rate, stratify_covariate=True)
    os.makedirs(c.RESULTS_DIR, exist_ok=True)
    res.to_csv(os.path.join(c.RESULTS_DIR, 'baselines.csv'), index=False)
    summ = M.summary_table(res)
    summ.to_csv(os.path.join(c.RESULTS_DIR, 'baselines_summary.csv'))
    with open(os.path.join(c.RESULTS_DIR, 'task_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    with pd.option_context('display.width', 400, 'display.max_columns', None, 'display.max_colwidth', 30):
        print(summ.to_string())
    print(f'\nDone in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
