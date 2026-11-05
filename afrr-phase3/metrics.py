"""
metrics.py — evaluation for every phase-3 model (see TASK.md).

evaluate(preds, train_spike_rate) takes a prediction table with the fixed
schema (common.PRED_SCHEMA; one or many models / horizons / periods) and
returns one row per (model, horizon, period, metric) with a point estimate and
a 95 % block-bootstrap CI.

Rows used: y and threshold not NaN (threshold NaN = label warm-up, excluded).
Label: spike = y > threshold.

Metrics
  spike:   pr_auc (average precision), roc_auc, brier (true Brier on
           spike_prob, no rescaling), brier_ref (constant train spike rate),
           brier_skill = 1 - brier / brier_ref
  price:   mae_median (|y - q50|), crps (2 x mean pinball over the 21
           levels; for a point forecast this equals the MAE), pinball_qXX
  calib.:  cov_10_90 = P(q10 <= y <= q90), cov_05_95 = P(q05 <= y <= q95)
Price metrics are NaN for models without quantiles.

Optional stratification (stratify_covariate=True): for horizons where some
origins have the T+h known-day-ahead values and some do not (h=96: T before /
from 13:00 local), metrics are also reported per stratum; column
covariate_known = all | known | unknown.

Uncertainty: block bootstrap over days. A block is one Swiss local calendar
day of the origin. Each resample draws n_days days with replacement; every
row gets the multiplicity of its day as a weight. 1000 resamples, fixed seed,
so all models of the same (period, horizon) and origin set see the same
resamples (paired). CI = 2.5 / 97.5 percentiles.

Also: prob_exceed(), the quantile -> spike probability rule for
quantile-only models.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as c  # noqa: E402

N_BOOT = 1000
SEED = 0
P_MIN, P_MAX = c.QUANTILES[0], c.QUANTILES[-1]   # clamp to [0.01, 0.99]


def pinball(q, y, tau):
    d = y - q
    return np.maximum(tau * d, (tau - 1) * d)


def prob_exceed(qvals, thr):
    """P(price > thr) from 21 sorted quantiles by linear interpolation of the
    CDF through the points (q_tau, tau). Ties are handled (the CDF is right-
    continuous: F(thr) uses all quantiles <= thr). Beyond the outer quantiles
    the probability is clamped: thr < q01 -> 0.99, thr >= q99 -> 0.01.
    NaN where thr or any quantile is NaN."""
    q = np.asarray(qvals, float)
    thr = np.asarray(thr, float)
    tau = c.QUANTILES
    out = np.full(len(q), np.nan)
    ok = ~np.isnan(thr) & ~np.isnan(q).any(axis=1)
    qo, to = q[ok], thr[ok]
    k = (qo <= to[:, None]).sum(axis=1)               # number of quantiles <= thr
    F = np.empty(len(qo))
    F[k == 0] = tau[0]
    F[k == len(tau)] = tau[-1]
    mid = (k > 0) & (k < len(tau))
    r = np.arange(len(qo))[mid]
    km = k[mid]
    q_lo, q_hi = qo[r, km - 1], qo[r, km]
    t_lo, t_hi = tau[km - 1], tau[km]
    F[mid] = t_lo + (t_hi - t_lo) * (to[mid] - q_lo) / (q_hi - q_lo)
    out[ok] = np.clip(1 - F, P_MIN, P_MAX)
    return out


def _row_losses(g, train_rate):
    y = g.y.to_numpy()
    lab = (y > g.threshold.to_numpy()).astype(float)
    p = g.spike_prob.to_numpy()
    Q = g[c.QCOLS].to_numpy()
    add = {'brier': (p - lab) ** 2, 'brier_ref': (train_rate - lab) ** 2,
           'spike_rate': lab}
    if not np.isnan(Q).any():
        pin = pinball(Q, y[:, None], c.QUANTILES[None, :])
        add['mae_median'] = np.abs(y - Q[:, c.QCOLS.index('q50')])
        add['crps'] = 2 * pin.mean(axis=1)
        for j, col in enumerate(c.QCOLS):
            add[f'pinball_{col}'] = pin[:, j]
        add['cov_10_90'] = ((y >= Q[:, c.QCOLS.index('q10')]) & (y <= Q[:, c.QCOLS.index('q90')])).astype(float)
        add['cov_05_95'] = ((y >= Q[:, c.QCOLS.index('q05')]) & (y <= Q[:, c.QCOLS.index('q95')])).astype(float)
    return lab, p, add


def _rank_metrics(lab, p, w=None):
    if np.isnan(p).any():
        return np.nan, np.nan
    pos = (lab * (w if w is not None else 1)).sum()
    tot = (w if w is not None else np.ones_like(lab)).sum()
    if pos == 0 or pos == tot:
        return np.nan, np.nan
    return (average_precision_score(lab, p, sample_weight=w),
            roc_auc_score(lab, p, sample_weight=w))


def evaluate_one(g, train_rate, n_boot=N_BOOT, seed=SEED):
    g = g[g.y.notna() & g.threshold.notna()].sort_values('origin_utc')
    lab, p, add = _row_losses(g, train_rate)
    day = g.origin_utc.dt.tz_convert(c.LOCAL_TZ).dt.date.to_numpy()
    days, day_idx = np.unique(day, return_inverse=True)
    D = len(days)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(D, np.full(D, 1 / D), size=n_boot).astype(float)   # (B, D)
    N_d = np.bincount(day_idx, minlength=D).astype(float)
    den = W @ N_d

    res = {}
    for k, v in add.items():
        S_d = np.bincount(day_idx, weights=v, minlength=D)
        res[k] = (v.mean(), W @ S_d / den)
    ref = res['brier_ref']
    res['brier_skill'] = (1 - res['brier'][0] / ref[0], 1 - res['brier'][1] / ref[1])

    ap0, auc0 = _rank_metrics(lab, p)
    ap_b, auc_b = np.full(n_boot, np.nan), np.full(n_boot, np.nan)
    if not np.isnan(ap0):
        for b in range(n_boot):
            ap_b[b], auc_b[b] = _rank_metrics(lab, p, W[b, day_idx])
    res['pr_auc'] = (ap0, ap_b)
    res['roc_auc'] = (auc0, auc_b)

    rows = []
    for k, (v0, vb) in res.items():
        lo, hi = (np.nanpercentile(vb, [2.5, 97.5]) if np.isfinite(vb).any() else (np.nan, np.nan))
        rows.append({'metric': k, 'value': float(v0), 'ci_low': float(lo), 'ci_high': float(hi)})
    out = pd.DataFrame(rows)
    out['n_origins'] = len(g)
    out['n_days'] = D
    out['n_spikes'] = int(lab.sum())
    return out


METRIC_ORDER = (['pr_auc', 'roc_auc', 'brier', 'brier_ref', 'brier_skill', 'spike_rate',
                 'mae_median', 'crps', 'cov_10_90', 'cov_05_95']
                + [f'pinball_{q}' for q in c.QCOLS])


def covariate_known(preds):
    """Boolean per row: are the known-day-ahead values of target_utc available
    at origin_utc (TASK.md rule 2: delivery day <= day of T, or next day and
    T >= 13:00 local)? Same rule as features.da_available."""
    from features import da_available   # local import: features -> labels -> common
    o = preds.origin_utc.dt.tz_convert(c.LOCAL_TZ).dt.tz_localize(None)
    t = preds.target_utc.dt.tz_convert(c.LOCAL_TZ).dt.tz_localize(None)
    return da_available(o, t)


def evaluate(preds, train_spike_rate, n_boot=N_BOOT, seed=SEED, stratify_covariate=False):
    """All metrics per model, horizon and period (long format).

    Column covariate_known: 'all' for the full origin set. With
    stratify_covariate=True, every (model, horizon, period) whose origins
    contain both cases (in practice h=96) is additionally evaluated separately
    on origins where the T+h known-day-ahead values are available ('known') and
    where they are not ('unknown'). Each stratum has its own day-block bootstrap
    (same seed, so paired across models)."""
    preds = c.validate_pred_table(preds)
    if stratify_covariate:
        preds = preds.assign(_cov=covariate_known(preds))
    parts = []
    for (m, h, per), g in preds.groupby(['model', 'horizon', 'period'], sort=True):
        subsets = [('all', g)]
        if stratify_covariate and g._cov.any() and not g._cov.all():
            subsets += [('known', g[g._cov]), ('unknown', g[~g._cov])]
        for tag, gs in subsets:
            r = evaluate_one(gs, train_spike_rate, n_boot, seed)
            r.insert(0, 'covariate_known', tag)
            r.insert(0, 'period', per)
            r.insert(0, 'horizon', h)
            r.insert(0, 'model', m)
            parts.append(r)
    out = pd.concat(parts, ignore_index=True)
    out['metric'] = pd.Categorical(out.metric, METRIC_ORDER, ordered=True)
    out['covariate_known'] = pd.Categorical(out.covariate_known, ['all', 'known', 'unknown'],
                                            ordered=True)
    return (out.sort_values(['period', 'horizon', 'covariate_known', 'model', 'metric'])
               .reset_index(drop=True))


def summary_table(res, metrics=('pr_auc', 'roc_auc', 'brier', 'brier_ref', 'mae_median',
                                'crps', 'cov_10_90', 'cov_05_95')):
    """Wide 'value [lo, hi]' table for printing."""
    r = res[res.metric.isin(metrics)].copy()
    r['txt'] = r.apply(lambda x: 'nan' if np.isnan(x.value) else
                       f'{x.value:.3f} [{x.ci_low:.3f}, {x.ci_high:.3f}]', axis=1)
    idx = ['period', 'horizon'] + (['covariate_known'] if 'covariate_known' in r else []) + ['model']
    return r.pivot_table(index=idx, columns='metric',
                         values='txt', aggfunc='first', observed=True)[list(metrics)]
