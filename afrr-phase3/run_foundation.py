"""
run_foundation.py — run the foundation-model configurations and evaluate them.

  1. for each configuration (default: all nine, foundation.CONFIGS), unless it
     is already complete (results/runtime/{name}.json and its four prediction
     tables exist): python foundation.py --model M --covariates C
     --batch B --threads 8, one after the other. foundation.py resumes from
     data/fm_cache/{name}.npz, so an interrupted run continues where it stopped.
  2. evaluation of every complete configuration (and the baselines):
       results/foundation.csv   metrics.evaluate, same columns as baselines.csv
                                (incl. covariate_known strata) + quantile_note
       results/all.csv          baselines.csv + foundation.csv
       results/common9.csv      secondary table, all models with quantiles, on
                                the 9 levels every model has (q10 .. q90):
                                crps9 = 2 x mean pinball over the 9 levels,
                                pinball_q10..q90, mae_median, cov_10_90; same
                                day-block bootstrap (1000, seed 0) and strata
       results/runtime.csv      seconds per call / per origin, peak memory
     quantile_note marks TimesFM: it has only the deciles, so q01 = q05 = q10,
     q95 = q99 = q90 and q15 .. q85 are interpolated linearly between deciles.

Batch sizes (BATCH): origins per forecast call. Batched calls must equal single
calls to within 1e-4 (foundation.py --batch-check, 20 origins). Only TimesFM
passes (max |dq| = 0 at batch 32); Timestral and Chronos-2 differ by up to
2.6e-3 and stay at one origin per call.

Usage (from the repo root, env afrr3-fm):
  python afrr-phase3/run_foundation.py [--configs chronos2_none timesfm_none ...] [--eval-only]
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as c  # noqa: E402
import foundation as fm  # noqa: E402
import metrics as M  # noqa: E402

BATCH = {'timesfm_none': 32}            # default 1
THREADS = 8
TIMESFM_NOTE = 'timesfm_deciles: q01=q05=q10, q95=q99=q90, q15..q85 interpolated'
BASELINE_QMODELS = ['persistence', 'seasonal_persistence', 'climatology', 'xgb_quantile']
ALL_NAMES = [fm.name_of(m, k) for m, k in fm.CONFIGS]


def complete(name):
    paths = [c.pred_path(name, h, p) for h in c.HORIZONS for p in c.TEST_PERIODS]
    return os.path.exists(os.path.join(fm.RUNTIME_DIR, f'{name}.json')) and all(map(os.path.exists, paths))


def run_config(name):
    model, cov = name.split('_', 1)
    cmd = [sys.executable, os.path.join(HERE, 'foundation.py'), '--model', model, '--covariates', cov,
           '--batch', str(BATCH.get(name, 1)), '--threads', str(THREADS)]
    print(f'\n[run] {name}: {" ".join(cmd[1:])}', flush=True)
    t = time.time()
    subprocess.run(cmd, check=True, env={**os.environ, 'HF_HUB_OFFLINE': '1'})
    print(f'[run] {name} finished in {(time.time() - t) / 3600:.2f} h', flush=True)


def load_preds(names):
    return pd.concat([pd.read_parquet(c.pred_path(n, h, p)) for n in names
                      for h in c.HORIZONS for p in c.TEST_PERIODS], ignore_index=True)


def note_of(model):
    return TIMESFM_NOTE if str(model).startswith('timesfm') else ''


# ── secondary table on the 9 common levels ──────────────────────────────────

COMMON9 = fm.COMMON9
TAU9 = np.array([c.QUANTILES[c.QCOLS.index(q)] for q in COMMON9])


def _common9_one(g, n_boot=M.N_BOOT, seed=M.SEED):
    """Same rows, day blocks and resamples as metrics.evaluate_one."""
    g = g[g.y.notna() & g.threshold.notna()].sort_values('origin_utc')
    y = g.y.to_numpy()
    Q = g[COMMON9].to_numpy()
    pin = M.pinball(Q, y[:, None], TAU9[None, :])
    loss = {'crps9': 2 * pin.mean(axis=1),
            'mae_median': np.abs(y - g['q50'].to_numpy()),
            'cov_10_90': ((y >= g['q10'].to_numpy()) & (y <= g['q90'].to_numpy())).astype(float),
            **{f'pinball_{q}': pin[:, j] for j, q in enumerate(COMMON9)}}
    day = g.origin_utc.dt.tz_convert(c.LOCAL_TZ).dt.date.to_numpy()
    days, day_idx = np.unique(day, return_inverse=True)
    D = len(days)
    W = np.random.default_rng(seed).multinomial(D, np.full(D, 1 / D), size=n_boot).astype(float)
    den = W @ np.bincount(day_idx, minlength=D).astype(float)
    rows = []
    for k, v in loss.items():
        vb = W @ np.bincount(day_idx, weights=v, minlength=D) / den
        lo, hi = np.percentile(vb, [2.5, 97.5])
        rows.append({'metric': k, 'value': float(v.mean()), 'ci_low': float(lo), 'ci_high': float(hi)})
    out = pd.DataFrame(rows)
    out['n_origins'], out['n_days'] = len(g), D
    return out


def evaluate_common9(preds):
    preds = c.validate_pred_table(preds)
    preds = preds.assign(_cov=M.covariate_known(preds))
    parts = []
    for (m, h, per), g in preds.groupby(['model', 'horizon', 'period'], sort=True):
        subsets = [('all', g)]
        if g._cov.any() and not g._cov.all():
            subsets += [('known', g[g._cov]), ('unknown', g[~g._cov])]
        for tag, gs in subsets:
            r = _common9_one(gs)
            r.insert(0, 'covariate_known', tag)
            r.insert(0, 'period', per)
            r.insert(0, 'horizon', h)
            r.insert(0, 'model', m)
            parts.append(r)
    out = pd.concat(parts, ignore_index=True)
    out['quantile_note'] = out.model.map(note_of)
    return out.sort_values(['period', 'horizon', 'covariate_known', 'model']).reset_index(drop=True)


# ── evaluation ──────────────────────────────────────────────────────────────

def evaluate(names):
    train_rate = json.load(open(os.path.join(c.DATA, 'labels_meta.json')))['train_spike_rate']
    print(f'\n[eval] {len(names)} foundation configurations: {", ".join(names)}', flush=True)
    res = M.evaluate(load_preds(names), train_rate, stratify_covariate=True)
    res['quantile_note'] = res.model.map(note_of)
    res.to_csv(os.path.join(c.RESULTS_DIR, 'foundation.csv'), index=False)

    base = pd.read_csv(os.path.join(c.RESULTS_DIR, 'baselines.csv'))
    base['quantile_note'] = ''
    allr = pd.concat([base, res], ignore_index=True)
    allr.to_csv(os.path.join(c.RESULTS_DIR, 'all.csv'), index=False)

    c9 = evaluate_common9(pd.concat([load_preds(BASELINE_QMODELS), load_preds(names)], ignore_index=True))
    c9.to_csv(os.path.join(c.RESULTS_DIR, 'common9.csv'), index=False)

    rt = []
    for n in names:
        r = json.load(open(os.path.join(fm.RUNTIME_DIR, f'{n}.json')))
        r.pop('versions', None)
        rt.append(r)
    rt = pd.DataFrame(rt)
    rt.to_csv(os.path.join(c.RESULTS_DIR, 'runtime.csv'), index=False)

    summ = M.summary_table(res)
    with pd.option_context('display.width', 400, 'display.max_columns', None, 'display.max_colwidth', 30):
        print(summ.to_string())
        print(rt[['model', 'batch_size', 'sec_per_call_mean', 'sec_per_origin', 'wall_seconds',
                  'peak_rss_mb']].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='+', default=ALL_NAMES, choices=ALL_NAMES)
    ap.add_argument('--eval-only', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    print(f'[start] {time.strftime("%Y-%m-%d %H:%M:%S")}  configs: {" ".join(a.configs)}', flush=True)
    if not a.eval_only:
        for n in a.configs:
            if complete(n):
                print(f'[skip] {n}: already complete', flush=True)
                continue
            run_config(n)
    done = [n for n in ALL_NAMES if complete(n)]
    evaluate(done)
    print(f'\n[done] {time.strftime("%Y-%m-%d %H:%M:%S")}  total {(time.time() - t0) / 3600:.2f} h', flush=True)


if __name__ == '__main__':
    main()
