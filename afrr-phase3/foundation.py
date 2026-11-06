"""
foundation.py — zero-shot foundation-model forecasts for the phase-3 task
(same task, origins, labels, metrics and prediction schema as Step 2; see
TASK.md).

Models:      timestral (Timestral-209M), chronos2 (Chronos-2), timesfm
             (TimesFM 2.5 200M, no covariates only; runs in the `timesfm`
             env through fm_timesfm_worker.py).
Covariates:  none | past_only | known_future | both.

Shared protocol
  * target pos_sec_price, univariate; context = the CONTEXT = 2016 rows
    (3 weeks) ending at T - 15 min, i.e. rows i-2016 .. i-1 for origin row i;
  * one forecast per origin, prediction_length PRED_LEN = 97; forecast step
    k (1-based) is row i-1+k, so horizon h (target row i+h) is step
    STEP_OF_H[h] = h + 1 (steps 5 and 97);
  * --batch B puts B origins in one forecast call (series in a batch share no
    information). Allowed only where --batch-check shows batched == single
    calls to within 1e-4; in practice TimesFM only (run_foundation.BATCH);
  * past-only covariates PAST_COLS over the context rows only;
  * known-future covariates KNOWN_COLS over context + horizon rows, NaN where
    the value is not yet published at T (features.da_available, 13:00 rule),
    plus categorical CAT_COLS (local hour, day of week, holiday) of each row;
  * spike_prob = metrics.prob_exceed(quantiles, threshold(T)), the same rule
    as the xgb_quantile baseline.

Outputs (model name = f'{model}_{covariates}', e.g. timestral_both)
  predictions/{name}_h{h}_{period}.parquet    Step-2 schema
  results/runtime/{name}.json                 seconds per call / per origin, peak memory
  data/fm_cache/{name}.npz                    resumable forecast cache

Other modes
  --batch-check   20 origins (seed 2): batched calls vs single calls; writes
                  results/runtime/batch_check_{name}.json
  --n-origins N   timing run on N random origins (--seed, default 0); writes
                  results/runtime/{name}_sample.json, no predictions
  --mps-check     20 random origins (seed 1) on CPU and on MPS, each in its
                  own process; writes
                  results/runtime/mps_check_{name}.json (max |dq|)

Usage (from the repo root, env afrr3-fm):
  python afrr-phase3/foundation.py --model timestral --covariates both
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as c  # noqa: E402
import features as F  # noqa: E402
from metrics import prob_exceed  # noqa: E402

CONTEXT = 2016
PRED_LEN = 97
STEP_OF_H = {4: 5, 96: 97}               # 1-based forecast step of target row i+h
assert all(STEP_OF_H[h] == h + 1 <= PRED_LEN for h in c.HORIZONS)

PAST_COLS = ['DE_WindSolar_Error', 'UF_DE', 'UF_FR', 'pos_sec_vol_mw', 'neg_sec_vol_mw', 'CH_Pump_Gen']
KNOWN_COLS = ['Sched_DE_CH', 'Sched_CH_IT', 'Sched_FR_CH', 'Sched_IT_CH', 'DA_Price_CH', 'DA_Price_DE',
              'CH_Load_Forecast']
assert set(KNOWN_COLS) <= set(c.KNOWN_DA_COLS)
CAT_COLS = ['cal_hour', 'cal_dow', 'cal_holiday']
MODELS = ('timestral', 'chronos2', 'timesfm')
COVS = ('none', 'past_only', 'known_future', 'both')
CONFIGS = [(m, k) for m in ('timestral', 'chronos2') for k in COVS] + [('timesfm', 'none')]

TIMESTRAL_REPO = 'mistralai/Timestral-209M'
CHRONOS_REPO = 'amazon/chronos-2'
TIMESFM_PY = '/opt/homebrew/Caskroom/miniforge/base/envs/timesfm/bin/python'
TIMESFM_LEVELS = np.round(np.arange(1, 10) / 10, 2)          # 0.1 .. 0.9
COMMON9 = [f'q{int(round(q * 100)):02d}' for q in TIMESFM_LEVELS]
CACHE_DIR = os.path.join(c.DATA, 'fm_cache')
RUNTIME_DIR = os.path.join(c.RESULTS_DIR, 'runtime')
THREADS = 8


def name_of(model, cov):
    return f'{model}_{cov}'


def uses_past(cov):
    return cov in ('past_only', 'both')


def uses_known(cov):
    return cov in ('known_future', 'both')


# ── inputs ──────────────────────────────────────────────────────────────────

def past_frame(df):
    """The past-only covariates; UF_DE / UF_FR (unplanned flows) are derived
    exactly as in features.build_features (TASK.md feature table)."""
    return pd.DataFrame({
        'DE_WindSolar_Error': df.DE_WindSolar_Error,
        'UF_DE': (df.DE_CH_mw - df.CH_DE_mw) - df.Sched_DE_CH,
        'UF_FR': (df.FR_CH_mw - df.CH_FR_mw) - df.Sched_FR_CH,
        'pos_sec_vol_mw': df.pos_sec_vol_mw,
        'neg_sec_vol_mw': df.neg_sec_vol_mw,
        'CH_Pump_Gen': df.CH_Pump_Gen,
    })[PAST_COLS].astype(float)


class Data:
    """Arrays of one (possibly masked) copy of the merged dataset."""

    def __init__(self, df):
        F.check_grid(df)
        self.n = len(df)
        self.price = df[c.TARGET].to_numpy(float)
        self.past = past_frame(df).to_numpy(float)
        self.known = df[KNOWN_COLS].to_numpy(float)
        self.ts_utc = pd.DatetimeIndex(df.timestamp_utc)
        self.local = pd.DatetimeIndex(df.timestamp_local)
        self.cat = np.column_stack([self.local.hour, self.local.dayofweek,
                                    pd.Index(self.local.date).isin(F.HOLIDAY_DATES)]).astype(float)

    def _rows(self, a, b, arr):
        """arr[a:b] padded with NaN past the end of the data."""
        out = np.full((b - a,) + arr.shape[1:], np.nan)
        hi = min(b, self.n)
        out[:hi - a] = arr[a:hi]
        return out

    def inputs(self, i, cov):
        """Model inputs for origin row i. Shapes: context (T,), past (6, T),
        known (7, T+P), cat (3, T+P), with T = CONTEXT, P = PRED_LEN."""
        a, b = i - CONTEXT, i + PRED_LEN
        assert a >= 0
        inp = {'context': self.price[a:i].copy()}
        if uses_past(cov):
            inp['past'] = self.past[a:i].T.copy()
        if uses_known(cov):
            kn = self._rows(a, b, self.known)
            hi = min(b, self.n)
            avail = np.zeros(b - a, bool)                  # rows past the data end: NaN
            avail[:hi - a] = F.da_available(self.local[[i]].repeat(hi - a), self.local[a:hi])
            kn[~avail] = np.nan
            inp['known'] = kn.T
            inp['cat'] = self._rows(a, b, self.cat).T
        inp['ts_utc'] = pd.date_range(self.ts_utc[a], periods=CONTEXT + PRED_LEN, freq=c.STEP)
        return inp


def check_inputs(df, cand, cov):
    """Leakage check for the foundation-model inputs: inputs built from a copy
    of the data with all post-T observed values and all unpublished day-ahead
    values masked (check_leakage.masked_slice) must equal the inputs built
    from the full data."""
    import check_leakage
    full = Data(df)
    for i in cand:
        s, j = check_leakage.masked_slice(df, int(i), PRED_LEN)
        a = full.inputs(int(i), cov)
        b = Data(s.reset_index(drop=True)).inputs(j, cov)
        for k in a:
            if k == 'ts_utc':
                assert (a[k] == b[k]).all()
            else:
                assert a[k].shape == b[k].shape, (k, a[k].shape, b[k].shape)
                assert np.array_equal(a[k], b[k], equal_nan=True), f'leak in {k} at origin {i}'
        # every input value beyond T - 15 min that is not a published DA / calendar value is absent
        assert len(a['context']) == CONTEXT and not np.isnan(a['context']).any()
    return len(cand)


def test_origins(df):
    """Union of the Step-2 test origins over horizons and periods (positions)."""
    return np.unique(np.concatenate([F.select_origins(df, h, p) for h in c.HORIZONS for p in c.TEST_PERIODS]))


# ── models ──────────────────────────────────────────────────────────────────

class TimestralRunner:
    def __init__(self, device):
        import torch
        from timestral import TimestralForecaster
        torch.set_num_threads(THREADS)
        self.f = TimestralForecaster.from_pretrained(TIMESTRAL_REPO, device=device)
        assert np.allclose(self.f.quantile_levels, c.QUANTILES)
        self.device = device

    def __call__(self, inps):
        """One forecast call for a batch of origins -> (B, P, 21). Series in a
        batch do not share information (each is its own group)."""
        kw = {}
        if 'past' in inps[0]:
            kw['past_covariates'] = np.stack([x['past'] for x in inps])                 # (B, 6, T)
        if 'known' in inps[0]:
            kw['known_covariates'] = np.stack([x['known'] for x in inps])               # (B, 7, T+P)
            kw['categorical_known_covariates'] = np.stack([x['cat'] for x in inps])     # (B, 3, T+P)
        ctx = np.stack([x['context'] for x in inps])[:, None]                           # (B, 1, T)
        fc = self.f.forecast(ctx, prediction_length=PRED_LEN, **kw)
        return fc.quantiles[:, 0].cpu().numpy()                                         # (B, P, 21)


class Chronos2Runner:
    def __init__(self, device):
        import torch
        from chronos import Chronos2Pipeline
        torch.set_num_threads(THREADS)
        self.p = Chronos2Pipeline.from_pretrained(CHRONOS_REPO, device_map=device)
        assert np.allclose(self.p.quantiles, c.QUANTILES)
        self.levels = [float(q) for q in c.QUANTILES]
        self.device = device

    @staticmethod
    def frames(inp, item):
        """(df, future_df) of one origin in the long format Chronos2Pipeline.predict_df
        expects. Timestamps are naive UTC (a regular 15-min grid across DST changes).
        Categoricals are strings so that Chronos treats them as categorical."""
        ts = inp['ts_utc'].tz_localize(None)
        df = pd.DataFrame({'item_id': item, 'timestamp': ts[:CONTEXT], 'target': inp['context']})
        if 'past' in inp:
            for k, col in enumerate(PAST_COLS):
                df[col] = inp['past'][k]
        fut = None
        if 'known' in inp:
            fut = pd.DataFrame({'item_id': item, 'timestamp': ts[CONTEXT:]})
            for k, col in enumerate(KNOWN_COLS):
                df[col] = inp['known'][k, :CONTEXT]
                fut[col] = inp['known'][k, CONTEXT:]
            for k, col in enumerate(CAT_COLS):
                lab = np.array([None if np.isnan(v) else f'{col}_{int(v)}' for v in inp['cat'][k]],
                               dtype=object)
                df[col] = pd.Series(lab[:CONTEXT], dtype='object')
                fut[col] = pd.Series(lab[CONTEXT:], dtype='object')
        return df, fut

    def __call__(self, inps):
        """One predict_df call for a batch of origins (one item_id each) -> (B, P, 21).
        cross_learning=False (default): items do not share information. batch_size
        is set so that the whole call is one model batch."""
        items = [f's{k:05d}' for k in range(len(inps))]       # sorted order = input order
        fr = [self.frames(x, it) for x, it in zip(inps, items)]
        df = pd.concat([f[0] for f in fr], ignore_index=True)
        fut = pd.concat([f[1] for f in fr], ignore_index=True) if fr[0][1] is not None else None
        rows = 1 + len(df.columns) - 3                         # target + covariates per item
        out = self.p.predict_df(df, future_df=fut, id_column='item_id', timestamp_column='timestamp',
                                target='target', prediction_length=PRED_LEN, quantile_levels=self.levels,
                                batch_size=rows * len(inps), cross_learning=False)
        assert list(pd.unique(out.item_id)) == items
        return out[[str(q) for q in self.levels]].to_numpy(float).reshape(len(inps), PRED_LEN, 21)


def timesfm_to_21(q9):
    """TimesFM's 9 levels (0.1..0.9) -> the 21-level table. Tails: q01 = q05 =
    q10 and q95 = q99 = q90. Levels between deciles (q15, ..., q85): linear
    interpolation of the quantile function between the two neighbouring deciles."""
    q9 = np.sort(np.asarray(q9, float), axis=-1)
    out = np.empty(q9.shape[:-1] + (21,))
    for j, tau in enumerate(c.QUANTILES):
        t = min(max(tau, 0.1), 0.9)
        out[..., j] = np.array([np.interp(t, TIMESFM_LEVELS, r) for r in q9.reshape(-1, 9)]).reshape(q9.shape[:-1])
    return out


def run_timesfm(contexts, device, batch):
    """One worker process for a segment of origins; `batch` origins per forecast call."""
    with tempfile.TemporaryDirectory() as tmp:
        inp, outp = os.path.join(tmp, 'in.npz'), os.path.join(tmp, 'out.npz')
        np.savez(inp, context=np.asarray(contexts, float), pred_len=PRED_LEN)
        subprocess.run([TIMESFM_PY, '-I', os.path.join(HERE, 'fm_timesfm_worker.py'), inp, outp,
                        '--device', device, '--threads', str(THREADS), '--batch', str(batch)], check=True,
                       env={**os.environ, 'PYTHONWARNINGS': 'ignore'})
        r = np.load(outp)
        return {k: r[k] for k in r.files}


def make_runner(model, device):
    return {'timestral': TimestralRunner, 'chronos2': Chronos2Runner}[model](device)


def mps_peak():
    try:
        import torch
        return int(torch.mps.driver_allocated_memory()) if torch.backends.mps.is_available() else 0
    except Exception:
        return 0


def forecast_many(model, cov, data, origins, device, batch=1, cache=None, segment=500):
    """Forecasts (len(origins), PRED_LEN, 21), seconds per origin (call time /
    origins in the call) and seconds per call. Work is done in segments of
    `segment` origins; with `cache` (path) progress is saved after every
    segment and resumed on restart (same origins required)."""
    n = len(origins)
    Q = np.full((n, PRED_LEN, 21), np.nan, np.float32)
    sec = np.full(n, np.nan)
    calls = []
    start = 0
    if cache and os.path.exists(cache):
        z = np.load(cache)
        if np.array_equal(z['origins'], origins) and int(z['batch']) == batch:
            start = int(z['done'])
            Q[:start], sec[:start], calls = z['Q'][:start], z['sec'][:start], list(z['calls'])
            print(f'  resuming {name_of(model, cov)} at {start}/{n}', flush=True)
    runner = make_runner(model, device) if model != 'timesfm' else None
    peak_rss = peak_mps = 0
    t0 = time.time()
    for s0 in range(start, n, segment):
        s1 = min(n, s0 + segment)
        inps = [data.inputs(int(i), cov) for i in origins[s0:s1]]
        if model == 'timesfm':
            assert cov == 'none'
            r = run_timesfm(np.stack([x['context'] for x in inps]), device, batch)
            Q[s0:s1] = timesfm_to_21(r['quantiles'])
            sec[s0:s1] = r['seconds']
            calls += list(r['call_seconds'])
            peak_rss = max(peak_rss, int(r['peak_rss_bytes']))
        else:
            for b0 in range(s0, s1, batch):
                b1 = min(s1, b0 + batch)
                t = time.perf_counter()
                Q[b0:b1] = runner(inps[b0 - s0:b1 - s0])
                dt = time.perf_counter() - t
                sec[b0:b1] = dt / (b1 - b0)
                calls.append(dt)
                if device == 'mps':
                    peak_mps = max(peak_mps, mps_peak())
        el = time.time() - t0
        print(f'  {name_of(model, cov)} {s1}/{n}  {np.nanmean(sec):.3f}s/origin  '
              f'eta {el / (s1 - start) * (n - s1) / 60:.0f} min', flush=True)
        if cache:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            np.savez(cache + '.tmp.npz', origins=origins, Q=Q, sec=sec, calls=np.array(calls),
                     done=s1, batch=batch)
            os.replace(cache + '.tmp.npz', cache)
    assert not np.isnan(Q).any()
    if model != 'timesfm':
        peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    info = {'peak_rss_bytes': peak_rss, 'device': device, 'batch': batch, 'calls': np.array(calls)}
    if device == 'mps':
        info['peak_mps_driver_bytes'] = int(peak_mps)
    return Q, sec, info


# ── modes ───────────────────────────────────────────────────────────────────

def library_versions():
    v = {}
    for mod in ('torch', 'timestral', 'chronos', 'transformers'):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            pass
    return v


def runtime_record(model, cov, sec, info, n_origins, wall):
    """Runtime summary. Peak RSS: whole inference process (for TimesFM the
    worker process); it includes the loaded model and, for resumed runs, only
    the last session."""
    cs = info['calls']
    return {'model': name_of(model, cov), 'batch_size': int(info['batch']), 'n_calls': int(len(cs)),
            'n_origins': int(n_origins),
            'sec_per_call_mean': float(np.mean(cs)), 'sec_per_call_median': float(np.median(cs)),
            'sec_per_call_p95': float(np.percentile(cs, 95)), 'sec_per_origin': float(np.mean(sec)),
            'sec_total_calls': float(np.sum(cs)),
            'wall_seconds': float(wall), 'peak_rss_mb': info['peak_rss_bytes'] / 2**20,
            'peak_mps_driver_mb': info.get('peak_mps_driver_bytes', 0) / 2**20, 'device': info['device'],
            'threads': THREADS, 'context': CONTEXT, 'pred_len': PRED_LEN, 'versions': library_versions()}


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def save_predictions(model, cov, df, data, origins, Q):
    lab = pd.read_parquet(c.LABELS)
    thr = lab.threshold.to_numpy()
    pos = {int(o): k for k, o in enumerate(origins)}
    name = name_of(model, cov)
    paths = []
    for h in c.HORIZONS:
        for per in c.TEST_PERIODS:
            idx = F.select_origins(df, h, per)
            q = np.sort(Q[[pos[int(i)] for i in idx], STEP_OF_H[h] - 1, :].astype(float), axis=1)
            p = prob_exceed(q, thr[idx])                            # thr_known = threshold(T)
            tab = c.make_pred_table(name, per, h, data.ts_utc[idx], data.ts_utc[idx + h],
                                    data.price[idx + h], thr[idx + h], q, p)
            paths.append(c.save_pred_table(tab))
    return paths


def main():
    global THREADS
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=MODELS)
    ap.add_argument('--covariates', required=True, choices=COVS)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'mps'])
    ap.add_argument('--n-origins', type=int, default=0, help='timing run on N random origins')
    ap.add_argument('--seed', type=int, default=0, help='seed of the --n-origins sample')
    ap.add_argument('--save-q', default='', help='with --n-origins: save forecasts here (used by --mps-check)')
    ap.add_argument('--threads', type=int, default=THREADS, help='torch CPU threads')
    ap.add_argument('--batch', type=int, default=1, help='origins per forecast call')
    ap.add_argument('--batch-check', action='store_true',
                    help='20 origins: batched calls (--batch, default 20) vs 20 single calls (max |dq| <= 1e-4)')
    ap.add_argument('--mps-check', action='store_true')
    ap.add_argument('--check-inputs', type=int, default=200, help='origins for the input leakage check')
    a = ap.parse_args()
    THREADS = a.threads
    if a.model == 'timesfm' and a.covariates != 'none':
        ap.error('TimesFM runs without covariates only')
    name = name_of(a.model, a.covariates)

    df = pd.read_parquet(c.MERGED)
    data = Data(df)
    origins = test_origins(df)
    rng = np.random.default_rng(0)

    if a.check_inputs:
        cand = np.concatenate([rng.choice(origins, a.check_inputs, replace=False), origins[:3], origins[-3:]])
        n = check_inputs(df, cand, a.covariates)
        print(f'input leakage check OK: {n} origins, covariates={a.covariates}')

    if a.mps_check:
        # each device in its own process, so that a crash of the MPS backend is recorded
        res = {}
        for dev in ('cpu', 'mps'):
            with tempfile.TemporaryDirectory() as tmp:
                out = os.path.join(tmp, 'q.npz')
                r = subprocess.run([sys.executable, os.path.abspath(__file__), '--model', a.model,
                                    '--covariates', a.covariates, '--device', dev, '--n-origins', '20',
                                    '--seed', '1', '--check-inputs', '0', '--save-q', out],
                                   capture_output=True, text=True)
                if r.returncode == 0:
                    z = np.load(out)
                    res[dev] = (z['Q'], float(z['sec'].mean()))
                else:
                    err = [ln for ln in (r.stdout + r.stderr).splitlines()
                           if 'Error' in ln or 'exception' in ln.lower()]
                    res[dev] = (None, f'exit {r.returncode}: ' + (err[-1].strip() if err else 'no message'))
        assert res['cpu'][0] is not None, res['cpu'][1]
        ok = res['mps'][0] is not None
        d = float(np.max(np.abs(res['mps'][0] - res['cpu'][0]))) if ok else None
        rec = {'model': name, 'n_origins': 20, 'max_abs_diff_quantiles': d,
               'sec_per_call_cpu': res['cpu'][1], 'sec_per_call_mps': res['mps'][1] if ok else None,
               'mps_error': None if ok else res['mps'][1], 'use_mps': bool(ok and d <= 1e-3),
               'versions': library_versions()}
        write_json(os.path.join(RUNTIME_DIR, f'mps_check_{name}.json'), rec)
        print(json.dumps(rec, indent=2))
        return

    if a.batch_check:
        sub = np.sort(np.random.default_rng(2).choice(origins, 20, replace=False))
        Q1, s1, _ = forecast_many(a.model, a.covariates, data, sub, a.device, batch=1)
        bs = a.batch if a.batch > 1 else 20
        Qb, sb, _ = forecast_many(a.model, a.covariates, data, sub, a.device, batch=bs)
        d = float(np.max(np.abs(Q1.astype(float) - Qb)))
        rec = {'model': name, 'n_origins': 20, 'batch_size': bs, 'max_abs_diff_quantiles': d,
               'pass_1e-4': bool(d <= 1e-4), 'sec_per_origin_single': float(s1.mean()),
               'sec_per_origin_batched': float(sb.mean()), 'versions': library_versions()}
        write_json(os.path.join(RUNTIME_DIR, f'batch_check_{name}.json'), rec)
        print(json.dumps({k: v for k, v in rec.items() if k != 'versions'}, indent=2))
        return

    if a.n_origins:
        sub = np.sort(np.random.default_rng(a.seed).choice(origins, a.n_origins, replace=False))
        t = time.time()
        Q, sec, info = forecast_many(a.model, a.covariates, data, sub, a.device, batch=a.batch)
        if a.save_q:
            np.savez(a.save_q, Q=Q, sec=sec, origins=sub)
            return
        rec = runtime_record(a.model, a.covariates, sec, info, len(sub), time.time() - t)
        write_json(os.path.join(RUNTIME_DIR, f'{name}_sample_b{a.batch}.json'), rec)
        print(json.dumps({k: v for k, v in rec.items() if k != 'versions'}, indent=2))
        return

    t = time.time()
    Q, sec, info = forecast_many(a.model, a.covariates, data, origins, a.device, batch=a.batch,
                                 cache=os.path.join(CACHE_DIR, f'{name}.npz'))
    rec = runtime_record(a.model, a.covariates, sec, info, len(origins), time.time() - t)
    write_json(os.path.join(RUNTIME_DIR, f'{name}.json'), rec)
    for p in save_predictions(a.model, a.covariates, df, data, origins, Q):
        print('  wrote', os.path.relpath(p, HERE))
    print(json.dumps({k: v for k, v in rec.items() if k != 'versions'}, indent=2))


if __name__ == '__main__':
    main()
