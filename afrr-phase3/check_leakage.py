"""
check_leakage.py — assert that no feature uses information unavailable at T.

For each sampled origin T and horizon h, build a copy of the data in which
  * every observed column (all except timestamps and KNOWN_DA_COLS) is NaN
    for rows at or after T (allowed: rows <= T - 15 min), and
  * every known-day-ahead column is NaN for rows whose delivery day is not
    available at T (features.da_available: day <= day(T), or next day if
    T >= 13:00 local),
recompute all features on that copy, and assert that every feature at T is
unchanged versus the features computed on the full data (NaN == NaN; floats
compared with atol = rtol = 1e-9, because rolling sums restart on the
shorter slice).

Sampled origins per horizon: random full hours over the whole dataset plus
edge cases (DST changes, 12:00-14:00 around the publication cutoff, period
boundaries, first labelled day).

A self-test first plants two leaky features (price at T; DA at T+h without
the publication rule) and asserts that both are detected.

Usage:  python afrr-phase3/check_leakage.py [--n-random 300]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as c  # noqa: E402
import features as F  # noqa: E402

WINDOW = 8640 + 8            # 90 days of 15-min rows plus a margin
TS_COLS = ['timestamp_utc', 'timestamp_local']

EDGE_LOCAL = ['2023-03-26 01:00', '2023-03-26 03:00', '2023-10-29 01:00', '2023-10-29 03:00',
              '2024-03-31 03:00', '2024-10-27 02:00', '2025-03-30 01:00', '2025-03-30 03:00',
              '2025-10-26 01:00', '2025-10-26 02:00', '2025-10-26 03:00', '2026-03-29 03:00',
              '2025-06-10 12:00', '2025-06-10 13:00', '2025-06-10 14:00', '2026-02-02 12:00',
              '2026-02-02 13:00', '2024-12-31 23:00', '2025-01-01 00:00', '2025-12-31 22:00',
              '2026-01-01 00:00', '2026-09-29 23:00', '2023-04-01 00:00', '2023-01-08 00:00']


def masked_slice(df, i, h):
    lo = max(0, i - WINDOW)
    s = df.iloc[lo:i + h + 1].copy()
    obs_cols = [k for k in s.columns if k not in TS_COLS + c.KNOWN_DA_COLS]
    pos = np.arange(lo, lo + len(s))
    s.loc[pos >= i, obs_cols] = np.nan
    T_local = df.timestamp_local.iloc[[i]].repeat(len(s))
    avail = F.da_available(T_local, s.timestamp_local)
    s.loc[~avail, c.KNOWN_DA_COLS] = np.nan
    return s, i - lo


def same(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    both_nan = np.isnan(a) & np.isnan(b)
    close = np.isclose(a, b, rtol=1e-9, atol=1e-9)
    return both_nan | close


def main(n_random=300, seed=0):
    df = pd.read_parquet(c.MERGED)
    ts = pd.DatetimeIndex(df.timestamp_utc)
    full_hour = np.where(ts.minute == 0)[0]
    tl = pd.DatetimeIndex(df.timestamp_local)
    # match on local wall-clock time: both copies of an autumn repeated hour
    edge = [int(k) for t in EDGE_LOCAL for k in np.where(tl == pd.Timestamp(t))[0]]
    assert len(edge) >= len(EDGE_LOCAL)
    rng = np.random.default_rng(seed)
    names = F.feature_names()
    total, failures = 0, []
    for h in c.HORIZONS:
        full = F.build_features(df, h)
        cand = np.concatenate([rng.choice(full_hour[full_hour < len(df) - h], n_random, replace=False),
                               [e for e in edge if e < len(df) - h]])
        nan_seen = pd.Series(0, index=names)
        for i in cand:
            s, j = masked_slice(df, int(i), h)
            Xm = F.build_features(s, h).iloc[j]
            Xf = full.iloc[int(i)]
            ok = same(Xf.to_numpy(), Xm.to_numpy())
            nan_seen += Xf.isna().astype(int)
            total += 1
            if not ok.all():
                bad = [n for n, o in zip(names, ok) if not o]
                failures.append((h, str(ts[i]), bad, Xf[bad].to_dict(), Xm[bad].to_dict()))
        print(f'h={h}: {len(cand)} origins checked, {len(names)} features each; '
              f'features NaN at >= 1 checked origin: '
              f'{int((nan_seen > 0).sum())}')
    if failures:
        for f in failures[:10]:
            print('LEAK', f)
        raise AssertionError(f'{len(failures)} origin(s) with features that change when '
                             f'future data is masked')
    print(f'OK: all {len(names)} features unchanged for {total} (origin, horizon) checks')


def self_test(n=20, seed=1):
    """Negative control: two deliberately leaky features must be detected.
    leak_p_T     = price of the row at T (not settled at T)
    leak_da_tph  = DA_Price_CH at T+h without the publication rule"""
    orig = F.build_features

    def leaky(d, h):
        X = orig(d, h)
        X['leak_p_T'] = d[c.TARGET].astype(float).to_numpy()
        X['leak_da_tph'] = d['DA_Price_CH'].astype(float).shift(-h).to_numpy()
        return X

    df = pd.read_parquet(c.MERGED)
    ts = pd.DatetimeIndex(df.timestamp_utc)
    rng = np.random.default_rng(seed)
    cand = rng.choice(np.where((ts.minute == 0) & (pd.DatetimeIndex(df.timestamp_local).hour < 13))[0][:-200], n,
                      replace=False)
    found = set()
    full = leaky(df, 96)
    for i in cand:
        s, j = masked_slice(df, int(i), 96)
        Xm = leaky(s, 96).iloc[j]
        ok = same(full.iloc[int(i)].to_numpy(), Xm.to_numpy())
        found |= {n for n, o in zip(full.columns, ok) if not o}
    assert {'leak_p_T', 'leak_da_tph'} <= found, f'self-test failed: detected only {found}'
    assert found == {'leak_p_T', 'leak_da_tph'}, f'unexpected detections {found}'
    print(f'self-test OK: both planted leaks detected ({n} origins, h=96, T before 13:00)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-random', type=int, default=300)
    a = ap.parse_args()
    self_test()
    main(a.n_random)
