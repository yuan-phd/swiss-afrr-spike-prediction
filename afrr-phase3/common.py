"""
common.py — shared definitions for phase-3 experiments (see TASK.md).

Paths, the fixed task constants (horizons, quantile levels, splits), the
prediction-table schema and small time helpers. Every later model imports
from here so that all experiments use the same task.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
MERGED = os.path.join(DATA, 'merged_2023_2026.parquet')
LABELS = os.path.join(DATA, 'labels.parquet')
PRED_DIR = os.path.join(HERE, 'predictions')
RESULTS_DIR = os.path.join(HERE, 'results')

LOCAL_TZ = 'Europe/Zurich'
STEP = pd.Timedelta('15min')
TARGET = 'pos_sec_price'

HORIZONS = (4, 96)                      # steps of 15 min: 1 h and 24 h
QUANTILES = np.array([0.01] + [round(0.05 * i, 2) for i in range(1, 20)] + [0.99])
QCOLS = [f'q{int(round(q * 100)):02d}' for q in QUANTILES]
assert len(QUANTILES) == 21 and QCOLS[0] == 'q01' and QCOLS[-1] == 'q99'

THRESH_WINDOW = pd.Timedelta(days=90)
THRESH_Q = 0.90

# Periods, in Swiss local wall-clock time, [start, end).
PERIODS = {
    'train': ('2023-01-01', '2025-01-01'),
    'test_A': ('2025-01-01', '2026-01-01'),
    'test_B': ('2026-01-01', '2026-10-01'),
}
TEST_PERIODS = ('test_A', 'test_B')
TAIL_FRAC = 0.15                        # last 15 % of train: early stopping + calibration

# Known-day-ahead columns (published on D-1 for delivery day D, ~12:45 local).
KNOWN_DA_COLS = ['Sched_DE_CH', 'Sched_CH_IT', 'Sched_FR_CH', 'Sched_IT_CH',
                 'DA_Price_CH', 'DA_Price_DE', 'DA_Price_Spread_DE_CH',
                 'CH_Load_Forecast']
DA_PUBLICATION_HOUR = 13                # next-day values usable from 13:00 local

# Fixed prediction-table schema (one parquet per model x horizon x period).
PRED_SCHEMA = {
    'model': 'string',
    'period': 'string',
    'horizon': 'int16',
    'origin_utc': 'datetime64[ns, UTC]',
    'target_utc': 'datetime64[ns, UTC]',
    'y': 'float64',                     # realised pos_sec_price at target_utc
    'threshold': 'float64',             # label threshold(target_utc)
    **{c: 'float64' for c in QCOLS},
    'spike_prob': 'float64',
}


def local_bound(s):
    return pd.Timestamp(s, tz=LOCAL_TZ).tz_convert('UTC')


def period_bounds_utc(name):
    a, b = PERIODS[name]
    return local_bound(a), local_bound(b)


def train_split_bounds():
    """UTC bounds (b0, b1, b2, b3): fit [b0,b1), early-stop [b1,b2), calibrate [b2,b3).
    b1 = start + 85 % of the training span; b2 = midpoint of the tail. Rounded
    down to a full hour."""
    b0, b3 = period_bounds_utc('train')
    span = b3 - b0
    b1 = (b0 + (1 - TAIL_FRAC) * span).floor('h')
    b2 = (b1 + (b3 - b1) / 2).floor('h')
    return b0, b1, b2, b3


def pred_path(model, horizon, period):
    return os.path.join(PRED_DIR, f'{model}_h{horizon}_{period}.parquet')


def make_pred_table(model, period, horizon, origin_utc, target_utc, y, threshold,
                    quantiles, spike_prob):
    """Build a prediction table with the fixed schema. quantiles: (n, 21) array
    or None (then all quantile columns are NaN)."""
    n = len(origin_utc)
    q = np.full((n, len(QCOLS)), np.nan) if quantiles is None else np.asarray(quantiles, float)
    assert q.shape == (n, len(QCOLS))
    df = pd.DataFrame({'model': model, 'period': period, 'horizon': horizon,
                       'origin_utc': pd.DatetimeIndex(origin_utc),
                       'target_utc': pd.DatetimeIndex(target_utc),
                       'y': np.asarray(y, float), 'threshold': np.asarray(threshold, float)})
    for i, c in enumerate(QCOLS):
        df[c] = q[:, i]
    df['spike_prob'] = np.asarray(spike_prob, float)
    return validate_pred_table(df)


def validate_pred_table(df):
    missing = [c for c in PRED_SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f'prediction table missing columns {missing}')
    df = df[list(PRED_SCHEMA)].astype(PRED_SCHEMA)
    q = df[QCOLS].to_numpy()
    ok = ~np.isnan(q).any(axis=1)
    if ok.any() and (np.diff(q[ok], axis=1) < -1e-9).any():
        raise ValueError('quantiles are not non-decreasing')
    p = df['spike_prob'].to_numpy()
    if np.nanmin(p) < 0 or np.nanmax(p) > 1:
        raise ValueError('spike_prob outside [0, 1]')
    return df


def save_pred_table(df):
    df = validate_pred_table(df)
    keys = df[['model', 'horizon', 'period']].drop_duplicates()
    assert len(keys) == 1, 'one model x horizon x period per file'
    m, h, p = keys.iloc[0]
    os.makedirs(PRED_DIR, exist_ok=True)
    path = pred_path(m, int(h), p)
    df.to_parquet(path, index=False)
    return path
