"""
features.py — feature matrix for origin T and horizon h (see TASK.md).

Information rules (fixed for all phase-3 models):
  * Observed columns: rows up to T - 15 min only (the row at T is the
    interval [T, T+15 min), not settled at T). "lag k" = row T - k*15 min, k >= 1.
  * Known-day-ahead columns (common.KNOWN_DA_COLS): a value for delivery day d
    (Swiss local date of the row) is available at T iff d <= local date of T,
    or d = next day and local hour of T >= 13. Otherwise NaN.
  * Calendar features are computed from timestamps only.

The dataset is a complete 15-min grid (asserted), so positional offsets are
time offsets: row i is origin T, row i - k is T - k steps, row i + h is T + h.

build_features(df, h) returns one row per dataset row (as if every row were an
origin). select_origins() keeps full-hour origins whose target is inside a period.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as c  # noqa: E402
from labels import compute_threshold  # noqa: E402

# Swiss public holidays used as a flag on the target's local date, hard-coded
# for the dataset years. The set: New Year, Good Friday, Easter Monday,
# Ascension, Whit Monday, National Day, Christmas, St Stephen's Day.
# Only 1 Aug is a federal holiday by law. New Year, Ascension and Christmas
# are holidays in every canton; the others in all or almost all cantons
# (Good Friday is not in VS/TI; St Stephen's Day is not in a few cantons).
SWISS_HOLIDAYS = {
    2023: ['01-01', '04-07', '04-10', '05-18', '05-29', '08-01', '12-25', '12-26'],
    2024: ['01-01', '03-29', '04-01', '05-09', '05-20', '08-01', '12-25', '12-26'],
    2025: ['01-01', '04-18', '04-21', '05-29', '06-09', '08-01', '12-25', '12-26'],
    2026: ['01-01', '04-03', '04-06', '05-14', '05-25', '08-01', '12-25', '12-26'],
}
HOLIDAY_DATES = {pd.Timestamp(f'{y}-{md}').date() for y, mds in SWISS_HOLIDAYS.items() for md in mds}


def _easter(y):
    """Anonymous Gregorian algorithm; used only to check the hard-coded dates."""
    a, b, cc = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    hh = (19 * a + b - d - g + 15) % 30
    i, k = cc // 4, cc % 4
    l = (32 + 2 * e + 2 * i - hh - k) % 7
    m = (a + 11 * hh + 22 * l) // 451
    month = (hh + l - 7 * m + 114) // 31
    day = (hh + l - 7 * m + 114) % 31 + 1
    return pd.Timestamp(y, month, day)


for _y, _mds in SWISS_HOLIDAYS.items():
    _e = _easter(_y)
    _exp = {pd.Timestamp(_y, 1, 1), _e - pd.Timedelta(days=2), _e + pd.Timedelta(days=1),
            _e + pd.Timedelta(days=39), _e + pd.Timedelta(days=50), pd.Timestamp(_y, 8, 1),
            pd.Timestamp(_y, 12, 25), pd.Timestamp(_y, 12, 26)}
    assert {pd.Timestamp(f'{_y}-{md}') for md in _mds} == _exp, f'holiday table wrong for {_y}'

OBS_LAGGED = {   # observed series -> (lags, rolling-mean windows), all ending at T - 15 min
    'DE_WindSolar_Error': ((1, 4), (96,)),
    'UF_DE': ((1, 4), (96,)),
    'UF_FR': ((1, 4), (96,)),
    'pos_sec_vol_mw': ((1, 4), (96,)),
    'neg_sec_vol_mw': ((1, 4), (96,)),
    'CH_Pump_Gen': ((1, 4), ()),
}


def feature_names():
    names = ['p_lag1', 'p_lag4', 'p_lag96', 'p_mean_96', 'p_std_96',
             'p_mean_672', 'p_std_672', 'thr_known']
    for col, (lags, wins) in OBS_LAGGED.items():
        names += [f'{col}_lag{k}' for k in lags] + [f'{col}_mean_{w}' for w in wins]
    for col in c.KNOWN_DA_COLS:
        names += [f'{col}_tph', f'{col}_chg']
    names += ['cal_hour', 'cal_dow', 'cal_month', 'cal_weekend', 'cal_holiday']
    return names


def check_grid(df):
    ts = pd.DatetimeIndex(df.timestamp_utc)
    if not (ts[1:] - ts[:-1] == c.STEP).all():
        raise ValueError('input is not a contiguous 15-min grid')


def da_available(origin_local, target_local):
    """Boolean array: known-day-ahead value of the target row usable at origin."""
    d_o = pd.DatetimeIndex(origin_local).normalize()
    d_t = pd.DatetimeIndex(target_local).normalize()
    hour_o = pd.DatetimeIndex(origin_local).hour
    same_or_before = d_t <= d_o
    next_day = (d_t == d_o + pd.Timedelta(days=1)) & (hour_o >= c.DA_PUBLICATION_HOUR)
    return np.asarray(same_or_before | next_day)


def build_features(df, h):
    """Features for every row of df treated as origin T, horizon h steps.
    Rows whose T+h is outside df get NaN for T+h features."""
    check_grid(df)
    n = len(df)
    out = {}
    p = df[c.TARGET].astype(float)
    past = p.shift(1)                                  # last allowed row = T - 15 min
    out['p_lag1'] = past
    out['p_lag4'] = p.shift(4)
    out['p_lag96'] = p.shift(96)
    for w in (96, 672):
        r = past.rolling(w, min_periods=w)
        out[f'p_mean_{w}'] = r.mean()
        out[f'p_std_{w}'] = r.std()
    # latest threshold known at T: window [T-90d, T), i.e. rows <= T - 15 min
    out['thr_known'] = pd.Series(compute_threshold(df.timestamp_utc, p), index=df.index)

    obs = pd.DataFrame({
        'DE_WindSolar_Error': df.DE_WindSolar_Error,
        'UF_DE': (df.DE_CH_mw - df.CH_DE_mw) - df.Sched_DE_CH,
        'UF_FR': (df.FR_CH_mw - df.CH_FR_mw) - df.Sched_FR_CH,
        'pos_sec_vol_mw': df.pos_sec_vol_mw,
        'neg_sec_vol_mw': df.neg_sec_vol_mw,
        'CH_Pump_Gen': df.CH_Pump_Gen,
    }, index=df.index).astype(float)
    for col, (lags, wins) in OBS_LAGGED.items():
        for k in lags:
            out[f'{col}_lag{k}'] = obs[col].shift(k)
        for w in wins:
            out[f'{col}_mean_{w}'] = obs[col].shift(1).rolling(w, min_periods=w).mean()

    tl = df.timestamp_local.reset_index(drop=True)
    target_local = tl.shift(-h)
    has_target = target_local.notna().to_numpy()
    avail = np.zeros(n, bool)
    avail[has_target] = da_available(tl[has_target], target_local[has_target])
    for col in c.KNOWN_DA_COLS:
        v = df[col].astype(float).reset_index(drop=True)
        at_tph = v.shift(-h).where(avail)
        out[f'{col}_tph'] = pd.Series(at_tph.to_numpy(), index=df.index)
        out[f'{col}_chg'] = pd.Series((at_tph - v).to_numpy(), index=df.index)

    tgt = pd.DatetimeIndex(target_local)
    cal = {
        'cal_hour': tgt.hour, 'cal_dow': tgt.dayofweek, 'cal_month': tgt.month,
        'cal_weekend': (tgt.dayofweek >= 5),
        'cal_holiday': np.array([ok and (d in HOLIDAY_DATES)
                                 for d, ok in zip(tgt.date, has_target)]),
    }
    for k, v in cal.items():
        arr = np.asarray(v, float)
        arr[~has_target] = np.nan
        out[k] = pd.Series(arr, index=df.index)

    X = pd.DataFrame(out, index=df.index)[feature_names()]
    return X


def da_nan_fraction(df, h, origin_mask):
    tl = df.timestamp_local.reset_index(drop=True)
    tgt = tl.shift(-h)
    ok = origin_mask & tgt.notna().to_numpy()
    return 1 - da_available(tl[ok], tgt[ok]).mean()


def select_origins(df, h, period):
    """Positions i of full-hour origins with origin and target in [start, end) of period."""
    a, b = c.period_bounds_utc(period)
    ts = pd.DatetimeIndex(df.timestamp_utc)
    i = np.arange(len(df) - h)
    tgt = ts[i + h]
    org = ts[i]
    keep = (org.minute == 0) & (org >= a) & (tgt < b)
    return i[keep]
