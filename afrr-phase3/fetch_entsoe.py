"""
fetch_entsoe.py — ENTSO-E fetcher for the phase-3 benchmark dataset
================================================================================
Based on 00_fetch_energy_data.py (v2). Same queries, same 11 output columns.
Changes from the original (see afrr-phase3/LOG.md for why):

  1. Date range and output file are command-line arguments.
  2. German day-ahead price uses bidding-zone code DE_LU (the code 'DE' returns
     NoMatchingDataError, which the original silently skipped).
  3. Errors are not swallowed. Any failed query raises with the query label and
     the month. Transient network/HTTP 429/5xx errors are retried up to
     3 times with a back-off, then raise.
  4. The output timestamp is tz-aware UTC ('YYYY-MM-DD HH:MM:SS+00:00'). Both copies of
     the repeated autumn hour are kept: they are distinct UTC instants.
  5. No gap filling. The final final_df.ffill() is removed. Hourly series are
     placed on the 15-min grid by repeating each value only inside its own
     native interval (for example 4 x 15 min for an hourly value). Missing
     source points stay NaN.
  6. Wind/solar actuals take the 'Actual Aggregated' sub-column explicitly
     (entsoe-py >= 0.7 returns two-level columns for DE generation). A missing
     generation type raises instead of being skipped.

Date arguments are calendar days in Swiss local time (Europe/Zurich).
--start is inclusive from 00:00 local; --end is inclusive up to 23:45 local.

Usage:
  python afrr-phase3/fetch_entsoe.py --start 2025-01-01 --end 2026-09-30 \
      --out afrr-phase3/data/raw/entsoe_2025_2026.csv

Requires ENTSOE_TOKEN in the repo-root .env (or in the environment).

OUTPUT COLUMNS (UTC index 'timestamp' = start of the 15-min interval):
  DE_WindSolar_Error    DE (wind on+offshore + solar) actual minus DA forecast (MW)
  Sched_DE_CH           Scheduled day-ahead commercial exchange DE->CH (MW)
  Sched_CH_IT           Scheduled day-ahead commercial exchange CH->IT (MW)
  CH_Pump_Gen           CH actual generation, Hydro Pumped Storage (MW)
  DA_Price_DE           Day-ahead price, bidding zone DE_LU (EUR/MWh)
  DA_Price_CH           Day-ahead price, CH (EUR/MWh)
  DA_Price_Spread_DE_CH DA_Price_DE - DA_Price_CH (EUR/MWh), NaN if either NaN
  Sched_FR_CH           Scheduled day-ahead commercial exchange FR->CH (MW)
  Sched_IT_CH           Scheduled day-ahead commercial exchange IT->CH (MW)
  CH_Load_Forecast      CH day-ahead total load forecast (MW)
================================================================================
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOCAL_TZ = 'Europe/Zurich'
GRID = pd.Timedelta('15min')
MAX_NATIVE_RES = pd.Timedelta('60min')   # coarsest resolution among the queries

CH, DE, DE_LU, IT, FR = 'CH', 'DE', 'DE_LU', 'IT', 'FR'
WINDSOLAR = ['Wind Onshore', 'Wind Offshore', 'Solar']
SLEEP_S = 1.5
MAX_RETRIES = 3

COLUMNS = ['DE_WindSolar_Error', 'Sched_DE_CH', 'Sched_CH_IT', 'CH_Pump_Gen',
           'DA_Price_DE', 'DA_Price_CH', 'DA_Price_Spread_DE_CH',
           'Sched_FR_CH', 'Sched_IT_CH', 'CH_Load_Forecast']


class FetchError(RuntimeError):
    pass


def fetch(label, month, func, *args, **kwargs):
    """Rate-limited query. Retries transient errors, raises on anything else."""
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            data = func(*args, **kwargs)
            time.sleep(SLEEP_S)
            if data is None or len(data) == 0:
                raise FetchError(f"query '{label}' returned no data for month {month}")
            return data
        except FetchError:
            raise
        except (requests.ConnectionError, requests.Timeout) as e:
            err = e
        except requests.HTTPError as e:
            code = getattr(e.response, 'status_code', None)
            if code not in (429, 500, 502, 503, 504):
                raise FetchError(f"query '{label}' failed for month {month}: "
                                 f"{type(e).__name__}: {e}") from e
            err = e
        except Exception as e:
            raise FetchError(f"query '{label}' failed for month {month}: "
                             f"{type(e).__name__}: {e}") from e
        if attempt <= MAX_RETRIES:
            wait = 10 * attempt
            print(f"    [retry {attempt}/{MAX_RETRIES} in {wait}s] {label}: {err}")
            time.sleep(wait)
    raise FetchError(f"query '{label}' failed for month {month} after "
                     f"{MAX_RETRIES} retries: {type(err).__name__}: {err}")


def to_series(obj, label, month, column=None):
    """Return a single float Series with a UTC index, no duplicate instants."""
    if isinstance(obj, pd.DataFrame):
        if column is None:
            if obj.shape[1] != 1:
                raise FetchError(f"query '{label}' month {month}: expected 1 column, "
                                 f"got {obj.columns.tolist()}")
            obj = obj.iloc[:, 0]
        else:
            obj = obj[column]
    s = obj.astype(float).copy()
    s.index = pd.DatetimeIndex(s.index).tz_convert('UTC')
    s = s.sort_index()
    dup = s.index.duplicated(keep=False)
    if dup.any():
        g = s[dup].groupby(level=0).nunique()
        if (g > 1).any():
            raise FetchError(f"query '{label}' month {month}: conflicting values at "
                             f"{g[g > 1].index[:3].tolist()}")
        s = s[~s.index.duplicated(keep='first')]
    return s


def gen_column(df, gen_type, label, month):
    """Pick 'Actual Aggregated' for a generation type (1- or 2-level columns)."""
    if isinstance(df.columns, pd.MultiIndex):
        key = (gen_type, 'Actual Aggregated')
        if key not in df.columns:
            raise FetchError(f"query '{label}' month {month}: column {key} missing; "
                             f"have {df.columns.tolist()}")
        return to_series(df[key], label, month)
    if gen_type not in df.columns:
        raise FetchError(f"query '{label}' month {month}: column '{gen_type}' missing; "
                         f"have {df.columns.tolist()}")
    return to_series(df[gen_type], label, month)


def to_grid(s, idx):
    """
    Put a series on the 15-min UTC grid without filling gaps.
    Each source point t_i covers [t_i, t_i + res_i), where res_i is its native
    resolution: the step to the next point, t_{i+1} - t_i. If that step is
    longer than 60 min (a real hole in the source) or there is no next point,
    the step from the previous point is used instead. Capped at 60 min.
    The resolution can change from one delivery day to the next (e.g. FR->CH
    schedules switch between 15-min and hourly), so it is estimated per point.
    Grid points not covered by any source interval stay NaN.
    """
    if len(s) == 0:
        return pd.Series(np.nan, index=idx)
    t = s.index.to_series()
    prev = t.diff()
    nxt = -t.diff(-1)
    res = nxt.where(nxt <= MAX_NATIVE_RES, prev)
    res = res.fillna(MAX_NATIVE_RES).clip(upper=MAX_NATIVE_RES)
    src = pd.DataFrame({'t': s.index, 'v': s.values, 'end': (s.index + pd.TimedeltaIndex(res.values))})
    g = pd.DataFrame({'t': idx})
    m = pd.merge_asof(g, src, on='t', direction='backward')
    covered = m['t'] < m['end']
    return pd.Series(np.where(covered, m['v'], np.nan), index=idx)


def month_windows(start_local, end_local_excl):
    starts = pd.date_range(start_local.normalize().replace(day=1),
                           end_local_excl, freq='MS', tz=LOCAL_TZ)
    for ms in starts:
        a = max(ms, start_local)
        b = min(ms + pd.offsets.MonthBegin(1), end_local_excl)
        if a < b:
            yield ms.strftime('%Y-%m'), a, b


def fetch_month(client, label_m, a, b):
    idx = pd.date_range(a.tz_convert('UTC'), b.tz_convert('UTC'),
                        freq=GRID, inclusive='left')
    q = dict(start=a, end=b)
    chunk = pd.DataFrame(index=idx)

    de_fc = fetch('DE wind+solar forecast', label_m,
                  client.query_wind_and_solar_forecast, DE, **q)
    de_act = fetch('DE generation actual', label_m, client.query_generation, DE, **q)
    sched_de_ch = fetch('Sched DE->CH', label_m, client.query_scheduled_exchanges,
                        DE, CH, day_ahead=True, **q)
    sched_ch_it = fetch('Sched CH->IT', label_m, client.query_scheduled_exchanges,
                        CH, IT, day_ahead=True, **q)
    ch_gen = fetch('CH generation', label_m, client.query_generation, CH, **q)
    da_de = fetch('DA price DE_LU', label_m, client.query_day_ahead_prices, DE_LU, **q)
    da_ch = fetch('DA price CH', label_m, client.query_day_ahead_prices, CH, **q)
    sched_fr_ch = fetch('Sched FR->CH', label_m, client.query_scheduled_exchanges,
                        FR, CH, day_ahead=True, **q)
    sched_it_ch = fetch('Sched IT->CH', label_m, client.query_scheduled_exchanges,
                        IT, CH, day_ahead=True, **q)
    ch_load_fc = fetch('CH load forecast', label_m, client.query_load_forecast, CH, **q)

    # DE wind+solar error: sum on the grid; NaN if any component is NaN
    act = sum(to_grid(gen_column(de_act, c, 'DE generation actual', label_m), idx)
              for c in WINDSOLAR)
    for c in WINDSOLAR:
        if c not in de_fc.columns:
            raise FetchError(f"query 'DE wind+solar forecast' month {label_m}: "
                             f"column '{c}' missing; have {de_fc.columns.tolist()}")
    fc = sum(to_grid(to_series(de_fc, 'DE wind+solar forecast', label_m, c), idx)
             for c in WINDSOLAR)
    chunk['DE_WindSolar_Error'] = act - fc

    chunk['Sched_DE_CH'] = to_grid(to_series(sched_de_ch, 'Sched DE->CH', label_m), idx)
    chunk['Sched_CH_IT'] = to_grid(to_series(sched_ch_it, 'Sched CH->IT', label_m), idx)
    chunk['CH_Pump_Gen'] = to_grid(gen_column(ch_gen, 'Hydro Pumped Storage',
                                              'CH generation', label_m), idx)
    chunk['DA_Price_DE'] = to_grid(to_series(da_de, 'DA price DE_LU', label_m), idx)
    chunk['DA_Price_CH'] = to_grid(to_series(da_ch, 'DA price CH', label_m), idx)
    chunk['DA_Price_Spread_DE_CH'] = chunk['DA_Price_DE'] - chunk['DA_Price_CH']
    chunk['Sched_FR_CH'] = to_grid(to_series(sched_fr_ch, 'Sched FR->CH', label_m), idx)
    chunk['Sched_IT_CH'] = to_grid(to_series(sched_it_ch, 'Sched IT->CH', label_m), idx)
    chunk['CH_Load_Forecast'] = to_grid(to_series(ch_load_fc, 'CH load forecast', label_m), idx)
    return chunk[COLUMNS]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--start', required=True, help='first day, YYYY-MM-DD (Swiss local)')
    ap.add_argument('--end', required=True, help='last day inclusive, YYYY-MM-DD (Swiss local)')
    ap.add_argument('--out', required=True, help='output CSV path')
    args = ap.parse_args()

    load_dotenv(os.path.join(ROOT, '.env'))
    token = os.getenv('ENTSOE_TOKEN')
    if not token:
        sys.exit('ENTSOE_TOKEN not found in .env or environment')
    client = EntsoePandasClient(api_key=token)

    start_local = pd.Timestamp(args.start, tz=LOCAL_TZ)
    # calendar day offset, so a DST change on the last day is handled
    end_local_excl = pd.Timestamp(args.end, tz=LOCAL_TZ) + pd.DateOffset(days=1)
    windows = list(month_windows(start_local, end_local_excl))
    print(f'ENTSO-E fetch | {start_local} -> {end_local_excl} (exclusive) | '
          f'{len(windows)} months')

    chunks = []
    for i, (label_m, a, b) in enumerate(windows, 1):
        print(f'[{i:02d}/{len(windows)}] {label_m}  {a} -> {b}')
        c = fetch_month(client, label_m, a, b)
        na = c.isna().sum()
        print(f'    rows={len(c)}  NaN per column: '
              + ', '.join(f'{k}={v}' for k, v in na.items() if v))
        chunks.append(c)

    df = pd.concat(chunks)
    df.index.name = 'timestamp'
    assert df.index.is_monotonic_increasing and not df.index.duplicated().any()
    expected = pd.date_range(start_local.tz_convert('UTC'),
                             end_local_excl.tz_convert('UTC'), freq=GRID, inclusive='left')
    assert df.index.equals(expected), 'output index is not the complete 15-min UTC grid'

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out)
    print(f'\nSaved {args.out}: {len(df):,} rows, {df.index.min()} -> {df.index.max()}')
    print('NaN per column:')
    print(df.isna().sum().to_string())


if __name__ == '__main__':
    main()
