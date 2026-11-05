"""
merge_raw_data.py — phase-3 benchmark dataset: Swissgrid + ENTSO-E, UTC-keyed
================================================================================
Based on the repo-root merge_raw_data.py. Raw measurements only: no feature
engineering, no gap filling, no clipping. The only transformation is kWh per
15 min -> MW (x 4 / 1000) for the Swissgrid energy columns, as in the original.

Rebuild (from the repo root):
    python afrr-phase3/merge_raw_data.py

Inputs (see DATA.md for provenance):
    data/raw/EnergieUebersichtCH-2023.xlsx                 Swissgrid, naive local
    data/raw/EnergieUebersichtCH-2024.xlsx                 Swissgrid, naive local
    afrr-phase3/data/raw/EnergieUebersichtCH-2025.xlsx     Swissgrid (republished 2026-05-19)
    afrr-phase3/data/raw/EnergieUebersichtCH-2026.xlsx     Swissgrid (to 2026-09-30)
    afrr-phase3/data/raw/entsoe_2023_2024.csv              ENTSO-E, UTC (fetch_entsoe.py)
    afrr-phase3/data/raw/entsoe_2025_2026.csv              ENTSO-E, UTC (fetch_entsoe.py)

Outputs:
    afrr-phase3/data/merged_2023_2026.parquet
    afrr-phase3/data/merged_2023_2026.csv
    afrr-phase3/data/reports/missing_values.csv     per-column NaN, overall + per year
    afrr-phase3/data/reports/gaps.csv               missing 15-min instants (if any)

Timezone handling (the bug fixed here is at the old merge_raw_data.py:126-130,
which parsed the naive-local v2 timestamps as UTC and shifted ENTSO-E by
+1 h in winter and +2 h in summer):
  * Swissgrid: naive Europe/Zurich. 2023/2024 files label the END of the
    15-min interval, 2025/2026 files the START (see SWISSGRID_FILES). The
    spring day has 92 rows (one +75 min step), the autumn day 100 rows (a
    repeated 02:00-02:45 block, CEST first, then CET). Localised in file order
    with ambiguous='infer', nonexistent='raise'; end labels then -15 min.
    Output timestamps always mark the interval START.
  * ENTSO-E (fetch_entsoe.py, all years): tz-aware UTC, parsed directly; both
    copies of the autumn hour are present, so no DST handling is needed.
  * The old v2 file (naive Europe/Zurich, 00_fetch_energy_data.py:262-270) is
    no longer an input. If loaded ('naive_local'), it is localised with
    ambiguous=True, because only the first (CEST) copy of the repeated autumn
    hour survived its keep='first'.
  * Merge key: timestamp_utc (tz-aware UTC). timestamp_local is naive Swiss
    time for calendar features; it repeats in the autumn hour by design.
================================================================================
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OLD_RAW = os.path.join(ROOT, 'data', 'raw')
NEW_RAW = os.path.join(HERE, 'data', 'raw')
OUT_DIR = os.path.join(HERE, 'data')
REPORT_DIR = os.path.join(OUT_DIR, 'reports')

LOCAL_TZ = 'Europe/Zurich'
START_LOCAL = '2023-01-01 00:00'          # inclusive
END_LOCAL = '2026-10-01 00:00'            # exclusive -> last row 2026-09-30 23:45
FREQ = '15min'

# (path, label convention). Swissgrid changed the label convention between the
# 2024 and 2025 files; the workbooks' own 'Uebersicht' formulas state it:
#   2023/2024: SUM over MATCH(dat_von+TIME(0,15,0)) .. MATCH(dat_bis)  -> label = interval END
#   2025/2026: SUM over MATCH(dat_von) .. MATCH(dat_bis-TIME(0,15,0))  -> label = interval START
# 'end' labels are converted to interval start by subtracting 15 min (after UTC conversion).
SWISSGRID_FILES = {
    2023: (os.path.join(OLD_RAW, 'EnergieUebersichtCH-2023.xlsx'), 'end'),
    2024: (os.path.join(OLD_RAW, 'EnergieUebersichtCH-2024.xlsx'), 'end'),
    2025: (os.path.join(NEW_RAW, 'EnergieUebersichtCH-2025.xlsx'), 'start'),
    2026: (os.path.join(NEW_RAW, 'EnergieUebersichtCH-2026.xlsx'), 'start'),
}

# ENTSO-E sources: (path, timestamp convention, local-time period it supplies)
# Period bounds are Swiss local wall-clock, [from, to).
# All four years come from fetch_entsoe.py (UTC, DE_LU, no forward fill).
# The old v2 file (naive local) is still supported via 'naive_local' and is
# used by compare_entsoe.py.
ENTSOE_SOURCES = [
    (os.path.join(NEW_RAW, 'entsoe_2023_2024.csv'), 'utc',
     '2023-01-01 00:00', '2025-01-01 00:00'),
    (os.path.join(NEW_RAW, 'entsoe_2025_2026.csv'), 'utc',
     '2025-01-01 00:00', '2026-10-01 00:00'),
]

# Swissgrid sheet Zeitreihen0h15, 0-based column index -> name
SG_COLS = {
    0: 'timestamp',
    6: 'pos_sec_vol_kwh', 7: 'neg_sec_vol_kwh',
    12: 'CH_DE_kwh', 13: 'DE_CH_kwh',
    14: 'CH_FR_kwh', 15: 'FR_CH_kwh',
    16: 'CH_IT_kwh', 17: 'IT_CH_kwh',
    21: 'pos_sec_price', 22: 'neg_sec_price',
}
# expected English header text (second line of row 0), checked on load
SG_HEADERS = {
    6: 'Positive secundary control energy', 7: 'Negative secundary control energy',
    12: 'Cross Border Exchange CH->DE', 13: 'Cross Border Exchange DE->CH',
    14: 'Cross Border Exchange CH->FR', 15: 'Cross Border Exchange FR->CH',
    16: 'Cross Border Exchange CH->IT', 17: 'Cross Border Exchange IT->CH',
    21: 'Average positive secondary control energy prices',
    22: 'Average negative secondary control energy prices',
}
KWH_TO_MW = 4 / 1000

ENTSOE_COLS = ['DE_WindSolar_Error', 'Sched_DE_CH', 'Sched_CH_IT', 'CH_Pump_Gen',
               'DA_Price_CH', 'Sched_FR_CH', 'Sched_IT_CH', 'CH_Load_Forecast',
               'DA_Price_DE', 'DA_Price_Spread_DE_CH']
OUT_COLS = (['timestamp_utc', 'timestamp_local',
             'pos_sec_price', 'neg_sec_price', 'pos_sec_vol_mw', 'neg_sec_vol_mw',
             'CH_DE_mw', 'DE_CH_mw', 'CH_FR_mw', 'FR_CH_mw', 'CH_IT_mw', 'IT_CH_mw']
            + ENTSOE_COLS)


def local_bound(s):
    return pd.Timestamp(s, tz=LOCAL_TZ).tz_convert('UTC')


# ── Swissgrid ──────────────────────────────────────────────────────────────────
def load_swissgrid_year(year, extra_cols=None):
    """Load one Swissgrid year. extra_cols: {index: name} for verification use."""
    path, label = SWISSGRID_FILES[year]
    cols = {**SG_COLS, **(extra_cols or {})}
    print(f'  Swissgrid {year}: {os.path.relpath(path, ROOT)}  (labels = interval {label})')
    hdr = pd.read_excel(path, sheet_name='Zeitreihen0h15', header=None, nrows=1)
    for i, txt in SG_HEADERS.items():
        got = str(hdr.iloc[0, i]).split('\n')[-1].strip()
        if got != txt:
            raise ValueError(f'{path}: column {i} header {got!r}, expected {txt!r}')
    df = pd.read_excel(path, sheet_name='Zeitreihen0h15', header=None, skiprows=2,
                       usecols=sorted(cols))
    df.columns = [cols[i] for i in sorted(cols)]
    if df['timestamp'].isna().any():
        raise ValueError(f'{path}: {df["timestamp"].isna().sum()} empty timestamps')
    # 2023-2025 files hold 'DD.MM.YYYY HH:MM' text, 2026 holds Excel datetimes
    naive = pd.to_datetime(df['timestamp'], dayfirst=True)
    # file order must be 15-min steps except one spring jump (+75 min) and one
    # autumn step back (-45 min, start of the repeated 02:00-02:45 block)
    step = naive.diff().dropna()
    odd = step[step != pd.Timedelta('15min')]
    allowed = {pd.Timedelta('75min'), pd.Timedelta('-45min')}
    if not set(odd) <= allowed or len(odd) > 2:
        raise ValueError(f'{path}: unexpected timestamp steps {odd.value_counts().to_dict()}')
    # file order is kept: 'infer' needs the repeated autumn hour in sequence
    utc = (pd.DatetimeIndex(naive)
           .tz_localize(LOCAL_TZ, ambiguous='infer', nonexistent='raise')
           .tz_convert('UTC'))
    if label == 'end':
        utc = utc - pd.Timedelta('15min')
    elif label != 'start':
        raise ValueError(label)
    if not (utc[1:] - utc[:-1] == pd.Timedelta('15min')).all():
        raise ValueError(f'{path}: UTC instants are not contiguous 15-min steps')
    df['timestamp_utc'] = utc
    df = df.drop(columns='timestamp')
    for c in df.columns:
        if c != 'timestamp_utc':
            df[c] = pd.to_numeric(df[c], errors='raise').astype(float)
    for c in [c for c in df.columns if c.endswith('_kwh')]:
        df[c.replace('_kwh', '_mw')] = df.pop(c) * KWH_TO_MW
    print(f'    {len(df):,} rows  {df.timestamp_utc.min()} -> {df.timestamp_utc.max()} (UTC)')
    return df


def load_swissgrid(extra_cols=None):
    sg = pd.concat([load_swissgrid_year(y, extra_cols) for y in SWISSGRID_FILES],
                   ignore_index=True)
    lo, hi = local_bound(START_LOCAL), local_bound(END_LOCAL)
    outside = ~sg.timestamp_utc.between(lo, hi, inclusive='left')
    if outside.any():
        print(f'    dropping {outside.sum()} Swissgrid rows outside the target range')
    sg = sg[~outside]
    dup = sg.timestamp_utc.duplicated(keep=False)
    if dup.any():
        raise ValueError(f'Swissgrid duplicate UTC instants: '
                         f'{sg.timestamp_utc[dup].unique()[:5].tolist()}')
    return sg.sort_values('timestamp_utc').reset_index(drop=True)


# ── ENTSO-E ────────────────────────────────────────────────────────────────────
def load_entsoe_source(path, convention, frm, to):
    print(f'  ENTSO-E: {os.path.relpath(path, ROOT)} ({convention})')
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: 'timestamp'})
    if convention == 'naive_local':
        # v2: naive Europe/Zurich. Only the first (CEST) copy of the repeated
        # autumn hour exists (00_fetch_energy_data.py:270, keep='first').
        naive = pd.DatetimeIndex(pd.to_datetime(df['timestamp']))
        if naive.has_duplicates:
            raise ValueError(f'{path}: duplicate naive timestamps')
        ts = naive.tz_localize(LOCAL_TZ, ambiguous=True, nonexistent='raise').tz_convert('UTC')
    elif convention == 'utc':
        ts = pd.DatetimeIndex(pd.to_datetime(df['timestamp'], utc=True))
    else:
        raise ValueError(convention)
    df['timestamp_utc'] = ts
    df = df.drop(columns='timestamp')
    missing = [c for c in ENTSOE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f'{path}: missing columns {missing}')
    df = df[['timestamp_utc'] + ENTSOE_COLS]
    df = df[df.timestamp_utc.between(local_bound(frm), local_bound(to), inclusive='left')]
    print(f'    {len(df):,} rows used for local [{frm}, {to})')
    return df


def load_entsoe():
    en = pd.concat([load_entsoe_source(*s) for s in ENTSOE_SOURCES], ignore_index=True)
    dup = en.timestamp_utc.duplicated(keep=False)
    if dup.any():
        raise ValueError(f'ENTSO-E duplicate UTC instants across sources: '
                         f'{en.timestamp_utc[dup].unique()[:5].tolist()}')
    return en.sort_values('timestamp_utc').reset_index(drop=True)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print('=' * 70)
    print('PHASE-3 MERGE (UTC key, no filling)')
    print('=' * 70)
    os.makedirs(REPORT_DIR, exist_ok=True)

    print('\n[1] Swissgrid')
    sg = load_swissgrid()
    print('\n[2] ENTSO-E')
    en = load_entsoe()

    print('\n[3] Index checks')
    grid = pd.date_range(local_bound(START_LOCAL), local_bound(END_LOCAL),
                         freq=FREQ, inclusive='left')
    sg_missing = grid.difference(pd.DatetimeIndex(sg.timestamp_utc))
    en_missing = grid.difference(pd.DatetimeIndex(en.timestamp_utc))
    en_extra = pd.DatetimeIndex(en.timestamp_utc).difference(grid)
    gaps = pd.concat([
        pd.DataFrame({'source': 'swissgrid', 'timestamp_utc': sg_missing}),
        pd.DataFrame({'source': 'entsoe', 'timestamp_utc': en_missing}),
    ], ignore_index=True)
    gaps['timestamp_local'] = gaps.timestamp_utc.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    gaps.to_csv(os.path.join(REPORT_DIR, 'gaps.csv'), index=False)
    print(f'  expected grid: {len(grid):,} instants  {grid[0]} -> {grid[-1]}')
    print(f'  Swissgrid rows: {len(sg):,}  missing instants: {len(sg_missing)}')
    print(f'  ENTSO-E rows:   {len(en):,}  missing instants: {len(en_missing)} '
          f'(left as NaN)  off-grid rows: {len(en_extra)}')
    for _, r in gaps.iterrows():
        print(f'    gap  {r.source:<9} {r.timestamp_utc}  (local {r.timestamp_local})')
    if len(en_extra):
        raise ValueError(f'ENTSO-E rows off the 15-min grid: {en_extra[:5].tolist()}')

    # Swissgrid defines the rows; ENTSO-E is left-joined so its gaps show as NaN
    merged = sg.merge(en, on='timestamp_utc', how='left', validate='one_to_one')
    merged['timestamp_local'] = (merged.timestamp_utc.dt.tz_convert(LOCAL_TZ)
                                 .dt.tz_localize(None))
    merged = merged[OUT_COLS].sort_values('timestamp_utc').reset_index(drop=True)

    ts = merged.timestamp_utc
    assert str(ts.dt.tz) == 'UTC'
    assert ts.is_monotonic_increasing and not ts.duplicated().any(), 'not strictly increasing'
    assert not merged.duplicated().any(), 'duplicate rows'
    if len(sg_missing):
        raise AssertionError(f'{len(sg_missing)} missing 15-min instants in Swissgrid; '
                             f'see {REPORT_DIR}/gaps.csv. Not filled; output not written.')
    assert pd.DatetimeIndex(ts).equals(grid), 'index is not the complete 15-min grid'
    print(f'  merged: {len(merged):,} rows, {len(merged.columns)} columns; '
          f'strictly increasing, no duplicates, complete 15-min grid')

    print('\n[4] Missing values (NaN count; year = Swiss local calendar year)')
    year = merged.timestamp_local.dt.year
    data_cols = OUT_COLS[2:]
    miss = merged[data_cols].isna().groupby(year).sum().T
    miss.columns = [str(c) for c in miss.columns]
    miss.insert(0, 'total', merged[data_cols].isna().sum())
    miss.loc['_rows'] = [len(merged)] + [int((year == int(y)).sum()) for y in miss.columns[1:]]
    miss.index.name = 'column'
    miss.to_csv(os.path.join(REPORT_DIR, 'missing_values.csv'))
    print(miss.to_string())

    print('\n[5] Save')
    pq = os.path.join(OUT_DIR, 'merged_2023_2026.parquet')
    csv = os.path.join(OUT_DIR, 'merged_2023_2026.csv')
    merged.to_parquet(pq, index=False)
    merged.to_csv(csv, index=False)
    print(f'  {os.path.relpath(pq, ROOT)}\n  {os.path.relpath(csv, ROOT)}')
    print('=' * 70)


if __name__ == '__main__':
    main()
