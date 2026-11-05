"""
verify_alignment.py — checks on afrr-phase3/data/merged_2023_2026.parquet

1. Offset test. Correlate ENTSO-E series with the Swissgrid measurement they
   should track, shifting the ENTSO-E series by k = -8..+8 steps of 15 min:
       CH_Load_Forecast  vs  Swissgrid "Total energy consumed by end users"
                             (sheet column 1, kWh -> MW; read from the raw
                             Swissgrid files and joined on timestamp_utc,
                             because it is not a column of the merged file)
       Sched_DE_CH       vs  DE_CH_mw - CH_DE_mw (measured net DE->CH)
   r(k) = corr(swissgrid(t), entsoe(t - k)). Shifts are applied to the full
   series, then each month is selected (Swiss local). Months: Jan and Jul of
   2024, 2025, 2026. Two variants per pair:
       'level' : correlation of the series themselves
       'diff'  : correlation of 15-min first differences (x(t) - x(t-1)),
                 which gives a sharp peak at the true offset
   Pass rule (decided 2026-11-05, see LOG.md):
       Sched_DE_CH       'diff' best k must be 0 (the level curve is flat in
                         some months, e.g. 2024-07, and cannot separate 0 / -1)
       CH_Load_Forecast  'level' |best k| <= 3: it is hourly and too smooth to
                         locate a 15-min offset, but this still catches a
                         whole-hour shift like the old +1 h / +2 h bug
   Rows that are not part of the pass rule are reported with pass = NaN.

2. 2025 spike rate: share of 2025 rows (Swiss local year) where pos_sec_price
   is above the P90 of pos_sec_price over the trailing 90 days strictly
   before T (time window [T-90d, T), rows with NaN price excluded).
   Computed on the new merged file and on the old data/raw_merged_*.csv.

Writes afrr-phase3/data/reports/offset_test.csv and spike_rate_2025.csv.
Usage:  python afrr-phase3/verify_alignment.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import merge_raw_data as m  # noqa: E402

MERGED = os.path.join(m.OUT_DIR, 'merged_2023_2026.parquet')
OLD_MERGED = [os.path.join(m.ROOT, 'data', 'raw_merged_2023_2024.csv'),
              os.path.join(m.ROOT, 'data', 'raw_merged_2025.csv')]
MONTHS = ['2024-01', '2024-07', '2025-01', '2025-07', '2026-01', '2026-07']
# pass rule: (entsoe column, variant) -> max |best offset|
TOL = {('Sched_DE_CH', 'diff'): 0, ('CH_Load_Forecast', 'level'): 3}
LAGS = range(-8, 9)


def offset_test(df):
    sg = pd.concat([m.load_swissgrid_year(y, extra_cols={1: 'consumption_kwh'})
                    [['timestamp_utc', 'consumption_mw']] for y in (2024, 2025, 2026)])
    d = df.merge(sg, on='timestamp_utc', how='left', validate='one_to_one')
    d['net_DE_CH_mw'] = d.DE_CH_mw - d.CH_DE_mw
    ym = d.timestamp_local.dt.strftime('%Y-%m')
    rows = []
    for variant in ('level', 'diff'):
      for ref, ent in [('consumption_mw', 'CH_Load_Forecast'), ('net_DE_CH_mw', 'Sched_DE_CH')]:
        x = d[ref] if variant == 'level' else d[ref].diff()
        y = d[ent] if variant == 'level' else d[ent].diff()
        shifted = {k: y.shift(k) for k in LAGS}
        for mon in MONTHS:
            sel = ym == mon
            r = {k: x[sel].corr(shifted[k][sel]) for k in LAGS}
            best = max(r, key=lambda k: r[k])
            row = {'variant': variant, 'month': mon, 'entsoe': ent, 'swissgrid': ref,
                   'n': int(sel.sum()),
                   'best_offset': best, 'r_best': round(r[best], 4), 'r_0': round(r[0], 4),
                   'r_-1': round(r[-1], 4), 'r_+1': round(r[1], 4),
                   'r_-4': round(r[-4], 4), 'r_+4': round(r[4], 4)}
            rows.append(row)
    return pd.DataFrame(rows)


def spike_rate(ts, price, year_local):
    """ts, price, year_local: aligned 1-D arrays, ts sorted (may repeat)."""
    ts = pd.DatetimeIndex(ts)
    price = np.asarray(price, dtype=float)
    year_local = np.asarray(year_local)
    ok = ~np.isnan(price)
    s = pd.Series(price[ok], index=ts[ok])
    thr = s.rolling('90D', closed='left', min_periods=1).quantile(0.9).to_numpy()
    in_year = year_local[ok] == 2025
    valid = in_year & ~np.isnan(thr)
    spike = s.to_numpy()[valid] > thr[valid]
    return {'n_rows_2025': int(in_year.sum()), 'n_nan_price_2025': int((year_local == 2025).sum() - in_year.sum()),
            'n_spikes': int(spike.sum()), 'spike_rate': round(float(spike.mean()), 5),
            'mean_threshold': round(float(thr[valid].mean()), 3)}


def main():
    df = pd.read_parquet(MERGED)
    print('[1] Offset test')
    ot = offset_test(df)
    tol = [TOL.get((e, v)) for e, v in zip(ot.entsoe, ot.variant)]
    ot['pass'] = [None if t is None else bool(abs(b) <= t) for b, t in zip(ot.best_offset, tol)]
    ot.to_csv(os.path.join(m.REPORT_DIR, 'offset_test.csv'), index=False)
    print(ot.to_string(index=False))
    ok = all(p for p in ot['pass'] if p is not None)
    print(f'  pass (Sched_DE_CH diff best=0, CH_Load_Forecast level |best|<=3): {ok}')

    print('\n[2] 2025 spike rate (trailing 90-day P90, strictly before T)')
    r_new = spike_rate(df.timestamp_utc, df.pos_sec_price, df.timestamp_local.dt.year)
    old = pd.concat([pd.read_csv(f, parse_dates=['timestamp']) for f in OLD_MERGED])
    old = old.sort_values('timestamp', kind='stable')
    r_old = spike_rate(old.timestamp, old.pos_sec_price, old.timestamp.dt.year)
    sr = pd.DataFrame([dict(file='old data/raw_merged_*.csv', **r_old),
                       dict(file='new merged_2023_2026.parquet', **r_new)])
    sr.to_csv(os.path.join(m.REPORT_DIR, 'spike_rate_2025.csv'), index=False)
    print(sr.to_string(index=False))
    if not ok:
        sys.exit('offset test FAILED')


if __name__ == '__main__':
    main()
