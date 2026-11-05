"""
compare_entsoe.py — Compare a re-fetched ENTSO-E file (fetch_entsoe.py: UTC,
DE_LU, no filling) with data/raw/entsoe_swiss_energy_data_v2.csv (naive
local, 'DE', forward-filled), column by column.

Rows are aligned on UTC instants, with v2 localised exactly as in
merge_raw_data.py (ambiguous=True). Each period is a range of Swiss local
calendar time [--from, --to). A cell "differs" if |new - old| > 1e-6, or
exactly one side is NaN. The counts are also split per Swiss local year,
and optionally at --split.

Usage (from the repo root):
  python afrr-phase3/compare_entsoe.py --new afrr-phase3/data/raw/entsoe_2025_2026.csv \
      --from 2025-01-01 --to 2026-01-01 --split 2025-10-01 \
      --out afrr-phase3/data/reports/entsoe_2025_diff.csv
  python afrr-phase3/compare_entsoe.py --new afrr-phase3/data/raw/entsoe_2023_2024.csv \
      --from 2023-01-01 --to 2025-01-01 \
      --out afrr-phase3/data/reports/entsoe_2023_2024_diff.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import merge_raw_data as m  # noqa: E402

OLD = os.path.join(m.OLD_RAW, 'entsoe_swiss_energy_data_v2.csv')
ATOL = 1e-6


def max_abs(absd, mask):
    return float(absd[mask].max()) if mask.any() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--new', required=True)
    ap.add_argument('--from', dest='frm', required=True, help='local date, inclusive')
    ap.add_argument('--to', required=True, help='local date, exclusive')
    ap.add_argument('--split', help='optional local date to split counts at')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    old = m.load_entsoe_source(OLD, 'naive_local', a.frm, a.to)
    new = m.load_entsoe_source(a.new, 'utc', a.frm, a.to)
    j = old.merge(new, on='timestamp_utc', how='outer', suffixes=('_old', '_new'),
                  indicator=True)
    loc = j.timestamp_utc.dt.tz_convert(m.LOCAL_TZ)
    print(f'rows: old={len(old):,} new={len(new):,}  '
          f'only_old={(j._merge == "left_only").sum()} only_new={(j._merge == "right_only").sum()}')
    print('instants only in new (local):', loc[j._merge == 'right_only'].tolist())
    print('instants only in old (local):', loc[j._merge == 'left_only'].tolist())
    both = j._merge == 'both'
    j, loc = j[both], loc[both]

    parts = [(str(y), loc.dt.year == y) for y in sorted(loc.dt.year.unique())]
    if a.split:
        s = pd.Timestamp(a.split, tz=m.LOCAL_TZ)
        parts += [(f'before_{a.split}', loc < s), (f'from_{a.split}', loc >= s)]

    rows = []
    for c in m.ENTSOE_COLS:
        x, y = j[f'{c}_old'], j[f'{c}_new']
        one_nan = x.isna() ^ y.isna()
        absd = (x - y).abs()
        differ = one_nan | (absd > ATOL)
        r = {'column': c, 'n_compared': len(j), 'n_differ': int(differ.sum()),
             'pct_differ': round(100 * differ.mean(), 2),
             'max_abs_diff': max_abs(absd, differ & ~one_nan),
             'n_differ_gt_1': int((absd > 1).sum() + one_nan.sum()),
             'nan_old': int(x.isna().sum()), 'nan_new': int(y.isna().sum())}
        for tag, mask in parts:
            r[f'n_differ_{tag}'] = int((differ & mask).sum())
            r[f'max_abs_{tag}'] = max_abs(absd, differ & ~one_nan & mask)
        ok = x.notna() & y.notna()
        r['corr'] = round(x[ok].corr(y[ok]), 6) if ok.sum() > 2 else np.nan
        rows.append(r)
    rep = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    rep.to_csv(a.out, index=False)
    with pd.option_context('display.width', 300, 'display.max_columns', None):
        print(rep.to_string(index=False))

    # months with differences > 1 MW / EUR, per column
    ym = loc.dt.strftime('%Y-%m')
    print('\ncells with |diff| > 1 (or one-sided NaN) by local month:')
    for c in m.ENTSOE_COLS:
        x, y = j[f'{c}_old'], j[f'{c}_new']
        big = ((x - y).abs() > 1) | (x.isna() ^ y.isna())
        if big.any():
            print(f'  {c}: {big.groupby(ym).sum()[lambda s: s > 0].to_dict()}')

    full = pd.read_csv(OLD, usecols=['timestamp', 'DA_Price_DE'], parse_dates=['timestamp'])
    print('\nold v2 DA_Price_DE rows / non-NaN by local year:')
    print(full.groupby(full.timestamp.dt.year).DA_Price_DE.agg(['size', 'count']).to_string())
    print(f'\nSaved {a.out}')


if __name__ == '__main__':
    main()
