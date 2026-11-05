"""
compare_swissgrid_2025.py — Compare the republished Swissgrid 2025 file
(afrr-phase3/data/raw/, last-modified 19 May 2026) with the old copy in data/raw/.

For every column of sheet Zeitreihen0h15: number of differing cells and max
absolute difference. Rows are aligned by position after checking that the
timestamp columns match. Writes afrr-phase3/data/reports/swissgrid_2025_diff.csv.

Usage:  python afrr-phase3/compare_swissgrid_2025.py
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OLD = os.path.join(ROOT, 'data', 'raw', 'EnergieUebersichtCH-2025.xlsx')
NEW = os.path.join(HERE, 'data', 'raw', 'EnergieUebersichtCH-2025.xlsx')
OUT = os.path.join(HERE, 'data', 'reports', 'swissgrid_2025_diff.csv')


def load(path):
    hdr = pd.read_excel(path, sheet_name='Zeitreihen0h15', header=None, nrows=2)
    df = pd.read_excel(path, sheet_name='Zeitreihen0h15', header=None, skiprows=2)
    names = ['timestamp'] + [str(hdr.iloc[0, i]).split('\n')[-1].strip()
                             for i in range(1, hdr.shape[1])]
    units = [str(hdr.iloc[1, i]) for i in range(hdr.shape[1])]
    df.columns = names
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
    return df, units


def main():
    old, u_old = load(OLD)
    new, u_new = load(NEW)
    print(f'old rows: {len(old):,}  new rows: {len(new):,}')
    assert list(old.columns) == list(new.columns), 'column headers differ'
    assert u_old == u_new, 'units differ'
    assert len(old) == len(new), 'row counts differ'
    ts_diff = int((old['timestamp'] != new['timestamp']).sum())
    print(f'timestamp cells differing: {ts_diff}')
    assert ts_diff == 0, 'timestamps differ; positional comparison invalid'

    rows = []
    for i, c in enumerate(old.columns[1:], start=1):
        a = pd.to_numeric(old[c], errors='coerce')
        b = pd.to_numeric(new[c], errors='coerce')
        both_nan = a.isna() & b.isna()
        differ = ~both_nan & ~np.isclose(a, b, rtol=0, atol=1e-9, equal_nan=False)
        absdiff = (a - b).abs()
        changed_ts = old.loc[differ, 'timestamp']
        rows.append({
            'col_idx': i, 'column': c, 'unit': u_old[i],
            'n_differ': int(differ.sum()),
            'max_abs_diff': float(absdiff[differ].max()) if differ.any() else 0.0,
            'old_nan': int(a.isna().sum()), 'new_nan': int(b.isna().sum()),
            'first_changed': changed_ts.min() if differ.any() else pd.NaT,
            'last_changed': changed_ts.max() if differ.any() else pd.NaT,
        })
    rep = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rep.to_csv(OUT, index=False)
    with pd.option_context('display.max_rows', None, 'display.width', 250,
                           'display.max_colwidth', 60):
        print(rep.to_string(index=False))
    print(f'\nSaved {OUT}')


if __name__ == '__main__':
    main()
