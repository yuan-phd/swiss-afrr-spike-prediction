"""
labels.py — spike threshold and label, computed once for every row.

threshold(t) = 90th percentile (linear interpolation) of pos_sec_price over
the time window [t - 90 days, t), i.e. strictly before t; the last row in the
window is t - 15 min. NaN if t - 90 days is before the first row of the
dataset (warm-up). spike(t) = pos_sec_price(t) > threshold(t), NaN where the
threshold is NaN.

The same column serves two purposes (TASK.md):
  * label threshold of a target t:     threshold(t)
  * threshold known to a model at T:   threshold(T)  (uses rows <= T - 15 min)

Writes data/labels.parquet and data/labels_meta.json.
Usage:  python afrr-phase3/labels.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as c  # noqa: E402

META = os.path.join(c.DATA, 'labels_meta.json')


def compute_threshold(ts_utc, price):
    """ts_utc: complete, strictly increasing UTC index. Returns threshold array."""
    s = pd.Series(np.asarray(price, float), index=pd.DatetimeIndex(ts_utc))
    thr = s.rolling(c.THRESH_WINDOW, closed='left').quantile(c.THRESH_Q)
    warm = s.index - c.THRESH_WINDOW < s.index[0]
    thr[warm] = np.nan
    return thr.to_numpy()


def build_labels(df):
    thr = compute_threshold(df.timestamp_utc, df[c.TARGET])
    y = df[c.TARGET].to_numpy()
    spike = np.where(np.isnan(thr), np.nan, (y > thr).astype(float))
    return pd.DataFrame({'timestamp_utc': df.timestamp_utc.reset_index(drop=True),
                         'timestamp_local': df.timestamp_local.reset_index(drop=True),
                         'threshold': thr, 'spike': spike})


def main():
    df = pd.read_parquet(c.MERGED)
    lab = build_labels(df)
    lab.to_parquet(c.LABELS, index=False)

    lab['year'] = lab.timestamp_local.dt.year
    lab['full_hour'] = lab.timestamp_local.dt.minute == 0
    per_year = {}
    for y, g in lab.groupby('year'):
        v = g.spike.dropna()
        vh = g.loc[g.full_hour, 'spike'].dropna()
        per_year[int(y)] = {'rows': int(len(g)), 'labelled': int(len(v)),
                            'spike_rate': float(v.mean()),
                            'labelled_full_hour': int(len(vh)),
                            'spike_rate_full_hour': float(vh.mean())}
    a, b = c.period_bounds_utc('train')
    tr = lab[(lab.timestamp_utc >= a) & (lab.timestamp_utc < b) & lab.full_hour].spike.dropna()
    first_label = lab.loc[lab.threshold.notna(), 'timestamp_utc'].min()
    meta = {'train_spike_rate': float(tr.mean()),
            'train_spike_rate_definition': 'mean spike label over full-hour rows of the '
                                           'training period with a valid label',
            'first_labelled_utc': str(first_label),
            'per_year': per_year}
    with open(META, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'labels: {len(lab):,} rows, first labelled row {first_label}')
    print(f'train spike rate (full-hour rows): {meta["train_spike_rate"]:.5f}')
    print(pd.DataFrame(per_year).T.to_string())


if __name__ == '__main__':
    main()
