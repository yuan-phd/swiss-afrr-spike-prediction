"""
================================================================================
STEP 00 — Prepare & Validate Features
================================================================================
PURPOSE:
    Load downloaded CSVs, validate all expected columns are present,
    check for nulls, confirm spike rates, and print a readiness report.
    Fails loudly if data is not ready for training.

INPUTS:
    data/processed/features_train_2023_2024.csv
    data/processed/features_val_2025.csv

OUTPUT:
    Console readiness report — no files written
================================================================================
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = os.path.join(os.path.dirname(__file__), "..", "..")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

TRAIN_PATH = os.path.join(ROOT, cfg["paths"]["train_csv"])
VAL_PATH   = os.path.join(ROOT, cfg["paths"]["val_csv"])
TARGET     = cfg["target"]

# All feature columns across all categories
ALL_FEATURES = (
    cfg["features"]["hammer"] +
    cfg["features"]["anvil"] +
    cfg["features"]["incentive"] +
    cfg["features"]["autoregressive"] +
    cfg["features"]["structural"]
)


def check_file(path: str, name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"  ❌ {name} not found: {path}")
        sys.exit(1)
    df = pd.read_csv(path, parse_dates=["timestamp"])
    print(f"  ✅ {name}: {len(df):,} rows loaded")
    return df


def check_columns(df: pd.DataFrame, name: str, required: list) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"\n  ❌ {name} missing columns:")
        for c in missing:
            print(f"       {c}")
        sys.exit(1)
    print(f"  ✅ {name}: all {len(required)} required columns present")


def check_nulls(df: pd.DataFrame, name: str, cols: list) -> None:
    null_counts = {c: df[c].isna().sum() for c in cols if df[c].isna().any()}
    if null_counts:
        print(f"\n  ⚠️  {name} null values in feature columns:")
        for col, count in null_counts.items():
            pct = count / len(df) * 100
            print(f"       {col:<35} {count:>6,} ({pct:.1f}%)")
    else:
        print(f"  ✅ {name}: no nulls in feature columns")


def check_spike_rates(train: pd.DataFrame, val: pd.DataFrame) -> None:
    print(f"\n  Spike rates:")
    for label in ["price_spike_fixed", "price_spike_rolling", "price_spike_pooled"]:
        if label in train.columns:
            t_rate = train[label].mean() * 100
            v_rate = val[label].mean()   * 100
            gap    = abs(v_rate - t_rate)
            status = "✅" if gap < 5 else "⚠️ " if gap < 10 else "🚨"
            print(f"  {status} {label:<30}  Train {t_rate:.1f}%  Val {v_rate:.1f}%  "
                  f"Gap {gap:.1f}%")


def check_date_ranges(train: pd.DataFrame, val: pd.DataFrame) -> None:
    print(f"\n  Date ranges:")
    print(f"    Train: {train['timestamp'].min().date()} → "
          f"{train['timestamp'].max().date()}")
    print(f"    Val:   {val['timestamp'].min().date()} → "
          f"{val['timestamp'].max().date()}")

    # Check no overlap
    train_max = train["timestamp"].max()
    val_min   = val["timestamp"].min()
    if val_min <= train_max:
        print(f"  ❌ OVERLAP detected between train and val — data leakage risk!")
        sys.exit(1)
    else:
        print(f"  ✅ No overlap — clean train/val split confirmed")


def feature_summary(train: pd.DataFrame) -> None:
    print(f"\n  Feature summary (training set):")
    print(f"  {'Feature':<35} {'Mean':>10} {'Std':>10} {'Null%':>7}")
    print(f"  {'-'*65}")

    categories = {
        "HAMMER":        cfg["features"]["hammer"],
        "ANVIL":         cfg["features"]["anvil"],
        "INCENTIVE":     cfg["features"]["incentive"],
        "AUTOREGRESSIVE":cfg["features"]["autoregressive"],
        "STRUCTURAL":    cfg["features"]["structural"],
    }

    for cat, cols in categories.items():
        print(f"\n  [{cat}]")
        for col in cols:
            if col not in train.columns:
                print(f"  ⚠️  {col:<33} MISSING")
                continue
            mean    = train[col].mean()
            std     = train[col].std()
            null_pct = train[col].isna().mean() * 100
            print(f"  {'':2}{col:<33} {mean:>10.2f} {std:>10.2f} {null_pct:>6.1f}%")


def main():
    print("=" * 60)
    print("STEP 00 — Feature Validation & Readiness Check")
    print("=" * 60)

    # ── 1. Load files ──────────────────────────────────────────────────────────
    print("\n[1] Loading files...")
    train = check_file(TRAIN_PATH, "Train (2023-2024)")
    val   = check_file(VAL_PATH,   "Val   (2025)")

    # ── 2. Check columns ───────────────────────────────────────────────────────
    print("\n[2] Checking required columns...")
    required = ALL_FEATURES + [TARGET, "timestamp", "pos_sec_price",
                                "rolling_p90_threshold"]
    check_columns(train, "Train", required)
    check_columns(val,   "Val",   required)

    # ── 3. Check column match ──────────────────────────────────────────────────
    print("\n[3] Checking train/val column match...")
    if train.columns.tolist() == val.columns.tolist():
        print("  ✅ Column lists match exactly")
    else:
        extra_train = set(train.columns) - set(val.columns)
        extra_val   = set(val.columns)   - set(train.columns)
        if extra_train:
            print(f"  ⚠️  Only in train: {extra_train}")
        if extra_val:
            print(f"  ⚠️  Only in val:   {extra_val}")

    # ── 4. Check nulls ─────────────────────────────────────────────────────────
    print("\n[4] Checking nulls in feature columns...")
    check_nulls(train, "Train", ALL_FEATURES)
    check_nulls(val,   "Val",   ALL_FEATURES)

    # ── 5. Check date ranges ───────────────────────────────────────────────────
    print("\n[5] Checking date ranges...")
    check_date_ranges(train, val)

    # ── 6. Check spike rates ───────────────────────────────────────────────────
    print("\n[6] Checking spike rates...")
    check_spike_rates(train, val)

    # ── 7. Check monotonic constraint columns exist ────────────────────────────
    print("\n[7] Checking monotonic constraint columns...")
    for col in cfg["monotonic_constraints"]:
        if col in train.columns:
            print(f"  ✅ {col}")
        else:
            print(f"  ❌ {col} — missing, check feature engineering")

    # ── 8. Feature summary ─────────────────────────────────────────────────────
    print("\n[8] Feature summary...")
    feature_summary(train)

    # ── 9. Rolling threshold sanity check ─────────────────────────────────────
    print("\n[9] Rolling threshold sanity check...")
    rt_min  = train["rolling_p90_threshold"].min()
    rt_mean = train["rolling_p90_threshold"].mean()
    rt_max  = train["rolling_p90_threshold"].max()
    print(f"  Min:  {rt_min:.1f} EUR/MWh  "
          f"{'✅' if rt_min > 50 else '⚠️  clip needed — too low'}")
    print(f"  Mean: {rt_mean:.1f} EUR/MWh")
    print(f"  Max:  {rt_max:.1f} EUR/MWh")

    if rt_min < 50:
        n_low = (train["rolling_p90_threshold"] < 50).sum()
        print(f"  ⚠️  {n_low} rows have rolling threshold < 50 EUR/MWh")
        print(f"     These are early 2023 rows with no rolling history")
        print(f"     Will be clipped to min=50 during training")

    # ── Final verdict ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅ DATA IS READY FOR TRAINING")
    print(f"   Train: {len(train):,} rows  |  "
          f"Val: {len(val):,} rows  |  "
          f"Features: {len(ALL_FEATURES)}")
    print(f"   Target: {TARGET}")
    print(f"   Spike rate train: {train[TARGET].mean()*100:.1f}%  |  "
          f"Val: {val[TARGET].mean()*100:.1f}%")
    print("=" * 60)
    print("\n[STEP 00 COMPLETE]")


if __name__ == "__main__":
    main()
