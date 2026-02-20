"""
================================================================================
STEP 01 — Feature Engineering & Train/Validation Split
================================================================================
PURPOSE:
    Merge Swissgrid and ENTSO-E data (both now covering 2023–2025), compute
    all derived features, then split explicitly into training (2023–2024) and
    validation (2025) sets.

INPUTS:
    data/processed/swissgrid_clean_2023_2025.csv
    data/processed/entsoe_clean_2023_2025.csv

OUTPUTS:
    data/processed/features_train_2023_2024.csv  — training set (model learns from this)
    data/processed/features_val_2025.csv         — validation set (held out until final eval)

KEY DERIVED FEATURES:
    ┌─────────────────────────────────────────────────────────────────────┐
    │ Feature                  │ Formula                                  │
    ├─────────────────────────────────────────────────────────────────────┤
    │ actual_net_DE_to_CH      │ DE_CH_mw − CH_DE_mw                     │
    │                          │ Net physical flow Germany → Switzerland  │
    │                          │ Positive = DE pushes power into CH       │
    │                          │ Negative = CH exports to DE              │
    ├─────────────────────────────────────────────────────────────────────┤
    │ Unplanned_Flow           │ actual_net_DE_to_CH − Sched_DE_CH       │
    │                          │ = Loop Flow / Unscheduled Transit        │
    │                          │ The "cause" variable in our hypothesis   │
    ├─────────────────────────────────────────────────────────────────────┤
    │ abs_Unplanned_Flow       │ |Unplanned_Flow|                         │
    │                          │ Used for non-linear U-shape analysis     │
    ├─────────────────────────────────────────────────────────────────────┤
    │ actual_net_FR_to_CH      │ FR_CH_mw − CH_FR_mw                     │
    │                          │ Net flow France → Switzerland            │
    ├─────────────────────────────────────────────────────────────────────┤
    │ neg_sec_abs_mw           │ |neg_sec_vol_mw|                         │
    │                          │ Absolute value for downward activation   │
    ├─────────────────────────────────────────────────────────────────────┤
    │ net_sec_mw               │ pos_sec_vol_mw + neg_sec_vol_mw         │
    │                          │ Net aFRR balance (pos-biased = deficit)  │
    ├─────────────────────────────────────────────────────────────────────┤
    │ price_spike              │ pos_sec_price > P90(pos_sec_price)       │
    │                          │ Binary flag for extreme price events     │
    ├─────────────────────────────────────────────────────────────────────┤
    │ hour, minute, hh_mm      │ Time-of-day features                    │
    ├─────────────────────────────────────────────────────────────────────┤
    │ is_turnover              │ minute ∈ {0, 45}                         │
    │                          │ Schedule transition windows              │
    └─────────────────────────────────────────────────────────────────────┘

SPLIT STRATEGY:
    Training set:   2023–2024 (model learns causal relationships)
    Validation set: 2025      (completely held out for final evaluation)

    The spike threshold (P90 of positive price) is computed on the TRAINING
    SET ONLY, then applied to the validation set. This prevents data leakage —
    the model cannot "know" what counts as a spike in 2025 during training.

NOTE ON SIGN CONVENTIONS:
    - pos_sec_vol_mw  ≥ 0   (upward regulation: system in deficit)
    - neg_sec_vol_mw  ≤ 0   (downward regulation: system in surplus)
    - Sched_DE_CH > 0 means Germany scheduled to export to Switzerland
    - actual_net_DE_to_CH > 0 means Germany physically pushed more than zero

NOTE ON WEAK LINEAR CORRELATIONS:
    The Pearson r between Unplanned_Flow and aFRR metrics is intentionally
    low (~0.01–0.03). This is NOT a data problem — it reflects the physical
    mechanism: aFRR volume is capacity-bounded (~500 MW), and price responds
    non-linearly (threshold / bid-ladder exhaustion effect). Tree-based models
    (XGBoost, LightGBM) capture this; linear regression cannot.
================================================================================
"""

import os
import pandas as pd
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

# Input filenames (outputs from Step 00)
SG_IN     = "swissgrid_clean_2023_2025.csv"
ENTSOE_IN = "entsoe_clean_2023_2025.csv"

# Output filenames — clearly named by purpose and date range
TRAIN_OUT = "features_train_2023_2024.csv"
VAL_OUT   = "features_val_2025.csv"

# Price spike threshold: top decile of positive aFRR price
SPIKE_QUANTILE = 0.90


# ── Feature Engineering Functions ────────────────────────────────────────────

def compute_net_flows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute net directional flows on key borders.
    Convention: positive = import into Switzerland, negative = export from CH.
    """
    df["actual_net_DE_to_CH"] = df["DE_CH_mw"] - df["CH_DE_mw"]
    df["actual_net_FR_to_CH"] = df["FR_CH_mw"] - df["CH_FR_mw"]
    return df


def compute_unplanned_flow(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Unplanned Loop Flow on the DE-CH border.

    Formula:
        Unplanned_Flow = Actual Physical Net Flow − Scheduled Commercial Flow

    Interpretation:
        > 0 → Germany delivered MORE than scheduled → surplus pushed into CH
        < 0 → Germany delivered LESS than scheduled → CH must compensate

    Physical basis:
        The scheduled commercial flow (Sched_DE_CH from ENTSO-E) represents
        what market participants planned at day-ahead gate closure. Deviations
        are caused by real-time surprises — primarily German wind/solar forecast
        errors — and manifest as unplanned loop flows through Switzerland.
    """
    df["Unplanned_Flow"]     = df["actual_net_DE_to_CH"] - df["Sched_DE_CH"]
    df["abs_Unplanned_Flow"] = df["Unplanned_Flow"].abs()
    return df


def compute_afrr_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived aFRR volume features.

    pos_sec_vol_mw  ≥ 0 (upward: Swissgrid buys more generation)
    neg_sec_vol_mw  ≤ 0 (downward: Swissgrid reduces generation)
    """
    df["neg_sec_abs_mw"] = df["neg_sec_vol_mw"].abs()
    df["net_sec_mw"]     = df["pos_sec_vol_mw"] + df["neg_sec_vol_mw"]
    return df


def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-of-day features used in structural analysis.

    hour        : 0–23
    minute      : 0, 15, 30, or 45 (15-min grid)
    hh_mm       : total minutes from midnight (0–1425)
    is_turnover : 1 for XX:00 and XX:45 (schedule transition windows)
    """
    df["hour"]        = df["timestamp"].dt.hour
    df["minute"]      = df["timestamp"].dt.minute
    df["hh_mm"]       = df["hour"] * 60 + df["minute"]
    df["is_turnover"] = df["minute"].isin([0, 45]).astype(int)
    return df


def apply_spike_flag(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Apply a pre-computed price spike threshold to label extreme price events.
    The threshold must always come from the training set to avoid data leakage.
    """
    df["price_spike"]    = (df["pos_sec_price"] > threshold).astype(int)
    df["spike_threshold"] = threshold
    return df


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def main():
    # ── 1. Load processed files ────────────────────────────────────────────────
    print("Loading processed data...")
    sg = pd.read_csv(
        os.path.join(PROCESSED_DIR, SG_IN), parse_dates=["timestamp"])
    entsoe = pd.read_csv(
        os.path.join(PROCESSED_DIR, ENTSOE_IN), parse_dates=["timestamp"])

    print(f"  Swissgrid: {len(sg):,} rows  "
          f"({sg['timestamp'].min().date()} → {sg['timestamp'].max().date()})")
    print(f"  ENTSO-E:   {len(entsoe):,} rows  "
          f"({entsoe['timestamp'].min().date()} → {entsoe['timestamp'].max().date()})")

    # ── 2. Merge on timestamp (inner join) ────────────────────────────────────
    # Both datasets now cover 2023–2025 fully.
    # Inner join keeps only timestamps present in both.
    df = pd.merge(sg, entsoe, on="timestamp", how="inner")
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"\nMerged: {len(df):,} rows")
    print(f"  Date range: {df['timestamp'].min().date()}  →  {df['timestamp'].max().date()}")

    # ── 3. Feature engineering ────────────────────────────────────────────────
    print("\nComputing features...")
    df = compute_net_flows(df)
    df = compute_unplanned_flow(df)
    df = compute_afrr_features(df)
    df = compute_time_features(df)

    # Drop rows missing any key causal variable
    key_cols = ["Unplanned_Flow", "DE_WindSolar_Error",
                "pos_sec_price", "neg_sec_price"]
    before = len(df)
    df = df.dropna(subset=key_cols)
    print(f"  Dropped {before - len(df):,} rows with missing key variables")
    print(f"  Rows after cleaning: {len(df):,}")

    # ── 4. Train / validation split ───────────────────────────────────────────
    # Split AFTER feature engineering so all features are computed once.
    # 2025 is held out completely — do not touch it until final evaluation.
    train = df[df["timestamp"].dt.year <= 2024].copy()
    val   = df[df["timestamp"].dt.year == 2025].copy()

    print(f"\nTrain set: {len(train):,} rows  "
          f"({train['timestamp'].min().date()} → {train['timestamp'].max().date()})")
    print(f"Val set:   {len(val):,} rows  "
          f"({val['timestamp'].min().date()} → {val['timestamp'].max().date()})")

    # ── 5. Price spike flag — computed on train, applied to val ───────────────
    # IMPORTANT: threshold comes from training set only.
    # If we computed P90 on all data including 2025, we would be leaking
    # future information into the training labels.
    print(f"\nComputing price spike flag (P{SPIKE_QUANTILE*100:.0f} threshold)...")
    threshold = train["pos_sec_price"].quantile(SPIKE_QUANTILE)
    print(f"  Threshold (from training set): {threshold:.2f} EUR/MWh")

    train = apply_spike_flag(train, threshold)
    val   = apply_spike_flag(val,   threshold)

    print(f"  Train spike rate: {train['price_spike'].mean()*100:.1f}%  "
          f"({train['price_spike'].sum():,} events)")
    print(f"  Val spike rate:   {val['price_spike'].mean()*100:.1f}%  "
          f"({val['price_spike'].sum():,} events)")

    # ── 6. Feature summary ─────────────────────────────────────────────────────
    print("\nFeature summary (training set):")
    feature_cols = [
        "actual_net_DE_to_CH", "Unplanned_Flow", "abs_Unplanned_Flow",
        "actual_net_FR_to_CH", "neg_sec_abs_mw", "net_sec_mw",
        "DE_WindSolar_Error", "Sched_DE_CH", "CH_Pump_Gen",
    ]
    print(train[feature_cols].describe().round(1).to_string())

    # ── 7. Save outputs ────────────────────────────────────────────────────────
    train_out = os.path.join(PROCESSED_DIR, TRAIN_OUT)
    val_out   = os.path.join(PROCESSED_DIR, VAL_OUT)

    train.to_csv(train_out, index=False)
    val.to_csv(val_out,   index=False)

    print(f"\nSaved:")
    print(f"  {train_out}")
    print(f"  {val_out}")
    print("\n[STEP 01 COMPLETE]")


if __name__ == "__main__":
    main()
