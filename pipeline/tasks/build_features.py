"""
================================================================================
tasks/build_features.py — Task 3: Build Features
================================================================================
PURPOSE:
    Merge ENTSO-E data (from S3) with Swissgrid data (from local XLSX),
    compute all 24 features the model expects, save as parquet to S3.

INPUTS:
    S3: raw/{date}/entsoe.csv           (from Task 1)
    Local: data/raw/EnergieUebersichtCH-{year}.xlsx  (latest Swissgrid file)

OUTPUT:
    S3: features/{date}/features.parquet

FEATURE ENGINEERING:
    Same logic as 01_feature_engineering.py — adapted for a single day.
    All 24 features from model_meta.json are computed here.

WHY PARQUET:
    Parquet stores data types natively (float stays float).
    CSV re-parses everything as text on every read.
    For ML feature data, parquet is faster and safer.

AIRFLOW CONCEPTS:
    xcom_pull   : read S3 path from Task 2 (validate_data)
    xcom_push   : send features parquet path to Task 4 (predict)
================================================================================
"""

import io
import os
import glob
import pandas as pd
import numpy as np
import boto3
import pyarrow as pa
import pyarrow.parquet as pq
from airflow.models import Variable

# ── Configuration ──────────────────────────────────────────────────────────────
S3_BUCKET       = Variable.get('S3_BUCKET', default_var='energy-pipeline')
S3_ENDPOINT_URL = 'http://localhost:4566'

# Path to Swissgrid raw data folder
PROJECT_DIR  = '/Users/yuan/Work/profolio/swiss-afrr-spike-prediction'
RAW_DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'raw')

# Swissgrid column indices and names (from 00_load_and_preprocess.py)
SG_COL_INDICES = [0, 6, 7, 12, 13, 14, 15, 21, 22]
SG_COL_NAMES   = [
    'timestamp',
    'pos_sec_vol_kwh',
    'neg_sec_vol_kwh',
    'CH_DE_kwh',
    'DE_CH_kwh',
    'CH_FR_kwh',
    'FR_CH_kwh',
    'pos_sec_price',
    'neg_sec_price',
]
KWH_TO_MW = 4 / 1000  # kWh per 15-min → MW


# ── S3 helpers ─────────────────────────────────────────────────────────────────
def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id='test',
        aws_secret_access_key='test',
        region_name='us-east-1',
    )


def read_csv_from_s3(s3_key: str) -> pd.DataFrame:
    s3  = get_s3_client()
    obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
    return pd.read_csv(
        io.StringIO(obj['Body'].read().decode('utf-8')),
        parse_dates=['timestamp']
    )


def upload_parquet_to_s3(df: pd.DataFrame, s3_key: str):
    """Convert DataFrame to parquet bytes and upload to S3."""
    buffer = io.BytesIO()
    table  = pa.Table.from_pandas(df)
    pq.write_table(table, buffer)
    buffer.seek(0)

    s3 = get_s3_client()
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=s3_key,
        Body=buffer.getvalue(),
        ContentType='application/octet-stream',
    )


# ── Swissgrid loading (from 00_load_and_preprocess.py) ────────────────────────
def load_latest_swissgrid(execution_date) -> pd.DataFrame:
    """
    Load the most recent Swissgrid XLSX that covers the execution date.
    Looks for EnergieUebersichtCH-{year}.xlsx files in data/raw/.
    Falls back to the latest available year if exact year not found.
    """
    year = execution_date.year

    # Try exact year first, then fall back to latest available
    pattern  = os.path.join(RAW_DATA_DIR, 'EnergieUebersichtCH-*.xlsx')
    files    = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No Swissgrid XLSX files found in {RAW_DATA_DIR}. "
            f"Please download EnergieUebersichtCH-{year}.xlsx manually."
        )

    # Find the file for this year, or use the latest available
    target = os.path.join(RAW_DATA_DIR, f'EnergieUebersichtCH-{year}.xlsx')
    if os.path.exists(target):
        filepath = target
        print(f"[build_features] Loading Swissgrid {year}: {filepath}")
    else:
        filepath = files[-1]  # latest available
        print(f"[build_features] ⚠️ {year} XLSX not found, using: {filepath}")

    # Load — same logic as load_swissgrid_year() in 00_load_and_preprocess.py
    df = pd.read_excel(
        filepath,
        sheet_name='Zeitreihen0h15',
        header=None,
        skiprows=2,
    )
    df = df.iloc[:, SG_COL_INDICES].copy()
    df.columns = SG_COL_NAMES

    # Parse timestamps (DD.MM.YYYY HH:MM format)
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Convert kWh → MW (same formula as original)
    for col in ['pos_sec_vol_kwh', 'neg_sec_vol_kwh',
                'CH_DE_kwh', 'DE_CH_kwh', 'CH_FR_kwh', 'FR_CH_kwh']:
        mw_col = col.replace('_kwh', '_mw')
        df[mw_col] = df[col] * KWH_TO_MW

    df = df.drop(columns=['pos_sec_vol_kwh', 'neg_sec_vol_kwh',
                           'CH_DE_kwh', 'DE_CH_kwh', 'CH_FR_kwh', 'FR_CH_kwh'])

    print(f"[build_features] Swissgrid loaded: {len(df):,} rows "
          f"({df['timestamp'].min().date()} → {df['timestamp'].max().date()})")
    return df


# ── Feature engineering (from 01_feature_engineering.py) ──────────────────────
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 24 features from model_meta.json.
    Same logic as 01_feature_engineering.py — adapted for single-day data.

    NOTE: Lag features (price_delta_lag1, lag4, lag96) and rolling windows
    require HISTORICAL data to compute correctly for the first row of the day.
    We load the full Swissgrid history and filter to the target date AFTER
    computing rolling/lag features — this prevents NaN at day boundaries.
    """

    # ── Net flows (from 01_feature_engineering.py: compute_net_flows) ─────────
    df['actual_net_DE_to_CH'] = df['DE_CH_mw'] - df['CH_DE_mw']
    df['actual_net_FR_to_CH'] = df['FR_CH_mw'] - df['CH_FR_mw']

    # ── Unplanned flows (compute_unplanned_flow) ───────────────────────────────
    # Unplanned_Flow = actual physical flow - scheduled commercial flow
    # This is the core causal variable: German wind errors → loop flows → aFRR spikes
    df['Unplanned_Flow']     = df['actual_net_DE_to_CH'] - df['Sched_DE_CH']
    df['abs_Unplanned_Flow'] = df['Unplanned_Flow'].abs()

    # FR-CH unplanned flow
    df['Unplanned_Flow_FR_CH']     = df['actual_net_FR_to_CH'] - df['Sched_FR_CH']
    df['abs_Unplanned_Flow_FR_CH'] = df['Unplanned_Flow_FR_CH'].abs()

    # Total unplanned flow (sum of both borders)
    df['Total_Unplanned_Flow'] = df['Unplanned_Flow'] + df['Unplanned_Flow_FR_CH']

    # ── Rolling unplanned flow (4-period = 1 hour lookback) ───────────────────
    # .shift(1) prevents look-ahead bias — excludes current row from window
    df['Unplanned_Flow_rolling4'] = (
        df['Unplanned_Flow'].shift(1).rolling(4).mean()
    )

    # ── Schedule features ──────────────────────────────────────────────────────
    # Delta between consecutive scheduled values — detects ramp changes
    df['Sched_DE_CH_delta'] = df['Sched_DE_CH'].diff()

    # Interaction: unplanned flow × scheduled flow
    # Captures when loop flows AND schedules both push in same direction
    df['Unplanned_x_Sched'] = df['Unplanned_Flow'] * df['Sched_DE_CH']

    # ── Rolling P90 threshold (from training set) ──────────────────────────────
    # 90th percentile of pos_sec_price over trailing 30 days (2880 periods)
    # .shift(1) prevents current price leaking into its own threshold
    df['rolling_p90_threshold'] = (
        df['pos_sec_price']
        .shift(1)
        .rolling(2880, min_periods=96)
        .quantile(0.90)
    )
    # Clip to minimum 50 EUR/MWh (early rows have no rolling history)
    df['rolling_p90_threshold'] = df['rolling_p90_threshold'].clip(lower=50)

    # ── Price lag features ─────────────────────────────────────────────────────
    # lag1 = 15 min ago, lag4 = 1 hour ago, lag96 = 24 hours ago
    # These capture autocorrelation in aFRR prices
    df['price_delta_lag1']  = df['pos_sec_price'].shift(1)
    df['price_delta_lag4']  = df['pos_sec_price'].shift(4)
    df['price_delta_lag96'] = df['pos_sec_price'].shift(96)

    # ── Price vs threshold lag features ───────────────────────────────────────
    # Was price above the rolling threshold 15 min / 1 hour ago?
    # Binary signal: 1 = was spiking recently
    df['price_vs_threshold_lag1'] = (
        (df['price_delta_lag1'] > df['rolling_p90_threshold'])
        .astype(int)
    )
    df['price_vs_threshold_lag4'] = (
        (df['price_delta_lag4'] > df['rolling_p90_threshold'])
        .astype(int)
    )

    # ── Time features ──────────────────────────────────────────────────────────
    df['minute']      = df['timestamp'].dt.minute
    df['is_turnover'] = df['minute'].isin([0, 45]).astype(int)

    return df


# ── Main task function ─────────────────────────────────────────────────────────
def build_features(**context):
    """
    Airflow task entry point.

    Steps:
        1. Pull validated ENTSO-E path from XCom (Task 2)
        2. Read ENTSO-E data from S3
        3. Load full Swissgrid history from local XLSX
        4. Merge on timestamp
        5. Compute all 24 features
        6. Filter to target date only
        7. Upload as parquet to S3
        8. Push parquet path to XCom for Task 4
    """

    # ── 1. Get date and S3 path ────────────────────────────────────────────────
    test_date = Variable.get('PIPELINE_TEST_DATE', default_var=None)
    if test_date:
        date_str = test_date
        print(f"[build_features] Using test date: {date_str}")
    else:
        logical_date = context.get('logical_date') or context.get('execution_date')
        date_str = logical_date.strftime('%Y-%m-%d')
    print(f"[build_features] Processing date: {date_str}")

    entsoe_s3key = context['ti'].xcom_pull(
        task_ids='validate_data', key='validated_path'
    ) or f'raw/{date_str}/entsoe.csv'

    # ── 2. Read ENTSO-E data from S3 ──────────────────────────────────────────
    print(f"[build_features] Reading ENTSO-E from S3: {entsoe_s3key}")
    entsoe = read_csv_from_s3(entsoe_s3key)
    print(f"[build_features] ENTSO-E: {len(entsoe)} rows, columns: {entsoe.columns.tolist()}")

    # ── 3. Load full Swissgrid history ────────────────────────────────────────
    # We load the FULL history (not just today) because:
    # - Rolling features (rolling_p90_threshold, Unplanned_Flow_rolling4)
    #   need historical rows to be meaningful for today's first row
    # - Without history, first row of the day would have NaN lag features
    sg_full = load_latest_swissgrid(pd.Timestamp(date_str))

    # ── 4. Merge ENTSO-E + Swissgrid on timestamp ──────────────────────────────
    # Inner join — only keep timestamps present in both datasets
    merged = pd.merge(sg_full, entsoe, on='timestamp', how='inner')
    merged = merged.sort_values('timestamp').reset_index(drop=True)
    print(f"[build_features] Merged: {len(merged):,} rows total")

    # ── 5. Compute all 24 features on full history ─────────────────────────────
    # We compute on the full merged dataset so rolling/lag features
    # have enough history at the start of the target date
    merged = compute_features(merged)

    # ── 6. Filter to target date only ─────────────────────────────────────────
    target_date = pd.Timestamp(date_str).date()
    daily = merged[merged['timestamp'].dt.date == target_date].copy()
    print(f"[build_features] Filtered to {date_str}: {len(daily)} rows")

    if len(daily) == 0:
        raise ValueError(
            f"[build_features] No rows found for {date_str} after merge. "
            f"Swissgrid data may not cover this date yet."
        )

    # ── 7. Select only the 24 model features + metadata ───────────────────────
    # Exactly the features from model_meta.json
    feature_cols = [
        'timestamp',
        'pos_sec_price',          # target variable (for evaluate task)
        'Unplanned_Flow',
        'abs_Unplanned_Flow',
        'Unplanned_Flow_rolling4',
        'DE_WindSolar_Error',
        'Unplanned_Flow_FR_CH',
        'abs_Unplanned_Flow_FR_CH',
        'Total_Unplanned_Flow',
        'Sched_DE_CH',
        'Sched_DE_CH_delta',
        'Sched_FR_CH',
        'Unplanned_x_Sched',
        'rolling_p90_threshold',
        'CH_Load_Forecast',
        'CH_Pump_Gen',
        'DA_Price_DE',
        'DA_Price_CH',
        'DA_Price_Spread_DE_CH',
        'price_delta_lag1',
        'price_delta_lag4',
        'price_delta_lag96',
        'price_vs_threshold_lag1',
        'price_vs_threshold_lag4',
        'minute',
        'is_turnover',
    ]

    # Only keep columns that exist (DA_Price_DE may be missing if fetch failed)
    available_cols = [c for c in feature_cols if c in daily.columns]
    missing_cols   = [c for c in feature_cols if c not in daily.columns]

    if missing_cols:
        print(f"[build_features] ⚠️ Missing columns (will be NaN): {missing_cols}")
        for col in missing_cols:
            daily[col] = np.nan

    daily = daily[feature_cols].copy()

    # ── 8. Summary ─────────────────────────────────────────────────────────────
    print(f"[build_features] Feature summary:")
    for col in feature_cols[2:]:  # skip timestamp and pos_sec_price
        null_pct = daily[col].isna().mean() * 100
        status   = "✅" if null_pct == 0 else "⚠️" if null_pct < 20 else "❌"
        print(f"  {status} {col:<35} {null_pct:.0f}% null")

    # ── 9. Upload parquet to S3 ────────────────────────────────────────────────
    s3_key = f"features/{date_str}/features.parquet"
    upload_parquet_to_s3(daily, s3_key)
    print(f"[build_features] Uploaded to s3://{S3_BUCKET}/{s3_key}")

    # ── 10. Push path to XCom for Task 4 (predict) ────────────────────────────
    context['ti'].xcom_push(key='features_path', value=s3_key)
    print(f"[build_features] XCom pushed features_path: {s3_key}")