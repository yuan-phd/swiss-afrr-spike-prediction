"""
================================================================================
tasks/predict.py — Task 4: Generate Predictions
================================================================================
PURPOSE:
    Load the champion XGBoost model and isotonic calibrator,
    generate spike probability predictions for today's features,
    save predictions to S3.

INPUTS:
    S3: features/{date}/features.parquet     (from Task 3)
    Local: 02_model_training/models/xgboost_spike_classifier.json
    Local: 02_model_training/models/isotonic_calibrator.pkl

OUTPUT:
    S3: predictions/{date}/predictions.parquet

MODEL:
    Two-step prediction (same as original evaluate script):
    1. XGBoost outputs raw probabilities
    2. Isotonic calibrator maps raw → calibrated probabilities
    Calibrated probabilities are better for decision-making (Brier score optimised)

AIRFLOW CONCEPTS:
    xcom_pull : read features path from Task 3 (build_features)
    xcom_push : send predictions path to Task 5 (evaluate)
================================================================================
"""

import io
import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import boto3
import pyarrow as pa
import pyarrow.parquet as pq
from airflow.models import Variable

# ── Configuration ──────────────────────────────────────────────────────────────
S3_BUCKET       = Variable.get('S3_BUCKET', default_var='energy-pipeline')
S3_ENDPOINT_URL = 'http://localhost:4566'

PROJECT_DIR  = '/Users/yuan/Work/profolio/swiss-afrr-spike-prediction'
MODELS_DIR   = os.path.join(PROJECT_DIR, '02_model_training', 'models')

XGB_MODEL_PATH   = os.path.join(MODELS_DIR, 'xgboost_spike_classifier.json')
CALIBRATOR_PATH  = os.path.join(MODELS_DIR, 'isotonic_calibrator.pkl')

# Exact feature order from model_meta.json — must match training order
FEATURE_COLS = [
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


# ── S3 helpers ─────────────────────────────────────────────────────────────────
def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id='test',
        aws_secret_access_key='test',
        region_name='us-east-1',
    )


def read_parquet_from_s3(s3_key: str) -> pd.DataFrame:
    """Read a parquet file from S3 into a DataFrame."""
    s3  = get_s3_client()
    obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
    return pd.read_parquet(io.BytesIO(obj['Body'].read()))


def upload_parquet_to_s3(df: pd.DataFrame, s3_key: str):
    """Upload a DataFrame as parquet to S3."""
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


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model():
    """
    Load XGBoost model and isotonic calibrator from local model directory.

    Why two separate files?
    - XGBoost outputs raw probabilities (not well calibrated)
    - Isotonic calibrator maps raw → calibrated probabilities
    - Both must be loaded and applied in sequence

    In a real production system these would be loaded from S3 (champion model).
    For this pipeline we load from local disk for simplicity.
    """
    if not os.path.exists(XGB_MODEL_PATH):
        raise FileNotFoundError(
            f"XGBoost model not found: {XGB_MODEL_PATH}\n"
            f"Expected: 02_model_training/models/xgboost_spike_classifier.json"
        )
    if not os.path.exists(CALIBRATOR_PATH):
        raise FileNotFoundError(
            f"Calibrator not found: {CALIBRATOR_PATH}\n"
            f"Expected: 02_model_training/models/isotonic_calibrator.pkl"
        )

    # Load XGBoost model from its native JSON format
    model = xgb.XGBClassifier()
    model.load_model(XGB_MODEL_PATH)
    print(f"[predict] XGBoost model loaded from: {XGB_MODEL_PATH}")

    # Load isotonic calibrator (sklearn object saved with joblib)
    calibrator = joblib.load(CALIBRATOR_PATH)
    print(f"[predict] Calibrator loaded from: {CALIBRATOR_PATH}")

    return model, calibrator


# ── Main task function ─────────────────────────────────────────────────────────
def predict(**context):
    """
    Airflow task entry point.

    Steps:
        1. Pull features parquet path from XCom (Task 3)
        2. Read features from S3
        3. Load XGBoost model and calibrator
        4. Generate raw probabilities with XGBoost
        5. Calibrate probabilities with isotonic calibrator
        6. Save predictions parquet to S3
        7. Push predictions path to XCom for Task 5
    """

    # ── 1. Get date and features path ──────────────────────────────────────────
    test_date = Variable.get('PIPELINE_TEST_DATE', default_var=None)
    if test_date:
        date_str = test_date
    else:
        logical_date = context.get('logical_date') or context.get('execution_date')
        date_str = logical_date.strftime('%Y-%m-%d')
    features_key = context['ti'].xcom_pull(
        task_ids='build_features', key='features_path'
    ) or f'features/{date_str}/features.parquet'

    print(f"[predict] Processing date: {date_str}")
    print(f"[predict] Reading features from S3: {features_key}")

    # ── 2. Read features from S3 ───────────────────────────────────────────────
    df = read_parquet_from_s3(features_key)
    print(f"[predict] Features loaded: {len(df)} rows, {len(df.columns)} columns")

    # ── 3. Load model and calibrator ───────────────────────────────────────────
    model, calibrator = load_model()

    # ── 4. Prepare feature matrix ──────────────────────────────────────────────
    # Select exactly the 24 features in the correct order
    # Missing columns (DA_Price_DE etc) stay as NaN — XGBoost handles this
    X = df[FEATURE_COLS].copy()
    print(f"[predict] Feature matrix shape: {X.shape}")
    print(f"[predict] NaN counts: {X.isna().sum().sum()} total NaN values")

    # ── 5. Generate raw probabilities ──────────────────────────────────────────
    # predict_proba returns [[prob_class_0, prob_class_1], ...]
    # We want prob_class_1 = probability of spike
    raw_probs = model.predict_proba(X)[:, 1]
    print(f"[predict] Raw probabilities — "
          f"min: {raw_probs.min():.3f}, "
          f"mean: {raw_probs.mean():.3f}, "
          f"max: {raw_probs.max():.3f}")

    # ── 6. Calibrate probabilities ─────────────────────────────────────────────
    # Isotonic calibration maps raw probabilities to better-calibrated ones
    # The calibrator was trained to minimise Brier score on the validation set
    calibrated_probs = calibrator.transform(raw_probs)
    print(f"[predict] Calibrated probabilities — "
          f"min: {calibrated_probs.min():.3f}, "
          f"mean: {calibrated_probs.mean():.3f}, "
          f"max: {calibrated_probs.max():.3f}")

    # ── 7. Build predictions DataFrame ────────────────────────────────────────
    predictions = pd.DataFrame({
        'timestamp':        df['timestamp'],
        'pos_sec_price':    df['pos_sec_price'],    # actual price (ground truth)
        'prob_raw':         raw_probs,              # before calibration
        'prob_calibrated':  calibrated_probs,       # after calibration (use this)
        'spike_predicted':  (calibrated_probs > 0.3415).astype(int),  # optimal threshold from model_meta
    })

    # Summary statistics
    spike_count = predictions['spike_predicted'].sum()
    print(f"[predict] Predicted {spike_count} spike intervals "
          f"({spike_count/len(predictions)*100:.1f}% of day)")

    # ── 8. Upload predictions to S3 ────────────────────────────────────────────
    s3_key = f"predictions/{date_str}/predictions.parquet"
    upload_parquet_to_s3(predictions, s3_key)
    print(f"[predict] Uploaded to s3://{S3_BUCKET}/{s3_key}")

    # ── 9. Push path to XCom for Task 5 (evaluate) ────────────────────────────
    context['ti'].xcom_push(key='predictions_path', value=s3_key)
    print(f"[predict] XCom pushed predictions_path: {s3_key}")