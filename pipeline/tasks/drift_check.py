"""
================================================================================
tasks/drift_check.py — Task 6: Monitor Feature Drift
================================================================================
PURPOSE:
    Compute PSI (Population Stability Index) on key features.
    Compare today's distribution against the training set baseline.
    Log PSI scores to MLflow.
    Trigger retrain_dag if drift exceeds threshold.

PSI INTERPRETATION:
    PSI < 0.1   → stable, no action needed
    PSI 0.1-0.2 → moderate drift, log warning
    PSI > 0.2   → significant drift, trigger retraining

WHY PSI:
    PSI measures how much a feature's distribution has shifted.
    If the distribution of Unplanned_Flow today looks very different
    from what the model was trained on, predictions will degrade.
    PSI catches this before Brier score degrades significantly.

AIRFLOW CONCEPTS:
    xcom_pull              : read brier_score from Task 5
    TriggerDagRunOperator  : trigger retrain_dag if drift detected
================================================================================
"""

import io
import os
import numpy as np
import pandas as pd
import boto3
import mlflow
from airflow.models import Variable

# ── Configuration ──────────────────────────────────────────────────────────────
S3_BUCKET           = Variable.get('S3_BUCKET', default_var='energy-pipeline')
S3_ENDPOINT_URL     = 'http://localhost:4566'
MLFLOW_TRACKING_URI = Variable.get('MLFLOW_TRACKING_URI',
                                    default_var='http://localhost:5001')
EXPERIMENT_NAME     = 'afrr_spike_prediction'

PROJECT_DIR         = '/Users/yuan/Work/profolio/swiss-afrr-spike-prediction'
TRAIN_DATA_PATH     = os.path.join(PROJECT_DIR, 'data', 'processed',
                                    'features_train_2023_2024.csv')

# PSI thresholds
PSI_WARNING_THRESHOLD = 0.1
PSI_RETRAIN_THRESHOLD = 0.2

# Features to monitor — the most important ones from SHAP analysis
MONITOR_FEATURES = [
    'abs_Unplanned_Flow',
    'DE_WindSolar_Error',
    'Sched_DE_CH',
    'DA_Price_CH',
    'CH_Load_Forecast',
]


# ── S3 helper ──────────────────────────────────────────────────────────────────
def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id='test',
        aws_secret_access_key='test',
        region_name='us-east-1',
    )


def read_parquet_from_s3(s3_key: str) -> pd.DataFrame:
    s3  = get_s3_client()
    obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
    return pd.read_parquet(io.BytesIO(obj['Body'].read()))


# ── PSI calculation ────────────────────────────────────────────────────────────
def compute_psi(expected: np.ndarray,
                actual: np.ndarray,
                n_bins: int = 10) -> float:
    """
    Compute Population Stability Index between two distributions.

    Steps:
        1. Bucket the expected (training) distribution into n_bins equal bins
        2. Count what % of training data falls in each bin
        3. Count what % of today's data falls in each bin
        4. PSI = sum((actual% - expected%) * ln(actual% / expected%))

    Args:
        expected : training set values (baseline distribution)
        actual   : today's values (current distribution)
        n_bins   : number of bins (10 is standard)

    Returns:
        PSI score (float)
    """
    # Remove NaN values
    expected = expected[~np.isnan(expected)]
    actual   = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Create bins from the training set distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)  # remove duplicates

    if len(breakpoints) < 2:
        return 0.0

    # Count observations in each bin
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts   = np.histogram(actual,   bins=breakpoints)[0]

    # Convert to percentages — add small epsilon to avoid log(0)
    epsilon          = 1e-6
    expected_percents = (expected_counts / len(expected)) + epsilon
    actual_percents   = (actual_counts   / len(actual))   + epsilon

    # PSI formula
    psi_values = (actual_percents - expected_percents) * np.log(
        actual_percents / expected_percents
    )
    return float(np.sum(psi_values))


# ── Main task function ─────────────────────────────────────────────────────────
def drift_check(**context):
    """
    Airflow task entry point.

    Steps:
        1. Pull features path and brier_score from XCom
        2. Load today's features from S3
        3. Load training set baseline from local CSV
        4. Compute PSI for each monitored feature
        5. Log PSI scores to MLflow
        6. Trigger retrain_dag if any PSI > threshold
    """

    # ── 1. Get date, features path, and brier score ────────────────────────────
    test_date = Variable.get('PIPELINE_TEST_DATE', default_var=None)
    if test_date:
        date_str = test_date
    else:
        logical_date = context.get('logical_date') or context.get('execution_date')
        date_str = logical_date.strftime('%Y-%m-%d')
    features_key = context['ti'].xcom_pull(
        task_ids='build_features', key='features_path'
    ) or f'features/{date_str}/features.parquet'

    brier_score  = context['ti'].xcom_pull(
        task_ids='evaluate', key='brier_score'
    )

    print(f"[drift_check] Processing date: {date_str}")
    print(f"[drift_check] Brier score from evaluate: {brier_score}")

    # ── 2. Load today's features from S3 ──────────────────────────────────────
    today_df = read_parquet_from_s3(features_key)
    print(f"[drift_check] Today's features: {len(today_df)} rows")

    # ── 3. Load training set baseline ─────────────────────────────────────────
    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"[drift_check] ⚠️ Training data not found: {TRAIN_DATA_PATH}")
        print(f"[drift_check] Skipping PSI check")
        return

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    print(f"[drift_check] Training baseline: {len(train_df):,} rows")

    # ── 4. Compute PSI for each monitored feature ──────────────────────────────
    psi_scores   = {}
    drift_status = {}

    print(f"\n[drift_check] PSI scores:")
    print(f"  {'Feature':<30} {'PSI':>8}  Status")
    print(f"  {'-'*50}")

    for feature in MONITOR_FEATURES:
        if feature not in today_df.columns or feature not in train_df.columns:
            print(f"  {'⚠️'} {feature:<28} MISSING")
            continue

        expected = train_df[feature].values
        actual   = today_df[feature].values
        psi      = compute_psi(expected, actual)

        psi_scores[feature] = psi

        if psi > PSI_RETRAIN_THRESHOLD:
            status = "❌ RETRAIN"
            drift_status[feature] = 'retrain'
        elif psi > PSI_WARNING_THRESHOLD:
            status = "⚠️  WARNING"
            drift_status[feature] = 'warning'
        else:
            status = "✅ stable"
            drift_status[feature] = 'stable'

        print(f"  {status}  {feature:<28} {psi:>8.4f}")

    # Overall drift decision
    max_psi          = max(psi_scores.values()) if psi_scores else 0.0
    should_retrain   = max_psi > PSI_RETRAIN_THRESHOLD
    n_retrain_feats  = sum(1 for s in drift_status.values() if s == 'retrain')
    n_warning_feats  = sum(1 for s in drift_status.values() if s == 'warning')

    print(f"\n[drift_check] Summary:")
    print(f"  Max PSI:          {max_psi:.4f}")
    print(f"  Retrain features: {n_retrain_feats}")
    print(f"  Warning features: {n_warning_feats}")
    print(f"  Should retrain:   {should_retrain}")

    # ── 5. Log PSI scores to MLflow ────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"drift_check_{date_str}"):
        mlflow.log_param("date", date_str)
        mlflow.log_param("should_retrain", should_retrain)
        mlflow.log_metric("max_psi", max_psi)
        mlflow.log_metric("n_retrain_features", n_retrain_feats)

        for feature, psi in psi_scores.items():
            # MLflow metric names can't have special characters
            safe_name = feature.replace('_', '_').lower()
            mlflow.log_metric(f"psi_{safe_name}", psi)

        if brier_score is not None:
            mlflow.log_metric("brier_score", brier_score)

    print(f"[drift_check] ✅ PSI scores logged to MLflow")

    # ── 6. Trigger retrain_dag if drift detected ───────────────────────────────
    if should_retrain:
        print(f"[drift_check] 🔄 Drift detected — triggering retrain_dag")
        # In a full production setup we would use TriggerDagRunOperator here
        # For now we log the trigger decision and note it for manual action
        # TriggerDagRunOperator is defined in the DAG file, not in the task
        print(f"[drift_check] ⚠️  retrain_dag should be triggered")
        print(f"[drift_check] Max PSI {max_psi:.4f} exceeds threshold {PSI_RETRAIN_THRESHOLD}")
    else:
        print(f"[drift_check] ✅ No significant drift — retrain not needed")

    # Push drift result to XCom
    context['ti'].xcom_push(key='should_retrain', value=should_retrain)
    context['ti'].xcom_push(key='max_psi', value=float(max_psi))
    print(f"[drift_check] XCom pushed should_retrain: {should_retrain}, max_psi: {max_psi:.4f}")