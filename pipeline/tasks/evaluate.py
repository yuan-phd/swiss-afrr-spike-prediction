"""
================================================================================
tasks/evaluate.py — Task 5: Evaluate Predictions and Log to MLflow
================================================================================
PURPOSE:
    Read predictions and actual prices from S3, compute Brier score,
    log all metrics to MLflow for monitoring over time.

INPUTS:
    S3: predictions/{date}/predictions.parquet  (from Task 4)

OUTPUT:
    MLflow run with metrics logged (visible at http://localhost:5001)
    XCom: brier_score (passed to Task 6 for drift check)

WHY BRIER SCORE:
    Brier score measures calibration quality of probability predictions.
    Formula: mean((predicted_prob - actual_outcome)²)
    Lower is better. Our model v2 baseline: 0.0623.
    If today's Brier score is much higher → model may be degrading.

MLFLOW CONCEPTS:
    mlflow.set_tracking_uri()  : tell MLflow where the server is
    mlflow.set_experiment()    : which experiment to log to
    mlflow.start_run()         : open a new run (like a new row in a table)
    mlflow.log_metric()        : log a number that changes per run
    mlflow.log_param()         : log a fixed value for this run
================================================================================
"""

import io
import pandas as pd
import numpy as np
import boto3
import mlflow
from sklearn.metrics import brier_score_loss, average_precision_score
from airflow.models import Variable

# ── Configuration ──────────────────────────────────────────────────────────────
S3_BUCKET            = Variable.get('S3_BUCKET', default_var='energy-pipeline')
S3_ENDPOINT_URL      = 'http://localhost:4566'
MLFLOW_TRACKING_URI  = Variable.get('MLFLOW_TRACKING_URI',
                                     default_var='http://localhost:5001')
EXPERIMENT_NAME      = 'afrr_spike_prediction'

# Baseline Brier score from model v2 validation
BASELINE_BRIER       = 0.0623
DEGRADATION_WARNING  = 0.05   # warn if Brier increases by more than 5%


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


# ── Main task function ─────────────────────────────────────────────────────────
def evaluate(**context):
    """
    Airflow task entry point.

    Steps:
        1. Pull predictions path from XCom (Task 4)
        2. Read predictions from S3
        3. Compute Brier score and other metrics
        4. Log everything to MLflow
        5. Push Brier score to XCom for Task 6 (drift_check)
    """

    # ── 1. Get date and predictions path ──────────────────────────────────────
    test_date = Variable.get('PIPELINE_TEST_DATE', default_var=None)
    if test_date:
        date_str = test_date
    else:
        logical_date = context.get('logical_date') or context.get('execution_date')
        date_str = logical_date.strftime('%Y-%m-%d')
    predictions_key  = context['ti'].xcom_pull(
        task_ids='predict', key='predictions_path'
    ) or f'predictions/{date_str}/predictions.parquet'

    print(f"[evaluate] Processing date: {date_str}")
    print(f"[evaluate] Reading predictions from S3: {predictions_key}")

    # ── 2. Read predictions from S3 ───────────────────────────────────────────
    df = read_parquet_from_s3(predictions_key)
    print(f"[evaluate] Loaded {len(df)} rows")

    # ── 3. Compute metrics ────────────────────────────────────────────────────
    # We need actual spike labels to compute Brier score
    # pos_sec_price is in the predictions file (we saved it in Task 4)
    # Spike = price above rolling P90 threshold
    # For evaluation we use a fixed threshold from training: ~100 EUR/MWh
    # (approximate — exact threshold would come from the training set P90)

    # Drop rows with null prices (can't evaluate without ground truth)
    df_eval = df.dropna(subset=['pos_sec_price', 'prob_calibrated'])

    if len(df_eval) == 0:
        print(f"[evaluate] ⚠️ No rows with valid prices — skipping evaluation")
        context['ti'].xcom_push(key='brier_score', value=None)
        return

    # Compute actual spike labels using training threshold from model_meta.json
    # rolling_p90 from training was approximately 100 EUR/MWh
    SPIKE_THRESHOLD = 245.22  # EUR/MWh — approximate P90 from training
    df_eval = df_eval.copy()
    df_eval['actual_spike'] = (df_eval['pos_sec_price'] > SPIKE_THRESHOLD).astype(int)

    y_true = df_eval['actual_spike'].values
    y_pred = df_eval['prob_calibrated'].values

    # Brier score: mean((predicted_prob - actual_outcome)²)
    brier = brier_score_loss(y_true, y_pred)

    # Spike rate stats
    actual_spike_rate    = y_true.mean()
    predicted_spike_rate = y_pred.mean()
    n_actual_spikes      = y_true.sum()
    n_predicted_spikes   = df['spike_predicted'].sum()

    # PR-AUC (only meaningful if there are actual spikes)
    pr_auc = None
    if n_actual_spikes > 0 and n_actual_spikes < len(y_true):
        pr_auc = average_precision_score(y_true, y_pred)

    # Degradation check
    brier_degradation = (brier - BASELINE_BRIER) / BASELINE_BRIER * 100
    if brier_degradation > DEGRADATION_WARNING * 100:
        print(f"[evaluate] ⚠️ Brier score degraded by {brier_degradation:.1f}% "
              f"vs baseline {BASELINE_BRIER}")
    else:
        print(f"[evaluate] ✅ Brier score within acceptable range")

    print(f"[evaluate] Metrics:")
    print(f"  Brier score:          {brier:.4f}  (baseline: {BASELINE_BRIER})")
    print(f"  Actual spike rate:    {actual_spike_rate:.1%}")
    print(f"  Predicted spike rate: {predicted_spike_rate:.1%}")
    print(f"  Actual spikes:        {n_actual_spikes}")
    print(f"  Predicted spikes:     {n_predicted_spikes}")
    if pr_auc:
        print(f"  PR-AUC:               {pr_auc:.4f}")

    # ── 4. Log to MLflow ──────────────────────────────────────────────────────
    # Set tracking URI — tells MLflow where the server is
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Set experiment — creates it if it doesn't exist
    # All daily runs go into the same experiment so you can compare over time
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"daily_eval_{date_str}"):

        # log_param: fixed values for this run (not metrics, just context)
        mlflow.log_param("date",           date_str)
        mlflow.log_param("n_rows",         len(df_eval))
        mlflow.log_param("spike_threshold", SPIKE_THRESHOLD)
        mlflow.log_param("baseline_brier", BASELINE_BRIER)

        # log_metric: numbers that change per run — these are what you track over time
        mlflow.log_metric("brier_score",          brier)
        mlflow.log_metric("actual_spike_rate",    actual_spike_rate)
        mlflow.log_metric("predicted_spike_rate", predicted_spike_rate)
        mlflow.log_metric("n_actual_spikes",      int(n_actual_spikes))
        mlflow.log_metric("n_predicted_spikes",   int(n_predicted_spikes))
        mlflow.log_metric("brier_degradation_pct", brier_degradation)

        if pr_auc:
            mlflow.log_metric("pr_auc", pr_auc)

    print(f"[evaluate] ✅ Logged to MLflow experiment: {EXPERIMENT_NAME}")
    print(f"[evaluate] View at: {MLFLOW_TRACKING_URI}")

    # ── 5. Push Brier score to XCom for Task 6 (drift_check) ─────────────────
    # drift_check uses this to decide whether to trigger retraining
    context['ti'].xcom_push(key='brier_score', value=float(brier))
    print(f"[evaluate] XCom pushed brier_score: {brier:.4f}")