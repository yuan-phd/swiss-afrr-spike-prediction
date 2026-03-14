"""
================================================================================
dags/retrain_dag.py — Retrain DAG (Champion/Challenger)
================================================================================
PURPOSE:
    Triggered by drift_check when PSI > 0.2.
    Runs the full retrain pipeline with champion/challenger pattern.

TASK ORDER:
    train_challenger → compare_models → promote_or_reject

HOW IT'S TRIGGERED:
    drift_check (Task 6 of energy_pipeline_daily) detects PSI > 0.2
    and triggers this DAG automatically via TriggerDagRunOperator.
    Can also be triggered manually for testing.

CHAMPION/CHALLENGER PATTERN:
    - train_challenger: runs train_model.py, produces challenger model
    - compare_models:   reads Brier scores from MLflow, decides winner
    - promote_or_reject: if challenger wins → overwrite champion files
                         if challenger loses → keep existing champion

SCHEDULE:
    None — only runs when triggered by drift_check or manually
================================================================================
"""

import os
import json
import shutil
import joblib
import mlflow
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR  = '/Users/yuan/Work/profolio/swiss-afrr-spike-prediction'
TRAINING_DIR = os.path.join(PROJECT_DIR, 'pipeline', 'training')

CHALLENGER_MODEL_PATH      = os.path.join(TRAINING_DIR, 'challenger_model.json')
CHALLENGER_CALIBRATOR_PATH = os.path.join(TRAINING_DIR, 'challenger_calibrator.pkl')
CHALLENGER_META_PATH       = os.path.join(TRAINING_DIR, 'challenger_meta.json')

CHAMPION_MODEL_PATH      = os.path.join(PROJECT_DIR, '02_model_training',
                                         'models', 'xgboost_spike_classifier.json')
CHAMPION_CALIBRATOR_PATH = os.path.join(PROJECT_DIR, '02_model_training',
                                         'models', 'isotonic_calibrator.pkl')

MLFLOW_TRACKING_URI = 'http://localhost:5001'

# ── Default arguments ──────────────────────────────────────────────────────────
default_args = {
    'owner': 'yuan',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': False,
}

# ── Task functions ─────────────────────────────────────────────────────────────

def train_challenger(**context):
    """
    Task 1: Run train_model.py to produce a challenger model.

    Imports and calls main() from train_model.py directly.
    The challenger model is saved to pipeline/training/.
    """
    import sys
    sys.path.insert(0, TRAINING_DIR)
    from train_model import main as retrain_main

    print("[retrain_dag] Starting challenger training...")
    promoted, brier = retrain_main()

    # Push challenger Brier to XCom for next task
    context['ti'].xcom_push(key='challenger_brier', value=float(brier))
    context['ti'].xcom_push(key='promoted', value=promoted)
    print(f"[retrain_dag] Challenger training complete. Brier: {brier:.4f}")


def compare_models(**context):
    challenger_brier = context['ti'].xcom_pull(
        task_ids='train_challenger', key='challenger_brier'
    )

    # Read champion Brier from MLflow — no hardcoding
    client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    runs   = client.search_runs(
        experiment_ids=['1'],
        filter_string="params.promoted = 'True'",
        order_by=["start_time DESC"],
        max_results=1
    )

    if not runs:
        raise ValueError(
            "[retrain_dag] No promoted run found in MLflow. "
            "Run train_model.py manually first to establish a champion baseline."
        )

    champion_brier = runs[0].data.metrics['challenger_brier']
    print(f"[retrain_dag] Champion Brier from MLflow: {champion_brier:.4f}")

    should_promote = challenger_brier < champion_brier

    print(f"[retrain_dag] Comparison:")
    print(f"  Champion Brier:   {champion_brier:.4f}")
    print(f"  Challenger Brier: {challenger_brier:.4f}")
    print(f"  Decision:         {'PROMOTE' if should_promote else 'KEEP CHAMPION'}")

    context['ti'].xcom_push(key='should_promote', value=should_promote)
    context['ti'].xcom_push(key='champion_brier', value=float(champion_brier))


def promote_or_reject(**context):
    """
    Task 3: Promote challenger to champion or reject it.

    If promote:
        - Copy challenger files → champion files
        - Log promotion to MLflow
    If reject:
        - Keep existing champion files unchanged
        - Log rejection to MLflow
    """
    should_promote   = context['ti'].xcom_pull(
        task_ids='compare_models', key='should_promote'
    )
    challenger_brier = context['ti'].xcom_pull(
        task_ids='train_challenger', key='challenger_brier'
    )
    champion_brier   = context['ti'].xcom_pull(
        task_ids='compare_models', key='champion_brier'
    )

    if should_promote:
        print(f"[retrain_dag] ✅ Promoting challenger to champion...")

        # Copy challenger → champion (overwrites existing champion)
        shutil.copy2(CHALLENGER_MODEL_PATH,      CHAMPION_MODEL_PATH)
        shutil.copy2(CHALLENGER_CALIBRATOR_PATH, CHAMPION_CALIBRATOR_PATH)

        print(f"[retrain_dag] Champion model updated:")
        print(f"  {CHAMPION_MODEL_PATH}")
        print(f"  {CHAMPION_CALIBRATOR_PATH}")
        print(f"  Brier improved: {champion_brier:.4f} → {challenger_brier:.4f}")

    else:
        print(f"[retrain_dag] ❌ Challenger rejected — keeping champion")
        print(f"  Champion Brier: {champion_brier:.4f}")
        print(f"  Challenger Brier: {challenger_brier:.4f}")
        print(f"  Challenger did not improve — no files changed")

    # Log final decision to MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment('afrr_spike_prediction')

    with mlflow.start_run(run_name=f"promotion_decision_{datetime.now().strftime('%Y-%m-%d')}"):
        mlflow.log_param("decision",         "promoted" if should_promote else "rejected")
        mlflow.log_metric("champion_brier",   champion_brier)
        mlflow.log_metric("challenger_brier", challenger_brier)
        mlflow.log_metric("brier_improvement", champion_brier - challenger_brier)
        if should_promote:
            mlflow.set_tag("promoted", "True")

    print(f"[retrain_dag] Decision logged to MLflow")


# ── DAG definition ─────────────────────────────────────────────────────────────
with DAG(
    dag_id='retrain_dag',
    description='Champion/challenger retrain pipeline — triggered by drift_check',
    default_args=default_args,
    schedule_interval=None,     # no schedule — only triggered externally
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['energy', 'afrr', 'retrain'],
) as dag:

    t1_train = PythonOperator(
        task_id='train_challenger',
        python_callable=train_challenger,
        provide_context=True,
    )

    t2_compare = PythonOperator(
        task_id='compare_models',
        python_callable=compare_models,
        provide_context=True,
    )

    t3_promote = PythonOperator(
        task_id='promote_or_reject',
        python_callable=promote_or_reject,
        provide_context=True,
    )

    # Linear dependency — must train before comparing, must compare before promoting
    t1_train >> t2_compare >> t3_promote