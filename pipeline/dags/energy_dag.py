"""
================================================================================
energy_dag.py — Daily Energy Pipeline DAG
================================================================================
PURPOSE:
    Orchestrates the daily aFRR spike prediction pipeline.
    Each task is a Python function defined in the tasks/ folder.

SCHEDULE:
    Runs daily at 2am. Each run processes the previous day's data.
    Example: run at 2026-03-14 02:00 → processes data for 2026-03-13

TASK ORDER:
    fetch_data → validate_data → build_features → predict → evaluate → drift_check

HOW TO READ THIS FILE:
    1. The DAG object defines the schedule and default settings
    2. Each PythonOperator connects one Python function to one task
    3. The >> operator defines the order (fetch must finish before validate, etc.)

AIRFLOW CONCEPTS USED:
    DAG             : the pipeline definition (schedule, retries, etc.)
    PythonOperator  : runs a Python function as a task
    task_id         : the name you see in the Airflow UI
    >>              : dependency arrow (left must finish before right starts)
    catchup=False   : don't backfill historical runs on startup
    **context       : Airflow passes execution_date and other info via context
================================================================================
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# ── Import task functions ──────────────────────────────────────────────────────
# Each task lives in its own file under tasks/
# Right now they are all stubs that just print "hello"
# We will replace them one by one over the weekend
import sys
import os
sys.path.insert(0, '/Users/yuan/Work/profolio/swiss-afrr-spike-prediction/pipeline')

from tasks.fetch_data import fetch_data
from tasks.validate_data import validate_data
from tasks.build_features import build_features
from tasks.predict import predict
from tasks.evaluate import evaluate
from tasks.drift_check import drift_check

# ── Default arguments ──────────────────────────────────────────────────────────
# These apply to every task in the DAG unless overridden
default_args = {
    'owner': 'yuan',
    'retries': 2,                           # retry a failed task 2 times
    'retry_delay': timedelta(minutes=5),    # wait 5 min between retries
    'email_on_failure': False,              # no email alerts for now
}

# ── DAG definition ─────────────────────────────────────────────────────────────
with DAG(
    dag_id='energy_pipeline_daily',         # name shown in Airflow UI
    description='Daily aFRR spike prediction pipeline',
    default_args=default_args,
    schedule_interval='0 2 * * *',          # 2am every day (cron syntax)
    start_date=datetime(2026, 1, 1),        # earliest date Airflow will run for
    catchup=False,                          # don't backfill old dates on startup
    tags=['energy', 'afrr', 'ml'],          # tags for filtering in UI
) as dag:

    # ── Task 1: Fetch data from ENTSO-E API ────────────────────────────────────
    # Fetches yesterday's energy data and uploads to S3
    t1_fetch = PythonOperator(
        task_id='fetch_data',
        python_callable=fetch_data,         # the function to run
        provide_context=True,               # passes execution_date etc. to function
    )

    # ── Task 2: Validate raw data ──────────────────────────────────────────────
    # Checks schema, nulls, ranges — halts pipeline if data is bad
    t2_validate = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data,
        provide_context=True,
    )

    # ── Task 3: Build features ─────────────────────────────────────────────────
    # Merges ENTSO-E + Swissgrid, computes all 24 model features
    t3_features = PythonOperator(
        task_id='build_features',
        python_callable=build_features,
        provide_context=True,
    )

    # ── Task 4: Predict ────────────────────────────────────────────────────────
    # Loads champion model, generates spike probability predictions
    t4_predict = PythonOperator(
        task_id='predict',
        python_callable=predict,
        provide_context=True,
    )

    # ── Task 5: Evaluate ───────────────────────────────────────────────────────
    # Computes Brier score, logs metrics to MLflow
    t5_evaluate = PythonOperator(
        task_id='evaluate',
        python_callable=evaluate,
        provide_context=True,
    )

    # ── Task 6: Drift check ────────────────────────────────────────────────────
    # Computes PSI on key features, triggers retrain_dag if drift detected
    t6_drift = PythonOperator(
        task_id='drift_check',
        python_callable=drift_check,
        provide_context=True,
    )

    # ── Task dependencies ──────────────────────────────────────────────────────
    # This single line defines the entire execution order:
    # fetch → validate → build_features → predict → evaluate → drift_check
    # Each task only starts after the previous one succeeds
    t1_fetch >> t2_validate >> t3_features >> t4_predict >> t5_evaluate >> t6_drift