"""
================================================================================
tasks/validate_data.py — Task 2: Validate Raw ENTSO-E Data
================================================================================
PURPOSE:
    Read yesterday's ENTSO-E CSV from S3, validate it with Pandera.
    If validation fails, raise an exception — Airflow will retry the task.

WHAT IT CHECKS:
    - All expected columns are present
    - No null values in critical columns
    - Values are within physically plausible ranges
    - Correct number of rows (96 rows = 24 hours × 4 quarter-hours)

AIRFLOW CONCEPTS:
    xcom_pull   : read the S3 path that Task 1 (fetch_data) pushed
    raise       : raising any exception marks the task as FAILED
                  Airflow will then retry up to 2 times (from default_args)
================================================================================
"""

import io
import pandas as pd
import pandera as pa
import boto3
from airflow.models import Variable

# ── S3 configuration ───────────────────────────────────────────────────────────
S3_BUCKET       = Variable.get('S3_BUCKET', default_var='energy-pipeline')
S3_ENDPOINT_URL = 'http://localhost:4566'


def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id='test',
        aws_secret_access_key='test',
        region_name='us-east-1',
    )


def read_csv_from_s3(s3_key: str) -> pd.DataFrame:
    """
    Read a CSV file from S3 and return as a DataFrame.
    This is the opposite of what Task 1 did (put_object → get_object).
    """
    s3  = get_s3_client()
    obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
    csv_content = obj['Body'].read().decode('utf-8')
    df = pd.read_csv(io.StringIO(csv_content), parse_dates=['timestamp'])
    return df


# ── Pandera schema ─────────────────────────────────────────────────────────────
# Define once — validates everything automatically when schema.validate(df) runs
# pa.Check.* are built-in checks: greater_than, less_than, not_null, etc.
schema = pa.DataFrameSchema(
    columns={
        # Wind/solar forecast error — can be negative (over-forecast) or positive
        'DE_WindSolar_Error': pa.Column(
            float,
            nullable=False,
            checks=pa.Check.in_range(-50000, 50000),
            description='German wind+solar forecast error (MW)'
        ),

        # Scheduled DE→CH exchange — positive means Germany exports to CH
        'Sched_DE_CH': pa.Column(
            float,
            nullable=False,
            checks=pa.Check.in_range(-5000, 5000),
            description='Scheduled DE→CH exchange (MW)'
        ),

        # Scheduled CH→IT exchange
        'Sched_CH_IT': pa.Column(
            float,
            nullable=True,  # nullable — sometimes missing for certain days
            checks=pa.Check.in_range(-5000, 5000),
        ),

        # Scheduled FR→CH exchange
        'Sched_FR_CH': pa.Column(
            float,
            nullable=True,
            checks=pa.Check.in_range(-5000, 5000),
        ),

        # Scheduled IT→CH exchange
        'Sched_IT_CH': pa.Column(
            float,
            nullable=True,
            checks=pa.Check.in_range(-5000, 5000),
        ),

        # Swiss hydro pumped storage — always positive (generation)
        'CH_Pump_Gen': pa.Column(
            float,
            nullable=True,
            checks=pa.Check.greater_than_or_equal_to(0),
            description='Swiss hydro pumped storage (MW)'
        ),

        # DA price Switzerland — must be positive (EUR/MWh)
        # nullable=True because DA_Price_DE sometimes fails to fetch
        'DA_Price_CH': pa.Column(
            float,
            nullable=True,
            checks=pa.Check.in_range(-500, 5000),
            description='Day-ahead price Switzerland (EUR/MWh)'
        ),

        # Swiss load forecast — always positive, typically 4000-12000 MW
        'CH_Load_Forecast': pa.Column(
            float,
            nullable=True,
            checks=pa.Check.in_range(0, 20000),
            description='Swiss forecasted load (MW)'
        ),
    },
    # Allow extra columns — DA_Price_DE and DA_Price_Spread may or may not exist
    # depending on whether the fetch succeeded that day
    strict=False,
)


# ── Main task function ─────────────────────────────────────────────────────────
def validate_data(**context):
    """
    Airflow task entry point.

    Steps:
        1. Pull S3 path from XCom (left by fetch_data)
        2. Read CSV from S3
        3. Check row count (should be 96 for a full day)
        4. Run Pandera schema validation
        5. Push validated path to XCom for Task 3
    """

    # ── 1. Pull S3 path from XCom ──────────────────────────────────────────────
    # Task 1 pushed: context['ti'].xcom_push(key='raw_path', value='raw/2026-03-12/entsoe.csv')
    # Task 2 pulls:  xcom_pull(task_ids='fetch_data', key='raw_path')
    # This is how tasks communicate — Task 2 knows exactly where Task 1 left the data
    date_str = context['execution_date'].strftime('%Y-%m-%d')
    s3_key = context['ti'].xcom_pull(task_ids='fetch_data', key='raw_path') \
         or f'raw/{date_str}/entsoe.csv'
    print(f"[validate_data] Reading from S3: {s3_key}")

    # ── 2. Read CSV from S3 ────────────────────────────────────────────────────
    df = read_csv_from_s3(s3_key)
    print(f"[validate_data] Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"[validate_data] Columns: {df.columns.tolist()}")

    # ── 3. Check row count ─────────────────────────────────────────────────────
    # A full day has 96 rows (24 hours × 4 quarter-hours)
    # We allow 92+ to handle rare DST transition days (23-hour days)
    if len(df) < 92:
        raise ValueError(
            f"[validate_data] Expected 92-96 rows, got {len(df)}. "
            f"Data may be incomplete for {context['execution_date'].date()}"
        )
    print(f"[validate_data] ✅ Row count OK: {len(df)} rows")

    # ── 4. Run Pandera schema validation ───────────────────────────────────────
    # This single line checks all columns defined in the schema above
    # If ANY check fails, Pandera raises SchemaError with exact details:
    #   - which column failed
    #   - which rows failed
    #   - what the failing values were
    try:
        schema.validate(df, lazy=True)  # lazy=True collects ALL errors at once
        print(f"[validate_data] ✅ Schema validation passed")
    except pa.errors.SchemaErrors as e:
        # Log the full error details before raising
        print(f"[validate_data] ❌ Schema validation FAILED:")
        print(e.failure_cases)
        raise  # re-raise so Airflow marks task as FAILED and retries

    # ── 5. Log coverage summary ────────────────────────────────────────────────
    print(f"[validate_data] Column coverage:")
    for col in df.columns:
        if col == 'timestamp':
            continue
        coverage = df[col].notna().mean() * 100
        status   = "✅" if coverage > 95 else "⚠️" if coverage > 80 else "❌"
        print(f"  {status} {col:<30} {coverage:.1f}%")

    # ── 6. Push validated path to XCom for Task 3 ─────────────────────────────
    # We pass the same S3 path forward — Task 3 will read the same file
    context['ti'].xcom_push(key='validated_path', value=s3_key)
    print(f"[validate_data] XCom pushed validated_path: {s3_key}")