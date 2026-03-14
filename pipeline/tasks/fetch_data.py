"""
================================================================================
tasks/fetch_data.py — Task 1: Fetch ENTSO-E Data
================================================================================
PURPOSE:
    Fetch yesterday's ENTSO-E energy data and upload to S3.
    This is an Airflow task — it is called by energy_dag.py.

WHAT IT FETCHES (same columns as original 00_fetch_energy_data.py):
    DE_WindSolar_Error        German wind+solar forecast error (MW)
    Sched_DE_CH               Scheduled DE→CH exchange (MW)
    Sched_CH_IT               Scheduled CH→IT exchange (MW)
    Sched_FR_CH               Scheduled FR→CH exchange (MW)
    Sched_IT_CH               Scheduled IT→CH exchange (MW)
    CH_Pump_Gen               Swiss hydro pumped storage (MW)
    DA_Price_DE               Day-ahead price Germany (EUR/MWh)
    DA_Price_CH               Day-ahead price Switzerland (EUR/MWh)
    DA_Price_Spread_DE_CH     DE minus CH price spread (EUR/MWh)
    CH_Load_Forecast          Swiss forecasted load (MW)

KEY DIFFERENCES FROM ORIGINAL SCRIPT:
    1. Fetches ONE day only (yesterday) instead of 3 years
    2. Reads API key from Airflow Variable (not .env file)
    3. Uploads result to S3 (not local CSV)
    4. Pushes S3 path via XCom so Task 2 knows where to find the file
    5. Entry point is fetch_data(**context) not main()

AIRFLOW CONCEPTS:
    context['execution_date']  : the date this DAG run is processing
    context['ti']              : task instance — used for XCom push/pull
    ti.xcom_push()             : send a small value to the next task
    Variable.get()             : read a secret from Airflow Variables
================================================================================
"""

import io
import time
import pandas as pd
import boto3
from entsoe import EntsoePandasClient
from airflow.models import Variable

# ── Constants ──────────────────────────────────────────────────────────────────
# Country codes used by ENTSO-E API
CH, DE, IT, FR = 'CH', 'DE', 'IT', 'FR'

# S3 configuration — reads from Airflow Variables set during setup
S3_BUCKET        = Variable.get('S3_BUCKET', default_var='energy-pipeline')
S3_ENDPOINT_URL  = 'http://localhost:4566'  # LocalStack endpoint


# ── S3 client ──────────────────────────────────────────────────────────────────
def get_s3_client():
    """
    Returns a boto3 S3 client pointing at LocalStack.
    In production this would point at real AWS S3 — just remove endpoint_url.
    """
    return boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id='test',
        aws_secret_access_key='test',
        region_name='us-east-1',
    )


# ── Helpers (same logic as original script) ────────────────────────────────────
def safe_fetch(label, func, *args, **kwargs):
    """
    Rate-limited fetch with exception handling.
    Returns None on failure — pipeline continues and logs the skip.
    """
    try:
        data = func(*args, **kwargs)
        time.sleep(1.5)  # Respect ENTSO-E rate limit
        return data
    except Exception as e:
        print(f"    [Skip] {label}: {e}")
        return None


def resample_hourly_to_15min(series, target_index):
    """
    Day-ahead prices are hourly. Forward-fill to 15-min grid.
    Each 15-min interval within an hour gets the same price.
    """
    if series is None:
        return pd.Series(index=target_index, dtype=float)
    return series.reindex(target_index, method='ffill')


# ── Main task function ─────────────────────────────────────────────────────────
def fetch_data(**context):
    """
    Airflow task entry point.

    The **context argument is how Airflow passes information to your function.
    The most important things in context:
        context['execution_date'] : the date this run is processing
        context['ti']             : task instance, used for XCom

    IMPORTANT: execution_date is the DAG's logical date, NOT today.
    For a 2am daily run, execution_date = yesterday.
    This means if the pipeline fails and you rerun it for a specific date,
    it will always fetch the correct date's data — not whatever today is.
    """

    # ── 1. Get execution date ──────────────────────────────────────────────────
    # Uses PIPELINE_TEST_DATE variable if set (for demos)
    # Falls back to logical_date/execution_date in production
    test_date = Variable.get('PIPELINE_TEST_DATE', default_var=None)
    if test_date:
        date_str = test_date
        print(f"[fetch_data] Using test date: {date_str}")
    else:
        logical_date = context.get('logical_date') or context.get('execution_date')
        date_str = logical_date.strftime('%Y-%m-%d')
    print(f"[fetch_data] Processing date: {date_str}")

    # ── 2. Define fetch window ─────────────────────────────────────────────────
    # Fetch extra buffer on both sides to cover CET timezone offset
    start = pd.Timestamp(date_str, tz='UTC') - pd.Timedelta(hours=1)
    end   = pd.Timestamp(date_str, tz='UTC') + pd.Timedelta(hours=23, minutes=45)
    print(f"[fetch_data] Fetch window: {start} → {end}")

    # ── 3. Get API key from Airflow Variables ──────────────────────────────────
    # Variable.get() reads from the Airflow Variables we set in setup
    # Never hardcode API keys in task files
    api_key = Variable.get('ENTSOE_API_KEY')
    client  = EntsoePandasClient(api_key=api_key)

    # ── 4. Fetch all ENTSO-E data ──────────────────────────────────────────────
    # Same fetches as original script — just for one day instead of 3 years
    print("[fetch_data] Fetching from ENTSO-E API...")

    df_de_forecast  = safe_fetch("DE wind+solar forecast",
                                  client.query_wind_and_solar_forecast,
                                  DE, start=start, end=end)

    df_de_actual    = safe_fetch("DE generation actual",
                                  client.query_generation,
                                  DE, start=start, end=end)

    sched_de_ch     = safe_fetch("Sched DE→CH",
                                  client.query_scheduled_exchanges,
                                  DE, CH, start=start, end=end, day_ahead=True)

    sched_ch_it     = safe_fetch("Sched CH→IT",
                                  client.query_scheduled_exchanges,
                                  CH, IT, start=start, end=end, day_ahead=True)

    df_ch_gen       = safe_fetch("CH generation",
                                  client.query_generation,
                                  CH, start=start, end=end)

    da_price_de     = safe_fetch("DA price DE",
                                  client.query_day_ahead_prices,
                                  DE, start=start, end=end)

    da_price_ch     = safe_fetch("DA price CH",
                                  client.query_day_ahead_prices,
                                  CH, start=start, end=end)

    sched_fr_ch     = safe_fetch("Sched FR→CH",
                                  client.query_scheduled_exchanges,
                                  FR, CH, start=start, end=end, day_ahead=True)

    sched_it_ch     = safe_fetch("Sched IT→CH",
                                  client.query_scheduled_exchanges,
                                  IT, CH, start=start, end=end, day_ahead=True)

    ch_load_forecast = safe_fetch("CH load forecast",
                                   client.query_load_forecast,
                                   CH, start=start, end=end)

    # ── 5. Assemble into a single DataFrame ───────────────────────────────────
    # Build a 15-min index for the full day (96 rows)
    idx   = pd.date_range(start=start, end=end, freq='15min', tz='UTC')
    chunk = pd.DataFrame(index=idx)

    # DE wind/solar forecast error
    if df_de_actual is not None and df_de_forecast is not None:
        actual_cols    = ['Wind Onshore', 'Wind Offshore', 'Solar']
        available_cols = [c for c in actual_cols if c in df_de_actual.columns]
        actual_sum     = df_de_actual[available_cols].sum(axis=1)
        forecast_sum   = df_de_forecast.sum(axis=1)
        chunk['DE_WindSolar_Error'] = (
            actual_sum.reindex(idx, method='ffill')
            - forecast_sum.reindex(idx, method='ffill'))

    # Scheduled flows
    if sched_de_ch is not None:
        chunk['Sched_DE_CH'] = sched_de_ch.reindex(idx, method='ffill')
    if sched_ch_it is not None:
        chunk['Sched_CH_IT'] = sched_ch_it.reindex(idx, method='ffill')
    if sched_fr_ch is not None:
        chunk['Sched_FR_CH'] = sched_fr_ch.reindex(idx, method='ffill')
    if sched_it_ch is not None:
        chunk['Sched_IT_CH'] = sched_it_ch.reindex(idx, method='ffill')

    # CH hydro pumped storage
    if df_ch_gen is not None:
        if 'Hydro Pumped Storage' in df_ch_gen.columns:
            chunk['CH_Pump_Gen'] = (df_ch_gen['Hydro Pumped Storage']
                                    .reindex(idx, method='ffill'))

    # Day-ahead prices (hourly → 15-min forward fill)
    if da_price_de is not None:
        chunk['DA_Price_DE'] = resample_hourly_to_15min(da_price_de, idx)
    if da_price_ch is not None:
        chunk['DA_Price_CH'] = resample_hourly_to_15min(da_price_ch, idx)
    if da_price_de is not None and da_price_ch is not None:
        chunk['DA_Price_Spread_DE_CH'] = chunk['DA_Price_DE'] - chunk['DA_Price_CH']

    # Swiss load forecast
    if ch_load_forecast is not None:
        if isinstance(ch_load_forecast, pd.DataFrame):
            load_series = ch_load_forecast.iloc[:, 0]
        else:
            load_series = ch_load_forecast
        chunk['CH_Load_Forecast'] = load_series.reindex(idx, method='ffill')

    # ── 6. Convert UTC → CET (same as original script) ────────────────────────
    chunk.index = chunk.index.tz_convert('Europe/Zurich')
    chunk.index = chunk.index.tz_localize(None)  # strip tzinfo → naive CET
    chunk.index.name = 'timestamp'

    # Forward-fill any remaining gaps
    chunk = chunk.ffill()

    # ── 7. Log what we fetched ─────────────────────────────────────────────────
    print(f"[fetch_data] Fetched {len(chunk)} rows, {len(chunk.columns)} columns")
    for col in chunk.columns:
        coverage = chunk[col].notna().mean() * 100
        status   = "✅" if coverage > 95 else "⚠️" if coverage > 80 else "❌"
        print(f"  {status} {col:<30} {coverage:.1f}% coverage")

    # ── 8. Upload to S3 ────────────────────────────────────────────────────────
    # S3 path follows Medallion architecture: raw/YYYY-MM-DD/entsoe.csv
    s3_key = f"raw/{date_str}/entsoe.csv"

    # Convert DataFrame to CSV in memory (no local file needed)
    csv_buffer = io.StringIO()
    chunk.to_csv(csv_buffer)
    csv_bytes = csv_buffer.getvalue().encode('utf-8')

    # Upload to LocalStack S3
    s3 = get_s3_client()
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=s3_key,
        Body=csv_bytes,
        ContentType='text/csv',
    )
    print(f"[fetch_data] Uploaded to s3://{S3_BUCKET}/{s3_key}")

    # ── 9. Push S3 path to XCom ────────────────────────────────────────────────
    # XCom passes this path to Task 2 (validate_data)
    # Task 2 will call: context['ti'].xcom_pull(task_ids='fetch_data', key='raw_path')
    context['ti'].xcom_push(key='raw_path', value=s3_key)
    print(f"[fetch_data] XCom pushed raw_path: {s3_key}")