import pandas as pd
from entsoe import EntsoePandasClient
from datetime import datetime
import time
import os
from dotenv import load_dotenv

# --- 1. Configuration ---
# Load variables from .env file
load_dotenv()
API_KEY = os.getenv('ENTSOE_TOKEN')
client = EntsoePandasClient(api_key=API_KEY)

# Define time range - UPDATED to include full year 2025
TOTAL_START = pd.Timestamp('2023-01-01', tz='UTC')
TOTAL_END = pd.Timestamp('2025-12-31 23:45', tz='UTC')

# Country Codes
CH, DE, IT = 'CH', 'DE', 'IT'


def safe_fetch(func, *args, **kwargs):
    """Industrial protection: rate limiting and exception handling"""
    try:
        data = func(*args, **kwargs)
        time.sleep(1.5)  # Prevent 429 Too Many Requests
        return data
    except Exception as e:
        print(f"  [Error] Fetching failed: {e}")
        return None


def main():
    if not API_KEY:
        print("❌ Error: ENTSOE_TOKEN not found. Check your .env file.")
        return

    # Monthly chunks to avoid connection timeouts for large ranges
    months = pd.date_range(start=TOTAL_START, end=TOTAL_END, freq='MS')
    all_chunks = []

    print(f"🚀 Starting task: {TOTAL_START.date()} to {TOTAL_END.date()}")

    for start_date in months:
        end_date = start_date + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23, minutes=45)
        if end_date > TOTAL_END: end_date = TOTAL_END

        print(f"📅 Processing: {start_date.date()} -> {end_date.date()}...")

        # --- Data Fetching Logic ---
        # 1. DE Wind/Solar (Forecast vs Actual)
        df_de_forecast = safe_fetch(client.query_wind_and_solar_forecast, DE, start=start_date, end=end_date)
        df_de_actual = safe_fetch(client.query_generation, DE, start=start_date, end=end_date)

        # 2. Scheduled Exchanges (Day-Ahead)
        sched_de_ch = safe_fetch(client.query_scheduled_exchanges, DE, CH, start=start_date, end=end_date,
                                 day_ahead=True)
        sched_ch_it = safe_fetch(client.query_scheduled_exchanges, CH, IT, start=start_date, end=end_date,
                                 day_ahead=True)

        # 3. CH Hydro Pumped Storage
        df_ch_gen = safe_fetch(client.query_generation, CH, start=start_date, end=end_date)

        # --- Data Assembly ---
        try:
            chunk = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq='15min', tz='UTC'))

            if df_de_actual is not None and df_de_forecast is not None:
                # Target specific columns for calculation
                actual_cols = ['Wind Onshore', 'Wind Offshore', 'Solar']
                available_cols = [c for c in actual_cols if c in df_de_actual.columns]
                actual_sum = df_de_actual[available_cols].sum(axis=1)
                chunk['DE_WindSolar_Error'] = actual_sum - df_de_forecast.sum(axis=1)

            chunk['Sched_DE_CH'] = sched_de_ch
            chunk['Sched_CH_IT'] = sched_ch_it

            if df_ch_gen is not None and 'Hydro Pumped Storage' in df_ch_gen.columns:
                chunk['CH_Pump_Gen'] = df_ch_gen['Hydro Pumped Storage']

            all_chunks.append(chunk)
        except Exception as e:
            print(f"  [Skip] Processing error for this chunk: {e}")

    # --- Consolidation and Save ---
    if all_chunks:
        final_df = pd.concat(all_chunks)
        # Convert to local time and clean (Pandas 3.0 syntax)
        final_df.index = final_df.index.tz_convert('Europe/Zurich')
        final_df = final_df.ffill()

        output_file = 'entsoe_swiss_energy_data.csv'
        final_df.to_csv(output_file)
        print(f"✅ Success! Data saved to: {output_file}")
    else:
        print("❌ No data fetched. Check API Key or Network.")


if __name__ == "__main__":
    main()
