"""
================================================================================
fetch_energy_data.py — ENTSO-E Data Fetcher  (v2)
================================================================================
PURPOSE:
    Fetch all ENTSO-E data needed for Swiss aFRR price spike prediction.
    Runs locally, saves one CSV, which is then uploaded to Databricks Volume.

WHAT'S NEW IN V2:
    + DA_Price_DE             Day-Ahead price Germany
    + DA_Price_CH             Day-Ahead price Switzerland
    + DA_Price_Spread_DE_CH   DE minus CH price spread (derived)
    + Sched_FR_CH             Scheduled FR→CH exchange
    + Sched_IT_CH             Scheduled IT→CH exchange (reverse direction)
    + CH_Load_Forecast        Swiss forecasted load

WHY THESE ADDITIONS:
    DA prices explain the +51 EUR/MWh uniform price shift we observed in 2025
    (PSI of pos_sec_price = 0.85 — the single biggest drift factor).
    FR-CH and IT-CH scheduled flows complete the Swiss border picture —
    Switzerland is squeezed from all four sides simultaneously during
    worst loop flow events, not just from Germany.

PIPELINE POSITION:
    1. Run this script locally          → entsoe_swiss_energy_data_v2.csv
    2. Upload to Databricks Volume      → replace old file
    3. Re-run Databricks notebooks 00-02
    4. Export updated feature tables
    5. Download for local ML training

OUTPUT:
    entsoe_swiss_energy_data_v2.csv

COLUMNS:
    timestamp             CET local time (Europe/Zurich, timezone-naive)
    DE_WindSolar_Error    German wind+solar actual minus forecast (MW)
    Sched_DE_CH           Scheduled DE→CH day-ahead exchange (MW)
    Sched_CH_IT           Scheduled CH→IT day-ahead exchange (MW)
    Sched_FR_CH           Scheduled FR→CH day-ahead exchange (MW)     [NEW]
    Sched_IT_CH           Scheduled IT→CH day-ahead exchange (MW)     [NEW]
    CH_Pump_Gen           Swiss hydro pumped storage generation (MW)
    DA_Price_DE           Day-Ahead price Germany (EUR/MWh)           [NEW]
    DA_Price_CH           Day-Ahead price Switzerland (EUR/MWh)       [NEW]
    DA_Price_Spread_DE_CH DE minus CH day-ahead spread (EUR/MWh)      [NEW]
    CH_Load_Forecast      Swiss forecasted load (MW)                  [NEW]
================================================================================
"""

import pandas as pd
from entsoe import EntsoePandasClient
import time
import os
from dotenv import load_dotenv

# ── Configuration ─────────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv('ENTSOE_TOKEN')
client  = EntsoePandasClient(api_key=API_KEY)

TOTAL_START = pd.Timestamp('2023-01-01', tz='UTC')
TOTAL_END   = pd.Timestamp('2025-12-31 23:45', tz='UTC')
OUTPUT_FILE = 'entsoe_swiss_energy_data_v2.csv'

# Country codes
CH, DE, IT, FR = 'CH', 'DE', 'IT', 'FR'

# DA prices are hourly — resample to 15-min by forward filling
DA_FREQ = '15min'


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_fetch(label: str, func, *args, **kwargs):
    """
    Rate-limited fetch with labelled exception handling.
    Returns None on failure so the chunk continues without crashing.
    """
    try:
        data = func(*args, **kwargs)
        time.sleep(1.5)  # Respect ENTSO-E rate limit (429 prevention)
        return data
    except Exception as e:
        print(f"    [Skip] {label}: {e}")
        return None


def resample_hourly_to_15min(series: pd.Series,
                              target_index: pd.DatetimeIndex) -> pd.Series:
    """
    Day-Ahead prices are published hourly.
    Forward-fill to 15-min grid so they align with Swissgrid data.
    Each 15-min interval within an hour gets the same hourly price.
    """
    if series is None:
        return pd.Series(index=target_index, dtype=float)
    return series.reindex(target_index, method='ffill')


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def main():
    if not API_KEY:
        print("❌ ENTSOE_TOKEN not found. Check your .env file.")
        return

    months     = pd.date_range(start=TOTAL_START, end=TOTAL_END, freq='MS')
    all_chunks = []

    print(f"🚀 ENTSO-E Fetch v2  |  {TOTAL_START.date()} → {TOTAL_END.date()}")
    print(f"   Months to process: {len(months)}")
    print(f"   New fields: DA_Price_DE, DA_Price_CH, Sched_FR_CH, "
          f"Sched_IT_CH, CH_Load_Forecast\n")

    for i, start_date in enumerate(months):
        end_date = (start_date
                    + pd.offsets.MonthEnd(1)
                    + pd.Timedelta(hours=23, minutes=45))
        if end_date > TOTAL_END:
            end_date = TOTAL_END

        print(f"📅 [{i+1:02d}/{len(months)}] "
              f"{start_date.date()} → {end_date.date()}")

        # ── EXISTING FETCHES ──────────────────────────────────────────────────

        df_de_forecast = safe_fetch(
            "DE wind+solar forecast",
            client.query_wind_and_solar_forecast,
            DE, start=start_date, end=end_date)

        df_de_actual = safe_fetch(
            "DE generation actual",
            client.query_generation,
            DE, start=start_date, end=end_date)

        sched_de_ch = safe_fetch(
            "Sched DE→CH",
            client.query_scheduled_exchanges,
            DE, CH, start=start_date, end=end_date, day_ahead=True)

        sched_ch_it = safe_fetch(
            "Sched CH→IT",
            client.query_scheduled_exchanges,
            CH, IT, start=start_date, end=end_date, day_ahead=True)

        df_ch_gen = safe_fetch(
            "CH generation",
            client.query_generation,
            CH, start=start_date, end=end_date)

        # ── NEW FETCHES ───────────────────────────────────────────────────────

        # Day-Ahead prices — hourly, will be resampled to 15-min
        da_price_de = safe_fetch(
            "DA price DE",
            client.query_day_ahead_prices,
            DE, start=start_date, end=end_date)

        da_price_ch = safe_fetch(
            "DA price CH",
            client.query_day_ahead_prices,
            CH, start=start_date, end=end_date)

        # Scheduled FR→CH exchange
        # Needed to compute Unplanned_Flow_FR_CH = actual_net_FR_to_CH - Sched_FR_CH
        # Same logic as DE-CH unplanned flow
        sched_fr_ch = safe_fetch(
            "Sched FR→CH",
            client.query_scheduled_exchanges,
            FR, CH, start=start_date, end=end_date, day_ahead=True)

        # Scheduled IT→CH exchange (reverse direction of existing CH→IT)
        # Italy is a net importer from Switzerland — this captures southbound flow
        sched_it_ch = safe_fetch(
            "Sched IT→CH",
            client.query_scheduled_exchanges,
            IT, CH, start=start_date, end=end_date, day_ahead=True)

        # Swiss forecasted load
        # High Swiss load → less flexible domestic generation → expensive aFRR
        ch_load_forecast = safe_fetch(
            "CH load forecast",
            client.query_load_forecast,
            CH, start=start_date, end=end_date)

        # ── ASSEMBLY ──────────────────────────────────────────────────────────
        try:
            idx   = pd.date_range(start=start_date, end=end_date,
                                  freq='15min', tz='UTC')
            chunk = pd.DataFrame(index=idx)

            # DE wind/solar forecast error (existing)
            if df_de_actual is not None and df_de_forecast is not None:
                actual_cols    = ['Wind Onshore', 'Wind Offshore', 'Solar']
                available_cols = [c for c in actual_cols
                                  if c in df_de_actual.columns]
                actual_sum     = df_de_actual[available_cols].sum(axis=1)
                forecast_sum   = df_de_forecast.sum(axis=1)
                chunk['DE_WindSolar_Error'] = (
                    actual_sum.reindex(idx, method='ffill')
                    - forecast_sum.reindex(idx, method='ffill'))

            # Scheduled flows (existing)
            if sched_de_ch is not None:
                chunk['Sched_DE_CH'] = sched_de_ch.reindex(idx, method='ffill')
            if sched_ch_it is not None:
                chunk['Sched_CH_IT'] = sched_ch_it.reindex(idx, method='ffill')

            # CH hydro (existing)
            if df_ch_gen is not None:
                if 'Hydro Pumped Storage' in df_ch_gen.columns:
                    chunk['CH_Pump_Gen'] = (df_ch_gen['Hydro Pumped Storage']
                                            .reindex(idx, method='ffill'))

            # Day-Ahead prices — hourly → 15-min forward fill (NEW)
            if da_price_de is not None:
                chunk['DA_Price_DE'] = resample_hourly_to_15min(da_price_de, idx)
            if da_price_ch is not None:
                chunk['DA_Price_CH'] = resample_hourly_to_15min(da_price_ch, idx)

            # DA price spread DE-CH (NEW — derived)
            # Positive = DE more expensive → economic incentive to export to CH
            # Negative = CH more expensive → economic incentive to import from DE
            # This directly explains loop flow magnitude and direction
            if da_price_de is not None and da_price_ch is not None:
                chunk['DA_Price_Spread_DE_CH'] = (
                    chunk['DA_Price_DE'] - chunk['DA_Price_CH'])

            # Scheduled FR→CH (NEW)
            if sched_fr_ch is not None:
                chunk['Sched_FR_CH'] = sched_fr_ch.reindex(idx, method='ffill')

            # Scheduled IT→CH (NEW)
            if sched_it_ch is not None:
                chunk['Sched_IT_CH'] = sched_it_ch.reindex(idx, method='ffill')

            # Swiss load forecast (NEW)
            if ch_load_forecast is not None:
                if isinstance(ch_load_forecast, pd.DataFrame):
                    load_series = ch_load_forecast.iloc[:, 0]
                else:
                    load_series = ch_load_forecast
                chunk['CH_Load_Forecast'] = load_series.reindex(
                    idx, method='ffill')

            all_chunks.append(chunk)
            fetched = [c for c in chunk.columns if chunk[c].notna().any()]
            print(f"    ✓ {len(fetched)} columns: {', '.join(fetched)}")

        except Exception as e:
            print(f"    [Skip] Assembly error: {e}")

    # ── CONSOLIDATION ─────────────────────────────────────────────────────────
    if not all_chunks:
        print("❌ No data fetched. Check API key or network.")
        return

    print(f"\n📊 Consolidating {len(all_chunks)} monthly chunks...")
    final_df = pd.concat(all_chunks)

    # Convert UTC → CET local time (matches Swissgrid timestamps)
    final_df.index = final_df.index.tz_convert('Europe/Zurich')
    final_df.index = final_df.index.tz_localize(None)  # Strip tzinfo → naive CET
    final_df.index.name = 'timestamp'

    # Forward-fill remaining gaps (DST transitions, brief API gaps)
    final_df = final_df.ffill()

    # Remove duplicate timestamps (DST clock-back creates duplicates)
    final_df = final_df[~final_df.index.duplicated(keep='first')]
    final_df = final_df.sort_index()

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"FETCH COMPLETE")
    print(f"{'='*60}")
    print(f"  Rows:       {len(final_df):,}")
    print(f"  Date range: {final_df.index.min()} → {final_df.index.max()}")
    print(f"  Columns ({len(final_df.columns)}):")
    for col in final_df.columns:
        coverage = final_df[col].notna().mean() * 100
        status   = "✅" if coverage > 95 else "⚠️ " if coverage > 80 else "❌"
        print(f"    {status} {col:<30} {coverage:.1f}% coverage")

    # ── SAVE ──────────────────────────────────────────────────────────────────
    final_df.to_csv(OUTPUT_FILE)
    print(f"\n✅ Saved → {OUTPUT_FILE}")
    print(f"\nNext steps:")
    print(f"  1. Upload {OUTPUT_FILE} to Databricks Volume")
    print(f"     (replace or alongside old entsoe_swiss_energy_data.csv)")
    print(f"  2. Re-run Databricks notebook 00 (update ENTSOE_FILE variable)")
    print(f"  3. Re-run notebooks 01 and 02")
    print(f"  4. Export and download updated feature tables")


if __name__ == "__main__":
    main()
