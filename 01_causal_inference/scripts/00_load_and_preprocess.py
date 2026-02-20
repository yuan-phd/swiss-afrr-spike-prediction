"""
================================================================================
STEP 00 — Data Loading & Preprocessing
================================================================================
PURPOSE:
    Load raw Swissgrid Excel files (2023–2025) and ENTSO-E CSV (2023–2025),
    clean and standardise them into two unified CSVs ready for merging.

INPUTS:
    data/raw/EnergieUebersichtCH-2023.xlsx
    data/raw/EnergieUebersichtCH-2024.xlsx
    data/raw/EnergieUebersichtCH-2025.xlsx
    data/raw/entsoe_swiss_energy_data.csv        ← your full 2023–2025 ENTSO-E file

OUTPUTS:
    data/processed/swissgrid_clean_2023_2025.csv — 15-min Swissgrid panel
    data/processed/entsoe_clean_2023_2025.csv    — 15-min ENTSO-E panel

COLUMN REFERENCE (Swissgrid Zeitreihen0h15 sheet, zero-indexed):
    Col  0 : Zeitstempel                    → timestamp
    Col  6 : Positive Sekundär-Regelenergie → pos_sec_vol_kwh   (kWh)
    Col  7 : Negative Sekundär-Regelenergie → neg_sec_vol_kwh   (kWh)
    Col 12 : Verbundaustausch CH->DE        → CH_DE_kwh         (kWh)
    Col 13 : Verbundaustausch DE->CH        → DE_CH_kwh         (kWh)
    Col 14 : Verbundaustausch CH->FR        → CH_FR_kwh         (kWh)
    Col 15 : Verbundaustausch FR->CH        → FR_CH_kwh         (kWh)
    Col 21 : Avg Positive aFRR Price        → pos_sec_price     (EUR/MWh)
    Col 22 : Avg Negative aFRR Price        → neg_sec_price     (EUR/MWh)

UNIT CONVERSION:
    Source unit: kWh per 15-minute interval
    Target unit: MW (average power over interval)
    Formula: MW = (kWh / 1000) * 4    [because 15 min = 1/4 hour]

ENTSO-E COLUMN REFERENCE:
    Unnamed: 0         → timestamp  (timezone-aware, converted to CET naive)
    DE_WindSolar_Error → combined German wind + solar forecast error (MW)
    Sched_DE_CH        → day-ahead scheduled commercial exchange DE→CH (MW)
    Sched_CH_IT        → day-ahead scheduled commercial exchange CH→IT (MW)
    CH_Pump_Gen        → Swiss hydro pump storage net generation (MW)
================================================================================
"""

import os
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────
RAW_DIR       = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

SWISSGRID_FILES = {
    2023: "EnergieUebersichtCH-2023.xlsx",
    2024: "EnergieUebersichtCH-2024.xlsx",
    2025: "EnergieUebersichtCH-2025.xlsx",
}
ENTSOE_FILE = "entsoe_swiss_energy_data.csv"

# Output filenames — descriptive and date-ranged
SG_OUT     = "swissgrid_clean_2023_2025.csv"
ENTSOE_OUT = "entsoe_clean_2023_2025.csv"

# Column indices to extract from Zeitreihen0h15 sheet (0-indexed)
SG_COL_INDICES = [0, 6, 7, 12, 13, 14, 15, 21, 22]
SG_COL_NAMES   = [
    "timestamp",
    "pos_sec_vol_kwh",   # Positive secondary control energy (kWh)
    "neg_sec_vol_kwh",   # Negative secondary control energy (kWh)
    "CH_DE_kwh",         # Physical flow: Switzerland → Germany (kWh)
    "DE_CH_kwh",         # Physical flow: Germany → Switzerland (kWh)
    "CH_FR_kwh",         # Physical flow: Switzerland → France (kWh)
    "FR_CH_kwh",         # Physical flow: France → Switzerland (kWh)
    "pos_sec_price",     # Avg positive aFRR price (EUR/MWh)
    "neg_sec_price",     # Avg negative aFRR price (EUR/MWh)
]

KWH_TO_MW_FACTOR = 4 / 1000  # 15-min interval: MW = kWh * 4 / 1000


# ── Helper Functions ──────────────────────────────────────────────────────────

def load_swissgrid_year(filepath: str) -> pd.DataFrame:
    """
    Load one year of Swissgrid 15-minute data from the Zeitreihen0h15 sheet.

    The sheet layout:
        Row 0  → long German/English column description (skip)
        Row 1  → unit string e.g. 'kWh', 'Euro/MWh' (skip)
        Row 2+ → actual data rows

    We read with header=None and skiprows=2 to get raw data, then select
    only the columns we need by index position.
    """
    df = pd.read_excel(
        filepath,
        sheet_name="Zeitreihen0h15",
        header=None,
        skiprows=2,
    )
    df = df.iloc[:, SG_COL_INDICES].copy()
    df.columns = SG_COL_NAMES
    return df


def convert_kwh_flows_to_mw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert physical flow and volume columns from kWh to MW.

    Swissgrid stores 15-minute energy totals in kWh.
    Average power (MW) = Energy (kWh) / 1000 / (15/60 hours)
                       = Energy (kWh) * 4 / 1000
    """
    kwh_cols = ["pos_sec_vol_kwh", "neg_sec_vol_kwh",
                "CH_DE_kwh", "DE_CH_kwh", "CH_FR_kwh", "FR_CH_kwh"]
    for col in kwh_cols:
        mw_col = col.replace("_kwh", "_mw")
        df[mw_col] = df[col] * KWH_TO_MW_FACTOR
    return df


def parse_swissgrid_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the Swissgrid timestamp column.

    Format in source: 'DD.MM.YYYY HH:MM'  e.g. '01.01.2023 00:15'
    The dayfirst=True flag handles the DD.MM.YYYY European date format.
    """
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True)
    return df


def load_entsoe(filepath: str) -> pd.DataFrame:
    """
    Load ENTSO-E data and standardise the timestamp column.

    The ENTSO-E timestamp is timezone-aware (Europe/Zurich / CET+1).
    We convert to timezone-naive CET local time so it aligns with
    the Swissgrid timestamps (which are also CET naive).

    Steps:
        1. Parse as UTC-aware string
        2. Convert timezone to Europe/Zurich (handles DST correctly)
        3. Strip timezone info (tz_localize(None)) → naive CET
    """
    df = pd.read_csv(filepath)
    df.columns = ["timestamp", "DE_WindSolar_Error", "Sched_DE_CH",
                  "Sched_CH_IT", "CH_Pump_Gen"]
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], utc=True)
          .dt.tz_convert("Europe/Zurich")
          .dt.tz_localize(None)
    )
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def main():
    # ── 1. Load and concatenate all Swissgrid years ───────────────────────────
    print("Loading Swissgrid data...")
    sg_frames = []
    for year, fname in SWISSGRID_FILES.items():
        path = os.path.join(RAW_DIR, fname)
        print(f"  Reading {fname}...")
        frame = load_swissgrid_year(path)
        sg_frames.append(frame)
        print(f"    → {len(frame):,} rows loaded")

    sg = pd.concat(sg_frames, ignore_index=True)

    # ── 2. Parse timestamps and sort ──────────────────────────────────────────
    sg = parse_swissgrid_timestamps(sg)
    sg = sg.sort_values("timestamp").reset_index(drop=True)

    # ── 3. Convert kWh → MW ───────────────────────────────────────────────────
    sg = convert_kwh_flows_to_mw(sg)

    # Drop original kWh columns (keep only MW versions)
    sg = sg.drop(columns=["pos_sec_vol_kwh", "neg_sec_vol_kwh",
                           "CH_DE_kwh", "DE_CH_kwh", "CH_FR_kwh", "FR_CH_kwh"])

    print(f"\nSwissgrid combined: {len(sg):,} rows")
    print(f"  Date range: {sg['timestamp'].min()}  →  {sg['timestamp'].max()}")
    print(f"  Non-zero positive price rows: {(sg['pos_sec_price'].fillna(0) != 0).sum():,}")
    print(f"  Non-zero negative price rows: {(sg['neg_sec_price'].fillna(0) != 0).sum():,}")

    # ── 4. Load ENTSO-E data ──────────────────────────────────────────────────
    print("\nLoading ENTSO-E data...")
    entsoe = load_entsoe(os.path.join(RAW_DIR, ENTSOE_FILE))

    print(f"ENTSO-E: {len(entsoe):,} rows")
    print(f"  Date range: {entsoe['timestamp'].min()}  →  {entsoe['timestamp'].max()}")
    print(f"  DE_WindSolar_Error range: {entsoe['DE_WindSolar_Error'].min():.0f}"
          f" → {entsoe['DE_WindSolar_Error'].max():.0f} MW")

    # ── 5. Verify timestamp frequency ─────────────────────────────────────────
    sg_freq     = sg["timestamp"].diff().mode()[0]
    entsoe_freq = entsoe["timestamp"].diff().mode()[0]
    print(f"\nTimestamp frequency check:")
    print(f"  Swissgrid:  {sg_freq}  (expected 00:15:00)")
    print(f"  ENTSO-E:    {entsoe_freq}  (expected 00:15:00)")

    # ── 6. Save processed files ───────────────────────────────────────────────
    sg_out     = os.path.join(PROCESSED_DIR, SG_OUT)
    entsoe_out = os.path.join(PROCESSED_DIR, ENTSOE_OUT)

    sg.to_csv(sg_out, index=False)
    entsoe.to_csv(entsoe_out, index=False)

    print(f"\nSaved:")
    print(f"  {sg_out}")
    print(f"  {entsoe_out}")
    print("\n[STEP 00 COMPLETE]")


if __name__ == "__main__":
    main()
