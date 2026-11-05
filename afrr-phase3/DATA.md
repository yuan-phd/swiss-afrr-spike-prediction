# Phase-3 benchmark dataset — `merged_2023_2026`

Built 2026-11-05. This file is the single source of truth for the follow-up experiments.

| | |
|---|---|
| Files | `afrr-phase3/data/merged_2023_2026.parquet` (canonical), `afrr-phase3/data/merged_2023_2026.csv` (copy) |
| Snapshot | The parquet committed to git is a **snapshot built on 2026-11-05** from raw files downloaded and fetched that day. The raw files (Swissgrid xlsx, ENTSO-E csv) and the CSV copy are not committed; rebuild them with the commands in *Rebuild from scratch*. A rebuild on a later date can differ where the sources were revised since (see *Revisions*). |
| Period | 2023-01-01 00:00 → 2026-09-30 23:45 Swiss local time (2022-12-31 23:00 → 2026-09-30 21:45 UTC) |
| Resolution | 15 min; each row is the interval **starting** at `timestamp_utc` |
| Rows | 131,420. The 15-min UTC grid is complete: no gaps, no duplicates, strictly increasing. **No missing values in any column.** |
| Columns | 22 (2 timestamps, 10 Swissgrid, 10 ENTSO-E) |
| Processing | Raw values only. No filling, clipping or features. The only transformation is Swissgrid kWh per 15 min → MW (× 4 / 1000). |
| Environment | conda env **`afrr3`** (Python 3.11.18, pandas 2.3.3, pyarrow 26.0.0, openpyxl 3.1.5, entsoe-py 0.8.1). The spec is in `afrr-phase3/environment.yml`. |

## Columns

**Availability at forecast time** (for a forecast made before delivery day D):
- **D-1**: known day-ahead, published on D-1 after the day-ahead auction (about 12:45 CET, ENTSO-E publication in the early afternoon).
- **ex post**: measured or settled, only known after the interval.

These are statements about the quantity itself. The publication timestamp of each value is not recorded in this dataset.

| Column | Source | Unit | Meaning | Availability |
|---|---|---|---|---|
| `timestamp_utc` | derived | tz-aware UTC | Start of the 15-min interval. This is the merge key. | — |
| `timestamp_local` | derived | naive Europe/Zurich | Same instant in Swiss wall-clock time, for calendar features. It **repeats** 02:00–02:45 on autumn DST days; use `timestamp_utc` for ordering. | — |
| `pos_sec_price` | Swissgrid col 21 "Average positive secondary control energy prices" | EUR/MWh | Positive aFRR energy price | ex post |
| `neg_sec_price` | Swissgrid col 22 "Average negative secondary control energy prices" | EUR/MWh | Negative aFRR energy price | ex post |
| `pos_sec_vol_mw` | Swissgrid col 6 "Positive secundary control energy" | MW (from kWh/15 min) | Activated positive aFRR | ex post |
| `neg_sec_vol_mw` | Swissgrid col 7 "Negative secundary control energy" | MW | Activated negative aFRR (≤ 0) | ex post |
| `CH_DE_mw` | Swissgrid col 12 "Cross Border Exchange CH->DE" | MW | Physical flow CH→DE | ex post |
| `DE_CH_mw` | Swissgrid col 13 "Cross Border Exchange DE->CH" | MW | Physical flow DE→CH | ex post |
| `CH_FR_mw` | Swissgrid col 14 "Cross Border Exchange CH->FR" | MW | Physical flow CH→FR | ex post |
| `FR_CH_mw` | Swissgrid col 15 "Cross Border Exchange FR->CH" | MW | Physical flow FR→CH | ex post |
| `CH_IT_mw` | Swissgrid col 16 "Cross Border Exchange CH->IT" | MW | Physical flow CH→IT (**new**) | ex post |
| `IT_CH_mw` | Swissgrid col 17 "Cross Border Exchange IT->CH" | MW | Physical flow IT→CH (**new**) | ex post |
| `DE_WindSolar_Error` | ENTSO-E A75 actual generation (DE) minus A69 day-ahead wind+solar forecast (DE) | MW | (Wind Onshore + Wind Offshore + Solar) actual − forecast | ex post (the forecast part is D-1) |
| `Sched_DE_CH` | ENTSO-E scheduled commercial exchanges, day-ahead, DE→CH | MW | | D-1 |
| `Sched_CH_IT` | same, CH→IT | MW | | D-1 |
| `CH_Pump_Gen` | ENTSO-E A75 actual generation CH, "Hydro Pumped Storage" (Actual Aggregated) | MW | | ex post |
| `DA_Price_CH` | ENTSO-E day-ahead price, CH | EUR/MWh | | D-1 |
| `Sched_FR_CH` | ENTSO-E scheduled exchanges, day-ahead, FR→CH | MW | | D-1 |
| `Sched_IT_CH` | same, IT→CH | MW | | D-1 |
| `CH_Load_Forecast` | ENTSO-E day-ahead total load forecast, CH | MW | | D-1 |
| `DA_Price_DE` | ENTSO-E day-ahead price, bidding zone **DE_LU** | EUR/MWh | | D-1 |
| `DA_Price_Spread_DE_CH` | `DA_Price_DE − DA_Price_CH` | EUR/MWh | Computed in the fetch scripts; NaN if either side is NaN | D-1 |

Notes:
- **Native resolution.** Some ENTSO-E series are hourly: `DA_Price_CH`, `CH_Load_Forecast`, `CH_Pump_Gen`, and `DA_Price_DE` before 2025-10-01. Each hourly value is repeated on its four 15-min rows. `DA_Price_DE` is 15-min from 2025-10-01, when the day-ahead market moved to 15-min products. Scheduled exchanges switch between hourly and 15-min from one delivery day to the next.
- **Swissgrid revisions.** Swissgrid values are settlement data and can be revised. The 2025 file was republished on 2026-05-19 (see below).

## Revisions

- **ENTSO-E measured (ex-post) series are revised after the fact.** This applies to `DE_WindSolar_Error` (through DE actual generation) and `CH_Pump_Gen`. Occasionally scheduled exchanges are corrected too, for example `Sched_DE_CH` on 2025-08-04.
- **This dataset uses the values as published on the fetch date:**
  - Both ENTSO-E files, `entsoe_2023_2024.csv` and `entsoe_2025_2026.csv`, were fetched on **2026-11-05**.
  - These are the revised values available on that date, **not** what was known in real time. A forecasting experiment that needs as-of-time values would need a separate source.
- **Evidence:** compared with the v2 fetch of 2026-02-21, 2025 measured values changed in Mar, Oct and Dec 2025. 2023–24 values were unchanged (see *Comparison*).
- **Swissgrid** settlement data is revised the same way: 2025 was republished on 2026-05-19.

## Sources and raw files

| Period (local) | Swissgrid | ENTSO-E |
|---|---|---|
| 2023 | `data/raw/EnergieUebersichtCH-2023.xlsx` (existing repo copy) | `afrr-phase3/data/raw/entsoe_2023_2024.csv` (fetched 2026-11-05) |
| 2024 | `data/raw/EnergieUebersichtCH-2024.xlsx` (existing repo copy) | same 2023–24 file |
| 2025 | `afrr-phase3/data/raw/EnergieUebersichtCH-2025.xlsx` (Last-Modified 2026-05-19) | `afrr-phase3/data/raw/entsoe_2025_2026.csv` (fetched 2026-11-05) |
| 2026-01 → 09 | `afrr-phase3/data/raw/EnergieUebersichtCH-2026.xlsx` (Last-Modified 2026-10-20; ends 2026-09-30 23:45) | same new file |

- Swissgrid downloads (sheet `Zeitreihen0h15`) come from https://www.swissgrid.ch/en/home/customers/topics/energy-data-ch.html:
  - 2025: `https://www.swissgrid.ch/dam/jcr:bbbbadc6-3656-4dfe-b81e-bbb9ad01be79/EnergieUebersichtCH-2025.xlsx`
  - 2026: `https://www.swissgrid.ch/dam/jcr:805e525c-44fe-4701-a227-6144193257ac/EnergieUebersichtCH-2026.xlsx`
- ENTSO-E data is fetched through the Transparency Platform API with entsoe-py `EntsoePandasClient`. The API key is `ENTSOE_TOKEN` in the repo-root `.env`.

## Timezone and interval conventions

### Merge key and output columns
- The merge key is `timestamp_utc`. Swissgrid defines the rows, and ENTSO-E is left-joined, so a missing ENTSO-E instant appears as NaN instead of dropping the row.
- In the output, every timestamp is the **start** of its 15-min interval.

### Swissgrid (all files: naive Swiss local time, with DST)
- **Label convention changed between files.** Each workbook's sheet header says only `Zeitstempel`. The convention is stated by the workbook's own `Uebersicht` formulas, which sum a From–To period:
  - **2023 and 2024:** `SUM(... MATCH(dat_von+TIME(0,15,0), Datetime!$A:$A, 0) ... MATCH(dat_bis, ...))`. The first interval of a period carries the label From + 15 min, so **the label is the interval end**. The files run from `01.01.YYYY 00:15` to `01.01.YYYY+1 00:00`.
  - **2025 and 2026:** `SUM(... MATCH(dat_von, ...) ... MATCH(dat_bis-TIME(0,15,0), ...))`. **The label is the interval start.** The files run from `01.01.YYYY 00:00` to `31.12.YYYY 23:45` (2026: to `30.09.2026 23:45`).
- **Spring-forward day** (92 rows): the sequence jumps 01:45 → 03:00, so local 02:00–02:45 never appears. Localisation uses `nonexistent='raise'`; nothing raises, because no non-existent time is present.
- **Autumn repeated hour** (100 rows): 02:00–02:45 appears twice in file order, first CEST, then CET. It is localised **in file order** with `ambiguous='infer'`, so both copies are kept as distinct UTC instants.
- **End labels → start (2023–24):** after converting to UTC, 15 min is subtracted. The resulting UTC instants are checked to be contiguous 15-min steps.

### ENTSO-E, old v2 file (no longer an input; naive Swiss local time)
v2 is used only for comparison (`compare_entsoe.py`). For reference:
- `00_fetch_energy_data.py:262-263` wrote naive local time.
- Line 270 (`keep='first'`) kept only the CEST copy of the repeated autumn hour, so v2 is localised with `ambiguous=True` (CEST). The CET copy (4 rows per year) is missing.
- The file starts at 2023-01-01 00:00 **UTC** (= 01:00 local), so local 2023-01-01 00:00–00:45 is missing.
- Spring forward: there are no non-existent local times, because the file came from UTC.
- The old script forward-filled every gap (line 266) and reindexed hourly data with an unlimited `ffill`. Any API gap in 2023–24 is therefore hidden in v2 and can't be detected from the file.

### ENTSO-E, `entsoe_2023_2024.csv` and `entsoe_2025_2026.csv` (tz-aware UTC)
- Both are written by `fetch_entsoe.py` with `+00:00` offsets and parsed directly as UTC.
- All four years therefore come from the same fetch code, with `DE_LU` and no forward fill.
- Both copies of every autumn repeated hour are present.
- DST needs no special handling.
- No gap filling. Each source point covers its own native interval only: the step to the next point, or, if that step is over 60 min, the step from the previous point.
- Neither fetch returned any NaN: 70,176 rows for 2023–24, 61,244 rows for 2025–26.

## Missing values (NaN count; year = Swiss local calendar year)

**0 NaN in every column and every year.** `data/reports/gaps.csv` is empty.

| | total | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|
| each of the 20 data columns | 0 | 0 | 0 | 0 | 0 |
| rows | 131,420 | 35,040 | 35,136 | 35,040 | 26,204 |

The earlier build (with v2 for 2023–24) had 12 NaN per ENTSO-E column: 2023-01-01 00:00–00:45, and the CET autumn hour on 2023-10-29 and 2024-10-27. The 2023–24 re-fetch removed them. The per-column table is in `data/reports/missing_values.csv`.

## Known issues fixed relative to the old `data/raw_merged_*.csv`

1. **ENTSO-E timezone bug.** `merge_raw_data.py:126-130` (repo root) parsed v2's naive local timestamps as UTC and converted them to local time again. That shifted every ENTSO-E column by +1 h in winter and +2 h in summer relative to Swissgrid. Fixed: ENTSO-E is now fetched and parsed as UTC.
2. **Swissgrid 2023–24 one-step misalignment.** The old merge treated the 2023–24 end labels as start labels, so those Swissgrid rows were 15 min late relative to the 2025 rows. The 2024 file's last row (label 2025-01-01 00:00, interval 2024-12-31 23:45–24:00) also collided with the 2025 file's first row. The old `raw_merged_2025.csv` therefore has two rows at 2025-01-01 00:00 (prices 209.83 and 294.81). Fixed by the −15 min shift for 2023–24.
3. **Duplicate timestamps in old files.** The old 2025 file has every autumn 02:00–02:45 timestamp 4 times, and the old 2023–24 file has 24 duplicate timestamps. The new file has none.
4. **German day-ahead price code.** `00_fetch_energy_data.py:156` queries `'DE'`, which now returns `NoMatchingDataError` (entsoe-py 0.7.10 and 0.8.1). `safe_fetch` swallowed this error, so a new fetch would silently produce no `DA_Price_DE` and no spread. The new fetch uses `DE_LU`.

   The existing v2 file is **not** affected. Its `DA_Price_DE` is non-empty for every row in 2023–2025, and it is identical to the DE_LU re-fetch for all 35,036 shared 2025 rows (0 differing cells, including Oct–Dec at 15-min resolution).

## Comparison: re-fetched ENTSO-E vs old v2

Both comparisons are made with `compare_entsoe.py`. A cell differs if |Δ| > 1e-6 or exactly one side is NaN.

### 2023–2024

- There are 70,164 shared rows. The new file also has 12 rows that v2 lacks: local 2023-01-01 00:00–00:45, and the CET autumn hour on 2023-10-29 and 2024-10-27.
- **0 differing cells in all 10 columns** (max abs diff 0; correlation 1.0 for every column).
- `DA_Price_DE` from v2 (`'DE'`, fetched 2026-02-21) is identical to `DE_LU`.
- The report is `data/reports/entsoe_2023_2024_diff.csv`.

As a control, shifting v2 by one 15-min step makes 70,162 of 70,163 `DE_WindSolar_Error` cells differ, so the comparison is sensitive.

What this shows:
- No 2023–24 value was revised between the two fetches.
- v2's forward-fill did not hide any API gap in 2023–24: the gap-free re-fetch has the same values everywhere.

### 2025

2025 has 35,036 shared rows. The new file also has the 4 CET autumn rows that v2 lacks. A cell differs if |Δ| > 1e-6 or exactly one side is NaN. The report is `data/reports/entsoe_2025_diff.csv`.

| Column | Differing cells | Max abs diff | Where |
|---|---|---|---|
| DA_Price_DE | 0 | 0 | — |
| DA_Price_CH | 0 | 0 | — |
| DA_Price_Spread_DE_CH | 0 | 0 | — |
| CH_Load_Forecast | 0 | 0 | — |
| Sched_CH_IT | 0 | 0 | — |
| Sched_IT_CH | 0 | 0 | — |
| Sched_FR_CH | 4 | 8.0 MW | 2025-05-26 21:00–21:45 UTC: v2 has one hourly value, the new file has 15-min values |
| Sched_DE_CH | 16 | 203.7 MW | only 2025-08-04, between 11:00 and 20:45 UTC |
| CH_Pump_Gen | 696 | 2290.5 MW | Oct 2025 (564 cells, mostly 0.01 MW rounding, some larger) and 2025-12-02 / 12-31 (132 cells) |
| DE_WindSolar_Error | 5,948 | 4129.9 MW | Mar 2025: 2,972 cells, 99.8 % below 1 MW (rounding), a handful around 4,000 MW. Dec 2025: all 2,976 cells, median 396 MW, mean new − old +289 MW |

Interpretation:
- The two sources **agree** on all day-ahead quantities, prices included.
- The ex-post quantities (DE actual generation, CH pumped storage) and one day of DE→CH schedules differ in a few months. This is consistent with ENTSO-E data being revised after v2 was fetched on 2026-02-21, especially December 2025, which was only weeks old at the time. That explanation is an inference: entsoe-py does not expose a revision history.

## Swissgrid 2025: republished file (2026-05-19) vs old repo copy

- Same 35,040 rows, identical timestamps, headers and units. The report is `data/reports/swissgrid_2025_diff.csv`.
- **None of the columns used in this dataset changed:** cols 6, 7, 12–17, 21, 22 have 0 differing cells.
- Changed:
  - col 1 "Total energy consumed by end users": 17,668 cells from 2025-07-01, max 16,514 kWh;
  - col 2 production: 28,043 cells;
  - col 3 consumption: all 35,040 cells;
  - col 5 grid feed-in: all 35,040 cells, max 1.81 GWh;
  - most canton production and consumption columns (25–60), mainly from July 2025.
- Col 1 is used only by the offset test in `verify_alignment.py`.

## Alignment verification (`verify_alignment.py`)

- **Method.** Compute r(k) = corr(Swissgrid(t), ENTSO-E(t−k)) for k = −8…+8 steps, in Jan and Jul of 2024, 2025 and 2026. Pairs:
  - `CH_Load_Forecast` vs Swissgrid col 1;
  - `Sched_DE_CH` vs `DE_CH_mw − CH_DE_mw`.
- **Two variants:** levels, and 15-min first differences.
- **Pass rule** (decided 2026-11-05, see LOG.md):
  - `Sched_DE_CH`, first differences: best k = 0.
  - `CH_Load_Forecast`, levels: |best k| ≤ 3. The load forecast is hourly and smooth, so it can't locate a 15-min offset, but this rule still catches the old ±1–2 h bug.
- **Result:** all 12 rows pass. On first differences, Sched_DE_CH peaks at k = 0 in every month with r 0.32–0.34, except Jul 2024 at 0.107. At k = ±1, r is at most 0.14. Secondary peaks at k = ±4 (one hour) reach 0.24, and in Jul 2024 r(−4) = 0.103 is close to r(0) = 0.107. The table is in `data/reports/offset_test.csv`.

## 2025 spike rate check

Spike = `pos_sec_price` above the P90 of `pos_sec_price` over [T − 90 days, T).

| File | 2025 rows | Spikes | Rate |
|---|---|---|---|
| old `data/raw_merged_*.csv` | 35,045 | 2,893 | 0.08255 |
| new `merged_2023_2026.parquet` | 35,040 | 2,892 | 0.08253 |

Rebuilding with the re-fetched 2023–24 ENTSO-E data gives identical numbers: the label depends on Swissgrid prices only, so ENTSO-E changes leave it unchanged. Removing the duplicate rows and applying the 2023–24 shift changes it by about 0.00002.

## Rebuild from scratch

Run from the repo root:

```bash
# 1. environment
conda env create -f afrr-phase3/environment.yml        # creates env "afrr3"
conda activate afrr3

# 2. Swissgrid raw files (2023/2024 are the existing data/raw/ copies)
mkdir -p afrr-phase3/data/raw
curl -fL -o afrr-phase3/data/raw/EnergieUebersichtCH-2025.xlsx \
  https://www.swissgrid.ch/dam/jcr:bbbbadc6-3656-4dfe-b81e-bbb9ad01be79/EnergieUebersichtCH-2025.xlsx
curl -fL -o afrr-phase3/data/raw/EnergieUebersichtCH-2026.xlsx \
  https://www.swissgrid.ch/dam/jcr:805e525c-44fe-4701-a227-6144193257ac/EnergieUebersichtCH-2026.xlsx
# expected SHA-256 (as fetched 2026-11-05):
#   2025: 48ba9c57086fc0f00c96ba5cf7f449b0f51fd098451783bc4f3c36628a2944de
#   2026: 42726ea9e0e075831250d92048a13b98d772a1201af52c995a20cca809b5ee4a

# 3. ENTSO-E (needs ENTSOE_TOKEN in .env; roughly 1-2 min per month)
python afrr-phase3/fetch_entsoe.py --start 2023-01-01 --end 2024-12-31 \
  --out afrr-phase3/data/raw/entsoe_2023_2024.csv
python afrr-phase3/fetch_entsoe.py --start 2025-01-01 --end 2026-09-30 \
  --out afrr-phase3/data/raw/entsoe_2025_2026.csv

# 4. merge (the one command that rebuilds the dataset from the raw files)
python afrr-phase3/merge_raw_data.py

# 5. optional checks / reports
python afrr-phase3/compare_swissgrid_2025.py
python afrr-phase3/compare_entsoe.py --new afrr-phase3/data/raw/entsoe_2023_2024.csv \
  --from 2023-01-01 --to 2025-01-01 --out afrr-phase3/data/reports/entsoe_2023_2024_diff.csv
python afrr-phase3/compare_entsoe.py --new afrr-phase3/data/raw/entsoe_2025_2026.csv \
  --from 2025-01-01 --to 2026-01-01 --split 2025-10-01 \
  --out afrr-phase3/data/reports/entsoe_2025_diff.csv
python afrr-phase3/verify_alignment.py
```

A re-download can change the result: Swissgrid may republish its files, and ENTSO-E revises ex-post data. Check the SHA-256 above, and compare a new ENTSO-E fetch against the previous one with `compare_entsoe.py`. Its `OLD` constant points at v2 and can be repointed. Steps 4–5 need the old `data/raw/` Swissgrid 2023/2024 files and, for the comparisons and the spike-rate check, the old v2 and `data/raw_merged_*.csv` files. All of these are local-only: `data/` is git-ignored.
