# afrr-phase3 log

## 2026-11-05: rebuild of the benchmark dataset (data only)

### What was done
- **Environment.** Neither the `timesfm` nor the `base` conda env had entsoe-py (and both lacked openpyxl). With your approval I created conda env **`afrr3`**: Python 3.11.18, pandas 2.3.3, pyarrow 26.0.0, openpyxl 3.1.5, python-dotenv, entsoe-py 0.8.1. It is exported to `environment.yml`, and xgboost, timestral and chronos will be added later.
- **Swissgrid.**
  - Downloaded `EnergieUebersichtCH-2025.xlsx` (republished, Last-Modified 2026-05-19) and `EnergieUebersichtCH-2026.xlsx` (Last-Modified 2026-10-20, data to 2026-09-30 23:45) into `data/raw/`. `../data/raw/` is untouched.
  - Compared the new 2025 file with the old copy (`compare_swissgrid_2025.py`). Rows and timestamps are identical, and none of the columns used in the merge changed. Consumption, production, grid feed-in and canton columns changed, mainly from 2025-07.
- **ENTSO-E.**
  - Wrote `fetch_entsoe.py`, based on `00_fetch_energy_data.py`. Changes: CLI date range and output path, `DE_LU`, errors raised with query label and month, UTC output, both autumn copies kept, no gap filling.
  - Fetched 2025-01-01 → 2026-09-30 to `data/raw/entsoe_2025_2026.csv`: 61,244 rows, 0 NaN.
- **Comparison with v2** (`compare_entsoe_2025.py`, renamed to `compare_entsoe.py` in the next entry).
  - `DA_Price_DE` is identical to DE_LU for all 2025 rows. Every other day-ahead column is also identical.
  - Ex-post columns differ in a few months, likely because of source revisions. See DATA.md.
- **Merge.** Wrote `merge_raw_data.py` with a UTC merge key, 22 columns, and CH_IT/IT_CH added. It produced `merged_2023_2026.parquet` and `.csv`: 131,420 rows, a complete 15-min grid, and 12 NaN per ENTSO-E column.
- **Verification.** Wrote `verify_alignment.py`. The offset test passes and the 2025 spike rate is 0.08255 (old) vs 0.08253 (new).

### What went wrong
1. **ENTSO-E timestamps in the old merge.** As known from the earlier inspection, the old merge parsed naive-local v2 timestamps as UTC (`merge_raw_data.py:126-130`), shifting ENTSO-E +1 h / +2 h. Fixed.
2. **New finding: the old merged file has a one-step (15 min) misalignment in 2023–2024.** The Swissgrid 2023 and 2024 files label the **end** of each interval, while 2025 and 2026 label the **start**. The workbooks' own `Uebersicht` SUM formulas state this (`dat_von+TIME(0,15,0) … dat_bis` vs `dat_von … dat_bis-TIME(0,15,0)`), and the year spans confirm it (00:15 → next year 00:00 vs 00:00 → 23:45).
   - The old merge treated every label as a start. So in `data/raw_merged_2023_2024.csv` (and the copies in afrr-phase1/2) every Swissgrid value sits one row too late relative to ENTSO-E and relative to 2025.
   - It also produced a duplicate row at 2025-01-01 00:00 in `raw_merged_2025.csv`.
   - The lag test on 15-min first differences of Sched_DE_CH vs net DE→CH flow confirms it: best k = −1 in 2024 without the shift, 0 with it.
3. **Bug in my first fetch run, fixed before the final run.** The first full fetch estimated each source point's native interval as min(step to previous, step to next). FR→CH, CH→IT and IT→CH schedules switch between 15-min and hourly resolution from one delivery day to the next. So the first hourly point after a 15-min day got a 15-min interval, and its next three quarter-hours became NaN (30 false NaN cells). The rule now uses the step to the next point, falling back to the step from the previous point when the next step is over 60 min. It is unit-tested on a switch and on a real hole. Re-fetched; the buggy output was moved to scratch, not kept.
4. **The level-correlation offset test can't separate offsets 0 and −1 in some months.** Sched_DE_CH in Jul 2024 gives r(−1) = 0.4937 vs r(0) = 0.4930. The hourly load forecast is too smooth to resolve 15 min at all; its best k ranges from −2 to +2 even when aligned.

### Decisions (all confirmed with you before acting)
- **Env:** new conda env `afrr3` instead of installing into base/timesfm.
- **New ENTSO-E file is UTC with both autumn copies.** v2 and Swissgrid are parsed as naive local time. Swissgrid uses `ambiguous='infer'`. v2 uses `ambiguous=True`, because `infer` fails on v2: only the first (CEST) copy survived `keep='first'`. The missing CET copy is left as NaN.
- **No forward fill in the fetch.** The final `ffill()` was dropped, and hourly → 15-min repetition is limited to each point's own interval.
- **Column `timestamp` dropped** from the 19 old columns: it would duplicate `timestamp_local`. The output has 22 columns.
- **Swissgrid 2023–24 shifted by −15 min** so that all years label the interval start.
- **Offset-test pass rule:** Sched_DE_CH on 15-min first differences must give best k = 0; CH_Load_Forecast on levels must give |best k| ≤ 3 (catches whole-hour shifts). Both tables are reported.
- **2023–2024 ENTSO-E kept from the old v2 file, not re-fetched.** Your rule was to re-fetch if `DA_Price_DE` in v2 is empty or differs from DE_LU for 2025. It is neither: it is non-empty in every row of 2023–2025 and has 0 differing cells against DE_LU in 2025.
  - Caveats: v2 forward-filled any API gaps (undetectable now), lost 4 CET autumn rows per year and 4 rows on 2023-01-01, and may contain values that ENTSO-E has revised since 2026-02-21 (as seen in 2025 for DE actuals and CH pumped storage).
  - Re-fetching 2023–24 with `fetch_entsoe.py` (about 45 min) would remove all three. It's left as an option for you.
- **Swissgrid rows define the dataset and ENTSO-E is left-joined.** The old merge used an inner join. Missing ENTSO-E instants appear as NaN instead of dropping rows.
- **The offset test reads Swissgrid col 1** (end-user consumption) directly from the raw files, joined on `timestamp_utc`. Adding it to the dataset would go beyond the column spec.

### Files
- Code: `fetch_entsoe.py`, `merge_raw_data.py`, `compare_swissgrid_2025.py`, `compare_entsoe_2025.py`, `verify_alignment.py`
- Data: `data/raw/` (2 xlsx, 1 csv), `data/merged_2023_2026.{parquet,csv}`
- Reports: `data/reports/{swissgrid_2025_diff,entsoe_2025_diff,offset_test,spike_rate_2025,missing_values,gaps}.csv`
- Logs: `data/logs/{fetch_2025_2026,merge,compare_entsoe,verify}.log`
- Docs: `DATA.md`, `LOG.md`, `environment.yml`

## 2026-11-05 (later): ENTSO-E 2023–24 re-fetched; first commit

### What was done
- **Re-fetch.** Fetched 2023-01-01 → 2024-12-31 with `fetch_entsoe.py` into `data/raw/entsoe_2023_2024.csv`: 70,176 rows, 0 NaN, no retries. All four years now come from the same UTC fetch, with `DE_LU` and no forward fill.
- **Comparison script.** Generalised `compare_entsoe_2025.py` into `compare_entsoe.py` (`--new --from --to --split --out`, plus per-year counts and a per-month breakdown of differences > 1) and re-ran the 2025 comparison with it. The 2025 results are unchanged.
- **2023–24 vs v2.** 70,164 shared rows: **0 differing cells in every column**, `DA_Price_DE` included. v2 lacks 12 rows (2023-01-01 00:00–00:45 local, and the CET autumn hour in 2023 and 2024). A control with v2 shifted by one step differs in 70,162 of 70,163 cells, so the zero is real.
- **Merge.** `merge_raw_data.py` now uses `entsoe_2023_2024.csv` for 2023–24; v2 is no longer an input. Rebuilt the dataset: 131,420 rows, **0 NaN in every column**, gaps.csv empty.
- **Checks re-run.** The Swissgrid 2025 comparison is unchanged. The offset test passes (same table: the 2023–24 ENTSO-E values are identical, and the 12 restored rows don't fall in the test months). The 2025 spike rate is unchanged at 0.08253, vs 0.08255 in the old file.
- **DATA.md.** Added a *Revisions* section (ENTSO-E measured series are revised after the fact, and this dataset uses the values as of the 2026-11-05 fetch) and a *Snapshot* line. Updated the sources, conventions, missing values, comparison and rebuild commands.

### Decisions (confirmed with you)
- **What gets committed.** The repo `.gitignore` excludes `*.csv`, `*.xlsx` and `*.parquet` ("Data files — never commit"), and origin is public. I committed code, docs, logs and the small report CSVs, force-adding the ignored report CSVs, plus `merged_2023_2026.parquet` as the snapshot. Raw files and the merged CSV are not committed. `.gitignore` is unchanged.
