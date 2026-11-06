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

## 2026-11-05: Step 2, evaluation framework and baselines

### What was done
- **Environment.** Added xgboost 3.4.2 and scikit-learn 1.9.1 to `afrr3` and re-exported `environment.yml`.
- **New code:** `common.py`, `labels.py`, `features.py`, `metrics.py`, `baselines.py`, `check_leakage.py`, `run_baselines.py`.
- **Docs:** TASK.md.
- **Outputs:**
  - `data/labels.parquet` and `labels_meta.json`;
  - 20 prediction tables in `predictions/`;
  - `results/baselines.csv` (long, with CIs), `results/baselines_summary.csv`, `results/task_stats.json`, `results/run_baselines.log`.
- `python afrr-phase3/run_baselines.py` rebuilds everything in about 3 min.

### Decisions (confirmed with you before coding)
1. Observed rows are allowed up to T − 15 min. `p_lag0` is dropped, and every observed-series lag starts at 1.
2. Day-ahead availability: the next day's values are available only from 13:00 local. For h=96, 54.17 % of origins have NaN T+h day-ahead features.
3. The label threshold is NaN during the 90-day warm-up. Those rows are excluded from training and evaluation.
4. Models use `thr_known` = threshold(T); labels and metrics use threshold(T+h).
5. The training tail (last 15 %) is split in half: early stopping, then isotonic calibration. Every part boundary has an h-step purge gap, and there is no refit.
6. Swiss national holidays are hard-coded and checked against an Easter computation.

### Decisions made while implementing (flag if you disagree)
- **`thr_known` as an XGBoost feature.** It is not in the requested list, but the label is relative to this rolling threshold, and it uses only past prices.
- **Seasonal persistence at h=96** uses the row at T−96 (2 days before the target). "Same time one day before the target" would be the row at T, which isn't observed at T.
- **`xgb_clf` has no price quantiles.** All 21 quantile columns are NaN, and the price metrics are NaN for it. The schema allows this for spike-only models.
- **Training origins are full hours only**, as in the test periods. The climatology quantiles are also computed on full-hour rows.
- **Quantile → probability clamp** is [0.01, 0.99] beyond the outer quantiles.
- **`pr_auc`** is average precision (sklearn), not trapezoidal area under the PR curve.
- **Leakage-check tolerance** is 1e-9. Rolling sums restart on the masked slice, so the comparison can't be bit-exact.

### What went wrong (all caught before the final run)
- **Probability clamp bug.** `metrics.py` first set the upper clamp to 1 − 0.99 = 0.01, so every interpolated probability came out as 0.01. A unit test caught it before any results were produced.
- **Lost timezone.** `labels.py` used `.values` on the tz-aware timestamp, which drops the timezone, and the comparison with the UTC split bounds then failed. Fixed by keeping the series.
- **Ambiguous edge cases.** Leakage-check edge cases at the autumn repeated hour raised `AmbiguousTimeError`. They are now matched on local wall-clock time, which picks up both copies.
- **The leakage check got a negative control.** It plants two leaky features (price at T, and DA at T+h without the 13:00 rule) and asserts that both are detected. It ran green before this was added, so it needed proof that it can fail.

### Observations to keep in mind (not acted on: no tuning on test)
- **Shifted early-stopping part.** train/es (2024-09-13 → 11-07) has spike rate 0.19, against 0.12 in train/fit and 0.116 in train/cal. The classifier early-stopped after only 81 trees (h=4) and 46 (h=96).
- **The classifier is weaker than the quantile model's spike probability.** At h=4, test A PR-AUC is 0.164 for `xgb_clf` and 0.197 for `xgb_quantile`; test B is 0.310 vs 0.398.
- **Quantile intervals are too narrow.** `xgb_quantile` covers less than nominal: [q10, q90] 0.70–0.77 against 0.80, [q05, q95] 0.82–0.87 against 0.90.
- **Base rate drops in test A.** Spike rate is 0.088 at full hours in test A vs 0.125 in train, so Brier skill is small for every model (`xgb_quantile` h=4: 0.078 vs ref 0.082).
- **These numbers are much lower than the earlier quoted figures** (PR-AUC 0.390 etc.). Those came from a nowcast with target leakage. They are not comparable.

## 2026-11-05: Step 2 additions (baselines accepted as is, no retuning)
- **Stratification.** Added `metrics.covariate_known()` and `evaluate(..., stratify_covariate=True)`. For h = 96, every metric is also reported for origins with and without the T+h day-ahead values (T ≥ 13:00 local or not). `results/baselines.csv` now has a `covariate_known` column (all | known | unknown) and 1,040 rows. Rerunning `run_baselines.py` reproduced every `all` value exactly.
- **What the split shows (h = 96, `xgb_quantile`):**
  - Test B PR-AUC is 0.445 [0.365, 0.514] when the values are known and 0.205 [0.171, 0.245] when they are not.
  - Test A shows almost no difference: 0.178 vs 0.158.
  - The strata also differ by target time of day and spike rate, so this is not a pure covariate effect (TASK.md).
- **TASK.md:** `xgb_quantile` is the primary XGBoost model. `xgb_clf` is secondary, with early stopping cut short (81 / 46 trees) by the 0.19 spike rate in Sep–Nov 2024.

## 2026-11-06: Step 3 setup, foundation models (before the full run)
- **Environment.** New env `afrr3-fm`, exported to `environment-fm.yml`. Timestral and Chronos-2 installed together without conflicts, so there is one env:
  - Python 3.11, torch 2.15.0 (CPU), timestral 1.0.0 (git `a534bdbd`), chronos-forecasting 2.3.2, transformers 5.22.1, accelerate, einops.
  - pandas is pinned to 2.3.3, as in `afrr3`.
  - TimesFM 2.5 runs in the existing `timesfm` env through `fm_timesfm_worker.py`. That env has no pyarrow, so data goes over `.npz`; the env was not modified.
  - Weights: `mistralai/Timestral-209M` @ `9874d252`, `amazon/chronos-2` @ `29ec3766`, `google/timesfm-2.5-200m-pytorch` (cached copy from phase 2).
- **Protocol.**
  - Context is the 2016 rows ending at T − 15 min.
  - `prediction_length` = 97. Step k is row T + (k−1)·15 min, so h=4 is step 5 and h=96 is step 97. Decided with the user: steps 4 and 96 would forecast one row too early.
  - A dummy forecast equal to the true price gives MAE 0 at both horizons, which confirms the mapping.
  - Known-future inputs: 7 day-ahead columns with the 13:00 rule (NaN until published), plus categorical hour, day of week and holiday. Rows after the end of the data are NaN, which affects only the last day of test B.
  - TimesFM has deciles only. q01 = q05 = q10 and q95 = q99 = q90; q15 … q85 are linear interpolation between deciles. This is flagged as `quantile_note` in the results tables.
  - 15,309 origins: the union of the Step-2 test origins. Of the 15,312 test-period hours, the last origin of each period has its T+4 target outside the period.
- **Input leakage check.** For every configuration, inputs built from data masked with `check_leakage.masked_slice` equal the inputs from the full data (206 origins each).
- **MPS vs CPU, 20 origins.** Everything runs on CPU with 8 threads.
  - Timestral fails on MPS: it uses float64 internally.
  - Chronos-2 aborts with a native Metal NSException (torch 2.15, macOS 14.6).
  - TimesFM matches (max |Δq| 5.2e-4) but is slower on MPS (0.38 vs 0.22 s per call). Passing the check only allows MPS, so CPU was kept. Records are in `results/runtime/mps_check_*.json`.
- **Batching, 20 origins: batched vs single calls, tolerance 1e-4.**
  - TimesFM: max |Δq| = 0 at batch 32, so it is batched with B = 32 (0.054 s per origin vs 0.22).
  - Timestral and Chronos-2: differences of 0.9e-3 to 2.6e-3 (about 100 float32 ULPs; up to 0.5 % relative near zero). They fail the tolerance and stay at one origin per call.
  - Records: `results/runtime/batch_check_*.json`.
- **Two configurations in parallel at 4 threads each** was not faster (Chronos-2 both 0.87 s per origin vs 0.54 s at 8 threads alone; Timestral past_only 1.30 vs 0.72), so configurations run sequentially.
- **Timing, 50 origins.** Seconds per origin for none / past_only / known_future / both:
  - Timestral: 0.20 / 0.72 / 1.06 / 1.71
  - Chronos-2: 0.10 / 0.25 / 0.37 / 0.54
  - TimesFM: 0.22, or 0.054 batched
  - All nine would take about 23 h. Batch 1, about 10.4 h, as decided by the user: the four Chronos-2 configurations, TimesFM, `timestral_none` and `timestral_past_only`. Batch 2 (`timestral_known_future`, `timestral_both`, about 12 h) is not started.
- **Defaults kept.** Timestral `symmetric_averaging=True` and `clip_nonnegative=True` (clipping only applies if the whole context is ≥ 0). TimesFM uses the phase-2 `ForecastConfig` (no positivity, flip invariance, continuous quantile head, quantile-crossing fix). Chronos-2 uses `cross_learning=False`.

## 2026-11-06: Step 3 batch 1 results (7 foundation configurations)
- **Run.** `run_foundation.py` ran timesfm_none, the four Chronos-2 configurations, timestral_none and timestral_past_only.
  - 01:12 → 10:57 CET, 9.76 h, against an estimate of 10.4 h. CPU, 8 threads, sequential.
  - The log (`results/foundation_batch1.log`) has no warnings, errors, NaN or resumes, only torch's macOS "Redirects are currently not supported" note.
  - Batch 2 (`timestral_known_future`, `timestral_both`) is not run.
- **Results.** All origins (`covariate_known = all`), with 95 % day-block bootstrap CIs in `results/foundation.csv` and `all.csv`.
  - Each cell is PR-AUC / CRPS / Brier.
  - Brier ref: h=4 0.0818 (A), 0.1175 (B); h=96 0.0820 (A), 0.1179 (B).

| model | h=4 test A | h=4 test B | h=96 test A | h=96 test B |
|---|---|---|---|---|
| climatology | 0.088 / 58.5 / 0.0818 | 0.136 / 45.6 / 0.1175 | 0.088 / 58.5 / 0.0820 | 0.136 / 45.7 / 0.1179 |
| xgb_quantile | 0.197 / 42.3 / 0.0778 | 0.398 / 27.1 / 0.1014 | 0.168 / 45.4 / 0.0801 | 0.349 / 31.3 / 0.1067 |
| timesfm_none | 0.209 / 41.6 / 0.0759 | 0.391 / 27.8 / 0.1015 | 0.138 / 44.8 / 0.0818 | 0.275 / 32.4 / 0.1131 |
| chronos2_none | 0.258 / 39.3 / 0.0735 | 0.433 / 26.2 / 0.0974 | 0.183 / 42.7 / 0.0783 | 0.315 / 30.4 / 0.1084 |
| chronos2_past_only | 0.274 / 38.8 / 0.0725 | 0.458 / 26.0 / 0.0957 | 0.193 / 42.8 / 0.0787 | 0.326 / 30.6 / 0.1095 |
| chronos2_known_future | 0.268 / 38.7 / 0.0733 | 0.461 / 25.5 / 0.0944 | 0.218 / 41.5 / 0.0768 | 0.374 / 28.8 / 0.1042 |
| chronos2_both | 0.271 / 38.6 / 0.0729 | 0.470 / 25.4 / 0.0937 | 0.217 / 41.3 / 0.0762 | 0.377 / 28.6 / 0.1037 |
| timestral_none | 0.275 / 38.4 / 0.0721 | 0.442 / 25.8 / 0.0961 | 0.194 / 41.8 / 0.0772 | 0.314 / 30.1 / 0.1079 |
| timestral_past_only | 0.288 / 37.9 / 0.0713 | 0.458 / 25.4 / 0.0947 | 0.199 / 41.7 / 0.0768 | 0.310 / 30.0 / 0.1085 |

- **Coverage, [q10, q90] / [q05, q95]** (nominal 0.80 / 0.90):
  - Chronos-2 and Timestral: 0.76–0.83 / 0.87–0.92.
  - xgb_quantile: 0.70–0.77 / 0.82–0.87.
  - TimesFM: 0.77–0.81 / the same values, by construction of its tails.
- **h=96 PR-AUC by `covariate_known` (known / unknown):**
  - Test B: chronos2_none 0.374 / 0.233; chronos2_known_future 0.467 / 0.246; chronos2_both 0.473 / 0.243; xgb_quantile 0.445 / 0.205.
  - Test A, "known": every model without day-ahead covariates is worse here (e.g. chronos2_none 0.139 vs 0.226). The strata differ by time of day (TASK.md caveat).
- **Reading.**
  - h=4: every Chronos-2 and Timestral configuration beats xgb_quantile on PR-AUC, CRPS and Brier in both periods.
  - h=96: only Chronos-2 with known-future covariates beats xgb_quantile on PR-AUC, and the CIs overlap.
  - Past-only covariates add little, and slightly hurt at h=96 test B. Known-future covariates are what helps at h=96.
- **Runtime** (`results/runtime.csv`). Seconds per origin: timestral none 0.214, past_only 0.754; chronos2 0.111 / 0.241 / 0.369 / 0.530; timesfm 0.060 at batch 32. Peak RSS is 1.4–2.3 GB.
- **Observations, not acted on.**
  - TimesFM spike_prob = 0.01 for 69 % of rows. With q95 = q99 = q90, `prob_exceed` jumps from about 0.10 to the 0.01 clamp whenever the threshold is ≥ q90. Its spike metrics partly reflect this tail rule.
  - Chronos-2 q01 < 0 in 50 % of rows, against 0.06 % negative prices (Timestral 6 %, TimesFM 3 %).
  - Timestral `clip_nonnegative` applies whenever the whole context is ≥ 0, which is nearly always.
  - `results/common9.csv` holds the secondary table on q10..q90 for all quantile models.

## 2026-11-06: TimesFM tail sensitivity (timesfm_none_extrap)
- **What.** The same TimesFM forecasts with different outer quantiles, built from the saved `timesfm_none` tables (no rerun; `run_foundation.make_timesfm_extrap`).
  - q95 = q90 + 0.5 (q90 − q80) and q99 = q90 + 0.9 (q90 − q80).
  - q05 = q10 − 0.5 (q20 − q10) and q01 = q10 − 0.9 (q20 − q10), clipped at 0.
  - Where q10 < 0 (3.3 % of rows), clipping at 0 would put q05 and q01 above q10, so they are capped at q10 to keep the quantiles monotone.
  - spike_prob = `prob_exceed(q, threshold(T))`, as for every model.
  - The original `timesfm_none` row is kept. The variant has its own `quantile_note`.
  - Re-evaluating reproduced all 1,736 existing rows of `foundation.csv` exactly. On the 9 common levels the variant equals the original by construction (`common9.csv`).
- **Results** (PR-AUC / Brier / CRPS / cov 05–95, original → extrap):
  - h=4 A: 0.209 → 0.226 / 0.0759 → 0.0750 / 41.6 → 41.1 / 0.802 → 0.843
  - h=4 B: 0.391 → 0.409 / 0.1015 → 0.1001 / 27.8 → 27.5 / 0.773 → 0.814
  - h=96 A: 0.138 → 0.151 / 0.0818 → 0.0803 / 44.8 → 44.3 / 0.811 → 0.860
  - h=96 B: 0.275 → 0.285 / 0.1131 → 0.1117 / 32.4 → 31.9 / 0.769 → 0.822
- **Reading.**
  - The flat tails cost TimesFM about 0.01–0.02 PR-AUC. The share of spike_prob at the 0.01 clamp drops from 69 % to 51 %.
  - The ranking does not change: TimesFM stays below Chronos-2 and Timestral everywhere. With extrapolated tails it slightly passes xgb_quantile at h=4 (test B 0.409 vs 0.398, CIs overlap) but not at h=96.
