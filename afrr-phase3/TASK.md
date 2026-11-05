# Phase-3 forecasting task (fixed for all experiments)

| | |
|---|---|
| Data | `data/merged_2023_2026.parquet` only (see DATA.md) |
| Code | `common.py` (constants, schema), `labels.py`, `features.py`, `metrics.py`, `baselines.py`, `check_leakage.py`, `run_baselines.py` |
| Rebuild | `python afrr-phase3/run_baselines.py` (conda env `afrr3`, about 3 min) |

## Target, origins, horizons
- **Target:** `pos_sec_price` (EUR/MWh) of the 15-min interval starting at T+h.
- **Forecast origins T:** every full hour, in Swiss local time. Because local offsets are whole hours, this is also every full hour in UTC.
- **Horizons:** h = 4 steps (1 h) and h = 96 steps (24 h). A step is 15 min, and T+h is computed on the UTC grid. Each horizon is trained and evaluated separately.
- **Origins per period:** an origin is used only if both T and T+h fall in the period. Counts:

| Period | h = 4 | h = 96 |
|---|---|---|
| Test A | 8,759 | 8,736 |
| Test B | 6,550 | 6,527 |

## Information available at T (applies to every phase-3 model, foundation models included)
1. **Observed columns:** only rows with timestamp ≤ T − 15 min.
   - Timestamps mark the interval **start**, so the row at T covers [T, T+15 min) and is not settled at T.
   - Every lag k ≥ 1 is the row at T − k·15 min, and rolling windows end at T − 15 min.
   - For foundation models, the context window ends at T − 15 min.
   - **Assumption:** aFRR prices and volumes, physical flows and ENTSO-E actuals for every interval that ended by T are available at T. In practice Swissgrid publishes with a delay, and ENTSO-E actuals are later revised (DATA.md, *Revisions*). This is an optimistic assumption.
2. **Known-day-ahead columns:** `Sched_DE_CH`, `Sched_CH_IT`, `Sched_FR_CH`, `Sched_IT_CH`, `DA_Price_CH`, `DA_Price_DE`, `DA_Price_Spread_DE_CH`, `CH_Load_Forecast`.
   - A value for delivery day d (the Swiss local date of the row) is available at T only if d is on or before the local date of T, or d is the next day and T is at or after 13:00 local.
   - Otherwise the value is NaN.
   - This also applies to known-future covariates of foundation models.
   - **Assumption:** the day-ahead auction results, schedules and load forecast for day D+1 are usable from 13:00 on day D.
3. **Calendar features** come from timestamps only.

**Share of origins whose T+h day-ahead values are NaN** under rule 2:

| | h = 4 | h = 96 |
|---|---|---|
| Train (all 3 parts) | 0.0000 | 0.5416 |
| Test A | 0.0000 | 0.5417 |
| Test B | 0.0000 | 0.5417 |

At h = 96 the target is the next day, so its day-ahead values exist only for origins at 13:00–23:00, which is 11/24 of origins (13/24 = 0.5417 are NaN). At h = 4 the target is either the same day, or, for origins from 23:00, the next day; the next-day values are already available then.

`check_leakage.py` enforces both rules. It runs first in `run_baselines.py`.

## Spike label
- **Threshold:** threshold(t) is the 90th percentile (linear interpolation) of `pos_sec_price` over the time window [t − 90 days, t). The last row in the window is t − 15 min.
- **Label:** spike(t) = `pos_sec_price(t) > threshold(t)`. Both are computed once and stored in `data/labels.parquet`.
- **Warm-up:** if t − 90 days is before the first row (2023-01-01 00:00 local), threshold and label are NaN. The first labelled row is 2023-04-01 01:00 local. Rows without a label are excluded from training labels and from evaluation.
- **Which threshold is used where:**
  - Labels and metrics use threshold(T+h), stored as the `threshold` column of every prediction table.
  - Models use the latest threshold known at T, which is threshold(T) (window ends at T − 15 min). This is feature `thr_known`. It differs from threshold(T+h) only through the last h steps of the window.

**Spike rate per year** (labelled rows):

| Year | Labelled rows | Spike rate | Full-hour rows | Spike rate at full hours |
|---|---|---|---|---|
| 2023 (from 2023-04-01) | 26,400 | 0.0866 | 6,600 | 0.1036 |
| 2024 | 35,136 | 0.1189 | 8,784 | 0.1413 |
| 2025 (test A) | 35,040 | 0.0825 | 8,760 | 0.0882 |
| 2026 to Sep (test B) | 26,204 | 0.1195 | 6,551 | 0.1359 |

**Training spike rate** (full-hour rows of the training period, which are the rows targets can fall on): **0.12513**. It is used as the constant prediction for `brier_ref` and as the climatology spike probability.

## Splits (no tuning on test data)

| Part | Swiss local | UTC bounds | Use |
|---|---|---|---|
| train/fit | 2023-01-01 → 2024-09-13 | [2022-12-31 23:00, 2024-09-13 07:00) | model fitting |
| train/es | 2024-09-13 → 2024-11-07 | [2024-09-13 07:00, 2024-11-07 03:00) | early stopping (XGBoost) |
| train/cal | 2024-11-07 → 2024-12-31 | [2024-11-07 03:00, 2024-12-31 23:00) | isotonic calibration (classifier) |
| test A | 2025-01-01 → 2026-01-01 | | evaluation |
| test B | 2026-01-01 → 2026-10-01 | | evaluation |

- **How the parts are built:**
  - es + cal are the last 15 % of the training span, in time order. The tail is split in half, and the bounds are rounded down to a full hour.
  - A part [lo, hi) contains the origins with T ≥ lo and T+h < hi. This leaves a purge gap of h steps between consecutive parts.
  - Models are not refitted on the full training period.
- **Training origins:** full hours only, as in the test periods.
  - h=4: fit 12,752, es 1,315, cal 1,315.
  - h=96: fit 12,752, es 1,292, cal 1,292.
- **Spike rates differ between the parts:** fit 0.119, es 0.192, cal 0.116.

## Features (46; `features.feature_names()`)

Offsets are in 15-min steps relative to the origin T, and "row T−k" means timestamp T − k·15 min. All windows require complete history (`min_periods` = window); before that the feature is NaN, which XGBoost handles natively. There is no normalisation.

| Feature | Definition | Rows used |
|---|---|---|
| `p_lag1`, `p_lag4`, `p_lag96` | `pos_sec_price` | T−1, T−4, T−96 |
| `p_mean_96`, `p_std_96` | mean and sample std of price, last day | T−96 … T−1 |
| `p_mean_672`, `p_std_672` | mean and sample std of price, last week | T−672 … T−1 |
| `thr_known` | P90 of price over [T − 90 d, T) | rows in [T−90 d, T−1] |
| `DE_WindSolar_Error_lag1`, `_lag4`, `_mean_96` | DE wind+solar actual − forecast | T−1, T−4; T−96 … T−1 |
| `UF_DE_lag1`, `_lag4`, `_mean_96` | unplanned flow DE→CH = (`DE_CH_mw` − `CH_DE_mw`) − `Sched_DE_CH` | same |
| `UF_FR_lag1`, `_lag4`, `_mean_96` | unplanned flow FR→CH = (`FR_CH_mw` − `CH_FR_mw`) − `Sched_FR_CH` | same |
| `pos_sec_vol_mw_lag1`, `_lag4`, `_mean_96` | activated positive aFRR | same |
| `neg_sec_vol_mw_lag1`, `_lag4`, `_mean_96` | activated negative aFRR | same |
| `CH_Pump_Gen_lag1`, `_lag4` | CH pumped-storage generation | T−1, T−4 |
| `{col}_tph` for each of the 8 known-day-ahead columns | value at T+h; NaN if not yet published (rule 2) | T+h |
| `{col}_chg` for the same 8 | value(T+h) − value(T); NaN if T+h not yet published | T+h, T |
| `cal_hour`, `cal_dow`, `cal_month`, `cal_weekend` | local hour, day of week (Mon = 0), month, weekend flag of T+h | timestamp of T+h |
| `cal_holiday` | T+h falls on a Swiss national public holiday (hard-coded below) | timestamp of T+h |

- **Value at T for `_chg`:** the day-ahead value of the row at T has delivery day = day of T, so it is always available.
- **`thr_known`** was not in the requested feature list. I added it because the label is defined relative to this rolling threshold. It uses only past prices.
- **Holidays (hard-coded in `features.py` and checked against an Easter computation):** New Year, Good Friday, Easter Monday, Ascension, Whit Monday, National Day (1 Aug), Christmas, St Stephen's Day.
  - Only 1 Aug is a federal holiday by law.
  - New Year, Ascension and Christmas are holidays in every canton; the others in all or almost all cantons.

## Models (baselines)

| Model | Quantiles | Spike probability |
|---|---|---|
| `persistence` | all 21 = `p_lag1` (last observed price, T − 15 min) | 1 if `p_lag1` > `thr_known` else 0 |
| `seasonal_persistence` | all 21 = price at T+h − 96k, with the smallest k ≥ 1 such that the row is ≤ T − 15 min: h=4 → row T−92 (same time yesterday); h=96 → row T−96 (same time 2 days before the target, because "1 day before" would be the row at T itself, not yet observed) | 1/0 vs `thr_known` |
| `climatology` | 21 quantiles of training-period price at full-hour rows with the same local hour as T+h | training spike rate 0.12513 |
| `xgb_clf` | **none (NaN)**: a classifier has no price distribution, so price metrics are NaN | XGBoost `binary:logistic`, early-stopped on train/es (logloss), isotonic regression fitted on train/cal |
| `xgb_quantile` | XGBoost `reg:quantileerror` with all 21 `quantile_alpha` in one model, early-stopped on train/es, quantiles sorted per row | `metrics.prob_exceed(quantiles, thr_known)` |

- **Primary and secondary XGBoost models:**
  - **`xgb_quantile` is the primary XGBoost model for all later comparisons.** It is the reference both for price metrics and for spike metrics, through its derived `spike_prob`.
  - **`xgb_clf` is a secondary baseline.** Its early stopping was cut short by the high spike rate in the train/es part: Sep–Nov 2024, spike rate 0.19 against 0.12 in train/fit. Only **81 trees (h = 4) and 46 trees (h = 96)** were kept. These results were accepted as is, with no retuning.
- **XGBoost settings** (fixed, not tuned): `learning_rate` 0.03, `max_depth` 6, `min_child_weight` 5, `subsample` 0.8, `colsample_bytree` 0.8, `reg_lambda` 1, `tree_method` hist, at most 4,000 trees, early stopping after 100 rounds, `random_state` 0.
- **Trees kept by early stopping:**

| Model | h = 4 | h = 96 |
|---|---|---|
| `xgb_clf` | 81 | 46 |
| `xgb_quantile` | 921 | 723 |

- **Quantile → spike probability** (`prob_exceed`): linear interpolation of the CDF through the points (q_τ, τ), right-continuous at ties. Beyond the outer quantiles it is clamped: P = 0.99 if threshold < q01, P = 0.01 if threshold ≥ q99. Persistence-type point forecasts use the 1/0 rule instead.

## Metrics (`metrics.py`; used by every later experiment)
- **Spike:**
  - `pr_auc`: average precision.
  - `roc_auc`.
  - `brier`: mean (p − label)², no rescaling.
  - `brier_ref`: Brier of always predicting the training spike rate 0.12513.
  - `brier_skill` = 1 − brier / brier_ref.
  - `spike_rate`.
- **Price:**
  - `mae_median` = mean |y − q50|.
  - `crps` = 2 × mean pinball loss over the 21 levels (quantile approximation). For a point forecast this equals the MAE.
  - `pinball_qXX` for each level.
- **Calibration:** `cov_10_90` and `cov_05_95`, the empirical coverage of [q10, q90] and [q05, q95].
- **Stratification by covariate availability** (`evaluate(..., stratify_covariate=True)`, used in `run_baselines.py`):
  - **When it applies:** to every (model, horizon, period) whose origins include both cases. In practice that is h = 96; at h = 4 every origin has its T+h day-ahead values.
  - **What it adds:** metrics computed separately on origins where the T+h known-day-ahead values are available (`covariate_known = known`: T at or after 13:00 local, 11/24 of origins) and where they are not (`unknown`: T before 13:00). The full set is `covariate_known = all`.
  - **Caveat:** the strata also differ by time of day. `known` targets fall at 13:00–23:00 and `unknown` targets at 00:00–12:00, and their spike rates differ (test B: 0.160 vs 0.116). Compare models within a stratum; differences between strata are not purely a covariate effect.
- **Uncertainty:** 95 % CI from a block bootstrap over Swiss local days of the origin (1,000 resamples, seed 0). The resamples are the same for all models on the same origin set, so model comparisons are paired.
- **API:** `metrics.evaluate(preds, train_spike_rate, stratify_covariate=False)` takes any prediction table(s) and returns long format: model, horizon, period, covariate_known, metric, value, ci_low, ci_high, n_origins, n_days, n_spikes.

## Prediction table schema (`common.PRED_SCHEMA`)

One parquet per model × horizon × period: `predictions/{model}_h{h}_{period}.parquet`.

| Column | Type | Meaning |
|---|---|---|
| model, period | string | model name; `test_A` / `test_B` |
| horizon | int16 | 4 or 96 |
| origin_utc, target_utc | datetime64[ns, UTC] | T and T+h |
| y | float64 | realised `pos_sec_price` at T+h |
| threshold | float64 | threshold(T+h), the label threshold |
| q01, q05, q10, …, q90, q95, q99 | float64 | 21 price quantiles, non-decreasing (NaN allowed for spike-only models) |
| spike_prob | float64 | P(spike at T+h), in [0, 1] |

`common.make_pred_table` and `common.validate_pred_table` enforce the schema, sorted quantiles and probabilities in [0, 1].
