# Swiss aFRR Causal Inference Pipeline

## What This Does

This project validates the **physical foundation** of a predictive model for Swiss Secondary Control Energy prices (aFRR). Before building any ML model, we must prove the causal chain:

```
German Wind/Solar Forecast Error
        ↓
Unplanned Loop Flow (DE → CH Border)
        ↓
aFRR Bid Ladder Exhaustion
        ↓
aFRR Price Spike
```

The Go/No-Go decision is based on a **Granger causality test**: if p < 0.05 for "German Wind Error → Swiss aFRR Price" (at any lag 1–4h), the project is commercially viable.

---

## Input Files

Place in `data/raw/`:

| File | Source | Description |
|------|--------|-------------|
| `EnergieUebersichtCH-2023.xlsx` | Swissgrid | 15-min physical flows + aFRR volumes & prices |
| `EnergieUebersichtCH-2024.xlsx` | Swissgrid | Same, 2024 |
| `EnergieUebersichtCH-2025.xlsx` | Swissgrid | Same, 2025 (validation set) |
| `entsoe_swiss_energy_data.csv`  | ENTSO-E   | German wind/solar errors, scheduled flows, pump storage |

---

## Quickstart

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn scipy openpyxl

# 2. Place raw data files in data/raw/

# 3. Run full pipeline
python run_all.py

# Or run individual steps
python scripts/00_load_and_preprocess.py
python scripts/01_feature_engineering.py
python scripts/02_granger_causality.py
python scripts/03_eda_analysis.py
python scripts/04_visualise.py
```

---

## Pipeline Steps

### Step 00 — Load & Preprocess (`00_load_and_preprocess.py`)

**What it does:**
- Reads the `Zeitreihen0h15` sheet from each Swissgrid Excel file
- Extracts 9 key columns (timestamp, aFRR volumes, border flows, prices)
- Converts energy units from **kWh → MW** using: `MW = kWh × 4 / 1000`
  - (15-minute intervals: 1 MWh = 4 × 15-min periods)
- Parses ENTSO-E CSV and converts timezone-aware timestamps to naive CET
- Saves two clean CSVs to `data/processed/`

**Key unit conversion explained:**
```
Swissgrid stores 15-minute energy totals in kWh.
To get average power (MW):
    MW = Energy (kWh) / 1000 kW/MW / (15/60 hours)
       = kWh × 4 / 1000
```

**Outputs:**
- `data/processed/swissgrid_2023_2025.csv`
- `data/processed/entsoe_2023_2024.csv`

---

### Step 01 — Feature Engineering (`01_feature_engineering.py`)

**What it does:**
- Merges Swissgrid and ENTSO-E on timestamp (inner join → 2023–2024 training window)
- Computes all derived features:

| Feature | Formula | Physical Meaning |
|---------|----------|------------------|
| `actual_net_DE_to_CH` | `DE_CH_mw − CH_DE_mw` | Net physical flow Germany→Switzerland |
| `Unplanned_Flow` | `actual_net_DE_to_CH − Sched_DE_CH` | Loop flow: actual minus scheduled |
| `abs_Unplanned_Flow` | `\|Unplanned_Flow\|` | Magnitude of deviation |
| `actual_net_FR_to_CH` | `FR_CH_mw − CH_FR_mw` | Net physical flow France→Switzerland |
| `neg_sec_abs_mw` | `\|neg_sec_vol_mw\|` | Absolute downward activation |
| `net_sec_mw` | `pos_sec_vol_mw + neg_sec_vol_mw` | Net aFRR balance |
| `price_spike` | `pos_sec_price > P90` | Binary flag for extreme prices |
| `hour`, `minute`, `hh_mm` | From timestamp | Time-of-day features |
| `is_turnover` | `minute ∈ {0, 45}` | Schedule transition windows |

**Why `Unplanned_Flow` is the key feature:**
The scheduled commercial exchange (`Sched_DE_CH`) is agreed at day-ahead gate closure. Any deviation in the actual physical flow is unscheduled — it cannot be physically prevented due to Kirchhoff's laws. When German wind overproduces, electrons flow where physics dictates, not where traders agreed. This "loop flow" forces Switzerland's Area Control Error (ACE) above zero, triggering aFRR activation.

**Outputs:**
- `data/processed/merged_training.csv` (2023–2024, ~68,000 rows)
- `data/processed/merged_2025.csv` (2025 Swissgrid only, for validation)

---

### Step 02 — Granger Causality (`02_granger_causality.py`)

**What it does:**
- Downsamples training data to hourly (reduces autocorrelation, improves test validity)
- Runs 3 Granger causality F-tests at lags 1–4 hours:
  1. `DE_WindSolar_Error → Unplanned_Flow` (Link 1)
  2. `Unplanned_Flow → neg_sec_abs_mw` (Link 2)
  3. `DE_WindSolar_Error → pos_sec_price` (Link 3 — **GO/NO-GO**)
- Prints verdict and saves results CSV

**How Granger causality works (manual OLS implementation):**
```
For each lag k = 1, 2, 3, 4:

  Restricted model:   Y(t) ~ intercept + Y(t-1) + ... + Y(t-k)
  Unrestricted model: Y(t) ~ intercept + Y(t-1) + ... + Y(t-k) + X(t-1) + ... + X(t-k)

  F = [(RSS_restricted - RSS_unrestricted) / k] / [RSS_unrestricted / (n - 2k - 1)]

  If F is large → X lags help predict Y → X Granger-causes Y
```

**Results:**
- Link 1 (Wind → Flow): p < 0.001 at all lags ✓
- Link 2 (Flow → aFRR): p < 0.001 at all lags ✓
- Link 3 (Wind → Price): p = 0.008 at lag 3h ✓ → **PROJECT IS GO**

**Why lag 3–4h (not immediate)?**
The ENTSO-E wind error is computed from day-ahead forecasts made ~36 hours before delivery. By the time the imbalance physically manifests (real-time) and clears through the balancing market auction, approximately 3–4 hours of statistical delay is observable in the hourly-aggregated data.

**Outputs:**
- `results/granger_results.csv`

---

### Step 03 — EDA Analysis (`03_eda_analysis.py`)

**What it does:**
Computes 5 sets of statistics that validate each link in the causal chain:

1. **OLS regressions** — Pearson r and R² for all feature pairs
2. **Cross-correlations by lag** — how correlation changes as we shift one variable forward/backward in time
3. **Spike probability by |Unplanned Flow| quintile** — the key non-linear diagnostic
4. **Price by minute-of-hour** — structural imbalance at schedule transitions
5. **Sensitivity map** — binned mean price as a function of Wind Error

**Understanding the weak Pearson r (~0.01–0.03):**

This is NOT a data quality problem. Three physical reasons:

**(A) Capacity bound:** aFRR volume is bounded by contracted reserves (~500 MW). At extreme imbalances, volume hits the ceiling and **price spikes instead**. The relationship volume↔flow is non-linear (flat until threshold, then bounded), which linear correlation cannot capture.

**(B) Dilution across the European grid:** German wind errors affect the entire synchronous European grid. Only a fraction of each MW routes through Switzerland based on Power Transfer Distribution Factors (PTDFs). The signal-to-noise ratio is low in a linear correlation.

**(C) Local noise:** ~40–60% of Swiss aFRR events are driven by purely local factors (domestic plant outages, Swiss load forecast errors) uncorrelated with German renewables.

**The non-linear evidence is in the spike probability table:**
| Quintile | |Unplanned Flow| | Spike Prob | Ratio vs Q1 |
|----------|------------------|------------|-------------|
| Q1 | < 140 MW | ~7.1% | 1.0× |
| Q2 | 140–370 MW | ~7.3% | 1.0× |
| Q3 | 370–720 MW | ~8.9% | 1.3× |
| Q4 | 720–1250 MW | ~12.1% | 1.7× |
| Q5 | > 1250 MW | ~14.5% | **2.1×** |

**Outputs:**
- `results/eda_stats.csv`
- `results/ols_stats.csv`
- `results/cross_correlations.csv`
- `results/spike_prob_by_quintile.csv`
- `results/price_by_minute.csv`
- `results/sensitivity_map.csv`

---

### Step 04 — Visualisation (`04_visualise.py`)

**What it does:**
Generates the 8-panel Go/No-Go report figure using all pre-computed statistics.

**Panel descriptions:**
| Panel | Content |
|-------|---------|
| Row 0 (Banner) | Causal chain diagram + GO/NO-GO verdict badge |
| ① Scatter of Truth | Hexbin: Unplanned Flow vs \|Neg aFRR\| + OLS line |
| ② Wind → Flow | Hexbin: Wind Error vs Unplanned Flow + OLS line |
| ③ Cross-Correlation | Pearson r vs lag (minutes) for each causal link |
| ④ Hour-Turnover | Bar chart: avg price at XX:00, XX:15, XX:30, XX:45 |
| ⑤ Spike Quintile | Bar chart: spike probability per |Unplanned Flow| quintile |
| ⑥ Granger p-values | Log-scale bar chart of p-values per lag and link |
| ⑦ Sensitivity Map | Binned mean price vs Wind Error (U-shape analysis) |
| ⑧ Summary | Text summary + GO/NO-GO verdict box |

**Output:**
- `results/causal_validation_swiss_afrr.png`

---

## Results Interpretation

### What "Granger Causality" Means Commercially

A statistically confirmed Granger link from German Wind Error → Swiss aFRR Price means:
- German wind forecast errors contain **lagged predictive information** about Swiss balancing costs
- This information is not yet "priced in" to the aFRR market at the time of activation
- A model that reads wind forecast errors at H-0 can predict, with statistical edge, whether the next 3–4 hour window will see elevated Swiss aFRR prices

This translates directly to:
- **Hydro pump storage dispatch**: pre-position storage charge/discharge before predicted imbalance windows
- **aFRR bid strategy**: adjust bid prices anticipating market tightness
- **Risk management**: hedge exposure during high-|loop flow| periods

### Why XGBoost/LightGBM (Not Linear Regression)

The causal mechanism is non-linear (price spikes 2.1× more often at extreme loop flows). Linear regression models a constant marginal effect and will systematically underestimate spike probability at the tails. Tree-based models naturally partition the feature space and can learn:
- "When |Unplanned_Flow| > 1,250 MW AND minute == 0 → high spike probability"
- "When Sched_DE_CH is near 0 AND DE_WindSolar_Error < -3,000 MW → positive aFRR likely"

These interaction effects are precisely what gradient boosted trees are designed for.

---

## Dependencies

```
pandas>=1.5
numpy>=1.23
matplotlib>=3.6
scikit-learn>=1.1
scipy>=1.9
openpyxl>=3.0   (for reading .xlsx files)
```

No `statsmodels` required — Granger causality is implemented from scratch using `numpy.linalg.lstsq` + `scipy.stats.f`.
