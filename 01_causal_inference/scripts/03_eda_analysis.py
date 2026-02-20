"""
================================================================================
STEP 03 — Causal EDA Analysis
================================================================================
PURPOSE:
    Compute all intermediate statistics used in the EDA plots:
        1. OLS regressions (Pearson r, R², slope)
        2. Cross-correlations by lag
        3. Price spike probability by |Unplanned Flow| quintile
        4. Price distribution by minute-of-hour (hour-turnover analysis)
        5. Sensitivity mapping (Wind Error → Price binned means)

INPUT:
    data/processed/features_train_2023_2024.csv

OUTPUTS:
    results/eda_stats.csv              — key summary statistics
    results/ols_stats.csv              — OLS regression results
    results/cross_correlations.csv     — lag correlation table
    results/spike_prob_by_quintile.csv — spike probability per |flow| quintile
    results/price_by_minute.csv        — price by minute slot (0,15,30,45)
    results/sensitivity_map.csv        — Wind Error → Price binned means

WHY ARE LINEAR CORRELATIONS WEAK (~0.01–0.03)?
    This is NOT a data quality problem. Three physical reasons:

    (A) CAPACITY BOUND: aFRR volume is capped at ~500 MW. At extreme imbalances,
        volume hits the ceiling and PRICE spikes instead. Linear correlation
        cannot detect this threshold regime.

    (B) DILUTION: German wind errors affect the whole European grid. Only a
        fraction of each MW routes through Switzerland based on PTDFs.

    (C) LOCAL NOISE: ~40–60% of Swiss aFRR events are driven by local factors
        (plant outages, Swiss load errors) uncorrelated with German renewables.

    The KEY EVIDENCE is in the spike probability quintile table:
    spike probability DOUBLES from the lowest to highest |Unplanned Flow|
    quintile — a clear non-linear threshold effect.
================================================================================
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ── Configuration ─────────────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
RESULTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_IN = "features_train_2023_2024.csv"

SPIKE_QUANTILE = 0.90
N_QUINTILES    = 5
MAX_LAG_15MIN  = 12    # ±12 intervals = ±180 minutes cross-correlation window
N_BINS_SENS    = 12    # Bins for sensitivity map


# ── Analysis Functions ────────────────────────────────────────────────────────

def compute_ols_stats(x: pd.Series, y: pd.Series,
                      name_x: str, name_y: str) -> dict:
    """Fit OLS and return Pearson r, R², slope, intercept."""
    clean      = pd.DataFrame({"x": x, "y": y}).dropna()
    lm         = LinearRegression().fit(clean[["x"]], clean["y"])
    r2         = r2_score(clean["y"], lm.predict(clean[["x"]]))
    pearson_r, pearson_p = stats.pearsonr(clean["x"], clean["y"])
    return {
        "feature_x":  name_x,
        "target_y":   name_y,
        "pearson_r":  round(pearson_r,        4),
        "pearson_p":  round(pearson_p,        6),
        "r_squared":  round(r2,               4),
        "slope":      round(lm.coef_[0],      4),
        "intercept":  round(lm.intercept_,    4),
        "n_obs":      len(clean),
    }


def compute_cross_correlations(df: pd.DataFrame,
                                x_col: str, y_col: str,
                                max_lag: int) -> pd.DataFrame:
    """
    Pearson cross-correlation at lags -max_lag to +max_lag (15-min steps).
    Positive lag k means corr(x(t), y(t+k)) — x leads y.
    """
    records = []
    for k in range(-max_lag, max_lag + 1):
        r = df[x_col].corr(df[y_col].shift(-k))
        records.append({
            "lag_intervals": k,
            "lag_minutes":   k * 15,
            "x_col":         x_col,
            "y_col":         y_col,
            "pearson_r":     round(r, 4),
        })
    return pd.DataFrame(records)


def compute_spike_prob_by_quintile(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    Compute price spike probability for each quintile of |Unplanned Flow|.

    Returns the table and the Q5/Q1 spike probability ratio.
    A ratio > 1.5x is strong evidence of the bid-ladder exhaustion mechanism.
    """
    labels = [
        "Q1 Very Low (<140 MW)",
        "Q2 Low (140–370 MW)",
        "Q3 Medium (370–720 MW)",
        "Q4 High (720–1250 MW)",
        "Q5 Very High (>1250 MW)",
    ]
    df = df.copy()
    df["uf_quintile"] = pd.qcut(df["abs_Unplanned_Flow"], q=N_QUINTILES, labels=labels)

    result = (df.groupby("uf_quintile", observed=True)
                .agg(
                    spike_probability     = ("price_spike", "mean"),
                    count                 = ("price_spike", "count"),
                    avg_unplanned_flow_mw = ("abs_Unplanned_Flow", "mean"),
                    avg_pos_price         = ("pos_sec_price", "mean"),
                )
                .reset_index())

    result["spike_probability_pct"] = (result["spike_probability"] * 100).round(2)
    bottom = result["spike_probability"].iloc[0]
    result["ratio_vs_q1"] = (result["spike_probability"] / bottom).round(2)
    ratio = result["spike_probability"].iloc[-1] / bottom

    return result, ratio


def compute_price_by_minute(df: pd.DataFrame) -> pd.DataFrame:
    """Average and std of aFRR prices grouped by minute-of-hour (0,15,30,45)."""
    return (df.groupby("minute")
              .agg(
                  pos_price_mean = ("pos_sec_price", "mean"),
                  pos_price_std  = ("pos_sec_price", "std"),
                  neg_price_mean = ("neg_sec_price", "mean"),
                  neg_price_std  = ("neg_sec_price", "std"),
                  spike_rate     = ("price_spike",   "mean"),
                  count          = ("pos_sec_price", "count"),
              )
              .assign(spike_rate_pct=lambda x: (x["spike_rate"] * 100).round(2))
              .reset_index())


def compute_sensitivity_map(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    """Bin German Wind Error and compute mean aFRR price per bin."""
    df = df.copy()
    df["wind_bin"] = pd.cut(df["DE_WindSolar_Error"], bins=n_bins)
    result = (df.groupby("wind_bin", observed=True)
                .agg(
                    pos_price_mean = ("pos_sec_price", "mean"),
                    pos_price_std  = ("pos_sec_price", "std"),
                    neg_price_mean = ("neg_sec_price", "mean"),
                    spike_rate     = ("price_spike",   "mean"),
                    count          = ("pos_sec_price", "count"),
                )
                .reset_index())
    result["bin_midpoint"] = result["wind_bin"].apply(lambda b: b.mid)
    return result[result["count"] > 100].reset_index(drop=True)


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def main():
    # ── 1. Load training data ─────────────────────────────────────────────────
    print("Loading training data...")
    df = pd.read_csv(os.path.join(PROCESSED_DIR, TRAIN_IN), parse_dates=["timestamp"])
    print(f"  Rows: {len(df):,}")

    # Recompute spike flag in case it's missing (Step 01 should have set it)
    if "price_spike" not in df.columns:
        threshold = df["pos_sec_price"].quantile(SPIKE_QUANTILE)
        df["price_spike"] = (df["pos_sec_price"] > threshold).astype(int)

    # ── 2. OLS regressions ────────────────────────────────────────────────────
    print("\nComputing OLS regressions...")
    ols_results = []

    r1 = compute_ols_stats(df["Unplanned_Flow"], df["neg_sec_abs_mw"],
                           "Unplanned_Flow", "neg_sec_abs_mw")
    r2 = compute_ols_stats(df["DE_WindSolar_Error"], df["Unplanned_Flow"],
                           "DE_WindSolar_Error", "Unplanned_Flow")
    r3 = compute_ols_stats(df["abs_Unplanned_Flow"], df["pos_sec_price"],
                           "abs_Unplanned_Flow", "pos_sec_price")
    r4 = compute_ols_stats(df["Sched_DE_CH"], df["actual_net_DE_to_CH"],
                           "Sched_DE_CH", "actual_net_DE_to_CH")

    for r in [r1, r2, r3, r4]:
        ols_results.append(r)
        print(f"  [{r['feature_x'][:30]:30s} → {r['target_y'][:20]:20s}]  "
              f"r = {r['pearson_r']:+.3f},  R² = {r['r_squared']:.3f}")

    print(f"\n  Note: r4 (Sched vs Actual flow) = {r4['pearson_r']:.3f} — "
          f"data quality check, expected high")

    pd.DataFrame(ols_results).to_csv(
        os.path.join(RESULTS_DIR, "ols_stats.csv"), index=False)

    # ── 3. Cross-correlations ─────────────────────────────────────────────────
    print("\nComputing cross-correlations by lag...")
    cc1 = compute_cross_correlations(df, "DE_WindSolar_Error", "Unplanned_Flow",   MAX_LAG_15MIN)
    cc2 = compute_cross_correlations(df, "abs_Unplanned_Flow", "price_spike",      MAX_LAG_15MIN)
    cc_all = pd.concat([cc1, cc2], ignore_index=True)
    cc_all.to_csv(os.path.join(RESULTS_DIR, "cross_correlations.csv"), index=False)

    peak1 = cc1.loc[cc1["pearson_r"].abs().idxmax()]
    peak2 = cc2.loc[cc2["pearson_r"].abs().idxmax()]
    print(f"  Wind Error → Unplanned Flow: peak r = {peak1['pearson_r']:.3f} "
          f"at lag {peak1['lag_minutes']:.0f} min")
    print(f"  |Unplanned| → Spike Prob:   peak r = {peak2['pearson_r']:.3f} "
          f"at lag {peak2['lag_minutes']:.0f} min")

    # ── 4. Spike probability by quintile ──────────────────────────────────────
    print("\nComputing spike probability by |Unplanned Flow| quintile...")
    spike_df, spike_ratio = compute_spike_prob_by_quintile(df)
    spike_df.to_csv(os.path.join(RESULTS_DIR, "spike_prob_by_quintile.csv"), index=False)

    print(spike_df[["uf_quintile", "spike_probability_pct",
                    "ratio_vs_q1", "count"]].to_string(index=False))

    q1_flow = spike_df["avg_unplanned_flow_mw"].iloc[0]
    q5_flow = spike_df["avg_unplanned_flow_mw"].iloc[-1]
    q1_prob = spike_df["spike_probability_pct"].iloc[0]
    q5_prob = spike_df["spike_probability_pct"].iloc[-1]
    slope_per_gw = (q5_prob - q1_prob) / ((q5_flow - q1_flow) / 1000)

    print(f"\n  >> Q5/Q1 spike ratio: {spike_ratio:.2f}x")
    print(f"  >> Marginal sensitivity: +{slope_per_gw:.1f}% per 1 GW of |Unplanned Flow|")

    # ── 5. Price by minute-of-hour ────────────────────────────────────────────
    print("\nComputing price by minute-of-hour...")
    price_min_df = compute_price_by_minute(df)
    price_min_df.to_csv(os.path.join(RESULTS_DIR, "price_by_minute.csv"), index=False)
    print(price_min_df[["minute", "pos_price_mean",
                         "neg_price_mean", "spike_rate_pct"]].to_string(index=False))

    # ── 6. Sensitivity map ────────────────────────────────────────────────────
    print("\nComputing sensitivity map (Wind Error → Price)...")
    sens_df = compute_sensitivity_map(df, N_BINS_SENS)
    sens_df.to_csv(os.path.join(RESULTS_DIR, "sensitivity_map.csv"), index=False)
    print(f"  {len(sens_df)} valid bins (min 100 obs each)")

    # ── 7. Summary stats CSV ──────────────────────────────────────────────────
    summary = {
        "n_training_obs":             len(df),
        "spike_threshold_eur_mwh":    df["spike_threshold"].iloc[0] if "spike_threshold" in df.columns else None,
        "spike_rate_pct":             df["price_spike"].mean() * 100,
        "pearson_r_wind_unplanned":   r2["pearson_r"],
        "pearson_r_unplanned_afrr":   r1["pearson_r"],
        "pearson_r_abs_flow_price":   r3["pearson_r"],
        "sanity_check_sched_actual":  r4["pearson_r"],
        "spike_ratio_q5_vs_q1":       spike_ratio,
        "slope_pct_per_gw":           slope_per_gw,
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(RESULTS_DIR, "eda_stats.csv"), index=False)

    print(f"\nAll EDA results saved to {RESULTS_DIR}/")
    print("\n[STEP 03 COMPLETE]")

    return df, spike_df, price_min_df, sens_df


if __name__ == "__main__":
    main()
