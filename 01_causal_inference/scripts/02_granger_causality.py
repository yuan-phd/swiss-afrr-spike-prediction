"""
================================================================================
STEP 02 — Granger Causality Tests
================================================================================
PURPOSE:
    Statistically test whether German wind/solar forecast errors Granger-cause
    Swiss aFRR price movements. This is the primary Go/No-Go decision metric.

INPUT:
    data/processed/features_train_2023_2024.csv

OUTPUT:
    results/granger_results.csv    — p-values and F-stats for all three chains
    Console: GO / NO-GO verdict

WHAT IS GRANGER CAUSALITY?
    A variable X "Granger-causes" Y if knowing the past values of X
    significantly improves our ability to predict Y, beyond what Y's own
    past values alone can tell us.

    We test this by comparing two OLS models:
        Restricted model:   Y(t) = f( Y(t-1), Y(t-2), ..., Y(t-lag) )
        Unrestricted model: Y(t) = f( Y(t-1), ..., Y(t-lag), X(t-1), ..., X(t-lag) )

    If the unrestricted model fits significantly better (F-test), we reject
    the null hypothesis that X does NOT Granger-cause Y.

    IMPORTANT: Implemented manually using scipy (OLS + F-test) — identical
    math to statsmodels.tsa.stattools.grangercausalitytests().

THREE CAUSAL LINKS TESTED:
    Link 1: DE_WindSolar_Error  →  Unplanned_Flow
            "German renewable surprise creates unplanned loop flow"

    Link 2: Unplanned_Flow      →  neg_sec_abs_mw
            "Unplanned loop flow forces aFRR downward activation"

    Link 3: DE_WindSolar_Error  →  pos_sec_price       ← GO/NO-GO TEST
            "German renewable surprise ultimately moves Swiss balancing price"

WHY DOWNSAMPLE TO HOURLY?
    The 15-minute data has strong autocorrelation at lag-1. Downsampling:
    (a) reduces multicollinearity in the lag matrix
    (b) better captures the propagation delay (minutes to hours)
    (c) speeds up computation significantly

GO/NO-GO DECISION RULE:
    If any lag (1h–4h) has p-value < 0.05 for Link 3 → PROJECT IS GO
    The lag 3-4h significance reflects the physical delay between when
    the day-ahead wind forecast was made and when the imbalance clears
    through the balancing market.
================================================================================
"""

import os
import pandas as pd
import numpy as np
from scipy import stats

# ── Configuration ─────────────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
RESULTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_IN    = "features_train_2023_2024.csv"
RESULTS_OUT = "granger_results.csv"

MAX_LAG     = 4       # Maximum lag in hours (1h, 2h, 3h, 4h)
P_THRESHOLD = 0.05


# ── Granger F-Test Implementation ────────────────────────────────────────────

def granger_f_test(y_series: pd.Series, x_series: pd.Series,
                   max_lag: int = 4) -> list[dict]:
    """
    Manually compute the Granger causality F-test for lags 1..max_lag.

    Parameters
    ----------
    y_series : pd.Series — the dependent variable (effect)
    x_series : pd.Series — the candidate cause variable
    max_lag  : int       — maximum number of lags to test

    Returns
    -------
    List of dicts: lag, F_stat, p_value, df1, df2, significant

    Algorithm for each lag k:
        1. Y_lags = [y(t-1), ..., y(t-k)],  X_lags = [x(t-1), ..., x(t-k)]
        2. Restricted:   y ~ intercept + Y_lags
        3. Unrestricted: y ~ intercept + Y_lags + X_lags
        4. F = ((RSS_r - RSS_u) / k) / (RSS_u / df2)
           df2 = n - 2k - 1
        5. p from F(k, df2) distribution
    """
    y = y_series.reset_index(drop=True)
    x = x_series.reset_index(drop=True)
    results = []

    for lag in range(1, max_lag + 1):
        n = len(y) - lag
        if n <= 2 * lag + 1:
            results.append({"lag": lag, "F_stat": np.nan, "p_value": np.nan,
                            "df1": lag, "df2": np.nan, "significant": False})
            continue

        Y      = y.values[lag:]
        Y_lags = np.column_stack([y.values[lag-k : len(y)-k] for k in range(1, lag+1)])
        X_lags = np.column_stack([x.values[lag-k : len(x)-k] for k in range(1, lag+1)])

        def ols_rss(X_design, y_target):
            beta = np.linalg.lstsq(X_design, y_target, rcond=None)[0]
            return np.sum((y_target - X_design @ beta) ** 2)

        ones = np.ones((n, 1))
        rss_r = ols_rss(np.hstack([ones, Y_lags]),           Y)
        rss_u = ols_rss(np.hstack([ones, Y_lags, X_lags]),   Y)

        df1 = lag
        df2 = n - 2 * lag - 1

        if df2 <= 0 or rss_u <= 0:
            results.append({"lag": lag, "F_stat": np.nan, "p_value": np.nan,
                            "df1": df1, "df2": df2, "significant": False})
            continue

        F_stat  = ((rss_r - rss_u) / df1) / (rss_u / df2)
        p_value = 1 - stats.f.cdf(F_stat, df1, df2)

        results.append({
            "lag":         lag,
            "F_stat":      round(F_stat,  3),
            "p_value":     round(p_value, 6),
            "df1":         df1,
            "df2":         df2,
            "significant": p_value < P_THRESHOLD,
        })

    return results


def print_granger_results(name: str, results: list[dict]) -> None:
    """Pretty-print Granger test results for one causal link."""
    print(f"\n  Granger: {name}")
    print(f"  {'Lag':>6}  {'F-stat':>8}  {'p-value':>10}  {'Result':>15}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*15}")
    for r in results:
        if np.isnan(r["p_value"]):
            result_str = "insufficient data"
        elif r["significant"]:
            result_str = "✓ SIGNIFICANT"
        else:
            result_str = "✗ not significant"
        print(f"  {r['lag']:>4}h   {r['F_stat']:>8.2f}  {r['p_value']:>10.5f}  {result_str}")


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def main():
    # ── 1. Load training data ─────────────────────────────────────────────────
    print("Loading training data...")
    df = pd.read_csv(os.path.join(PROCESSED_DIR, TRAIN_IN), parse_dates=["timestamp"])
    print(f"  Rows: {len(df):,}  "
          f"({df['timestamp'].min().date()} → {df['timestamp'].max().date()})")

    # ── 2. Downsample to hourly ───────────────────────────────────────────────
    granger_df = (
        df.set_index("timestamp")
          .resample("1h")
          .mean(numeric_only=True)
          .dropna(subset=["DE_WindSolar_Error", "Unplanned_Flow",
                          "neg_sec_abs_mw", "pos_sec_price"])
          .reset_index()
    )
    print(f"  Downsampled to hourly: {len(granger_df):,} rows")

    # ── 3. Run all three Granger tests ────────────────────────────────────────
    print("\n" + "="*65)
    print("  GRANGER CAUSALITY TESTS")
    print("="*65)
    print(f"  (Hourly data, lags 1–{MAX_LAG}h, significance threshold p < {P_THRESHOLD})")

    gc_link1 = granger_f_test(
        granger_df["Unplanned_Flow"],
        granger_df["DE_WindSolar_Error"], MAX_LAG)
    print_granger_results("Link 1: DE_WindSolar_Error → Unplanned_Flow", gc_link1)

    gc_link2 = granger_f_test(
        granger_df["neg_sec_abs_mw"],
        granger_df["Unplanned_Flow"], MAX_LAG)
    print_granger_results("Link 2: Unplanned_Flow → |Neg aFRR|", gc_link2)

    gc_link3 = granger_f_test(
        granger_df["pos_sec_price"],
        granger_df["DE_WindSolar_Error"], MAX_LAG)
    print_granger_results("Link 3: DE_WindSolar_Error → pos_sec_price  [GO/NO-GO]", gc_link3)

    # ── 4. GO/NO-GO decision ──────────────────────────────────────────────────
    valid_p = [r["p_value"] for r in gc_link3 if not np.isnan(r["p_value"])]
    min_p   = min(valid_p)
    go      = min_p < P_THRESHOLD

    print("\n" + "="*65)
    print(f"  GO/NO-GO DECISION")
    print(f"  Minimum p-value (Link 3, across all lags): {min_p:.6f}")
    print(f"  Threshold: {P_THRESHOLD}")
    print(f"  Decision:  {'✅  PROJECT IS GO' if go else '❌  PROJECT IS NO-GO'}")
    print("="*65)

    # ── 5. Save results ───────────────────────────────────────────────────────
    records = []
    for link_name, link_results in [
        ("DE_WindSolar_Error → Unplanned_Flow", gc_link1),
        ("Unplanned_Flow → neg_sec_abs_mw",     gc_link2),
        ("DE_WindSolar_Error → pos_sec_price",   gc_link3),
    ]:
        for r in link_results:
            records.append({"causal_link": link_name, **r})

    results_df = pd.DataFrame(records)
    out_path   = os.path.join(RESULTS_DIR, RESULTS_OUT)
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print("\n[STEP 02 COMPLETE]")

    return gc_link1, gc_link2, gc_link3, min_p, go


if __name__ == "__main__":
    main()
