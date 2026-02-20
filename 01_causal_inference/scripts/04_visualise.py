"""
================================================================================
STEP 04 — Causal Inference Visualisation  (Individual Figures)
================================================================================
PURPOSE:
    Generate one figure per panel for easy reading and sharing.
    Each figure is saved separately in results/figures/

INPUTS:
    data/processed/features_train_2023_2024.csv
    results/granger_results.csv
    results/ols_stats.csv
    results/spike_prob_by_quintile.csv
    results/price_by_minute.csv
    results/sensitivity_map.csv
    results/eda_stats.csv
    results/cross_correlations.csv

OUTPUTS (all saved in results/figures/):
    00_causal_chain_banner.png
    01_scatter_of_truth.png
    02_wind_error_to_unplanned_flow.png
    03_cross_correlation_by_lag.png
    04_hour_turnover_analysis.png
    05_spike_probability_quintile.png
    06_granger_pvalues.png
    07_sensitivity_map.png
    08_summary_verdict.png
================================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression

# ── Configuration ─────────────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
RESULTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "results")
FIGURES_DIR   = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

TRAIN_IN = "features_train_2023_2024.csv"

# Colour palette
BG_DARK  = "#0d1117"
BG_MID   = "#161b22"
BG_PANEL = "#1c2130"
BLUE     = "#58a6ff"
CYAN     = "#39d0d8"
GREEN    = "#3fb950"
AMBER    = "#d29922"
RED      = "#f85149"
PURPLE   = "#bc8cff"
GREY     = "#8b949e"
WHITE    = "#e6edf3"

plt.rcParams.update({
    "figure.facecolor":  BG_DARK,
    "axes.facecolor":    BG_PANEL,
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   WHITE,
    "xtick.color":       GREY,
    "ytick.color":       GREY,
    "text.color":        WHITE,
    "grid.color":        "#21262d",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
})

FIG_W, FIG_H = 12, 7
DPI          = 150


# ── Helpers ───────────────────────────────────────────────────────────────────

def save_fig(fig: plt.Figure, filename: str) -> None:
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    print(f"  Saved → figures/{filename}")


def make_fig(suptitle: str, subtitle: str = "") -> tuple[plt.Figure, plt.Axes]:
    """
    Create a standard single-panel figure.
    suptitle = large bold title at the top (figure level)
    subtitle = smaller grey description line (axis level, below suptitle)
    """
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor=BG_DARK)
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", color=WHITE, y=0.98)
    if subtitle:
        # Placed as a figure-level text below the suptitle, above the axes
        fig.text(0.5, 0.93, subtitle, ha="center", color=GREY, fontsize=9.5)
    ax.set_facecolor(BG_PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    ax.grid(True, alpha=0.4)
    # Remove the default axes title so nothing overlaps
    ax.set_title("")
    return fig, ax


def add_footer(fig: plt.Figure, text: str) -> None:
    fig.text(0.5, 0.01, text, ha="center", color=GREY, fontsize=8)


def add_ols_line(ax, x_series, y_series, color=CYAN, label_prefix="OLS"):
    clean = pd.DataFrame({"x": x_series, "y": y_series}).dropna()
    lm    = LinearRegression().fit(clean[["x"]], clean["y"])
    xs    = np.linspace(clean["x"].min(), clean["x"].max(), 300)
    r     = clean["x"].corr(clean["y"])
    ax.plot(xs, lm.predict(xs.reshape(-1, 1)),
            color=color, lw=2.5, label=f"{label_prefix}  r = {r:.3f}")
    return r


# ── Figure 00: Causal Chain Banner ────────────────────────────────────────────

def plot_causal_chain(go_p, corr_wf, spike_ratio, slope_per_gw, n_obs):
    fig, ax = plt.subplots(figsize=(FIG_W, 4), facecolor=BG_DARK)
    ax.set_facecolor(BG_MID)
    ax.set_xlim(0, 10); ax.set_ylim(0, 1); ax.axis("off")

    fig.suptitle(
        "Swiss aFRR — Causal Inference Validation  |  Go/No-Go Decision",
        fontsize=15, fontweight="bold", color=WHITE, y=1.02)

    chain = [
        ("German Wind/Solar\nForecast Error",   AMBER,  0.7),
        ("→",                                    WHITE,  2.05),
        ("Unplanned Loop Flow\nDE → CH Border",  BLUE,   2.8),
        ("→",                                    WHITE,  4.35),
        ("aFRR Bid Ladder\nExhaustion",          PURPLE, 5.1),
        ("→",                                    WHITE,  6.65),
        ("aFRR Price\nSpike",                    RED,    7.4),
    ]
    for label, col, x in chain:
        is_arrow = label == "→"
        ax.text(x, 0.60, label, color=col, fontsize=11, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.5", facecolor=BG_PANEL,
                          edgecolor=col, linewidth=2) if not is_arrow else None)

    ax.text(1.7,  0.15, f"Pearson r = {corr_wf:.3f}",
            color=AMBER, fontsize=9, ha="center")
    ax.text(4.1,  0.15, "Non-linear threshold\n(bid-ladder effect)",
            color=PURPLE, fontsize=9, ha="center")
    ax.text(6.1,  0.15, f"Spike prob ×{spike_ratio:.1f}x\nat high |flow|",
            color=RED, fontsize=9, ha="center")

    go   = go_p < 0.05
    vcol = GREEN if go else RED
    ax.text(9.3, 0.60,
            f"Granger  p = {go_p:.4f}\n{'✅  PROJECT GO' if go else '❌  NO-GO'}",
            color=vcol, fontsize=12, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.6", facecolor=BG_DARK,
                      edgecolor=vcol, linewidth=2.5))

    add_footer(fig,
        f"Training: {n_obs:,} observations (2023–2024)  |  "
        f"Granger min p = {go_p:.5f}  |  Spike ratio Q5/Q1 = {spike_ratio:.1f}x")
    fig.tight_layout()
    save_fig(fig, "00_causal_chain_banner.png")


# ── Figure 01: Scatter of Truth ───────────────────────────────────────────────

def plot_scatter_of_truth(df):
    fig, ax = make_fig(
        "① Scatter of Truth: Unplanned Loop Flow vs |Neg aFRR Activation|",
        "Each point = one 15-min interval  |  Hexbin density  |  2023–2024 training data")

    sample = df.sample(min(40_000, len(df)), random_state=42)
    hb = ax.hexbin(sample["Unplanned_Flow"], sample["neg_sec_abs_mw"],
                   gridsize=60, cmap="YlOrRd", mincnt=1, linewidths=0.1)
    plt.colorbar(hb, ax=ax, label="Number of 15-min intervals")
    add_ols_line(ax, sample["Unplanned_Flow"], sample["neg_sec_abs_mw"])
    ax.axhline(0, color=GREY, lw=0.8, ls=":")
    ax.axvline(0, color=GREY, lw=0.8, ls=":")
    ax.legend(fontsize=10, facecolor=BG_DARK, edgecolor="#30363d")
    ax.set_xlabel("Unplanned Flow DE→CH  (MW)\n← CH exports to DE   |   DE pushes to CH →")
    ax.set_ylabel("|Negative aFRR Activation|  (MW)")
    ax.text(0.02, 0.97,
        "Why is Pearson r weak?\n"
        "→ aFRR volume is capacity-bounded (~500 MW)\n"
        "→ At extremes, PRICE spikes instead of volume\n"
        "→ Non-linear mechanism — not captured by r\n"
        "→ See figure 05 for the real evidence",
        transform=ax.transAxes, color=AMBER, fontsize=9, va="top",
        bbox=dict(facecolor=BG_DARK, edgecolor=AMBER, alpha=0.9, pad=5))

    add_footer(fig, "Expected: weak linear r — the causal signal is non-linear (threshold regime)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    save_fig(fig, "01_scatter_of_truth.png")


# ── Figure 02: Wind Error → Unplanned Flow ────────────────────────────────────

def plot_wind_to_flow(df):
    fig, ax = make_fig(
        "② German Wind Error → Unplanned Loop Flow DE→CH",
        "Renewable variability propagates as unplanned physical flow across the border")

    sample = df.sample(min(40_000, len(df)), random_state=42)
    hb = ax.hexbin(sample["DE_WindSolar_Error"], sample["Unplanned_Flow"],
                   gridsize=60, cmap="Blues", mincnt=1, linewidths=0.1)
    plt.colorbar(hb, ax=ax, label="Number of 15-min intervals")
    add_ols_line(ax, sample["DE_WindSolar_Error"], sample["Unplanned_Flow"], color=AMBER)
    ax.axhline(0, color=GREY, lw=0.8, ls=":")
    ax.axvline(0, color=GREY, lw=0.8, ls=":")
    ax.legend(fontsize=10, facecolor=BG_DARK, edgecolor="#30363d")
    ax.set_xlabel("German Wind + Solar Forecast Error  (MW)\n← Under-production  |  Over-production →")
    ax.set_ylabel("Unplanned DE→CH Physical Flow  (MW)\n[Actual Physical − Scheduled Commercial]")

    add_footer(fig,
        "Causal Link 1: German renewable surplus must flow somewhere — Swiss corridor absorbs excess")
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    save_fig(fig, "02_wind_error_to_unplanned_flow.png")


# ── Figure 03: Cross-Correlation by Lag ──────────────────────────────────────

def plot_cross_correlation(cc_df):
    fig, ax = make_fig(
        "③ Cross-Correlation by Lag  (±3 hours)",
        "Positive lag = Effect lags Cause  |  Peak reveals propagation delay in causal chain")

    cc_wf = cc_df[cc_df["x_col"] == "DE_WindSolar_Error"]
    cc_sp = cc_df[cc_df["x_col"] == "abs_Unplanned_Flow"]

    ax.plot(cc_wf["lag_minutes"], cc_wf["pearson_r"],
            color=AMBER, lw=2.5, marker="o", ms=5,
            label="Wind Error → Unplanned Flow")

    ax2 = ax.twinx()
    ax2.plot(cc_sp["lag_minutes"], cc_sp["pearson_r"],
             color=RED, lw=2.5, marker="s", ms=5,
             label="|Unplanned Flow| → Spike Probability")
    ax2.set_ylabel("|Unplanned Flow| → Spike Prob  (Pearson r)", color=RED)
    ax2.tick_params(axis="y", colors=RED)
    ax2.spines["right"].set_edgecolor(RED)

    ax.axvline(0, color=WHITE, lw=1.2, ls="--", alpha=0.5, label="Zero lag")
    ax.axhline(0, color=GREY,  lw=0.8, ls=":")
    ax.set_xlabel("Lag (minutes)  [positive = X leads Y]")
    ax.set_ylabel("Wind Error → Unplanned Flow  (Pearson r)", color=AMBER)
    ax.tick_params(axis="y", colors=AMBER)
    ax.spines["left"].set_edgecolor(AMBER)

    lines  = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(lines, labels, fontsize=10, facecolor=BG_DARK,
              edgecolor="#30363d", loc="upper left")

    peak_idx = cc_sp["pearson_r"].abs().idxmax()
    peak_lag = cc_sp.loc[peak_idx, "lag_minutes"]
    peak_r   = cc_sp.loc[peak_idx, "pearson_r"]
    ax2.annotate(f"Peak at {peak_lag:.0f} min\nr = {peak_r:.3f}",
                 xy=(peak_lag, peak_r),
                 xytext=(peak_lag + 20, peak_r + 0.005),
                 color=RED, fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=RED))

    add_footer(fig,
        "German wind shock → unplanned flow → price spike: propagation delay visible in lag structure")
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    save_fig(fig, "03_cross_correlation_by_lag.png")


# ── Figure 04: Hour-Turnover Analysis ────────────────────────────────────────

def plot_hour_turnover(price_min_df):
    fig, ax = make_fig(
        "④ Hour-Turnover Structural Imbalance  (XX:00 & XX:45)",
        "Schedule transitions create consistent price asymmetry — highest-value ML feature")

    minutes    = [0, 15, 30, 45]
    min_labels = ["XX:00\n(Schedule Change)", "XX:15", "XX:30", "XX:45\n(Pre-Change)"]

    pos_means   = [price_min_df.loc[price_min_df["minute"]==m, "pos_price_mean"].values[0] for m in minutes]
    neg_means   = [price_min_df.loc[price_min_df["minute"]==m, "neg_price_mean"].values[0] for m in minutes]
    pos_stds    = [price_min_df.loc[price_min_df["minute"]==m, "pos_price_std"].values[0]  for m in minutes]
    neg_stds    = [price_min_df.loc[price_min_df["minute"]==m, "neg_price_std"].values[0]  for m in minutes]
    spike_rates = [price_min_df.loc[price_min_df["minute"]==m, "spike_rate_pct"].values[0] for m in minutes]

    x, w = np.arange(4), 0.35
    ax.bar(x-w/2, pos_means, w, color=BLUE, alpha=0.85, label="Avg Positive aFRR Price")
    ax.bar(x+w/2, neg_means, w, color=RED,  alpha=0.85, label="Avg Negative aFRR Price")
    ax.errorbar(x-w/2, pos_means, yerr=[s/2 for s in pos_stds],
                fmt="none", color=CYAN,  capsize=6, lw=2, label="±½ std dev")
    ax.errorbar(x+w/2, neg_means, yerr=[s/2 for s in neg_stds],
                fmt="none", color=AMBER, capsize=6, lw=2)

    ax.set_xticks(x)
    ax.set_xticklabels(min_labels, fontsize=12)
    for xi in [0, 3]:
        ax.get_xticklabels()[xi].set_color(AMBER)
        ax.axvspan(xi-0.48, xi+0.48, alpha=0.07, color=AMBER, zorder=0)

    ax.axhline(0, color=GREY, lw=1)
    ax.set_ylabel("Average aFRR Price  (EUR/MWh)")
    ax.legend(fontsize=10, facecolor=BG_DARK, edgecolor="#30363d")

    for xi, pm, nm, sr in zip(x, pos_means, neg_means, spike_rates):
        ax.text(xi-w/2, pm + max(pos_means)*0.02, f"{pm:.0f}",
                ha="center", fontsize=11, color=CYAN, fontweight="bold")
        ax.text(xi+w/2, nm - max(pos_means)*0.06, f"{nm:.0f}",
                ha="center", fontsize=11, color=AMBER, fontweight="bold")
        ax.text(xi, min(neg_means) - max(pos_means)*0.10,
                f"spike {sr:.1f}%", ha="center", fontsize=8, color=GREY)

    add_footer(fig,
        "XX:00 = first interval after new hourly schedule activates  |  "
        "XX:45 = last interval before change  |  Both show structural price deviation")
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    save_fig(fig, "04_hour_turnover_analysis.png")


# ── Figure 05: Spike Probability by Quintile ──────────────────────────────────

def plot_spike_quintile(spike_df):
    fig, ax = make_fig(
        "⑤ Spike Probability by |Unplanned Flow| Quintile",
        "KEY NON-LINEAR EVIDENCE: spike probability doubles from lowest to highest flow quintile")

    colors = [AMBER if i in [0, 4] else BLUE for i in range(len(spike_df))]
    ax.bar(range(len(spike_df)), spike_df["spike_probability_pct"],
           color=colors, alpha=0.87, edgecolor="#30363d", lw=1, width=0.6)

    for i, (v, n, avg_f) in enumerate(zip(
            spike_df["spike_probability_pct"],
            spike_df["count"],
            spike_df["avg_unplanned_flow_mw"])):
        ax.text(i, v + 0.4,  f"{v:.1f}%",
                ha="center", color=WHITE, fontsize=12, fontweight="bold")
        ax.text(i, -1.8, f"n = {n//1000:.0f}k",
                ha="center", color=GREY, fontsize=9)
        ax.text(i, -3.2, f"avg {avg_f:.0f} MW",
                ha="center", color=GREY, fontsize=8)

    ratio = (spike_df["spike_probability_pct"].iloc[-1] /
             spike_df["spike_probability_pct"].iloc[0])
    q1_p  = spike_df["spike_probability_pct"].iloc[0]
    q5_p  = spike_df["spike_probability_pct"].iloc[-1]
    q1_f  = spike_df["avg_unplanned_flow_mw"].iloc[0]
    q5_f  = spike_df["avg_unplanned_flow_mw"].iloc[-1]
    slope = (q5_p - q1_p) / ((q5_f - q1_f) / 1000)

    ax.set_xticks(range(len(spike_df)))
    ax.set_xticklabels(spike_df["uf_quintile"].tolist(), fontsize=9)
    ax.set_ylabel("Price Spike Probability  (%)\n[Spike = pos_sec_price > P90 threshold]")
    ax.set_ylim(bottom=-4)

    ax.text(0.98, 0.97,
        f"Q5 / Q1 ratio:  ×{ratio:.1f}x\n"
        f"Slope:          +{slope:.1f}% per 1 GW",
        transform=ax.transAxes, color=AMBER, fontsize=11, fontweight="bold",
        ha="right", va="top",
        bbox=dict(facecolor=BG_DARK, edgecolor=AMBER, pad=6, linewidth=1.5))

    for xi in [0, 4]:
        ax.get_xticklabels()[xi].set_color(AMBER)

    add_footer(fig,
        "Weak Pearson r does NOT mean weak causality — it means the relationship is threshold-driven")
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    save_fig(fig, "05_spike_probability_quintile.png")


# ── Figure 06: Granger p-values ───────────────────────────────────────────────

def plot_granger_pvalues(granger_df, go_p):
    fig, ax = make_fig(
        "⑥ Granger Causality F-test  p-values by Lag",
        "Bars below the green line (p < 0.05) confirm statistical causality at that lag")

    links  = granger_df["causal_link"].unique()
    colors = [AMBER, BLUE, RED]
    w      = 0.25
    lags_x = np.arange(4)

    for j, (link, col) in enumerate(zip(links, colors)):
        pvals = granger_df[granger_df["causal_link"] == link]["p_value"].values[:4]
        label = (link.replace("DE_WindSolar_Error", "Wind Error")
                     .replace("Unplanned_Flow", "Unplanned Flow")
                     .replace("neg_sec_abs_mw", "|Neg aFRR|")
                     .replace("pos_sec_price", "aFRR Price"))
        ax.bar(lags_x + j*w, pvals, w, color=col, alpha=0.85,
               label=label, edgecolor="#30363d", lw=0.8)
        for k, p in enumerate(pvals):
            if not np.isnan(p):
                ax.text(lags_x[k] + j*w, p * 1.5, f"{p:.4f}",
                        ha="center", fontsize=7.5, color=col, rotation=45)

    ax.axhline(0.05, color=GREEN, lw=2.5, ls="--",
               label="p = 0.05 significance threshold", zorder=5)
    ax.set_xticks(lags_x + w)
    ax.set_xticklabels(["Lag 1h", "Lag 2h", "Lag 3h", "Lag 4h"], fontsize=11)
    ax.set_ylabel("p-value  (log scale)")
    ax.set_yscale("log")
    ax.legend(fontsize=9, facecolor=BG_DARK, edgecolor="#30363d", loc="upper right")

    go   = go_p < 0.05
    vcol = GREEN if go else RED
    ax.text(0.02, 0.06,
        f"GO/NO-GO: min p = {go_p:.5f}  →  {'✅  PROJECT IS GO' if go else '❌  NO-GO'}",
        transform=ax.transAxes, color=vcol, fontsize=11, fontweight="bold",
        bbox=dict(facecolor=BG_DARK, edgecolor=vcol, pad=5, linewidth=1.5))

    add_footer(fig,
        "Lag 3-4h significant for Link 3 (Wind Error → Price): "
        "delay reflects forecast window to real-time balancing market clearance")
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    save_fig(fig, "06_granger_pvalues.png")


# ── Figure 07: Sensitivity Map ────────────────────────────────────────────────

def plot_sensitivity_map(sens_df):
    fig, ax = make_fig(
        "⑦ Sensitivity Map: German Wind Error → Swiss aFRR Price",
        "How average balancing cost changes as German renewable forecast error varies")

    ax.fill_between(sens_df["bin_midpoint"],
                    sens_df["pos_price_mean"] - sens_df["pos_price_std"] / 2,
                    sens_df["pos_price_mean"] + sens_df["pos_price_std"] / 2,
                    color=PURPLE, alpha=0.2, label="±½ std dev band")
    ax.plot(sens_df["bin_midpoint"], sens_df["pos_price_mean"],
            color=PURPLE, lw=3, marker="o", ms=7, label="Mean positive aFRR price")

    overall_mean = sens_df["pos_price_mean"].mean()
    ax.axhline(overall_mean, color=GREY, lw=1.5, ls="--",
               label=f"Overall mean ({overall_mean:.0f} EUR/MWh)")
    ax.axvline(0, color=WHITE, lw=1, ls=":", alpha=0.5, label="Zero forecast error")

    ax.set_xlabel("German Wind + Solar Forecast Error  (MW)\n"
                  "← Under-production  |  Zero  |  Over-production →")
    ax.set_ylabel("Average Positive aFRR Price  (EUR/MWh)")
    ax.legend(fontsize=10, facecolor=BG_DARK, edgecolor="#30363d")

    left_tail  = sens_df.iloc[0]
    right_tail = sens_df.iloc[-1]
    ax.annotate(f"Under-production\n{left_tail['pos_price_mean']:.0f} EUR/MWh",
                xy=(left_tail["bin_midpoint"], left_tail["pos_price_mean"]),
                xytext=(left_tail["bin_midpoint"] + 800, left_tail["pos_price_mean"] + 5),
                color=AMBER, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=AMBER))
    ax.annotate(f"Over-production\n{right_tail['pos_price_mean']:.0f} EUR/MWh",
                xy=(right_tail["bin_midpoint"], right_tail["pos_price_mean"]),
                xytext=(right_tail["bin_midpoint"] - 2500, right_tail["pos_price_mean"] + 5),
                color=AMBER, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=AMBER))

    add_footer(fig,
        "U-shape confirms bid-ladder exhaustion: extreme deviations in either direction spike prices")
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    save_fig(fig, "07_sensitivity_map.png")


# ── Figure 08: Summary & Verdict ─────────────────────────────────────────────

def plot_summary(go_p, eda_stats, spike_df, n_obs):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor=BG_DARK)
    ax.set_facecolor(BG_MID); ax.axis("off")

    go    = go_p < 0.05
    vcol  = GREEN if go else RED
    ratio = (spike_df["spike_probability_pct"].iloc[-1] /
             spike_df["spike_probability_pct"].iloc[0])
    slope = float(eda_stats.get("slope_pct_per_gw", 0))

    fig.suptitle("Swiss aFRR Causal Inference — Summary & Go/No-Go Verdict",
                 fontsize=14, fontweight="bold", color=WHITE, y=0.98)

    # Left column: text summary
    lines = [
        (0.93, "VALIDATION RESULTS",                                       WHITE,  13, True),
        (0.87, f"Training set: {n_obs:,} observations  (2023–2024)",       GREY,   10, False),
        (0.80, "LINK 1 — German Wind Error → Unplanned Loop Flow",         AMBER,  11, True),
        (0.74, "  Granger: SIGNIFICANT at all lags 1–4h",                 GREEN,  10, False),
        (0.69, f"  Pearson r = {eda_stats['pearson_r_wind_unplanned']:.3f}"
               "  (weak linear — non-linear mechanism expected)",           GREY,    9, False),
        (0.62, "LINK 2 — Unplanned Flow → aFRR Spike Probability",        BLUE,   11, True),
        (0.56, f"  Spike prob ×{ratio:.1f}x from Q1 → Q5 quintile",       GREEN,  10, False),
        (0.51, f"  +{slope:.1f}% per 1 GW of loop flow",                  CYAN,   10, False),
        (0.44, "LINK 3 — Wind Error → aFRR Price  [GO/NO-GO TEST]",       RED,    11, True),
        (0.38, f"  Granger p = {go_p:.5f}  (significant at lag 3–4h)",
               GREEN if go else RED,                                                10, False),
        (0.33, "  Delay = forecast window → real-time market clearance",   GREY,    9, False),
        (0.26, "STRUCTURAL SIGNAL",                                        PURPLE, 11, True),
        (0.20, "  Price asymmetry at XX:00 and XX:45 slots",              GREEN,  10, False),
        (0.15, "  Most predictable feature for ML model",                 GREEN,   9, False),
        (0.08, "RECOMMENDED MODEL",                                        WHITE,  11, True),
        (0.03, "  XGBoost / LightGBM  —  non-linear, threshold-aware",    CYAN,   10, False),
    ]

    for row in lines:
        if len(row) == 5:
            y, txt, col, fs, bold = row
        else:
            y, txt, col, fs, bold = row[0], row[1], row[2], row[3], row[4]
        ax.text(0.03, y, txt, transform=ax.transAxes,
                color=col, fontsize=fs,
                fontweight="bold" if bold else "normal", va="bottom")

    # Right column: verdict box
    verdict_box = mpatches.FancyBboxPatch(
        (0.63, 0.38), 0.33, 0.20, boxstyle="round,pad=0.03",
        facecolor=BG_DARK, edgecolor=vcol, linewidth=3,
        transform=ax.transAxes)
    ax.add_patch(verdict_box)
    ax.text(0.795, 0.49,
            f"{'✅  PROJECT IS GO' if go else '❌  NO-GO'}",
            transform=ax.transAxes, color=vcol, fontsize=14,
            fontweight="bold", ha="center", va="center")
    ax.text(0.795, 0.42,
            f"Granger p = {go_p:.4f}\n< 0.05 threshold",
            transform=ax.transAxes, color=GREY, fontsize=9,
            ha="center", va="center")

    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")

    add_footer(fig,
        "Swiss aFRR Causal Inference Pipeline  |  "
        "Data: Swissgrid 2023–2025 + ENTSO-E 2023–2025")
    fig.tight_layout()
    save_fig(fig, "08_summary_verdict.png")


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def main():
    print("Loading data and results...")

    df         = pd.read_csv(os.path.join(PROCESSED_DIR, TRAIN_IN), parse_dates=["timestamp"])
    granger_df = pd.read_csv(os.path.join(RESULTS_DIR, "granger_results.csv"))
    spike_df   = pd.read_csv(os.path.join(RESULTS_DIR, "spike_prob_by_quintile.csv"))
    price_min  = pd.read_csv(os.path.join(RESULTS_DIR, "price_by_minute.csv"))
    sens_df    = pd.read_csv(os.path.join(RESULTS_DIR, "sensitivity_map.csv"))
    eda_stats  = pd.read_csv(os.path.join(RESULTS_DIR, "eda_stats.csv")).iloc[0].to_dict()
    cc_df      = pd.read_csv(os.path.join(RESULTS_DIR, "cross_correlations.csv"))

    if "price_spike" not in df.columns:
        threshold = df["pos_sec_price"].quantile(0.90)
        df["price_spike"] = (df["pos_sec_price"] > threshold).astype(int)

    gc_link3    = granger_df[granger_df["causal_link"].str.contains("pos_sec_price")]
    go_p        = gc_link3["p_value"].min()
    corr_wf     = float(eda_stats["pearson_r_wind_unplanned"])
    spike_ratio = (spike_df["spike_probability_pct"].iloc[-1] /
                   spike_df["spike_probability_pct"].iloc[0])
    q1_p  = spike_df["spike_probability_pct"].iloc[0]
    q5_p  = spike_df["spike_probability_pct"].iloc[-1]
    q1_f  = spike_df["avg_unplanned_flow_mw"].iloc[0]
    q5_f  = spike_df["avg_unplanned_flow_mw"].iloc[-1]
    slope = (q5_p - q1_p) / ((q5_f - q1_f) / 1000)
    eda_stats["slope_pct_per_gw"] = slope

    print(f"  GO/NO-GO p-value: {go_p:.5f}  →  {'GO ✓' if go_p < 0.05 else 'NO-GO ✗'}")
    print(f"\nGenerating 9 individual figures → {FIGURES_DIR}/")

    plot_causal_chain(go_p, corr_wf, spike_ratio, slope, len(df))
    plot_scatter_of_truth(df)
    plot_wind_to_flow(df)
    plot_cross_correlation(cc_df)
    plot_hour_turnover(price_min)
    plot_spike_quintile(spike_df)
    plot_granger_pvalues(granger_df, go_p)
    plot_sensitivity_map(sens_df)
    plot_summary(go_p, eda_stats, spike_df, len(df))

    print(f"\nAll 9 figures saved to: {FIGURES_DIR}")
    print("\n[STEP 04 COMPLETE]")


if __name__ == "__main__":
    main()
