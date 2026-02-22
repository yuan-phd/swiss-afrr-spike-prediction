"""
================================================================================
STEP 03 — XAI Explainability Suite (SHAP)
================================================================================
PURPOSE:
    Full SHAP explainability analysis with four critical validation checks:

    CHECK 1 — Threshold Validation
        SHAP dependence plot for abs_Unplanned_Flow.
        Model is valid ONLY if we see a non-linear step-up at 1250 MW.
        This proves the model learned the physical bid-ladder mechanism.

    CHECK 2 — Interaction Audit
        SHAP interaction values between Unplanned_Flow and Sched_DE_CH.
        Proves the model understands "Market Scheduling" amplifies
        "Physical Loop Flows."

    CHECK 3 — Bias Check
        SHAP values for temporal features (month, day_of_week, hour).
        If these dominate, model is overfitting to 2025 price level rise
        rather than learning physical causality.

    CHECK 4 — Global Feature Importance
        Beeswarm summary plot — direction and magnitude of all features.
        Confirms Hammer features dominate over Structural features.

INPUTS:
    data/processed/features_train_2023_2024.csv  (SHAP background)
    data/processed/features_val_2025.csv          (SHAP explanation target)
    02_model_training/models/xgboost_spike_classifier.json
    02_model_training/models/model_meta.json

OUTPUTS:
    MLflow artifacts:
      → shap_summary_beeswarm.png       (global importance + direction)
      → shap_dependence_abs_flow.png    (1250 MW threshold check)
      → shap_dependence_da_spread.png   (economic incentive check)
      → shap_interaction_flow_sched.png (amplification audit)
      → shap_bias_check.png             (temporal feature audit)
      → shap_local_example.png          (single prediction waterfall)
      → shap_category_importance.png    (hammer vs anvil vs incentive)
================================================================================
"""

import os
import sys
import json
import yaml
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import xgboost as xgb
import shap

# ── Configuration ──────────────────────────────────────────────────────────────
ROOT        = os.path.join(os.path.dirname(__file__), "..", "..")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

TRAIN_PATH  = os.path.join(ROOT, cfg["paths"]["train_csv"])
VAL_PATH    = os.path.join(ROOT, cfg["paths"]["val_csv"])
MODEL_DIR   = os.path.join(ROOT, cfg["paths"]["model_dir"])
FIGURES_DIR = os.path.join(ROOT, cfg["paths"]["figures_dir"])
os.makedirs(FIGURES_DIR, exist_ok=True)

TARGET       = cfg["target"]
ALL_FEATURES = (cfg["features"]["hammer"] + cfg["features"]["anvil"] +
                cfg["features"]["incentive"] + cfg["features"]["autoregressive"] +
                cfg["features"]["structural"])
N_SHAP       = cfg["xai"]["n_shap_samples"]

# Colour palette
BG_DARK  = "#0d1117"
BG_PANEL = "#1c2130"
BLUE     = "#58a6ff"
GREEN    = "#3fb950"
AMBER    = "#d29922"
RED      = "#f85149"
PURPLE   = "#bc8cff"
WHITE    = "#e6edf3"
GREY     = "#8b949e"

plt.rcParams.update({
    "figure.facecolor": BG_DARK,
    "axes.facecolor":   BG_PANEL,
    "text.color":       WHITE,
    "axes.labelcolor":  WHITE,
    "xtick.color":      GREY,
    "ytick.color":      GREY,
    "axes.edgecolor":   "#30363d",
    "grid.color":       "#21262d",
    "grid.alpha":       0.4,
    "font.size":        11,
})

THRESHOLD_1250MW = 1250  # Physical bid-ladder bottleneck from causal analysis


def style_ax(ax):
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    ax.grid(True, alpha=0.3)


# ── SHAP Computation ───────────────────────────────────────────────────────────

def add_regime_invariant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute price_delta and price_vs_threshold if not already present."""
    if "price_delta_lag1" not in df.columns:
        rolling_median = df["pos_sec_price"].rolling(96, min_periods=48).median()
        rolling_std    = df["pos_sec_price"].rolling(96, min_periods=48).std()
        rolling_std    = rolling_std.clip(lower=1.0)
        lag1  = df["pos_sec_price"].shift(1)
        lag4  = df["pos_sec_price"].shift(4)
        lag96 = df["pos_sec_price"].shift(96)
        df["price_delta_lag1"]  = (lag1  - rolling_median) / rolling_std
        df["price_delta_lag4"]  = (lag4  - rolling_median) / rolling_std
        df["price_delta_lag96"] = (lag96 - rolling_median) / rolling_std
        threshold = df["rolling_p90_threshold"].clip(lower=50)
        df["price_vs_threshold_lag1"] = lag1  / threshold
        df["price_vs_threshold_lag4"] = lag4  / threshold
    return df


def compute_shap(model, X_background, X_explain):
    """
    Use TreeExplainer — fastest and exact for XGBoost.
    X_background: training data sample (provides baseline distribution)
    X_explain: data to explain (validation set sample)
    """
    print(f"  Computing SHAP values for {len(X_explain):,} samples...")
    explainer   = shap.TreeExplainer(model, X_background)
    shap_values = explainer.shap_values(X_explain)
    print(f"  SHAP values shape: {shap_values.shape}")
    return explainer, shap_values


# ── CHECK 1: Threshold Validation ─────────────────────────────────────────────

def plot_shap_dependence_threshold(X_explain, shap_values,
                                   feature: str) -> plt.Figure:
    """
    SHAP dependence plot for abs_Unplanned_Flow.
    Critical validation: model is only valid if SHAP values show
    a non-linear step-up at the 1250 MW physical bottleneck.
    If the relationship is linear, the model hasn't learned the
    bid-ladder mechanism — it's just fitting noise.
    """
    feat_idx = ALL_FEATURES.index(feature)
    feat_vals = X_explain[feature].values
    shap_vals = shap_values[:, feat_idx]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG_DARK)
    fig.suptitle(f"CHECK 1: Threshold Validation — {feature}\n"
                 f"Model valid only if step-up at {THRESHOLD_1250MW} MW",
                 fontsize=13, fontweight="bold", color=WHITE)

    # Left: raw scatter
    sc = axes[0].scatter(feat_vals, shap_vals,
                         c=shap_vals, cmap="RdYlGn", alpha=0.3, s=8)
    axes[0].axvline(THRESHOLD_1250MW, color=AMBER, lw=2.5, ls="--",
                    label=f"1250 MW bottleneck")
    axes[0].axhline(0, color=GREY, lw=1, ls=":")
    axes[0].set_xlabel(feature)
    axes[0].set_ylabel(f"SHAP value (impact on spike probability)")
    axes[0].set_title("Raw SHAP Scatter", color=WHITE)
    axes[0].legend(fontsize=9, facecolor=BG_DARK, edgecolor="#30363d",
                   labelcolor=WHITE)
    plt.colorbar(sc, ax=axes[0], label="SHAP value")
    style_ax(axes[0])

    # Right: binned mean SHAP — clearer view of the threshold effect
    bins = np.linspace(feat_vals.min(), feat_vals.max(), 25)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means   = []
    bin_stds    = []
    for i in range(len(bins) - 1):
        mask = (feat_vals >= bins[i]) & (feat_vals < bins[i+1])
        if mask.sum() > 0:
            bin_means.append(shap_vals[mask].mean())
            bin_stds.append(shap_vals[mask].std())
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)

    bin_means = np.array(bin_means)
    bin_stds  = np.array(bin_stds)

    axes[1].plot(bin_centers, bin_means, color=BLUE, lw=2.5, marker="o", ms=6)
    axes[1].fill_between(bin_centers,
                         bin_means - bin_stds/2,
                         bin_means + bin_stds/2,
                         color=BLUE, alpha=0.2, label="±½ std")
    axes[1].axvline(THRESHOLD_1250MW, color=AMBER, lw=2.5, ls="--",
                    label=f"1250 MW bottleneck")
    axes[1].axhline(0, color=GREY, lw=1, ls=":")

    # Check if step-up exists
    below_1250 = bin_means[bin_centers < THRESHOLD_1250MW]
    above_1250 = bin_means[bin_centers >= THRESHOLD_1250MW]
    if len(below_1250) > 0 and len(above_1250) > 0:
        step_up = np.nanmean(above_1250) - np.nanmean(below_1250)
        verdict = "✅ STEP-UP CONFIRMED" if step_up > 0.01 else "⚠️  STEP-UP WEAK"
        color   = GREEN if step_up > 0.01 else AMBER
        axes[1].text(0.98, 0.97, f"{verdict}\nStep: {step_up:+.4f}",
                     transform=axes[1].transAxes, color=color,
                     fontsize=11, fontweight="bold", ha="right", va="top",
                     bbox=dict(facecolor=BG_DARK, edgecolor=color, pad=5))

    axes[1].set_xlabel(feature)
    axes[1].set_ylabel("Mean SHAP value (binned)")
    axes[1].set_title("Binned Mean SHAP — Non-linearity Check", color=WHITE)
    axes[1].legend(fontsize=9, facecolor=BG_DARK, edgecolor="#30363d",
                   labelcolor=WHITE)
    style_ax(axes[1])

    fig.tight_layout()
    return fig


# ── CHECK 2: Interaction Audit ─────────────────────────────────────────────────

def plot_shap_interaction(X_explain, shap_values,
                          feat1: str, feat2: str) -> plt.Figure:
    """
    SHAP dependence coloured by interaction feature.
    Proves that Sched_DE_CH amplifies the effect of Unplanned_Flow.
    High Sched_DE_CH + high Unplanned_Flow should show highest SHAP values.
    """
    idx1 = ALL_FEATURES.index(feat1)
    idx2 = ALL_FEATURES.index(feat2)

    feat1_vals = X_explain[feat1].values
    feat2_vals = X_explain[feat2].values
    shap_vals  = shap_values[:, idx1]

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG_DARK)
    sc = ax.scatter(feat1_vals, shap_vals,
                    c=feat2_vals, cmap="RdYlBu_r",
                    alpha=0.4, s=10)
    plt.colorbar(sc, ax=ax, label=feat2)

    ax.axvline(THRESHOLD_1250MW, color=AMBER, lw=2, ls="--",
               label="1250 MW bottleneck")
    ax.axhline(0, color=GREY, lw=1, ls=":")
    ax.set_xlabel(feat1)
    ax.set_ylabel(f"SHAP value of {feat1}")
    ax.set_title(f"CHECK 2: Interaction Audit\n"
                 f"{feat1} SHAP values, coloured by {feat2}\n"
                 f"Warm colours = high {feat2} (amplifies physical flow impact)",
                 fontsize=12, fontweight="bold", color=WHITE)
    ax.legend(fontsize=9, facecolor=BG_DARK, edgecolor="#30363d",
              labelcolor=WHITE)
    style_ax(ax)
    fig.tight_layout()
    return fig


# ── CHECK 3: Bias Check ────────────────────────────────────────────────────────

def plot_shap_bias_check(X_explain, shap_values) -> plt.Figure:
    """
    Temporal feature SHAP audit.
    If month, day_of_week, or hour dominate the SHAP values,
    the model has learned the 2025 price level rise (overfitting)
    rather than the physical causal mechanism.
    Verdict: temporal features should have LOW mean |SHAP| relative
    to Hammer features.
    """
    bias_features   = cfg["xai"]["bias_check_features"]
    hammer_features = cfg["features"]["hammer"]

    def mean_abs_shap(feat):
        idx = ALL_FEATURES.index(feat)
        return np.abs(shap_values[:, idx]).mean()

    bias_shap   = {f: mean_abs_shap(f) for f in bias_features
                   if f in ALL_FEATURES}
    hammer_shap = {f: mean_abs_shap(f) for f in hammer_features
                   if f in ALL_FEATURES}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG_DARK)
    fig.suptitle("CHECK 3: Temporal Bias Audit\n"
                 "Physical features must dominate temporal features",
                 fontsize=13, fontweight="bold", color=WHITE)

    # Left: temporal features
    sorted_bias = dict(sorted(bias_shap.items(), key=lambda x: x[1]))
    axes[0].barh(list(sorted_bias.keys()), list(sorted_bias.values()),
                 color=AMBER, alpha=0.85, edgecolor="#30363d")
    axes[0].set_xlabel("Mean |SHAP value|")
    axes[0].set_title("Temporal Features (should be LOW)", color=AMBER)
    style_ax(axes[0])

    # Right: hammer features
    sorted_hammer = dict(sorted(hammer_shap.items(), key=lambda x: x[1]))
    axes[1].barh(list(sorted_hammer.keys()), list(sorted_hammer.values()),
                 color=GREEN, alpha=0.85, edgecolor="#30363d")
    axes[1].set_xlabel("Mean |SHAP value|")
    axes[1].set_title("Physical Features / Hammer (should be HIGH)", color=GREEN)
    style_ax(axes[1])

    # Verdict
    max_bias   = max(bias_shap.values())   if bias_shap   else 0
    mean_hammer= np.mean(list(hammer_shap.values())) if hammer_shap else 0
    ratio      = mean_hammer / (max_bias + 1e-8)
    verdict    = "✅ NO BIAS" if ratio > 2 else "⚠️  POSSIBLE BIAS"
    color      = GREEN if ratio > 2 else AMBER

    fig.text(0.5, 0.01,
             f"{verdict}  |  "
             f"Hammer mean SHAP: {mean_hammer:.4f}  |  "
             f"Max temporal SHAP: {max_bias:.4f}  |  "
             f"Ratio: {ratio:.1f}x  (>2x = safe)",
             ha="center", color=color, fontsize=11, fontweight="bold")

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig, ratio


# ── CHECK 4: Global Beeswarm ───────────────────────────────────────────────────

def plot_shap_beeswarm(X_explain, shap_values) -> plt.Figure:
    """
    SHAP summary beeswarm plot — shows direction AND magnitude for all features.
    Red = high feature value increases spike probability.
    Blue = high feature value decreases spike probability.
    Note: shap 0.45+ does not accept 'ax' argument for dot plots —
    we let SHAP create its own figure then apply styling to gcf/gca.
    """
    plt.close('all')

    with plt.rc_context({
        'axes.facecolor':   BG_PANEL,
        'figure.facecolor': BG_DARK,
        'text.color':       WHITE,
        'axes.labelcolor':  WHITE,
        'xtick.color':      GREY,
        'ytick.color':      WHITE,
    }):
        shap.summary_plot(
            shap_values,
            X_explain,
            feature_names=ALL_FEATURES,
            plot_type="dot",
            show=False,
            max_display=len(ALL_FEATURES))

        fig = plt.gcf()
        ax  = plt.gca()

        fig.set_facecolor(BG_DARK)
        ax.set_facecolor(BG_PANEL)
        ax.set_title(
            "SHAP Summary: Global Feature Importance & Direction\n"
            "Red = High value increases spike prob | Blue = Decreases",
            fontsize=12, fontweight="bold", color=WHITE, pad=20)
        ax.set_xlabel("SHAP Value (impact on model output)",
                      color=WHITE, fontsize=10)
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")

        plt.tight_layout()

    return fig


# ── Category Importance ────────────────────────────────────────────────────────

def plot_category_importance(shap_values) -> plt.Figure:
    """
    Aggregate SHAP importance by feature category (Hammer / Anvil / Incentive).
    Validates that the physical causal architecture dominates.
    """
    categories = {
        "HAMMER\n(Physical Causal)":   cfg["features"]["hammer"],
        "ANVIL\n(Market Context)":     cfg["features"]["anvil"],
        "INCENTIVE\n(Economic Signal)":cfg["features"]["incentive"],
        "AUTOREGRESSIVE\n(Price Memory)":cfg["features"]["autoregressive"],
        "STRUCTURAL\n(Time Patterns)": cfg["features"]["structural"],
    }
    category_colors = [AMBER, BLUE, GREEN, "#bc8cff", GREY]

    cat_importance = {}
    for cat, cols in categories.items():
        indices = [ALL_FEATURES.index(c) for c in cols if c in ALL_FEATURES]
        if indices:
            cat_importance[cat] = np.abs(shap_values[:, indices]).mean()

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG_DARK)
    ax.set_facecolor(BG_PANEL)

    bars = ax.bar(range(len(cat_importance)),
                  list(cat_importance.values()),
                  color=category_colors[:len(cat_importance)],
                  alpha=0.85, edgecolor="#30363d", width=0.6)

    for bar, val in zip(bars, cat_importance.values()):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.0005,
                f"{val:.4f}",
                ha="center", color=WHITE, fontsize=11, fontweight="bold")

    ax.set_xticks(range(len(cat_importance)))
    ax.set_xticklabels(list(cat_importance.keys()), fontsize=10)
    ax.set_ylabel("Mean |SHAP value|")
    ax.set_title("Feature Category Importance\n"
                 "HAMMER (physical) should dominate",
                 fontsize=13, fontweight="bold", color=WHITE)
    style_ax(ax)

    # Verdict
    hammer_imp = cat_importance.get(
        "HAMMER\n(Physical Causal)", 0)
    max_other  = max([v for k, v in cat_importance.items()
                      if "HAMMER" not in k], default=0)
    verdict = ("✅ Physical causal features dominate"
               if hammer_imp > max_other
               else "⚠️  Non-physical features dominate — investigate")
    color = GREEN if hammer_imp > max_other else AMBER
    ax.text(0.98, 0.97, verdict,
            transform=ax.transAxes, color=color,
            fontsize=11, fontweight="bold", ha="right", va="top",
            bbox=dict(facecolor=BG_DARK, edgecolor=color, pad=5))

    fig.tight_layout()
    return fig


# ── Local Waterfall ────────────────────────────────────────────────────────────

def plot_local_waterfall(explainer, X_explain, y_prob,
                         y_true) -> plt.Figure:
    """
    Waterfall plot for a single high-confidence spike prediction.
    Shows exactly which features pushed the prediction above threshold.
    """
    # Find a true positive spike with high confidence
    spike_idx = np.where((y_true.values == 1) & (y_prob > 0.7))[0]
    if len(spike_idx) == 0:
        spike_idx = np.where(y_true.values == 1)[0]
    idx = spike_idx[0]

    shap_exp = explainer(X_explain.iloc[[idx]])

    fig, ax = plt.subplots(figsize=(12, 8), facecolor=BG_DARK)
    ax.set_facecolor(BG_PANEL)

    shap.waterfall_plot(shap_exp[0], show=False, max_display=15)

    ax.set_title(f"Local Explanation — Single Spike Prediction\n"
                 f"Sample #{idx}  |  Predicted P(spike)={y_prob[idx]:.3f}  |  "
                 f"Actual={'SPIKE' if y_true.values[idx]==1 else 'NO SPIKE'}",
                 fontsize=11, fontweight="bold", color=WHITE)
    ax.set_facecolor(BG_PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")

    fig.tight_layout()
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STEP 03 — XAI Explainability Suite (SHAP)")
    print("=" * 60)

    # ── Load model ─────────────────────────────────────────────────────────────
    print("\n[1] Loading model...")
    model_path = os.path.join(MODEL_DIR, "xgboost_spike_classifier.json")
    meta_path  = os.path.join(MODEL_DIR, "model_meta.json")

    if not os.path.exists(model_path):
        print("❌ Model not found — run 01_train_xgboost.py first")
        sys.exit(1)

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    with open(meta_path) as f:
        meta = json.load(f)
    optimal_threshold = meta.get("optimal_threshold", 0.5)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\n[2] Loading data...")
    train = pd.read_csv(TRAIN_PATH, parse_dates=["timestamp"])
    val   = pd.read_csv(VAL_PATH,   parse_dates=["timestamp"])

    for df in [train, val]:
        if "rolling_p90_threshold" in df.columns:
            df["rolling_p90_threshold"] = df["rolling_p90_threshold"].clip(lower=50)
        add_regime_invariant_features(df)

    X_train = train[ALL_FEATURES].fillna(train[ALL_FEATURES].median())
    X_val   = val[ALL_FEATURES].fillna(val[ALL_FEATURES].median())
    y_val   = val[TARGET]

    # Sample for SHAP (TreeExplainer is fast but val set is large)
    np.random.seed(42)
    background_idx = np.random.choice(len(X_train), min(N_SHAP, len(X_train)),
                                      replace=False)
    explain_idx    = np.random.choice(len(X_val), min(N_SHAP, len(X_val)),
                                      replace=False)

    X_background = X_train.iloc[background_idx].reset_index(drop=True)
    X_explain    = X_val.iloc[explain_idx].reset_index(drop=True)
    y_explain    = y_val.iloc[explain_idx].reset_index(drop=True)

    print(f"  Background samples: {len(X_background):,}")
    print(f"  Explain samples:    {len(X_explain):,}")

    # ── Compute SHAP values ────────────────────────────────────────────────────
    print("\n[3] Computing SHAP values...")
    explainer, shap_values = compute_shap(model, X_background, X_explain)

    y_prob = model.predict_proba(X_explain)[:, 1]

    # ── MLflow logging ─────────────────────────────────────────────────────────
    tracking_uri = os.path.join(ROOT, cfg["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(f"file://{os.path.abspath(tracking_uri)}")
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="xai_shap_analysis",
                          run_id=meta["run_id"]):

        print("\n[4] Generating XAI plots...")

        # CHECK 1 — Threshold validation
        print("  CHECK 1: Threshold validation (1250 MW)...")
        fig1 = plot_shap_dependence_threshold(
            X_explain, shap_values, "abs_Unplanned_Flow")
        path1 = os.path.join(FIGURES_DIR, "shap_dependence_abs_flow.png")
        fig1.savefig(path1, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
        mlflow.log_artifact(path1)
        plt.close(fig1)

        # Also plot DA spread dependence
        if "DA_Price_Spread_DE_CH" in ALL_FEATURES:
            fig1b = plot_shap_dependence_threshold(
                X_explain, shap_values, "DA_Price_Spread_DE_CH")
            path1b = os.path.join(FIGURES_DIR, "shap_dependence_da_spread.png")
            fig1b.savefig(path1b, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
            mlflow.log_artifact(path1b)
            plt.close(fig1b)

        # CHECK 2 — Interaction audit
        print("  CHECK 2: Interaction audit (Flow × Sched)...")
        fig2 = plot_shap_interaction(
            X_explain, shap_values, "Unplanned_Flow", "Sched_DE_CH")
        path2 = os.path.join(FIGURES_DIR, "shap_interaction_flow_sched.png")
        fig2.savefig(path2, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
        mlflow.log_artifact(path2)
        plt.close(fig2)

        # CHECK 3 — Bias check
        print("  CHECK 3: Temporal bias audit...")
        fig3, bias_ratio = plot_shap_bias_check(X_explain, shap_values)
        path3 = os.path.join(FIGURES_DIR, "shap_bias_check.png")
        fig3.savefig(path3, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
        mlflow.log_artifact(path3)
        mlflow.log_metric("shap_bias_ratio_hammer_vs_temporal", bias_ratio)
        plt.close(fig3)

        verdict = "PASS" if bias_ratio > 2 else "INVESTIGATE"
        print(f"    Bias ratio (hammer/temporal): {bias_ratio:.1f}x — {verdict}")

        # CHECK 4 — Global beeswarm
        print("  CHECK 4: Global beeswarm summary...")
        fig4 = plot_shap_beeswarm(X_explain, shap_values)
        path4 = os.path.join(FIGURES_DIR, "shap_summary_beeswarm.png")
        fig4.savefig(path4, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
        mlflow.log_artifact(path4)
        plt.close(fig4)

        # Category importance
        print("  Category importance plot...")
        fig5 = plot_category_importance(shap_values)
        path5 = os.path.join(FIGURES_DIR, "shap_category_importance.png")
        fig5.savefig(path5, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
        mlflow.log_artifact(path5)
        plt.close(fig5)

        # Local waterfall — single prediction
        print("  Local waterfall (single prediction)...")
        try:
            fig6 = plot_local_waterfall(explainer, X_explain, y_prob, y_explain)
            path6 = os.path.join(FIGURES_DIR, "shap_local_example.png")
            fig6.savefig(path6, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
            mlflow.log_artifact(path6)
            plt.close(fig6)
        except Exception as e:
            print(f"    ⚠️  Waterfall plot skipped: {e}")

        # Log top SHAP features as metrics
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            "feature":        ALL_FEATURES,
            "mean_abs_shap":  mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False)

        print("\n  Top 10 features by mean |SHAP|:")
        for _, row in shap_df.head(10).iterrows():
            print(f"    {row['feature']:<35} {row['mean_abs_shap']:.5f}")
            mlflow.log_metric(f"shap_{row['feature']}", row["mean_abs_shap"])

        # Save SHAP importance CSV
        shap_path = os.path.join(
            ROOT, cfg["paths"]["results_dir"], "shap_importance.csv")
        shap_df.to_csv(shap_path, index=False)
        mlflow.log_artifact(shap_path)

    print(f"\n{'='*60}")
    print(f"✅ XAI analysis complete")
    print(f"   7 figures saved to: {FIGURES_DIR}")
    print(f"   All artifacts logged to MLflow")
    print(f"{'='*60}")
    print("\n[STEP 03 COMPLETE]")


if __name__ == "__main__":
    main()
