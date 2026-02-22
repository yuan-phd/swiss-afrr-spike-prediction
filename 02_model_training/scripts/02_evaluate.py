"""
================================================================================
STEP 02 — Model Evaluation v2
================================================================================
Changes from v1:
  - Loads isotonic calibrator and evaluates calibrated predictions
  - Computes both raw and calibrated metrics for comparison
  - Computes regime-invariant features if not already in CSV
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
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_score, recall_score, brier_score_loss,
    precision_recall_curve, roc_curve, confusion_matrix)
from sklearn.calibration import calibration_curve

# ── Configuration ──────────────────────────────────────────────────────────────
ROOT        = os.path.join(os.path.dirname(__file__), "..", "..")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

VAL_PATH    = os.path.join(ROOT, cfg["paths"]["val_csv"])
MODEL_DIR   = os.path.join(ROOT, cfg["paths"]["model_dir"])
RESULTS_DIR = os.path.join(ROOT, cfg["paths"]["results_dir"])
FIGURES_DIR = os.path.join(ROOT, cfg["paths"]["figures_dir"])
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

TARGET       = cfg["target"]
ALL_FEATURES = (cfg["features"]["hammer"] + cfg["features"]["anvil"] +
                cfg["features"]["incentive"] + cfg["features"]["autoregressive"] +
                cfg["features"]["structural"])

BG_DARK  = "#0d1117"
BG_PANEL = "#1c2130"
BLUE     = "#58a6ff"
GREEN    = "#3fb950"
AMBER    = "#d29922"
RED      = "#f85149"
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


def style_ax(ax):
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    ax.grid(True, alpha=0.3)


def load_and_transform(path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if "rolling_p90_threshold" in df.columns:
        df["rolling_p90_threshold"] = df["rolling_p90_threshold"].clip(lower=50)

    # Compute regime-invariant features if not already present
    if "price_delta_lag1" not in df.columns:
        print("  Computing price_delta features locally...")
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

    X = df[ALL_FEATURES].fillna(df[ALL_FEATURES].median())
    y = df[TARGET]
    return df, X, y


def plot_pr_curve(y_true, y_prob_raw, y_prob_cal, pr_auc_raw,
                  pr_auc_cal) -> tuple[plt.Figure, float]:
    precision_r, recall_r, thresh_r = precision_recall_curve(y_true, y_prob_raw)
    precision_c, recall_c, thresh_c = precision_recall_curve(y_true, y_prob_cal)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG_DARK)
    ax.plot(recall_r, precision_r, color=AMBER, lw=2,
            label=f"Raw model (PR-AUC={pr_auc_raw:.4f})", alpha=0.7)
    ax.plot(recall_c, precision_c, color=BLUE, lw=2.5,
            label=f"Calibrated (PR-AUC={pr_auc_cal:.4f})")
    ax.axhline(baseline, color=GREY, lw=1.5, ls="--",
               label=f"Random baseline ({baseline:.3f})")

    # Optimal threshold on calibrated
    f1_scores = 2 * precision_c * recall_c / (precision_c + recall_c + 1e-8)
    opt_idx   = np.argmax(f1_scores[:-1])
    opt_thresh = thresh_c[opt_idx]
    ax.scatter(recall_c[opt_idx], precision_c[opt_idx],
               color=GREEN, s=120, zorder=5,
               label=f"Optimal F1={f1_scores[opt_idx]:.3f} "
                     f"@ thresh={opt_thresh:.3f}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve v2 — Raw vs Calibrated",
                 fontsize=13, fontweight="bold", color=WHITE)
    ax.legend(fontsize=9, facecolor=BG_DARK, edgecolor="#30363d",
              labelcolor=WHITE)
    style_ax(ax)
    fig.tight_layout()
    return fig, opt_thresh


def plot_calibration(y_true, y_prob_raw, y_prob_cal) -> plt.Figure:
    frac_r, mean_r = calibration_curve(y_true, y_prob_raw, n_bins=10)
    frac_c, mean_c = calibration_curve(y_true, y_prob_cal, n_bins=10)
    brier_r = brier_score_loss(y_true, y_prob_raw)
    brier_c = brier_score_loss(y_true, y_prob_cal)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG_DARK)
    fig.suptitle("Calibration Analysis v2 — Raw vs Calibrated",
                 fontsize=13, fontweight="bold", color=WHITE)

    for ax in axes:
        ax.plot([0, 1], [0, 1], color=GREY, lw=1.5, ls="--",
                label="Perfect calibration")
        style_ax(ax)

    axes[0].plot(mean_r, frac_r, color=AMBER, lw=2.5, marker="o", ms=7,
                 label=f"Raw (Brier={brier_r:.4f})")
    axes[0].set_title("Before Calibration", color=WHITE)
    axes[0].set_xlabel("Mean Predicted Probability")
    axes[0].set_ylabel("Fraction of Positives")
    axes[0].legend(fontsize=9, facecolor=BG_DARK, edgecolor="#30363d",
                   labelcolor=WHITE)

    axes[1].plot(mean_c, frac_c, color=GREEN, lw=2.5, marker="s", ms=7,
                 label=f"Calibrated (Brier={brier_c:.4f})")
    axes[1].set_title("After Isotonic Calibration", color=WHITE)
    axes[1].set_xlabel("Mean Predicted Probability")
    axes[1].legend(fontsize=9, facecolor=BG_DARK, edgecolor="#30363d",
                   labelcolor=WHITE)

    fig.tight_layout()
    return fig


def plot_monthly(val_df, y_prob_cal, y_true, threshold) -> plt.Figure:
    val_df = val_df.copy()
    val_df["y_prob"] = y_prob_cal
    val_df["y_true"] = y_true.values
    val_df["y_pred"] = (y_prob_cal >= threshold).astype(int)
    val_df["month_label"] = val_df["timestamp"].dt.strftime("%Y-%m")

    monthly = (val_df.groupby("month_label")
               .apply(lambda g: pd.Series({
                   "pr_auc":     average_precision_score(g["y_true"], g["y_prob"])
                                 if g["y_true"].sum() > 0 else np.nan,
                   "spike_rate": g["y_true"].mean() * 100,
                   "pred_rate":  g["y_pred"].mean() * 100,
               }), include_groups=False)
               .reset_index())

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), facecolor=BG_DARK)
    fig.suptitle("Monthly Performance v2 — Calibrated Predictions",
                 fontsize=13, fontweight="bold", color=WHITE)

    x = range(len(monthly))

    axes[0].bar(x, monthly["pr_auc"], color=BLUE, alpha=0.85,
                edgecolor="#30363d")
    axes[0].axhline(monthly["pr_auc"].mean(), color=AMBER, lw=2, ls="--",
                    label=f"Mean PR-AUC = {monthly['pr_auc'].mean():.3f}")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(monthly["month_label"], rotation=45, ha="right")
    axes[0].set_ylabel("PR-AUC")
    axes[0].set_title("PR-AUC by Month (Calibrated)", color=WHITE)
    axes[0].legend(fontsize=9, facecolor=BG_DARK, edgecolor="#30363d",
                   labelcolor=WHITE)
    style_ax(axes[0])

    w = 0.4
    axes[1].bar([i-w/2 for i in x], monthly["spike_rate"], w,
                color=RED,   alpha=0.85, edgecolor="#30363d",
                label="Actual spike rate %")
    axes[1].bar([i+w/2 for i in x], monthly["pred_rate"],  w,
                color=GREEN, alpha=0.85, edgecolor="#30363d",
                label="Predicted spike rate %")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(monthly["month_label"], rotation=45, ha="right")
    axes[1].set_ylabel("Spike Rate (%)")
    axes[1].set_title("Actual vs Predicted Spike Rate by Month", color=WHITE)
    axes[1].legend(fontsize=9, facecolor=BG_DARK, edgecolor="#30363d",
                   labelcolor=WHITE)
    style_ax(axes[1])

    fig.tight_layout()
    return fig


def plot_confusion(y_true, y_pred) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=BG_DARK)
    ax.set_facecolor(BG_PANEL)
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color=WHITE, fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: No Spike", "Pred: Spike"])
    ax.set_yticklabels(["Actual: No Spike", "Actual: Spike"])
    ax.set_title("Confusion Matrix v2 — Calibrated Predictions",
                 fontsize=13, fontweight="bold", color=WHITE, pad=12)
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    fig.tight_layout()
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STEP 02 v2 — Model Evaluation (Calibrated)")
    print("=" * 60)

    # ── Load model + calibrator ────────────────────────────────────────────────
    print("\n[1] Loading model and calibrator...")
    model_path = os.path.join(MODEL_DIR, "xgboost_spike_classifier.json")
    cal_path   = os.path.join(MODEL_DIR, "isotonic_calibrator.pkl")
    meta_path  = os.path.join(MODEL_DIR, "model_meta.json")

    if not os.path.exists(model_path):
        print("❌ Model not found — run 01_train_xgboost.py first")
        sys.exit(1)

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    with open(cal_path, "rb") as f:
        iso_reg = pickle.load(f)
    with open(meta_path) as f:
        meta = json.load(f)

    print(f"  Model loaded (v{meta.get('version','1')})")
    print(f"  CV PR-AUC from training: {meta['cv_pr_auc']:.4f}")

    # ── Load validation data ───────────────────────────────────────────────────
    print("\n[2] Loading validation data...")
    val_df, X_val, y_val = load_and_transform(VAL_PATH)
    print(f"  Val rows: {len(val_df):,}  |  Spike rate: {y_val.mean()*100:.1f}%")

    # ── Predict — raw and calibrated ──────────────────────────────────────────
    print("\n[3] Generating predictions...")
    y_prob_raw = model.predict_proba(X_val)[:, 1]
    y_prob_cal = iso_reg.predict(y_prob_raw)

    # ── Compute metrics ────────────────────────────────────────────────────────
    print("\n[4] Computing metrics...")
    pr_auc_raw = average_precision_score(y_val, y_prob_raw)
    pr_auc_cal = average_precision_score(y_val, y_prob_cal)
    roc_raw    = roc_auc_score(y_val, y_prob_raw)
    roc_cal    = roc_auc_score(y_val, y_prob_cal)
    brier_raw  = brier_score_loss(y_val, y_prob_raw)
    brier_cal  = brier_score_loss(y_val, y_prob_cal)

    # MLflow
    tracking_uri = os.path.join(ROOT, cfg["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(f"file://{os.path.abspath(tracking_uri)}")
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="evaluation_v3_calibrated",
                          run_id=meta["run_id"]):

        fig_pr, opt_thresh = plot_pr_curve(y_val, y_prob_raw, y_prob_cal,
                                            pr_auc_raw, pr_auc_cal)
        y_pred = (y_prob_cal >= opt_thresh).astype(int)

        f1  = f1_score(y_val, y_pred, zero_division=0)
        prec= precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)

        val_metrics = {
            "val_pr_auc_raw":        pr_auc_raw,
            "val_pr_auc_calibrated": pr_auc_cal,
            "val_roc_auc_raw":       roc_raw,
            "val_roc_auc_calibrated":roc_cal,
            "val_brier_raw":         brier_raw,
            "val_brier_calibrated":  brier_cal,
            "val_f1_calibrated":     f1,
            "val_precision":         prec,
            "val_recall":            rec,
            "val_optimal_threshold": opt_thresh,
        }
        mlflow.log_metrics(val_metrics)

        print(f"\n  {'Metric':<30} {'Raw':>10} {'Calibrated':>12}")
        print(f"  {'-'*54}")
        print(f"  {'PR-AUC':<30} {pr_auc_raw:>10.4f} {pr_auc_cal:>12.4f}")
        print(f"  {'ROC-AUC':<30} {roc_raw:>10.4f} {roc_cal:>12.4f}")
        print(f"  {'Brier Score':<30} {brier_raw:>10.4f} {brier_cal:>12.4f}")
        print(f"  {'F1 (opt threshold)':<30} {'—':>10} {f1:>12.4f}")
        print(f"  {'Precision':<30} {'—':>10} {prec:>12.4f}")
        print(f"  {'Recall':<30} {'—':>10} {rec:>12.4f}")
        print(f"  {'Optimal threshold':<30} {'—':>10} {opt_thresh:>12.4f}")

        # Save plots
        print("\n[5] Generating plots...")
        plots = {
            "precision_recall_curve_v2.png": fig_pr,
            "calibration_v2.png":            plot_calibration(y_val,
                                                y_prob_raw, y_prob_cal),
            "monthly_performance_v2.png":    plot_monthly(val_df, y_prob_cal,
                                                y_val, opt_thresh),
            "confusion_matrix_v2.png":       plot_confusion(y_val, y_pred),
        }
        for fname, fig in plots.items():
            path = os.path.join(FIGURES_DIR, fname)
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
            mlflow.log_artifact(path)
            plt.close(fig)
            print(f"  Saved → {fname}")

        # Save report
        report = pd.DataFrame([val_metrics])
        report["run_id"] = meta["run_id"]
        report_path = os.path.join(RESULTS_DIR, "evaluation_report_v2.csv")
        report.to_csv(report_path, index=False)
        mlflow.log_artifact(report_path)

        # Update meta
        meta["optimal_threshold"] = float(opt_thresh)
        meta["val_pr_auc_cal"]    = float(pr_auc_cal)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ Evaluation v3 complete")
    print(f"   Val PR-AUC (raw):        {pr_auc_raw:.4f}")
    print(f"   Val PR-AUC (calibrated): {pr_auc_cal:.4f}")
    print(f"   Brier improvement:       "
          f"{(brier_raw-brier_cal)/brier_raw*100:.1f}%")
    print(f"   Optimal threshold:       {opt_thresh:.4f}")
    print(f"   F1 at threshold:         {f1:.4f}")
    print(f"{'='*60}")
    print("\n[STEP 02 v3 COMPLETE]")


if __name__ == "__main__":
    main()
