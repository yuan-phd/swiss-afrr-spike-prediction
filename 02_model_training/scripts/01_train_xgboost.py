"""
================================================================================
STEP 01 — Train XGBoost Spike Classifier v2
================================================================================
Changes from v1:
  - Removed hour, month, day_of_week from features (temporal bias fix)
  - Uses regime-invariant price_delta features instead of raw lags
  - Stronger regularisation (max_depth=4, min_child_weight=30, gamma=3)
  - Isotonic regression calibration layer added after training
  - Calibration model saved separately for downstream use
================================================================================
"""

import os
import sys
import yaml
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             f1_score, precision_score, recall_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression

# ── Configuration ──────────────────────────────────────────────────────────────
ROOT        = os.path.join(os.path.dirname(__file__), "..", "..")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

TRAIN_PATH  = os.path.join(ROOT, cfg["paths"]["train_csv"])
MODEL_DIR   = os.path.join(ROOT, cfg["paths"]["model_dir"])
FIGURES_DIR = os.path.join(ROOT, cfg["paths"]["figures_dir"])
os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

TARGET       = cfg["target"]
ALL_FEATURES = (cfg["features"]["hammer"] + cfg["features"]["anvil"] +
                cfg["features"]["incentive"] + cfg["features"]["autoregressive"] +
                cfg["features"]["structural"])
XGB_PARAMS   = cfg["xgboost"]
CV_SPLITS    = cfg["cross_validation"]["n_splits"]
CAL_FRAC     = cfg["calibration"]["calibration_frac"]
CAL_METHOD   = cfg["calibration"]["method"]

# Colour palette
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_and_transform(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load CSV and compute regime-invariant features.
    v3 change: clip price_vs_threshold to max 2.0
    Ratios above 2.0 are extreme outliers (price = 2x spike threshold)
    that XGBoost splits on aggressively, crowding out physical features.
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])

    # Clip rolling threshold edge case
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

    # Validate all features present
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        print(f"❌ Missing features: {missing}")
        sys.exit(1)

    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()
    X = X.fillna(X.median())

    # v3: clip price_vs_threshold outliers
    # Ratios > 2.0 = price is more than 2x the spike threshold
    # These extreme values create dominant splits that crowd out
    # physical features. Clipping forces model to rely on other features
    # for the most extreme cases.
    for col in ["price_vs_threshold_lag1", "price_vs_threshold_lag4"]:
        if col in X.columns:
            n_clipped = (X[col] > 2.0).sum()
            if n_clipped > 0:
                print(f"  v3 clip: {col} — {n_clipped} values above 2.0 clipped")
            X[col] = X[col].clip(upper=2.0)

    print(f"  Rows: {len(X):,}  |  "
          f"Spike rate: {y.mean()*100:.1f}%  |  "
          f"Features: {len(ALL_FEATURES)}")
    return X, y


def build_monotonic_constraints(features: list, constraints: dict) -> tuple:
    return tuple(constraints.get(f, 0) for f in features)


def get_xgb_params(features: list) -> dict:
    monotonic = build_monotonic_constraints(features,
                                            cfg["monotonic_constraints"])
    return {
        "n_estimators":          XGB_PARAMS["n_estimators"],
        "max_depth":             XGB_PARAMS["max_depth"],
        "learning_rate":         XGB_PARAMS["learning_rate"],
        "subsample":             XGB_PARAMS["subsample"],
        "colsample_bytree":      XGB_PARAMS["colsample_bytree"],
        "min_child_weight":      XGB_PARAMS["min_child_weight"],
        "gamma":                 XGB_PARAMS["gamma"],
        "reg_alpha":             XGB_PARAMS["reg_alpha"],
        "reg_lambda":            XGB_PARAMS["reg_lambda"],
        "scale_pos_weight":      XGB_PARAMS["scale_pos_weight"],
        "eval_metric":           XGB_PARAMS["eval_metric"],
        "early_stopping_rounds": XGB_PARAMS["early_stopping_rounds"],
        "random_state":          XGB_PARAMS["random_state"],
        "n_jobs":                XGB_PARAMS["n_jobs"],
        "monotone_constraints":  monotonic,
        "verbosity":             0,
    }


def cv_score(X: pd.DataFrame, y: pd.Series) -> dict:
    tscv    = TimeSeriesSplit(n_splits=CV_SPLITS)
    pr_aucs, roc_aucs, f1s = [], [], []

    print(f"\n  Running {CV_SPLITS}-fold TimeSeriesSplit CV...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        m = xgb.XGBClassifier(**get_xgb_params(X_tr.columns.tolist()))
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        y_prob  = m.predict_proba(X_val)[:, 1]
        y_pred  = (y_prob >= 0.5).astype(int)
        pr_auc  = average_precision_score(y_val, y_prob)
        roc_auc = roc_auc_score(y_val, y_prob)
        f1      = f1_score(y_val, y_pred, zero_division=0)

        pr_aucs.append(pr_auc)
        roc_aucs.append(roc_auc)
        f1s.append(f1)
        print(f"    Fold {fold+1}: PR-AUC={pr_auc:.4f}  "
              f"ROC-AUC={roc_auc:.4f}  F1={f1:.4f}")

    return {
        "cv_pr_auc_mean":  np.mean(pr_aucs),
        "cv_pr_auc_std":   np.std(pr_aucs),
        "cv_roc_auc_mean": np.mean(roc_aucs),
        "cv_roc_auc_std":  np.std(roc_aucs),
        "cv_f1_mean":      np.mean(f1s),
        "cv_f1_std":       np.std(f1s),
    }


def plot_feature_importance(model, features: list) -> plt.Figure:
    importance = model.feature_importances_
    df_imp = pd.DataFrame({
        "feature":    features,
        "importance": importance,
    }).sort_values("importance", ascending=True)

    category_colors = {
        "hammer":        AMBER,
        "anvil":         BLUE,
        "incentive":     GREEN,
        "autoregressive": "#bc8cff",
        "structural":    GREY,
    }
    feature_categories = {}
    for cat, cols in cfg["features"].items():
        for col in cols:
            feature_categories[col] = cat

    colors = [category_colors.get(feature_categories.get(f, ""), GREY)
              for f in df_imp["feature"]]

    fig, ax = plt.subplots(figsize=(12, max(8, len(features) * 0.38)),
                           facecolor=BG_DARK)
    ax.set_facecolor(BG_PANEL)
    ax.barh(df_imp["feature"], df_imp["importance"],
            color=colors, alpha=0.85, edgecolor="#30363d")
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title("XGBoost Feature Importance by Category (v2)",
                 fontsize=13, fontweight="bold", color=WHITE, pad=12)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=k.upper())
                       for k, c in category_colors.items()]
    ax.legend(handles=legend_elements, fontsize=9,
              facecolor=BG_DARK, edgecolor="#30363d",
              labelcolor=WHITE, loc="lower right")
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_calibration_comparison(y_true, y_prob_raw,
                                y_prob_cal) -> plt.Figure:
    """
    Compare calibration before and after isotonic regression.
    Shows the improvement from the calibration layer.
    """
    frac_raw, mean_raw = calibration_curve(y_true, y_prob_raw, n_bins=10)
    frac_cal, mean_cal = calibration_curve(y_true, y_prob_cal, n_bins=10)

    fig, ax = plt.subplots(figsize=(10, 7), facecolor=BG_DARK)
    ax.set_facecolor(BG_PANEL)
    ax.plot([0, 1], [0, 1], color=GREY, lw=1.5, ls="--",
            label="Perfect calibration")
    ax.plot(mean_raw, frac_raw, color=AMBER, lw=2.5, marker="o", ms=7,
            label="Before isotonic calibration")
    ax.plot(mean_cal, frac_cal, color=GREEN, lw=2.5, marker="s", ms=7,
            label="After isotonic calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (actual spike rate)")
    ax.set_title("Calibration Improvement — Isotonic Regression",
                 fontsize=13, fontweight="bold", color=WHITE)
    ax.legend(fontsize=10, facecolor=BG_DARK, edgecolor="#30363d",
              labelcolor=WHITE)
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STEP 01 v2 — XGBoost Spike Classifier Training")
    print("Changes: de-calendarized, regime-invariant lags, isotonic cal")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\n[1] Loading training data...")
    X, y = load_and_transform(TRAIN_PATH)

    # ── Split: fit set + calibration set ──────────────────────────────────────
    # Use last CAL_FRAC of training data for isotonic calibration
    # This is separate from the early-stopping eval set
    cal_split  = int(len(X) * (1 - CAL_FRAC))
    X_fit      = X.iloc[:cal_split]
    y_fit      = y.iloc[:cal_split]
    X_cal      = X.iloc[cal_split:]
    y_cal      = y.iloc[cal_split:]

    # Early stopping uses last 20% of fit set
    es_split   = int(len(X_fit) * 0.8)
    X_train_es = X_fit.iloc[:es_split]
    y_train_es = y_fit.iloc[:es_split]
    X_es       = X_fit.iloc[es_split:]
    y_es       = y_fit.iloc[es_split:]

    print(f"  Fit set:         {len(X_fit):,} rows")
    print(f"  Calibration set: {len(X_cal):,} rows "
          f"(spike rate: {y_cal.mean()*100:.1f}%)")

    # ── MLflow setup ───────────────────────────────────────────────────────────
    tracking_uri = os.path.join(ROOT, cfg["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(f"file://{os.path.abspath(tracking_uri)}")
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    print(f"\n[2] MLflow experiment: {cfg['mlflow']['experiment_name']}")

    with mlflow.start_run(run_name="xgboost_v3_clipped_colsample") as run:
        print(f"\n[3] MLflow run: {run.info.run_id}")

        # Log config
        mlflow.log_param("version",          "v3")
        mlflow.log_param("target",           TARGET)
        mlflow.log_param("n_features",       len(ALL_FEATURES))
        mlflow.log_param("removed_features", "hour,month,day_of_week")
        mlflow.log_param("lag_transform",    "price_delta_zscore+price_vs_threshold")
        mlflow.log_param("v3_changes",       "colsample_bytree=0.6, price_vs_threshold clipped at 2.0")
        mlflow.log_param("calibration",      CAL_METHOD)
        mlflow.log_params({f"xgb_{k}": v for k, v in XGB_PARAMS.items()})

        # ── Cross-validation ───────────────────────────────────────────────────
        print("\n[4] Cross-validation on fit set...")
        cv_metrics = cv_score(X_fit, y_fit)
        mlflow.log_metrics(cv_metrics)

        print(f"\n  CV Results:")
        print(f"    PR-AUC:  {cv_metrics['cv_pr_auc_mean']:.4f} "
              f"± {cv_metrics['cv_pr_auc_std']:.4f}")
        print(f"    ROC-AUC: {cv_metrics['cv_roc_auc_mean']:.4f} "
              f"± {cv_metrics['cv_roc_auc_std']:.4f}")
        print(f"    F1:      {cv_metrics['cv_f1_mean']:.4f} "
              f"± {cv_metrics['cv_f1_std']:.4f}")

        # ── Train final model ──────────────────────────────────────────────────
        print("\n[5] Training final model...")
        model = xgb.XGBClassifier(**get_xgb_params(ALL_FEATURES))
        model.fit(X_train_es, y_train_es,
                  eval_set=[(X_es, y_es)],
                  verbose=100)

        best_iteration = model.best_iteration
        print(f"  Best iteration: {best_iteration}")
        mlflow.log_metric("best_iteration", best_iteration)

        # ── Isotonic calibration ───────────────────────────────────────────────
        print(f"\n[6] Fitting isotonic calibration on {len(X_cal):,} samples...")
        y_prob_raw_cal = model.predict_proba(X_cal)[:, 1]

        # Fit isotonic regression on calibration set probabilities
        from sklearn.isotonic import IsotonicRegression
        iso_reg = IsotonicRegression(out_of_bounds="clip")
        iso_reg.fit(y_prob_raw_cal, y_cal)

        y_prob_cal_cal = iso_reg.predict(y_prob_raw_cal)

        # Calibration improvement on calibration set
        from sklearn.metrics import brier_score_loss
        brier_raw = brier_score_loss(y_cal, y_prob_raw_cal)
        brier_cal = brier_score_loss(y_cal, y_prob_cal_cal)
        print(f"  Brier score before calibration: {brier_raw:.4f}")
        print(f"  Brier score after calibration:  {brier_cal:.4f}")
        print(f"  Improvement: {(brier_raw - brier_cal)/brier_raw*100:.1f}%")

        mlflow.log_metric("cal_brier_before", brier_raw)
        mlflow.log_metric("cal_brier_after",  brier_cal)

        # Plot calibration comparison
        fig_cal = plot_calibration_comparison(y_cal, y_prob_raw_cal,
                                              y_prob_cal_cal)
        cal_path = os.path.join(FIGURES_DIR, "calibration_improvement.png")
        fig_cal.savefig(cal_path, dpi=150, bbox_inches="tight",
                        facecolor=BG_DARK)
        mlflow.log_artifact(cal_path)
        plt.close(fig_cal)

        # ── Training set metrics (raw model) ───────────────────────────────────
        print("\n[7] Computing training set metrics...")
        y_prob_train = model.predict_proba(X)[:, 1]
        y_pred_train = (y_prob_train >= 0.5).astype(int)

        train_metrics = {
            "train_pr_auc":   average_precision_score(y, y_prob_train),
            "train_roc_auc":  roc_auc_score(y, y_prob_train),
            "train_f1":       f1_score(y, y_pred_train, zero_division=0),
            "train_precision":precision_score(y, y_pred_train, zero_division=0),
            "train_recall":   recall_score(y, y_pred_train, zero_division=0),
        }
        mlflow.log_metrics(train_metrics)

        print(f"  Train PR-AUC:  {train_metrics['train_pr_auc']:.4f}")
        print(f"  Train ROC-AUC: {train_metrics['train_roc_auc']:.4f}")
        print(f"  Train F1:      {train_metrics['train_f1']:.4f}")

        # ── Feature importance ─────────────────────────────────────────────────
        print("\n[8] Feature importance...")
        fig_imp = plot_feature_importance(model, ALL_FEATURES)
        imp_path = os.path.join(FIGURES_DIR, "feature_importance_v2.png")
        fig_imp.savefig(imp_path, dpi=150, bbox_inches="tight",
                        facecolor=BG_DARK)
        mlflow.log_artifact(imp_path)
        plt.close(fig_imp)

        importance_df = pd.DataFrame({
            "feature":    ALL_FEATURES,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        print("  Top 10 features:")
        for _, row in importance_df.head(10).iterrows():
            print(f"    {row['feature']:<35} {row['importance']:.4f}")
            mlflow.log_metric(f"imp_{row['feature']}", row["importance"])

        # ── Save model + calibrator ────────────────────────────────────────────
        print("\n[9] Saving model and calibrator...")
        model_path = os.path.join(MODEL_DIR, "xgboost_spike_classifier.json")
        model.save_model(model_path)

        cal_model_path = os.path.join(MODEL_DIR, "isotonic_calibrator.pkl")
        with open(cal_model_path, "wb") as f:
            pickle.dump(iso_reg, f)

        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=cfg["mlflow"]["model_name"])
        mlflow.log_artifact(cal_model_path)

        # Save meta
        meta = {
            "version":        "v3",
            "features":       ALL_FEATURES,
            "target":         TARGET,
            "run_id":         run.info.run_id,
            "best_iteration": best_iteration,
            "cv_pr_auc":      cv_metrics["cv_pr_auc_mean"],
            "cal_method":     CAL_METHOD,
            "removed_features": ["hour", "month", "day_of_week"],
            "lag_transform":  "price_delta_zscore",
            "v3_changes":     "colsample_bytree=0.6, price_vs_threshold clipped at 2.0",
        }
        meta_path = os.path.join(MODEL_DIR, "model_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        mlflow.log_artifact(meta_path)

        print(f"\n{'='*60}")
        print(f"✅ Training v3 complete")
        print(f"   Changes: colsample_bytree=0.6, price_vs_threshold clipped at 2.0")
        print(f"   CV PR-AUC: {cv_metrics['cv_pr_auc_mean']:.4f} "
              f"± {cv_metrics['cv_pr_auc_std']:.4f}")
        print(f"   Calibration Brier improvement: "
              f"{(brier_raw-brier_cal)/brier_raw*100:.1f}%")
        print(f"   Run ID: {run.info.run_id}")
        print(f"{'='*60}")

    print("\n[STEP 01 v2 COMPLETE]")


if __name__ == "__main__":
    main()
