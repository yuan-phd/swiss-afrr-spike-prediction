"""
================================================================================
training/train_model.py — Full Retrain Script
================================================================================
PURPOSE:
    Load full historical feature data, retrain XGBoost with isotonic
    calibration, log everything to MLflow, save challenger model locally.

    This script is called by retrain_dag.py when drift_check detects
    PSI > 0.2 on key features.

HOW IT FITS IN THE PIPELINE:
    daily pipeline:   fetch → validate → build_features → predict → evaluate → drift_check
                                                                                     ↓
                                                                          (if PSI > 0.2)
                                                                                     ↓
    retrain_dag:      load_data → train_challenger → evaluate_challenger → promote_or_reject
                                       ↑
                                  THIS SCRIPT

CHAMPION/CHALLENGER PATTERN:
    Current champion: existing model in 02_model_training/models/
    New challenger:   model trained by this script
    Promotion:        challenger only replaces champion if Brier score improves

INPUTS:
    Local: data/processed/features_train_2023_2024.csv  (training set)
    Local: data/processed/features_val_2025.csv         (validation set)

OUTPUTS:
    Local: pipeline/training/challenger_model.json      (XGBoost challenger)
    Local: pipeline/training/challenger_calibrator.pkl  (isotonic calibrator)
    MLflow: retrain run with all params and metrics logged

USAGE:
    Called by retrain_dag.py automatically when drift detected.
    Can also be run manually:
        python3 training/train_model.py
================================================================================
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss, average_precision_score
from airflow.models import Variable

# ── Configuration ──────────────────────────────────────────────────────────────
PROJECT_DIR  = '/Users/yuan/Work/profolio/swiss-afrr-spike-prediction'
TRAINING_DIR = os.path.join(PROJECT_DIR, 'pipeline', 'training')

TRAIN_PATH = os.path.join(PROJECT_DIR, 'data', 'processed',
                           'features_train_2023_2024.csv')
VAL_PATH   = os.path.join(PROJECT_DIR, 'data', 'processed',
                           'features_val_2025.csv')

# Challenger model output paths
CHALLENGER_MODEL_PATH      = os.path.join(TRAINING_DIR, 'challenger_model.json')
CHALLENGER_CALIBRATOR_PATH = os.path.join(TRAINING_DIR, 'challenger_calibrator.pkl')
CHALLENGER_META_PATH       = os.path.join(TRAINING_DIR, 'challenger_meta.json')

# Champion model paths (for comparison)
CHAMPION_MODEL_PATH      = os.path.join(PROJECT_DIR, '02_model_training',
                                         'models', 'xgboost_spike_classifier.json')
CHAMPION_CALIBRATOR_PATH = os.path.join(PROJECT_DIR, '02_model_training',
                                         'models', 'isotonic_calibrator.pkl')

MLFLOW_TRACKING_URI = 'http://localhost:5001'
EXPERIMENT_NAME     = 'afrr_spike_prediction'

# Target variable
TARGET = 'price_spike_rolling'

# Exact feature list from model_meta.json
FEATURE_COLS = [
    'Unplanned_Flow',
    'abs_Unplanned_Flow',
    'Unplanned_Flow_rolling4',
    'DE_WindSolar_Error',
    'Unplanned_Flow_FR_CH',
    'abs_Unplanned_Flow_FR_CH',
    'Total_Unplanned_Flow',
    'Sched_DE_CH',
    'Sched_DE_CH_delta',
    'Sched_FR_CH',
    'Unplanned_x_Sched',
    'rolling_p90_threshold',
    'CH_Load_Forecast',
    'CH_Pump_Gen',
    'DA_Price_DE',
    'DA_Price_CH',
    'DA_Price_Spread_DE_CH',
    'price_delta_lag1',
    'price_delta_lag4',
    'price_delta_lag96',
    'price_vs_threshold_lag1',
    'price_vs_threshold_lag4',
    'minute',
    'is_turnover',
]

# XGBoost hyperparameters (same as original training)
XGB_PARAMS = {
    'n_estimators':       300,
    'max_depth':          4,
    'learning_rate':      0.05,
    'subsample':          0.8,
    'colsample_bytree':   0.8,
    'min_child_weight':   10,
    'scale_pos_weight':   9,      # handles class imbalance (~10% spikes)
    'eval_metric':        'aucpr',
    'early_stopping_rounds': 20,
    'random_state':       42,
    'tree_method':        'hist',
}


def load_data():
    """Load training and validation feature sets."""
    print("[train_model] Loading training data...")

    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(
            f"Training data not found: {TRAIN_PATH}\n"
            f"Run 01_feature_engineering.py first."
        )
    if not os.path.exists(VAL_PATH):
        raise FileNotFoundError(
            f"Validation data not found: {VAL_PATH}\n"
            f"Run 01_feature_engineering.py first."
        )

    train = pd.read_csv(TRAIN_PATH, parse_dates=['timestamp'])
    val   = pd.read_csv(VAL_PATH,   parse_dates=['timestamp'])

    print(f"[train_model] Train: {len(train):,} rows "
          f"({train['timestamp'].min().date()} → {train['timestamp'].max().date()})")
    print(f"[train_model] Val:   {len(val):,} rows "
          f"({val['timestamp'].min().date()} → {val['timestamp'].max().date()})")

    # Check target exists
    if TARGET not in train.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found in training data.\n"
            f"Available columns: {train.columns.tolist()}"
        )

    return train, val


def prepare_features(train, val):
    """Extract feature matrices and target vectors."""
    # Only keep feature columns that exist in both datasets
    available = [c for c in FEATURE_COLS if c in train.columns]
    missing   = [c for c in FEATURE_COLS if c not in train.columns]

    if missing:
        print(f"[train_model] ⚠️ Missing features: {missing}")

    X_train = train[available].copy()
    y_train = train[TARGET].copy()
    X_val   = val[available].copy()
    y_val   = val[TARGET].copy()

    print(f"[train_model] Features: {len(available)}")
    print(f"[train_model] Train spike rate: {y_train.mean():.1%}")
    print(f"[train_model] Val spike rate:   {y_val.mean():.1%}")

    return X_train, y_train, X_val, y_val


def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Train XGBoost with early stopping on validation set.
    Same setup as original 01_train_xgboost.py.
    """
    print("[train_model] Training XGBoost...")

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    best_iter = model.best_iteration
    print(f"[train_model] Best iteration: {best_iter}")
    return model


def train_calibrator(model, X_val, y_val):
    """
    Train isotonic calibrator on validation set.
    Maps raw XGBoost probabilities → calibrated probabilities.

    Why isotonic regression?
    - XGBoost probabilities are often over/under-confident
    - Isotonic regression learns a monotonic mapping to fix this
    - Trained on val set (not train set) to avoid overfitting
    """
    print("[train_model] Training isotonic calibrator...")

    raw_probs  = model.predict_proba(X_val)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(raw_probs, y_val)

    calibrated_probs = calibrator.transform(raw_probs)
    print(f"[train_model] Calibrator trained on {len(raw_probs)} val samples")
    return calibrator, raw_probs, calibrated_probs


def evaluate_challenger(model, calibrator, X_val, y_val):
    """Compute Brier score and PR-AUC for the challenger model."""
    raw_probs        = model.predict_proba(X_val)[:, 1]
    calibrated_probs = calibrator.transform(raw_probs)

    brier  = brier_score_loss(y_val, calibrated_probs)
    pr_auc = average_precision_score(y_val, calibrated_probs)

    print(f"[train_model] Challenger metrics:")
    print(f"  Brier score: {brier:.4f}")
    print(f"  PR-AUC:      {pr_auc:.4f}")

    return brier, pr_auc


def get_champion_brier():
    """
    Get champion Brier score from the last promoted MLflow run.
    Falls back to 0.0623 if no promoted run exists yet.
    """
    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        runs = client.search_runs(
            experiment_ids=['1'],
            filter_string="params.promoted = 'True'",
            order_by=["start_time DESC"],
            max_results=1
        )
        if runs:
            brier = runs[0].data.metrics['challenger_brier']
            print(f"[train_model] Champion Brier from MLflow: {brier:.4f}")
            return brier
        else:
            print("[train_model] No promoted run found — using default 0.0623")
            return 0.0623
    except Exception as e:
        print(f"[train_model] MLflow query failed: {e} — using default 0.0623")
        return 0.0623


def save_challenger(model, calibrator, brier, pr_auc):
    """Save challenger model files and metadata."""
    os.makedirs(TRAINING_DIR, exist_ok=True)

    model.save_model(CHALLENGER_MODEL_PATH)
    joblib.dump(calibrator, CHALLENGER_CALIBRATOR_PATH)

    meta = {
        'brier_score': brier,
        'pr_auc':      pr_auc,
        'n_features':  len(FEATURE_COLS),
        'target':      TARGET,
    }
    with open(CHALLENGER_META_PATH, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"[train_model] Challenger saved:")
    print(f"  Model:      {CHALLENGER_MODEL_PATH}")
    print(f"  Calibrator: {CHALLENGER_CALIBRATOR_PATH}")
    print(f"  Meta:       {CHALLENGER_META_PATH}")


def main():
    """
    Full retrain pipeline:
    1. Load data
    2. Train XGBoost
    3. Train calibrator
    4. Evaluate challenger
    5. Compare vs champion
    6. Log to MLflow
    7. Save challenger files
    """
    print("=" * 60)
    print("RETRAIN PIPELINE")
    print("=" * 60)

    # ── 1. Load data ───────────────────────────────────────────────────────────
    train, val = load_data()
    X_train, y_train, X_val, y_val = prepare_features(train, val)

    # ── 2. Train XGBoost ───────────────────────────────────────────────────────
    model = train_xgboost(X_train, y_train, X_val, y_val)

    # ── 3. Train calibrator ────────────────────────────────────────────────────
    calibrator, raw_probs, calibrated_probs = train_calibrator(model, X_val, y_val)

    # ── 4. Evaluate challenger ─────────────────────────────────────────────────
    brier, pr_auc = evaluate_challenger(model, calibrator, X_val, y_val)

    # ── 5. Compare vs champion ─────────────────────────────────────────────────
    champion_brier = get_champion_brier()
    promoted = brier < champion_brier

    print(f"\n[train_model] Champion vs Challenger:")
    print(f"  Champion Brier:   {champion_brier:.4f}")
    print(f"  Challenger Brier: {brier:.4f}")
    print(f"  Decision:         {'✅ PROMOTE challenger' if promoted else '❌ KEEP champion'}")

    # ── 6. Log to MLflow ───────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="retrain_challenger"):
        # Log all XGBoost hyperparameters
        for param, value in XGB_PARAMS.items():
            if param not in ['eval_metric']:
                mlflow.log_param(param, value)

        mlflow.log_param("target",         TARGET)
        mlflow.log_param("n_train_rows",   len(X_train))
        mlflow.log_param("n_val_rows",     len(X_val))
        mlflow.log_param("n_features",     len(FEATURE_COLS))
        mlflow.log_param("promoted",       promoted)

        mlflow.log_metric("challenger_brier",  brier)
        mlflow.log_metric("challenger_pr_auc", pr_auc)
        mlflow.log_metric("champion_brier",    champion_brier)
        mlflow.log_metric("brier_improvement", champion_brier - brier)

        # Tag this run as promoted so next retrain can find it
        if promoted:
            mlflow.set_tag("promoted", "True")
            print(f"[train_model] ✅ Tagged MLflow run as promoted")

    print(f"[train_model] ✅ Logged to MLflow experiment: {EXPERIMENT_NAME}")

    # ── 7. Save challenger ─────────────────────────────────────────────────────
    save_challenger(model, calibrator, brier, pr_auc)

    print("\n" + "=" * 60)
    print(f"RETRAIN COMPLETE")
    print(f"  Challenger Brier: {brier:.4f}")
    print(f"  Champion Brier:   {champion_brier:.4f}")
    print(f"  Decision:         {'PROMOTE' if promoted else 'KEEP CHAMPION'}")
    print("=" * 60)

    return promoted, brier


if __name__ == "__main__":
    main()