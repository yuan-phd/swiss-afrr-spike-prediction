"""
================================================================================
statistical_anomaly_detector.py — Statistical Anomaly Detector
================================================================================
PURPOSE:
    Detect anomalies that domain rules and logical rules CANNOT catch:
    multivariate outliers — feature combinations that are individually valid
    but collectively unusual compared to training data.

WHY MULTIVARIATE MATTERS:
    Domain rule:  abs_Unplanned_Flow < 10000  ✅
    Domain rule:  DE_WindSolar_Error < 25000  ✅
    Domain rule:  DA_Price_CH < 4000          ✅

    But what about all THREE happening simultaneously at unusual values?
    Each individual value passes domain checks. Each individual feature passes
    PSI. But the COMBINATION is unprecedented.

    This is what IsolationForest catches.

WHAT THIS CATCHES:
    - Subtle multivariate drift below PSI threshold
    - Novel feature combinations never seen in training
    - Coordinated anomalies across multiple features
    - Slow drift that hasn't yet triggered PSI > 0.2

WHAT THIS DOES NOT CATCH:
    - Single-feature out-of-bounds (use domain detector)
    - Time-series gaps or pipeline bugs (use logical detector)
    - Distribution-level shift (use drift_check / PSI)

ALGORITHM:
    IsolationForest — fits 100 random trees, isolates each point by random
    splits. Anomalies require fewer splits to isolate. Score is the average
    path length, normalised so:
        score < -0.3  → likely anomaly
        score > 0     → likely normal

    Why IsolationForest over OneClassSVM or EllipticEnvelope?
        - OneClassSVM: O(n²) — too slow on 70k training rows
        - EllipticEnvelope: assumes Gaussian — our features are heavily skewed
        - IsolationForest: no distribution assumption, fast, well-suited

USAGE:
    detector = StatisticalAnomalyDetector()
    detector.fit(X_train)              # train once on clean training data
    detector.save('artifacts/detector.pkl')

    # Daily in pipeline:
    detector.load('artifacts/detector.pkl')
    anomalies = detector.score(X_today)  # returns scores per row
================================================================================
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


# ── Result type ──────────────────────────────────────────────────────────────
@dataclass
class StatisticalResult:
    n_rows: int                      # total rows scored
    n_anomalies: int                 # rows flagged as anomalies
    anomaly_rate: float              # fraction flagged
    score_min: float                 # most anomalous score
    score_max: float                 # most normal score
    score_mean: float
    threshold: float                 # threshold used to flag
    anomaly_indices: list            # row indices flagged
    sample_anomalies: list           # up to 5 most anomalous rows with details


# ── Main detector class ───────────────────────────────────────────────────────
class StatisticalAnomalyDetector:
    """
    IsolationForest wrapper with feature scaling, model persistence,
    and integration with the Airflow pipeline.

    Two-phase usage:
        1. fit() — train on clean training data (one-time)
        2. score() — predict on new data (daily)
    """

    def __init__(
        self,
        contamination: float = 0.05,    # fraction expected anomalies in training
        n_estimators: int = 100,        # number of trees
        max_samples: int = 256,         # samples per tree
        random_state: int = 42,
        threshold: float = -0.05,       # score below = anomaly (tune later)
        feature_cols: Optional[list] = None,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.threshold = threshold
        self.feature_cols = feature_cols  # if None, uses all numeric columns

        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[RobustScaler] = None
        self.train_metadata: dict = {}

    # ── Feature preparation ──────────────────────────────────────────────────
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select feature columns and handle NaN.

        IsolationForest does NOT handle NaN natively (unlike XGBoost), so:
          - Drop rows with NaN in critical features
          - This is conservative — we'd rather skip a row than give a bogus score
        """
        if self.feature_cols is not None:
            available = [c for c in self.feature_cols if c in df.columns]
            X = df[available].copy()
        else:
            # Use all numeric columns except metadata
            exclude = {'timestamp', 'price_spike', 'price_spike_fixed',
                       'price_spike_rolling', 'price_spike_pooled',
                       'spike_threshold', 'pos_sec_price', 'neg_sec_price'}
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            X = df[[c for c in numeric_cols if c not in exclude]].copy()

        # Drop rows with NaN
        before = len(X)
        X = X.dropna()
        dropped = before - len(X)
        if dropped > 0:
            print(f"[StatDetector] Dropped {dropped} rows with NaN features")

        return X

    # ── Training ─────────────────────────────────────────────────────────────
    def fit(self, df_train: pd.DataFrame) -> dict:
        """
        Fit the IsolationForest model on clean training data.

        Steps:
          1. Select feature columns
          2. Drop NaN rows
          3. Fit RobustScaler (handles outliers better than StandardScaler)
          4. Fit IsolationForest on scaled features
          5. Compute training score distribution for threshold tuning
        """
        print(f"[StatDetector] Training on {len(df_train):,} rows")

        X = self._prepare_features(df_train)
        print(f"[StatDetector] Feature matrix: {X.shape}")
        print(f"[StatDetector] Features used: {list(X.columns)}")

        # RobustScaler uses median + IQR — handles outliers better than mean+std
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Fit IsolationForest
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)
        print(f"[StatDetector] IsolationForest trained")

        # Compute training score distribution
        train_scores = self.model.score_samples(X_scaled)
        train_predictions = self.model.predict(X_scaled)
        train_anomaly_rate = float((train_predictions == -1).mean())

        # Set dynamic threshold to P5 of training scores (used for reporting)
        self._dynamic_threshold = float(np.percentile(train_scores, 5))

        self.train_metadata = {
            'n_train_rows':    len(X),
            'feature_cols':    list(X.columns),
            'score_min':       float(train_scores.min()),
            'score_max':       float(train_scores.max()),
            'score_mean':      float(train_scores.mean()),
            'score_p1':        float(np.percentile(train_scores, 1)),
            'score_p5':        float(np.percentile(train_scores, 5)),
            'score_p50':       float(np.percentile(train_scores, 50)),
            'score_p95':       float(np.percentile(train_scores, 95)),
            'dynamic_threshold': self._dynamic_threshold,
            'training_anomaly_rate': train_anomaly_rate,
        }

        # Save the actual feature_cols for inference
        self.feature_cols = list(X.columns)

        print(f"[StatDetector] Training complete:")
        print(f"  Score range: {train_scores.min():.3f} to {train_scores.max():.3f}")
        print(f"  P5 score:    {self.train_metadata['score_p5']:.3f}")
        print(f"  P50 score:   {self.train_metadata['score_p50']:.3f}")
        print(f"  Dynamic threshold (P5): {self._dynamic_threshold:.3f}")
        print(f"  Training anomaly rate (contamination-based): "
              f"{self.train_metadata['training_anomaly_rate']*100:.1f}%")

        return self.train_metadata

    # ── Inference ────────────────────────────────────────────────────────────
    def score(self, df: pd.DataFrame) -> StatisticalResult:
        """
        Score new data against the trained model.

        Returns a StatisticalResult with per-row scores and flagged anomalies.
        """
        if self.model is None:
            raise RuntimeError(
                "Model not trained. Call fit() first or load() from disk."
            )

        X = self._prepare_features(df)
        if len(X) == 0:
            return StatisticalResult(
                n_rows=0, n_anomalies=0, anomaly_rate=0.0,
                score_min=0.0, score_max=0.0, score_mean=0.0,
                threshold=self.threshold,
                anomaly_indices=[], sample_anomalies=[],
            )

        X_scaled = self.scaler.transform(X)
        scores = self.model.score_samples(X_scaled)

        # Use the model's own contamination-based decision (not arbitrary threshold)
        # predict() returns -1 for anomaly, +1 for normal, based on contamination
        # This is more robust than a hardcoded threshold.
        predictions = self.model.predict(X_scaled)
        is_anomaly = predictions == -1

        # Update threshold dynamically based on training distribution
        # (for reporting purposes — the actual decision uses predict())
        if not hasattr(self, '_dynamic_threshold'):
            self._dynamic_threshold = float(np.percentile(scores, 5))

        # Get the 5 most anomalous rows for inspection
        # Sort by score ascending (most negative = most anomalous)
        anomaly_indices = X.index[is_anomaly].tolist()
        score_series = pd.Series(scores, index=X.index)
        most_anomalous = score_series.nsmallest(5)

        sample_anomalies = []
        for idx, score in most_anomalous.items():
            row_data = X.loc[idx].to_dict()
            sample_anomalies.append({
                'index':     int(idx),
                'score':     float(score),
                'top_features': self._explain_anomaly(X.loc[idx], X),
            })

        return StatisticalResult(
            n_rows=len(X),
            n_anomalies=int(is_anomaly.sum()),
            anomaly_rate=float(is_anomaly.mean()),
            score_min=float(scores.min()),
            score_max=float(scores.max()),
            score_mean=float(scores.mean()),
            threshold=getattr(self, '_dynamic_threshold', self.threshold),
            anomaly_indices=anomaly_indices,
            sample_anomalies=sample_anomalies,
        )

    def _explain_anomaly(self, row: pd.Series, X_train: pd.DataFrame) -> list:
        """
        Identify which features in this row deviate most from training distribution.
        Uses robust z-score (median + MAD) for skewed distributions.

        Skips binary/low-variance features where MAD = 0 would cause infinite z-scores.
        These features (like is_turnover) carry no anomaly signal individually.
        """
        median = X_train.median()
        mad = (X_train - median).abs().median()

        # Filter out features with near-zero MAD (binary, constant, low-variance)
        # MAD < 0.01 means the feature barely varies — z-score is meaningless
        meaningful_features = mad[mad > 0.01].index.tolist()

        if not meaningful_features:
            return []

        row_filtered = row[meaningful_features]
        median_filtered = median[meaningful_features]
        mad_filtered = mad[meaningful_features]

        z_scores = ((row_filtered - median_filtered) / mad_filtered).abs()
        top_3 = z_scores.nlargest(3)

        return [
            {
                'feature': str(feat),
                'value':   float(row[feat]),
                'z_score': float(z),
            }
            for feat, z in top_3.items()
        ]

    # ── Persistence ──────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        """Save model + scaler + metadata to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        artifact = {
            'model':              self.model,
            'scaler':             self.scaler,
            'feature_cols':       self.feature_cols,
            'threshold':          self.threshold,
            'dynamic_threshold':  getattr(self, '_dynamic_threshold', None),
            'contamination':      self.contamination,
            'train_metadata':     self.train_metadata,
        }
        joblib.dump(artifact, path)
        print(f"[StatDetector] Saved to {path}")

    def load(self, path: str) -> None:
        """Load model + scaler + metadata from disk."""
        artifact = joblib.load(path)
        self.model = artifact['model']
        self.scaler = artifact['scaler']
        self.feature_cols = artifact['feature_cols']
        self.threshold = artifact['threshold']
        self.contamination = artifact['contamination']
        self.train_metadata = artifact['train_metadata']
        if artifact.get('dynamic_threshold') is not None:
            self._dynamic_threshold = artifact['dynamic_threshold']
        print(f"[StatDetector] Loaded from {path}")
        print(f"  Trained on {self.train_metadata['n_train_rows']:,} rows")
        print(f"  Features: {len(self.feature_cols)}")


# ── CLI for standalone training and testing ──────────────────────────────────
def main():
    """
    Two-step demo:
      Step A: Train detector on 2023-2024 features, save artifact
      Step B: Score 2025 features, see how many are flagged as anomalies
    """
    import sys

    PROJECT_DIR = '/Users/ye/work/portfolio/swiss-afrr-spike-prediction'
    TRAIN_PATH  = os.path.join(PROJECT_DIR, 'data', 'processed',
                               'features_train_2023_2024.csv')
    VAL_PATH    = os.path.join(PROJECT_DIR, 'data', 'processed',
                               'features_val_2025.csv')
    ARTIFACT_PATH = os.path.join(PROJECT_DIR, 'pipeline', 'monitoring',
                                  'statistical', 'artifacts',
                                  'isolation_forest.pkl')

    if not os.path.exists(TRAIN_PATH):
        print(f"Training data not found: {TRAIN_PATH}")
        sys.exit(1)

    print("=" * 70)
    print("STATISTICAL ANOMALY DETECTION — Train + Test")
    print("=" * 70)

    # Define which features to use for anomaly detection
    # Same 24 features as the XGBoost model — this is intentional:
    # we want the anomaly detector to see exactly what the model sees
    FEATURE_COLS = [
        'Unplanned_Flow', 'abs_Unplanned_Flow', 'Unplanned_Flow_rolling4',
        'DE_WindSolar_Error', 'Unplanned_Flow_FR_CH', 'abs_Unplanned_Flow_FR_CH',
        'Total_Unplanned_Flow', 'Sched_DE_CH', 'Sched_DE_CH_delta', 'Sched_FR_CH',
        'Unplanned_x_Sched', 'rolling_p90_threshold', 'CH_Load_Forecast',
        'CH_Pump_Gen', 'DA_Price_DE', 'DA_Price_CH', 'DA_Price_Spread_DE_CH',
        'price_delta_lag1', 'price_delta_lag4', 'price_delta_lag96',
        'price_vs_threshold_lag1', 'price_vs_threshold_lag4', 'minute',
        'is_turnover',
    ]

    # ── Step A: Train ─────────────────────────────────────────────────────────
    print("\n[Step A] Training IsolationForest on 2023-2024 data")
    print("-" * 70)

    df_train = pd.read_csv(TRAIN_PATH)
    print(f"Training rows: {len(df_train):,}")

    detector = StatisticalAnomalyDetector(
        contamination=0.05,
        threshold=-0.05,
        feature_cols=FEATURE_COLS,
    )
    detector.fit(df_train)
    detector.save(ARTIFACT_PATH)

    # ── Step B: Score 2025 data ───────────────────────────────────────────────
    if not os.path.exists(VAL_PATH):
        print(f"\n[Step B] Validation data not found at {VAL_PATH} — skipping")
        return

    print("\n" + "=" * 70)
    print("[Step B] Scoring 2025 validation data")
    print("-" * 70)

    df_val = pd.read_csv(VAL_PATH)
    print(f"Validation rows: {len(df_val):,}")

    result = detector.score(df_val)

    print(f"\nResults:")
    print(f"  Total rows scored:    {result.n_rows:,}")
    print(f"  Flagged as anomaly:   {result.n_anomalies:,} ({result.anomaly_rate*100:.1f}%)")
    print(f"  Score range:          {result.score_min:.3f} to {result.score_max:.3f}")
    print(f"  Score mean:           {result.score_mean:.3f}")
    print(f"  Threshold used:       {result.threshold}")

    # Compare to training baseline
    print(f"\nComparison vs training:")
    train_anomaly_rate = detector.train_metadata['training_anomaly_rate']
    print(f"  Training anomaly rate (baseline): {train_anomaly_rate*100:.1f}%")
    print(f"  2025 anomaly rate:                {result.anomaly_rate*100:.1f}%")

    if result.anomaly_rate > train_anomaly_rate * 2:
        print(f"  ⚠️  2025 has 2x+ more anomalies than training — significant drift")
    elif result.anomaly_rate > train_anomaly_rate * 1.5:
        print(f"  ⚠️  2025 has 1.5x+ more anomalies than training — moderate drift")
    else:
        print(f"  ✅ 2025 anomaly rate similar to training — no major drift")

    # Show top anomalies
    print(f"\nTop 5 most anomalous rows in 2025:")
    print("-" * 70)
    for sample in result.sample_anomalies:
        print(f"\nRow {sample['index']}: score = {sample['score']:.3f}")
        print(f"  Most unusual features (z-score from training median):")
        for feat in sample['top_features']:
            print(f"    {feat['feature']:<30} value={feat['value']:>10.2f}  z={feat['z_score']:.2f}")


if __name__ == "__main__":
    main()
