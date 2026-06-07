"""
================================================================================
domain_anomaly_detector.py — Domain Anomaly Detector
================================================================================
PURPOSE:
    Apply physical and business knowledge rules from domain_rules.yaml
    to a DataFrame. Detect violations of physical laws, capacity limits,
    sign conventions, and derived field consistency.

WHAT THIS CATCHES:
    - Values outside physical bounds (capacity, price floors)
    - Wrong signs (negative volumes, bidirectional flows)
    - Broken derived fields (spread != DE - CH)
    - Physically impossible combinations

WHAT THIS DOES NOT CATCH:
    - Statistical outliers (use isolation_forest.py)
    - Time series gaps (use logical_anomaly_detector.py)
    - Cross-source disagreements (use logical_anomaly_detector.py)

USAGE:
    detector = DomainAnomalyDetector('domain_rules.yaml')
    violations = detector.check(df)
    # violations is a DataFrame with one row per violation
================================================================================
"""

import os
import yaml
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ── Result types ──────────────────────────────────────────────────────────────
@dataclass
class Violation:
    """One detected anomaly."""
    rule_type: str           # 'range', 'non_negative', 'not_simultaneous', etc.
    severity: str            # 'error', 'warning', 'info'
    columns: list            # column(s) involved
    n_violations: int        # how many rows failed
    sample_values: list      # up to 5 sample failing values
    sample_indices: list     # up to 5 sample failing row indices
    reason: str              # human-readable explanation
    rule_config: dict = field(default_factory=dict)  # original rule for debugging


# ── Main detector class ───────────────────────────────────────────────────────
class DomainAnomalyDetector:
    """
    Applies domain rules from YAML config to a DataFrame.

    Rules are grouped into:
        swissgrid_rules    — for Swissgrid raw columns
        entsoe_rules       — for ENTSO-E raw columns
        feature_rules      — for engineered features
        consistency_rules  — for derived field checks
    """

    def __init__(self, rules_path: str):
        if not os.path.exists(rules_path):
            raise FileNotFoundError(f"Rules file not found: {rules_path}")

        with open(rules_path) as f:
            self.rules = yaml.safe_load(f)

        print(f"[DomainDetector] Loaded rules from {rules_path}")
        for group, rules in self.rules.items():
            print(f"  {group}: {len(rules)} rules")

    # ── Individual rule checkers ─────────────────────────────────────────────

    def _check_range(self, df: pd.DataFrame, rule: dict) -> Optional[Violation]:
        """value must be between min and max."""
        col = rule['column']
        if col not in df.columns:
            return None  # silently skip — column may not be in this DataFrame

        mask = (df[col] < rule['min']) | (df[col] > rule['max'])
        if not mask.any():
            return None

        failing = df.loc[mask, col]
        return Violation(
            rule_type='range',
            severity=rule['severity'],
            columns=[col],
            n_violations=int(mask.sum()),
            sample_values=failing.head(5).tolist(),
            sample_indices=failing.head(5).index.tolist(),
            reason=f"{rule['reason']} (range: [{rule['min']}, {rule['max']}])",
            rule_config=rule,
        )

    def _check_non_negative(self, df: pd.DataFrame, rule: dict) -> Optional[Violation]:
        """value must be >= 0."""
        col = rule['column']
        if col not in df.columns:
            return None

        mask = df[col] < 0
        if not mask.any():
            return None

        failing = df.loc[mask, col]
        return Violation(
            rule_type='non_negative',
            severity=rule['severity'],
            columns=[col],
            n_violations=int(mask.sum()),
            sample_values=failing.head(5).tolist(),
            sample_indices=failing.head(5).index.tolist(),
            reason=rule['reason'],
            rule_config=rule,
        )

    def _check_non_positive(self, df: pd.DataFrame, rule: dict) -> Optional[Violation]:
        """value must be <= 0."""
        col = rule['column']
        if col not in df.columns:
            return None

        mask = df[col] > 0
        if not mask.any():
            return None

        failing = df.loc[mask, col]
        return Violation(
            rule_type='non_positive',
            severity=rule['severity'],
            columns=[col],
            n_violations=int(mask.sum()),
            sample_values=failing.head(5).tolist(),
            sample_indices=failing.head(5).index.tolist(),
            reason=rule['reason'],
            rule_config=rule,
        )

    def _check_not_simultaneous(self, df: pd.DataFrame, rule: dict) -> Optional[Violation]:
        """two columns must not both be > tolerance at the same time."""
        cols = rule['columns']
        tolerance = rule.get('tolerance', 0)

        if not all(c in df.columns for c in cols):
            return None

        # Both columns must exceed tolerance simultaneously
        mask = (df[cols[0]] > tolerance) & (df[cols[1]] > tolerance)
        if not mask.any():
            return None

        return Violation(
            rule_type='not_simultaneous',
            severity=rule['severity'],
            columns=cols,
            n_violations=int(mask.sum()),
            sample_values=df.loc[mask, cols].head(5).values.tolist(),
            sample_indices=df.loc[mask].head(5).index.tolist(),
            reason=f"{rule['reason']} (tolerance: {tolerance} MW)",
            rule_config=rule,
        )

    def _check_derived_field(self, df: pd.DataFrame, rule: dict) -> Optional[Violation]:
        """derived field must match its formula within tolerance."""
        derived = rule['derived']
        formula = rule['formula']
        tolerance = rule.get('tolerance', 0.01)

        if derived not in df.columns:
            return None

        # Evaluate the formula safely using df.eval
        try:
            expected = df.eval(formula)
        except Exception as e:
            return Violation(
                rule_type='derived_field',
                severity='warning',
                columns=[derived],
                n_violations=0,
                sample_values=[],
                sample_indices=[],
                reason=f"Could not evaluate formula '{formula}': {e}",
                rule_config=rule,
            )

        actual = df[derived]
        diff = (expected - actual).abs()
        mask = diff > tolerance

        if not mask.any():
            return None

        return Violation(
            rule_type='derived_field',
            severity=rule['severity'],
            columns=[derived],
            n_violations=int(mask.sum()),
            sample_values=[
                {'expected': float(expected.iloc[i]), 'actual': float(actual.iloc[i])}
                for i in df.loc[mask].head(5).index.tolist()
            ],
            sample_indices=df.loc[mask].head(5).index.tolist(),
            reason=f"{rule['reason']} (formula: {formula}, tolerance: {tolerance})",
            rule_config=rule,
        )

    # ── Dispatcher ────────────────────────────────────────────────────────────

    def _check_rule(self, df: pd.DataFrame, rule: dict) -> Optional[Violation]:
        """Route a rule to the right checker based on its type."""
        rule_type = rule['type']
        checker = {
            'range':            self._check_range,
            'non_negative':     self._check_non_negative,
            'non_positive':     self._check_non_positive,
            'not_simultaneous': self._check_not_simultaneous,
            'derived_field':    self._check_derived_field,
        }.get(rule_type)

        if checker is None:
            print(f"[DomainDetector] Unknown rule type: {rule_type}")
            return None

        return checker(df, rule)

    # ── Public API ────────────────────────────────────────────────────────────

    def check(self, df: pd.DataFrame, groups: Optional[list] = None) -> pd.DataFrame:
        """
        Apply all rules in the specified groups.

        Args:
            df: DataFrame to check
            groups: list of rule groups to apply (None = apply all)

        Returns:
            DataFrame with one row per violation, columns:
                rule_type, severity, columns, n_violations,
                sample_values, sample_indices, reason
        """
        if groups is None:
            groups = list(self.rules.keys())

        violations = []
        for group in groups:
            if group not in self.rules:
                print(f"[DomainDetector] Unknown group: {group}")
                continue

            for rule in self.rules[group]:
                violation = self._check_rule(df, rule)
                if violation is not None:
                    violations.append(violation)

        if not violations:
            return pd.DataFrame()

        return pd.DataFrame([v.__dict__ for v in violations])

    def has_errors(self, violations_df: pd.DataFrame) -> bool:
        """Check if any error-severity violations exist."""
        if violations_df.empty:
            return False
        return (violations_df['severity'] == 'error').any()

    def summary(self, violations_df: pd.DataFrame) -> dict:
        """Return a quick summary of violations by severity."""
        if violations_df.empty:
            return {'total': 0, 'errors': 0, 'warnings': 0, 'info': 0}

        return {
            'total':    len(violations_df),
            'errors':   int((violations_df['severity'] == 'error').sum()),
            'warnings': int((violations_df['severity'] == 'warning').sum()),
            'info':     int((violations_df['severity'] == 'info').sum()),
        }


# ── CLI for standalone testing ────────────────────────────────────────────────
def main():
    """
    Test the detector against the existing training data.
    Run from project root:
        python3 pipeline/monitoring/domain/domain_anomaly_detector.py
    """
    import sys

    PROJECT_DIR = '/Users/ye/work/portfolio/swiss-afrr-spike-prediction'
    RULES_PATH  = os.path.join(PROJECT_DIR, 'pipeline', 'monitoring',
                               'domain', 'domain_rules.yaml')
    DATA_PATH   = os.path.join(PROJECT_DIR, 'data', 'processed',
                               'features_train_2023_2024.csv')

    if not os.path.exists(DATA_PATH):
        print(f"Data file not found: {DATA_PATH}")
        sys.exit(1)

    print("=" * 70)
    print("DOMAIN ANOMALY DETECTION — Test Run")
    print("=" * 70)

    # Load data
    print(f"\nLoading: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Rows: {len(df):,}, Columns: {len(df.columns)}")

    # Run detector
    print(f"\nApplying rules from: {RULES_PATH}")
    detector = DomainAnomalyDetector(RULES_PATH)
    violations = detector.check(df)

    # Report
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    summary = detector.summary(violations)
    print(f"Total violations: {summary['total']}")
    print(f"  Errors:   {summary['errors']}")
    print(f"  Warnings: {summary['warnings']}")
    print(f"  Info:     {summary['info']}")

    if violations.empty:
        print("\n✅ No violations detected")
        return

    print("\n" + "-" * 70)
    print("VIOLATION DETAILS")
    print("-" * 70)
    for _, v in violations.iterrows():
        icon = {'error': '❌', 'warning': '⚠️', 'info': 'ℹ️'}[v['severity']]
        print(f"\n{icon} [{v['severity'].upper()}] {v['rule_type']} on {v['columns']}")
        print(f"   {v['n_violations']:,} violations")
        print(f"   Reason: {v['reason']}")
        print(f"   Sample values: {v['sample_values'][:3]}")


if __name__ == "__main__":
    main()
