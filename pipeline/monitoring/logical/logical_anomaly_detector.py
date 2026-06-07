"""
================================================================================
logical_anomaly_detector.py — Logical Anomaly Detector
================================================================================
PURPOSE:
    Detect anomalies in RELATIONSHIPS between values, not in values themselves.
    Even when every individual value passes domain rules, the data can still
    be wrong — missing intervals, broken derived fields, orphaned outputs,
    cross-source disagreements, sudden unnatural changes.

WHAT THIS CATCHES:
    1. Time series continuity     — missing intervals, duplicate timestamps
    2. Pipeline output consistency — features without raw, predictions without
                                     features, orphaned MLflow runs
    3. Cross-source agreement      — same quantity, multiple sources, must agree
    4. Schedule stability          — Sched_DE_CH should not jump suddenly
    5. Feature freshness           — data is for the right date
    6. Row count consistency       — 96 rows per day for 15-min data

WHAT THIS DOES NOT CATCH:
    - Out-of-bounds values (use domain_anomaly_detector.py)
    - Statistical outliers (use statistical_anomaly_detector.py)

USAGE:
    detector = LogicalAnomalyDetector()
    issues = detector.check_all(df, expected_date='2025-12-15')
================================================================================
"""

import os
import boto3
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import timedelta


# ── Result type (shared structure with domain detector) ──────────────────────
@dataclass
class LogicalIssue:
    check_type: str          # which check fired ('time_continuity', etc.)
    severity: str            # 'error', 'warning', 'info'
    n_issues: int            # how many problems found
    sample_details: list     # up to 5 examples
    reason: str              # human-readable explanation
    metadata: dict = field(default_factory=dict)


# ── Main detector class ───────────────────────────────────────────────────────
class LogicalAnomalyDetector:
    """
    Six independent logical checks, each can be called individually
    or all together via check_all().
    """

    def __init__(
        self,
        expected_freq: str = '15min',
        rows_per_day: int = 96,
        rows_per_day_tolerance: int = 4,  # DST days have 92 or 100
        schedule_jump_threshold: float = 1500.0,  # MW
    ):
        self.expected_freq = expected_freq
        self.rows_per_day = rows_per_day
        self.rows_per_day_tolerance = rows_per_day_tolerance
        self.schedule_jump_threshold = schedule_jump_threshold

    # ── Check 1: Time series continuity ───────────────────────────────────────
    def check_time_continuity(self, df: pd.DataFrame) -> Optional[LogicalIssue]:
        """
        Checks:
          - No missing 15-min intervals
          - No duplicate timestamps (except DST transitions)
          - Timestamps are monotonically non-decreasing
        """
        if 'timestamp' not in df.columns:
            return None

        ts = pd.to_datetime(df['timestamp'])

        issues = []
        sample_details = []

        # Sub-check 1a: monotonically non-decreasing
        if not ts.is_monotonic_increasing:
            non_monotonic = (ts.diff() < pd.Timedelta(0)).sum()
            issues.append(f"{non_monotonic} timestamps go backwards")
            sample_details.append({
                'issue': 'non_monotonic',
                'count': int(non_monotonic),
            })

        # Sub-check 1b: missing intervals
        expected = pd.date_range(ts.min(), ts.max(), freq=self.expected_freq)
        missing = expected.difference(ts)
        # Allow up to 4 missing intervals per day for DST (1h spring forward)
        if len(missing) > self.rows_per_day_tolerance:
            issues.append(f"{len(missing)} missing intervals")
            sample_details.append({
                'issue': 'missing_intervals',
                'count': len(missing),
                'first_5': [str(t) for t in missing[:5]],
            })

        # Sub-check 1c: duplicate timestamps
        # Expected DST fall-back duplicates: 4 timestamps × 4 duplicates each
        # = 16 per autumn (per data source). With 2 sources merged: 32 per autumn.
        # Allow up to 40 to cover legitimate DST patterns.
        duplicates = ts[ts.duplicated()]
        DST_DUPLICATE_TOLERANCE = 40
        if len(duplicates) > DST_DUPLICATE_TOLERANCE:
            issues.append(f"{len(duplicates)} duplicate timestamps")
            sample_details.append({
                'issue': 'duplicates',
                'count': len(duplicates),
                'first_5': duplicates.head(5).astype(str).tolist(),
                'note': 'DST fall-back creates ~16-32 expected duplicates per year',
            })

        if not issues:
            return None

        return LogicalIssue(
            check_type='time_continuity',
            severity='error',
            n_issues=sum(d['count'] if isinstance(d, dict) and 'count' in d else 1
                         for d in sample_details),
            sample_details=sample_details,
            reason="; ".join(issues),
        )

    # ── Check 2: Pipeline output consistency ─────────────────────────────────
    def check_pipeline_consistency(
        self,
        date_str: str,
        s3_bucket: str = 'energy-pipeline',
        s3_endpoint: str = 'http://localhost:4566',
    ) -> Optional[LogicalIssue]:
        """
        Verifies that pipeline outputs exist in the expected order:
            raw exists → features exists → predictions exists

        If predictions exist but features don't, the predictions are orphaned.
        """
        s3 = boto3.client(
            's3',
            endpoint_url=s3_endpoint,
            aws_access_key_id='test',
            aws_secret_access_key='test',
            region_name='us-east-1',
        )

        keys = {
            'raw':         f"raw/{date_str}/entsoe.csv",
            'features':    f"features/{date_str}/features.parquet",
            'predictions': f"predictions/{date_str}/predictions.parquet",
        }

        existence = {}
        for stage, key in keys.items():
            try:
                s3.head_object(Bucket=s3_bucket, Key=key)
                existence[stage] = True
            except Exception:
                existence[stage] = False

        # Detect inconsistent state — downstream exists but upstream doesn't
        problems = []
        if existence['features'] and not existence['raw']:
            problems.append("features exist but raw data missing")
        if existence['predictions'] and not existence['features']:
            problems.append("predictions exist but features missing")
        if existence['predictions'] and not existence['raw']:
            problems.append("predictions exist but raw data missing")

        if not problems:
            return None

        return LogicalIssue(
            check_type='pipeline_consistency',
            severity='error',
            n_issues=len(problems),
            sample_details=[{
                'date': date_str,
                'existence': existence,
                'problems': problems,
            }],
            reason="; ".join(problems),
            metadata={'date': date_str},
        )

    # ── Check 3: Cross-source agreement ──────────────────────────────────────
    def check_cross_source_agreement(
        self,
        df: pd.DataFrame,
        tolerance_pct: float = 5.0,
    ) -> Optional[LogicalIssue]:
        """
        Checks that derived/secondary representations of the same quantity
        agree with their primary source within tolerance.

        In our data, the most useful cross-source check is:
            DA_Price_Spread_DE_CH should equal DA_Price_DE - DA_Price_CH

        This catches cases where one of the prices was updated but the
        spread was computed from a stale value.
        """
        if not all(c in df.columns for c in
                   ['DA_Price_DE', 'DA_Price_CH', 'DA_Price_Spread_DE_CH']):
            return None

        expected_spread = df['DA_Price_DE'] - df['DA_Price_CH']
        actual_spread   = df['DA_Price_Spread_DE_CH']

        # Use tolerance as percentage of typical price magnitude
        # to handle small-value vs large-value cases
        magnitude = (df['DA_Price_DE'].abs() + df['DA_Price_CH'].abs()) / 2
        tolerance = magnitude * (tolerance_pct / 100)
        tolerance = tolerance.clip(lower=0.1)  # minimum 0.1 EUR/MWh tolerance

        diff = (expected_spread - actual_spread).abs()
        mask = diff > tolerance

        if not mask.any():
            return None

        sample = df.loc[mask, [
            'timestamp', 'DA_Price_DE', 'DA_Price_CH', 'DA_Price_Spread_DE_CH'
        ]].head(5)

        return LogicalIssue(
            check_type='cross_source_agreement',
            severity='warning',
            n_issues=int(mask.sum()),
            sample_details=sample.to_dict('records'),
            reason=f"DA_Price_Spread_DE_CH disagrees with DA_Price_DE - DA_Price_CH "
                   f"in {mask.sum():,} rows (tolerance: {tolerance_pct}%)",
        )

    # ── Check 4: Schedule stability ──────────────────────────────────────────
    def check_schedule_stability(self, df: pd.DataFrame) -> Optional[LogicalIssue]:
        """
        Day-ahead schedules are HOURLY products, so they are flat for 4
        consecutive 15-min intervals and can legitimately jump at hour
        boundaries (minute 0).

        Two thresholds applied:
          - Within hour (any jump): suspicious — schedule should be flat
          - At hour boundary: only flag if abnormally large

        Catches:
          - Mid-fetch updates (within-hour jumps)
          - API returning intraday update mixed with day-ahead
          - Extreme market events (very large hour-boundary jumps)
        """
        if 'timestamp' not in df.columns:
            return None

        sched_cols = [c for c in df.columns if c.startswith('Sched_')]
        if not sched_cols:
            return None

        ts = pd.to_datetime(df['timestamp'])
        is_hour_boundary = ts.dt.minute == 0
        # within-hour rows are everywhere except hour boundary
        within_hour_mask = ~is_hour_boundary

        all_jumps = []
        for col in sched_cols:
            jumps = df[col].diff().abs()

            # Sub-check 4a: within-hour jumps — should be ~0
            within_hour_jumps = (jumps > 100) & within_hour_mask
            if within_hour_jumps.any():
                all_jumps.append({
                    'column': col,
                    'context': 'within_hour',
                    'count': int(within_hour_jumps.sum()),
                    'max_jump_mw': float(jumps[within_hour_mask].max() if within_hour_mask.any() else 0),
                    'reason': 'schedule should be flat within an hour',
                })

            # Sub-check 4b: hour-boundary jumps — abnormal only if very large
            boundary_jumps = (jumps > self.schedule_jump_threshold) & is_hour_boundary
            if boundary_jumps.any():
                all_jumps.append({
                    'column': col,
                    'context': 'hour_boundary',
                    'count': int(boundary_jumps.sum()),
                    'max_jump_mw': float(jumps[is_hour_boundary].max()),
                    'reason': f'jump > {self.schedule_jump_threshold} MW at hour boundary',
                })

        if not all_jumps:
            return None

        return LogicalIssue(
            check_type='schedule_stability',
            severity='warning',
            n_issues=sum(j['count'] for j in all_jumps),
            sample_details=all_jumps,
            reason="Scheduled flows show suspicious jumps "
                   "(within-hour > 100 MW or hour-boundary > "
                   f"{self.schedule_jump_threshold} MW)",
        )

    # ── Check 5: Feature freshness ───────────────────────────────────────────
    def check_freshness(
        self,
        df: pd.DataFrame,
        expected_date: str,
        max_age_hours: int = 48,
    ) -> Optional[LogicalIssue]:
        """
        Verifies the data is for the expected date.
        Catches cases where stale data is being processed as fresh.
        """
        if 'timestamp' not in df.columns:
            return None

        ts = pd.to_datetime(df['timestamp'])
        expected_dt = pd.Timestamp(expected_date).date()

        # Check that at least some rows are for the expected date
        rows_on_date = (ts.dt.date == expected_dt).sum()
        if rows_on_date == 0:
            return LogicalIssue(
                check_type='freshness',
                severity='error',
                n_issues=1,
                sample_details=[{
                    'expected_date': expected_date,
                    'min_date': str(ts.min().date()),
                    'max_date': str(ts.max().date()),
                    'rows_on_expected_date': 0,
                }],
                reason=f"No rows found for expected date {expected_date}. "
                       f"Data covers {ts.min().date()} to {ts.max().date()}",
            )

        # Check that the most recent timestamp isn't stale
        age_hours = (pd.Timestamp(expected_date) - ts.max()).total_seconds() / 3600
        if age_hours > max_age_hours:
            return LogicalIssue(
                check_type='freshness',
                severity='warning',
                n_issues=1,
                sample_details=[{
                    'expected_date': expected_date,
                    'last_timestamp': str(ts.max()),
                    'age_hours': round(age_hours, 1),
                }],
                reason=f"Data is stale — last timestamp is {age_hours:.1f}h "
                       f"before expected date",
            )

        return None

    # ── Check 6: Row count consistency ───────────────────────────────────────
    def check_row_count(
        self,
        df: pd.DataFrame,
        expected_date: Optional[str] = None,
    ) -> Optional[LogicalIssue]:
        """
        Daily DataFrames should have ~96 rows (24h × 4 quarter-hours).
        DST days can have 92 or 100.
        """
        if expected_date is None:
            return None

        if 'timestamp' not in df.columns:
            return None

        ts = pd.to_datetime(df['timestamp'])
        rows_on_date = (ts.dt.date == pd.Timestamp(expected_date).date()).sum()

        diff = abs(rows_on_date - self.rows_per_day)
        if diff <= self.rows_per_day_tolerance:
            return None

        return LogicalIssue(
            check_type='row_count',
            severity='error' if diff > 10 else 'warning',
            n_issues=1,
            sample_details=[{
                'expected_date': expected_date,
                'rows': int(rows_on_date),
                'expected': self.rows_per_day,
                'tolerance': self.rows_per_day_tolerance,
            }],
            reason=f"Expected {self.rows_per_day}±{self.rows_per_day_tolerance} "
                   f"rows for {expected_date}, got {rows_on_date}",
        )

    # ── Public API: run everything ───────────────────────────────────────────
    def check_all(
        self,
        df: pd.DataFrame,
        expected_date: Optional[str] = None,
        check_pipeline_outputs: bool = False,
        s3_bucket: str = 'energy-pipeline',
    ) -> pd.DataFrame:
        """
        Run all 6 logical checks. Returns a DataFrame of issues.

        Args:
            df: data to check (must have 'timestamp' column)
            expected_date: 'YYYY-MM-DD' for freshness/row count/pipeline checks
            check_pipeline_outputs: if True, also check S3 file consistency
                                    (only useful in pipeline context)
        """
        issues = []

        # 1. Time continuity
        issue = self.check_time_continuity(df)
        if issue:
            issues.append(issue)

        # 2. Pipeline consistency (only if requested)
        if check_pipeline_outputs and expected_date:
            issue = self.check_pipeline_consistency(expected_date, s3_bucket)
            if issue:
                issues.append(issue)

        # 3. Cross-source agreement
        issue = self.check_cross_source_agreement(df)
        if issue:
            issues.append(issue)

        # 4. Schedule stability
        issue = self.check_schedule_stability(df)
        if issue:
            issues.append(issue)

        # 5. Freshness
        if expected_date:
            issue = self.check_freshness(df, expected_date)
            if issue:
                issues.append(issue)

        # 6. Row count
        if expected_date:
            issue = self.check_row_count(df, expected_date)
            if issue:
                issues.append(issue)

        if not issues:
            return pd.DataFrame()

        return pd.DataFrame([i.__dict__ for i in issues])

    def has_errors(self, issues_df: pd.DataFrame) -> bool:
        if issues_df.empty:
            return False
        return (issues_df['severity'] == 'error').any()

    def summary(self, issues_df: pd.DataFrame) -> dict:
        if issues_df.empty:
            return {'total': 0, 'errors': 0, 'warnings': 0, 'info': 0}
        return {
            'total':    len(issues_df),
            'errors':   int((issues_df['severity'] == 'error').sum()),
            'warnings': int((issues_df['severity'] == 'warning').sum()),
            'info':     int((issues_df['severity'] == 'info').sum()),
        }


# ── CLI for standalone testing ────────────────────────────────────────────────
def main():
    import sys

    PROJECT_DIR = '/Users/ye/work/portfolio/swiss-afrr-spike-prediction'
    DATA_PATH   = os.path.join(PROJECT_DIR, 'data', 'processed',
                               'features_train_2023_2024.csv')

    if not os.path.exists(DATA_PATH):
        print(f"Data file not found: {DATA_PATH}")
        sys.exit(1)

    print("=" * 70)
    print("LOGICAL ANOMALY DETECTION — Test Run")
    print("=" * 70)

    print(f"\nLoading: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Rows: {len(df):,}, Columns: {len(df.columns)}")

    detector = LogicalAnomalyDetector()
    issues = detector.check_all(df)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    summary = detector.summary(issues)
    print(f"Total issues: {summary['total']}")
    print(f"  Errors:   {summary['errors']}")
    print(f"  Warnings: {summary['warnings']}")
    print(f"  Info:     {summary['info']}")

    if issues.empty:
        print("\n✅ No logical issues detected")
        return

    print("\n" + "-" * 70)
    print("ISSUE DETAILS")
    print("-" * 70)
    for _, i in issues.iterrows():
        icon = {'error': '❌', 'warning': '⚠️', 'info': 'ℹ️'}[i['severity']]
        print(f"\n{icon} [{i['severity'].upper()}] {i['check_type']}")
        print(f"   Issues: {i['n_issues']:,}")
        print(f"   Reason: {i['reason']}")
        if i['sample_details']:
            print(f"   Sample: {i['sample_details'][0]}")


if __name__ == "__main__":
    main()
