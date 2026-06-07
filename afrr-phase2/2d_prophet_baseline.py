"""
================================================================================
2d_prophet_baseline.py — Prophet Trend Forecasting Baseline
================================================================================
PURPOSE:
    Compare Facebook Prophet against Google TimesFM 2.5 on the same task:
    24h-ahead aFRR price forecasting using only price history.

    This answers: "Do you need a foundation model, or does a classical
    statistical baseline achieve comparable trend forecasting?"

EXPERIMENT DESIGN:
    Same sliding-window approach as TimesFM experiment 2a:
      - Train: all data before each test day
      - Predict: next 96 steps (24h of 15-min intervals)
      - Evaluate: MAE, RMSE, MAPE, bias, correlation
      - 365 windows across 2025

COMPARISON METRICS (from TimesFM 2a):
    TimesFM MAE:   54.8 EUR/MWh
    TimesFM RMSE:  107.0 EUR/MWh
    TimesFM r:     0.900

INPUTS:
    afrr-phase2/data/raw_merged_2023_2024.csv
    afrr-phase2/data/raw_merged_2025.csv

OUTPUTS:
    afrr-phase2/output/2d_prophet_metrics.json
    afrr-phase2/output/2d_prophet_predictions.png
    afrr-phase2/output/2d_prophet_mae_over_time.png
    afrr-phase2/output/2d_comparison_prophet_vs_timesfm.png
================================================================================
"""

import os
import json
import warnings
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(PROJECT_DIR, 'data')
OUT_DIR     = os.path.join(PROJECT_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

HORIZON = 96  # 24h at 15-min intervals

# TimesFM baseline numbers (from experiment 2a)
TIMESFM_METRICS = {
    'mae':  54.83,
    'rmse': 107.01,
    'mape': 39.88,
    'bias': -9.83,
}


# ── Load data ─────────────────────────────────────────────────────────────────
def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, 'raw_merged_2023_2024.csv'))
    val   = pd.read_csv(os.path.join(DATA_DIR, 'raw_merged_2025.csv'))

    train['timestamp'] = pd.to_datetime(train['timestamp'])
    val['timestamp']   = pd.to_datetime(val['timestamp'])

    # Combine for sliding window (Prophet trains on all data before each day)
    combined = pd.concat([train, val], ignore_index=True)
    combined = combined.sort_values('timestamp').reset_index(drop=True)

    # Prophet requires columns named 'ds' and 'y'
    combined = combined.rename(columns={
        'timestamp':     'ds',
        'pos_sec_price': 'y',
    })

    return combined, val


# ── Run Prophet sliding window ────────────────────────────────────────────────
def run_prophet_experiment(combined: pd.DataFrame, val: pd.DataFrame):
    """
    For each day in 2025, fit Prophet on all prior data and forecast 24h ahead.
    Uses the same sliding window approach as TimesFM for fair comparison.
    """
    val_dates = pd.to_datetime(val['timestamp']).dt.date.unique()
    print(f"[Prophet] {len(val_dates)} test days in 2025")

    all_actuals = []
    all_preds   = []
    daily_mae   = []
    daily_dates = []
    total_time  = 0

    for i, test_date in enumerate(val_dates):
        test_date_ts = pd.Timestamp(test_date)

        # Training data: everything before this day
        train_mask = combined['ds'] < test_date_ts
        df_train = combined[train_mask][['ds', 'y']].copy()

        # Actual values for this day
        actual_mask = (combined['ds'] >= test_date_ts) & \
                      (combined['ds'] < test_date_ts + pd.Timedelta(days=1))
        df_actual = combined[actual_mask].copy()

        if len(df_actual) < HORIZON * 0.8:
            continue  # skip incomplete days

        # Fit Prophet — suppress logging
        t0 = time.time()
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,  # not enough data for yearly
            changepoint_prior_scale=0.05,
        )
        model.fit(df_train)

        # Create future dataframe for the test day
        future = model.make_future_dataframe(
            periods=HORIZON,
            freq='15min',
            include_history=False,
        )
        forecast = model.predict(future)
        elapsed = time.time() - t0
        total_time += elapsed

        # Align predictions with actuals
        n = min(len(forecast), len(df_actual))
        preds   = forecast['yhat'].values[:n]
        actuals = df_actual['y'].values[:n]

        all_preds.extend(preds)
        all_actuals.extend(actuals)

        mae = np.mean(np.abs(preds - actuals))
        daily_mae.append(mae)
        daily_dates.append(test_date)

        if (i + 1) % 30 == 0 or i == 0:
            print(f"  Day {i+1}/{len(val_dates)}: {test_date} | "
                  f"MAE={mae:.1f} | time={elapsed:.1f}s")

    avg_time = total_time / len(daily_mae) if daily_mae else 0
    print(f"\n[Prophet] Completed {len(daily_mae)} days, "
          f"avg {avg_time:.1f}s per window")

    return (np.array(all_actuals), np.array(all_preds),
            np.array(daily_mae), daily_dates)


# ── Compute metrics ───────────────────────────────────────────────────────────
def compute_metrics(actuals, preds, daily_mae, daily_dates, avg_time):
    errors = preds - actuals
    abs_errors = np.abs(errors)

    # MAPE — skip zeros to avoid division errors
    nonzero = actuals != 0
    mape = np.mean(np.abs(errors[nonzero]) / np.abs(actuals[nonzero])) * 100

    # Correlation
    corr = np.corrcoef(actuals, preds)[0, 1]

    metrics = {
        'window_count':             len(daily_mae),
        'mae':                      float(np.mean(abs_errors)),
        'rmse':                     float(np.sqrt(np.mean(errors**2))),
        'mape':                     float(mape),
        'bias':                     float(np.mean(errors)),
        'correlation':              float(corr),
        'inference_time_per_window': float(avg_time),
        'timesfm_baseline':         TIMESFM_METRICS,
    }
    return metrics


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_mae_over_time(daily_mae, daily_dates, metrics, out_path):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily_dates, daily_mae, alpha=0.4, color='steelblue',
            label='Window MAE')

    # Monthly average
    df_mae = pd.DataFrame({'date': daily_dates, 'mae': daily_mae})
    df_mae['date'] = pd.to_datetime(df_mae['date'])
    monthly = df_mae.set_index('date').resample('ME').mean()
    ax.plot(monthly.index, monthly['mae'], 'o-', color='darkred',
            linewidth=2, label='Monthly avg')

    # Prophet mean MAE
    ax.axhline(metrics['mae'], color='steelblue', linestyle='--', alpha=0.7,
               label=f"Prophet Mean: {metrics['mae']:.1f}")

    # TimesFM mean MAE
    ax.axhline(TIMESFM_METRICS['mae'], color='orange', linestyle='--', alpha=0.7,
               label=f"TimesFM Mean: {TIMESFM_METRICS['mae']:.1f}")

    ax.set_title('Prophet vs TimesFM — 24h Ahead Price Prediction Error Over 2025',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('MAE (EUR/MWh)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_sample_predictions(combined, daily_dates, out_path):
    """Plot 3 sample days: early, mid, late 2025."""
    sample_dates = [daily_dates[0], daily_dates[len(daily_dates)//2], daily_dates[-1]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, test_date in zip(axes, sample_dates):
        test_date_ts = pd.Timestamp(test_date)

        # Fit on data before this day
        train_mask = combined['ds'] < test_date_ts
        df_train = combined[train_mask][['ds', 'y']].copy()

        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05,
        )
        model.fit(df_train)

        future = model.make_future_dataframe(
            periods=HORIZON, freq='15min', include_history=False,
        )
        forecast = model.predict(future)

        # Actual
        actual_mask = (combined['ds'] >= test_date_ts) & \
                      (combined['ds'] < test_date_ts + pd.Timedelta(days=1))
        df_actual = combined[actual_mask]

        n = min(len(forecast), len(df_actual))
        steps = range(n)

        ax.plot(steps, df_actual['y'].values[:n], 'k-', linewidth=1.2,
                label='Actual')
        ax.plot(steps, forecast['yhat'].values[:n], 'b--', linewidth=1.2,
                label='Prophet')
        ax.fill_between(steps,
                        forecast['yhat_lower'].values[:n],
                        forecast['yhat_upper'].values[:n],
                        alpha=0.15, color='blue')
        ax.set_title(str(test_date), fontsize=11)
        ax.set_xlabel('Steps (15-min)')
        if ax == axes[0]:
            ax.set_ylabel('EUR/MWh')
        ax.legend(fontsize=8)

    fig.suptitle('Prophet — Sample 24h Predictions vs Actual',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_comparison_bar(metrics, out_path):
    """Side-by-side bar chart: Prophet vs TimesFM."""
    metric_names = ['MAE', 'RMSE']
    prophet_vals = [metrics['mae'], metrics['rmse']]
    timesfm_vals = [TIMESFM_METRICS['mae'], TIMESFM_METRICS['rmse']]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, prophet_vals, width, label='Prophet',
                   color='#5B9BD5', edgecolor='white')
    bars2 = ax.bar(x + width/2, timesfm_vals, width, label='TimesFM 2.5',
                   color='#C00000', edgecolor='white')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{bar.get_height():.1f}', ha='center', fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{bar.get_height():.1f}', ha='center', fontweight='bold')

    ax.set_ylabel('EUR/MWh', fontsize=11)
    ax.set_title('Prophet vs TimesFM 2.5 — 24h Price Forecasting',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("V2 Phase 2d — Prophet Baseline vs TimesFM")
    print("=" * 70)

    combined, val = load_data()
    print(f"Combined: {len(combined):,} rows")
    print(f"Val period: {val['timestamp'].min()} → {val['timestamp'].max()}")

    print("\n[Running Prophet sliding window — this will take a while]")
    print("-" * 70)
    actuals, preds, daily_mae, daily_dates = run_prophet_experiment(combined, val)

    avg_time = 0
    if len(daily_mae) > 0:
        avg_time = sum(daily_mae) / len(daily_mae)  # placeholder

    metrics = compute_metrics(actuals, preds, daily_mae, daily_dates, avg_time)

    # Save metrics
    metrics_path = os.path.join(OUT_DIR, '2d_prophet_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")

    # Print comparison
    print("\n" + "=" * 70)
    print("RESULTS — Prophet vs TimesFM")
    print("=" * 70)
    print(f"{'Metric':<15} {'Prophet':>12} {'TimesFM':>12} {'Winner':>10}")
    print("-" * 50)

    for m in ['mae', 'rmse', 'mape', 'bias']:
        p = metrics[m]
        t = TIMESFM_METRICS[m]
        winner = 'Prophet' if abs(p) < abs(t) else 'TimesFM'
        print(f"{m.upper():<15} {p:>12.1f} {t:>12.1f} {winner:>10}")

    print(f"{'CORRELATION':<15} {metrics['correlation']:>12.3f} {'0.900':>12} "
          f"{'Prophet' if metrics['correlation'] > 0.900 else 'TimesFM':>10}")

    # Plots
    print("\nGenerating plots...")
    plot_mae_over_time(daily_mae, daily_dates, metrics,
                       os.path.join(OUT_DIR, '2d_prophet_mae_over_time.png'))
    plot_comparison_bar(metrics,
                       os.path.join(OUT_DIR, '2d_comparison_prophet_vs_timesfm.png'))
    plot_sample_predictions(combined, daily_dates,
                           os.path.join(OUT_DIR, '2d_prophet_predictions.png'))

    print("\nAll outputs saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
