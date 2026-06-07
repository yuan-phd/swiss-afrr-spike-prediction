"""
================================================================================
spike_incident_clustering.py — Cluster Spike Incidents via Sentence Embeddings
================================================================================
PURPOSE:
    Treat each spike period as an incident. Generate a natural-language
    description capturing the physical mechanism. Embed with three different
    models. Cluster with HDBSCAN. Compare which model produces the most
    interpretable taxonomy of root causes.

METHODOLOGY:
    1. Detect spike events
       - A spike row is one where pos_sec_price > rolling P90 threshold
       - Consecutive spike rows (gap < 30 min) merge into one event
       - Each event gets ONE description

    2. Generate descriptions (mechanism-focused)
       Avoid calendar features that would dominate embeddings.
       Include: cause variables, magnitude, key context only.

    3. Embed with three models
       - MiniLM   (all-MiniLM-L6-v2)    : lightweight 384-dim baseline
       - MPNet    (all-mpnet-base-v2)   : strong general 768-dim
       - BioBERT  (PubMedBERT)          : domain-specific (medical, expected
                                           to underperform — comparison is story)

    4. Cluster each embedding set
       UMAP → 10D → HDBSCAN with min_cluster_size=20

    5. Evaluate
       - Silhouette score (numeric)
       - Sample 3-5 spike descriptions per cluster (qualitative)
       - Manual cluster interpretation

INPUTS:
    data/processed/features_train_2023_2024.csv
    data/processed/features_val_2025.csv

OUTPUTS:
    pipeline/monitoring/clustering/results/
        cluster_assignments_<model>.csv
        cluster_samples_<model>.txt
        umap_2d_<model>.png
        comparison_summary.json
================================================================================
"""

import os
import json
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────────
PROJECT_DIR = '/Users/ye/work/portfolio/swiss-afrr-spike-prediction'
DATA_DIR    = os.path.join(PROJECT_DIR, 'data', 'processed')
OUT_DIR     = os.path.join(PROJECT_DIR, 'pipeline', 'monitoring',
                            'clustering', 'results')
os.makedirs(OUT_DIR, exist_ok=True)

# Three embedding models to compare
EMBEDDING_MODELS = {
    'minilm':  'sentence-transformers/all-MiniLM-L6-v2',
    'mpnet':   'sentence-transformers/all-mpnet-base-v2',
    'biobert': 'pritamdeka/S-PubMedBert-MS-MARCO',  # domain-specific (medical)
}

# Event detection — merge spike rows that are close in time
EVENT_GAP_MINUTES = 30
MIN_EVENT_DURATION_MIN = 15  # at least one 15-min interval

# Clustering parameters
UMAP_COMPONENTS = 10
UMAP_NEIGHBORS = 15
HDBSCAN_MIN_CLUSTER_SIZE = 20
HDBSCAN_MIN_SAMPLES = 5


# ── Step 1: Detect spike events from row-level data ──────────────────────────
def detect_events(df: pd.DataFrame, spike_col: str = 'price_spike_rolling') -> pd.DataFrame:
    """
    Merge consecutive spike rows into single events.

    An event = group of rows where:
      - All rows are flagged as spikes
      - Time gap to next spike row is <= EVENT_GAP_MINUTES

    Returns a DataFrame with one row per event:
      event_id, start_ts, end_ts, duration_min, peak_price,
      mean_price, n_intervals, plus aggregated feature stats
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    spikes = df[df[spike_col] == 1].copy()
    if len(spikes) == 0:
        return pd.DataFrame()

    # Detect event boundaries: gap > EVENT_GAP_MINUTES means new event
    spikes['gap'] = spikes['timestamp'].diff().dt.total_seconds() / 60
    spikes['new_event'] = (spikes['gap'].isna()) | (spikes['gap'] > EVENT_GAP_MINUTES)
    spikes['event_id'] = spikes['new_event'].cumsum()

    # Aggregate features per event
    feature_cols_to_aggregate = [
        'pos_sec_price', 'DE_WindSolar_Error', 'abs_Unplanned_Flow',
        'Sched_DE_CH', 'Unplanned_Flow_FR_CH', 'CH_Load_Forecast',
        'CH_Pump_Gen', 'DA_Price_DE', 'DA_Price_CH', 'DA_Price_Spread_DE_CH',
        'rolling_p90_threshold',
    ]
    available = [c for c in feature_cols_to_aggregate if c in spikes.columns]

    events = spikes.groupby('event_id').agg(
        start_ts=('timestamp', 'min'),
        end_ts=('timestamp', 'max'),
        n_intervals=('timestamp', 'count'),
        peak_price=('pos_sec_price', 'max'),
        mean_price=('pos_sec_price', 'mean'),
        **{f'mean_{c}': (c, 'mean') for c in available},
        **{f'max_{c}': (c, 'max')   for c in available},
    ).reset_index()

    events['duration_min'] = (
        (events['end_ts'] - events['start_ts']).dt.total_seconds() / 60 + 15
    ).astype(int)

    # Filter very short events (single isolated 15-min spikes are noise)
    events = events[events['duration_min'] >= MIN_EVENT_DURATION_MIN].reset_index(drop=True)

    print(f"[events] Detected {len(events):,} spike events from "
          f"{spikes[spike_col].sum():,} spike rows")
    print(f"[events] Duration: mean={events['duration_min'].mean():.0f}min, "
          f"max={events['duration_min'].max():.0f}min")

    return events


# ── Step 2: Generate mechanism-focused descriptions ──────────────────────────
def describe_event(event: pd.Series) -> str:
    """
    Generate a natural language description focused on physical mechanism.

    Style guidance applied:
      - Mechanism-focused (cause → effect → key context)
      - NO weekday/season — avoid calendar dominance in embeddings
      - Severity, German wind error, unplanned flow, prices, duration
      - Compact: single paragraph, ~2-4 sentences
    """
    # Severity tier from peak price relative to rolling threshold
    threshold = event.get('mean_rolling_p90_threshold', 200)
    ratio = event['peak_price'] / max(threshold, 1)
    if ratio > 5:
        severity = "extreme"
    elif ratio > 3:
        severity = "severe"
    elif ratio > 2:
        severity = "moderate"
    else:
        severity = "mild"

    # German wind error context
    de_err = event.get('mean_DE_WindSolar_Error', 0)
    if abs(de_err) > 5000:
        wind_phrase = f"large German wind+solar forecast error of {de_err:.0f} MW"
    elif abs(de_err) > 2000:
        wind_phrase = f"moderate German forecast deviation of {de_err:.0f} MW"
    else:
        wind_phrase = f"small German forecast deviation of {de_err:.0f} MW"

    # Unplanned flow context — the causal variable
    flow = event.get('mean_abs_Unplanned_Flow', 0)
    flow_max = event.get('max_abs_Unplanned_Flow', 0)
    if flow > 1500:
        flow_phrase = f"high unplanned cross-border flow averaging {flow:.0f} MW " \
                      f"(peaking at {flow_max:.0f} MW, well above the 1250 MW grid bottleneck)"
    elif flow > 800:
        flow_phrase = f"elevated unplanned flow averaging {flow:.0f} MW " \
                      f"(peaking at {flow_max:.0f} MW, near the 1250 MW bottleneck)"
    else:
        flow_phrase = f"low unplanned flow averaging {flow:.0f} MW " \
                      f"(peaking at {flow_max:.0f} MW, below the bottleneck threshold)"

    # Price spread — economic incentive to push power
    spread = event.get('mean_DA_Price_Spread_DE_CH', 0)
    if abs(spread) > 50:
        spread_phrase = f"large day-ahead price spread DE-CH of {spread:.0f} EUR/MWh"
    elif abs(spread) > 10:
        spread_phrase = f"moderate day-ahead price spread DE-CH of {spread:.0f} EUR/MWh"
    else:
        spread_phrase = f"narrow day-ahead price spread DE-CH of {spread:.0f} EUR/MWh"

    # Swiss demand context
    load = event.get('mean_CH_Load_Forecast', 0)
    pump = event.get('mean_CH_Pump_Gen', 0)
    if load > 8500 and pump < 500:
        demand_phrase = "during Swiss high demand with low pump generation"
    elif load > 8500:
        demand_phrase = "during Swiss high demand"
    elif pump > 2000:
        demand_phrase = "during high pump-storage activity"
    else:
        demand_phrase = "during normal Swiss demand"

    # Duration
    if event['duration_min'] >= 120:
        duration_phrase = f"sustained {event['duration_min']:.0f}-minute event"
    elif event['duration_min'] >= 45:
        duration_phrase = f"prolonged {event['duration_min']:.0f}-minute event"
    else:
        duration_phrase = f"brief {event['duration_min']:.0f}-minute event"

    # Assemble — mechanism-first ordering
    description = (
        f"{severity.capitalize()} aFRR price spike: {duration_phrase}, "
        f"peak {event['peak_price']:.0f} EUR/MWh ({ratio:.1f}x threshold). "
        f"Driven by {wind_phrase} and {flow_phrase}. "
        f"Market context: {spread_phrase}, {demand_phrase}."
    )
    return description


# ── Step 3: Embed with three models ──────────────────────────────────────────
def embed_descriptions(descriptions: list, model_name: str) -> np.ndarray:
    """Embed list of strings with a sentence-transformer model."""
    from sentence_transformers import SentenceTransformer

    print(f"[embed] Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"[embed] Encoding {len(descriptions)} descriptions...")
    embeddings = model.encode(
        descriptions,
        show_progress_bar=True,
        batch_size=64,
        convert_to_numpy=True,
    )
    print(f"[embed] Output shape: {embeddings.shape}")
    return embeddings


# ── Step 4: Cluster with UMAP + HDBSCAN ──────────────────────────────────────
def cluster_embeddings(embeddings: np.ndarray) -> dict:
    """
    Reduce embeddings with UMAP, cluster with HDBSCAN.

    Returns:
        cluster_labels:  array of cluster IDs (-1 = noise)
        umap_2d:         2D projection for visualisation
        umap_10d:        10D projection used for clustering
        silhouette:      cluster quality score
        n_clusters:      number of clusters found
        n_noise:         number of noise points
    """
    import umap
    import hdbscan
    from sklearn.metrics import silhouette_score

    # 2D for visualisation
    print("[cluster] UMAP 2D for visualisation...")
    umap_2d = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_NEIGHBORS,
        random_state=42,
    ).fit_transform(embeddings)

    # 10D for clustering input
    print("[cluster] UMAP 10D for clustering...")
    umap_10d = umap.UMAP(
        n_components=UMAP_COMPONENTS,
        n_neighbors=UMAP_NEIGHBORS,
        random_state=42,
    ).fit_transform(embeddings)

    print("[cluster] HDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
    )
    labels = clusterer.fit_predict(umap_10d)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())

    # Silhouette score (only on clustered points, excluding noise)
    sil = None
    if n_clusters >= 2:
        mask = labels != -1
        if mask.sum() >= 10:
            sil = float(silhouette_score(umap_10d[mask], labels[mask]))

    print(f"[cluster] Found {n_clusters} clusters, {n_noise} noise points")
    print(f"[cluster] Silhouette score: {sil}")

    return {
        'labels':     labels,
        'umap_2d':    umap_2d,
        'umap_10d':   umap_10d,
        'silhouette': sil,
        'n_clusters': n_clusters,
        'n_noise':    n_noise,
    }


# ── Step 5: Sample inspection per cluster ────────────────────────────────────
def sample_clusters(events: pd.DataFrame, descriptions: list, labels: np.ndarray,
                    n_samples: int = 5) -> dict:
    """For each cluster, return n_samples representative descriptions."""
    samples = {}
    unique_clusters = sorted(set(labels))
    for cluster_id in unique_clusters:
        idx = np.where(labels == cluster_id)[0]
        if len(idx) == 0:
            continue
        # Take evenly spaced samples
        sample_indices = idx[np.linspace(0, len(idx) - 1, min(n_samples, len(idx))).astype(int)]
        samples[int(cluster_id)] = {
            'size': len(idx),
            'mean_peak_price': float(events.iloc[idx]['peak_price'].mean()),
            'mean_duration_min': float(events.iloc[idx]['duration_min'].mean()),
            'descriptions': [descriptions[i] for i in sample_indices],
        }
    return samples


# ── Step 6: Plot UMAP 2D ────────────────────────────────────────────────────
def plot_umap(umap_2d: np.ndarray, labels: np.ndarray, model_key: str, out_path: str):
    """Save UMAP 2D scatter plot coloured by cluster."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 8))

    unique_clusters = sorted(set(labels))
    cmap = plt.cm.tab20

    for i, cluster_id in enumerate(unique_clusters):
        mask = labels == cluster_id
        if cluster_id == -1:
            ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1],
                       c='lightgrey', s=8, alpha=0.5, label=f'noise ({mask.sum()})')
        else:
            ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1],
                       c=[cmap(i % 20)], s=12, alpha=0.7,
                       label=f'cluster {cluster_id} ({mask.sum()})')

    ax.set_title(f"UMAP 2D — {model_key} embeddings\n{len(unique_clusters)-1} clusters")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# ── Main pipeline ─────────────────────────────────────────────────────────────
def main():
    # ── Load and detect events ───────────────────────────────────────────────
    print("=" * 70)
    print("SPIKE INCIDENT CLUSTERING — Three-Model Comparison")
    print("=" * 70)

    train_path = os.path.join(DATA_DIR, 'features_train_2023_2024.csv')
    print(f"\nLoading: {train_path}")
    df = pd.read_csv(train_path)
    print(f"Rows: {len(df):,}")

    print("\n[Step 1] Detecting spike events")
    print("-" * 70)
    events = detect_events(df)
    if events.empty:
        print("No events found")
        return

    print("\n[Step 2] Generating descriptions")
    print("-" * 70)
    descriptions = events.apply(describe_event, axis=1).tolist()
    print(f"Generated {len(descriptions)} descriptions")
    print("\nSample description:")
    print(f"  {descriptions[0]}")
    print(f"  {descriptions[len(descriptions) // 2]}")

    # ── Compare embedding models ──────────────────────────────────────────────
    summary = {}
    for model_key, model_name in EMBEDDING_MODELS.items():
        print("\n" + "=" * 70)
        print(f"[Model: {model_key}] {model_name}")
        print("=" * 70)

        try:
            embeddings = embed_descriptions(descriptions, model_name)
        except Exception as e:
            print(f"Failed to load {model_key}: {e}")
            summary[model_key] = {'error': str(e)}
            continue

        result = cluster_embeddings(embeddings)
        samples = sample_clusters(events, descriptions, result['labels'])

        # Save cluster assignments
        out_csv = os.path.join(OUT_DIR, f'cluster_assignments_{model_key}.csv')
        events_with_clusters = events.copy()
        events_with_clusters[f'cluster_{model_key}'] = result['labels']
        events_with_clusters[f'umap_x_{model_key}'] = result['umap_2d'][:, 0]
        events_with_clusters[f'umap_y_{model_key}'] = result['umap_2d'][:, 1]
        events_with_clusters['description'] = descriptions
        events_with_clusters.to_csv(out_csv, index=False)
        print(f"\nSaved assignments: {out_csv}")

        # Save samples for inspection
        out_txt = os.path.join(OUT_DIR, f'cluster_samples_{model_key}.txt')
        with open(out_txt, 'w') as f:
            f.write(f"Cluster Samples — {model_key} ({model_name})\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total events: {len(events)}\n")
            f.write(f"Clusters: {result['n_clusters']}\n")
            f.write(f"Noise points: {result['n_noise']}\n")
            f.write(f"Silhouette score: {result['silhouette']}\n\n")

            for cluster_id in sorted(samples.keys()):
                s = samples[cluster_id]
                tag = "NOISE" if cluster_id == -1 else f"CLUSTER {cluster_id}"
                f.write(f"\n{tag} (size={s['size']}, "
                        f"mean_peak={s['mean_peak_price']:.0f} EUR/MWh, "
                        f"mean_duration={s['mean_duration_min']:.0f}min)\n")
                f.write("-" * 70 + "\n")
                for i, desc in enumerate(s['descriptions'], 1):
                    f.write(f"{i}. {desc}\n\n")
        print(f"Saved samples: {out_txt}")

        # Plot
        out_png = os.path.join(OUT_DIR, f'umap_2d_{model_key}.png')
        plot_umap(result['umap_2d'], result['labels'], model_key, out_png)
        print(f"Saved plot: {out_png}")

        summary[model_key] = {
            'model_name': model_name,
            'n_events':   len(events),
            'n_clusters': result['n_clusters'],
            'n_noise':    result['n_noise'],
            'noise_rate': result['n_noise'] / len(events),
            'silhouette': result['silhouette'],
        }

    # ── Final comparison summary ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<10} {'Clusters':>10} {'Noise':>10} {'Silhouette':>12}")
    print("-" * 50)
    for key, s in summary.items():
        if 'error' in s:
            print(f"{key:<10} ERROR: {s['error']}")
            continue
        sil = f"{s['silhouette']:.3f}" if s['silhouette'] is not None else 'N/A'
        print(f"{key:<10} {s['n_clusters']:>10} "
              f"{s['n_noise']:>5}/{s['n_events']:<4}  {sil:>12}")

    summary_path = os.path.join(OUT_DIR, 'comparison_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull summary saved: {summary_path}")
    print(f"\nAll outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
