"""
Tutorial 04g: Per-Metric DTW Investigation

Research Question:
For multi-metric trajectory analysis, does computing DTW separately for each metric
and then combining the distances produce more robust/interpretable results than
the current multivariate DTW approach?

Current Approach (Multivariate DTW):
- Combine all metrics into single feature vector at each timepoint
- Distance = Euclidean distance in feature space
- Requires Z-score normalization to balance scales

Proposed Approach (Per-Metric DTW):
- Compute DTW separately for each metric using raw values
- Combine resulting distance matrices using various strategies
- Hypothesis: More robust to cross-experiment differences, more interpretable

This tutorial compares both approaches on experiment 20250512.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.stats import pearsonr
from sklearn.metrics import adjusted_rand_score
import warnings

# Import core DTW functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

from analyze.utils.timeseries.dtw import compute_md_dtw_distance_matrix
from analyze.trajectory_analysis.utilities.dtw_utils import prepare_multivariate_array
from analyze.trajectory_analysis.distance import compute_trajectory_distances

# Setup paths
DATA_FILE = Path(__file__).parent.parent.parent / 'mcolon' / '20251229_cep290_phenotype_extraction' / 'final_data' / 'embryo_data_with_labels.csv'
OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("TUTORIAL 04g: Per-Metric DTW Investigation")
print("="*80)

# ============================================================================
# SECTION 1: Data Loading
# ============================================================================

print("\n" + "="*80)
print("SECTION 1: Data Loading")
print("="*80)

# Load data
print(f"\nLoading data from: {DATA_FILE}")
df_full = pd.read_csv(DATA_FILE, low_memory=False)

# Filter to experiment 20250512 (excellent coverage, 88 embryos)
EXPERIMENT = '20250512'
df_exp = df_full[df_full['experiment_id'] == EXPERIMENT].copy()

# Get metrics
METRICS = ['baseline_deviation_normalized', 'total_length_um']

# Filter to embryos with sufficient data
MIN_TIMEPOINTS = 50
embryo_counts = df_exp.groupby('embryo_id').size()
valid_embryos = embryo_counts[embryo_counts >= MIN_TIMEPOINTS].index
df_exp = df_exp[df_exp['embryo_id'].isin(valid_embryos)].copy()

print(f"\nExperiment: {EXPERIMENT}")
print(f"Total embryos: {len(valid_embryos)}")
print(f"Metrics: {METRICS}")
print(f"Timepoint threshold: >={MIN_TIMEPOINTS}")

# Check data availability
print("\nData availability per metric:")
for metric in METRICS:
    n_valid = df_exp.groupby('embryo_id')[metric].apply(lambda x: x.notna().sum()).min()
    print(f"  {metric}: min {n_valid} timepoints per embryo")

# ============================================================================
# SECTION 2: Helper Functions
# ============================================================================

print("\n" + "="*80)
print("SECTION 2: Helper Functions")
print("="*80)

def normalize_distance_matrix(D: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize a distance matrix.

    Parameters
    ----------
    D : np.ndarray
        Distance matrix (symmetric, non-negative)
    method : str
        'minmax': (D - min) / (max - min)
        'zscore': (D - mean) / std

    Returns
    -------
    D_norm : np.ndarray
        Normalized distance matrix
    """
    if method == 'minmax':
        D_min = D.min()
        D_max = D.max()
        if D_max == D_min:
            return np.zeros_like(D)
        return (D - D_min) / (D_max - D_min)
    elif method == 'zscore':
        # Only use upper triangle (avoid double-counting)
        triu_idx = np.triu_indices_from(D, k=1)
        D_mean = D[triu_idx].mean()
        D_std = D[triu_idx].std()
        if D_std == 0:
            return np.zeros_like(D)
        return (D - D_mean) / D_std
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_per_metric_dtw(
    df: pd.DataFrame,
    metrics: List[str],
    combination_method: str = 'mean',
    normalize_distances: bool = True,
    normalize_method: str = 'minmax',
    weights: Optional[List[float]] = None,
    sakoe_chiba_radius: Optional[int] = 20,
    time_col: str = 'predicted_stage_hpf',
    embryo_id_col: str = 'embryo_id',
    time_grid: Optional[np.ndarray] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, List[str], np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute DTW separately for each metric and combine distances.

    This function computes DTW distance matrices independently for each metric
    (using raw values, no normalization), then combines the resulting distance
    matrices using the specified strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Trajectory data
    metrics : List[str]
        List of metric column names
    combination_method : str
        How to combine per-metric distances:
        - 'mean': Simple average (default)
        - 'weighted': Weighted average (requires weights parameter)
        - 'euclidean': sqrt(D1² + D2² + ...)
        - 'min': Take minimum across metrics
        - 'max': Take maximum across metrics
    normalize_distances : bool
        Whether to normalize each distance matrix before combining
    normalize_method : str
        Normalization method: 'minmax' or 'zscore'
    weights : Optional[List[float]]
        Weights for 'weighted' combination method (must sum to 1)
    sakoe_chiba_radius : Optional[int]
        Sakoe-Chiba band radius for DTW
    time_col : str
        Name of time column
    embryo_id_col : str
        Name of embryo ID column
    time_grid : Optional[np.ndarray]
        Optional time grid (passed to prepare_multivariate_array)
    verbose : bool
        Whether to print verbose output

    Returns
    -------
    D_combined : np.ndarray
        Combined distance matrix (n_embryos x n_embryos)
    embryo_ids : List[str]
        List of embryo IDs (order matches rows/cols of D_combined)
    time_grid : np.ndarray
        Time grid used for alignment
    per_metric_results : Dict[str, np.ndarray]
        Dictionary mapping metric names to their distance matrices
    """
    # Validate parameters
    if combination_method == 'weighted':
        if weights is None:
            raise ValueError("weights parameter required for weighted combination")
        if len(weights) != len(metrics):
            raise ValueError(f"weights length ({len(weights)}) must match metrics length ({len(metrics)})")
        if not np.isclose(sum(weights), 1.0):
            raise ValueError(f"weights must sum to 1.0, got {sum(weights)}")

    print(f"\nComputing per-metric DTW:")
    print(f"  Metrics: {metrics}")
    print(f"  Combination method: {combination_method}")
    print(f"  Normalize distances: {normalize_distances} ({normalize_method if normalize_distances else 'N/A'})")
    if weights is not None:
        print(f"  Weights: {weights}")

    per_metric_distances = {}
    embryo_ids = None
    time_grid = None

    # Compute DTW for each metric separately
    for metric in metrics:
        print(f"\n  Processing {metric}...")

        # Prepare single-metric array (NO normalization - use raw values)
        X_metric, ids, grid = prepare_multivariate_array(
            df,
            metrics=[metric],
            normalize=False,  # KEY: No normalization for per-metric approach
            time_col=time_col,
            embryo_id_col=embryo_id_col,
            time_grid=time_grid,
            verbose=verbose
        )

        print(f"    Array shape: {X_metric.shape} (n_embryos={X_metric.shape[0]}, n_timepoints={X_metric.shape[1]}, n_metrics=1)")
        print(f"    Value range: [{np.nanmin(X_metric):.3f}, {np.nanmax(X_metric):.3f}]")

        # Ensure consistent embryo ordering across metrics
        if embryo_ids is None:
            embryo_ids = ids
            time_grid = grid
        else:
            if ids != embryo_ids:
                raise ValueError(f"Embryo IDs differ between metrics (this shouldn't happen)")

        # Compute DTW distance matrix for this metric
        D_metric = compute_md_dtw_distance_matrix(
            X_metric,
            sakoe_chiba_radius=sakoe_chiba_radius
        )

        print(f"    Distance range: [{D_metric.min():.3f}, {D_metric.max():.3f}]")
        print(f"    Distance mean: {D_metric[np.triu_indices_from(D_metric, k=1)].mean():.3f}")

        per_metric_distances[metric] = D_metric

    # Normalize distance matrices if requested
    if normalize_distances:
        print(f"\n  Normalizing distance matrices ({normalize_method})...")
        for metric in metrics:
            D = per_metric_distances[metric]
            D_norm = normalize_distance_matrix(D, method=normalize_method)
            per_metric_distances[metric] = D_norm
            print(f"    {metric}: [{D_norm.min():.3f}, {D_norm.max():.3f}]")

    # Combine distance matrices
    print(f"\n  Combining distances using '{combination_method}' method...")
    distance_arrays = [per_metric_distances[m] for m in metrics]

    if combination_method == 'mean':
        D_combined = np.mean(distance_arrays, axis=0)
    elif combination_method == 'weighted':
        D_combined = sum(w * D for w, D in zip(weights, distance_arrays))
    elif combination_method == 'euclidean':
        D_combined = np.sqrt(sum(D**2 for D in distance_arrays))
    elif combination_method == 'min':
        D_combined = np.min(distance_arrays, axis=0)
    elif combination_method == 'max':
        D_combined = np.max(distance_arrays, axis=0)
    else:
        raise ValueError(f"Unknown combination method: {combination_method}")

    print(f"    Combined distance range: [{D_combined.min():.3f}, {D_combined.max():.3f}]")
    print(f"    Combined distance mean: {D_combined[np.triu_indices_from(D_combined, k=1)].mean():.3f}")

    return D_combined, embryo_ids, time_grid, per_metric_distances


def compute_cluster_assignments(D: np.ndarray, n_clusters: int = 4, method: str = 'ward') -> np.ndarray:
    """
    Compute hierarchical cluster assignments from distance matrix.

    Parameters
    ----------
    D : np.ndarray
        Distance matrix (symmetric)
    n_clusters : int
        Number of clusters
    method : str
        Linkage method

    Returns
    -------
    clusters : np.ndarray
        Cluster assignments (1-indexed)
    """
    D_condensed = squareform(D)
    Z = linkage(D_condensed, method=method)
    clusters = fcluster(Z, t=n_clusters, criterion='maxclust')
    return clusters


print("\nHelper functions defined:")
print("  - normalize_distance_matrix()")
print("  - compute_per_metric_dtw()")
print("  - compute_cluster_assignments()")

# ============================================================================
# SECTION 3: Approach A - Current Multivariate DTW
# ============================================================================

print("\n" + "="*80)
print("SECTION 3: Approach A - Current Multivariate DTW")
print("="*80)
print("\nThis approach:")
print("  1. Z-score normalizes each metric")
print("  2. Combines metrics into single feature vector per timepoint")
print("  3. Computes DTW using Euclidean distance in feature space")

# Compute multivariate DTW (current approach)
D_multivariate, ids_multi, grid_multi = compute_trajectory_distances(
    df_exp,
    metrics=METRICS,
    normalize=True,  # Z-score normalization
    sakoe_chiba_radius=20,
    time_col='predicted_stage_hpf',  # Use predicted_stage_hpf as time
    embryo_id_col='embryo_id'
)

print(f"\nMultivariate DTW results:")
print(f"  Distance matrix shape: {D_multivariate.shape}")
print(f"  Distance range: [{D_multivariate.min():.3f}, {D_multivariate.max():.3f}]")
print(f"  Distance mean: {D_multivariate[np.triu_indices_from(D_multivariate, k=1)].mean():.3f}")
print(f"  Distance std: {D_multivariate[np.triu_indices_from(D_multivariate, k=1)].std():.3f}")

# Cluster using hierarchical clustering
N_CLUSTERS = 4
clusters_multi = compute_cluster_assignments(D_multivariate, n_clusters=N_CLUSTERS)

print(f"\nClustering (n={N_CLUSTERS}):")
cluster_counts_multi = pd.Series(clusters_multi).value_counts().sort_index()
for cluster_id, count in cluster_counts_multi.items():
    pct = 100 * count / len(clusters_multi)
    print(f"  Cluster {cluster_id}: {count:2d} embryos ({pct:5.1f}%)")

# ============================================================================
# SECTION 4: Approach B - Per-Metric DTW + Combination
# ============================================================================

print("\n" + "="*80)
print("SECTION 4: Approach B - Per-Metric DTW + Combination")
print("="*80)
print("\nThis approach:")
print("  1. Computes DTW separately for each metric (raw values, no normalization)")
print("  2. Normalizes each distance matrix independently")
print("  3. Combines distance matrices using various strategies")

# Test multiple combination strategies
combination_strategies = {
    'mean': {'method': 'mean', 'normalize': True},
    'euclidean': {'method': 'euclidean', 'normalize': True},
    'weighted_70_30': {'method': 'weighted', 'weights': [0.7, 0.3], 'normalize': True},
    'weighted_50_50': {'method': 'weighted', 'weights': [0.5, 0.5], 'normalize': True},
    'mean_unnormalized': {'method': 'mean', 'normalize': False},
}

per_metric_results = {}

for strategy_name, params in combination_strategies.items():
    print(f"\n--- Strategy: {strategy_name} ---")

    D_combined, ids_per, grid_per, per_metric_dists = compute_per_metric_dtw(
        df_exp,
        metrics=METRICS,
        combination_method=params['method'],
        normalize_distances=params['normalize'],
        weights=params.get('weights', None),
        sakoe_chiba_radius=20
    )

    # Cluster
    clusters_per = compute_cluster_assignments(D_combined, n_clusters=N_CLUSTERS)

    print(f"\nClustering results:")
    cluster_counts_per = pd.Series(clusters_per).value_counts().sort_index()
    for cluster_id, count in cluster_counts_per.items():
        pct = 100 * count / len(clusters_per)
        print(f"  Cluster {cluster_id}: {count:2d} embryos ({pct:5.1f}%)")

    # Store results
    per_metric_results[strategy_name] = {
        'D_combined': D_combined,
        'clusters': clusters_per,
        'per_metric_distances': per_metric_dists,
        'embryo_ids': ids_per
    }

# ============================================================================
# SECTION 5: Comparison Analysis
# ============================================================================

print("\n" + "="*80)
print("SECTION 5: Comparison Analysis")
print("="*80)

comparison_results = {}

for strategy_name, results in per_metric_results.items():
    print(f"\n{'='*80}")
    print(f"Comparing: Multivariate DTW vs Per-Metric DTW ({strategy_name})")
    print('='*80)

    D_per = results['D_combined']
    clusters_per = results['clusters']

    # 1. Distance Matrix Correlation
    triu_idx = np.triu_indices_from(D_multivariate, k=1)
    r_distances, p_distances = pearsonr(
        D_multivariate[triu_idx],
        D_per[triu_idx]
    )

    print(f"\n1. Distance Matrix Correlation:")
    print(f"   Pearson r = {r_distances:.4f} (p = {p_distances:.2e})")
    if r_distances > 0.9:
        print(f"   Interpretation: Very similar distance matrices")
    elif r_distances > 0.7:
        print(f"   Interpretation: Moderately similar distance matrices")
    else:
        print(f"   Interpretation: Different distance matrices")

    # 2. Cluster Agreement
    ari = adjusted_rand_score(clusters_multi, clusters_per)

    print(f"\n2. Cluster Agreement (Adjusted Rand Index):")
    print(f"   ARI = {ari:.4f}")
    if ari == 1.0:
        print(f"   Interpretation: Perfect agreement")
    elif ari > 0.8:
        print(f"   Interpretation: Good agreement")
    elif ari > 0.5:
        print(f"   Interpretation: Moderate agreement")
    else:
        print(f"   Interpretation: Poor agreement")

    # 3. Cluster Size Distributions
    print(f"\n3. Cluster Size Distributions:")
    counts_multi = pd.Series(clusters_multi).value_counts(normalize=True).sort_index()
    counts_per = pd.Series(clusters_per).value_counts(normalize=True).sort_index()

    print(f"   {'Cluster':<10} {'Multivariate':<15} {'Per-Metric':<15} {'Diff':<10}")
    print(f"   {'-'*50}")
    for cluster_id in range(1, N_CLUSTERS + 1):
        pct_multi = counts_multi.get(cluster_id, 0) * 100
        pct_per = counts_per.get(cluster_id, 0) * 100
        diff = pct_per - pct_multi
        print(f"   {cluster_id:<10} {pct_multi:>6.1f}%         {pct_per:>6.1f}%         {diff:>+6.1f}%")

    # 4. Per-Metric Contribution Analysis
    print(f"\n4. Per-Metric Contribution Analysis:")
    print(f"   (Correlation of each metric's distance matrix with combined)")

    for metric in METRICS:
        D_metric = results['per_metric_distances'][metric]
        r_contrib, _ = pearsonr(D_metric[triu_idx], D_per[triu_idx])
        print(f"   {metric:<35} r = {r_contrib:+.4f}")

    # Determine dominant metric
    contribs = {
        metric: pearsonr(results['per_metric_distances'][metric][triu_idx], D_per[triu_idx])[0]
        for metric in METRICS
    }
    dominant_metric = max(contribs, key=contribs.get)
    print(f"   Dominant metric: {dominant_metric}")

    # Store comparison results
    comparison_results[strategy_name] = {
        'r_distances': r_distances,
        'p_distances': p_distances,
        'ari': ari,
        'cluster_counts_multi': counts_multi,
        'cluster_counts_per': counts_per,
        'metric_contributions': contribs,
        'dominant_metric': dominant_metric
    }

# ============================================================================
# SECTION 6: Visualization
# ============================================================================

print("\n" + "="*80)
print("SECTION 6: Visualization")
print("="*80)

# Choose best per-metric strategy for detailed visualization
best_strategy = max(comparison_results.items(), key=lambda x: x[1]['ari'])[0]
print(f"\nUsing '{best_strategy}' strategy for detailed visualizations (highest ARI)")

D_per_best = per_metric_results[best_strategy]['D_combined']
clusters_per_best = per_metric_results[best_strategy]['clusters']
per_metric_dists_best = per_metric_results[best_strategy]['per_metric_distances']

# Plot 1: Distance Matrix Comparison
print("\nGenerating Plot 1: Distance Matrix Comparison...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Heatmap: Multivariate
ax = axes[0, 0]
im1 = ax.imshow(D_multivariate, cmap='viridis', aspect='auto')
ax.set_title('Multivariate DTW Distance Matrix', fontsize=12, fontweight='bold')
ax.set_xlabel('Embryo Index')
ax.set_ylabel('Embryo Index')
plt.colorbar(im1, ax=ax, label='Distance')

# Heatmap: Per-Metric
ax = axes[0, 1]
im2 = ax.imshow(D_per_best, cmap='viridis', aspect='auto')
ax.set_title(f'Per-Metric DTW Distance Matrix ({best_strategy})', fontsize=12, fontweight='bold')
ax.set_xlabel('Embryo Index')
ax.set_ylabel('Embryo Index')
plt.colorbar(im2, ax=ax, label='Distance')

# Scatterplot: Multivariate vs Per-Metric
ax = axes[1, 0]
triu_idx = np.triu_indices_from(D_multivariate, k=1)
ax.scatter(D_multivariate[triu_idx], D_per_best[triu_idx], alpha=0.3, s=10)
ax.plot([0, max(D_multivariate.max(), D_per_best.max())],
        [0, max(D_multivariate.max(), D_per_best.max())],
        'r--', label='y=x', linewidth=2)
ax.set_xlabel('Multivariate DTW Distance')
ax.set_ylabel(f'Per-Metric DTW Distance ({best_strategy})')
ax.set_title(f'Distance Correlation (r={comparison_results[best_strategy]["r_distances"]:.3f})',
             fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Distribution comparison
ax = axes[1, 1]
ax.hist(D_multivariate[triu_idx], bins=50, alpha=0.5, label='Multivariate', density=True)
ax.hist(D_per_best[triu_idx], bins=50, alpha=0.5, label=f'Per-Metric ({best_strategy})', density=True)
ax.set_xlabel('Distance')
ax.set_ylabel('Density')
ax.set_title('Distance Distribution Comparison', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_file = OUTPUT_DIR / '04g_distance_comparison.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Saved: {output_file}")
plt.close()

# Plot 2: Cluster Assignment Comparison
print("\nGenerating Plot 2: Cluster Assignment Comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Confusion matrix
ax = axes[0]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(clusters_multi, clusters_per_best)
im = ax.imshow(cm, cmap='Blues', aspect='auto')
ax.set_xticks(range(N_CLUSTERS))
ax.set_yticks(range(N_CLUSTERS))
ax.set_xticklabels(range(1, N_CLUSTERS + 1))
ax.set_yticklabels(range(1, N_CLUSTERS + 1))
ax.set_xlabel(f'Per-Metric Cluster ({best_strategy})')
ax.set_ylabel('Multivariate Cluster')
ax.set_title(f'Cluster Assignment Confusion Matrix\nARI = {comparison_results[best_strategy]["ari"]:.3f}',
             fontsize=12, fontweight='bold')

# Add text annotations
for i in range(N_CLUSTERS):
    for j in range(N_CLUSTERS):
        text = ax.text(j, i, cm[i, j], ha="center", va="center", color="black" if cm[i, j] < cm.max()/2 else "white")

plt.colorbar(im, ax=ax, label='Count')

# Cluster size comparison
ax = axes[1]
cluster_ids = range(1, N_CLUSTERS + 1)
counts_multi = [pd.Series(clusters_multi).value_counts().get(i, 0) for i in cluster_ids]
counts_per = [pd.Series(clusters_per_best).value_counts().get(i, 0) for i in cluster_ids]

x = np.arange(len(cluster_ids))
width = 0.35

ax.bar(x - width/2, counts_multi, width, label='Multivariate', alpha=0.8)
ax.bar(x + width/2, counts_per, width, label=f'Per-Metric ({best_strategy})', alpha=0.8)

ax.set_xlabel('Cluster')
ax.set_ylabel('Number of Embryos')
ax.set_title('Cluster Size Comparison', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(cluster_ids)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_file = OUTPUT_DIR / '04g_cluster_comparison.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Saved: {output_file}")
plt.close()

# Plot 3: Per-Metric Contribution
print("\nGenerating Plot 3: Per-Metric Contribution...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Individual metric distance matrices
for idx, metric in enumerate(METRICS):
    ax = axes[0, idx]
    D_metric = per_metric_dists_best[metric]
    im = ax.imshow(D_metric, cmap='viridis', aspect='auto')
    ax.set_title(f'{metric}\nDistance Matrix', fontsize=11, fontweight='bold')
    ax.set_xlabel('Embryo Index')
    ax.set_ylabel('Embryo Index')
    plt.colorbar(im, ax=ax, label='Distance')

# Correlation with combined distance
triu_idx = np.triu_indices_from(D_per_best, k=1)

for idx, metric in enumerate(METRICS):
    ax = axes[1, idx]
    D_metric = per_metric_dists_best[metric]
    r_contrib = comparison_results[best_strategy]['metric_contributions'][metric]

    ax.scatter(D_metric[triu_idx], D_per_best[triu_idx], alpha=0.3, s=10)
    ax.set_xlabel(f'{metric} Distance')
    ax.set_ylabel('Combined Distance')
    ax.set_title(f'Contribution to Combined\nr = {r_contrib:.3f}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
output_file = OUTPUT_DIR / '04g_metric_contributions.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Saved: {output_file}")
plt.close()

# Plot 4: Comparison Across All Strategies
print("\nGenerating Plot 4: Strategy Comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ARI comparison
ax = axes[0]
strategies = list(comparison_results.keys())
aris = [comparison_results[s]['ari'] for s in strategies]
colors = ['#1f77b4' if s == best_strategy else '#7f7f7f' for s in strategies]

bars = ax.barh(strategies, aris, color=colors, alpha=0.8)
ax.set_xlabel('Adjusted Rand Index (ARI)')
ax.set_title('Cluster Agreement: Multivariate vs Per-Metric Strategies',
             fontsize=12, fontweight='bold')
ax.set_xlim(0, 1)
ax.axvline(0.8, color='green', linestyle='--', alpha=0.5, label='Good agreement (0.8)')
ax.axvline(0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate agreement (0.5)')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (strategy, ari) in enumerate(zip(strategies, aris)):
    ax.text(ari + 0.01, i, f'{ari:.3f}', va='center')

# Distance correlation comparison
ax = axes[1]
r_values = [comparison_results[s]['r_distances'] for s in strategies]
colors = ['#1f77b4' if s == best_strategy else '#7f7f7f' for s in strategies]

bars = ax.barh(strategies, r_values, color=colors, alpha=0.8)
ax.set_xlabel('Pearson Correlation (r)')
ax.set_title('Distance Matrix Correlation: Multivariate vs Per-Metric',
             fontsize=12, fontweight='bold')
ax.set_xlim(0, 1)
ax.axvline(0.9, color='green', linestyle='--', alpha=0.5, label='Very similar (0.9)')
ax.axvline(0.7, color='orange', linestyle='--', alpha=0.5, label='Moderately similar (0.7)')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (strategy, r) in enumerate(zip(strategies, r_values)):
    ax.text(r + 0.01, i, f'{r:.3f}', va='center')

plt.tight_layout()
output_file = OUTPUT_DIR / '04g_strategy_comparison.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Saved: {output_file}")
plt.close()

# ============================================================================
# SECTION 7: Summary and Recommendations
# ============================================================================

print("\n" + "="*80)
print("SECTION 7: Summary and Recommendations")
print("="*80)

# Find best and worst strategies
best_ari_strategy = max(comparison_results.items(), key=lambda x: x[1]['ari'])
worst_ari_strategy = min(comparison_results.items(), key=lambda x: x[1]['ari'])

print(f"\nBest Agreement Strategy: {best_ari_strategy[0]}")
print(f"  ARI = {best_ari_strategy[1]['ari']:.4f}")
print(f"  Distance Correlation r = {best_ari_strategy[1]['r_distances']:.4f}")

print(f"\nWorst Agreement Strategy: {worst_ari_strategy[0]}")
print(f"  ARI = {worst_ari_strategy[1]['ari']:.4f}")
print(f"  Distance Correlation r = {worst_ari_strategy[1]['r_distances']:.4f}")

# Overall assessment
avg_ari = np.mean([r['ari'] for r in comparison_results.values()])
avg_r = np.mean([r['r_distances'] for r in comparison_results.values()])

print(f"\nAverage Across All Per-Metric Strategies:")
print(f"  Mean ARI = {avg_ari:.4f}")
print(f"  Mean Distance Correlation r = {avg_r:.4f}")

print("\n" + "-"*80)
print("INTERPRETATION")
print("-"*80)

if avg_ari > 0.9:
    print("\n✓ HIGH AGREEMENT: Per-metric DTW produces essentially the same results")
    print("  as multivariate DTW.")
    print("\n  Recommendation:")
    print("    - Use current multivariate DTW for simplicity (one function call)")
    print("    - Consider per-metric approach when interpretability is critical")
    print("    - Per-metric allows identifying which metrics drive clustering")
elif avg_ari > 0.5:
    print("\n≈ MODERATE AGREEMENT: Methods differ but not dramatically.")
    print("\n  Recommendation:")
    print("    - Document trade-offs between approaches")
    print("    - Per-metric may be preferred for:")
    print("      * Better interpretability (see metric contributions)")
    print("      * Avoiding cross-metric normalization artifacts")
    print("      * Flexible metric weighting based on domain knowledge")
    print("    - Multivariate may be preferred for:")
    print("      * Simpler API (single function call)")
    print("      * True multivariate distance (metrics interact)")
else:
    print("\n✗ LOW AGREEMENT: Methods produce substantially different clusterings.")
    print("\n  Recommendation:")
    print("    - Need additional validation to determine which is more biologically")
    print("      meaningful (e.g., correlation with phenotypes)")
    print("    - Investigate which embryos get reassigned and why")

print("\n" + "-"*80)
print("KEY BENEFITS OF PER-METRIC APPROACH (Regardless of Agreement)")
print("-"*80)

print("\n1. INTERPRETABILITY")
print("   Can quantify which metric drives clustering decisions")
for metric in METRICS:
    contrib = comparison_results[best_strategy]['metric_contributions'][metric]
    print(f"   - {metric}: r = {contrib:.3f}")

print("\n2. ROBUSTNESS")
print("   Each metric uses its own scale (no cross-metric normalization)")
print("   May be more stable across experiments with different value ranges")

print("\n3. FLEXIBILITY")
print("   Can weight metrics based on domain knowledge:")
for strategy in ['weighted_70_30', 'weighted_50_50']:
    if strategy in comparison_results:
        ari = comparison_results[strategy]['ari']
        print(f"   - {strategy}: ARI = {ari:.3f}")

print("\n4. DEBUGGING")
print("   Can identify if one metric is problematic or uninformative")

# Write summary to file
print("\n" + "="*80)
print("Writing summary to file...")

summary_file = OUTPUT_DIR / '04g_results_summary.txt'
with open(summary_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("Tutorial 04g: Per-Metric DTW Investigation - Results Summary\n")
    f.write("="*80 + "\n\n")

    f.write(f"Experiment: {EXPERIMENT}\n")
    f.write(f"Metrics: {', '.join(METRICS)}\n")
    f.write(f"Number of embryos: {len(ids_multi)}\n")
    f.write(f"Number of clusters: {N_CLUSTERS}\n\n")

    f.write("-"*80 + "\n")
    f.write("COMPARISON RESULTS\n")
    f.write("-"*80 + "\n\n")

    for strategy_name, results in comparison_results.items():
        f.write(f"\nStrategy: {strategy_name}\n")
        f.write(f"  ARI (Cluster Agreement): {results['ari']:.4f}\n")
        f.write(f"  Distance Correlation (r): {results['r_distances']:.4f}\n")
        f.write(f"  Dominant Metric: {results['dominant_metric']}\n")
        f.write(f"  Metric Contributions:\n")
        for metric, contrib in results['metric_contributions'].items():
            f.write(f"    - {metric}: r = {contrib:.4f}\n")

    f.write("\n" + "-"*80 + "\n")
    f.write("OVERALL ASSESSMENT\n")
    f.write("-"*80 + "\n\n")
    f.write(f"Best Strategy: {best_ari_strategy[0]} (ARI = {best_ari_strategy[1]['ari']:.4f})\n")
    f.write(f"Average ARI: {avg_ari:.4f}\n")
    f.write(f"Average Distance Correlation: {avg_r:.4f}\n")

print(f"Saved: {summary_file}")

print("\n" + "="*80)
print("TUTORIAL COMPLETE")
print("="*80)
print(f"\nOutput files saved to: {OUTPUT_DIR}")
print("  - 04g_distance_comparison.png")
print("  - 04g_cluster_comparison.png")
print("  - 04g_metric_contributions.png")
print("  - 04g_strategy_comparison.png")
print("  - 04g_results_summary.txt")
print("\n" + "="*80)
