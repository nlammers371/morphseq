"""
Tutorial 04h: Cross-Experiment Validation of Per-Metric DTW

Research Question:
Does per-metric DTW reduce cross-experiment cluster assignment disagreement
compared to multivariate DTW?

Background:
- Tutorial 04d found ~25% disagreement when projecting experiments separately
  vs clustering all together (multivariate DTW)
- Root cause: Global Z-score normalization depends on time window coverage
- Hypothesis: Per-metric DTW uses raw values → more robust

Test Design:
1. Select reference experiment (20250512 - good coverage)
2. Cluster reference with both methods (multivariate, per-metric euclidean)
3. Add new experiment (different time window)
4. Compare cluster assignments: reference-only vs reference+new combined
5. Measure disagreement for each method

Expected Outcome:
- Multivariate DTW: ~25% disagreement (known from 04d)
- Per-metric DTW: <10% disagreement (hypothesis)

Success Criteria:
- If per-metric ARI > 0.90 (vs multivariate ARI ~0.75) → strong evidence to switch
- If per-metric ARI ~0.75 → no improvement, keep multivariate
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import adjusted_rand_score, confusion_matrix
import warnings

# Import core functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

from analyze.trajectory_analysis.distance import compute_trajectory_distances
from analyze.utils.timeseries.dtw import compute_md_dtw_distance_matrix
from analyze.trajectory_analysis.utilities.dtw_utils import prepare_multivariate_array

# Setup paths
DATA_FILE = Path(__file__).parent.parent.parent / 'mcolon' / '20251229_cep290_phenotype_extraction' / 'final_data' / 'embryo_data_with_labels.csv'
OUTPUT_DIR = Path(__file__).parent / 'output' / '04h'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'figures').mkdir(exist_ok=True)
(OUTPUT_DIR / 'figures' / 'confusion_matrices').mkdir(exist_ok=True)
(OUTPUT_DIR / 'results').mkdir(exist_ok=True)
(OUTPUT_DIR / 'logs').mkdir(exist_ok=True)

# Redirect output to log file
import sys
log_file = open(OUTPUT_DIR / 'logs' / 'run.log', 'w')
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

print("="*80)
print("TUTORIAL 04h: Cross-Experiment Validation of Per-Metric DTW")
print("="*80)

# ============================================================================
# SECTION 1: Helper Functions
# ============================================================================

print("\n" + "="*80)
print("SECTION 1: Helper Functions")
print("="*80)

def normalize_distance_matrix(D: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """Normalize a distance matrix."""
    if method == 'minmax':
        D_min = D.min()
        D_max = D.max()
        if D_max == D_min:
            return np.zeros_like(D)
        return (D - D_min) / (D_max - D_min)
    elif method == 'zscore':
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
    combination_method: str = 'euclidean',
    normalize_distances: bool = True,
    normalize_method: str = 'minmax',
    sakoe_chiba_radius: Optional[int] = 20,
    time_col: str = 'predicted_stage_hpf',
    embryo_id_col: str = 'embryo_id',
    verbose: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute DTW separately for each metric and combine distances.

    Returns
    -------
    D_combined : np.ndarray
        Combined distance matrix
    embryo_ids : List[str]
        List of embryo IDs
    """
    print(f"\nComputing per-metric DTW:")
    print(f"  Metrics: {metrics}")
    print(f"  Combination: {combination_method}")
    print(f"  Normalize: {normalize_distances} ({normalize_method if normalize_distances else 'N/A'})")

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
            verbose=False
        )

        print(f"    Array shape: {X_metric.shape}")
        print(f"    Value range: [{np.nanmin(X_metric):.3f}, {np.nanmax(X_metric):.3f}]")

        # Ensure consistent embryo ordering
        if embryo_ids is None:
            embryo_ids = ids
            time_grid = grid
        else:
            if ids != embryo_ids:
                raise ValueError(f"Embryo IDs differ between metrics")

        # Compute DTW distance matrix
        D_metric = compute_md_dtw_distance_matrix(
            X_metric,
            sakoe_chiba_radius=sakoe_chiba_radius
        )

        print(f"    Distance range: [{D_metric.min():.3f}, {D_metric.max():.3f}]")

        per_metric_distances[metric] = D_metric

    # Normalize distance matrices if requested
    if normalize_distances:
        print(f"\nNormalizing distance matrices ({normalize_method})...")
        for metric in metrics:
            per_metric_distances[metric] = normalize_distance_matrix(
                per_metric_distances[metric],
                method=normalize_method
            )

    # Combine distance matrices
    print(f"\nCombining distance matrices ({combination_method})...")
    distance_arrays = [per_metric_distances[m] for m in metrics]

    if combination_method == 'mean':
        D_combined = np.mean(distance_arrays, axis=0)
    elif combination_method == 'euclidean':
        D_combined = np.sqrt(np.sum([D**2 for D in distance_arrays], axis=0))
    else:
        raise ValueError(f"Unknown combination method: {combination_method}")

    print(f"  Combined distance range: [{D_combined.min():.3f}, {D_combined.max():.3f}]")

    return D_combined, embryo_ids


def cluster_from_distance_matrix(
    D: np.ndarray,
    embryo_ids: List[str],
    n_clusters: int = 5,
    linkage_method: str = 'ward'
) -> Dict[str, int]:
    """
    Perform hierarchical clustering from distance matrix.

    Returns
    -------
    cluster_map : Dict[str, int]
        Mapping from embryo_id to cluster label
    """
    # Hierarchical clustering
    Z = linkage(D[np.triu_indices(len(D), k=1)], method=linkage_method)
    cluster_labels = fcluster(Z, n_clusters, criterion='maxclust')

    # Create mapping
    cluster_map = {eid: int(label) for eid, label in zip(embryo_ids, cluster_labels)}

    return cluster_map


def compute_disagreement(
    cluster_map_ref: Dict[str, int],
    cluster_map_combined: Dict[str, int],
    ref_embryo_ids: List[str]
) -> Tuple[float, np.ndarray, List[str]]:
    """
    Compute cluster assignment disagreement between reference-only
    and reference+new combined clustering.

    Returns
    -------
    ari : float
        Adjusted Rand Index (1.0 = perfect agreement, 0.0 = random)
    confusion : np.ndarray
        Confusion matrix
    labels : List[str]
        Cluster labels
    """
    # Get cluster assignments for reference embryos only
    clusters_ref = [cluster_map_ref[eid] for eid in ref_embryo_ids]
    clusters_combined = [cluster_map_combined[eid] for eid in ref_embryo_ids]

    # Compute ARI
    ari = adjusted_rand_score(clusters_ref, clusters_combined)

    # Compute confusion matrix
    conf = confusion_matrix(clusters_ref, clusters_combined)

    # Get unique labels
    labels = sorted(set(clusters_ref) | set(clusters_combined))

    return ari, conf, labels


# ============================================================================
# SECTION 2: Data Loading
# ============================================================================

print("\n" + "="*80)
print("SECTION 2: Data Loading")
print("="*80)

# Load data
print(f"\nLoading data from: {DATA_FILE}")
df_full = pd.read_csv(DATA_FILE, low_memory=False)

# Define experiments to test
REFERENCE_EXP = '20250512'
TEST_EXPERIMENTS = ['20251017_combined', '20251106', '20251112']

# Metrics to use
METRICS = ['baseline_deviation_normalized', 'total_length_um']
MIN_TIMEPOINTS = 50

print(f"\nReference experiment: {REFERENCE_EXP}")
print(f"Test experiments: {TEST_EXPERIMENTS}")
print(f"Metrics: {METRICS}")
print(f"Min timepoints: {MIN_TIMEPOINTS}")

# Filter and prepare reference experiment
df_ref = df_full[df_full['experiment_id'] == REFERENCE_EXP].copy()
embryo_counts_ref = df_ref.groupby('embryo_id').size()
valid_embryos_ref = embryo_counts_ref[embryo_counts_ref >= MIN_TIMEPOINTS].index
df_ref = df_ref[df_ref['embryo_id'].isin(valid_embryos_ref)].copy()

print(f"\nReference experiment ({REFERENCE_EXP}):")
print(f"  Total embryos: {len(valid_embryos_ref)}")
print(f"  Time range: {df_ref['predicted_stage_hpf'].min():.1f} - {df_ref['predicted_stage_hpf'].max():.1f} hpf")

# Check which test experiments are available
available_test_exps = []
for exp in TEST_EXPERIMENTS:
    if exp in df_full['experiment_id'].values:
        available_test_exps.append(exp)
        df_exp = df_full[df_full['experiment_id'] == exp].copy()
        embryo_counts = df_exp.groupby('embryo_id').size()
        valid_embryos = embryo_counts[embryo_counts >= MIN_TIMEPOINTS].index
        print(f"\nTest experiment ({exp}):")
        print(f"  Total embryos: {len(valid_embryos)}")
        if len(valid_embryos) > 0:
            df_exp_filtered = df_exp[df_exp['embryo_id'].isin(valid_embryos)]
            print(f"  Time range: {df_exp_filtered['predicted_stage_hpf'].min():.1f} - {df_exp_filtered['predicted_stage_hpf'].max():.1f} hpf")
    else:
        print(f"\nTest experiment ({exp}): NOT FOUND")

if len(available_test_exps) == 0:
    print("\n" + "="*80)
    print("ERROR: No test experiments found!")
    print("="*80)
    sys.exit(1)

print(f"\nWill test with {len(available_test_exps)} experiments: {available_test_exps}")

# ============================================================================
# SECTION 3: Reference-Only Clustering
# ============================================================================

print("\n" + "="*80)
print("SECTION 3: Reference-Only Clustering")
print("="*80)

N_CLUSTERS = 5

print(f"\nClustering reference experiment with both methods...")
print(f"Number of clusters: {N_CLUSTERS}")

# Method 1: Multivariate DTW (current approach)
print("\n" + "-"*80)
print("Method 1: Multivariate DTW")
print("-"*80)

D_multi_ref, embryo_ids_multi_ref, _ = compute_trajectory_distances(
    df_ref,
    metrics=METRICS,
    normalize=True,
    sakoe_chiba_radius=20,
    time_col='predicted_stage_hpf',
    embryo_id_col='embryo_id',
    verbose=True
)

print(f"\nMultivariate DTW - Reference only:")
print(f"  Distance matrix shape: {D_multi_ref.shape}")
print(f"  Distance range: [{D_multi_ref.min():.3f}, {D_multi_ref.max():.3f}]")

cluster_map_multi_ref = cluster_from_distance_matrix(
    D_multi_ref,
    embryo_ids_multi_ref,
    n_clusters=N_CLUSTERS
)

# Print cluster sizes
cluster_sizes_multi_ref = pd.Series(list(cluster_map_multi_ref.values())).value_counts().sort_index()
print(f"\nCluster sizes (multivariate):")
for cluster_id, size in cluster_sizes_multi_ref.items():
    print(f"  Cluster {cluster_id}: {size} embryos ({100*size/len(cluster_map_multi_ref):.1f}%)")

# Method 2: Per-Metric DTW (proposed approach)
print("\n" + "-"*80)
print("Method 2: Per-Metric DTW")
print("-"*80)

D_per_ref, embryo_ids_per_ref = compute_per_metric_dtw(
    df_ref,
    metrics=METRICS,
    combination_method='euclidean',
    normalize_distances=True,
    normalize_method='minmax',
    sakoe_chiba_radius=20,
    verbose=True
)

print(f"\nPer-Metric DTW - Reference only:")
print(f"  Distance matrix shape: {D_per_ref.shape}")
print(f"  Distance range: [{D_per_ref.min():.3f}, {D_per_ref.max():.3f}]")

cluster_map_per_ref = cluster_from_distance_matrix(
    D_per_ref,
    embryo_ids_per_ref,
    n_clusters=N_CLUSTERS
)

# Print cluster sizes
cluster_sizes_per_ref = pd.Series(list(cluster_map_per_ref.values())).value_counts().sort_index()
print(f"\nCluster sizes (per-metric):")
for cluster_id, size in cluster_sizes_per_ref.items():
    print(f"  Cluster {cluster_id}: {size} embryos ({100*size/len(cluster_map_per_ref):.1f}%)")

# Compare reference-only clusterings
ari_ref_comparison = adjusted_rand_score(
    list(cluster_map_multi_ref.values()),
    list(cluster_map_per_ref.values())
)
print(f"\nAgreement between methods on reference-only: ARI = {ari_ref_comparison:.3f}")
print("(This should be ~0.87-0.89 based on Tutorial 04g)")

# ============================================================================
# SECTION 4: Cross-Experiment Validation
# ============================================================================

print("\n" + "="*80)
print("SECTION 4: Cross-Experiment Validation")
print("="*80)

# Store results
results = []

for test_exp in available_test_exps:
    print("\n" + "="*80)
    print(f"Testing with: {REFERENCE_EXP} + {test_exp}")
    print("="*80)

    # Prepare test experiment data
    df_test = df_full[df_full['experiment_id'] == test_exp].copy()
    embryo_counts_test = df_test.groupby('embryo_id').size()
    valid_embryos_test = embryo_counts_test[embryo_counts_test >= MIN_TIMEPOINTS].index
    df_test = df_test[df_test['embryo_id'].isin(valid_embryos_test)].copy()

    print(f"\nTest experiment ({test_exp}):")
    print(f"  Total embryos: {len(valid_embryos_test)}")

    if len(valid_embryos_test) == 0:
        print(f"  Skipping {test_exp} (no valid embryos)")
        continue

    # Combine datasets
    df_combined = pd.concat([df_ref, df_test], ignore_index=True)
    print(f"\nCombined dataset:")
    print(f"  Total embryos: {len(df_combined['embryo_id'].unique())}")
    print(f"  Reference embryos: {len(valid_embryos_ref)}")
    print(f"  Test embryos: {len(valid_embryos_test)}")

    # --- Multivariate DTW: Combined clustering ---
    print("\n" + "-"*80)
    print("Multivariate DTW: Reference + Test Combined")
    print("-"*80)

    D_multi_combined, embryo_ids_multi_combined, _ = compute_trajectory_distances(
        df_combined,
        metrics=METRICS,
        normalize=True,
        sakoe_chiba_radius=20,
        time_col='predicted_stage_hpf',
        embryo_id_col='embryo_id',
        verbose=False
    )

    cluster_map_multi_combined = cluster_from_distance_matrix(
        D_multi_combined,
        embryo_ids_multi_combined,
        n_clusters=N_CLUSTERS
    )

    # Compute disagreement for multivariate DTW
    ari_multi, conf_multi, labels_multi = compute_disagreement(
        cluster_map_multi_ref,
        cluster_map_multi_combined,
        list(valid_embryos_ref)
    )

    print(f"\nMultivariate DTW Disagreement:")
    print(f"  ARI: {ari_multi:.3f}")
    print(f"  Interpretation: {100*(1-ari_multi):.1f}% disagreement")

    # --- Per-Metric DTW: Combined clustering ---
    print("\n" + "-"*80)
    print("Per-Metric DTW: Reference + Test Combined")
    print("-"*80)

    D_per_combined, embryo_ids_per_combined = compute_per_metric_dtw(
        df_combined,
        metrics=METRICS,
        combination_method='euclidean',
        normalize_distances=True,
        normalize_method='minmax',
        sakoe_chiba_radius=20,
        verbose=False
    )

    cluster_map_per_combined = cluster_from_distance_matrix(
        D_per_combined,
        embryo_ids_per_combined,
        n_clusters=N_CLUSTERS
    )

    # Compute disagreement for per-metric DTW
    ari_per, conf_per, labels_per = compute_disagreement(
        cluster_map_per_ref,
        cluster_map_per_combined,
        list(valid_embryos_ref)
    )

    print(f"\nPer-Metric DTW Disagreement:")
    print(f"  ARI: {ari_per:.3f}")
    print(f"  Interpretation: {100*(1-ari_per):.1f}% disagreement")

    # --- Comparison ---
    print("\n" + "-"*80)
    print("Comparison")
    print("-"*80)

    improvement = ari_per - ari_multi
    print(f"\nMultivariate DTW ARI: {ari_multi:.3f} ({100*(1-ari_multi):.1f}% disagreement)")
    print(f"Per-Metric DTW ARI:   {ari_per:.3f} ({100*(1-ari_per):.1f}% disagreement)")
    print(f"Improvement: {improvement:+.3f} ARI points")

    if improvement > 0.10:
        print("→ STRONG IMPROVEMENT: Per-metric DTW is more robust")
    elif improvement > 0.05:
        print("→ MODERATE IMPROVEMENT: Per-metric DTW shows benefit")
    elif improvement > -0.05:
        print("→ NO SIGNIFICANT DIFFERENCE: Methods are equivalent")
    else:
        print("→ WORSE: Multivariate DTW is more robust")

    # Store results
    results.append({
        'test_experiment': test_exp,
        'n_ref_embryos': len(valid_embryos_ref),
        'n_test_embryos': len(valid_embryos_test),
        'ari_multivariate': ari_multi,
        'ari_per_metric': ari_per,
        'disagreement_multivariate_pct': 100 * (1 - ari_multi),
        'disagreement_per_metric_pct': 100 * (1 - ari_per),
        'improvement_ari': improvement
    })

    # --- Visualize confusion matrices ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Multivariate confusion matrix
    sns.heatmap(conf_multi, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=labels_multi, yticklabels=labels_multi)
    axes[0].set_title(f'Multivariate DTW\nARI = {ari_multi:.3f}')
    axes[0].set_xlabel('Combined Clustering')
    axes[0].set_ylabel('Reference-Only Clustering')

    # Per-metric confusion matrix
    sns.heatmap(conf_per, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
                xticklabels=labels_per, yticklabels=labels_per)
    axes[1].set_title(f'Per-Metric DTW\nARI = {ari_per:.3f}')
    axes[1].set_xlabel('Combined Clustering')
    axes[1].set_ylabel('Reference-Only Clustering')

    plt.suptitle(f'Cluster Reassignment: {REFERENCE_EXP} + {test_exp}', fontsize=14, y=1.02)
    plt.tight_layout()

    output_path = OUTPUT_DIR / 'figures' / 'confusion_matrices' / f'confusion_{test_exp}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved confusion matrix to: {output_path}")
    plt.close()

# ============================================================================
# SECTION 5: Summary and Visualization
# ============================================================================

print("\n" + "="*80)
print("SECTION 5: Summary and Visualization")
print("="*80)

# Create results dataframe
df_results = pd.DataFrame(results)

print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)
print(df_results.to_string(index=False))

# Save results
results_path = OUTPUT_DIR / 'results' / 'disagreement_summary.csv'
df_results.to_csv(results_path, index=False)
print(f"\nResults saved to: {results_path}")

# Overall statistics
print("\n" + "-"*80)
print("Overall Statistics")
print("-"*80)
print(f"\nMultivariate DTW:")
print(f"  Mean ARI: {df_results['ari_multivariate'].mean():.3f}")
print(f"  Mean disagreement: {df_results['disagreement_multivariate_pct'].mean():.1f}%")
print(f"\nPer-Metric DTW:")
print(f"  Mean ARI: {df_results['ari_per_metric'].mean():.3f}")
print(f"  Mean disagreement: {df_results['disagreement_per_metric_pct'].mean():.1f}%")
print(f"\nMean improvement: {df_results['improvement_ari'].mean():+.3f} ARI points")

# Create comparison plot
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(df_results))
width = 0.35

bars1 = ax.bar(x - width/2, df_results['ari_multivariate'], width,
               label='Multivariate DTW', color='#1f77b4', alpha=0.8)
bars2 = ax.bar(x + width/2, df_results['ari_per_metric'], width,
               label='Per-Metric DTW', color='#ff7f0e', alpha=0.8)

ax.set_xlabel('Test Experiment', fontsize=12)
ax.set_ylabel('Adjusted Rand Index (ARI)', fontsize=12)
ax.set_title('Cross-Experiment Robustness: Multivariate vs Per-Metric DTW', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(df_results['test_experiment'], rotation=45, ha='right')
ax.legend()
ax.axhline(y=0.75, color='red', linestyle='--', alpha=0.5, label='25% disagreement threshold')
ax.axhline(y=0.90, color='green', linestyle='--', alpha=0.5, label='10% disagreement threshold')
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
comparison_path = OUTPUT_DIR / 'figures' / 'disagreement_comparison.png'
plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
print(f"\nComparison plot saved to: {comparison_path}")
plt.close()

# ============================================================================
# SECTION 6: Recommendation
# ============================================================================

print("\n" + "="*80)
print("SECTION 6: RECOMMENDATION")
print("="*80)

mean_ari_multi = df_results['ari_multivariate'].mean()
mean_ari_per = df_results['ari_per_metric'].mean()
mean_improvement = df_results['improvement_ari'].mean()

print("\nBased on cross-experiment validation results:\n")

if mean_ari_per > 0.90 and mean_improvement > 0.10:
    print("✅ RECOMMENDATION: SWITCH TO PER-METRIC DTW AS DEFAULT")
    print("\nRationale:")
    print(f"  • Per-metric DTW achieves {mean_ari_per:.3f} ARI (< 10% disagreement)")
    print(f"  • {mean_improvement:+.3f} ARI improvement over multivariate DTW")
    print(f"  • Much more robust to cross-experiment normalization issues")
    print("\nNext steps:")
    print("  1. Add compute_per_metric_dtw() to src/analyze/trajectory_analysis/distance.py")
    print("  2. Add method='multivariate'|'per_metric' parameter to compute_trajectory_distances()")
    print("  3. Update projection functions to use per-metric DTW by default")
    print("  4. Update tutorials 01-09 to demonstrate per-metric approach")

elif mean_ari_per > 0.85 and mean_improvement > 0.05:
    print("⚠️  RECOMMENDATION: OFFER BOTH METHODS, DOCUMENT TRADE-OFFS")
    print("\nRationale:")
    print(f"  • Per-metric DTW shows moderate improvement ({mean_improvement:+.3f} ARI)")
    print(f"  • Still some disagreement ({100*(1-mean_ari_per):.1f}%) but better than multivariate ({100*(1-mean_ari_multi):.1f}%)")
    print(f"  • Trade-off between simplicity (multivariate) and robustness (per-metric)")
    print("\nNext steps:")
    print("  1. Add per-metric as optional method in core library")
    print("  2. Document: 'Use per-metric for cross-experiment robustness'")
    print("  3. Keep multivariate as default for backward compatibility")

else:
    print("❌ RECOMMENDATION: KEEP MULTIVARIATE DTW AS DEFAULT")
    print("\nRationale:")
    print(f"  • Per-metric DTW does not significantly reduce disagreement")
    print(f"  • Mean ARI: {mean_ari_per:.3f} (per-metric) vs {mean_ari_multi:.3f} (multivariate)")
    print(f"  • Improvement: {mean_improvement:+.3f} ARI (not substantial)")
    print(f"  • Multivariate DTW is simpler and equally robust")
    print("\nNext steps:")
    print("  1. Document per-metric DTW in tutorials as interpretability tool only")
    print("  2. Do NOT add to core library (complexity not justified)")
    print("  3. Explore other solutions: separate projections, K-NN posteriors")

print("\n" + "="*80)
print("TUTORIAL 04h COMPLETE")
print("="*80)
print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print("\nGenerated files:")
print(f"  • Figures: {OUTPUT_DIR / 'figures'}")
print(f"  • Results: {OUTPUT_DIR / 'results' / 'disagreement_summary.csv'}")
print(f"  • Logs: {OUTPUT_DIR / 'logs' / 'run.log'}")

log_file.close()
