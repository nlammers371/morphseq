"""
Tutorial 04f: Clustering with vs without Normalization

Simple question: For CLUSTERING a single experiment (not projection),
do raw vs normalized values give the same cluster structure?

Test:
----
1. Cluster 20260122 with normalize=True (current default)
2. Cluster 20260122 with normalize=False (raw values)
3. Compare:
   - Distance matrix correlation
   - Cluster assignments (after hierarchical clustering)
   - Cluster distribution

Key Question:
-------------
For single-metric clustering, can we just use raw values and avoid the
normalization dependency issue entirely?
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.analyze.trajectory_analysis import (
    prepare_multivariate_array,
    compute_md_dtw_distance_matrix,
)
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import spearmanr

# Setup
OUTPUT_DIR = Path(__file__).parent / "output"
FIGURES_DIR = OUTPUT_DIR / "figures" / "04f"
RESULTS_DIR = OUTPUT_DIR / "results"

for d in [FIGURES_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

METRICS = ['baseline_deviation_normalized']
SAKOE_CHIBA_RADIUS = 20
N_CLUSTERS = 4  # For CEP290 phenotypes

print("="*80)
print("Tutorial 04f: Clustering with/without Normalization")
print("="*80)

# Load data
meta_dir = project_root / 'morphseq_playground' / 'metadata' / 'build04_output'
df = pd.read_csv(meta_dir / 'qc_staged_20260122.csv')
df = df[df['use_embryo_flag']].copy()

print(f"\nData: {df['embryo_id'].nunique()} embryos from experiment 20260122")
print(f"Time range: {df['predicted_stage_hpf'].min():.1f} - {df['predicted_stage_hpf'].max():.1f} hpf")
print(f"Genotypes: {sorted(df['genotype'].unique())}")

# ============================================================================
# Method A: WITH Normalization (current default)
# ============================================================================
print("\n" + "="*80)
print("METHOD A: WITH Z-SCORE NORMALIZATION")
print("="*80)

X_norm, embryo_ids_norm, time_grid = prepare_multivariate_array(
    df,
    metrics=METRICS,
    normalize=True,
    verbose=False
)

print(f"Array shape: {X_norm.shape}")
print(f"Value range: [{np.nanmin(X_norm):.4f}, {np.nanmax(X_norm):.4f}]")
print(f"Mean: {np.nanmean(X_norm):.4f}, Std: {np.nanstd(X_norm):.4f}")

D_norm = compute_md_dtw_distance_matrix(
    X_norm,
    sakoe_chiba_radius=SAKOE_CHIBA_RADIUS,
    verbose=False
)

print(f"Distance matrix shape: {D_norm.shape}")
print(f"Distance range: [{D_norm[D_norm > 0].min():.4f}, {D_norm.max():.4f}]")

# Cluster
Z_norm = linkage(D_norm[np.triu_indices(len(D_norm), k=1)], method='average')
clusters_norm = fcluster(Z_norm, N_CLUSTERS, criterion='maxclust')

print(f"\nClusters (n={N_CLUSTERS}):")
for c in range(1, N_CLUSTERS+1):
    count = (clusters_norm == c).sum()
    pct = count / len(clusters_norm) * 100
    print(f"  Cluster {c}: {count} ({pct:.1f}%)")

# ============================================================================
# Method B: WITHOUT Normalization (raw values)
# ============================================================================
print("\n" + "="*80)
print("METHOD B: WITHOUT NORMALIZATION (RAW VALUES)")
print("="*80)

X_raw, embryo_ids_raw, _ = prepare_multivariate_array(
    df,
    metrics=METRICS,
    normalize=False,
    verbose=False
)

print(f"Array shape: {X_raw.shape}")
print(f"Value range: [{np.nanmin(X_raw):.4f}, {np.nanmax(X_raw):.4f}]")
print(f"Mean: {np.nanmean(X_raw):.4f}, Std: {np.nanstd(X_raw):.4f}")

D_raw = compute_md_dtw_distance_matrix(
    X_raw,
    sakoe_chiba_radius=SAKOE_CHIBA_RADIUS,
    verbose=False
)

print(f"Distance matrix shape: {D_raw.shape}")
print(f"Distance range: [{D_raw[D_raw > 0].min():.4f}, {D_raw.max():.4f}]")

# Cluster
Z_raw = linkage(D_raw[np.triu_indices(len(D_raw), k=1)], method='average')
clusters_raw = fcluster(Z_raw, N_CLUSTERS, criterion='maxclust')

print(f"\nClusters (n={N_CLUSTERS}):")
for c in range(1, N_CLUSTERS+1):
    count = (clusters_raw == c).sum()
    pct = count / len(clusters_raw) * 100
    print(f"  Cluster {c}: {count} ({pct:.1f}%)")

# ============================================================================
# Compare
# ============================================================================
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

# Distance matrix correlation
corr_dist = np.corrcoef(D_norm.flatten(), D_raw.flatten())[0, 1]
print(f"\nDistance matrix correlation: {corr_dist:.6f}")

# Rank correlation (are embryos ranked similarly?)
rank_corrs = []
for i in range(D_norm.shape[0]):
    rho, _ = spearmanr(D_norm[i, :], D_raw[i, :])
    rank_corrs.append(rho)
mean_rank_corr = np.mean(rank_corrs)
print(f"Mean Spearman rank correlation: {mean_rank_corr:.6f}")

# Cluster assignment agreement (need to match cluster labels)
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ari = adjusted_rand_score(clusters_norm, clusters_raw)
nmi = normalized_mutual_info_score(clusters_norm, clusters_raw)

print(f"\nCluster similarity:")
print(f"  Adjusted Rand Index: {ari:.4f} (1.0 = identical, 0.0 = random)")
print(f"  Normalized Mutual Info: {nmi:.4f} (1.0 = identical, 0.0 = independent)")

# Create confusion matrix
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(clusters_norm, clusters_raw)

print(f"\nConfusion matrix (Normalized vs Raw):")
print(f"         Raw: ", end='')
for c in range(1, N_CLUSTERS+1):
    print(f"  C{c}", end='')
print()
for i, c in enumerate(range(1, N_CLUSTERS+1)):
    print(f"  Norm C{c}:", end='')
    for j in range(N_CLUSTERS):
        print(f" {conf_mat[i, j]:4d}", end='')
    print()

# Check if cluster distributions are similar (Chi-square test)
from scipy.stats import chisquare

dist_norm = [(clusters_norm == c).sum() for c in range(1, N_CLUSTERS+1)]
dist_raw = [(clusters_raw == c).sum() for c in range(1, N_CLUSTERS+1)]

chi2, p_value = chisquare(dist_norm, dist_raw)
print(f"\nCluster distribution similarity:")
print(f"  Chi-square: {chi2:.4f}, p-value: {p_value:.4f}")
print(f"  Normalized: {dist_norm}")
print(f"  Raw:        {dist_raw}")

# ============================================================================
# Interpretation
# ============================================================================
print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

if ari > 0.9:
    print("\n✓ HIGHLY SIMILAR: Clusters are nearly identical!")
    print(f"  ARI = {ari:.4f} (>0.9)")
    print("  → Normalization doesn't meaningfully change clustering")
    print("  → For single-metric analysis, raw values are fine!")
elif ari > 0.7:
    print("\n⚠ MODERATELY SIMILAR: Clusters are similar but not identical")
    print(f"  ARI = {ari:.4f} (0.7-0.9)")
    print("  → Normalization affects clustering somewhat")
    print("  → Results are broadly consistent")
elif ari > 0.5:
    print("\n⚠ SOMEWHAT SIMILAR: Noticeable differences")
    print(f"  ARI = {ari:.4f} (0.5-0.7)")
    print("  → Normalization affects clustering")
else:
    print("\n✗ VERY DIFFERENT: Clusters differ significantly")
    print(f"  ARI = {ari:.4f} (<0.5)")
    print("  → Normalization strongly affects clustering")
    print("  → Cannot skip normalization")

if p_value > 0.05:
    print(f"\nCluster distributions are similar (p={p_value:.4f} > 0.05)")
    print("  → Proportions in each cluster are comparable")
else:
    print(f"\nCluster distributions differ (p={p_value:.4f} < 0.05)")
    print("  → Different proportions in clusters")

# ============================================================================
# Visualization
# ============================================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Distance matrix correlation
ax1 = axes[0, 0]
ax1.scatter(D_norm.flatten(), D_raw.flatten(), alpha=0.2, s=10)
max_val = max(D_norm.max(), D_raw.max())
ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
ax1.set_xlabel('Distance (Normalized)', fontsize=11)
ax1.set_ylabel('Distance (Raw)', fontsize=11)
ax1.set_title(f'Distance Correlation\nr={corr_dist:.4f}', fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3)

# Confusion matrix
ax2 = axes[0, 1]
im = ax2.imshow(conf_mat, cmap='Blues', aspect='auto')
ax2.set_xlabel('Raw Clusters', fontsize=11)
ax2.set_ylabel('Normalized Clusters', fontsize=11)
ax2.set_title(f'Cluster Assignment Confusion\nARI={ari:.4f}', fontsize=12, fontweight='bold')
ax2.set_xticks(range(N_CLUSTERS))
ax2.set_yticks(range(N_CLUSTERS))
ax2.set_xticklabels([f'C{i+1}' for i in range(N_CLUSTERS)])
ax2.set_yticklabels([f'C{i+1}' for i in range(N_CLUSTERS)])

# Add counts to cells
for i in range(N_CLUSTERS):
    for j in range(N_CLUSTERS):
        text = ax2.text(j, i, conf_mat[i, j],
                       ha="center", va="center",
                       color="white" if conf_mat[i, j] > conf_mat.max()/2 else "black")
plt.colorbar(im, ax=ax2)

# Cluster distributions
ax3 = axes[1, 0]
x = np.arange(N_CLUSTERS)
width = 0.35
ax3.bar(x - width/2, dist_norm, width, label='Normalized', alpha=0.7)
ax3.bar(x + width/2, dist_raw, width, label='Raw', alpha=0.7)
ax3.set_xlabel('Cluster', fontsize=11)
ax3.set_ylabel('Count', fontsize=11)
ax3.set_title(f'Cluster Distributions\nχ²={chi2:.2f}, p={p_value:.4f}', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels([f'C{i+1}' for i in range(N_CLUSTERS)])
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# Summary stats
ax4 = axes[1, 1]
summary_text = f'''CLUSTERING COMPARISON

Similarity Metrics:
  ARI:  {ari:.4f}
  NMI:  {nmi:.4f}

Distance Correlation:
  Pearson:  {corr_dist:.4f}
  Spearman: {mean_rank_corr:.4f}

Distribution Test:
  χ² = {chi2:.4f}
  p  = {p_value:.4f}

Recommendation:
'''

if ari > 0.9:
    summary_text += "  ✓ Raw values OK\n  for single metric!"
elif ari > 0.7:
    summary_text += "  ⚠ Similar but\n  prefer normalized"
else:
    summary_text += "  ✗ Keep using\n  normalization"

ax4.text(0.5, 0.5, summary_text,
        ha='center', va='center', fontsize=12, family='monospace',
        transform=ax4.transAxes,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax4.axis('off')

plt.suptitle('Clustering: Normalized vs Raw Features', fontsize=14, fontweight='bold')
plt.tight_layout()

plot_path = FIGURES_DIR / 'clustering_normalization_comparison.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: {plot_path}")
plt.close()

# Save results
results = pd.DataFrame([{
    'method': 'normalized',
    'n_embryos': len(embryo_ids_norm),
    'n_clusters': N_CLUSTERS,
    **{f'cluster_{i+1}_count': dist_norm[i] for i in range(N_CLUSTERS)},
}, {
    'method': 'raw',
    'n_embryos': len(embryo_ids_raw),
    'n_clusters': N_CLUSTERS,
    **{f'cluster_{i+1}_count': dist_raw[i] for i in range(N_CLUSTERS)},
}])

results_path = RESULTS_DIR / 'clustering_normalization_comparison.csv'
results.to_csv(results_path, index=False)
print(f"✓ Saved: {results_path}")

# Save comparison metrics
metrics = pd.DataFrame([{
    'distance_correlation_pearson': corr_dist,
    'distance_correlation_spearman': mean_rank_corr,
    'adjusted_rand_index': ari,
    'normalized_mutual_info': nmi,
    'chisquare_statistic': chi2,
    'chisquare_pvalue': p_value,
}])

metrics_path = RESULTS_DIR / 'clustering_normalization_metrics.csv'
metrics.to_csv(metrics_path, index=False)
print(f"✓ Saved: {metrics_path}")

print("\n" + "="*80)
print("✓ TUTORIAL 04f COMPLETE")
print("="*80)
