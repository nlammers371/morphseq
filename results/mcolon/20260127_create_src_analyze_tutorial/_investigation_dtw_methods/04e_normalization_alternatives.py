"""
Tutorial 04e: Normalization Alternatives Testing

Test two potential solutions to the normalization problem:

1. NO NORMALIZATION (for single metric):
   - Does raw curvature give same distances as normalized?
   - Would this eliminate the time window dependency?

2. DISTANCE NORMALIZATION (instead of feature normalization):
   - Compute distances on raw values
   - Then normalize the distance matrix
   - Does this give consistent results?

Key Question:
-------------
If we're only using ONE metric (baseline_deviation_normalized), do we even
need Z-score normalization? The metric is already "normalized" in some sense
(deviation from baseline).

Test Strategy:
-------------
For experiment 20260122 with common reference embryos:

Test 1: Raw vs Normalized Features
  - Compute distances with normalize=True
  - Compute distances with normalize=False
  - Compare: correlation, cluster assignments

Test 2: Distance Normalization
  - Compute separate distances (raw, different time windows)
  - Normalize each distance matrix to [0,1] or z-score
  - Compare if normalized distances are more consistent

If Test 1 shows high correlation, we can recommend:
  "For single-metric analysis, use normalize=False for robustness"

If Test 2 works, we could add distance normalization as an option.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.analyze.trajectory_analysis import (
    compute_cross_dtw_distance_matrix,
    prepare_multivariate_array,
)

# Setup paths
OUTPUT_DIR = Path(__file__).parent / "output"
FIGURES_DIR = OUTPUT_DIR / "figures" / "04e"
RESULTS_DIR = OUTPUT_DIR / "results"

for d in [FIGURES_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

METRICS = ['baseline_deviation_normalized']
SAKOE_CHIBA_RADIUS = 20

print("="*80)
print("Tutorial 04e: Normalization Alternatives")
print("="*80)
print("\nTesting:")
print("  1. Does single-metric analysis need normalization?")
print("  2. Can we normalize distances instead of features?")

# ============================================================================
# Load Data (just one experiment for testing)
# ============================================================================
print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

CEP290_REF_DIR = project_root / "results" / "mcolon" / "20251229_cep290_phenotype_extraction" / "final_data"
df_cep290_labels = pd.read_csv(CEP290_REF_DIR / "embryo_cluster_labels.csv", low_memory=False)
df_cep290_data = pd.read_csv(CEP290_REF_DIR / "embryo_data_with_labels.csv", low_memory=False)

labels_valid = df_cep290_labels.drop_duplicates(subset='embryo_id')
labels_valid = labels_valid[labels_valid['cluster_categories'].notna()].copy()
ref_cluster_map = dict(zip(labels_valid['embryo_id'], labels_valid['clusters']))

df_cep290_ref = df_cep290_data[df_cep290_data['embryo_id'].isin(labels_valid['embryo_id'])].copy()

meta_dir = project_root / 'morphseq_playground' / 'metadata' / 'build04_output'
df_20260122 = pd.read_csv(meta_dir / 'qc_staged_20260122.csv')
df_20260122 = df_20260122[df_20260122['use_embryo_flag']].copy()

print(f"Reference: {df_cep290_ref['embryo_id'].nunique()} embryos")
print(f"Source: {df_20260122['embryo_id'].nunique()} embryos")

# ============================================================================
# TEST 1: Raw vs Normalized Features (Single Metric)
# ============================================================================
print("\n" + "="*80)
print("TEST 1: RAW vs NORMALIZED FEATURES")
print("="*80)
print("\nFor single metric, do we need Z-score normalization?")

# Get common time window
source_min = df_20260122['predicted_stage_hpf'].min()
source_max = df_20260122['predicted_stage_hpf'].max()
ref_min = df_cep290_ref['predicted_stage_hpf'].min()
ref_max = df_cep290_ref['predicted_stage_hpf'].max()

window_start = max(source_min, ref_min)
window_end = min(source_max, ref_max)

source_filtered = df_20260122[
    (df_20260122['predicted_stage_hpf'] >= window_start) &
    (df_20260122['predicted_stage_hpf'] <= window_end)
].copy()

ref_filtered = df_cep290_ref[
    (df_cep290_ref['predicted_stage_hpf'] >= window_start) &
    (df_cep290_ref['predicted_stage_hpf'] <= window_end)
].copy()

# --- Method A: WITH normalization (current default) ---
print("\nMethod A: WITH Z-score normalization")

X_ref_norm, ref_ids, time_grid = prepare_multivariate_array(
    ref_filtered,
    metrics=METRICS,
    time_col='predicted_stage_hpf',
    embryo_id_col='embryo_id',
    normalize=True,  # Z-score normalization
    verbose=False
)

X_source_norm, source_ids, _ = prepare_multivariate_array(
    source_filtered,
    metrics=METRICS,
    time_col='predicted_stage_hpf',
    embryo_id_col='embryo_id',
    time_grid=time_grid,
    normalize=True,  # Z-score normalization
    verbose=False
)

D_normalized = compute_cross_dtw_distance_matrix(
    X_source_norm,
    X_ref_norm,
    sakoe_chiba_radius=SAKOE_CHIBA_RADIUS,
    n_jobs=-1,
    verbose=False
)

print(f"  Shape: {D_normalized.shape}")
print(f"  Range: [{D_normalized.min():.4f}, {D_normalized.max():.4f}]")
print(f"  Mean: {D_normalized.mean():.4f}")

# --- Method B: WITHOUT normalization (raw values) ---
print("\nMethod B: WITHOUT normalization (raw values)")

X_ref_raw, ref_ids_raw, time_grid_raw = prepare_multivariate_array(
    ref_filtered,
    metrics=METRICS,
    time_col='predicted_stage_hpf',
    embryo_id_col='embryo_id',
    normalize=False,  # NO normalization
    verbose=False
)

X_source_raw, source_ids_raw, _ = prepare_multivariate_array(
    source_filtered,
    metrics=METRICS,
    time_col='predicted_stage_hpf',
    embryo_id_col='embryo_id',
    time_grid=time_grid_raw,
    normalize=False,  # NO normalization
    verbose=False
)

D_raw = compute_cross_dtw_distance_matrix(
    X_source_raw,
    X_ref_raw,
    sakoe_chiba_radius=SAKOE_CHIBA_RADIUS,
    n_jobs=-1,
    verbose=False
)

print(f"  Shape: {D_raw.shape}")
print(f"  Range: [{D_raw.min():.4f}, {D_raw.max():.4f}]")
print(f"  Mean: {D_raw.mean():.4f}")

# --- Compare ---
print("\n" + "-"*80)
print("COMPARISON: Raw vs Normalized")
print("-"*80)

# Correlation of distance matrices
correlation_features = np.corrcoef(D_normalized.flatten(), D_raw.flatten())[0, 1]
print(f"\nDistance correlation: {correlation_features:.6f}")

# Nearest neighbor assignments
nn_idx_normalized = np.argmin(D_normalized, axis=1)
nn_idx_raw = np.argmin(D_raw, axis=1)

# Do they assign to same reference embryos?
nn_agreement = (nn_idx_normalized == nn_idx_raw).sum()
nn_total = len(nn_idx_normalized)
nn_agreement_pct = nn_agreement / nn_total * 100

print(f"\nNearest neighbor agreement: {nn_agreement}/{nn_total} ({nn_agreement_pct:.1f}%)")

# Get cluster assignments
clusters_normalized = [ref_cluster_map[ref_ids[idx]] for idx in nn_idx_normalized]
clusters_raw = [ref_cluster_map[ref_ids_raw[idx]] for idx in nn_idx_raw]

cluster_agreement = sum(c1 == c2 for c1, c2 in zip(clusters_normalized, clusters_raw))
cluster_agreement_pct = cluster_agreement / len(clusters_normalized) * 100

print(f"Cluster assignment agreement: {cluster_agreement}/{len(clusters_normalized)} ({cluster_agreement_pct:.1f}%)")

# Rank correlation (do they rank references similarly?)
from scipy.stats import spearmanr
rank_correlations = []
for i in range(D_normalized.shape[0]):
    rho, _ = spearmanr(D_normalized[i, :], D_raw[i, :])
    rank_correlations.append(rho)

mean_rank_corr = np.mean(rank_correlations)
print(f"\nMean Spearman rank correlation: {mean_rank_corr:.6f}")
print(f"  (Are reference embryos ranked similarly?)")

if cluster_agreement_pct > 95:
    print("\n✓ TEST 1 PASSED: Raw and normalized give same cluster assignments!")
    print("  → For single-metric analysis, normalization is NOT needed")
    print("  → Recommend using normalize=False for robustness")
elif cluster_agreement_pct > 85:
    print("\n⚠ TEST 1 MARGINAL: Mostly same assignments")
    print(f"  → {100 - cluster_agreement_pct:.1f}% differ")
    print("  → May still be acceptable for robustness")
else:
    print("\n✗ TEST 1 FAILED: Significant differences in assignments")
    print("  → Normalization does affect results")
    print("  → Cannot skip normalization")

# ============================================================================
# TEST 2: Distance Normalization Approach
# ============================================================================
print("\n" + "="*80)
print("TEST 2: NORMALIZE DISTANCES (not features)")
print("="*80)
print("\nCan we compute distances on raw values, then normalize distances?")

# We already have D_raw from experiment-specific window
# Now compute in combined window

df_combined_all = pd.concat([
    pd.read_csv(meta_dir / 'qc_staged_20260122.csv').query('use_embryo_flag'),
    pd.read_csv(meta_dir / 'qc_staged_20260124.csv').query('use_embryo_flag'),
], ignore_index=True)

combined_min = df_combined_all['predicted_stage_hpf'].min()
combined_max = df_combined_all['predicted_stage_hpf'].max()
window_start_comb = max(combined_min, ref_min)
window_end_comb = min(combined_max, ref_max)

source_filtered_comb = df_20260122[
    (df_20260122['predicted_stage_hpf'] >= window_start_comb) &
    (df_20260122['predicted_stage_hpf'] <= window_end_comb)
].copy()

ref_filtered_comb = df_cep290_ref[
    (df_cep290_ref['predicted_stage_hpf'] >= window_start_comb) &
    (df_cep290_ref['predicted_stage_hpf'] <= window_end_comb)
].copy()

X_ref_raw_comb, ref_ids_comb, time_grid_comb = prepare_multivariate_array(
    ref_filtered_comb,
    metrics=METRICS,
    time_col='predicted_stage_hpf',
    embryo_id_col='embryo_id',
    normalize=False,
    verbose=False
)

X_source_raw_comb, source_ids_comb, _ = prepare_multivariate_array(
    source_filtered_comb,
    metrics=METRICS,
    time_col='predicted_stage_hpf',
    embryo_id_col='embryo_id',
    time_grid=time_grid_comb,
    normalize=False,
    verbose=False
)

D_raw_combined = compute_cross_dtw_distance_matrix(
    X_source_raw_comb,
    X_ref_raw_comb,
    sakoe_chiba_radius=SAKOE_CHIBA_RADIUS,
    n_jobs=-1,
    verbose=False
)

print(f"\nSeparate window raw distances: {D_raw.shape}")
print(f"Combined window raw distances: {D_raw_combined.shape}")

# Find common references
common_refs = sorted(set(ref_ids_raw) & set(ref_ids_comb))
ref_idx_sep = [ref_ids_raw.index(rid) for rid in common_refs]
ref_idx_comb = [ref_ids_comb.index(rid) for rid in common_refs]

D_raw_sep_common = D_raw[:, ref_idx_sep]
D_raw_comb_common = D_raw_combined[:, ref_idx_comb]

print(f"Common references: {len(common_refs)}")

# Now try different distance normalizations

# Option A: Min-max normalization per embryo
def normalize_distances_minmax(D):
    """Normalize each embryo's distances to [0,1]"""
    D_norm = D.copy()
    for i in range(D.shape[0]):
        d_min = D[i, :].min()
        d_max = D[i, :].max()
        if d_max > d_min:
            D_norm[i, :] = (D[i, :] - d_min) / (d_max - d_min)
    return D_norm

# Option B: Z-score normalization per embryo
def normalize_distances_zscore(D):
    """Z-score normalize each embryo's distances"""
    D_norm = D.copy()
    for i in range(D.shape[0]):
        mean = D[i, :].mean()
        std = D[i, :].std()
        if std > 0:
            D_norm[i, :] = (D[i, :] - mean) / std
    return D_norm

print("\n--- Option A: Min-Max Normalize Distances ---")
D_sep_minmax = normalize_distances_minmax(D_raw_sep_common)
D_comb_minmax = normalize_distances_minmax(D_raw_comb_common)

corr_minmax = np.corrcoef(D_sep_minmax.flatten(), D_comb_minmax.flatten())[0, 1]
print(f"Correlation after min-max: {corr_minmax:.6f}")

# Check if nearest neighbors match
nn_sep_minmax = np.argmin(D_sep_minmax, axis=1)
nn_comb_minmax = np.argmin(D_comb_minmax, axis=1)
nn_agree_minmax = (nn_sep_minmax == nn_comb_minmax).sum() / len(nn_sep_minmax) * 100
print(f"NN agreement: {nn_agree_minmax:.1f}%")

print("\n--- Option B: Z-Score Normalize Distances ---")
D_sep_zscore = normalize_distances_zscore(D_raw_sep_common)
D_comb_zscore = normalize_distances_zscore(D_raw_comb_common)

corr_zscore = np.corrcoef(D_sep_zscore.flatten(), D_comb_zscore.flatten())[0, 1]
print(f"Correlation after z-score: {corr_zscore:.6f}")

nn_sep_zscore = np.argmin(D_sep_zscore, axis=1)
nn_comb_zscore = np.argmin(D_comb_zscore, axis=1)
nn_agree_zscore = (nn_sep_zscore == nn_comb_zscore).sum() / len(nn_sep_zscore) * 100
print(f"NN agreement: {nn_agree_zscore:.1f}%")

print("\n--- Compare to Raw (unnormalized) distances ---")
corr_raw = np.corrcoef(D_raw_sep_common.flatten(), D_raw_comb_common.flatten())[0, 1]
print(f"Correlation (raw, no norm): {corr_raw:.6f}")

nn_sep_raw = np.argmin(D_raw_sep_common, axis=1)
nn_comb_raw = np.argmin(D_raw_comb_common, axis=1)
nn_agree_raw = (nn_sep_raw == nn_comb_raw).sum() / len(nn_sep_raw) * 100
print(f"NN agreement (raw): {nn_agree_raw:.1f}%")

print("\n" + "-"*80)
if nn_agree_raw > 95:
    print("✓ TEST 2A: Raw distances already give consistent assignments!")
    print("  → No need for distance normalization")
    print("  → Just use normalize=False for features")
elif nn_agree_minmax > 95 or nn_agree_zscore > 95:
    print("✓ TEST 2B: Distance normalization improves consistency!")
    print(f"  Raw agreement: {nn_agree_raw:.1f}%")
    print(f"  MinMax agreement: {nn_agree_minmax:.1f}%")
    print(f"  ZScore agreement: {nn_agree_zscore:.1f}%")
    print("  → Could implement distance normalization option")
else:
    print("✗ TEST 2 FAILED: Distance normalization doesn't solve the problem")
    print(f"  Raw: {nn_agree_raw:.1f}%")
    print(f"  MinMax: {nn_agree_minmax:.1f}%")
    print(f"  ZScore: {nn_agree_zscore:.1f}%")

# ============================================================================
# Visualization
# ============================================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Test 1: Raw vs Normalized features
ax1 = axes[0, 0]
ax1.scatter(D_raw.flatten(), D_normalized.flatten(), alpha=0.1, s=1)
max_val = max(D_raw.max(), D_normalized.max())
ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
ax1.set_xlabel('Distance (Raw Features)', fontsize=11)
ax1.set_ylabel('Distance (Normalized Features)', fontsize=11)
ax1.set_title(f'Test 1: Raw vs Normalized Features\nr={correlation_features:.4f}',
              fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3)

# Test 1: Cluster agreement
ax2 = axes[0, 1]
ax2.text(0.5, 0.5, f'TEST 1 RESULTS\n\n'
         f'Cluster Agreement:\n{cluster_agreement_pct:.1f}%\n\n'
         f'NN Agreement:\n{nn_agreement_pct:.1f}%\n\n'
         f'Rank Correlation:\n{mean_rank_corr:.4f}',
         ha='center', va='center', fontsize=14, family='monospace',
         transform=ax2.transAxes,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax2.axis('off')

# Test 2: Raw distances (sep vs comb)
ax3 = axes[1, 0]
ax3.scatter(D_raw_sep_common.flatten(), D_raw_comb_common.flatten(), alpha=0.1, s=1, label='Raw')
ax3.scatter(D_sep_minmax.flatten(), D_comb_minmax.flatten(), alpha=0.1, s=1, label='MinMax')
max_val = max(D_raw_sep_common.max(), D_raw_comb_common.max())
ax3.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
ax3.set_xlabel('Distance (Separate Window)', fontsize=11)
ax3.set_ylabel('Distance (Combined Window)', fontsize=11)
ax3.set_title(f'Test 2: Distance Normalization\nRaw r={corr_raw:.4f}, MinMax r={corr_minmax:.4f}',
              fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# Test 2: Results summary
ax4 = axes[1, 1]
results_text = f'''TEST 2 RESULTS

NN Agreement:
  Raw:    {nn_agree_raw:.1f}%
  MinMax: {nn_agree_minmax:.1f}%
  ZScore: {nn_agree_zscore:.1f}%

Distance Correlation:
  Raw:    {corr_raw:.4f}
  MinMax: {corr_minmax:.4f}
  ZScore: {corr_zscore:.4f}
'''
ax4.text(0.5, 0.5, results_text,
         ha='center', va='center', fontsize=13, family='monospace',
         transform=ax4.transAxes,
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
ax4.axis('off')

plt.suptitle('Normalization Alternatives Testing', fontsize=14, fontweight='bold')
plt.tight_layout()

plot_path = FIGURES_DIR / 'normalization_alternatives.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: {plot_path}")
plt.close()

# Save results
results = pd.DataFrame([{
    'test': 'Raw_vs_Normalized_Features',
    'correlation': correlation_features,
    'cluster_agreement_pct': cluster_agreement_pct,
    'nn_agreement_pct': nn_agreement_pct,
    'mean_rank_correlation': mean_rank_corr,
}, {
    'test': 'Distance_Norm_Raw',
    'correlation': corr_raw,
    'nn_agreement_pct': nn_agree_raw,
}, {
    'test': 'Distance_Norm_MinMax',
    'correlation': corr_minmax,
    'nn_agreement_pct': nn_agree_minmax,
}, {
    'test': 'Distance_Norm_ZScore',
    'correlation': corr_zscore,
    'nn_agreement_pct': nn_agree_zscore,
}])

results_path = RESULTS_DIR / 'normalization_alternatives.csv'
results.to_csv(results_path, index=False)
print(f"✓ Saved: {results_path}")

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print("\nTest 1: Do we need normalization for single metric?")
if cluster_agreement_pct > 95:
    print(f"  ✓ NO - raw values work great ({cluster_agreement_pct:.1f}% agreement)")
    print(f"  RECOMMENDATION: Use normalize=False for single-metric analysis")
    print(f"  BENEFIT: Eliminates time window dependency issue!")
else:
    print(f"  ✗ YES - normalization affects results ({cluster_agreement_pct:.1f}% agreement)")

print("\nTest 2: Can we normalize distances instead?")
if nn_agree_raw > 95:
    print(f"  → Already consistent with raw distances ({nn_agree_raw:.1f}%)")
    print(f"  → No need for distance normalization")
elif max(nn_agree_minmax, nn_agree_zscore) > 95:
    best = 'MinMax' if nn_agree_minmax > nn_agree_zscore else 'ZScore'
    best_pct = max(nn_agree_minmax, nn_agree_zscore)
    print(f"  ✓ YES - {best} normalization helps ({best_pct:.1f}% vs {nn_agree_raw:.1f}%)")
    print(f"  → Could implement as option in projection functions")
else:
    print(f"  ✗ NO - distance normalization doesn't solve the problem")

print("\n" + "="*80)
print("✓ TUTORIAL 04e COMPLETE")
print("="*80)
