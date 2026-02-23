"""
Tutorial 04d: Direct Distance Comparison - Same Embryo Pairs

PROPER test: For the SAME embryo-reference pairs, do we get the same DTW distance
whether computed separately or combined?

Key Insight:
-----------
If NaN-aware DTW is working correctly, then:
  distance(embryo_A, ref_B) computed with experiment 1's time window
  SHOULD EQUAL
  distance(embryo_A, ref_B) computed with combined time window

Both should ignore NaNs and only use overlapping valid timepoints.

Test Strategy:
-------------
1. For experiment 20260122:
   - Get distances to ALL 550 reference embryos (separate method)
   - Get distances to SAME 550 reference embryos (combined method)
   - Correlate the distance vectors

2. For experiment 20260124:
   - Get distances to ALL 571 reference embryos (separate method)
   - Get distances to SAME 571 reference embryos (combined method)
   - Correlate the distance vectors

If correlation is ~1.0, NaN-aware DTW is working perfectly.
If correlation is lower, there's an issue with how NaNs are handled.
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
FIGURES_DIR = OUTPUT_DIR / "figures" / "04d"
RESULTS_DIR = OUTPUT_DIR / "results"

for d in [FIGURES_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Configuration
SOURCE_EXPERIMENTS = ['20260122', '20260124']
METRICS = ['baseline_deviation_normalized']
SAKOE_CHIBA_RADIUS = 20

print("="*80)
print("Tutorial 04d: Direct Distance Comparison Test")
print("="*80)
print("\nTesting: Do SAME embryo-reference pairs get SAME distances?")
print("\nIf NaN-aware DTW works correctly:")
print("  - Correlation should be ~1.0")
print("  - Max difference should be ~0")
print("  - This would prove NaN padding is working correctly")

# ============================================================================
# Load Data
# ============================================================================
print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

CEP290_REF_DIR = project_root / "results" / "mcolon" / "20251229_cep290_phenotype_extraction" / "final_data"

df_cep290_labels = pd.read_csv(CEP290_REF_DIR / "embryo_cluster_labels.csv", low_memory=False)
df_cep290_data = pd.read_csv(CEP290_REF_DIR / "embryo_data_with_labels.csv", low_memory=False)

labels_valid = df_cep290_labels.drop_duplicates(subset='embryo_id')
labels_valid = labels_valid[labels_valid['cluster_categories'].notna()].copy()

df_cep290_ref = df_cep290_data[df_cep290_data['embryo_id'].isin(labels_valid['embryo_id'])].copy()

print(f"Reference data: {df_cep290_ref['embryo_id'].nunique()} embryos")

meta_dir = project_root / 'morphseq_playground' / 'metadata' / 'build04_output'

source_dfs = {}
for exp_id in SOURCE_EXPERIMENTS:
    df_exp = pd.read_csv(meta_dir / f'qc_staged_{exp_id}.csv')
    df_exp = df_exp[df_exp['use_embryo_flag']].copy()
    source_dfs[exp_id] = df_exp
    print(f"  {exp_id}: {df_exp['embryo_id'].nunique()} embryos")

# ============================================================================
# Test Each Experiment
# ============================================================================

results_by_experiment = {}

for exp_id in SOURCE_EXPERIMENTS:
    print("\n" + "="*80)
    print(f"EXPERIMENT: {exp_id}")
    print("="*80)

    source_df = source_dfs[exp_id]

    # ========================================================================
    # Method 1: Separate (experiment-specific time window)
    # ========================================================================
    print("\nMethod 1: SEPARATE (experiment-specific time window)")

    source_min = source_df['predicted_stage_hpf'].min()
    source_max = source_df['predicted_stage_hpf'].max()
    ref_min = df_cep290_ref['predicted_stage_hpf'].min()
    ref_max = df_cep290_ref['predicted_stage_hpf'].max()

    window_start_sep = max(source_min, ref_min)
    window_end_sep = min(source_max, ref_max)

    print(f"  Time window: {window_start_sep:.1f} - {window_end_sep:.1f} hpf")

    source_filtered_sep = source_df[
        (source_df['predicted_stage_hpf'] >= window_start_sep) &
        (source_df['predicted_stage_hpf'] <= window_end_sep)
    ].copy()

    ref_filtered_sep = df_cep290_ref[
        (df_cep290_ref['predicted_stage_hpf'] >= window_start_sep) &
        (df_cep290_ref['predicted_stage_hpf'] <= window_end_sep)
    ].copy()

    X_ref_sep, ref_ids_sep, time_grid_sep = prepare_multivariate_array(
        ref_filtered_sep,
        metrics=METRICS,
        time_col='predicted_stage_hpf',
        embryo_id_col='embryo_id',
        normalize=True,
        verbose=False
    )

    X_source_sep, source_ids_sep, _ = prepare_multivariate_array(
        source_filtered_sep,
        metrics=METRICS,
        time_col='predicted_stage_hpf',
        embryo_id_col='embryo_id',
        time_grid=time_grid_sep,
        normalize=True,
        verbose=False
    )

    print(f"  Source: {X_source_sep.shape}")
    print(f"  Reference: {X_ref_sep.shape}")
    print(f"  Reference IDs: {len(ref_ids_sep)}")

    D_separate = compute_cross_dtw_distance_matrix(
        X_source_sep,
        X_ref_sep,
        sakoe_chiba_radius=SAKOE_CHIBA_RADIUS,
        n_jobs=-1,
        verbose=False
    )

    print(f"  Distance matrix: {D_separate.shape}")

    # ========================================================================
    # Method 2: Combined (full time window)
    # ========================================================================
    print("\nMethod 2: COMBINED (full time window)")

    # Use the FULL combined dataset's time range
    df_combined_full = pd.concat(source_dfs.values(), ignore_index=True)
    combined_min = df_combined_full['predicted_stage_hpf'].min()
    combined_max = df_combined_full['predicted_stage_hpf'].max()

    window_start_comb = max(combined_min, ref_min)
    window_end_comb = min(combined_max, ref_max)

    print(f"  Time window: {window_start_comb:.1f} - {window_end_comb:.1f} hpf")

    source_filtered_comb = source_df[
        (source_df['predicted_stage_hpf'] >= window_start_comb) &
        (source_df['predicted_stage_hpf'] <= window_end_comb)
    ].copy()

    ref_filtered_comb = df_cep290_ref[
        (df_cep290_ref['predicted_stage_hpf'] >= window_start_comb) &
        (df_cep290_ref['predicted_stage_hpf'] <= window_end_comb)
    ].copy()

    X_ref_comb, ref_ids_comb, time_grid_comb = prepare_multivariate_array(
        ref_filtered_comb,
        metrics=METRICS,
        time_col='predicted_stage_hpf',
        embryo_id_col='embryo_id',
        normalize=True,
        verbose=False
    )

    X_source_comb, source_ids_comb, _ = prepare_multivariate_array(
        source_filtered_comb,
        metrics=METRICS,
        time_col='predicted_stage_hpf',
        embryo_id_col='embryo_id',
        time_grid=time_grid_comb,
        normalize=True,
        verbose=False
    )

    print(f"  Source: {X_source_comb.shape}")
    print(f"  Reference: {X_ref_comb.shape}")
    print(f"  Reference IDs: {len(ref_ids_comb)}")

    D_combined = compute_cross_dtw_distance_matrix(
        X_source_comb,
        X_ref_comb,
        sakoe_chiba_radius=SAKOE_CHIBA_RADIUS,
        n_jobs=-1,
        verbose=False
    )

    print(f"  Distance matrix: {D_combined.shape}")

    # ========================================================================
    # Compare: Find common reference embryos
    # ========================================================================
    print("\n" + "-"*80)
    print("COMPARISON: Same embryo-reference pairs")
    print("-"*80)

    # Find reference embryos common to both
    common_ref_ids = sorted(set(ref_ids_sep) & set(ref_ids_comb))
    print(f"\nCommon reference embryos: {len(common_ref_ids)}")
    print(f"  Separate only: {len(set(ref_ids_sep) - set(ref_ids_comb))}")
    print(f"  Combined only: {len(set(ref_ids_comb) - set(ref_ids_sep))}")

    if len(common_ref_ids) == 0:
        print("\n⚠ ERROR: No common reference embryos!")
        continue

    # Get indices for common refs in both matrices
    ref_idx_sep = [ref_ids_sep.index(rid) for rid in common_ref_ids]
    ref_idx_comb = [ref_ids_comb.index(rid) for rid in common_ref_ids]

    # Extract submatrices for common refs
    D_sep_common = D_separate[:, ref_idx_sep]
    D_comb_common = D_combined[:, ref_idx_comb]

    print(f"\nSubmatrices for common refs:")
    print(f"  Separate: {D_sep_common.shape}")
    print(f"  Combined: {D_comb_common.shape}")

    # Element-wise comparison
    diff = D_comb_common - D_sep_common
    abs_diff = np.abs(diff)

    print(f"\nElement-wise comparison:")
    print(f"  Max absolute difference: {abs_diff.max():.6f}")
    print(f"  Mean absolute difference: {abs_diff.mean():.6f}")
    print(f"  Std of differences: {diff.std():.6f}")

    # Correlation
    D_sep_flat = D_sep_common.flatten()
    D_comb_flat = D_comb_common.flatten()
    correlation = np.corrcoef(D_sep_flat, D_comb_flat)[0, 1]

    print(f"\n  Pearson correlation: {correlation:.10f}")

    # Percentage identical (within tolerance)
    tolerance = 1e-6
    n_identical = np.sum(abs_diff < tolerance)
    n_total = abs_diff.size
    pct_identical = n_identical / n_total * 100

    print(f"  Within tolerance ({tolerance}): {n_identical}/{n_total} ({pct_identical:.2f}%)")

    # Store results
    results_by_experiment[exp_id] = {
        'correlation': correlation,
        'max_abs_diff': abs_diff.max(),
        'mean_abs_diff': abs_diff.mean(),
        'std_diff': diff.std(),
        'pct_identical': pct_identical,
        'n_common_refs': len(common_ref_ids),
        'D_sep_common': D_sep_common,
        'D_comb_common': D_comb_common,
        'diff': diff,
    }

    # Test result for this experiment
    print("\n" + "-"*80)
    if correlation > 0.9999 and abs_diff.max() < 0.01:
        print(f"✓ {exp_id}: PERFECT MATCH")
        print(f"  NaN-aware DTW produces identical distances for same pairs!")
    elif correlation > 0.99:
        print(f"⚠ {exp_id}: VERY CLOSE")
        print(f"  Correlation: {correlation:.6f}, Max diff: {abs_diff.max():.6f}")
        print(f"  Small numerical differences (likely precision/normalization)")
    else:
        print(f"✗ {exp_id}: SIGNIFICANT DIFFERENCES")
        print(f"  Correlation: {correlation:.6f}, Max diff: {abs_diff.max():.6f}")
        print(f"  This suggests NaN-aware DTW is NOT working consistently!")
    print("-"*80)

# ============================================================================
# Overall Summary
# ============================================================================
print("\n" + "="*80)
print("OVERALL SUMMARY")
print("="*80)

avg_correlation = np.mean([r['correlation'] for r in results_by_experiment.values()])
max_max_diff = np.max([r['max_abs_diff'] for r in results_by_experiment.values()])
avg_mean_diff = np.mean([r['mean_abs_diff'] for r in results_by_experiment.values()])

print(f"\nAcross all experiments:")
print(f"  Average correlation: {avg_correlation:.6f}")
print(f"  Max absolute difference: {max_max_diff:.6f}")
print(f"  Average mean difference: {avg_mean_diff:.6f}")

print(f"\n" + "="*80)
if avg_correlation > 0.9999 and max_max_diff < 0.01:
    print("✓✓✓ TEST PASSED: NaN-aware DTW is working PERFECTLY!")
    print("="*80)
    print("\nConclusion:")
    print("  Same embryo-reference pairs get identical distances")
    print("  regardless of time window used.")
    print("\n  This proves:")
    print("  - NaN padding is working correctly")
    print("  - NaN-aware DTW ignores NaNs properly")
    print("  - Distance calculations are consistent")
    print("\n  The 25% disagreement in 04 is NOT due to NaN handling.")
    print("  It must be due to different nearest neighbor selection")
    print("  when reference sets differ in size.")
elif avg_correlation > 0.99:
    print("⚠ TEST MARGINAL: Very high correlation but not perfect")
    print("="*80)
    print(f"\n  Correlation: {avg_correlation:.6f}")
    print(f"  Max diff: {max_max_diff:.6f}")
    print("\n  Small differences may be due to:")
    print("  - Floating point precision")
    print("  - Normalization differences")
    print("  - Numerical artifacts in DTW computation")
    print("\n  Overall, NaN-aware DTW appears to be working correctly.")
else:
    print("✗✗✗ TEST FAILED: Significant differences detected!")
    print("="*80)
    print(f"\n  Correlation: {avg_correlation:.6f}")
    print(f"  Max diff: {max_max_diff:.6f}")
    print("\n  This suggests NaN-aware DTW is NOT working correctly.")
    print("  Investigation needed:")
    print("  - Check NaN handling in _nan_aware_cost_matrix")
    print("  - Verify normalization is consistent")
    print("  - Examine time grid generation")
print("="*80)

# ============================================================================
# Visualization
# ============================================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, len(SOURCE_EXPERIMENTS), figsize=(14, 10))

for idx, exp_id in enumerate(SOURCE_EXPERIMENTS):
    res = results_by_experiment[exp_id]

    # Scatter plot
    ax1 = axes[0, idx]
    ax1.scatter(res['D_sep_common'].flatten(), res['D_comb_common'].flatten(),
               alpha=0.1, s=1)
    max_val = max(res['D_sep_common'].max(), res['D_comb_common'].max())
    ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
    ax1.set_xlabel('Distance (Separate)', fontsize=10)
    ax1.set_ylabel('Distance (Combined)', fontsize=10)
    ax1.set_title(f'{exp_id}\nr={res["correlation"]:.6f}', fontsize=11, fontweight='bold')
    ax1.grid(alpha=0.3)

    # Difference histogram
    ax2 = axes[1, idx]
    ax2.hist(res['diff'].flatten(), bins=100, edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Difference (Combined - Separate)', fontsize=10)
    ax2.set_ylabel('Count', fontsize=10)
    ax2.set_title(f'Max diff: {res["max_abs_diff"]:.4f}', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)

plt.suptitle('Direct Distance Comparison: Same Embryo-Reference Pairs',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()

plot_path = FIGURES_DIR / 'direct_distance_comparison.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: {plot_path}")
plt.close()

# Save results
results_df = pd.DataFrame([
    {
        'experiment': exp_id,
        'correlation': res['correlation'],
        'max_abs_diff': res['max_abs_diff'],
        'mean_abs_diff': res['mean_abs_diff'],
        'std_diff': res['std_diff'],
        'pct_identical': res['pct_identical'],
        'n_common_refs': res['n_common_refs'],
    }
    for exp_id, res in results_by_experiment.items()
])

results_path = RESULTS_DIR / 'direct_distance_comparison.csv'
results_df.to_csv(results_path, index=False)
print(f"✓ Saved: {results_path}")

print("\n" + "="*80)
print("✓ TUTORIAL 04d COMPLETE")
print("="*80)
