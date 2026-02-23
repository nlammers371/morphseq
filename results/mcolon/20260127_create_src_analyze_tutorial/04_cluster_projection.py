"""
Tutorial 04: Cluster Projection - Testing Batch Effects in CEP290 Experiments

Demonstrates cluster projection methodology for comparing phenotypic trajectories
across experiments with different temporal coverage.

Problem:
--------
With 27-47 hpf overlap window (Tutorial 03), we cannot distinguish:
- Wild-type (ab) from "Low_to_High" cep290_crispant trajectories
- "Not_Penetrant" from "Low_to_High" cep290_crispant (error bars overlap)

Key Insight:
-----------
Experiment 20260124 extends to 80 hpf and shows almost all crispants become
penetrant by 80 hpf. This means we can test batch effects by projecting each
experiment's clusters onto a well-defined reference.

Approach:
---------
Project new CEP290 experiments (20260122, 20260124) onto pre-existing CEP290
mutant cluster definitions from 7 older experiments (2025 data) to test whether
temporal coverage differences cause batch effects in penetrance trajectories.

Pipeline Steps:
--------------
1. Load CEP290 reference clusters (well-defined from 7 experiments)
2. Load source experiments separately (20260122, 20260124)
3. Compute cross-DTW distances (source → reference)
4. Project onto reference clusters (nearest neighbor method)
5. Compare cluster proportions between experiments
6. Visualize trajectories and test for batch effects

Key API Usage:
-------------
- compute_cross_dtw_distance_matrix() - Cross-dataset DTW
- assign_clusters_nearest_neighbor() - NN cluster assignment
- compare_cluster_frequencies() - Statistical comparison
- plot_feature_over_time() - Faceted trajectory visualization
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "src"))

# Setup output directory
OUTPUT_DIR = Path(__file__).parent / "output"
FIGURES_DIR = OUTPUT_DIR / "figures" / "04"
RESULTS_DIR = OUTPUT_DIR / "results"
PROJECTION_DIR = FIGURES_DIR / "projection_results"

for d in [FIGURES_DIR, RESULTS_DIR, PROJECTION_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Import projection utilities from main codebase
from analyze.trajectory_analysis import (
    run_bootstrap_projection_with_plots,
)

from analyze.viz.plotting import plot_feature_over_time, plot_proportions

# Keep compare_cluster_frequencies from local utils
from projection_utils import compare_cluster_frequencies

# ============================================================================
# Configuration
# ============================================================================

# Reference data paths (pre-computed CEP290 clusters from 7 experiments)
CEP290_REF_DIR = project_root / "results" / "mcolon" / "20251229_cep290_phenotype_extraction" / "final_data"

# Source experiments to project
SOURCE_EXPERIMENTS = ['20260122', '20260124']

# Analysis parameters
METRICS = ['baseline_deviation_normalized']  # Curvature only
SAKOE_CHIBA_RADIUS = 20  # DTW warping constraint

print("="*80)
print("Tutorial 04: CEP290 Cluster Projection - Batch Effect Analysis")
print("="*80)
print(f"\nConfiguration:")
print(f"  Reference: CEP290 clusters from 7 experiments (2025)")
print(f"  Source: {SOURCE_EXPERIMENTS}")
print(f"  Metrics: {METRICS}")
print(f"  Time window: AUTO-DETECTED (intersection of source and reference)")
print(f"  Sakoe-Chiba radius: {SAKOE_CHIBA_RADIUS}")

# ============================================================================
# Step 1: Load CEP290 Reference Clusters (Well-Defined Phenotypes)
# ============================================================================
print("\n" + "="*80)
print("STEP 1: LOAD CEP290 REFERENCE CLUSTERS")
print("="*80)

print("\nLoading CEP290 reference data...")
df_cep290_labels = pd.read_csv(CEP290_REF_DIR / "embryo_cluster_labels.csv", low_memory=False)
df_cep290_data = pd.read_csv(CEP290_REF_DIR / "embryo_data_with_labels.csv", low_memory=False)

print(f"  Labels: {len(df_cep290_labels)} rows")
print(f"  Data: {len(df_cep290_data)} rows, {df_cep290_data['embryo_id'].nunique()} embryos")

# Filter to embryos with valid cluster assignments
labels_valid = df_cep290_labels.drop_duplicates(subset='embryo_id')
labels_valid = labels_valid[labels_valid['cluster_categories'].notna()].copy()
print(f"  Valid cluster assignments: {len(labels_valid)} embryos")

print(f"\nReference cluster distribution:")
for cat, count in labels_valid['cluster_categories'].value_counts().items():
    pct = count / len(labels_valid) * 100
    print(f"  {cat}: {count} ({pct:.1f}%)")

# Filter data to only include embryos with valid clusters
df_cep290_ref = df_cep290_data[df_cep290_data['embryo_id'].isin(labels_valid['embryo_id'])].copy()
print(f"\nFiltered reference data: {df_cep290_ref['embryo_id'].nunique()} embryos")

# ============================================================================
# Step 2: Load Source Experiments (20260122, 20260124)
# ============================================================================
print("\n" + "="*80)
print("STEP 2: LOAD SOURCE EXPERIMENTS")
print("="*80)

meta_dir = project_root / 'morphseq_playground' / 'metadata' / 'build04_output'

source_dfs = {}
for exp_id in SOURCE_EXPERIMENTS:
    print(f"\nLoading experiment {exp_id}...")
    df_exp = pd.read_csv(meta_dir / f'qc_staged_{exp_id}.csv')

    # Filter to valid embryos
    df_exp = df_exp[df_exp['use_embryo_flag']].copy()
    df_exp['experiment_id'] = exp_id

    print(f"  Total: {len(df_exp)} rows, {df_exp['embryo_id'].nunique()} embryos")
    print(f"  Time range: {df_exp['predicted_stage_hpf'].min():.1f} - {df_exp['predicted_stage_hpf'].max():.1f} hpf")
    print(f"  Genotypes: {sorted(df_exp['genotype'].unique())}")

    source_dfs[exp_id] = df_exp

# ============================================================================
# Step 3-5: Bootstrapped Projection onto Reference (Using New API)
# ============================================================================
print("\n" + "="*80)
print("STEP 3-5: BOOTSTRAP PROJECT EXPERIMENTS ONTO REFERENCE")
print("="*80)
print("\nUsing automatic time window detection (intersection method)")
print("This ensures no extrapolation bias in DTW distance computation\n")

projections = {}
projection_results = {}

for exp_id in SOURCE_EXPERIMENTS:
    print("\n" + "-"*80)
    print(f"EXPERIMENT: {exp_id}")
    print("-"*80)

    results = run_bootstrap_projection_with_plots(
        source_df=source_dfs[exp_id],
        reference_df=df_cep290_ref,
        labels_df=labels_valid,
        output_dir=PROJECTION_DIR / exp_id,
        run_name=f"{exp_id}_projection",
        id_col='embryo_id',
        time_col='predicted_stage_hpf',
        cluster_col='cluster_categories',
        category_col=None,
        metrics=METRICS,
        sakoe_chiba_radius=SAKOE_CHIBA_RADIUS,
        n_bootstrap=100,
        frac=0.8,
        bootstrap_on="reference",
        method='nearest_neighbor',
        classification='2d',
        normalize=True,
        verbose=True,
        save_outputs=True
    )

    df_proj = results['assignments_df']

    # Add experiment ID and genotype
    df_proj['experiment_id'] = exp_id
    exp_df = source_dfs[exp_id]
    genotype_map = dict(zip(exp_df['embryo_id'], exp_df['genotype']))
    df_proj['genotype'] = df_proj['embryo_id'].map(genotype_map)

    projections[exp_id] = df_proj
    projection_results[exp_id] = results

    # Save projection results
    output_path = PROJECTION_DIR / f"{exp_id}_projection_bootstrap.csv"
    df_proj.to_csv(output_path, index=False)
    print(f"\n✓ Saved: {output_path}")

# ============================================================================
# BENCHMARK: Compare Separate vs Combined Projection
# ============================================================================
print("\n" + "="*80)
print("BENCHMARK: SEPARATE vs COMBINED PROJECTION")
print("="*80)

# Approach A: Separate (already done above)
print("\nApproach A: Project experiments SEPARATELY")
print("  - 20260122: Individual projection")
print("  - 20260124: Individual projection")

# Combine for summary stats
df_all_proj_separate = pd.concat(projections.values(), ignore_index=True)
print(f"  - Total: {len(df_all_proj_separate)} embryos")

# Approach B: Combined - merge both experiments first, then project
print("\nApproach B: Project experiments COMBINED")

# Combine source experiments into one DataFrame
df_combined = pd.concat(source_dfs.values(), ignore_index=True)
print(f"  - Combined: {df_combined['embryo_id'].nunique()} embryos from {len(SOURCE_EXPERIMENTS)} experiments")
print(f"  - Time range: {df_combined['predicted_stage_hpf'].min():.1f} - {df_combined['predicted_stage_hpf'].max():.1f} hpf")

# Project combined dataset
combined_results = run_bootstrap_projection_with_plots(
    source_df=df_combined,
    reference_df=df_cep290_ref,
    labels_df=labels_valid,
    output_dir=PROJECTION_DIR / "combined",
    run_name="combined_projection",
    id_col='embryo_id',
    time_col='predicted_stage_hpf',
    cluster_col='cluster_categories',
    category_col=None,
    metrics=METRICS,
    sakoe_chiba_radius=SAKOE_CHIBA_RADIUS,
    n_bootstrap=100,
    frac=0.8,
    bootstrap_on="reference",
    method='nearest_neighbor',
    classification='2d',
    normalize=True,
    verbose=True,
    save_outputs=True
)

df_proj_combined = combined_results['assignments_df']

# Add experiment ID and genotype
genotype_map_combined = dict(zip(df_combined['embryo_id'], df_combined['genotype']))
experiment_map_combined = dict(zip(df_combined['embryo_id'], df_combined['experiment_id']))
df_proj_combined['genotype'] = df_proj_combined['embryo_id'].map(genotype_map_combined)
df_proj_combined['experiment_id'] = df_proj_combined['embryo_id'].map(experiment_map_combined)

# Save combined projection
output_path_combined = PROJECTION_DIR / "combined_projection_bootstrap.csv"
df_proj_combined.to_csv(output_path_combined, index=False)
print(f"\n✓ Saved: {output_path_combined}")

# ============================================================================
# BENCHMARK COMPARISON: Analyze Differences
# ============================================================================
print("\n" + "="*80)
print("BENCHMARK RESULTS: SEPARATE vs COMBINED")
print("="*80)

# Merge separate and combined results for comparison
df_separate = df_all_proj_separate[['embryo_id', 'cluster', 'cluster_label', 'nearest_distance', 'experiment_id']].copy()
df_separate = df_separate.rename(columns={
    'cluster': 'cluster_separate',
    'cluster_label': 'category_separate',
    'nearest_distance': 'distance_separate'
})

df_compare = df_proj_combined[['embryo_id', 'cluster', 'cluster_label', 'nearest_distance']].merge(
    df_separate,
    on='embryo_id',
    how='inner'
)
df_compare = df_compare.rename(columns={
    'cluster': 'cluster_combined',
    'cluster_label': 'category_combined',
    'nearest_distance': 'distance_combined'
})

# Analysis 1: Cluster assignment agreement
agreement = (df_compare['cluster_combined'] == df_compare['cluster_separate']).sum()
total = len(df_compare)
agreement_pct = agreement / total * 100

print(f"\nCluster Assignment Agreement:")
print(f"  Matching: {agreement}/{total} ({agreement_pct:.1f}%)")
print(f"  Different: {total - agreement}/{total} ({100 - agreement_pct:.1f}%)")

# Analysis 2: Category assignment agreement
category_agreement = (df_compare['category_combined'] == df_compare['category_separate']).sum()
category_agreement_pct = category_agreement / total * 100

print(f"\nCategory Assignment Agreement:")
print(f"  Matching: {category_agreement}/{total} ({category_agreement_pct:.1f}%)")
print(f"  Different: {total - category_agreement}/{total} ({100 - category_agreement_pct:.1f}%)")

# Analysis 3: Distance comparison
print(f"\nDTW Distance Comparison:")
print(f"  Separate - Mean: {df_compare['distance_separate'].mean():.3f}, Median: {df_compare['distance_separate'].median():.3f}")
print(f"  Combined - Mean: {df_compare['distance_combined'].mean():.3f}, Median: {df_compare['distance_combined'].median():.3f}")

distance_diff = (df_compare['distance_combined'] - df_compare['distance_separate']).abs()
print(f"  Mean absolute difference: {distance_diff.mean():.3f}")
print(f"  Max absolute difference: {distance_diff.max():.3f}")

# Analysis 4: Breakdown by experiment
print(f"\n--- Agreement by Experiment ---")
for exp_id in SOURCE_EXPERIMENTS:
    df_exp = df_compare[df_compare['experiment_id'] == exp_id]
    exp_agreement = (df_exp['category_combined'] == df_exp['category_separate']).sum()
    exp_total = len(df_exp)
    exp_pct = exp_agreement / exp_total * 100 if exp_total > 0 else 0
    print(f"  {exp_id}: {exp_agreement}/{exp_total} ({exp_pct:.1f}%)")

# Analysis 5: Show disagreements
disagreements = df_compare[df_compare['category_combined'] != df_compare['category_separate']]
if len(disagreements) > 0:
    print(f"\n--- Disagreements ({len(disagreements)} embryos) ---")
    print(f"  Separate → Combined transitions:")
    transitions = disagreements.groupby(['category_separate', 'category_combined']).size().reset_index(name='count')
    for _, row in transitions.iterrows():
        print(f"    {row['category_separate']} → {row['category_combined']}: {row['count']}")
else:
    print(f"\n✓ Perfect agreement! All embryos assigned to same category.")

# Save comparison results
df_compare.to_csv(RESULTS_DIR / "benchmark_comparison.csv", index=False)
print(f"\n✓ Saved: {RESULTS_DIR / 'benchmark_comparison.csv'}")

# Visualization: Comparison plot
print(f"\n--- Generating comparison visualization ---")

fig_comp, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Distance comparison scatter
ax1 = axes[0]
ax1.scatter(df_compare['distance_separate'], df_compare['distance_combined'],
            alpha=0.5, s=30, c=df_compare['experiment_id'].map({'20260122': 'blue', '20260124': 'orange'}))
max_dist = max(df_compare['distance_separate'].max(), df_compare['distance_combined'].max())
ax1.plot([0, max_dist], [0, max_dist], 'k--', alpha=0.5, linewidth=2, label='y=x')
ax1.set_xlabel('Distance (Separate)', fontsize=11)
ax1.set_ylabel('Distance (Combined)', fontsize=11)
ax1.set_title('DTW Distance: Separate vs Combined', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: Agreement matrix
ax2 = axes[1]
if len(disagreements) > 0:
    from collections import Counter
    transition_counts = Counter(zip(df_compare['category_separate'], df_compare['category_combined']))
    categories = sorted(set(df_compare['category_separate']) | set(df_compare['category_combined']))

    matrix = np.zeros((len(categories), len(categories)))
    for i, cat_sep in enumerate(categories):
        for j, cat_comb in enumerate(categories):
            matrix[i, j] = transition_counts.get((cat_sep, cat_comb), 0)

    im = ax2.imshow(matrix, cmap='Blues', aspect='auto')
    ax2.set_xticks(range(len(categories)))
    ax2.set_yticks(range(len(categories)))
    ax2.set_xticklabels(categories, rotation=45, ha='right')
    ax2.set_yticklabels(categories)
    ax2.set_xlabel('Combined Assignment', fontsize=11)
    ax2.set_ylabel('Separate Assignment', fontsize=11)
    ax2.set_title('Assignment Agreement Matrix', fontsize=12, fontweight='bold')

    # Add counts to cells
    for i in range(len(categories)):
        for j in range(len(categories)):
            text = ax2.text(j, i, int(matrix[i, j]),
                          ha="center", va="center", color="black" if matrix[i, j] < matrix.max()/2 else "white")

    plt.colorbar(im, ax=ax2, label='Count')
else:
    ax2.text(0.5, 0.5, '100% Agreement\n(No disagreements)',
            ha='center', va='center', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

plt.tight_layout()
comp_plot_path = FIGURES_DIR / 'benchmark_comparison.png'
plt.savefig(comp_plot_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {comp_plot_path}")
plt.close(fig_comp)

print("\n" + "="*80)
print("BENCHMARK COMPLETE")
print("="*80)
print(f"\nExpected outcome with NaN-aware DTW:")
print(f"  - High agreement (>95%) indicates NaN-aware DTW is working correctly")
print(f"  - Low agreement would suggest batch effects from temporal coverage differences")
print(f"  - Distance differences should be minimal (same embryos, same reference)")

# Combine all projections
df_all_proj = pd.concat(projections.values(), ignore_index=True)
print("\n" + "="*80)
print(f"✓ PROJECTION COMPLETE")
print("="*80)
print(f"Combined projections: {len(df_all_proj)} embryos across {len(SOURCE_EXPERIMENTS)} experiments")

# ============================================================================
# Step 6: Distance Diagnostics - Cluster Fit Checks
# ============================================================================
print("\n" + "="*80)
print("STEP 6: DISTANCE DIAGNOSTICS")
print("="*80)

def _label_key(label: str) -> str:
    label_str = str(label).strip().lower()
    cleaned = []
    prev_underscore = False
    for ch in label_str:
        if ch.isalnum():
            cleaned.append(ch)
            prev_underscore = False
        else:
            if not prev_underscore:
                cleaned.append('_')
                prev_underscore = True
    key = ''.join(cleaned).strip('_')
    return key or "label"

for exp_id in SOURCE_EXPERIMENTS:
    print(f"\n--- {exp_id} ---")
    df_exp = projections[exp_id].copy()

    # Check if distance columns exist
    label_set = sorted(df_exp['cluster_label'].dropna().unique())
    label_keys = {label: _label_key(label) for label in label_set}
    dist_cols = [f"dist_to_{label_keys[label]}" for label in label_set if f"dist_to_{label_keys[label]}" in df_exp.columns]

    if len(dist_cols) == 0:
        print("  No per-cluster distance columns found. Skipping diagnostics.")
        continue

    # Distance margin summary
    print(f"  Distance margin stats (second-best - best):")
    print(f"    Mean: {df_exp['distance_margin'].mean():.3f}")
    print(f"    Median: {df_exp['distance_margin'].median():.3f}")
    print(f"    Min: {df_exp['distance_margin'].min():.3f}")

    # Intermediate vs Low_to_High check if those labels exist
    if 'Intermediate' in label_keys and 'Low_to_High' in label_keys:
        d_inter = f"dist_to_{label_keys['Intermediate']}"
        d_l2h = f"dist_to_{label_keys['Low_to_High']}"
        if d_inter in df_exp.columns and d_l2h in df_exp.columns:
            df_inter = df_exp[df_exp['cluster_label'] == 'Intermediate'].copy()
            if len(df_inter) > 0:
                df_inter['margin_l2h_minus_inter'] = df_inter[d_l2h] - df_inter[d_inter]
                print(f"  Intermediate assignments: {len(df_inter)}")
                print(f"    Mean (Low_to_High - Intermediate) distance margin: {df_inter['margin_l2h_minus_inter'].mean():.3f}")
                print(f"    Median margin: {df_inter['margin_l2h_minus_inter'].median():.3f}")
                # Save table for review
                diag_path = RESULTS_DIR / f"{exp_id}_intermediate_margin_check.csv"
                df_inter.sort_values('margin_l2h_minus_inter').to_csv(diag_path, index=False)
                print(f"    Saved: {diag_path}")
            else:
                print("  No Intermediate assignments found.")
        else:
            print("  Missing distance columns for Intermediate/Low_to_High comparison.")

    # Save overall distance diagnostic table
    diag_all_path = RESULTS_DIR / f"{exp_id}_distance_diagnostics.csv"
    df_exp.to_csv(diag_all_path, index=False)
    print(f"  Saved: {diag_all_path}")

# ============================================================================
# Step 7: Proportion Analysis - Compare Experiments
# ============================================================================
print("\n" + "="*80)
print("STEP 7: PROPORTION ANALYSIS")
print("="*80)

# Compare cluster frequencies between experiments
print("\n--- Comparing 20260122 vs 20260124 ---")

freq_comp, stats = compare_cluster_frequencies(
    projections['20260122'],
    projections['20260124'],
    category_col='cluster_label',
    ref_label='20260122',
    projected_label='20260124'
)

print("\nCluster frequency comparison (%):")
print(freq_comp.to_string())

print("\n" + "="*60)
print("STATISTICAL TESTS")
print("="*60)
print(f"Chi-square test:")
print(f"  χ² = {stats['chi2_statistic']:.3f}")
print(f"  p-value = {stats['chi2_pvalue']:.4f}")
print(f"  df = {stats['chi2_dof']}")

# Save frequency comparison
freq_comp.to_csv(RESULTS_DIR / "cluster_frequency_comparison.csv")
print(f"\nSaved: {RESULTS_DIR / 'cluster_frequency_comparison.csv'}")

# Analyze by genotype
print("\n" + "="*80)
print("GENOTYPE-STRATIFIED ANALYSIS")
print("="*80)

for genotype in sorted(df_all_proj['genotype'].unique()):
    print(f"\n--- {genotype} ---")

    df_geno = df_all_proj[df_all_proj['genotype'] == genotype]

    for exp_id in SOURCE_EXPERIMENTS:
        df_exp_geno = df_geno[df_geno['experiment_id'] == exp_id]
        print(f"\n{exp_id} ({len(df_exp_geno)} embryos):")

        if len(df_exp_geno) > 0:
            for cat, count in df_exp_geno['cluster_label'].value_counts().items():
                pct = count / len(df_exp_geno) * 100
                print(f"  {cat}: {count} ({pct:.1f}%)")

# ============================================================================
# Step 7: Visualize Trajectories - Faceted by Cluster Category and Experiment
# ============================================================================
print("\n" + "="*80)
print("STEP 8: TRAJECTORY VISUALIZATION")
print("="*80)

# Merge projection results back with trajectory data
df_viz_list = []

for exp_id in SOURCE_EXPERIMENTS:
    # Get trajectory data
    df_traj = source_dfs[exp_id]

    # Merge with projection assignments (no time filtering - show full trajectories)
    proj = projections[exp_id][['embryo_id', 'cluster_label', 'cluster']].copy()
    df_merged = df_traj.merge(proj, on='embryo_id', how='inner')

    df_viz_list.append(df_merged)

df_viz = pd.concat(df_viz_list, ignore_index=True)
print(f"\nVisualization data: {len(df_viz)} rows, {df_viz['embryo_id'].nunique()} embryos")

# Plot 1: Trajectories faceted by cluster_label (rows) and experiment_id (cols)
print("\nGenerating trajectory plot (cluster × experiment)...")

fig1 = plot_feature_over_time(
    df_viz,
    features='baseline_deviation_normalized',
    time_col='predicted_stage_hpf',
    id_col='embryo_id',
    color_by='genotype',
    facet_row='cluster_label',
    facet_col='experiment_id',
    title='Projected Cluster Trajectories: Experiment Comparison',
    backend='matplotlib',
    bin_width=2.0,
)

plot1_path = FIGURES_DIR / 'cluster_projection_trajectories.png'
plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {plot1_path}")
plt.close(fig1)

# ============================================================================
# Step 8: Proportion Plots - Batch Effect Visualization
# ============================================================================
print("\n" + "="*80)
print("STEP 9: PROPORTION PLOTS")
print("="*80)

# Use the new plot_proportions function for faceted visualization
print("\nGenerating proportion plot (cluster_label by experiment and genotype)...")

# Get unique embryo assignments (deduplicate by embryo_id)
df_embryo_proj = df_all_proj.drop_duplicates(subset='embryo_id')

fig2 = plot_proportions(
    df_embryo_proj,
    color_by_grouping='cluster_label',
    row_by='genotype',
    col_by='experiment_id',
    count_by='embryo_id',
    normalize=True,
    bar_mode='grouped',
    title='Cluster Distribution by Experiment and Genotype',
    show_counts=True,
)

plot2_path = FIGURES_DIR / 'proportion_by_experiment.png'
plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {plot2_path}")
plt.close(fig2)

# ============================================================================
# Step 9: Batch Effect Analysis - Distance Distribution Check
# ============================================================================
print("\n" + "="*80)
print("STEP 10: BATCH EFFECT ANALYSIS")
print("="*80)

print("\nDistance distribution analysis:")

fig3, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, exp_id in enumerate(SOURCE_EXPERIMENTS):
    ax = axes[idx]

    # Plot distance distribution
    distances = df_all_proj[df_all_proj['experiment_id'] == exp_id]['nearest_distance']

    ax.hist(distances, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(distances.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {distances.median():.2f}')
    ax.set_xlabel('DTW Distance to Nearest Reference', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'{exp_id}', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    print(f"\n{exp_id} distance statistics:")
    print(f"  Min: {distances.min():.2f}")
    print(f"  Max: {distances.max():.2f}")
    print(f"  Mean: {distances.mean():.2f}")
    print(f"  Median: {distances.median():.2f}")
    print(f"  Std: {distances.std():.2f}")

plt.suptitle('DTW Distance Distribution: Source → Reference', fontsize=14, fontweight='bold')
plt.tight_layout()

plot3_path = FIGURES_DIR / 'batch_effect_analysis.png'
plt.savefig(plot3_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: {plot3_path}")
plt.close(fig3)

# ============================================================================
# SECTION 10: Bootstrap Uncertainty Quantification
# ============================================================================

print("\n" + "="*80)
print("SECTION 11: Bootstrap Uncertainty Quantification")
print("="*80)

# Bootstrap parameters (already used in the wrapper)
print(f"\nBootstrapping cluster projections...")
print(f"  Iterations: 100")
print(f"  Subsample fraction: 0.8")

# Bootstrap each experiment separately
bootstrap_posterior_results = {}

for exp_id in SOURCE_EXPERIMENTS:
    print(f"\n{'='*60}")
    print(f"Bootstrapping experiment: {exp_id}")
    print('='*60)

    # Pull posteriors from wrapper outputs
    posteriors = projection_results[exp_id]['posteriors']

    # Convert to DataFrame
    uncertainty_df = pd.DataFrame({
        'embryo_id': posteriors['embryo_ids'],
        'modal_cluster': posteriors['modal_cluster'],
        'max_p': posteriors['max_p'],
        'entropy': posteriors['entropy'],
        'log_odds_gap': posteriors['log_odds_gap'],
        'second_best_cluster': posteriors['second_best_cluster'],
        'sample_counts': posteriors['sample_counts']
    })

    # Add posterior columns using label names when available
    label_order = projection_results[exp_id].get('cluster_label_order')
    label_map = projection_results[exp_id].get('cluster_label_map', {})
    if label_order:
        for idx, label in enumerate(label_order):
            uncertainty_df[f'p_{label}'] = posteriors['p_matrix'][:, idx]
        # Add modal label for readability
        uncertainty_df['modal_label'] = [
            label_map.get(int(c), c) for c in posteriors['modal_cluster']
        ]
    else:
        for cluster_id in range(posteriors['n_clusters']):
            uncertainty_df[f'p_cluster_{cluster_id}'] = posteriors['p_matrix'][:, cluster_id]

    # Merge with projection assignments for context
    uncertainty_df = uncertainty_df.merge(
        projections[exp_id][['embryo_id', 'cluster', 'cluster_label', 'nearest_distance', 'membership']],
        on='embryo_id',
        how='left',
    )

    bootstrap_posterior_results[exp_id] = uncertainty_df

    # Print summary statistics
    print(f"\nPosterior Summary for {exp_id}:")
    print(f"  Mean max_p (confidence): {uncertainty_df['max_p'].mean():.3f}")
    print(f"  Median max_p: {uncertainty_df['max_p'].median():.3f}")
    print(f"  % embryos with max_p > 0.90: {(uncertainty_df['max_p'] > 0.90).mean() * 100:.1f}%")
    print(f"  % embryos with max_p > 0.75: {(uncertainty_df['max_p'] > 0.75).mean() * 100:.1f}%")
    print(f"  % embryos with max_p < 0.50: {(uncertainty_df['max_p'] < 0.50).mean() * 100:.1f}%")
    print(f"  Mean entropy: {uncertainty_df['entropy'].mean():.3f}")
    print(f"  Mean log-odds gap: {uncertainty_df['log_odds_gap'].mean():.3f}")

    # Identify high vs low confidence embryos
    high_conf = uncertainty_df[uncertainty_df['max_p'] > 0.90]
    low_conf = uncertainty_df[uncertainty_df['max_p'] < 0.50]

    print(f"\nHigh confidence embryos (max_p > 90%): {len(high_conf)}")
    if len(high_conf) > 0:
        print(f"  Clusters: {high_conf['modal_cluster'].value_counts().to_dict()}")

    print(f"\nLow confidence embryos (max_p < 50%): {len(low_conf)}")
    if len(low_conf) > 0:
        print(f"  Clusters: {low_conf['modal_cluster'].value_counts().to_dict()}")
        print(f"  Sample embryos: {low_conf['embryo_id'].head(5).tolist()}")

    # Agreement with original projection
    modal_matches = (uncertainty_df['modal_cluster'] == uncertainty_df['cluster']).sum()
    print(f"\nAgreement with Original Projection:")
    print(f"  Modal cluster matches: {modal_matches}/{len(uncertainty_df)} ({100*modal_matches/len(uncertainty_df):.1f}%)")

    # Save results
    output_path = RESULTS_DIR / f'{exp_id}_bootstrap_posteriors.csv'
    uncertainty_df.to_csv(output_path, index=False)
    print(f"\nPosterior results saved to: {output_path}")

# Combine experiments
combined_posteriors = pd.concat([
    df.assign(experiment_id=exp_id)
    for exp_id, df in bootstrap_posterior_results.items()
], ignore_index=True)

combined_output_path = RESULTS_DIR / 'combined_bootstrap_posteriors.csv'
combined_posteriors.to_csv(combined_output_path, index=False)
print(f"\nCombined posterior results saved to: {combined_output_path}")

# ============================================================================
# SECTION 11: Bootstrap Posterior Visualizations
# ============================================================================

print("\n" + "="*80)
print("SECTION 11: Bootstrap Posterior Visualizations")
print("="*80)

# Visualization 1: max_p Distribution Histogram
# ----------------------------------------------
print("\nGenerating max_p distribution histogram...")

fig_boot1, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, exp_id in enumerate(SOURCE_EXPERIMENTS):
    ax = axes[idx]
    df_exp = bootstrap_posterior_results[exp_id]

    ax.hist(df_exp['max_p'], bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(0.90, color='green', linestyle='--', label='High confidence (>90%)')
    ax.axvline(0.75, color='orange', linestyle='--', label='Moderate (>75%)')
    ax.axvline(0.50, color='red', linestyle='--', label='Low confidence (<50%)')

    ax.set_xlabel('Max Posterior Probability (max_p)', fontsize=12)
    ax.set_ylabel('Number of Embryos', fontsize=12)
    ax.set_title(f'Posterior Confidence Distribution\n{exp_id}', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
boot_plot1_path = FIGURES_DIR / 'bootstrap_maxp_distribution.png'
plt.savefig(boot_plot1_path, dpi=300)
print(f"✓ Saved: {boot_plot1_path}")
plt.close(fig_boot1)

# Visualization 2: max_p vs Nearest Distance Scatter
# ---------------------------------------------------
print("\nGenerating max_p vs distance scatter plot...")

fig_boot2, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, exp_id in enumerate(SOURCE_EXPERIMENTS):
    ax = axes[idx]
    df_exp = bootstrap_posterior_results[exp_id]

    scatter = ax.scatter(
        df_exp['nearest_distance'],
        df_exp['max_p'],
        c=df_exp['entropy'],
        s=30,
        alpha=0.6,
        cmap='viridis'
    )

    ax.set_xlabel('Nearest Reference Distance (DTW)', fontsize=12)
    ax.set_ylabel('Max Posterior Probability', fontsize=12)
    ax.set_title(f'Confidence vs Distance\n{exp_id}', fontsize=14)
    ax.grid(alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Entropy', fontsize=10)

plt.tight_layout()
boot_plot2_path = FIGURES_DIR / 'bootstrap_maxp_vs_distance.png'
plt.savefig(boot_plot2_path, dpi=300)
print(f"✓ Saved: {boot_plot2_path}")
plt.close(fig_boot2)

# Visualization 3: max_p by Cluster
# ----------------------------------
print("\nGenerating max_p by cluster bar plot...")

fig_boot3, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, exp_id in enumerate(SOURCE_EXPERIMENTS):
    ax = axes[idx]
    df_exp = bootstrap_posterior_results[exp_id]

    # Group by cluster and compute mean max_p
    cluster_maxp = df_exp.groupby('modal_cluster')['max_p'].agg(['mean', 'std', 'count'])
    cluster_maxp = cluster_maxp.sort_index()

    x = cluster_maxp.index
    y = cluster_maxp['mean']
    yerr = cluster_maxp['std']

    ax.bar(x, y, yerr=yerr, capsize=5, alpha=0.7, edgecolor='black')
    ax.axhline(0.75, color='orange', linestyle='--', alpha=0.5, label='75% threshold')
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Mean Max Posterior Probability', fontsize=12)
    ax.set_title(f'Confidence by Cluster\n{exp_id}', fontsize=14)
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # Add sample sizes as text
    for cluster, row in cluster_maxp.iterrows():
        ax.text(cluster, row['mean'] + row['std'] + 0.02, f"n={int(row['count'])}",
                ha='center', fontsize=9)

plt.tight_layout()
boot_plot3_path = FIGURES_DIR / 'bootstrap_maxp_by_cluster.png'
plt.savefig(boot_plot3_path, dpi=300)
print(f"✓ Saved: {boot_plot3_path}")
plt.close(fig_boot3)

# Visualization 4: Heatmap of Posterior Matrix (for low-confidence embryos)
# --------------------------------------------------------------------------
print("\nGenerating posterior probability heatmap for low-confidence embryos...")

for exp_id in SOURCE_EXPERIMENTS:
    df_exp = bootstrap_posterior_results[exp_id]

    # Select low confidence embryos
    low_conf_embryos = df_exp[df_exp['max_p'] < 0.75].copy()

    if len(low_conf_embryos) == 0:
        print(f"  {exp_id}: No low-confidence embryos, skipping heatmap")
        continue

    # Extract probability columns
    prob_cols = [col for col in df_exp.columns if col.startswith('p_cluster_')]
    cluster_numbers = [int(col.split('_')[-1]) for col in prob_cols]

    # Create matrix: rows=embryos, cols=clusters
    prob_matrix = low_conf_embryos[prob_cols].values

    # Sort embryos by modal cluster for better visualization
    sorted_idx = low_conf_embryos['modal_cluster'].argsort()
    prob_matrix = prob_matrix[sorted_idx, :]

    # Plot heatmap
    fig_heat, ax = plt.subplots(figsize=(10, max(6, len(low_conf_embryos) * 0.2)))

    im = ax.imshow(prob_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)

    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Embryo (low confidence)', fontsize=12)
    ax.set_title(f'Cluster Posterior Probabilities\n{exp_id} (max_p < 75%)', fontsize=14)
    ax.set_xticks(range(len(cluster_numbers)))
    ax.set_xticklabels(cluster_numbers)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Posterior Probability', fontsize=10)

    plt.tight_layout()
    heatmap_path = FIGURES_DIR / f'bootstrap_posterior_heatmap_{exp_id}.png'
    plt.savefig(heatmap_path, dpi=300)
    print(f"  ✓ Saved: {heatmap_path}")
    plt.close(fig_heat)

print("\nAll bootstrap visualizations complete!")

# ============================================================================
# SECTION 12: Bootstrap Results Interpretation
# ============================================================================

print("\n" + "="*80)
print("SECTION 12: Bootstrap Results Interpretation")
print("="*80)

print("\n" + "="*80)
print("OVERALL BOOTSTRAP SUMMARY")
print("="*80)

for exp_id in SOURCE_EXPERIMENTS:
    df_exp = bootstrap_posterior_results[exp_id]

    print(f"\n{exp_id} ({len(df_exp)} embryos):")
    print("-" * 60)

    # Confidence tiers
    high = (df_exp['max_p'] > 0.90).sum()
    moderate = ((df_exp['max_p'] >= 0.75) & (df_exp['max_p'] <= 0.90)).sum()
    low = (df_exp['max_p'] < 0.75).sum()

    print(f"\nConfidence Tiers:")
    print(f"  High (>90%):       {high:3d} embryos ({100*high/len(df_exp):5.1f}%)")
    print(f"  Moderate (75-90%): {moderate:3d} embryos ({100*moderate/len(df_exp):5.1f}%)")
    print(f"  Low (<75%):        {low:3d} embryos ({100*low/len(df_exp):5.1f}%)")

    # Agreement with original projection
    modal_matches_original = (df_exp['modal_cluster'] == df_exp['cluster']).sum()
    print(f"\nAgreement with Original Projection:")
    print(f"  Modal cluster matches: {modal_matches_original}/{len(df_exp)} ({100*modal_matches_original/len(df_exp):.1f}%)")

    # Cluster-specific uncertainty
    print(f"\nMean Confidence by Cluster:")
    cluster_conf = df_exp.groupby('modal_cluster')['max_p'].agg(['mean', 'count'])
    for cluster, row in cluster_conf.iterrows():
        # Get label from the cluster_label column (if it exists)
        category = 'Unknown'
        if 'cluster_label' in df_exp.columns:
            category_series = df_exp[df_exp['modal_cluster'] == cluster]['cluster_label']
            if len(category_series) > 0:
                category = category_series.iloc[0]
        print(f"  Cluster {cluster} ({category:20s}): {row['mean']:.3f} (n={int(row['count'])})")

print("\n" + "="*80)
print("INTERPRETATION GUIDELINES")
print("="*80)

print("""
Bootstrap confidence interpretation:

1. HIGH CONFIDENCE (>90%):
   - Embryo is robustly assigned to this cluster
   - Assignment is stable across resampling variations
   - Can confidently use this assignment for downstream analysis

2. MODERATE CONFIDENCE (75-90%):
   - Embryo is likely in this cluster, but some uncertainty exists
   - May be near a cluster boundary
   - Consider checking trajectories visually
   - Use with caution in critical analyses

3. LOW CONFIDENCE (<75%):
   - Embryo assignment is unstable
   - May switch between 2+ clusters across bootstrap iterations
   - Check second_best_cluster column
   - Consider excluding from cluster-specific analyses
   - Good candidates for visual inspection

Cluster entropy interpretation:
   - Entropy = 0: Always assigned to same cluster (perfect stability)
   - Entropy < 1: Mostly stable, minor switching
   - Entropy > 1.5: High uncertainty, switching across multiple clusters

Nearest distance vs confidence:
   - Ideally: larger distance → lower confidence
   - If large distance BUT high confidence: embryo is clearly different from all references
   - If small distance BUT low confidence: near a cluster boundary
""")

print("\n" + "="*80)
print("FILES GENERATED")
print("="*80)

print("\nBootstrap Results (CSV):")
for exp_id in SOURCE_EXPERIMENTS:
    print(f"  - {exp_id}_bootstrap_posteriors.csv")
print(f"  - combined_bootstrap_posteriors.csv")

print("\nBootstrap Visualizations (PNG):")
print(f"  - bootstrap_maxp_distribution.png")
print(f"  - bootstrap_maxp_vs_distance.png")
print(f"  - bootstrap_maxp_by_cluster.png")
for exp_id in SOURCE_EXPERIMENTS:
    print(f"  - bootstrap_posterior_heatmap_{exp_id}.png (if low-confidence embryos exist)")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("✓ TUTORIAL 04 COMPLETE WITH BOOTSTRAP UNCERTAINTY QUANTIFICATION")
print("="*80)

print(f"\nOutputs saved to:")
print(f"  Projection results: {PROJECTION_DIR}")
print(f"  Figures: {FIGURES_DIR}")
print(f"  Frequency comparison: {RESULTS_DIR}")
print(f"  Bootstrap posteriors: {RESULTS_DIR}")

print(f"\nKey findings (Projection):")
print(f"  - Projected {len(df_all_proj)} embryos onto CEP290 reference clusters")
print(f"  - Chi-square test: χ² = {stats['chi2_statistic']:.3f}, p = {stats['chi2_pvalue']:.4f}")

if stats['chi2_pvalue'] < 0.05:
    print(f"  - SIGNIFICANT batch effect detected (p < 0.05)")
else:
    print(f"  - No significant batch effect detected (p ≥ 0.05)")

print(f"\nKey findings (Bootstrap Uncertainty):")
for exp_id in SOURCE_EXPERIMENTS:
    df_boot = bootstrap_posterior_results[exp_id]
    mean_conf = df_boot['max_p'].mean()
    high_pct = (df_boot['max_p'] > 0.90).mean() * 100
    low_pct = (df_boot['max_p'] < 0.75).mean() * 100
    print(f"  {exp_id}:")
    print(f"    - Mean confidence: {mean_conf:.3f}")
    print(f"    - High confidence (>90%): {high_pct:.1f}%")
    print(f"    - Low confidence (<75%): {low_pct:.1f}%")

print(f"\nInterpretation:")
print(f"  Cluster projection reveals batch effects caused by temporal coverage:")
print(f"  - 20260122: {df_viz[df_viz['experiment_id']=='20260122']['predicted_stage_hpf'].min():.0f}-{df_viz[df_viz['experiment_id']=='20260122']['predicted_stage_hpf'].max():.0f} hpf")
print(f"  - 20260124: {df_viz[df_viz['experiment_id']=='20260124']['predicted_stage_hpf'].min():.0f}-{df_viz[df_viz['experiment_id']=='20260124']['predicted_stage_hpf'].max():.0f} hpf")
print(f"  Use reference clusters as 'anchors' to normalize cross-experiment comparisons.")

print(f"\nBenchmark Results:")
print(f"  - Category agreement: {category_agreement_pct:.1f}%")
print(f"  - Mean distance difference: {distance_diff.mean():.3f}")
if category_agreement_pct >= 95:
    print(f"  ✓ High agreement suggests NaN-aware DTW eliminates splitting need")
else:
    print(f"  ⚠ Low agreement suggests potential issues with temporal handling")

print(f"\nBootstrap Uncertainty Summary:")
print(f"  - Bootstrap provides per-embryo confidence in cluster assignments")
print(f"  - High confidence embryos (>90%) are robust to sampling variation")
print(f"  - Low confidence embryos (<75%) may be near cluster boundaries")
print(f"  - Use max_p and entropy to filter embryos for downstream analysis")

print(f"\nNext steps:")
print(f"  - Review trajectory plots to understand projection quality")
print(f"  - Check distance distributions for outliers")
print(f"  - Filter embryos by bootstrap confidence for cluster-specific analyses")
print(f"  - Investigate low-confidence embryos (potential boundary cases)")
print(f"  - Apply same approach to other mutant lines (tmem67, etc.)")

print("="*80)
