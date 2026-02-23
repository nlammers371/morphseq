"""
Tutorial 04b: Compare Separate vs Combined Clustering Methods

Follow-up to 04_cluster_projection.py that visualizes trajectory differences
between the two clustering approaches (separate vs combined projection).

This script helps determine if disagreements are meaningful phenotypic differences
or just noise from bootstrapping/boundary cases.

Approach:
---------
- Load benchmark comparison results from 04
- Merge cluster assignments with trajectory data
- Create faceted plots comparing:
  * Row: Clustering method (Separate vs Combined)
  * Color: Cluster assignment
  * Additional facet: Genotype (optional)

This allows visual inspection of whether embryos assigned to different clusters
actually show different phenotypic trajectories.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.analyze.viz.plotting import plot_feature_over_time

# Setup paths
OUTPUT_DIR = Path(__file__).parent / "output"
FIGURES_DIR = OUTPUT_DIR / "figures" / "04b"
RESULTS_DIR = OUTPUT_DIR / "results"

for d in [FIGURES_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("="*80)
print("Tutorial 04b: Compare Separate vs Combined Clustering Methods")
print("="*80)

# ============================================================================
# Step 1: Load Benchmark Comparison Results
# ============================================================================
print("\n" + "="*80)
print("STEP 1: LOAD BENCHMARK COMPARISON")
print("="*80)

comparison_path = RESULTS_DIR / "benchmark_comparison.csv"
df_compare = pd.read_csv(comparison_path)

print(f"\nLoaded comparison data: {len(df_compare)} embryos")
print(f"\nAgreement summary:")
print(f"  Cluster agreement: {(df_compare['cluster_combined'] == df_compare['cluster_separate']).sum()}/{len(df_compare)}")
print(f"  Category agreement: {(df_compare['category_combined'] == df_compare['category_separate']).sum()}/{len(df_compare)}")

# Identify agreement status
df_compare['category_agrees'] = df_compare['category_combined'] == df_compare['category_separate']
df_compare['cluster_agrees'] = df_compare['cluster_combined'] == df_compare['cluster_separate']

print(f"\nDisagreements by category:")
disagreements = df_compare[~df_compare['category_agrees']]
print(f"  Total: {len(disagreements)} embryos")

if len(disagreements) > 0:
    transitions = disagreements.groupby(['category_separate', 'category_combined']).size().reset_index(name='count')
    transitions = transitions.sort_values('count', ascending=False)
    print(f"\n  Top transitions:")
    for _, row in transitions.head(10).iterrows():
        print(f"    {row['category_separate']:20s} → {row['category_combined']:20s}: {row['count']}")

# ============================================================================
# Step 2: Load Trajectory Data
# ============================================================================
print("\n" + "="*80)
print("STEP 2: LOAD TRAJECTORY DATA")
print("="*80)

# Load source experiment data
SOURCE_EXPERIMENTS = ['20260122', '20260124']
meta_dir = project_root / 'morphseq_playground' / 'metadata' / 'build04_output'

source_dfs = {}
for exp_id in SOURCE_EXPERIMENTS:
    print(f"\nLoading experiment {exp_id}...")
    df_exp = pd.read_csv(meta_dir / f'qc_staged_{exp_id}.csv')
    df_exp = df_exp[df_exp['use_embryo_flag']].copy()
    df_exp['experiment_id'] = exp_id

    print(f"  Total: {len(df_exp)} rows, {df_exp['embryo_id'].nunique()} embryos")
    print(f"  Time range: {df_exp['predicted_stage_hpf'].min():.1f} - {df_exp['predicted_stage_hpf'].max():.1f} hpf")

    source_dfs[exp_id] = df_exp

# Combine trajectory data
df_traj = pd.concat(source_dfs.values(), ignore_index=True)
print(f"\nCombined trajectory data: {len(df_traj)} rows, {df_traj['embryo_id'].nunique()} embryos")

# ============================================================================
# Step 3: Create Comparison DataFrames for Visualization
# ============================================================================
print("\n" + "="*80)
print("STEP 3: PREPARE COMPARISON DATA")
print("="*80)

# Create two versions of the trajectory data with different cluster assignments

# Version 1: Separate method assignments
df_separate_assignments = df_compare[['embryo_id', 'category_separate', 'cluster_separate']].copy()
df_separate_assignments = df_separate_assignments.rename(columns={
    'category_separate': 'cluster_category',
    'cluster_separate': 'cluster'
})
df_separate_assignments['clustering_method'] = 'Separate'

df_traj_separate = df_traj.merge(df_separate_assignments, on='embryo_id', how='inner')
print(f"\nSeparate method trajectories: {len(df_traj_separate)} rows, {df_traj_separate['embryo_id'].nunique()} embryos")

# Version 2: Combined method assignments
df_combined_assignments = df_compare[['embryo_id', 'category_combined', 'cluster_combined']].copy()
df_combined_assignments = df_combined_assignments.rename(columns={
    'category_combined': 'cluster_category',
    'cluster_combined': 'cluster'
})
df_combined_assignments['clustering_method'] = 'Combined'

df_traj_combined = df_traj.merge(df_combined_assignments, on='embryo_id', how='inner')
print(f"Combined method trajectories: {len(df_traj_combined)} rows, {df_traj_combined['embryo_id'].nunique()} embryos")

# Combine both for faceted plotting
df_viz_both = pd.concat([df_traj_separate, df_traj_combined], ignore_index=True)
print(f"\nCombined visualization data: {len(df_viz_both)} rows")

# Add agreement status for filtering
agreement_map = dict(zip(df_compare['embryo_id'], df_compare['category_agrees']))
df_viz_both['category_agrees'] = df_viz_both['embryo_id'].map(agreement_map)

# ============================================================================
# Step 4: Plot All Embryos - Separate vs Combined
# ============================================================================
print("\n" + "="*80)
print("STEP 4: PLOT ALL EMBRYOS (SEPARATE VS COMBINED)")
print("="*80)

print("\nGenerating faceted plot: clustering_method (rows) × cluster_category (cols) × genotype (color)...")

fig1 = plot_feature_over_time(
    df_viz_both,
    features='baseline_deviation_normalized',
    time_col='predicted_stage_hpf',
    id_col='embryo_id',
    color_by='genotype',
    facet_row='clustering_method',
    facet_col='cluster_category',
    title='Cluster Assignments: Separate vs Combined Projection',
    backend='matplotlib',
    bin_width=2.0,
)

plot1_path = FIGURES_DIR / 'all_embryos_method_comparison.png'
plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {plot1_path}")
plt.close(fig1)

# ============================================================================
# Step 5: Plot Only Disagreements
# ============================================================================
print("\n" + "="*80)
print("STEP 5: PLOT DISAGREEMENTS ONLY")
print("="*80)

# Filter to embryos where category assignments differ
df_viz_disagree = df_viz_both[~df_viz_both['category_agrees']].copy()

print(f"\nDisagreement embryos: {df_viz_disagree['embryo_id'].nunique()} embryos")
print(f"  Total rows: {len(df_viz_disagree)}")

if len(df_viz_disagree) > 0:
    print("\nGenerating faceted plot for disagreements...")

    fig2 = plot_feature_over_time(
        df_viz_disagree,
        features='baseline_deviation_normalized',
        time_col='predicted_stage_hpf',
        id_col='embryo_id',
        color_by='genotype',
        facet_row='clustering_method',
        facet_col='cluster_category',
        title='Disagreement Cases: Separate vs Combined Assignment',
        backend='matplotlib',
        bin_width=2.0,
    )

    plot2_path = FIGURES_DIR / 'disagreements_method_comparison.png'
    plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {plot2_path}")
    plt.close(fig2)
else:
    print("\n✓ No disagreements - all embryos assigned to same category!")

# Skip the by-genotype section since we're already coloring by genotype

# ============================================================================
# Step 7: Detailed Disagreement Analysis
# ============================================================================
print("\n" + "="*80)
print("STEP 7: DISAGREEMENT ANALYSIS")
print("="*80)

if len(disagreements) > 0:
    # For each transition type, plot example trajectories
    print("\nAnalyzing major transition patterns...")

    # Get top 5 transition types
    top_transitions = transitions.head(5)

    for _, trans_row in top_transitions.iterrows():
        cat_sep = trans_row['category_separate']
        cat_comb = trans_row['category_combined']
        count = trans_row['count']

        print(f"\n  Transition: {cat_sep} → {cat_comb} ({count} embryos)")

        # Get embryos for this transition
        trans_embryos = disagreements[
            (disagreements['category_separate'] == cat_sep) &
            (disagreements['category_combined'] == cat_comb)
        ]['embryo_id'].tolist()

        # Filter visualization data
        df_trans = df_viz_both[df_viz_both['embryo_id'].isin(trans_embryos)].copy()

        if len(df_trans) > 0:
            # Create safe filename
            filename = f"transition_{cat_sep.replace(' ', '_')}_to_{cat_comb.replace(' ', '_')}.png"

            fig_trans = plot_feature_over_time(
                df_trans,
                features='baseline_deviation_normalized',
                time_col='predicted_stage_hpf',
                id_col='embryo_id',
                color_by='genotype',
                facet_row='clustering_method',
                facet_col='cluster_category',
                title=f'Transition: {cat_sep} → {cat_comb} ({count} embryos)',
                backend='matplotlib',
                bin_width=2.0,
            )

            plot_trans_path = FIGURES_DIR / filename
            plt.savefig(plot_trans_path, dpi=150, bbox_inches='tight')
            print(f"    ✓ Saved: {plot_trans_path}")
            plt.close(fig_trans)

# ============================================================================
# Step 8: Distance Analysis for Disagreements
# ============================================================================
print("\n" + "="*80)
print("STEP 8: DISTANCE ANALYSIS")
print("="*80)

# Compare distances for agreements vs disagreements
df_compare['distance_change'] = df_compare['distance_combined'] - df_compare['distance_separate']
df_compare['abs_distance_change'] = np.abs(df_compare['distance_change'])

print("\nDistance statistics:")
print(f"\nAgreements (category_agrees=True):")
agrees = df_compare[df_compare['category_agrees']]
print(f"  Count: {len(agrees)}")
print(f"  Mean distance change: {agrees['distance_change'].mean():.3f}")
print(f"  Mean |distance change|: {agrees['abs_distance_change'].mean():.3f}")
print(f"  Std distance change: {agrees['distance_change'].std():.3f}")

print(f"\nDisagreements (category_agrees=False):")
disagrees = df_compare[~df_compare['category_agrees']]
print(f"  Count: {len(disagrees)}")
print(f"  Mean distance change: {disagrees['distance_change'].mean():.3f}")
print(f"  Mean |distance change|: {disagrees['abs_distance_change'].mean():.3f}")
print(f"  Std distance change: {disagrees['distance_change'].std():.3f}")

# Plot distance comparison
fig_dist, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Distance change histogram
ax1 = axes[0]
ax1.hist(agrees['distance_change'], bins=30, alpha=0.5, label='Agreements', color='green', edgecolor='black')
ax1.hist(disagrees['distance_change'], bins=30, alpha=0.5, label='Disagreements', color='red', edgecolor='black')
ax1.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax1.set_xlabel('Distance Change (Combined - Separate)', fontsize=11)
ax1.set_ylabel('Count', fontsize=11)
ax1.set_title('Distance Change Distribution', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Scatter of separate vs combined distances colored by agreement
ax2 = axes[1]
ax2.scatter(agrees['distance_separate'], agrees['distance_combined'],
           alpha=0.5, s=30, c='green', label='Agreements')
ax2.scatter(disagrees['distance_separate'], disagrees['distance_combined'],
           alpha=0.5, s=30, c='red', label='Disagreements')
max_dist = max(df_compare['distance_separate'].max(), df_compare['distance_combined'].max())
ax2.plot([0, max_dist], [0, max_dist], 'k--', alpha=0.5, linewidth=2, label='y=x')
ax2.set_xlabel('Distance (Separate)', fontsize=11)
ax2.set_ylabel('Distance (Combined)', fontsize=11)
ax2.set_title('Distance Comparison by Agreement Status', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plot_dist_path = FIGURES_DIR / 'distance_analysis.png'
plt.savefig(plot_dist_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: {plot_dist_path}")
plt.close(fig_dist)

# ============================================================================
# Step 9: Statistical Test - Are Disagreements Random?
# ============================================================================
print("\n" + "="*80)
print("STEP 9: STATISTICAL TESTS")
print("="*80)

from scipy.stats import ttest_ind, mannwhitneyu

# Test if distance changes differ between agreements and disagreements
print("\nTesting if distance changes differ between agreements vs disagreements:")

# T-test
t_stat, t_pval = ttest_ind(agrees['abs_distance_change'], disagrees['abs_distance_change'])
print(f"\n  T-test (absolute distance change):")
print(f"    t-statistic: {t_stat:.3f}")
print(f"    p-value: {t_pval:.4f}")

# Mann-Whitney U test (non-parametric)
u_stat, u_pval = mannwhitneyu(agrees['abs_distance_change'], disagrees['abs_distance_change'])
print(f"\n  Mann-Whitney U test (absolute distance change):")
print(f"    U-statistic: {u_stat:.3f}")
print(f"    p-value: {u_pval:.4f}")

if t_pval < 0.05:
    print(f"\n  ✓ SIGNIFICANT: Disagreements have different distance changes (p < 0.05)")
    print(f"    This suggests disagreements are NOT random - they occur when distances change more")
else:
    print(f"\n  No significant difference in distance changes (p ≥ 0.05)")
    print(f"    Disagreements may be due to random variation near cluster boundaries")

# Test by experiment
print("\n" + "="*80)
print("DISAGREEMENT BREAKDOWN BY EXPERIMENT")
print("="*80)

for exp_id in SOURCE_EXPERIMENTS:
    df_exp_comp = df_compare[df_compare['experiment_id'] == exp_id]
    exp_agrees = df_exp_comp['category_agrees'].sum()
    exp_total = len(df_exp_comp)

    if exp_total > 0:
        exp_pct = exp_agrees / exp_total * 100

        print(f"\n{exp_id}:")
        print(f"  Total: {exp_total} embryos")
        print(f"  Agreements: {exp_agrees} ({exp_pct:.1f}%)")
        print(f"  Disagreements: {exp_total - exp_agrees} ({100 - exp_pct:.1f}%)")

        # Distance stats by experiment
        exp_disagrees = df_exp_comp[~df_exp_comp['category_agrees']]
        if len(exp_disagrees) > 0:
            print(f"  Mean |distance change| for disagreements: {exp_disagrees['abs_distance_change'].mean():.3f}")
    else:
        print(f"\n{exp_id}: No data (experiment_id may not be in comparison data)")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("✓ TUTORIAL 04b COMPLETE")
print("="*80)

print(f"\nOutputs saved to:")
print(f"  Figures: {FIGURES_DIR}")

print(f"\nKey visualizations:")
print(f"  - all_embryos_method_comparison.png - All embryos, both methods side-by-side")
print(f"  - disagreements_method_comparison.png - Only embryos with different assignments")
print(f"  - [genotype]_method_comparison.png - Per-genotype comparisons")
print(f"  - transition_*.png - Detailed views of major transition types")
print(f"  - distance_analysis.png - Distance change patterns")

print(f"\nInterpretation:")
print(f"  - Visual inspection can reveal if disagreements are meaningful or noise")
print(f"  - Look for systematic differences in trajectory shapes between methods")
print(f"  - Check if disagreements cluster at specific timepoints or genotypes")

if t_pval < 0.05:
    print(f"\n  Statistical analysis suggests disagreements are NON-RANDOM:")
    print(f"  - Embryos with larger distance changes more likely to be reassigned")
    print(f"  - This indicates clustering method affects results systematically")
else:
    print(f"\n  Statistical analysis suggests disagreements may be RANDOM:")
    print(f"  - Distance changes similar between agreements and disagreements")
    print(f"  - May indicate noise from bootstrapping or boundary cases")

print("="*80)
