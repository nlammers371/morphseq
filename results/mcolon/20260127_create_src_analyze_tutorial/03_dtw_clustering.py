"""
Tutorial 03: DTW Clustering Pipeline

Demonstrates DTW-based trajectory clustering on CEP290 embryos.

Pipeline steps:
1. Extract and interpolate trajectories (multi-feature)
2. Outlier filtering (IQR-based)
3. Compute MD-DTW distance matrix
4. K-selection analysis
5. Bootstrap hierarchical clustering
6. Posterior analysis and membership classification

Key API usage:
- extract_trajectories_df() + interpolate_to_common_grid_multi_df()
- prepare_multivariate_array() + compute_md_dtw_distance_matrix()
- evaluate_k_range() + plot_k_selection()
- run_bootstrap_hierarchical()
- analyze_bootstrap_results() + classify_membership_2d()
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

# Setup output directory
OUTPUT_DIR = Path(__file__).parent / "output"
FIGURES_DIR = OUTPUT_DIR / "figures" / "03"  # Tutorial 03 specific directory
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load data from both CEP290 experiments (direct CSV load)
print("Loading CEP290 data (20260122, 20260124) from build04...")
meta_dir = project_root / 'morphseq_playground' / 'metadata' / 'build04_output'
df1 = pd.read_csv(meta_dir / 'qc_staged_20260122.csv')
df2 = pd.read_csv(meta_dir / 'qc_staged_20260124.csv')
df = pd.concat([df1, df2], ignore_index=True)

# Filter to valid embryos
df = df[df['use_embryo_flag']].copy()
print(f"Loaded {len(df['embryo_id'].unique())} embryos across {len(df)} timepoints")

# ============================================================================
# Quick HPF coverage diagnostics (all experiments, pre-subsample)
# ============================================================================
from src.analyze.viz.hpf_coverage import experiment_hpf_coverage, plot_hpf_overlap_quick

print("\nComputing HPF coverage across experiments...")
bins_mid, cover_df, cov_count = experiment_hpf_coverage(
    df,
    experiment_col="experiment_id",
    hpf_col="predicted_stage_hpf",
    embryo_col="embryo_id",
    bin_width=0.5,
    min_embryos_per_bin=3,
)

hpf_plot_path = FIGURES_DIR / "00_hpf_coverage.png"
hpf_heatmap_path = FIGURES_DIR / "00_hpf_coverage_heatmap.png"
hpf_start, hpf_end = plot_hpf_overlap_quick(
    bins_mid,
    cov_count,
    cover_df=cover_df,
    min_experiments=2,  # CEP290 has 2 experiments (20260122, 20260124)
    show_heatmap=True,
    coverage_plot_path=hpf_plot_path,
    heatmap_path=hpf_heatmap_path,
    show=False,
)

hpf_start = 25
hpf_end   = 50 # overriding because theree are batch affects in the timing 
print()
print(f"HPF overlap window: {hpf_start} to {hpf_end}")
print(f"Saved coverage plots: {hpf_plot_path}, {hpf_heatmap_path}")

# Determine time window for clustering
if hpf_start is not None and hpf_end is not None:
    time_window = (hpf_start, hpf_end)
    print(f"\nUsing optimal overlap window for clustering: {hpf_start} - {hpf_end} hpf")
else:
    time_window = None
    print("\nNo optimal overlap window found, using full time range")

# Run on full dataset
print(f"Using full dataset: {len(df['embryo_id'].unique())} embryos")

# ============================================================================
# Step 1: Compute MD-DTW distance matrix with optimal time window
# ============================================================================
print("\n" + "="*80)
print("STEP 1: COMPUTE MD-DTW DISTANCE MATRIX")
print("="*80)

from src.analyze.trajectory_analysis.utilities.dtw_utils import compute_trajectory_distances

# Use two features for clustering: baseline_deviation_normalized + total_length_um
FEATURES = ['baseline_deviation_normalized']

print(f"Features: {FEATURES}")
print(f"Time window: {time_window}")

# Compute distances using the wrapper function that handles time filtering
D, embryo_ids, time_grid = compute_trajectory_distances(
    df,
    metrics=FEATURES,
    time_col='predicted_stage_hpf',
    time_window=time_window,  # Use the optimal overlap window
    embryo_id_col='embryo_id',
    normalize=True,  # Z-score normalization (recommended)
    sakoe_chiba_radius=20,  # Warping constraint: 10 steps * 0.5 hpf = ±5 hpf flexibility
    verbose=True
)

print(f"\nDistance matrix shape: {D.shape}")
print(f"Distance range: [{D.min():.2f}, {D.max():.2f}]")
print(f"Number of embryos: {len(embryo_ids)}")

# Save distance matrix
np.save(RESULTS_DIR / "dtw_distance_matrix.npy", D)
np.save(RESULTS_DIR / "embryo_ids.npy", embryo_ids)
print(f"\nSaved distance matrix to: {RESULTS_DIR / 'dtw_distance_matrix.npy'}")

# ============================================================================
# Step 2: K-selection with plots (hierarchical clustering)
# ===========================================================================
print("\n" + "="*80)
print("STEP 2: K-SELECTION WITH PLOTS")
print("="*80)

from src.analyze.trajectory_analysis.clustering import run_k_selection_with_plots

k_selection_dir = FIGURES_DIR / "k_selection_results"
print(f"\nRunning k-selection. Output: {k_selection_dir}")

# Filter df to only embryos in the distance matrix
df_filtered = df[df['embryo_id'].isin(embryo_ids)].copy()
print(f"Filtered DataFrame to {len(df_filtered['embryo_id'].unique())} embryos in distance matrix")

k_results = run_k_selection_with_plots(
    df=df_filtered,
    D=D,
    embryo_ids=embryo_ids,
    output_dir=k_selection_dir,
    k_range=[2, 3, 4, 5, 6],
    n_bootstrap=50,  # Reduced for speed
    method='kmedoids',
    plotting_metrics=['baseline_deviation_normalized', 'total_length_um'],
    x_col='predicted_stage_hpf',
    iqr_multiplier=2,  # Use 2x IQR for outlier filtering
    verbose=True
)

# ============================================================================
# Summary of k-selection
# ============================================================================
print("\n" + "="*80)
print("K-SELECTION SUMMARY")
print("="*80)
print(f"\nBest k (recommended): {k_results['best_k']}")
print(f"Note: k=3 also works well (94.7% core assignments)")
print(f"\nAll outputs saved to: {k_selection_dir}")
print("\nGenerated files:")
print(f"  - k2_membership_trajectories.png → k6_membership_trajectories.png")
print(f"  - k_selection_comparison.png (summary metrics)")
print(f"  - k_selection_summary.csv")
print(f"  - cluster_assignments.csv")
print(f"  - k_results.pkl")

# ============================================================================
# Step 3: Post-clustering analysis - Use k=3 clusters
# ============================================================================
print("\n" + "="*80)
print("STEP 3: POST-CLUSTERING ANALYSIS (k=3)")
print("="*80)

# Load cluster assignments
cluster_assignments = pd.read_csv(k_selection_dir / 'cluster_assignments.csv')
print(f"\nLoaded cluster assignments: {len(cluster_assignments)} embryos")

# Use k=3 clustering (both k=2 and k=3 had 94.7% core, but k=3 shows more structure)
cluster_col = 'clustering_k_3'
print(f"Using {cluster_col}")
print(f"Cluster distribution:\n{cluster_assignments[cluster_col].value_counts().sort_index()}")

# Map clusters back to the original dataframe
df_clustered = df.merge(
    cluster_assignments[['embryo_id', cluster_col]],
    on='embryo_id',
    how='left'
)

# Rename cluster column for clarity
df_clustered = df_clustered.rename(columns={cluster_col: 'cluster'})

print(f"\nMerged clusters with trajectory data:")
print(f"  Total rows: {len(df_clustered)}")
print(f"  Embryos with cluster assignments: {df_clustered['cluster'].notna().sum()} / {len(df_clustered)}")
print(f"  Unique embryos: {df_clustered['embryo_id'].nunique()}")

# Save the clustered dataframe for reuse
clustered_df_path = RESULTS_DIR / 'df_with_clusters_k3.csv'
df_clustered.to_csv(clustered_df_path, index=False)
print(f"\n✓ Saved clustered dataframe: {clustered_df_path}")
print("  (This can be reused for downstream analysis)")

# ============================================================================
# Step 4: Trajectory visualization by genotype and cluster
# ============================================================================
print("\n" + "="*80)
print("STEP 4: TRAJECTORY PLOTS - GENOTYPE × CLUSTER")
print("="*80)

from src.analyze.viz.plotting import plot_feature_over_time

# Filter to embryos with cluster assignments (those in time window)
df_plot = df_clustered[df_clustered['cluster'].notna()].copy()
df_plot['cluster'] = df_plot['cluster'].astype(int)

print(f"\nPlotting trajectories for {df_plot['embryo_id'].nunique()} clustered embryos")
print(f"Genotypes: {sorted(df_plot['genotype'].unique())}")
print(f"Clusters: {sorted(df_plot['cluster'].unique())}")

# Plot 1: Curvature trajectories - faceted by genotype (col) and cluster (row)
fig1 = plot_feature_over_time(
    df_plot,
    features='baseline_deviation_normalized',
    time_col='predicted_stage_hpf',
    id_col='embryo_id',
    color_by='genotype',
    facet_col='genotype',
    facet_row='cluster',
    title='Curvature Trajectories by Genotype and Cluster (k=3)',
    backend='matplotlib',
    bin_width=2.0,
)

plot1_path = FIGURES_DIR / 'cluster_analysis' / '01_trajectories_genotype_cluster.png'
plot1_path.parent.mkdir(exist_ok=True)
plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {plot1_path}")
plt.close(fig1)

# ============================================================================
# Step 5: Batch effect check - color by experiment
# ============================================================================
print("\n" + "="*80)
print("STEP 5: BATCH EFFECT CHECK - COLOR BY EXPERIMENT")
print("="*80)

# Plot 2: Same as above but colored by experiment to check for batch effects
fig2 = plot_feature_over_time(
    df_plot,
    features='baseline_deviation_normalized',
    time_col='predicted_stage_hpf',
    id_col='embryo_id',
    color_by='experiment_id',
    facet_col='genotype',
    facet_row='cluster',
    title='Batch Effect Check: Trajectories Colored by Experiment (k=3)',
    backend='matplotlib',
    bin_width=2.0,
)

plot2_path = FIGURES_DIR / 'cluster_analysis' / '02_batch_effect_check.png'
plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {plot2_path}")
plt.close(fig2)

# ============================================================================
# Step 6: Proportion analysis - genotype vs cluster distributions
# ============================================================================
print("\n" + "="*80)
print("STEP 6: PROPORTION ANALYSIS")
print("="*80)

# Get unique embryo-genotype-cluster mappings
embryo_summary = df_plot[['embryo_id', 'genotype', 'cluster', 'experiment_id']].drop_duplicates()

print(f"\nTotal embryos with clusters: {len(embryo_summary)}")
print(f"\nGenotype distribution:")
print(embryo_summary['genotype'].value_counts())
print(f"\nCluster distribution:")
print(embryo_summary['cluster'].value_counts().sort_index())

# Plot A: For each cluster, what % are each genotype?
print("\n--- Plot A: Genotype distribution within each cluster ---")
cluster_genotype_counts = embryo_summary.groupby(['cluster', 'genotype']).size().reset_index(name='count')
cluster_totals = embryo_summary.groupby('cluster').size().reset_index(name='total')
cluster_genotype_props = cluster_genotype_counts.merge(cluster_totals, on='cluster')
cluster_genotype_props['proportion'] = cluster_genotype_props['count'] / cluster_genotype_props['total']

fig3, ax3 = plt.subplots(figsize=(10, 6))
genotypes = sorted(embryo_summary['genotype'].unique())
clusters = sorted(embryo_summary['cluster'].unique())
x = np.arange(len(clusters))
width = 0.8 / len(genotypes)

for i, genotype in enumerate(genotypes):
    props = []
    for cluster in clusters:
        subset = cluster_genotype_props[(cluster_genotype_props['cluster'] == cluster) &
                                        (cluster_genotype_props['genotype'] == genotype)]
        props.append(subset['proportion'].values[0] if len(subset) > 0 else 0)
    ax3.bar(x + i * width, props, width, label=genotype)

ax3.set_xlabel('Cluster')
ax3.set_ylabel('Proportion')
ax3.set_title('Genotype Distribution within Each Cluster (k=3)')
ax3.set_xticks(x + width * (len(genotypes) - 1) / 2)
ax3.set_xticklabels([f'Cluster {c}' for c in clusters])
ax3.legend(title='Genotype')
ax3.set_ylim([0, 1])
plt.tight_layout()

plot3_path = FIGURES_DIR / 'cluster_analysis' / '03_genotype_per_cluster.png'
plt.savefig(plot3_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {plot3_path}")
plt.close(fig3)

# Plot B: For each genotype, what % are in each cluster?
print("\n--- Plot B: Cluster distribution within each genotype ---")
genotype_cluster_counts = embryo_summary.groupby(['genotype', 'cluster']).size().reset_index(name='count')
genotype_totals = embryo_summary.groupby('genotype').size().reset_index(name='total')
genotype_cluster_props = genotype_cluster_counts.merge(genotype_totals, on='genotype')
genotype_cluster_props['proportion'] = genotype_cluster_props['count'] / genotype_cluster_props['total']

fig4, ax4 = plt.subplots(figsize=(10, 6))
x = np.arange(len(genotypes))
width = 0.8 / len(clusters)

for i, cluster in enumerate(clusters):
    props = []
    for genotype in genotypes:
        subset = genotype_cluster_props[(genotype_cluster_props['genotype'] == genotype) &
                                        (genotype_cluster_props['cluster'] == cluster)]
        props.append(subset['proportion'].values[0] if len(subset) > 0 else 0)
    ax4.bar(x + i * width, props, width, label=f'Cluster {cluster}')

ax4.set_xlabel('Genotype')
ax4.set_ylabel('Proportion')
ax4.set_title('Cluster Distribution within Each Genotype (k=3)')
ax4.set_xticks(x + width * (len(clusters) - 1) / 2)
ax4.set_xticklabels(genotypes, rotation=45, ha='right')
ax4.legend(title='Cluster')
ax4.set_ylim([0, 1])
plt.tight_layout()

plot4_path = FIGURES_DIR / 'cluster_analysis' / '04_cluster_per_genotype.png'
plt.savefig(plot4_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {plot4_path}")
plt.close(fig4)

# Print numerical summaries
print("\n" + "="*80)
print("NUMERICAL SUMMARIES")
print("="*80)
print("\nGenotype distribution within each cluster:")
for cluster in clusters:
    print(f"\nCluster {cluster}:")
    subset = cluster_genotype_props[cluster_genotype_props['cluster'] == cluster]
    for _, row in subset.iterrows():
        print(f"  {row['genotype']}: {row['count']:.0f} ({row['proportion']*100:.1f}%)")

print("\nCluster distribution within each genotype:")
for genotype in genotypes:
    print(f"\n{genotype}:")
    subset = genotype_cluster_props[genotype_cluster_props['genotype'] == genotype]
    for _, row in subset.iterrows():
        print(f"  Cluster {row['cluster']:.0f}: {row['count']:.0f} ({row['proportion']*100:.1f}%)")

print("\n" + "="*80)
print("✓ Tutorial 03 complete!")
print("="*80)
print(f"\nAll results saved to:")
print(f"  K-selection: {k_selection_dir}")
print(f"  Cluster analysis: {FIGURES_DIR / 'cluster_analysis'}")
print(f"  Clustered dataframe: {clustered_df_path}")
print("\nNext steps:")
print("  - Review trajectory plots to understand cluster patterns")
print("  - Check batch effect plot (should show no experiment bias)")
print("  - Interpret genotype-cluster associations from proportion plots")
print("="*80)
