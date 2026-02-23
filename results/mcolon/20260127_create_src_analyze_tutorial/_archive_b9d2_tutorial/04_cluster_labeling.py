"""
Tutorial 04: Cluster Labeling + PCA Visualization

Demonstrates manual cluster annotation and visualization in PCA space.

Key steps:
1. Load clustering results from Tutorial 03
2. Map cluster IDs to meaningful phenotype labels
3. Visualize clusters in PCA space
4. Examine cluster composition by genotype

Key API usage:
- Manual cluster labeling (domain-specific knowledge)
- plot_3d_scatter() with cluster labels
- Cross-tabulation of clusters vs genotypes
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle

# Setup directories
OUTPUT_DIR = Path(__file__).parent / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"

# Load data from both B9D2 experiments
from src.analyze.trajectory_analysis.data_loading import load_experiment_dataframe

print("Loading experiment data...")
df1 = load_experiment_dataframe('20251121', format_version='df03')
df2 = load_experiment_dataframe('20251125', format_version='df03')
df = pd.concat([df1, df2], ignore_index=True)
df = df[df['use_embryo_flag']].copy()

# Load PCA results from Tutorial 02
from src.analyze.utils import fit_transform_pca
df_pca, pca, scaler, z_mu_cols = fit_transform_pca(df, n_components=3)
pca_cols = ['PCA_1', 'PCA_2', 'PCA_3']

# ============================================================================
# Step 1: Load clustering results
# ============================================================================
print("\n" + "="*80)
print("STEP 1: LOAD CLUSTERING RESULTS")
print("="*80)

# Load cluster membership from Tutorial 03
membership_df = pd.read_csv(RESULTS_DIR / "cluster_membership.csv")
print(f"\nLoaded cluster membership for {len(membership_df)} embryos")
print(f"Clusters: {membership_df['cluster_id'].unique()}")

# ============================================================================
# Step 2: Map clusters to phenotype labels
# ============================================================================
print("\n" + "="*80)
print("STEP 2: MAP CLUSTERS TO PHENOTYPE LABELS")
print("="*80)

# Based on manual inspection of trajectories and genotype composition:
# - Cluster 0: Short Body Axis (formerly CE-like, but includes many hets)
# - Cluster 1: Homozygous B9D2 (homo without short body axis)
# - Cluster 2: Not Penetrant (WT + Het that look wildtype-like)
#
# NOTE: These mappings are based on the previous B9D2 analysis.
# User should inspect clustering results and adjust mappings as needed.

CLUSTER_LABEL_MAP = {
    0: 'Short Body Axis',
    1: 'Homozygous B9D2',
    2: 'Not Penetrant',
}

# Apply cluster labels
membership_df['cluster_label'] = membership_df['cluster_id'].map(CLUSTER_LABEL_MAP)

print("\nCluster label mapping:")
for cluster_id, label in CLUSTER_LABEL_MAP.items():
    n = len(membership_df[membership_df['cluster_id'] == cluster_id])
    print(f"  Cluster {cluster_id} → '{label}' ({n} embryos)")

# Save labeled membership
membership_df.to_csv(RESULTS_DIR / "cluster_membership_labeled.csv", index=False)
print(f"\nSaved labeled membership to: {RESULTS_DIR / 'cluster_membership_labeled.csv'}")

# ============================================================================
# Step 3: Merge cluster labels into main dataframe
# ============================================================================
print("\n" + "="*80)
print("STEP 3: MERGE LABELS INTO DATAFRAME")
print("="*80)

# Create embryo-level lookup
embryo_labels = membership_df[['embryo_id', 'cluster_id', 'cluster_label', 'membership_class']].copy()

# Merge into PCA dataframe
df_pca = df_pca.merge(embryo_labels, on='embryo_id', how='left')

print(f"\nMerged labels into {len(df_pca)} rows")
print(f"Embryos with cluster labels: {df_pca['cluster_label'].notna().sum()}")

# ============================================================================
# Step 4: Examine cluster composition by genotype
# ============================================================================
print("\n" + "="*80)
print("STEP 4: CLUSTER COMPOSITION BY GENOTYPE")
print("="*80)

# Cross-tabulation
cluster_genotype_counts = pd.crosstab(
    df_pca.drop_duplicates('embryo_id')['cluster_label'],
    df_pca.drop_duplicates('embryo_id')['genotype'],
    margins=True,
)

print("\nCluster composition by genotype:")
print(cluster_genotype_counts)

# Save composition table
cluster_genotype_counts.to_csv(RESULTS_DIR / "cluster_genotype_composition.csv")
print(f"\nSaved composition table to: {RESULTS_DIR / 'cluster_genotype_composition.csv'}")

# ============================================================================
# Step 5: Visualize clusters in PCA space
# ============================================================================
print("\n" + "="*80)
print("STEP 5: VISUALIZE CLUSTERS IN PCA SPACE")
print("="*80)

from src.analyze.viz.plotting import plot_3d_scatter

# Define color lookup for cluster labels
CLUSTER_COLOR_MAP = {
    'Short Body Axis': '#d62728',      # Red
    'Homozygous B9D2': '#ff7f0e',      # Orange
    'Not Penetrant': '#2ca02c',        # Green
}

print("\n5a. 3D scatter colored by cluster label...")
fig = plot_3d_scatter(
    df_pca,
    coords=pca_cols,
    color_by='cluster_label',
    color_lookup=CLUSTER_COLOR_MAP,
    show_trajectories=True,
    show_mean_per_group=True,
)
fig.write_html(FIGURES_DIR / "11_pca_by_cluster_label.html")
print(f"   Saved: {FIGURES_DIR / '11_pca_by_cluster_label.html'}")

print("\n5b. 3D scatter colored by genotype (to compare with clustering)...")
from src.analyze.trajectory_analysis.viz.styling import get_genotype_color
genotypes = df_pca['genotype'].unique()
genotype_color_lookup = {gt: get_genotype_color(gt) for gt in genotypes}

fig = plot_3d_scatter(
    df_pca,
    coords=pca_cols,
    color_by='genotype',
    color_lookup=genotype_color_lookup,
    show_trajectories=True,
    show_mean_per_group=True,
)
fig.write_html(FIGURES_DIR / "12_pca_by_genotype_for_comparison.html")
print(f"   Saved: {FIGURES_DIR / '12_pca_by_genotype_for_comparison.html'}")

print("\n5c. 2D projection (PC1 vs PC2) by cluster label...")
fig = plot_3d_scatter(
    df_pca,
    coords=['PCA_1', 'PCA_2'],
    color_by='cluster_label',
    color_lookup=CLUSTER_COLOR_MAP,
    show_trajectories=True,
    show_mean_per_group=True,
)
fig.write_html(FIGURES_DIR / "13_pca_2d_by_cluster.html")
print(f"   Saved: {FIGURES_DIR / '13_pca_2d_by_cluster.html'}")

# ============================================================================
# Step 6: Filter to core members only
# ============================================================================
print("\n" + "="*80)
print("STEP 6: VISUALIZE CORE CLUSTER MEMBERS ONLY")
print("="*80)

df_pca_core = df_pca[df_pca['membership_class'] == 'core'].copy()
print(f"\nCore members: {df_pca_core['embryo_id'].nunique()} embryos")

fig = plot_3d_scatter(
    df_pca_core,
    coords=pca_cols,
    color_by='cluster_label',
    color_lookup=CLUSTER_COLOR_MAP,
    show_trajectories=True,
    show_mean_per_group=True,
)
fig.write_html(FIGURES_DIR / "14_pca_core_members_only.html")
print(f"   Saved: {FIGURES_DIR / '14_pca_core_members_only.html'}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("CLUSTER LABELING SUMMARY")
print("="*80)

for label in CLUSTER_LABEL_MAP.values():
    cluster_data = df_pca[df_pca['cluster_label'] == label].drop_duplicates('embryo_id')
    print(f"\n{label}:")
    print(f"  Total embryos: {len(cluster_data)}")
    print(f"  Genotype breakdown:")
    for gt in cluster_data['genotype'].value_counts().items():
        print(f"    {gt[0]}: {gt[1]}")

print("\n" + "="*80)
print("NOTE: Manual inspection shows sub-structure within 'Homozygous B9D2'")
print("  - Some embryos show distinct trajectory patterns")
print("  - Consider re-clustering this group separately for finer phenotyping")
print("="*80)

print("\n✓ Tutorial 04 complete!")
print(f"  Figures saved to: {FIGURES_DIR}")
print(f"  Results saved to: {RESULTS_DIR}")
