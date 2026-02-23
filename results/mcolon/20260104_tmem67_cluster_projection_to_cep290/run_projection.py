"""
TMEM67 Cluster Projection onto CEP290 Reference Clusters

Main script to project TMEM67 embryo trajectories onto pre-existing CEP290
cluster assignments using DTW distance.

Usage:
    python run_projection.py
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
import sys
sys.path.insert(0, '/net/trapnell/vol1/home/mdcolon/proj/morphseq')

from src.analyze.trajectory_analysis.data_loading import load_experiment_dataframe
from src.analyze.trajectory_analysis.dtw_distance import prepare_multivariate_array
from projection_utils import (
    compute_cross_dtw_distance_matrix,
    assign_clusters_nearest_neighbor,
    assign_clusters_knn_posterior,
    compare_cluster_frequencies
)

# Paths
CEP290_DIR = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251229_cep290_phenotype_extraction/final_data")
OUTPUT_DIR = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260104_tmem67_cluster_projection_to_cep290/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Analysis parameters
# Use curvature-only for DTW projection (avoid mixing in body size/length).
METRICS = ['baseline_deviation_normalized']
TMEM67_EXPERIMENTS = ['20250711', '20251205']
SAKOE_CHIBA_RADIUS = 3
K_NN = 5

print("="*80)
print("TMEM67 Cluster Projection Analysis")
print("="*80)

# =============================================================================
# Step 1: Load CEP290 Reference Data (PRE-COMPUTED CLUSTERS)
# =============================================================================
print("\n[1/7] Loading CEP290 reference data...")

# Load CEP290 trajectory data
df_cep290 = pd.read_csv(CEP290_DIR / "embryo_data_with_labels.csv", low_memory=False)
print(f"  Loaded CEP290 data: {len(df_cep290)} rows, {df_cep290['embryo_id'].nunique()} embryos")

# Load PRE-COMPUTED cluster labels (FIXED - do not modify)
df_cep290_labels = pd.read_csv(CEP290_DIR / "embryo_cluster_labels.csv", low_memory=False)
print(f"  Loaded CEP290 labels: {len(df_cep290_labels)} rows")

# Use ALL CEP290 data (not just spawn)
df_cep290_all = df_cep290.copy()
labels_all = df_cep290_labels.drop_duplicates(subset='embryo_id')

print(f"  CEP290 all embryos (before filtering): {len(labels_all)} embryos")

# Filter out embryos with no valid cluster assignment
# This prevents TMEM67 embryos from being assigned to unclassified CEP290 embryos
labels_all = labels_all[labels_all['cluster_categories'].notna()].copy()
print(f"  CEP290 all embryos (after filtering NaN clusters): {len(labels_all)} embryos with valid cluster labels")

# NOTE: Future improvement - implement bootstrap-based cluster assignment
# Use bootstrap utilities from src/analyze/trajectory_analysis/ to:
# 1. Compute DTW distances with uncertainty via bootstrap resampling
# 2. Assign clusters with confidence intervals
# 3. Handle borderline cases more robustly than nearest-neighbor alone

# Build embryo_id -> cluster mappings
cep290_cluster_map = dict(zip(labels_all['embryo_id'], labels_all['clusters']))
cep290_category_map = dict(zip(labels_all['embryo_id'], labels_all['cluster_categories']))

# Create cluster number -> category mapping for later use
cluster_to_category = dict(zip(labels_all['clusters'], labels_all['cluster_categories']))

print(f"  Cluster categories: {sorted(labels_all['cluster_categories'].dropna().unique())}")
print(f"  Cluster distribution:")
for cat, count in labels_all['cluster_categories'].value_counts(dropna=False).items():
    print(f"    {cat}: {count}")

# =============================================================================
# Step 2: Load TMEM67 Data
# =============================================================================
print("\n[2/7] Loading TMEM67 data...")

dfs = []
for exp_id in TMEM67_EXPERIMENTS:
    print(f"  Loading {exp_id}...")
    df_exp = load_experiment_dataframe(exp_id, format_version='df03')
    df_exp['experiment_id'] = exp_id
    dfs.append(df_exp)
    print(f"    {len(df_exp)} rows, {df_exp['embryo_id'].nunique()} embryos")

df_tmem67 = pd.concat(dfs, ignore_index=True)
print(f"  Combined TMEM67 data: {len(df_tmem67)} rows, {df_tmem67['embryo_id'].nunique()} embryos")

# Filter to TMEM67 genotypes only
df_tmem67 = df_tmem67[df_tmem67["genotype"].str.contains("tmem67", case=False, na=False)].copy()
print(f"  After genotype filter: {df_tmem67['embryo_id'].nunique()} embryos")

# Homozygous-only summary (likely penetrant subset)
df_tmem67_hom = df_tmem67[df_tmem67["genotype"].str.contains("tmem67_homozygous", case=False, na=False)].copy()
print(f"  TMEM67 homozygous: {df_tmem67_hom['embryo_id'].nunique()} embryos")

# Mark spawn (missing pair = spawn)
df_tmem67.loc[df_tmem67["pair"].isna(), "pair"] = "tmem67_spawn"
df_tmem67_hom.loc[df_tmem67_hom["pair"].isna(), "pair"] = "tmem67_spawn"

print(f"  TMEM67 pair distribution:")
for pair, count in df_tmem67['pair'].value_counts().items():
    embryo_count = df_tmem67[df_tmem67['pair'] == pair]['embryo_id'].nunique()
    print(f"    {pair}: {embryo_count} embryos")

if len(df_tmem67_hom) > 0:
    print(f"  TMEM67 homozygous pair distribution:")
    for pair, count in df_tmem67_hom['pair'].value_counts().items():
        embryo_count = df_tmem67_hom[df_tmem67_hom['pair'] == pair]['embryo_id'].nunique()
        print(f"    {pair}: {embryo_count} embryos")

# =============================================================================
# Step 3: Prepare Multivariate Arrays with Shared Time Grid
# =============================================================================
print("\n[3/7] Preparing multivariate arrays...")

# CEP290 reference (get time_grid from here)
# Filter to only include embryos with valid cluster assignments
print("  Preparing CEP290 array (all data)...")
valid_cep290_ids = labels_all['embryo_id'].unique()
df_cep290_valid = df_cep290_all[df_cep290_all['embryo_id'].isin(valid_cep290_ids)].copy()
print(f"  CEP290 embryos (filtered to valid clusters): {df_cep290_valid['embryo_id'].nunique()}")

X_cep290, cep290_ids, time_grid = prepare_multivariate_array(
    df_cep290_valid,
    metrics=METRICS,
    normalize=True,
    verbose=False
)
print(f"  CEP290 array shape: {X_cep290.shape}")
print(f"  Time grid: {len(time_grid)} points ({time_grid.min():.1f} - {time_grid.max():.1f} hpf)")

# TMEM67 (use SAME time_grid - critical for valid comparisons!)
print("\n  Preparing TMEM67 array (using same time grid)...")

# First check TMEM67 time range
tmem67_time_range = (df_tmem67['predicted_stage_hpf'].min(), df_tmem67['predicted_stage_hpf'].max())
print(f"  TMEM67 time range: {tmem67_time_range[0]:.1f} - {tmem67_time_range[1]:.1f} hpf")
print(f"  CEP290 time range: {time_grid.min():.1f} - {time_grid.max():.1f} hpf")

# Use the shared grid - this will handle interpolation/extrapolation
X_tmem67, tmem67_ids, _ = prepare_multivariate_array(
    df_tmem67,
    metrics=METRICS,
    time_grid=time_grid,  # CRITICAL: shared grid
    normalize=True,
    verbose=False
)
print(f"  TMEM67 array shape: {X_tmem67.shape}")

# Verify shapes match
if X_tmem67.shape[1] != X_cep290.shape[1]:
    raise ValueError(
        f"Shape mismatch after interpolation: TMEM67 has {X_tmem67.shape[1]} timepoints, "
        f"CEP290 has {X_cep290.shape[1]}. This should not happen with shared time_grid!"
    )

# =============================================================================
# Step 4: Compute Cross-Dataset DTW Distances
# =============================================================================
print("\n[4/7] Computing cross-dataset DTW distances...")
print(f"  This will compute {X_tmem67.shape[0]} x {X_cep290.shape[0]} = {X_tmem67.shape[0] * X_cep290.shape[0]} pairwise distances")

D_cross = compute_cross_dtw_distance_matrix(
    X_tmem67,
    X_cep290,
    sakoe_chiba_radius=SAKOE_CHIBA_RADIUS,
    n_jobs=-1,
    verbose=True
)

# Save distance matrix
np.save(OUTPUT_DIR / "cross_dtw_distance_matrix.npy", D_cross)
print(f"\n  Saved distance matrix to: {OUTPUT_DIR / 'cross_dtw_distance_matrix.npy'}")

# =============================================================================
# Step 5: Nearest Neighbor Projection
# =============================================================================
print("\n[5/7] Assigning clusters via nearest neighbor...")

df_nn = assign_clusters_nearest_neighbor(
    D_cross,
    tmem67_ids,
    cep290_ids,
    cep290_cluster_map,
    cep290_category_map
)

print(f"  Assigned {len(df_nn)} TMEM67 embryos to clusters")
print(f"\n  TMEM67 cluster distribution (nearest neighbor):")
for cat, count in df_nn['cluster_category'].value_counts().items():
    pct = count / len(df_nn) * 100
    print(f"    {cat}: {count} ({pct:.1f}%)")

# Add pair information to assignments
pair_map = dict(zip(df_tmem67['embryo_id'], df_tmem67['pair']))
df_nn['pair'] = df_nn['embryo_id'].map(pair_map)
genotype_map = dict(zip(df_tmem67['embryo_id'], df_tmem67['genotype']))
df_nn['genotype'] = df_nn['embryo_id'].map(genotype_map)

# Save nearest neighbor results
df_nn.to_csv(OUTPUT_DIR / "tmem67_nn_projection.csv", index=False)
print(f"\n  Saved NN results to: {OUTPUT_DIR / 'tmem67_nn_projection.csv'}")

df_nn_hom = df_nn[df_nn["genotype"].str.contains("tmem67_homozygous", case=False, na=False)]
if len(df_nn_hom) > 0:
    print(f"\n  TMEM67 homozygous cluster distribution (nearest neighbor):")
    for cat, count in df_nn_hom['cluster_category'].value_counts().items():
        pct = count / len(df_nn_hom) * 100
        print(f"    {cat}: {count} ({pct:.1f}%)")

# =============================================================================
# Step 6: K-NN Posterior Projection (for comparison)
# =============================================================================
print(f"\n[6/7] Assigning clusters via K-NN (k={K_NN})...")

df_knn = assign_clusters_knn_posterior(
    D_cross,
    tmem67_ids,
    cep290_ids,
    cep290_cluster_map,
    k=K_NN,
    target_category_map=cep290_category_map
)

print(f"  Assigned {len(df_knn)} TMEM67 embryos with posteriors")
print(f"\n  TMEM67 cluster distribution (K-NN):")
for cat, count in df_knn['cluster_category'].value_counts().items():
    pct = count / len(df_knn) * 100
    print(f"    {cat}: {count} ({pct:.1f}%)")

# Add pair information
df_knn['pair'] = df_knn['embryo_id'].map(pair_map)
df_knn['genotype'] = df_knn['embryo_id'].map(genotype_map)

# Save K-NN results
df_knn.to_csv(OUTPUT_DIR / "tmem67_knn_projection.csv", index=False)
print(f"\n  Saved K-NN results to: {OUTPUT_DIR / 'tmem67_knn_projection.csv'}")

df_knn_hom = df_knn[df_knn["genotype"].str.contains("tmem67_homozygous", case=False, na=False)]
if len(df_knn_hom) > 0:
    print(f"\n  TMEM67 homozygous cluster distribution (K-NN):")
    for cat, count in df_knn_hom['cluster_category'].value_counts(dropna=False).items():
        pct = count / len(df_knn_hom) * 100
        print(f"    {cat}: {count} ({pct:.1f}%)")

# =============================================================================
# Step 7: Compare Cluster Frequencies
# =============================================================================
print("\n[7/7] Comparing cluster frequencies...")

# CEP290 all vs TMEM67 spawn (for comparison)
df_tmem67_spawn = df_nn[df_nn['pair'] == 'tmem67_spawn']

print(f"\nComparing:")
print(f"  CEP290 all: {len(labels_all)} embryos")
print(f"  TMEM67 spawn: {len(df_tmem67_spawn)} embryos")

# Normalize column naming for frequency comparison
labels_all_comp = labels_all.copy()
if 'cluster_category' not in labels_all_comp.columns and 'cluster_categories' in labels_all_comp.columns:
    labels_all_comp = labels_all_comp.rename(columns={'cluster_categories': 'cluster_category'})

freq_comp, stats = compare_cluster_frequencies(
    labels_all_comp,
    df_tmem67_spawn,
    category_col='cluster_category',
    ref_label='CEP290_all',
    projected_label='TMEM67_spawn'
)

print("\n" + "="*60)
print("CLUSTER FREQUENCY COMPARISON (%)")
print("="*60)
print(freq_comp.to_string())

print("\n" + "="*60)
print("STATISTICAL TESTS")
print("="*60)
print(f"Chi-square test:")
print(f"  χ² = {stats['chi2_statistic']:.3f}")
print(f"  p-value = {stats['chi2_pvalue']:.4f}")
print(f"  df = {stats['chi2_dof']}")

if 'fisher_pvalue' in stats:
    print(f"\nFisher's exact test:")
    print(f"  Odds ratio = {stats['fisher_odds_ratio']:.3f}")
    print(f"  p-value = {stats['fisher_pvalue']:.4f}")

# Save frequency comparison
freq_comp.to_csv(OUTPUT_DIR / "cluster_frequency_comparison_cep290all_vs_tmem67spawn.csv")
print(f"\nSaved frequency comparison to: {OUTPUT_DIR / 'cluster_frequency_comparison_cep290all_vs_tmem67spawn.csv'}")

# Test hypothesis: TMEM67 has higher High_to_Low percentage
if 'High_to_Low' in freq_comp.index:
    cep290_h2l = freq_comp.loc['High_to_Low', 'CEP290_all']
    tmem67_h2l = freq_comp.loc['High_to_Low', 'TMEM67_spawn']
    diff = tmem67_h2l - cep290_h2l

    print("\n" + "="*60)
    print("HYPOTHESIS TEST: High_to_Low Frequency")
    print("="*60)
    print(f"  CEP290 spawn:  {cep290_h2l:.1f}%")
    print(f"  TMEM67 spawn:  {tmem67_h2l:.1f}%")
    print(f"  Difference:    {diff:+.1f}%")

    if diff > 0:
        print(f"  → TMEM67 has {diff:.1f}% MORE High_to_Low embryos (supports hypothesis)")
    else:
        print(f"  → TMEM67 has {abs(diff):.1f}% FEWER High_to_Low embryos (against hypothesis)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nResults saved to: {OUTPUT_DIR}")
print("\nNext steps:")
print("  1. Review distance distribution (check for outliers)")
print("  2. Create visualizations (bar charts, trajectory overlays)")
print("  3. Analyze TMEM67 pair_1 data separately if needed")
