"""
Tutorial 02: 3D Scatter + PCA Visualization

Demonstrates PCA dimensionality reduction and 3D visualization of trajectories.

Key API usage:
- fit_transform_pca() from src.analyze.utils (NEW LOCATION)
- plot_3d_scatter() with trajectory visualization options

API Reference
=============

fit_transform_pca(df, n_components=3)
--------------------------------------
Perform PCA on VAE embeddings (z_mu_b columns).

Parameters:
    df : DataFrame with z_mu_b_* columns
    n_components : int, default=3
        Number of principal components to compute

Returns:
    df_pca : DataFrame with PCA_1, PCA_2, ... columns added
    pca : sklearn PCA object (access .explained_variance_ratio_)
    scaler : sklearn StandardScaler object
    z_mu_cols : list of z_mu_b column names used

Example:
    df_pca, pca, scaler, z_mu_cols = fit_transform_pca(df, n_components=3)
    print(f"Variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")


plot_3d_scatter(df, coords, color_by, ...)
-------------------------------------------
Create interactive 3D scatter plot with Plotly.

Parameters:
    df : DataFrame
        Data containing coordinate columns
    coords : list of str
        Exactly 3 column names for x, y, z coordinates
        e.g., ['PCA_1', 'PCA_2', 'PCA_3']
    color_by : str
        Column to use for coloring points/lines (e.g., 'genotype')
    color_palette : dict, optional
        Custom color mapping {value: hex_color}
    line_by : str, default='id'
        Column identifying individual trajectories (e.g., 'embryo_id')
    x_col : str, optional
        Time column for sorting trajectory points (required if show_lines=True)
    show_lines : bool, default=False
        Connect points per line_by group to show trajectories
    show_mean : bool, default=False
        Show mean trajectory per color_by group
    min_points_per_line : int, default=20
        Minimum points required per line_by group to be included

Returns:
    go.Figure : Plotly figure object

Example:
    # Basic scatter
    fig = plot_3d_scatter(df, ['PCA_1', 'PCA_2', 'PCA_3'], color_by='genotype')

    # With individual trajectories
    fig = plot_3d_scatter(
        df, ['PCA_1', 'PCA_2', 'PCA_3'],
        color_by='genotype',
        color_palette=color_lookup,
        line_by='embryo_id',
        show_lines=True,
        x_col='time_hpf'
    )

    # With mean trajectories per group
    fig = plot_3d_scatter(..., show_mean=True)

Note: This function is Plotly-only (no matplotlib backend currently available)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

# Setup output directory
OUTPUT_DIR = Path(__file__).parent / "output"
FIGURES_DIR = OUTPUT_DIR / "figures" / "02"  # Tutorial 02 specific directory
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load data from both CEP290 experiments (direct CSV load)
print("Loading experiment data...")
meta_dir = project_root / 'morphseq_playground' / 'metadata' / 'build04_output'
df1 = pd.read_csv(meta_dir / 'qc_staged_20260122.csv')
df2 = pd.read_csv(meta_dir / 'qc_staged_20260124.csv')
df = pd.concat([df1, df2], ignore_index=True)

# Filter to valid embryos
df = df[df['use_embryo_flag']].copy()
print(f"Loaded {len(df['embryo_id'].unique())} embryos across {len(df)} timepoints")

# Define color lookup (CORRECTED: get_color_for_genotype)
from src.analyze.trajectory_analysis.viz.styling import get_color_for_genotype
genotypes = df['genotype'].unique()
color_lookup = {gt: get_color_for_genotype(gt) for gt in genotypes}

# ============================================================================
# PCA on VAE embeddings (NEW: from src.analyze.utils)
# ============================================================================
print("\nPerforming PCA on VAE embeddings...")
from src.analyze.utils import fit_transform_pca

df_pca, pca, scaler, z_mu_cols = fit_transform_pca(df, n_components=3)
pca_cols = ['PCA_1', 'PCA_2', 'PCA_3']

# Report variance explained
variance_explained = pca.explained_variance_ratio_
print(f"PCA variance explained:")
for i, var in enumerate(variance_explained, 1):
    print(f"  PC{i}: {var*100:.1f}%")
print(f"  Total (3 components): {variance_explained.sum()*100:.1f}%")

# Save PCA results
pca_summary = pd.DataFrame({
    'component': [f'PC{i+1}' for i in range(len(variance_explained))],
    'variance_explained': variance_explained,
    'cumulative_variance': variance_explained.cumsum()
})
pca_summary.to_csv(RESULTS_DIR / "pca_variance_explained.csv", index=False)
print(f"\nSaved PCA summary to: {RESULTS_DIR / 'pca_variance_explained.csv'}")

# Import 3D plotting function
from src.analyze.viz.plotting import plot_3d_scatter

# ============================================================================
# Example 1: Basic 3D scatter (points only)
# ============================================================================
print("\n1. Creating basic 3D scatter plot...")
fig = plot_3d_scatter(
    df_pca,
    coords=pca_cols,
    color_by='genotype',
    color_palette=color_lookup,
    line_by='embryo_id',
    show_lines=False,
    show_mean=False,
)
fig.write_html(FIGURES_DIR / "05_3d_scatter_points.html")
print(f"   Saved: {FIGURES_DIR / '05_3d_scatter_points.html'}")

# ============================================================================
# Example 2: 3D scatter with individual trajectories
# ============================================================================
print("\n2. Creating 3D scatter with individual trajectories...")
fig = plot_3d_scatter(
    df_pca,
    coords=pca_cols,
    color_by='genotype',
    color_palette=color_lookup,
    line_by='embryo_id',
    show_lines=True,  # Show individual trajectory lines
    x_col='time_hpf',  # Required for ordering trajectory points
    show_mean=False,
)
fig.write_html(FIGURES_DIR / "06_3d_scatter_trajectories.html")
print(f"   Saved: {FIGURES_DIR / '06_3d_scatter_trajectories.html'}")

# ============================================================================
# Example 3: 3D scatter with mean trajectory per group
# ============================================================================
print("\n3. Creating 3D scatter with mean trajectories...")
fig = plot_3d_scatter(
    df_pca,
    coords=pca_cols,
    color_by='genotype',
    color_palette=color_lookup,
    line_by='embryo_id',
    show_lines=True,
    x_col='time_hpf',
    show_mean=True,  # Show mean trajectory per genotype
)
fig.write_html(FIGURES_DIR / "07_3d_scatter_with_means.html")
print(f"   Saved: {FIGURES_DIR / '07_3d_scatter_with_means.html'}")

# ============================================================================
# Example 4: 2D projection (PCA_1 vs PCA_2)
# Note: plot_3d_scatter requires exactly 3 coordinates, so we duplicate PCA_2
# ============================================================================
print("\n4. Creating 2D projection (PC1 vs PC2)...")
fig = plot_3d_scatter(
    df_pca,
    coords=['PCA_1', 'PCA_2', 'PCA_2'],  # Third coord is dummy for 3D API
    color_by='genotype',
    color_palette=color_lookup,
    line_by='embryo_id',
    show_lines=True,
    x_col='time_hpf',
    show_mean=True,
)
fig.write_html(FIGURES_DIR / "08_2d_projection_pc1_pc2.html")
print(f"   Saved: {FIGURES_DIR / '08_2d_projection_pc1_pc2.html'}")

# ============================================================================
# Example 5: Alternative 2D projection (PCA_2 vs PCA_3)
# ============================================================================
print("\n5. Creating 2D projection (PC2 vs PC3)...")
fig = plot_3d_scatter(
    df_pca,
    coords=['PCA_2', 'PCA_3', 'PCA_3'],  # Third coord is dummy
    color_by='genotype',
    color_palette=color_lookup,
    line_by='embryo_id',
    show_lines=True,
    x_col='time_hpf',
    show_mean=True,
)
fig.write_html(FIGURES_DIR / "09_2d_projection_pc2_pc3.html")
print(f"   Saved: {FIGURES_DIR / '09_2d_projection_pc2_pc3.html'}")

print("\n✓ Tutorial 02 complete!")
print(f"  Generated 5 HTML output files")
print(f"  Figures saved to: {FIGURES_DIR}")
print(f"  Results saved to: {RESULTS_DIR}")
print(f"  PCA variance: PC1={variance_explained[0]*100:.1f}%, PC2={variance_explained[1]*100:.1f}%, PC3={variance_explained[2]*100:.1f}% (Total={variance_explained.sum()*100:.1f}%)")
