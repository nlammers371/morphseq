"""
Tutorial 07: Spline Fitting Through Found Clusters

Demonstrates spline fitting for cluster-specific trajectory modeling.

Key API usage:
- spline_fit_wrapper() for fitting splines to trajectory clusters
- plot_3d_with_spline() for visualization

NOTE: This tutorial uses spline fitting which may be deprecated.
If spline_fit_wrapper is not available, skip this tutorial or use
alternative trajectory smoothing methods.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Setup directories
OUTPUT_DIR = Path(__file__).parent / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"

# Load data
from src.analyze.trajectory_analysis.data_loading import load_experiment_dataframe

print("Loading experiment data...")
df1 = load_experiment_dataframe('20251121', format_version='df03')
df2 = load_experiment_dataframe('20251125', format_version='df03')
df = pd.concat([df1, df2], ignore_index=True)
df = df[df['use_embryo_flag']].copy()

# Load PCA results
from src.analyze.utils import fit_transform_pca
df_pca, pca, scaler, z_mu_cols = fit_transform_pca(df, n_components=3)
pca_cols = ['PCA_1', 'PCA_2', 'PCA_3']

# Load cluster labels
membership_df = pd.read_csv(RESULTS_DIR / "cluster_membership_labeled.csv")
embryo_labels = membership_df[['embryo_id', 'cluster_label']].copy()
df_pca = df_pca.merge(embryo_labels, on='embryo_id', how='left')
df_pca = df_pca.dropna(subset=['cluster_label'])

print(f"Loaded {len(df_pca['embryo_id'].unique())} embryos with cluster labels")

# ============================================================================
# Check if spline fitting is available
# ============================================================================
print("\n" + "="*80)
print("CHECKING SPLINE FITTING AVAILABILITY")
print("="*80)

try:
    from src.analyze.spline_fitting import spline_fit_wrapper
    SPLINE_AVAILABLE = True
    print("✓ spline_fit_wrapper available")
except ImportError:
    SPLINE_AVAILABLE = False
    print("✗ spline_fit_wrapper not available")
    print("\nAlternative: Use plot_3d_scatter() with show_mean_per_group=True")
    print("             to visualize mean trajectories per cluster")

if SPLINE_AVAILABLE:
    # ============================================================================
    # Fit splines per cluster
    # ============================================================================
    print("\n" + "="*80)
    print("FITTING SPLINES PER CLUSTER")
    print("="*80)

    spline_results = spline_fit_wrapper(
        df_pca,
        coords=pca_cols,
        group_by='cluster_label',
        time_col='predicted_stage_hpf',
        n_knots=10,  # Number of knots for spline
        smoothing_factor=0.5,  # Smoothing parameter (0 = interpolation, higher = smoother)
    )

    print(f"\nFitted splines for {len(spline_results)} cluster labels")

    # Save spline parameters
    import pickle
    with open(RESULTS_DIR / "spline_results_per_cluster.pkl", 'wb') as f:
        pickle.dump(spline_results, f)
    print(f"Saved to: {RESULTS_DIR / 'spline_results_per_cluster.pkl'}")

    # ============================================================================
    # Visualize splines in 3D
    # ============================================================================
    print("\n" + "="*80)
    print("VISUALIZING SPLINES IN 3D")
    print("="*80)

    from src.analyze.spline_fitting.viz import plot_3d_with_spline

    # Color lookup
    CLUSTER_COLOR_MAP = {
        'Short Body Axis': '#d62728',
        'Homozygous B9D2': '#ff7f0e',
        'Not Penetrant': '#2ca02c',
    }

    fig = plot_3d_with_spline(
        df_pca,
        coords=pca_cols,
        spline_results=spline_results,
        color_by='cluster_label',
        color_lookup=CLUSTER_COLOR_MAP,
        show_data_points=True,
        show_trajectories=False,  # Hide individual trajectories for clarity
    )
    fig.write_html(FIGURES_DIR / "24_splines_per_cluster.html")
    print(f"   Saved: {FIGURES_DIR / '24_splines_per_cluster.html'}")

    # ============================================================================
    # 2D projections with splines
    # ============================================================================
    print("\n2D projections with splines...")

    fig = plot_3d_with_spline(
        df_pca,
        coords=['PCA_1', 'PCA_2'],
        spline_results=spline_results,
        color_by='cluster_label',
        color_lookup=CLUSTER_COLOR_MAP,
        show_data_points=True,
    )
    fig.write_html(FIGURES_DIR / "25_splines_2d_pc1_pc2.html")
    print(f"   Saved: {FIGURES_DIR / '25_splines_2d_pc1_pc2.html'}")

    print("\n✓ Tutorial 07 complete (spline fitting)!")

else:
    # ============================================================================
    # Alternative: Mean trajectories without splines
    # ============================================================================
    print("\n" + "="*80)
    print("ALTERNATIVE: VISUALIZING MEAN TRAJECTORIES")
    print("="*80)

    from src.analyze.viz.plotting import plot_3d_scatter

    CLUSTER_COLOR_MAP = {
        'Short Body Axis': '#d62728',
        'Homozygous B9D2': '#ff7f0e',
        'Not Penetrant': '#2ca02c',
    }

    print("\nCreating 3D scatter with mean trajectories per cluster...")
    fig = plot_3d_scatter(
        df_pca,
        coords=pca_cols,
        color_by='cluster_label',
        color_lookup=CLUSTER_COLOR_MAP,
        show_trajectories=True,
        show_mean_per_group=True,  # Show mean trajectory per cluster
    )
    fig.write_html(FIGURES_DIR / "24_mean_trajectories_per_cluster.html")
    print(f"   Saved: {FIGURES_DIR / '24_mean_trajectories_per_cluster.html'}")

    print("\n2D projection with mean trajectories...")
    fig = plot_3d_scatter(
        df_pca,
        coords=['PCA_1', 'PCA_2'],
        color_by='cluster_label',
        color_lookup=CLUSTER_COLOR_MAP,
        show_trajectories=True,
        show_mean_per_group=True,
    )
    fig.write_html(FIGURES_DIR / "25_mean_trajectories_2d.html")
    print(f"   Saved: {FIGURES_DIR / '25_mean_trajectories_2d.html'}")

    print("\n✓ Tutorial 07 complete (mean trajectories - spline fitting not available)!")

print(f"\n  Figures saved to: {FIGURES_DIR}")
if SPLINE_AVAILABLE:
    print(f"  Results saved to: {RESULTS_DIR}")
