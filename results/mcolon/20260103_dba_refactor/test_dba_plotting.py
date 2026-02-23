#!/usr/bin/env python
"""
Test script to reload modules and create DBA trend line plots.

This script:
1. Reloads all updated modules to pick up changes
2. Creates a DBA trends subfolder
3. Generates comparison plots with DBA, median, and mean trend lines
"""

import sys
sys.path.insert(0, '/net/trapnell/vol1/home/mdcolon/proj/morphseq')

# Step 1: Thorough module reload
print("=" * 80)
print("STEP 1: Reloading updated modules")
print("=" * 80)

import importlib

# Reload DTW distance module (updated with return_path)
print("  - Reloading dtw_distance...")
from src.analyze.trajectory_analysis import dtw_distance
importlib.reload(dtw_distance)

# Reload DBA module
print("  - Reloading dba...")
from src.analyze.trajectory_analysis import dba
importlib.reload(dba)

# Reload facetted_plotting (updated with DBA support)
print("  - Reloading facetted_plotting...")
from src.analyze.trajectory_analysis import facetted_plotting
importlib.reload(facetted_plotting)

# Import the plotting function
from src.analyze.trajectory_analysis.facetted_plotting import plot_multimetric_trajectories

print("✓ All modules reloaded successfully!\n")

# Step 2: Verify you have the required variables
print("=" * 80)
print("STEP 2: Verifying required variables")
print("=" * 80)

try:
    # Check if df exists in the calling scope
    df
    print("✓ df found")
except NameError:
    print("✗ ERROR: 'df' not found. Please define your dataframe first.")
    sys.exit(1)

try:
    # Check if plot_dir exists
    plot_dir
    print(f"✓ plot_dir found: {plot_dir}")
except NameError:
    print("✗ ERROR: 'plot_dir' not found. Please define plot_dir first.")
    sys.exit(1)

# Step 3: Create DBA subfolder
print("\n" + "=" * 80)
print("STEP 3: Creating output directory")
print("=" * 80)

from pathlib import Path
dba_plot_dir = Path(plot_dir) / 'dba_trends'
dba_plot_dir.mkdir(parents=True, exist_ok=True)
print(f"✓ Created: {dba_plot_dir}\n")

# Step 4: Set up data column
print("=" * 80)
print("STEP 4: Preparing data")
print("=" * 80)

df["all_the_data_colored_by_cluster"] = "All the data colored by cluster"
n_embryos = len(df[~df["cluster"].isna()])
print(f"✓ Data prepared: {n_embryos} embryos with cluster assignments\n")

# Step 5: Generate plots
print("=" * 80)
print("STEP 5: Generating plots")
print("=" * 80)

# Plot 1: DBA trend only (no individual trajectories, no error bands)
print("\n[1/4] Generating DBA trend only...")
fig_dba = plot_multimetric_trajectories(
    df[~df["cluster"].isna()],
    metrics=['baseline_deviation_normalized', 'total_length_um'],
    col_by='all_the_data_colored_by_cluster',
    color_by_grouping='cluster',
    x_col='predicted_stage_hpf',
    metric_labels={
        'baseline_deviation_normalized': 'Curvature (normalized)',
        'total_length_um': 'Body Length (μm)',
    },
    title='Trajectories by Cluster (k=6) - DBA Trend Lines',
    x_label='Time (hpf)',
    backend='both',
    bin_width=2.0,
    show_individual=False,
    show_error_band=False,
    trend_statistic='dba',  # Use DBA!
    output_path=dba_plot_dir / 'cluster_trajectories_dba_trend_only.html'
)
print("    ✓ Saved: cluster_trajectories_dba_trend_only.html / .png")

# Plot 2: DBA trend with individual trajectories
print("\n[2/4] Generating DBA trend with individual trajectories...")
fig_dba_with_individual = plot_multimetric_trajectories(
    df[~df["cluster"].isna()],
    metrics=['baseline_deviation_normalized', 'total_length_um'],
    col_by='all_the_data_colored_by_cluster',
    color_by_grouping='cluster',
    x_col='predicted_stage_hpf',
    metric_labels={
        'baseline_deviation_normalized': 'Curvature (normalized)',
        'total_length_um': 'Body Length (μm)',
    },
    title='Trajectories by Cluster (k=6) - DBA Trend with Individual Trajectories',
    x_label='Time (hpf)',
    backend='both',
    bin_width=2.0,
    show_individual=True,   # Show individual trajectories
    show_error_band=False,
    trend_statistic='dba',
    output_path=dba_plot_dir / 'cluster_trajectories_dba_trend_with_individual.html'
)
print("    ✓ Saved: cluster_trajectories_dba_trend_with_individual.html / .png")

# Plot 3: Median trend with IQR error bands (for comparison)
print("\n[3/4] Generating median trend with IQR error bands...")
fig_median = plot_multimetric_trajectories(
    df[~df["cluster"].isna()],
    metrics=['baseline_deviation_normalized', 'total_length_um'],
    col_by='all_the_data_colored_by_cluster',
    color_by_grouping='cluster',
    x_col='predicted_stage_hpf',
    metric_labels={
        'baseline_deviation_normalized': 'Curvature (normalized)',
        'total_length_um': 'Body Length (μm)',
    },
    title='Trajectories by Cluster (k=6) - Median Trend with IQR',
    x_label='Time (hpf)',
    backend='both',
    bin_width=2.0,
    show_individual=False,
    show_error_band=True,
    trend_statistic='median',
    error_type='iqr',
    error_band_alpha=0.2,
    output_path=dba_plot_dir / 'cluster_trajectories_median_trend_iqr.html'
)
print("    ✓ Saved: cluster_trajectories_median_trend_iqr.html / .png")

# Plot 4: Mean trend with SD error bands (for comparison)
print("\n[4/4] Generating mean trend with SD error bands...")
fig_mean_sd = plot_multimetric_trajectories(
    df[~df["cluster"].isna()],
    metrics=['baseline_deviation_normalized', 'total_length_um'],
    col_by='all_the_data_colored_by_cluster',
    color_by_grouping='cluster',
    x_col='predicted_stage_hpf',
    metric_labels={
        'baseline_deviation_normalized': 'Curvature (normalized)',
        'total_length_um': 'Body Length (μm)',
    },
    title='Trajectories by Cluster (k=6) - Mean Trend with SD',
    x_label='Time (hpf)',
    backend='both',
    bin_width=2.0,
    show_individual=False,
    show_error_band=True,
    trend_statistic='mean',
    error_type='sd',
    error_band_alpha=0.2,
    output_path=dba_plot_dir / 'cluster_trajectories_mean_trend_sd.html'
)
print("    ✓ Saved: cluster_trajectories_mean_trend_sd.html / .png")

# Summary
print("\n" + "=" * 80)
print("COMPLETE!")
print("=" * 80)
print(f"\nAll plots saved to: {dba_plot_dir}")
print("\nFiles created:")
print("  1. cluster_trajectories_dba_trend_only.html / .png")
print("  2. cluster_trajectories_dba_trend_with_individual.html / .png")
print("  3. cluster_trajectories_median_trend_iqr.html / .png")
print("  4. cluster_trajectories_mean_trend_sd.html / .png")
print("\nCompare the DBA plots (#1, #2) with traditional methods (#3, #4)")
print("to see how DBA handles temporal alignment!\n")
