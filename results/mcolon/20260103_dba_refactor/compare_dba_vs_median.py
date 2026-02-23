#!/usr/bin/env python
"""
Visual comparison of DBA vs binned median trend lines.

This script generates side-by-side plots to validate that DBA
produces sensible consensus trajectories.

Output:
- trend_binned_median.html / .png
- trend_dba.html / .png
"""
from pathlib import Path
import sys
sys.path.insert(0, '/net/trapnell/vol1/home/mdcolon/proj/morphseq')

import pandas as pd
from src.analyze.trajectory_analysis.facetted_plotting import plot_multimetric_trajectories

# Output directory
output_dir = Path(__file__).parent
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading data...")
# TODO: Update this path to your actual data file
# df = pd.read_parquet('/path/to/your/data.parquet')

# For testing, you can use a subset:
# df = df[df['cluster'].isin([0, 1, 2])]

print(f"Generating plots in {output_dir}")

# Binned median (current default)
print("  Generating binned median plot...")
fig_median = plot_multimetric_trajectories(
    df,
    metrics=['baseline_deviation_normalized'],
    col_by='cluster',
    trend_statistic='median',
    title='Trend: Binned Median',
    output_path=output_dir / 'trend_binned_median.html',
    backend='both',
)

# DBA trend line
print("  Generating DBA plot...")
fig_dba = plot_multimetric_trajectories(
    df,
    metrics=['baseline_deviation_normalized'],
    col_by='cluster',
    trend_statistic='dba',  # NEW!
    title='Trend: DBA (DTW Barycenter Averaging)',
    output_path=output_dir / 'trend_dba.html',
    backend='both',
)

print()
print(f"Plots saved to {output_dir}")
print("  - trend_binned_median.html / .png")
print("  - trend_dba.html / .png")
print()
print("Compare the trend lines visually to verify DBA is working correctly.")
