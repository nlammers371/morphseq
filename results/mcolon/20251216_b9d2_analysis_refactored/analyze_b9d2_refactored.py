#!/usr/bin/env python3
"""
B9D2 Pair Analysis - Refactored Version Using New Plotting Framework

This is a streamlined version of the b9d2 pair analysis using the new
refactored plotting utilities. Reduces ~1255 lines to ~100 lines while
maintaining all functionality.

FEATURES:
- Uses new plot_pairs_overview(), plot_genotypes_by_pair(), plot_single_genotype_across_pairs()
- Automatic suffix-based genotype coloring (works for any gene)
- Dual-backend support (PNG and/or HTML)
- Gaussian smoothing enabled by default
- Descriptive filenames with experiment ID and metric

USAGE:
    python analyze_b9d2_refactored.py

OUTPUT:
    results/mcolon/20251020_b9d2_analysis_refactored/
      output_{exp_id}_{metric}/
        figures/
          {exp_id}_{metric}_all_pairs_overview.png/.html
          {exp_id}_{metric}_genotypes_by_pair.png/.html
          {exp_id}_{metric}_homozygous_across_pairs.png/.html
        tables/
          pair_summary_statistics.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, '/net/trapnell/vol1/home/mdcolon/proj/morphseq')

from src.analyze.trajectory_analysis.data_loading import load_experiment_dataframe
from src.analyze.trajectory_analysis import (
    plot_pairs_overview,
    plot_genotypes_by_pair,
    plot_single_genotype_across_pairs,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Experiments and metrics to analyze
EXPERIMENT_IDS = ['20251121']
METRICS = ['baseline_deviation_normalized', 'total_length_um', 'surface_area_um']

# Column names
TIME_COL = 'predicted_stage_hpf'
EMBRYO_ID_COL = 'embryo_id'
PAIR_COL = 'pair'
GENOTYPE_COL = 'genotype'

# Base output directory
BASE_OUTPUT_DIR = Path('/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251216_b9d2_analysis_refactored')

# Metric display names
METRIC_DISPLAY_NAMES = {
    'baseline_deviation_normalized': 'Normalized Baseline Deviation',
    'total_length_um': 'Total Length (µm)',
    'surface_area_um': 'Surface Area (µm²)',
}

# Output format control
GENERATE_PNG = True
GENERATE_PLOTLY = True


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_metric_abbreviation(metric_name):
    """Get abbreviated metric name for filenames."""
    abbrev_map = {
        'baseline_deviation_normalized': 'baseline_dev',
        'total_length_um': 'length',
        'surface_area_um': 'surface_area',
    }
    return abbrev_map.get(metric_name, metric_name)


def load_and_prepare_data(experiment_id, metric_name):
    """Load data and prepare for analysis."""
    print(f"Loading data for experiment {experiment_id}, metric {metric_name}...")

    df = load_experiment_dataframe(experiment_id, format_version='qc_staged')

    # Filter for valid embryos (if column exists)
    if 'use_embryo_flag' in df.columns:
        print(f"  Before use_embryo_flag filter: {len(df)} rows")
        df = df[df['use_embryo_flag'] == 1].copy()
        print(f"  After use_embryo_flag filter: {len(df)} rows")

    # First check if pair column exists or has valid data
    has_valid_pairs = PAIR_COL in df.columns and not df[PAIR_COL].isna().all()

    # Drop rows with missing values in key columns (excluding pair if it doesn't exist)
    required_cols = [EMBRYO_ID_COL, TIME_COL, metric_name, GENOTYPE_COL]
    if has_valid_pairs:
        required_cols.append(PAIR_COL)

    print(f"  Before dropna: {len(df)} rows")
    print(f"  Checking for NaNs in: {required_cols}")
    for col in required_cols:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                print(f"    {col}: {n_missing} missing values")

    df = df.dropna(subset=required_cols).copy()
    print(f"  After dropna: {len(df)} rows")

    # If no pair column or all NaN, the plotting functions will create {genotype}_unknown_pair
    if PAIR_COL in df.columns and not df[PAIR_COL].isna().all():
        print(f"  Unique pairs: {sorted(df[PAIR_COL].unique())}")
    else:
        print(f"  No pair column - will auto-create {{genotype}}_unknown_pair")

    print(f"  Genotypes: {sorted(df[GENOTYPE_COL].unique())}")

    return df


def create_summary_statistics(df, metric_name, output_dir):
    """Create summary statistics table."""
    summary_data = []

    for pair in sorted(df[PAIR_COL].unique()):
        for genotype in sorted(df[GENOTYPE_COL].unique()):
            subset = df[(df[PAIR_COL] == pair) & (df[GENOTYPE_COL] == genotype)]

            if len(subset) > 0:
                summary_data.append({
                    'pair': pair,
                    'genotype': genotype,
                    'n_embryos': subset[EMBRYO_ID_COL].nunique(),
                    'mean_value': subset[metric_name].mean(),
                    'std_value': subset[metric_name].std(),
                    'min_value': subset[metric_name].min(),
                    'max_value': subset[metric_name].max(),
                })

    summary_df = pd.DataFrame(summary_data)
    output_path = output_dir / 'pair_summary_statistics.csv'
    summary_df.to_csv(output_path, index=False)
    print(f"  Summary statistics saved to {output_path}")

    return summary_df


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_experiment_metric(experiment_id, metric_name):
    """Run complete analysis for one experiment and metric."""
    print("\n" + "=" * 80)
    print(f"Analyzing: {experiment_id} - {metric_name}")
    print("=" * 80)

    # Load data
    df = load_and_prepare_data(experiment_id, metric_name)

    # Create output directories
    metric_abbrev = get_metric_abbreviation(metric_name)
    output_dir = BASE_OUTPUT_DIR / f'output_{experiment_id}_{metric_abbrev}'
    figures_dir = output_dir / 'figures'
    tables_dir = output_dir / 'tables'
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Determine backend
    if GENERATE_PNG and GENERATE_PLOTLY:
        backend = 'both'
    elif GENERATE_PLOTLY:
        backend = 'plotly'
    else:
        backend = 'matplotlib'

    metric_label = METRIC_DISPLAY_NAMES.get(metric_name, metric_name)

    # 1. All pairs × genotypes overview (NxM grid)
    print("\n1. Creating all pairs overview (NxM grid)...")
    output_path = figures_dir / f'{experiment_id}_{metric_abbrev}_all_pairs_overview'
    plot_pairs_overview(
        df,
        x_col=TIME_COL,
        y_col=metric_name,
        line_by=EMBRYO_ID_COL,
        pair_col=PAIR_COL,
        genotype_col=GENOTYPE_COL,
        backend=backend,
        output_path=output_path,
        title=f'{experiment_id}: All Pairs × Genotypes - {metric_label}',
        y_label=metric_label,
    )
    print(f"  Saved to {output_path}.*")

    # 2. Genotypes by pair (1xN with overlay)
    print("\n2. Creating genotypes by pair plot...")
    output_path = figures_dir / f'{experiment_id}_{metric_abbrev}_genotypes_by_pair'
    plot_genotypes_by_pair(
        df,
        x_col=TIME_COL,
        y_col=metric_name,
        line_by=EMBRYO_ID_COL,
        pair_col=PAIR_COL,
        genotype_col=GENOTYPE_COL,
        backend=backend,
        output_path=output_path,
        title=f'{experiment_id}: Genotypes by Pair - {metric_label}',
        y_label=metric_label,
    )
    print(f"  Saved to {output_path}.*")

    # 3. Homozygous across pairs
    print("\n3. Creating homozygous across pairs plot...")
    # Find homozygous genotype
    homozygous = [g for g in df[GENOTYPE_COL].unique() if 'homo' in g.lower()]
    if homozygous:
        output_path = figures_dir / f'{experiment_id}_{metric_abbrev}_homozygous_across_pairs'
        plot_single_genotype_across_pairs(
            df,
            genotype=homozygous[0],
            x_col=TIME_COL,
            y_col=metric_name,
            line_by=EMBRYO_ID_COL,
            pair_col=PAIR_COL,
            genotype_col=GENOTYPE_COL,
            backend=backend,
            output_path=output_path,
            title=f'{experiment_id}: {homozygous[0]} Across Pairs - {metric_label}',
            y_label=metric_label,
        )
        print(f"  Saved to {output_path}.*")
    else:
        print("  No homozygous genotype found, skipping.")

    # 4. Create summary statistics
    print("\n4. Creating summary statistics...")
    create_summary_statistics(df, metric_name, tables_dir)

    print("\n" + "=" * 80)
    print(f"Analysis complete for {experiment_id} - {metric_name}!")
    print(f"Figures: {figures_dir}")
    print(f"Tables: {tables_dir}")
    print("=" * 80)


def main():
    """Main analysis function - loops over all experiments and metrics."""
    print("\n" + "=" * 80)
    print("B9D2 PAIR ANALYSIS - REFACTORED VERSION")
    print("Using new plotting framework (plot_pairs_overview, etc.)")
    print("=" * 80)
    print(f"Experiments: {EXPERIMENT_IDS}")
    print(f"Metrics: {METRICS}")
    print(f"Output formats: PNG={GENERATE_PNG}, Plotly={GENERATE_PLOTLY}")
    print("=" * 80)

    # Loop over all combinations
    for experiment_id in EXPERIMENT_IDS:
        for metric_name in METRICS:
            analyze_experiment_metric(experiment_id, metric_name)

    print("\n" + "=" * 80)
    print("ALL ANALYSES COMPLETE!")
    print(f"Results saved to: {BASE_OUTPUT_DIR}")
    print("=" * 80)


if __name__ == '__main__':
    main()
