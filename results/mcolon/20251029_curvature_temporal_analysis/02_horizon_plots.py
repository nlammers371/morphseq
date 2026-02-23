#!/usr/bin/env python3
"""
Create horizon plots showing temporal correlation structure of curvature measurements.

Horizon plots display a start_time × target_time correlation matrix as a heatmap.
This reveals how well curvature at one developmental timepoint predicts curvature
at another, showing consistency of developmental information through time.

The analysis uses binned timepoints to aggregate observations across multiple embryos:
- Each observation is binned to the nearest time bin (e.g., 2 hpf bins)
- For each pair of time bins, we compute the Pearson correlation of metric values
  across all embryos that have observations at both times
- Result shows genotype-specific temporal prediction patterns

Uses reusable utilities from analyze.difference_detection package.

Output
------
- Correlation matrices as CSV (one per genotype)
- Horizon grid PNG figure comparing all three genotypes
- Summary statistics on temporal correlation structure
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Setup imports from src
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / 'src'))

from analyze.difference_detection import (
    plot_horizon_grid,
    plot_single_horizon,
)
from analyze.difference_detection.time_matrix import build_metric_matrices

# Import data loading from this directory
from load_data import get_analysis_dataframe, get_genotype_short_name


# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR = Path(__file__).parent
FIGURE_DIR = RESULTS_DIR / 'outputs' / 'figures' / '02_horizon_plots'
TABLE_DIR = RESULTS_DIR / 'outputs' / 'tables' / '02_horizon_plots'

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# Analysis parameters
TIME_BIN_WIDTH = 2.0  # hpf
PRIMARY_METRICS = ['normalized_baseline_deviation']
METRIC_LABELS = {
    'normalized_baseline_deviation': 'Normalized Baseline Deviation'
}

# Genotypes in display order
GENOTYPES = ['cep290_wildtype', 'cep290_heterozygous', 'cep290_homozygous']


# ============================================================================
# Time Binning
# ============================================================================

def bin_timepoint(time_value: float, bin_width: float = TIME_BIN_WIDTH) -> float:
    """
    Bin a continuous timepoint to the nearest bin boundary.

    Parameters
    ----------
    time_value : float
        Timepoint in hpf
    bin_width : float
        Width of bins in hpf

    Returns
    -------
    float
        Binned timepoint (multiple of bin_width)
    """
    return round(time_value / bin_width) * bin_width


def prepare_time_matrix_data(
    df: pd.DataFrame,
    metric: str = 'arc_length_ratio',
    bin_width: float = TIME_BIN_WIDTH
) -> pd.DataFrame:
    """
    Prepare curvature data into time matrix format for correlation computation.

    Creates a dataframe with columns: start_time, target_time, correlation
    where each row represents the correlation between metric values at two
    different binned timepoints, computed across all embryos.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with embryo_id, predicted_stage_hpf, genotype, metric
    metric : str
        Name of the metric column to correlate
    bin_width : float
        Width of time bins in hpf

    Returns
    -------
    pd.DataFrame
        Long-form dataframe with columns: start_time, target_time, correlation, genotype
    """
    # Validate that metric column exists
    if metric not in df.columns:
        raise ValueError(f"Metric column '{metric}' not found in dataframe")

    # Add binned time column
    df = df.copy()
    df['time_bin'] = df['predicted_stage_hpf'].apply(lambda t: bin_timepoint(t, bin_width))

    # Get unique time bins
    time_bins = sorted(df['time_bin'].unique())
    print(f"    Found {len(time_bins)} unique time bins from {df['time_bin'].min():.1f} to {df['time_bin'].max():.1f} hpf")

    # Compute pairwise correlations across embryos
    # Only compute for forward-in-time pairs (start_time < target_time)
    results = []

    for i, time_i in enumerate(time_bins):
        for j, time_j in enumerate(time_bins):
            # Only correlate forward in time (start < target)
            if time_i >= time_j:
                continue

            # Get embryos with observations at both times
            data_i = df[df['time_bin'] == time_i]
            data_j = df[df['time_bin'] == time_j]

            # Find embryos present at both times
            embryos_i = set(data_i['embryo_id'].unique())
            embryos_j = set(data_j['embryo_id'].unique())
            common_embryos = sorted(embryos_i & embryos_j)

            if len(common_embryos) < 2:
                # Need at least 2 embryos for meaningful correlation
                continue

            # Collect metric values at both times for each embryo
            values_i = []
            values_j = []

            for embryo_id in common_embryos:
                embryo_i_data = data_i[data_i['embryo_id'] == embryo_id][metric].values
                embryo_j_data = data_j[data_j['embryo_id'] == embryo_id][metric].values

                # Take mean if multiple measurements at this timepoint
                if len(embryo_i_data) > 0 and len(embryo_j_data) > 0:
                    values_i.append(embryo_i_data.mean())
                    values_j.append(embryo_j_data.mean())

            if len(values_i) >= 2:
                try:
                    corr, _ = pearsonr(values_i, values_j)
                except:
                    corr = np.nan

                results.append({
                    'start_time': time_i,
                    'target_time': time_j,
                    'correlation': corr,
                })

    result_df = pd.DataFrame(results)
    return result_df


def compute_correlation_matrices_by_genotype(
    df: pd.DataFrame,
    metric: str = 'arc_length_ratio',
    bin_width: float = TIME_BIN_WIDTH
) -> dict:
    """
    Compute time × time correlation matrices for each genotype.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with embryo_id, genotype, predicted_stage_hpf, metric
    metric : str
        Name of metric column
    bin_width : float
        Width of time bins

    Returns
    -------
    dict
        Mapping of genotype -> correlation matrix (DataFrame)
    """
    matrices = {}

    for genotype in GENOTYPES:
        genotype_df = df[df['genotype'] == genotype].copy()

        if len(genotype_df) == 0:
            print(f"    No data for {genotype}")
            matrices[genotype] = None
            continue

        # Prepare long-form time matrix data
        time_matrix_df = prepare_time_matrix_data(genotype_df, metric=metric, bin_width=bin_width)

        if len(time_matrix_df) == 0:
            print(f"    Could not compute correlations for {genotype}")
            matrices[genotype] = None
            continue

        # Build matrix using analyze.difference_detection utility
        matrix = build_metric_matrices(
            time_matrix_df,
            metric='correlation',
            start_col='start_time',
            target_col='target_time',
            values_col='correlation',
            aggfunc='mean'
        )

        matrices[genotype] = matrix
        genotype_label = get_genotype_short_name(genotype)
        print(f"    {genotype_label}: {matrix.shape[0]} × {matrix.shape[1]} matrix")

    return matrices


# ============================================================================
# Plotting
# ============================================================================

def plot_genotype_horizon_comparison(
    matrices: dict,
    metric: str = 'arc_length_ratio',
    save_dir: Path = FIGURE_DIR
) -> Path:
    """
    Create horizon plots comparing all three genotypes side-by-side.

    Uses plot_horizon_grid from analyze.difference_detection.

    Parameters
    ----------
    matrices : dict
        Mapping of genotype -> correlation matrix
    metric : str
        Metric name (for title)
    save_dir : Path
        Directory to save figure

    Returns
    -------
    Path
        Path to saved figure
    """
    # Reformat matrices dict to nested structure expected by plot_horizon_grid
    # plot_horizon_grid expects: {row_label: {col_label: matrix}}
    # We want 1 row, 3 columns (one per genotype)
    matrices_nested = {'Correlation': {}}
    for genotype in GENOTYPES:
        if matrices[genotype] is not None:
            genotype_label = get_genotype_short_name(genotype)
            matrices_nested['Correlation'][genotype_label] = matrices[genotype]

    if not matrices_nested['Correlation']:
        print("    No matrices to plot")
        return None

    # Use plot_horizon_grid with larger figure size
    fig = plot_horizon_grid(
        matrices_nested,
        row_labels=[''],
        col_labels=list(matrices_nested['Correlation'].keys()),
        metric='Correlation',
        cmap='RdBu_r',
        clip_percentiles=None,  # Use full -1 to +1 range for correlation
        title=f'Temporal Prediction: {METRIC_LABELS.get(metric, metric)}',
        figsize=(18, 6),  # Larger figure to fit all labels
        save_path=save_dir / f'horizon_comparison_{metric}.png'
    )

    return save_dir / f'horizon_comparison_{metric}.png'


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    print("\n" + "="*80)
    print("CREATING HORIZON PLOTS: TEMPORAL CURVATURE CORRELATION")
    print("="*80)

    # Load data
    print("\nSTEP 1: LOADING AND PREPARING DATA")
    df, metadata = get_analysis_dataframe()

    # Create horizon plots for each metric
    print("\nSTEP 2: COMPUTING CORRELATION MATRICES")

    for metric in PRIMARY_METRICS:
        print(f"\n  Processing metric: {metric}")
        print(f"    Time bin width: {TIME_BIN_WIDTH} hpf")

        # Compute correlation matrices for each genotype
        matrices = compute_correlation_matrices_by_genotype(df, metric=metric, bin_width=TIME_BIN_WIDTH)

        # Plot genotype comparison
        fig_path = plot_genotype_horizon_comparison(matrices, metric=metric)
        if fig_path:
            print(f"    Saved genotype comparison: {fig_path}")

        # Save correlation matrices as CSV
        for genotype in GENOTYPES:
            if matrices[genotype] is not None:
                genotype_label = get_genotype_short_name(genotype)
                csv_path = TABLE_DIR / f'correlation_matrix_{metric}_{genotype_label}.csv'
                matrices[genotype].to_csv(csv_path)
                print(f"    Saved {genotype_label} correlations: {csv_path}")

    print("\n" + "="*80)
    print("HORIZON PLOT CREATION COMPLETE")
    print("="*80)
    print(f"\nOutputs saved to:")
    print(f"  Figures: {FIGURE_DIR}")
    print(f"  Tables:  {TABLE_DIR}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
