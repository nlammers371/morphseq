"""
Threshold Optimization via Horizon Plot Analysis

This script implements variance-minimization threshold optimization to identify
optimal curvature thresholds that separate embryos into groups with minimal
within-group variance at future timepoints.

Per-genotype analysis: WT, Het, and Homo analyzed independently (no mixing).
Bootstrap validation: 50 iterations with 20% embryo holdout per iteration.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
import random
from multiprocessing import Pool
import os

# Import utilities from existing analysis
from load_data import (
    get_analysis_dataframe,
    get_genotype_short_name,
    get_genotype_color,
)

# Import horizon plotting utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))
from analyze.difference_detection.horizon_plots import plot_single_horizon

warnings.filterwarnings('ignore')

# Configuration
METRIC_NAME = 'normalized_baseline_deviation'
TIME_BIN_WIDTH = 2.0  # hpf
THRESHOLD_PERCENTILES = [10, 25, 50, 75, 90]
N_BOOTSTRAP = 50
BOOTSTRAP_HOLDOUT_FRACTION = 0.2
RANDOM_SEED = 42

# Output directories
OUTPUT_DIR = Path(__file__).parent / 'outputs' / '05_threshold_optimization'
FIGURE_DIR = OUTPUT_DIR / 'figures'
TABLE_DIR = OUTPUT_DIR / 'tables'

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# Genotypes
GENOTYPES = [
    'cep290_wildtype',
    'cep290_heterozygous',
    'cep290_homozygous'
]

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def bin_data_by_time(df, bin_width=2.0, time_col='predicted_stage_hpf'):
    """
    Bin data by developmental time.

    Parameters
    ----------
    df : pd.DataFrame
        Data with time column
    bin_width : float
        Width of each bin in hpf
    time_col : str
        Column name for developmental time

    Returns
    -------
    df : pd.DataFrame
        Data with added 'time_bin' column
    time_bins : np.ndarray
        Array of bin centers
    """
    df = df.copy()
    min_time = df[time_col].min()
    max_time = df[time_col].max()

    # Create bins
    bin_edges = np.arange(
        np.floor(min_time / bin_width) * bin_width,
        np.ceil(max_time / bin_width) * bin_width + bin_width,
        bin_width
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Assign bins
    df['time_bin'] = pd.cut(df[time_col], bins=bin_edges, labels=bin_centers)
    df['time_bin'] = df['time_bin'].astype(float)

    return df, bin_centers


def get_embryos_at_time(df, time_bin, metric=METRIC_NAME):
    """
    Get metric values for all embryos at a specific time bin.

    Parameters
    ----------
    df : pd.DataFrame
        Data with time_bin column
    time_bin : float
        Target time bin
    metric : str
        Metric column name

    Returns
    -------
    dict
        Dictionary mapping embryo_id to metric value (average across frames in bin)
    """
    bin_df = df[df['time_bin'] == time_bin].copy()

    # Average metric per embryo within the bin
    embryo_values = {}
    for embryo_id in bin_df['embryo_id'].unique():
        values = bin_df[bin_df['embryo_id'] == embryo_id][metric].values
        if len(values) > 0:
            embryo_values[embryo_id] = np.mean(values)

    return embryo_values


def find_optimal_threshold(
    df,
    time_i,
    time_j,
    threshold_percentiles=THRESHOLD_PERCENTILES,
    metric=METRIC_NAME
):
    """
    Find threshold that minimizes within-group variance at future timepoint.

    For a given pair of timepoints (i, j), this function:
    1. Gets metric values at time_i for all embryos
    2. Tests various threshold values
    3. For each threshold, splits embryos and computes variance at time_j
    4. Returns threshold that minimizes total weighted variance

    Parameters
    ----------
    df : pd.DataFrame
        Data with time_bin and metric columns
    time_i : float
        Current time bin
    time_j : float
        Future time bin (must be > time_i)
    threshold_percentiles : list
        Percentiles to test as thresholds
    metric : str
        Metric column name

    Returns
    -------
    optimal_threshold : float
        Threshold that minimizes variance
    min_variance : float
        Minimized within-group variance
    """
    # Get metric values at time_i
    embryo_values_i = get_embryos_at_time(df, time_i, metric=metric)

    if len(embryo_values_i) == 0:
        return np.nan, np.nan

    # Generate threshold candidates from percentiles at time_i
    values_array = np.array(list(embryo_values_i.values()))
    thresholds = np.percentile(values_array, threshold_percentiles)

    # Get metric values at time_j
    embryo_values_j = get_embryos_at_time(df, time_j, metric=metric)

    if len(embryo_values_j) == 0:
        return np.nan, np.nan

    best_threshold = np.nan
    min_variance = np.inf

    # Test each threshold
    for tau in thresholds:
        # Split embryos into high/low groups at time_i
        high_embryos = [e for e, v in embryo_values_i.items() if v > tau]
        low_embryos = [e for e, v in embryo_values_i.items() if v <= tau]

        if len(high_embryos) == 0 or len(low_embryos) == 0:
            continue  # Skip if threshold splits 0 embryos to one group

        # Get variance at time_j for each group
        high_values_j = [embryo_values_j[e] for e in high_embryos if e in embryo_values_j]
        low_values_j = [embryo_values_j[e] for e in low_embryos if e in embryo_values_j]

        if len(high_values_j) == 0 or len(low_values_j) == 0:
            continue

        # Compute within-group variances
        var_high = np.var(high_values_j)
        var_low = np.var(low_values_j)

        # Weighted total variance
        n_high = len(high_values_j)
        n_low = len(low_values_j)
        total_var = (n_high * var_high + n_low * var_low) / (n_high + n_low)

        # Track best
        if total_var < min_variance:
            min_variance = total_var
            best_threshold = tau

    return best_threshold, min_variance


def _optimize_single_node(args):
    """
    Helper function for parallel optimization of single (i,j) node.
    Unpacks args to work with multiprocessing.
    """
    df, time_i, time_j, metric = args
    tau_opt, var_min = find_optimal_threshold(df, time_i, time_j, metric=metric)
    return (time_i, time_j, tau_opt, var_min)


def optimize_thresholds_for_genotype(
    df,
    genotype,
    time_bins,
    metric=METRIC_NAME,
    n_jobs=-1
):
    """
    Optimize thresholds for all (time_i, time_j) pairs for a genotype.
    Uses parallel processing for speedup.

    Parameters
    ----------
    df : pd.DataFrame
        Data for this genotype
    genotype : str
        Genotype name
    time_bins : np.ndarray
        Array of time bin centers
    metric : str
        Metric column name
    n_jobs : int
        Number of parallel jobs (-1 = all CPUs)

    Returns
    -------
    optimal_thresholds : np.ndarray
        (n_times, n_times) matrix of optimal thresholds
    variance_reduction : np.ndarray
        (n_times, n_times) matrix of minimized variance values
    """
    n_times = len(time_bins)
    optimal_thresholds = np.full((n_times, n_times), np.nan)
    variance_reduction = np.full((n_times, n_times), np.nan)

    print(f"\n{genotype}:")
    print(f"  Optimizing thresholds for {n_times} timepoints...")

    # Create task list for parallel processing
    tasks = []
    for i, time_i in enumerate(time_bins):
        for j, time_j in enumerate(time_bins):
            if time_i >= time_j:
                continue  # Only upper triangle (forward prediction)
            tasks.append((df, time_i, time_j, metric))

    # Determine number of workers
    if n_jobs == -1:
        n_workers = os.cpu_count() - 1 or 1
    else:
        n_workers = n_jobs

    print(f"  Using {n_workers} parallel workers for {len(tasks)} nodes...")

    # Parallel execution
    with Pool(n_workers) as pool:
        results = pool.map(_optimize_single_node, tasks)

    # Fill result matrices
    for time_i, time_j, tau_opt, var_min in results:
        i = np.where(time_bins == time_i)[0][0]
        j = np.where(time_bins == time_j)[0][0]
        optimal_thresholds[i, j] = tau_opt
        variance_reduction[i, j] = var_min

    print(f"  Completed all {len(tasks)} nodes")

    return optimal_thresholds, variance_reduction


def _optimize_serial(df, time_bins, metric=METRIC_NAME):
    """
    Serial optimization without multiprocessing (for nested calls).
    Used internally by bootstrap iterations.
    """
    n_times = len(time_bins)
    optimal_thresholds = np.full((n_times, n_times), np.nan)
    variance_reduction = np.full((n_times, n_times), np.nan)

    for i, time_i in enumerate(time_bins):
        for j, time_j in enumerate(time_bins):
            if time_i >= time_j:
                continue

            tau_opt, var_min = find_optimal_threshold(
                df, time_i, time_j, metric=metric
            )
            optimal_thresholds[i, j] = tau_opt
            variance_reduction[i, j] = var_min

    return optimal_thresholds, variance_reduction


def _bootstrap_single_iteration(args):
    """Helper function for parallel bootstrap iterations."""
    df, embryo_ids, time_bins, holdout_fraction, metric, iter_idx = args
    n_holdout = max(1, int(len(embryo_ids) * holdout_fraction))
    holdout_embryos = set(random.sample(list(embryo_ids), n_holdout))
    train_df = df[~df['embryo_id'].isin(holdout_embryos)].copy()

    # Use serial optimization to avoid nested multiprocessing
    tau_opt, _ = _optimize_serial(train_df, time_bins, metric=metric)
    return (iter_idx, tau_opt)


def bootstrap_threshold_stability(
    df,
    genotype,
    time_bins,
    n_iterations=N_BOOTSTRAP,
    holdout_fraction=BOOTSTRAP_HOLDOUT_FRACTION,
    metric=METRIC_NAME,
    n_jobs=-1
):
    """
    Bootstrap threshold optimization to assess stability.
    Uses parallel processing for speedup.

    For each iteration:
    1. Randomly hold out holdout_fraction of embryos
    2. Re-run threshold optimization on remaining embryos
    3. Track variation in optimal thresholds across iterations

    Parameters
    ----------
    df : pd.DataFrame
        Data for this genotype
    genotype : str
        Genotype name
    time_bins : np.ndarray
        Array of time bin centers
    n_iterations : int
        Number of bootstrap iterations
    holdout_fraction : float
        Fraction of embryos to hold out per iteration
    metric : str
        Metric column name
    n_jobs : int
        Number of parallel jobs (-1 = all CPUs)

    Returns
    -------
    threshold_mean : np.ndarray
        Mean optimal threshold across bootstraps
    threshold_sd : np.ndarray
        SD of optimal threshold across bootstraps
    bootstrap_thresholds : list
        List of threshold matrices from each bootstrap
    """
    embryo_ids = df['embryo_id'].unique()
    n_embryos = len(embryo_ids)
    n_holdout = max(1, int(n_embryos * holdout_fraction))

    print(f"  Bootstrapping {n_iterations} iterations (holding out {n_holdout}/{n_embryos} embryos)...")

    # Create task list for parallel processing
    tasks = [
        (df, embryo_ids, time_bins, holdout_fraction, metric, iter_idx)
        for iter_idx in range(n_iterations)
    ]

    # Determine number of workers
    if n_jobs == -1:
        n_workers = max(1, os.cpu_count() - 1)
    else:
        n_workers = n_jobs

    print(f"  Using {n_workers} parallel workers for {len(tasks)} bootstrap iterations...")

    # Parallel execution
    bootstrap_thresholds = [None] * n_iterations
    with Pool(n_workers) as pool:
        results = pool.map(_bootstrap_single_iteration, tasks)

    # Sort results by iteration index and extract threshold matrices
    results.sort(key=lambda x: x[0])
    bootstrap_thresholds = [tau_opt for _, tau_opt in results]

    print(f"  Completed all {n_iterations} bootstrap iterations")

    # Compute statistics across bootstraps
    bootstrap_array = np.array(bootstrap_thresholds)
    threshold_mean = np.nanmean(bootstrap_array, axis=0)
    threshold_sd = np.nanstd(bootstrap_array, axis=0)

    return threshold_mean, threshold_sd, bootstrap_thresholds


def plot_horizon_heatmap(
    matrix,
    time_bins,
    title,
    cmap='RdYlBu_r',
    vmin=None,
    vmax=None,
    output_path=None,
    figsize=(10, 9)
):
    """
    Plot a single horizon heatmap using the standard horizon_plots module.

    Automatically uses correct orientation:
    - Upper right corner: early times at top, late times at right
    - origin='upper' is default in plot_single_horizon()

    Parameters
    ----------
    matrix : np.ndarray
        (n_times, n_times) matrix to plot
    time_bins : np.ndarray
        Time bin centers (for labels)
    title : str
        Plot title
    cmap : str
        Colormap name
    vmin, vmax : float
        Color scale limits
    output_path : Path
        Path to save figure
    figsize : tuple
        Figure size
    """
    # Convert to DataFrame with proper index/columns for plot_single_horizon
    matrix_plot = np.copy(matrix)
    matrix_plot[np.tril_indices_from(matrix_plot, k=-1)] = np.nan

    df_matrix = pd.DataFrame(
        matrix_plot,
        index=time_bins,
        columns=time_bins
    )

    # Use the standard horizon plot function (uses origin='upper' by default)
    fig, ax = plt.subplots(figsize=figsize)

    plot_single_horizon(
        df_matrix,
        metric='value',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        title=title,
        xlabel='Future Time (hpf)',
        ylabel='Current Time (hpf)',
        annotate=False,
        add_colorbar=True
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path.name}")

    plt.close(fig)


def plot_threshold_comparison(
    thresholds_dict,
    time_bins,
    output_path=None,
    figsize=(15, 4)
):
    """
    Compare optimal thresholds across genotypes.

    Parameters
    ----------
    thresholds_dict : dict
        Dictionary mapping genotype -> threshold matrix
    time_bins : np.ndarray
        Time bin centers
    output_path : Path
        Path to save figure
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    genotype_list = ['cep290_wildtype', 'cep290_heterozygous', 'cep290_homozygous']
    genotype_short = ['WT', 'Het', 'Homo']

    for ax_idx, (genotype, short_name) in enumerate(zip(genotype_list, genotype_short)):
        if genotype not in thresholds_dict:
            continue

        matrix = thresholds_dict[genotype]
        matrix_plot = np.copy(matrix)
        matrix_plot[np.tril_indices_from(matrix_plot, k=-1)] = np.nan

        im = axes[ax_idx].imshow(
            matrix_plot,
            cmap='viridis',
            aspect='auto',
            origin='lower'
        )

        tick_positions = np.arange(len(time_bins))
        tick_labels = [f'{t:.0f}' for t in time_bins]

        axes[ax_idx].set_xticks(tick_positions)
        axes[ax_idx].set_yticks(tick_positions)
        axes[ax_idx].set_xticklabels(tick_labels, rotation=45)
        axes[ax_idx].set_yticklabels(tick_labels)

        axes[ax_idx].set_xlabel('Future Time (hpf)', fontsize=10)
        axes[ax_idx].set_ylabel('Current Time (hpf)', fontsize=10)
        axes[ax_idx].set_title(f'{short_name}', fontsize=12, fontweight='bold')

        plt.colorbar(im, ax=axes[ax_idx], label='Threshold')

    plt.suptitle('Optimal Curvature Thresholds (normalized_baseline_deviation)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path.name}")

    plt.close()


def plot_threshold_split_trajectories(
    df,
    genotype,
    time_bins,
    optimal_thresholds,
    metric=METRIC_NAME,
    output_dir=FIGURE_DIR,
    future_idx=None
):
    """
    Plot embryo trajectories split by optimal threshold at earliest time.

    For the earliest time bin, shows:
    - Embryos above threshold (high group) in one color
    - Embryos below threshold (low group) in another color
    - Optimal threshold value as horizontal line
    - Threshold was optimized to minimize variance at a specific future time

    Parameters
    ----------
    df : pd.DataFrame
        Data for this genotype
    genotype : str
        Genotype name
    time_bins : np.ndarray
        Time bin centers
    optimal_thresholds : np.ndarray
        Optimal threshold matrix
    metric : str
        Metric column name
    output_dir : Path
        Directory to save figures
    future_idx : int, optional
        Index of future time bin used for threshold optimization.
        If None, uses midpoint of time_bins.
    """
    genotype_short = get_genotype_short_name(genotype)
    genotype_color = get_genotype_color(genotype)

    # Find earliest time bin with valid threshold
    earliest_time = time_bins[0]
    earliest_idx = 0

    # Determine future time index
    if future_idx is None:
        future_idx = len(time_bins) // 2

    future_time = time_bins[future_idx]

    # Get optimal threshold for this pair
    tau_opt = optimal_thresholds[earliest_idx, future_idx]

    if np.isnan(tau_opt):
        print(f"  Skipping threshold split plot for {genotype_short} (no valid threshold at t{earliest_time:.0f}->t{future_time:.0f})")
        return

    # Get embryo values at earliest time
    embryo_values_early = get_embryos_at_time(df, earliest_time, metric=metric)

    if len(embryo_values_early) == 0:
        return

    # Split embryos into high/low groups
    high_embryos = [e for e, v in embryo_values_early.items() if v > tau_opt]
    low_embryos = [e for e, v in embryo_values_early.items() if v <= tau_opt]

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot low group trajectories
    for embryo_id in low_embryos:
        embryo_df = df[df['embryo_id'] == embryo_id].sort_values('predicted_stage_hpf')
        ax.plot(embryo_df['predicted_stage_hpf'], embryo_df[metric],
                color='blue', alpha=0.3, linewidth=1)

    # Plot high group trajectories
    for embryo_id in high_embryos:
        embryo_df = df[df['embryo_id'] == embryo_id].sort_values('predicted_stage_hpf')
        ax.plot(embryo_df['predicted_stage_hpf'], embryo_df[metric],
                color='red', alpha=0.3, linewidth=1)

    # Add threshold line at earliest time
    ax.axhline(tau_opt, color='black', linestyle='--', linewidth=2,
               label=f'Optimal τ = {tau_opt:.4f} (minimizes variance at {future_time:.0f} hpf)')
    ax.axvline(earliest_time, color='gray', linestyle=':', alpha=0.5, label='Classification time')

    # Add legend entries for groups
    ax.plot([], [], color='blue', linewidth=2, label=f'Low group (n={len(low_embryos)})')
    ax.plot([], [], color='red', linewidth=2, label=f'High group (n={len(high_embryos)})')

    ax.set_xlabel('Developmental Time (hpf)', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'{genotype_short}: Trajectories Split by τ at {earliest_time:.0f} hpf\n(Threshold optimized for {future_time:.0f} hpf)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / f'threshold_split_trajectories_{genotype_short}_t{earliest_time:.0f}hpf_pred{future_time:.0f}hpf.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path.name}")
    plt.close()


def save_matrix_to_csv(matrix, time_bins, output_path):
    """
    Save horizon matrix to CSV with time bin labels.

    Parameters
    ----------
    matrix : np.ndarray
        (n_times, n_times) matrix
    time_bins : np.ndarray
        Time bin centers
    output_path : Path
        Path to save CSV
    """
    df = pd.DataFrame(matrix, index=time_bins, columns=time_bins)
    df.index.name = 'current_time_hpf'
    df.columns.name = 'future_time_hpf'
    df.to_csv(output_path)
    print(f"  Saved: {output_path.name}")


def main():
    """Main analysis pipeline."""

    print("=" * 80)
    print("THRESHOLD OPTIMIZATION VIA HORIZON PLOT ANALYSIS")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    df, metadata = get_analysis_dataframe()
    print(f"  Loaded {len(df)} timepoints from {df['embryo_id'].nunique()} embryos")
    print(f"  Genotypes: {df['genotype'].unique()}")

    # Bin data by time
    print(f"\nBinning data by {TIME_BIN_WIDTH} hpf windows...")
    df, time_bins = bin_data_by_time(df, bin_width=TIME_BIN_WIDTH)
    print(f"  Created {len(time_bins)} time bins")
    print(f"  Time range: {time_bins[0]:.1f} - {time_bins[-1]:.1f} hpf")

    # Results storage
    results_all = {}

    # Per-genotype analysis
    for genotype in GENOTYPES[:-1]:
        print(f"\n{'='*80}")
        print(f"ANALYZING GENOTYPE: {genotype}")
        print(f"{'='*80}")

        genotype_df = df[df['genotype'] == genotype].copy()
        n_embryos = genotype_df['embryo_id'].nunique()
        n_timepoints = len(genotype_df)

        print(f"  {n_embryos} embryos, {n_timepoints} timepoints")

        # Optimize thresholds
        print(f"\nOptimizing thresholds (variance minimization)...")
        optimal_tau, variance_min = optimize_thresholds_for_genotype(
            genotype_df, genotype, time_bins
        )

        # Bootstrap stability
        print(f"\nAssessing threshold stability via bootstrap...")
        tau_mean, tau_sd, tau_bootstrap = bootstrap_threshold_stability(
            genotype_df, genotype, time_bins
        )

        # Store results
        results_all[genotype] = {
            'optimal_thresholds': optimal_tau,
            'variance_reduction': variance_min,
            'bootstrap_mean': tau_mean,
            'bootstrap_sd': tau_sd,
            'time_bins': time_bins
        }

        # Save matrices
        genotype_short = get_genotype_short_name(genotype)

        print(f"\nSaving results for {genotype_short}...")
        save_matrix_to_csv(
            optimal_tau,
            time_bins,
            TABLE_DIR / f'optimal_thresholds_matrix_{genotype_short}.csv'
        )
        save_matrix_to_csv(
            variance_min,
            time_bins,
            TABLE_DIR / f'variance_reduction_{genotype_short}.csv'
        )
        save_matrix_to_csv(
            tau_mean,
            time_bins,
            TABLE_DIR / f'bootstrap_mean_thresholds_{genotype_short}.csv'
        )
        save_matrix_to_csv(
            tau_sd,
            time_bins,
            TABLE_DIR / f'bootstrap_stability_{genotype_short}.csv'
        )

        # Plot horizon maps
        print(f"\nGenerating horizon plots for {genotype_short}...")

        plot_horizon_heatmap(
            optimal_tau,
            time_bins,
            f'{genotype_short}: Optimal Thresholds',
            cmap='viridis',
            output_path=FIGURE_DIR / f'optimal_thresholds_horizon_{genotype_short}.png'
        )

        plot_horizon_heatmap(
            tau_sd,
            time_bins,
            f'{genotype_short}: Threshold Stability (SD)',
            cmap='RdYlGn_r',
            output_path=FIGURE_DIR / f'threshold_stability_horizon_{genotype_short}.png'
        )

        plot_horizon_heatmap(
            variance_min,
            time_bins,
            f'{genotype_short}: Variance Reduction',
            cmap='RdYlGn_r',
            output_path=FIGURE_DIR / f'variance_reduction_horizon_{genotype_short}.png'
        )

        # Plot threshold split trajectories
        # Find index of time bin closest to 100 hpf for prediction
        pred_time_target = 100.0
        pred_idx = np.argmin(np.abs(time_bins - pred_time_target))

        print(f"\nGenerating threshold split visualization for {genotype_short}...")
        print(f"  Using prediction target: {time_bins[pred_idx]:.1f} hpf (index {pred_idx})")

        plot_threshold_split_trajectories(
            genotype_df,
            genotype,
            time_bins,
            optimal_tau,
            metric=METRIC_NAME,
            output_dir=FIGURE_DIR,
            future_idx=pred_idx
        )

    # Cross-genotype comparison
    print(f"\n{'='*80}")
    print("CROSS-GENOTYPE COMPARISON")
    print(f"{'='*80}\n")

    thresholds_dict = {
        g: results_all[g]['optimal_thresholds'] for g in GENOTYPES
    }

    plot_threshold_comparison(
        thresholds_dict,
        time_bins,
        output_path=FIGURE_DIR / 'threshold_comparison_across_genotypes.png'
    )

    # Summary statistics
    print("\nGenerating summary statistics...")
    summary_stats = []

    for genotype in GENOTYPES:
        tau_opt = results_all[genotype]['optimal_thresholds']
        tau_sd = results_all[genotype]['bootstrap_sd']
        var_min = results_all[genotype]['variance_reduction']

        # Count valid nodes
        valid_nodes = ~np.isnan(tau_opt)

        summary_stats.append({
            'genotype': genotype,
            'n_valid_nodes': np.sum(valid_nodes),
            'mean_threshold': np.nanmean(tau_opt),
            'sd_threshold': np.nanstd(tau_opt),
            'mean_stability': np.nanmean(tau_sd),
            'mean_variance': np.nanmean(var_min),
            'sd_variance': np.nanstd(var_min)
        })

    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(TABLE_DIR / 'summary_statistics.csv', index=False)
    print(f"  Saved: summary_statistics.csv\n")
    print(summary_df.to_string(index=False))

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutputs saved to:")
    print(f"  Figures: {FIGURE_DIR}")
    print(f"  Tables: {TABLE_DIR}")


if __name__ == '__main__':
    main()
