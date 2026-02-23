#!/usr/bin/env python3
"""
Analyze baseline deviation and length differences between pairs and genotypes for b9d2 experiments.

This script analyzes b9d2 data from experiments 20251104 and 20251119, creating faceted plots
comparing all genotypes (WT, Het, Homo) across pair groups for two metrics:
- baseline_deviation_normalized
- total_length_um
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys
from scipy.ndimage import gaussian_filter1d

# Add src to path
sys.path.insert(0, '/net/trapnell/vol1/home/mdcolon/proj/morphseq')

from src.analyze.trajectory_analysis.data_loading import _load_df03_format
from src.analyze.trajectory_analysis.plotting import (
    plot_cluster_trajectories_df,
)
from src.analyze.trajectory_analysis.pair_analysis import (
    get_trajectories_for_group,
    compute_binned_mean,
    plot_genotypes_overlaid,
    plot_faceted_trajectories,
    GENOTYPE_COLORS as DEFAULT_GENOTYPE_COLORS,
    GENOTYPE_ORDER as DEFAULT_GENOTYPE_ORDER,
)

# Configuration - will loop over these
# NOTE: 20251104 doesn't have df03 file yet in build06_output, so only analyzing 20251119
EXPERIMENT_IDS = ['20251104']  # ['20251104', '20251119'] when 20251104 is processed
METRICS = ['baseline_deviation_normalized', 'total_length_um', 'surface_area_um']

# Constants
TIME_COL = 'predicted_stage_hpf'
EMBRYO_ID_COL = 'embryo_id'
PAIR_COL = 'pair'
GENOTYPE_COL = 'genotype'

# Base output directory
BASE_OUTPUT_DIR = Path('/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251204_initial_b9d2_pair_analysis')

# Genotype configuration for b9d2
GENOTYPE_ORDER = ['b9d2_wildtype', 'b9d2_heterozygous', 'b9d2_homozygous']
GENOTYPE_COLORS = {
    'b9d2_wildtype': '#2E7D32',      # Green
    'b9d2_heterozygous': '#FFA500',  # Orange
    'b9d2_homozygous': '#D32F2F',    # Red
}

# Metric display names
METRIC_DISPLAY_NAMES = {
    'baseline_deviation_normalized': 'Normalized Baseline Deviation',
    'total_length_um': 'Total Length (µm)',
    'surface_area_um': 'Surface Area (µm²)',
}

# Smoothing configuration
SMOOTH_METHOD = 'gaussian'  # 'gaussian' or None for no smoothing
SMOOTH_PARAMS = {'sigma': 1.5}  # Parameters for Gaussian smoothing


def get_metric_label(metric_name):
    """Get human-readable label for a metric."""
    return METRIC_DISPLAY_NAMES.get(metric_name, metric_name)


def load_and_prepare_data(experiment_id, metric_name):
    """Load data and prepare for analysis."""
    print(f"Loading data for experiment {experiment_id}, metric {metric_name}...")

    df = _load_df03_format(experiment_id)

    # Handle column naming collisions from merge
    # When curvature_metrics and df03 both have total_length_um,
    # pandas creates _x and _y suffixes. Use the df03 version (_y) for consistency.
    if metric_name == 'total_length_um' and 'total_length_um_y' in df.columns:
        df['total_length_um'] = df['total_length_um_y']
        print(f"  Using total_length_um_y (from df03) as total_length_um")

    # Filter for valid embryos (use_embryo_flag == 1)
    df = df[df['use_embryo_flag'] == 1].copy()

    # Drop rows with missing values in key columns
    df = df.dropna(subset=[EMBRYO_ID_COL, TIME_COL, metric_name, PAIR_COL, GENOTYPE_COL])

    print(f"Data shape: {df.shape}")
    print(f"Unique pairs: {df[PAIR_COL].unique()}")
    print(f"Genotypes: {df[GENOTYPE_COL].unique()}")

    return df


def get_trajectories_for_pair_genotype(df, pair, genotype, metric_name):
    """Extract trajectories for a specific pair and genotype with smoothing."""
    filtered = df[(df[PAIR_COL] == pair) & (df[GENOTYPE_COL] == genotype)].copy()

    if len(filtered) == 0:
        return None, None, None

    # Group by embryo and get trajectories
    embryo_ids = filtered[EMBRYO_ID_COL].unique()
    trajectories = []

    for embryo_id in embryo_ids:
        embryo_data = filtered[filtered[EMBRYO_ID_COL] == embryo_id].sort_values(TIME_COL)
        if len(embryo_data) > 1:
            times = embryo_data[TIME_COL].values
            metrics = embryo_data[metric_name].values

            # Apply Gaussian smoothing if enabled
            if SMOOTH_METHOD == 'gaussian':
                sigma = SMOOTH_PARAMS.get('sigma', 1.5)
                metrics = gaussian_filter1d(metrics, sigma=sigma)

            trajectories.append({
                'embryo_id': embryo_id,
                'times': times,
                'metrics': metrics,
            })

    return trajectories, embryo_ids, len(trajectories)


def plot_trajectories_for_pair(df, pair, metric_name, output_path, global_time_min=None, global_time_max=None,
                               global_metric_min=None, global_metric_max=None):
    """Create a 1x3 faceted plot for a pair showing all three genotypes."""
    print(f"\nCreating plot for {pair}...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metric_label = get_metric_label(metric_name)
    fig.suptitle(f'{metric_label} Trajectories - {pair}', fontsize=14, fontweight='bold')

    # Collect data to determine axis ranges if not provided
    if global_time_min is None:
        all_data = {}
        global_time_min, global_time_max = float('inf'), float('-inf')
        global_metric_min, global_metric_max = float('inf'), float('-inf')

        for genotype in GENOTYPE_ORDER:
            trajectories, embryo_ids, n_embryos = get_trajectories_for_pair_genotype(df, pair, genotype, metric_name)
            all_data[genotype] = (trajectories, embryo_ids, n_embryos)

            if trajectories is not None and n_embryos > 0:
                for traj in trajectories:
                    global_time_min = min(global_time_min, traj['times'].min())
                    global_time_max = max(global_time_max, traj['times'].max())
                    global_metric_min = min(global_metric_min, traj['metrics'].min())
                    global_metric_max = max(global_metric_max, traj['metrics'].max())
    else:
        all_data = {}
        for genotype in GENOTYPE_ORDER:
            all_data[genotype] = get_trajectories_for_pair_genotype(df, pair, genotype, metric_name)

    # Add padding to metric axis
    metric_padding = (global_metric_max - global_metric_min) * 0.1
    global_metric_min -= metric_padding
    global_metric_max += metric_padding

    for ax_idx, genotype in enumerate(GENOTYPE_ORDER):
        ax = axes[ax_idx]

        trajectories, embryo_ids, n_embryos = all_data[genotype]

        if trajectories is None or n_embryos == 0:
            ax.text(0.5, 0.5, f'No data for\n{genotype.replace("b9d2_", "")}',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='gray')
            ax.set_xlabel('Time (hpf)')
            ax.set_ylabel(metric_label)
            ax.set_title(f'{genotype.replace("b9d2_", "").title()} (n={n_embryos})')
            ax.set_xlim(global_time_min, global_time_max)
            ax.set_ylim(global_metric_min, global_metric_max)
            continue

        # Plot individual trajectories
        color = GENOTYPE_COLORS[genotype]
        for traj in trajectories:
            ax.plot(traj['times'], traj['metrics'], alpha=0.3, linewidth=1, color=color)

        # Plot mean trajectory
        all_times = np.concatenate([t['times'] for t in trajectories])
        all_metrics = np.concatenate([t['metrics'] for t in trajectories])

        # Create mean by binning
        time_bins = np.arange(np.floor(all_times.min()), np.ceil(all_times.max()) + 1, 0.5)
        bin_means = []
        bin_times = []
        for i in range(len(time_bins) - 1):
            mask = (all_times >= time_bins[i]) & (all_times < time_bins[i+1])
            if mask.sum() > 0:
                bin_means.append(all_metrics[mask].mean())
                bin_times.append((time_bins[i] + time_bins[i+1]) / 2)

        if bin_times:
            ax.plot(bin_times, bin_means, color=color, linewidth=2.5, label='Mean', zorder=5)

        ax.set_xlabel('Time (hpf)')
        ax.set_ylabel(metric_label)
        ax.set_title(f'{genotype.replace("b9d2_", "").title()} (n={n_embryos})')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Set aligned axes
        ax.set_xlim(global_time_min, global_time_max)
        ax.set_ylim(global_metric_min, global_metric_max)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_all_pairs_overview(df, pairs, metric_name, output_path):
    """Create a comprehensive overview plot with all pairs.

    Layout: Rows = pairs, Columns = genotypes
    All axes are aligned for comparison.
    """
    print(f"\nCreating overview plot with all {len(pairs)} pairs...")

    n_pairs = len(pairs)
    n_genotypes = 3

    fig, axes = plt.subplots(n_pairs, n_genotypes, figsize=(15, 4.5 * n_pairs))

    # Ensure axes is always 2D even with single row
    if n_pairs == 1:
        axes = axes.reshape(1, -1)

    metric_label = get_metric_label(metric_name)
    fig.suptitle(f'{metric_label} Trajectories - All Pairs Overview',
                 fontsize=16, fontweight='bold', y=0.995)

    # First pass: collect all data to determine global axis ranges
    all_data = {}
    global_time_min, global_time_max = float('inf'), float('-inf')
    global_metric_min, global_metric_max = float('inf'), float('-inf')

    for pair in pairs:
        for genotype in GENOTYPE_ORDER:
            trajectories, embryo_ids, n_embryos = get_trajectories_for_pair_genotype(df, pair, genotype, metric_name)
            all_data[(pair, genotype)] = (trajectories, embryo_ids, n_embryos)

            if trajectories is not None and n_embryos > 0:
                for traj in trajectories:
                    global_time_min = min(global_time_min, traj['times'].min())
                    global_time_max = max(global_time_max, traj['times'].max())
                    global_metric_min = min(global_metric_min, traj['metrics'].min())
                    global_metric_max = max(global_metric_max, traj['metrics'].max())

    # Add padding to metric axis
    metric_padding = (global_metric_max - global_metric_min) * 0.1
    global_metric_min -= metric_padding
    global_metric_max += metric_padding

    # Second pass: plot with aligned axes
    for row_idx, pair in enumerate(pairs):
        for col_idx, genotype in enumerate(GENOTYPE_ORDER):
            ax = axes[row_idx, col_idx]

            trajectories, embryo_ids, n_embryos = all_data[(pair, genotype)]

            if trajectories is None or n_embryos == 0:
                ax.text(0.5, 0.5, f'No data',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, color='lightgray')
                ax.set_xlabel('Time (hpf)', fontsize=9)
                ax.set_ylabel(metric_label, fontsize=9)
                if row_idx == 0:
                    ax.set_title(f'{genotype.replace("b9d2_", "").title()}', fontweight='bold')
                if col_idx == 0:
                    ax.set_ylabel(f'{pair}\n\n{metric_label}', fontsize=9)
                ax.tick_params(labelsize=8)
                ax.set_xlim(global_time_min, global_time_max)
                ax.set_ylim(global_metric_min, global_metric_max)
                continue

            # Plot individual trajectories
            color = GENOTYPE_COLORS[genotype]
            for traj in trajectories:
                ax.plot(traj['times'], traj['metrics'], alpha=0.25, linewidth=0.8, color=color)

            # Plot mean trajectory
            all_times = np.concatenate([t['times'] for t in trajectories])
            all_metrics = np.concatenate([t['metrics'] for t in trajectories])

            # Create mean by binning
            time_bins = np.arange(np.floor(all_times.min()), np.ceil(all_times.max()) + 1, 0.5)
            bin_means = []
            bin_times = []
            for i in range(len(time_bins) - 1):
                mask = (all_times >= time_bins[i]) & (all_times < time_bins[i+1])
                if mask.sum() > 0:
                    bin_means.append(all_metrics[mask].mean())
                    bin_times.append((time_bins[i] + time_bins[i+1]) / 2)

            if bin_times:
                ax.plot(bin_times, bin_means, color=color, linewidth=2.2, label='Mean', zorder=5)

            # Set labels and title
            ax.set_xlabel('Time (hpf)', fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(f'{pair}\n\n{metric_label}', fontsize=9)
            else:
                ax.set_ylabel('')

            if row_idx == 0:
                ax.set_title(f'{genotype.replace("b9d2_", "").title()} (n={n_embryos})',
                           fontweight='bold', fontsize=10)
            else:
                ax.set_title(f'n={n_embryos}', fontsize=9)

            ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
            ax.tick_params(labelsize=8)

            # Set aligned axes
            ax.set_xlim(global_time_min, global_time_max)
            ax.set_ylim(global_metric_min, global_metric_max)

            # Add legend only to top-left plot
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_genotypes_by_pair(df, pairs, metric_name, output_path):
    """Create a 1xN plot showing all three genotypes overlaid for each pair.

    Layout: 1 row × N columns, where each column is a pair.
    Each subplot contains all three genotypes (WT, Het, Homo) overlaid for direct comparison.
    """
    print(f"\nCreating genotypes-by-pair comparison plot for {len(pairs)} pairs...")

    fig, axes = plt.subplots(1, len(pairs), figsize=(15, 4.5))

    # Ensure axes is always iterable
    if len(pairs) == 1:
        axes = [axes]

    metric_label = get_metric_label(metric_name)
    fig.suptitle(f'{metric_label} Trajectories by Pair - All Genotypes Compared',
                 fontsize=14, fontweight='bold')

    # First pass: collect all data to determine global axis ranges
    all_pair_genotype_data = {}
    global_time_min, global_time_max = float('inf'), float('-inf')
    global_metric_min, global_metric_max = float('inf'), float('-inf')

    for pair in pairs:
        for genotype in GENOTYPE_ORDER:
            trajectories, embryo_ids, n_embryos = get_trajectories_for_pair_genotype(df, pair, genotype, metric_name)
            all_pair_genotype_data[(pair, genotype)] = (trajectories, embryo_ids, n_embryos)

            if trajectories is not None and n_embryos > 0:
                for traj in trajectories:
                    global_time_min = min(global_time_min, traj['times'].min())
                    global_time_max = max(global_time_max, traj['times'].max())
                    global_metric_min = min(global_metric_min, traj['metrics'].min())
                    global_metric_max = max(global_metric_max, traj['metrics'].max())

    # Add padding to metric axis
    metric_padding = (global_metric_max - global_metric_min) * 0.1
    global_metric_min -= metric_padding
    global_metric_max += metric_padding

    # Second pass: plot with aligned axes
    for col_idx, pair in enumerate(pairs):
        ax = axes[col_idx]

        # Plot all three genotypes on same axis
        for genotype in GENOTYPE_ORDER:
            trajectories, embryo_ids, n_embryos = all_pair_genotype_data[(pair, genotype)]

            if trajectories is None or n_embryos == 0:
                continue

            color = GENOTYPE_COLORS[genotype]

            # Plot individual trajectories (faded)
            for traj in trajectories:
                ax.plot(traj['times'], traj['metrics'], alpha=0.2, linewidth=0.8, color=color)

            # Plot mean trajectory
            all_times = np.concatenate([t['times'] for t in trajectories])
            all_metrics = np.concatenate([t['metrics'] for t in trajectories])

            # Create mean by binning
            time_bins = np.arange(np.floor(all_times.min()), np.ceil(all_times.max()) + 1, 0.5)
            bin_means = []
            bin_times = []
            for i in range(len(time_bins) - 1):
                mask = (all_times >= time_bins[i]) & (all_times < time_bins[i+1])
                if mask.sum() > 0:
                    bin_means.append(all_metrics[mask].mean())
                    bin_times.append((time_bins[i] + time_bins[i+1]) / 2)

            if bin_times:
                ax.plot(bin_times, bin_means, color=color, linewidth=2.5,
                       label=f'{genotype.replace("b9d2_", "").title()} (n={n_embryos})', zorder=5)

        ax.set_xlabel('Time (hpf)', fontsize=10)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_title(f'{pair}', fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.legend(fontsize=9, loc='upper right')

        # Set aligned axes
        ax.set_xlim(global_time_min, global_time_max)
        ax.set_ylim(global_metric_min, global_metric_max)
        ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_homozygous_across_pairs(df, pairs, metric_name, output_path):
    """Create a 1xN plot showing only homozygous genotype across all pairs."""
    print(f"\nCreating homozygous-only plot across {len(pairs)} pairs...")

    fig, axes = plt.subplots(1, len(pairs), figsize=(15, 4.5))

    # Ensure axes is always iterable
    if len(pairs) == 1:
        axes = [axes]

    metric_label = get_metric_label(metric_name)
    fig.suptitle(f'{metric_label} Trajectories - Homozygous (b9d2) Genotype Across Pairs',
                 fontsize=14, fontweight='bold')

    homozygous_genotype = 'b9d2_homozygous'
    color = GENOTYPE_COLORS[homozygous_genotype]

    # First pass: collect all data to determine axis ranges
    all_pair_data = {}
    global_time_min, global_time_max = float('inf'), float('-inf')
    global_metric_min, global_metric_max = float('inf'), float('-inf')

    for pair in pairs:
        trajectories, embryo_ids, n_embryos = get_trajectories_for_pair_genotype(df, pair, homozygous_genotype, metric_name)
        all_pair_data[pair] = (trajectories, embryo_ids, n_embryos)

        if trajectories is not None and n_embryos > 0:
            for traj in trajectories:
                global_time_min = min(global_time_min, traj['times'].min())
                global_time_max = max(global_time_max, traj['times'].max())
                global_metric_min = min(global_metric_min, traj['metrics'].min())
                global_metric_max = max(global_metric_max, traj['metrics'].max())

    # Add some padding to the metric axis
    metric_padding = (global_metric_max - global_metric_min) * 0.1
    global_metric_min -= metric_padding
    global_metric_max += metric_padding

    # Second pass: plot with aligned axes
    for ax_idx, pair in enumerate(pairs):
        ax = axes[ax_idx]

        trajectories, embryo_ids, n_embryos = all_pair_data[pair]

        if trajectories is None or n_embryos == 0:
            ax.text(0.5, 0.5, f'No data for\n{pair}',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='gray')
            ax.set_xlabel('Time (hpf)')
            ax.set_ylabel(metric_label)
            ax.set_title(f'{pair} (n={n_embryos})')
            ax.set_xlim(global_time_min, global_time_max)
            ax.set_ylim(global_metric_min, global_metric_max)
            continue

        # Plot individual trajectories
        for traj in trajectories:
            ax.plot(traj['times'], traj['metrics'], alpha=0.3, linewidth=1, color=color)

        # Plot mean trajectory
        all_times = np.concatenate([t['times'] for t in trajectories])
        all_metrics = np.concatenate([t['metrics'] for t in trajectories])

        # Create mean by binning
        time_bins = np.arange(np.floor(all_times.min()), np.ceil(all_times.max()) + 1, 0.5)
        bin_means = []
        bin_times = []
        for i in range(len(time_bins) - 1):
            mask = (all_times >= time_bins[i]) & (all_times < time_bins[i+1])
            if mask.sum() > 0:
                bin_means.append(all_metrics[mask].mean())
                bin_times.append((time_bins[i] + time_bins[i+1]) / 2)

        if bin_times:
            ax.plot(bin_times, bin_means, color=color, linewidth=2.5, label='Mean', zorder=5)

        ax.set_xlabel('Time (hpf)')
        ax.set_ylabel(metric_label)
        ax.set_title(f'{pair} (n={n_embryos})')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Set aligned axes
        ax.set_xlim(global_time_min, global_time_max)
        ax.set_ylim(global_metric_min, global_metric_max)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_summary_statistics(df, metric_name, tables_dir):
    """Create summary statistics table."""
    print("\nCreating summary statistics...")

    summary_rows = []

    for pair in sorted(df[PAIR_COL].unique()):
        for genotype in GENOTYPE_ORDER:
            filtered = df[(df[PAIR_COL] == pair) & (df[GENOTYPE_COL] == genotype)]

            if len(filtered) == 0:
                continue

            n_embryos = filtered[EMBRYO_ID_COL].nunique()
            n_timepoints = len(filtered)
            time_range = f"{filtered[TIME_COL].min():.1f}-{filtered[TIME_COL].max():.1f}"
            mean_metric = filtered[metric_name].mean()
            std_metric = filtered[metric_name].std()

            summary_rows.append({
                'pair': pair,
                'genotype': genotype,
                'n_embryos': n_embryos,
                'n_timepoints': n_timepoints,
                'time_range_hpf': time_range,
                f'mean_{metric_name}': mean_metric,
                f'std_{metric_name}': std_metric,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = tables_dir / 'pair_summary_statistics.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")
    print("\nSummary Statistics:")
    print(summary_df.to_string(index=False))

    return summary_df


def analyze_experiment_metric(experiment_id, metric_name):
    """Run analysis for a single experiment and metric combination."""
    print("\n" + "=" * 80)
    print(f"Analyzing Experiment {experiment_id} - Metric: {metric_name}")
    print("=" * 80)

    # Set up output directories for this combination
    output_dir = BASE_OUTPUT_DIR / f'output_{experiment_id}_{metric_name}'
    figures_dir = output_dir / 'figures'
    tables_dir = output_dir / 'tables'

    # Create directories
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_and_prepare_data(experiment_id, metric_name)

    # Get unique pairs
    pairs = sorted(df[PAIR_COL].unique())
    print(f"\nFound {len(pairs)} pair groups: {pairs}")

    # Create per-pair plots (3 columns, one per genotype)
    for pair in pairs:
        output_path = figures_dir / f'pair_{pair.replace("b9d2_", "").replace("_", "")}_all_genotypes.png'
        plot_trajectories_for_pair(df, pair, metric_name, output_path)

    # Create overview plot with all pairs side-by-side (using reusable function)
    overview_path = figures_dir / 'all_pairs_overview.png'
    metric_label = get_metric_label(metric_name)
    plot_faceted_trajectories(
        df, pairs, GENOTYPE_ORDER,
        row_col=PAIR_COL, col_col=GENOTYPE_COL,
        time_col=TIME_COL, metric_col=metric_name,
        embryo_id_col=EMBRYO_ID_COL,
        group_colors=GENOTYPE_COLORS,
        output_path=overview_path,
        title=f'{metric_label} Trajectories - All Pairs Overview',
        smooth_method=SMOOTH_METHOD,
        smooth_params=SMOOTH_PARAMS,
    )

    # Create collapsed plot: all genotypes by pair (1xN layout, using reusable function)
    genotypes_by_pair_path = figures_dir / 'genotypes_by_pair_comparison.png'
    plot_genotypes_overlaid(
        df, pairs,
        group_col=PAIR_COL, genotype_col=GENOTYPE_COL,
        time_col=TIME_COL, metric_col=metric_name,
        embryo_id_col=EMBRYO_ID_COL,
        genotype_order=GENOTYPE_ORDER, genotype_colors=GENOTYPE_COLORS,
        output_path=genotypes_by_pair_path,
        title=f'{metric_label} Trajectories by Pair - All Genotypes Compared',
        smooth_method=SMOOTH_METHOD,
        smooth_params=SMOOTH_PARAMS,
    )

    # Create homozygous-only plot across pairs
    homozygous_path = figures_dir / 'homozygous_across_pairs.png'
    plot_homozygous_across_pairs(df, pairs, metric_name, homozygous_path)

    # Create summary statistics
    summary_df = create_summary_statistics(df, metric_name, tables_dir)

    print("\n" + "=" * 80)
    print(f"Analysis complete for {experiment_id} - {metric_name}!")
    print(f"Figures saved to: {figures_dir}")
    print(f"Tables saved to: {tables_dir}")
    print("=" * 80)


def main():
    """Main analysis function - loops over all experiments and metrics."""
    print("\n" + "=" * 80)
    print("B9D2 PAIR ANALYSIS")
    print("Analyzing baseline deviation and total length across genotypes and pairs")
    print("=" * 80)
    print(f"Experiments: {EXPERIMENT_IDS}")
    print(f"Metrics: {METRICS}")
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
