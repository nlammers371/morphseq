#!/usr/bin/env python3
"""
Analyze curvature differences between pairs and genotypes for experiment 20251106.

This script loads the 20251106 data and creates faceted plots comparing
all genotypes (WT, Het, Homo) across each pair group to identify penetrance biases.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys

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

# Configuration
EXPERIMENT_ID = '202512'
METRIC_NAME = 'baseline_deviation_normalized'
TIME_COL = 'predicted_stage_hpf'
EMBRYO_ID_COL = 'embryo_id'
PAIR_COL = 'pair'
GENOTYPE_COL = 'genotype'

# Output paths
OUTPUT_DIR = Path(f'/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251113_curvature_pair_analysis/output_{EXPERIMENT_ID}')
FIGURES_DIR = OUTPUT_DIR / 'figures'
TABLES_DIR = OUTPUT_DIR / 'tables'

# Create directories if they don't exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Color mapping based on genotype suffix (independent of prefix)
GENOTYPE_SUFFIX_COLORS = {
    'wildtype': '#2E7D32',      # Green
    'heterozygous': '#FFA500',  # Orange
    'homozygous': '#D32F2F',    # Red
}

# Genotype suffix order
GENOTYPE_SUFFIX_ORDER = ['wildtype', 'heterozygous', 'homozygous']


def load_and_prepare_data(experiment_id):
    print(f"Loading data for experiment {experiment_id}...")

    df = _load_df03_format(experiment_id)

    # Filter for valid embryos (use_embryo_flag == 1)
    df = df[df['use_embryo_flag'] == 1].copy()

    # Drop rows with missing values in key columns
    df = df.dropna(subset=[EMBRYO_ID_COL, TIME_COL, METRIC_NAME, PAIR_COL, GENOTYPE_COL])

    print(f"Data shape: {df.shape}")
    print(f"Unique pairs: {df[PAIR_COL].unique()}")
    print(f"Genotypes: {df[GENOTYPE_COL].unique()}")

    return df


def get_genotype_suffix(genotype):
    """Extract the suffix (wildtype, heterozygous, homozygous) from full genotype name."""
    for suffix in GENOTYPE_SUFFIX_ORDER:
        if genotype.endswith(suffix):
            return suffix
    return genotype


def get_color_for_genotype(genotype):
    """Get color for a genotype based on its suffix."""
    suffix = get_genotype_suffix(genotype)
    return GENOTYPE_SUFFIX_COLORS.get(suffix, '#808080')  # Gray fallback


def get_all_genotypes_sorted(df):
    """Get all unique genotypes sorted by suffix order."""
    genotypes = df[GENOTYPE_COL].unique()
    return sorted(genotypes,
                 key=lambda g: (GENOTYPE_SUFFIX_ORDER.index(get_genotype_suffix(g))
                               if get_genotype_suffix(g) in GENOTYPE_SUFFIX_ORDER else 999))


def get_trajectories_for_pair_genotype(df, pair, genotype):
    """Extract trajectories for a specific pair and genotype."""
    filtered = df[(df[PAIR_COL] == pair) & (df[GENOTYPE_COL] == genotype)].copy()

    if len(filtered) == 0:
        return None, None, None

    # Group by embryo and get mean trajectory
    embryo_ids = filtered[EMBRYO_ID_COL].unique()
    trajectories = []

    for embryo_id in embryo_ids:
        embryo_data = filtered[filtered[EMBRYO_ID_COL] == embryo_id].sort_values(TIME_COL)
        if len(embryo_data) > 1:
            trajectories.append({
                'embryo_id': embryo_id,
                'times': embryo_data[TIME_COL].values,
                'metrics': embryo_data[METRIC_NAME].values,
            })

    return trajectories, embryo_ids, len(trajectories)


def plot_trajectories_for_pair(df, pair, output_path, global_time_min=None, global_time_max=None,
                               global_metric_min=None, global_metric_max=None):
    """Create a faceted plot for a pair showing all genotypes for that pair."""
    print(f"\nCreating plot for {pair}...")

    # Get unique genotypes for this pair, sorted by suffix order
    pair_genotypes = sorted(df[df[PAIR_COL] == pair][GENOTYPE_COL].unique(),
                           key=lambda g: (GENOTYPE_SUFFIX_ORDER.index(get_genotype_suffix(g))
                                         if get_genotype_suffix(g) in GENOTYPE_SUFFIX_ORDER else 999))

    n_genotypes = len(pair_genotypes) if pair_genotypes else 3  # fallback to 3
    fig, axes = plt.subplots(1, n_genotypes, figsize=(5*n_genotypes, 4.5))
    if n_genotypes == 1:
        axes = [axes]

    fig.suptitle(f'Curvature Trajectories - {pair}', fontsize=14, fontweight='bold')

    # Collect data to determine axis ranges if not provided
    if global_time_min is None:
        all_data = {}
        global_time_min, global_time_max = float('inf'), float('-inf')
        global_metric_min, global_metric_max = float('inf'), float('-inf')

        for genotype in pair_genotypes:
            trajectories, embryo_ids, n_embryos = get_trajectories_for_pair_genotype(df, pair, genotype)
            all_data[genotype] = (trajectories, embryo_ids, n_embryos)

            if trajectories is not None and n_embryos > 0:
                for traj in trajectories:
                    global_time_min = min(global_time_min, traj['times'].min())
                    global_time_max = max(global_time_max, traj['times'].max())
                    global_metric_min = min(global_metric_min, traj['metrics'].min())
                    global_metric_max = max(global_metric_max, traj['metrics'].max())

        # If no data found for any genotype, use default ranges
        if global_time_min == float('inf'):
            global_time_min, global_time_max = 0, 10
            global_metric_min, global_metric_max = -1, 1
    else:
        all_data = {}
        for genotype in pair_genotypes:
            all_data[genotype] = get_trajectories_for_pair_genotype(df, pair, genotype)

    # Add padding to metric axis
    metric_padding = (global_metric_max - global_metric_min) * 0.1
    global_metric_min -= metric_padding
    global_metric_max += metric_padding

    for ax_idx, genotype in enumerate(pair_genotypes):
        ax = axes[ax_idx]

        trajectories, embryo_ids, n_embryos = all_data[genotype]

        if trajectories is None or n_embryos == 0:
            genotype_label = get_genotype_suffix(genotype).replace("_", " ").title()
            ax.text(0.5, 0.5, f'No data for\n{genotype_label}',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='gray')
            ax.set_xlabel('Time (hpf)')
            ax.set_ylabel('Normalized Baseline Deviation')
            ax.set_title(f'{genotype_label} (n={n_embryos})')
            ax.set_xlim(global_time_min, global_time_max)
            ax.set_ylim(global_metric_min, global_metric_max)
            continue

        # Plot individual trajectories
        color = get_color_for_genotype(genotype)
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

        genotype_label = get_genotype_suffix(genotype).replace("_", " ").title()
        ax.set_xlabel('Time (hpf)')
        ax.set_ylabel('Normalized Baseline Deviation')
        ax.set_title(f'{genotype_label} (n={n_embryos})')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Set aligned axes
        ax.set_xlim(global_time_min, global_time_max)
        ax.set_ylim(global_metric_min, global_metric_max)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_all_pairs_overview(df, pairs, output_path):
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

    fig.suptitle('Curvature Trajectories - All Pairs Overview',
                 fontsize=16, fontweight='bold', y=0.995)

    # First pass: collect all data to determine global axis ranges
    all_data = {}
    global_time_min, global_time_max = float('inf'), float('-inf')
    global_metric_min, global_metric_max = float('inf'), float('-inf')

    for pair in pairs:
        for genotype in GENOTYPE_ORDER:
            trajectories, embryo_ids, n_embryos = get_trajectories_for_pair_genotype(df, pair, genotype)
            all_data[(pair, genotype)] = (trajectories, embryo_ids, n_embryos)

            if trajectories is not None and n_embryos > 0:
                for traj in trajectories:
                    global_time_min = min(global_time_min, traj['times'].min())
                    global_time_max = max(global_time_max, traj['times'].max())
                    global_metric_min = min(global_metric_min, traj['metrics'].min())
                    global_metric_max = max(global_metric_max, traj['metrics'].max())

    # If no data found, use default ranges
    if global_time_min == float('inf'):
        global_time_min, global_time_max = 0, 10
        global_metric_min, global_metric_max = -1, 1

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
                ax.set_ylabel('Normalized Baseline Deviation', fontsize=9)
                if row_idx == 0:
                    ax.set_title(f'{genotype.replace("cep290_", "").title()}', fontweight='bold')
                if col_idx == 0:
                    ax.set_ylabel(f'{pair}\n\nNormalized Baseline Deviation', fontsize=9)
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
                ax.set_ylabel(f'{pair}\n\nNormalized Baseline Deviation', fontsize=9)
            else:
                ax.set_ylabel('')

            if row_idx == 0:
                ax.set_title(f'{genotype.replace("cep290_", "").title()} (n={n_embryos})',
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


def plot_genotypes_by_pair(df, pairs, output_path):
    """Create a 1x3 plot showing all three genotypes overlaid for each pair.

    Layout: 1 row × 3 columns, where each column is a pair.
    Each subplot contains all three genotypes (WT, Het, Homo) overlaid for direct comparison.
    """
    print(f"\nCreating genotypes-by-pair comparison plot for {len(pairs)} pairs...")

    fig, axes = plt.subplots(1, len(pairs), figsize=(15, 4.5))

    # Ensure axes is always iterable
    if len(pairs) == 1:
        axes = [axes]

    fig.suptitle('Curvature Trajectories by Pair - All Genotypes Compared',
                 fontsize=14, fontweight='bold')

    # First pass: collect all data to determine global axis ranges
    all_pair_genotype_data = {}
    global_time_min, global_time_max = float('inf'), float('-inf')
    global_metric_min, global_metric_max = float('inf'), float('-inf')

    for pair in pairs:
        for genotype in GENOTYPE_ORDER:
            trajectories, embryo_ids, n_embryos = get_trajectories_for_pair_genotype(df, pair, genotype)
            all_pair_genotype_data[(pair, genotype)] = (trajectories, embryo_ids, n_embryos)

            if trajectories is not None and n_embryos > 0:
                for traj in trajectories:
                    global_time_min = min(global_time_min, traj['times'].min())
                    global_time_max = max(global_time_max, traj['times'].max())
                    global_metric_min = min(global_metric_min, traj['metrics'].min())
                    global_metric_max = max(global_metric_max, traj['metrics'].max())

    # If no data found, use default ranges
    if global_time_min == float('inf'):
        global_time_min, global_time_max = 0, 10
        global_metric_min, global_metric_max = -1, 1

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
                       label=f'{genotype.replace("cep290_", "").title()} (n={n_embryos})', zorder=5)

        ax.set_xlabel('Time (hpf)', fontsize=10)
        ax.set_ylabel('Normalized Baseline Deviation', fontsize=10)
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


def plot_homozygous_across_pairs(df, pairs, output_path):
    """Create a 1xN plot showing only homozygous genotype across all pairs."""
    print(f"\nCreating homozygous-only plot across {len(pairs)} pairs...")

    fig, axes = plt.subplots(1, len(pairs), figsize=(5*len(pairs), 4.5))

    # Ensure axes is always iterable
    if len(pairs) == 1:
        axes = [axes]

    fig.suptitle('Curvature Trajectories - Homozygous (cep290) Genotype Across Pairs',
                 fontsize=14, fontweight='bold')

    homozygous_genotype = 'cep290_homozygous'
    color = GENOTYPE_COLORS[homozygous_genotype]

    # First pass: collect all data to determine axis ranges
    all_pair_data = {}
    global_time_min, global_time_max = float('inf'), float('-inf')
    global_metric_min, global_metric_max = float('inf'), float('-inf')

    for pair in pairs:
        trajectories, embryo_ids, n_embryos = get_trajectories_for_pair_genotype(df, pair, homozygous_genotype)
        all_pair_data[pair] = (trajectories, embryo_ids, n_embryos)

        if trajectories is not None and n_embryos > 0:
            for traj in trajectories:
                global_time_min = min(global_time_min, traj['times'].min())
                global_time_max = max(global_time_max, traj['times'].max())
                global_metric_min = min(global_metric_min, traj['metrics'].min())
                global_metric_max = max(global_metric_max, traj['metrics'].max())

    # If no data found, use default ranges
    if global_time_min == float('inf'):
        global_time_min, global_time_max = 0, 10
        global_metric_min, global_metric_max = -1, 1

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
            ax.set_ylabel('Normalized Baseline Deviation')
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
        ax.set_ylabel('Normalized Baseline Deviation')
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


def create_summary_statistics(df):
    """Create summary statistics table."""
    print("\nCreating summary statistics...")

    summary_rows = []
    all_genotypes = get_all_genotypes_sorted(df)

    for pair in sorted(df[PAIR_COL].unique()):
        for genotype in all_genotypes:
            filtered = df[(df[PAIR_COL] == pair) & (df[GENOTYPE_COL] == genotype)]

            if len(filtered) == 0:
                continue

            n_embryos = filtered[EMBRYO_ID_COL].nunique()
            n_timepoints = len(filtered)
            time_range = f"{filtered[TIME_COL].min():.1f}-{filtered[TIME_COL].max():.1f}"
            mean_metric = filtered[METRIC_NAME].mean()
            std_metric = filtered[METRIC_NAME].std()

            summary_rows.append({
                'pair': pair,
                'genotype': genotype,
                'n_embryos': n_embryos,
                'n_timepoints': n_timepoints,
                'time_range_hpf': time_range,
                'mean_curvature': mean_metric,
                'std_curvature': std_metric,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = TABLES_DIR / 'pair_summary_statistics.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")
    print("\nSummary Statistics:")
    print(summary_df.to_string(index=False))

    return summary_df


def main():
    """Main analysis function."""
    print("=" * 60)
    print("Curvature Analysis for Experiment 20251106")
    print("Comparing penetrance across pair groups")
    print("=" * 60)

    # Load data
    df = load_and_prepare_data(EXPERIMENT_ID)

    # Get unique pairs
    pairs = sorted(df[PAIR_COL].unique())
    print(f"\nFound {len(pairs)} pair groups: {pairs}")

    # Create per-pair plots (columns for each genotype in that pair)
    for pair in pairs:
        output_path = FIGURES_DIR / f'pair_{pair}_all_genotypes.png'
        plot_trajectories_for_pair(df, pair, output_path)

    # Create summary statistics
    summary_df = create_summary_statistics(df)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Tables saved to: {TABLES_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
