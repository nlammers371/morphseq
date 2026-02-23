#!/usr/bin/env python3
"""
B9D2 Pair Analysis Script (QC Staged) - Generates both PNG and interactive Plotly plots.

This script analyzes b9d2 data from build04 output (qc_staged CSV files), creating
trajectory plots comparing all genotypes (WT, Het, Homo) across pair groups for multiple metrics.

DATA SOURCE:
- Uses build04 output (qc_staged) from morphseq_playground/metadata/build04_output/
- Loads pre-merged single CSV files (no separate curvature/metadata merge needed)
- Filters for use_embryo_flag == 1

FEATURES:
- Dual-backend support: Generate PNG (matplotlib) and/or HTML (plotly) plots
- Descriptive filenames including experiment ID and abbreviated metric names
- Interactive plots show embryo_id on hover (plotly only)
- Gaussian smoothing support for trajectories

OUTPUT CONTROL:
Set GENERATE_PNG and GENERATE_PLOTLY flags in main() to control which formats are created:
- GENERATE_PNG=True: Creates static PNG plots (good for quick viewing)
- GENERATE_PLOTLY=True: Creates interactive HTML plots (hover to see embryo IDs)
- Both=True: Creates both formats side-by-side

HELPER FUNCTIONS:
- get_metric_abbreviation(): Converts long metric names to short versions for filenames
- get_output_path(): Generates standardized output paths with experiment ID and metric
- plot_*(..., plotly=True, png=True): Each plotting function accepts format flags

USAGE:
1. Configure EXPERIMENT_IDS and METRICS at the top
2. Set GENERATE_PNG and GENERATE_PLOTLY in main()
3. Run: python analyze_b9d2_pairs_qc_staged.py

OUTPUT STRUCTURE:
results/mcolon/20251209/
  output_{exp_id}_{metric_abbrev}/
    figures/
      {exp_id}_{metric_abbrev}_pair_{pair}_all_genotypes.png
      {exp_id}_{metric_abbrev}_pair_{pair}_all_genotypes.html
      {exp_id}_{metric_abbrev}_all_pairs_overview.png
      {exp_id}_{metric_abbrev}_all_pairs_overview.html
      ...
    tables/
      pair_summary_statistics.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
from scipy.ndimage import gaussian_filter1d

# Add src to path
sys.path.insert(0, '/net/trapnell/vol1/home/mdcolon/proj/morphseq')

from src.analyze.trajectory_analysis.data_loading import _load_qc_staged
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

# Base output directory - new folder for qc_staged analysis
BASE_OUTPUT_DIR = Path('/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251209')

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
SMOOTH_METHOD = 'gaussian'
SMOOTH_PARAMS = {'sigma': 1.5}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_metric_abbreviation(metric_name):
    """Get abbreviated metric name for filenames.

    Args:
        metric_name: Full metric name (e.g., 'baseline_deviation_normalized')

    Returns:
        str: Abbreviated metric name (e.g., 'baseline_dev')
    """
    abbrev_map = {
        'baseline_deviation_normalized': 'baseline_dev',
        'total_length_um': 'length',
        'surface_area_um': 'surface_area',
    }
    return abbrev_map.get(metric_name, metric_name)


def get_output_path(figures_dir, experiment_id, metric_name, plot_type, extension, **kwargs):
    """Generate standardized output path with experiment ID and metric.

    This function centralizes filename generation to ensure consistent naming
    across all plots. The format includes experiment ID and abbreviated metric
    name for easy identification.

    Args:
        figures_dir: Path to figures output directory
        experiment_id: Experiment identifier (e.g., '20251104')
        metric_name: Full metric name (e.g., 'baseline_deviation_normalized')
        plot_type: One of 'per_pair', 'overview', 'genotypes_by_pair', 'homozygous'
        extension: File extension ('png' or 'html')
        **kwargs: Additional args like 'pair' for per_pair plots

    Returns:
        Path: Full output path with standardized naming convention

    Example:
        >>> get_output_path(fig_dir, '20251104', 'baseline_deviation_normalized',
        ...                 'per_pair', 'png', pair='b9d2_pair1')
        Path('.../20251104_baseline_dev_pair_b9d2_pair1_all_genotypes.png')
    """
    metric_abbrev = get_metric_abbreviation(metric_name)

    # Build filename based on plot type
    if plot_type == 'per_pair':
        pair = kwargs['pair']
        filename = f'{experiment_id}_{metric_abbrev}_pair_{pair}_all_genotypes.{extension}'
    elif plot_type == 'overview':
        filename = f'{experiment_id}_{metric_abbrev}_all_pairs_overview.{extension}'
    elif plot_type == 'genotypes_by_pair':
        filename = f'{experiment_id}_{metric_abbrev}_genotypes_by_pair.{extension}'
    elif plot_type == 'homozygous':
        filename = f'{experiment_id}_{metric_abbrev}_homozygous_across_pairs.{extension}'
    elif plot_type == 'scatter':
        filename = f'{experiment_id}_length_vs_curvature_scatter.{extension}'
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")

    return figures_dir / filename


def get_metric_label(metric_name):
    """Get human-readable label for a metric."""
    return METRIC_DISPLAY_NAMES.get(metric_name, metric_name)


# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_and_prepare_data(experiment_id, metric_name):
    """Load data and prepare for analysis."""
    print(f"Loading data for experiment {experiment_id}, metric {metric_name}...")

    df = _load_qc_staged(experiment_id)

    # Filter for valid embryos
    df = df[df['use_embryo_flag'] == 1].copy()

    # Drop rows with missing values in key columns
    df = df.dropna(subset=[EMBRYO_ID_COL, TIME_COL, metric_name, PAIR_COL, GENOTYPE_COL])

    print(f"Data shape: {df.shape}")
    print(f"Unique pairs: {df[PAIR_COL].unique()}")
    print(f"Genotypes: {df[GENOTYPE_COL].unique()}")

    return df


def get_trajectories_for_pair_genotype(df, pair, genotype, metric_name):
    """Extract trajectories for a specific pair and genotype with smoothing.

    Enhanced to include genotype and pair in trajectory dict for hover tooltips.
    """
    filtered = df[(df[PAIR_COL] == pair) & (df[GENOTYPE_COL] == genotype)].copy()

    if len(filtered) == 0:
        return None, None, None

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
                'genotype': genotype,
                'pair': pair,
            })

    return trajectories, embryo_ids, len(trajectories)


# ============================================================================
# PLOTTING FUNCTIONS - DUAL BACKEND SUPPORT
# ============================================================================

def plot_trajectories_for_pair(df, pair, metric_name, experiment_id, figures_dir,
                               plotly=False, png=False,
                               global_time_min=None, global_time_max=None,
                               global_metric_min=None, global_metric_max=None):
    """Create 1x3 faceted plot for a pair showing all three genotypes.

    Supports dual backends: matplotlib (PNG) and plotly (interactive HTML).

    Args:
        df: DataFrame with trajectory data
        pair: Pair identifier (e.g., 'b9d2_pair1')
        metric_name: Full metric name
        experiment_id: Experiment identifier
        figures_dir: Output directory for figures
        plotly: If True, generate interactive HTML plot
        png: If True, generate static PNG plot
        global_time_min/max: Optional axis limits for time
        global_metric_min/max: Optional axis limits for metric

    Outputs:
        - If png=True: {exp_id}_{metric_abbrev}_pair_{pair}_all_genotypes.png
        - If plotly=True: {exp_id}_{metric_abbrev}_pair_{pair}_all_genotypes.html
    """
    print(f"\nCreating plot for {pair}...")
    metric_label = get_metric_label(metric_name)

    # ========================================================================
    # SHARED DATA PREPARATION (runs once regardless of backend)
    # ========================================================================

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

    # ========================================================================
    # PNG RENDERING (matplotlib)
    # ========================================================================

    if png:
        output_path = get_output_path(figures_dir, experiment_id, metric_name,
                                     'per_pair', 'png', pair=pair)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        fig.suptitle(f'{metric_label} Trajectories - {pair}', fontsize=14, fontweight='bold')

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
        print(f"Saved PNG: {output_path}")
        plt.close()

    # ========================================================================
    # PLOTLY RENDERING (interactive HTML)
    # ========================================================================

    if plotly:
        output_path = get_output_path(figures_dir, experiment_id, metric_name,
                                     'per_pair', 'html', pair=pair)

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=[f'{g.replace("b9d2_", "").title()}' for g in GENOTYPE_ORDER],
            specs=[[{}, {}, {}]]
        )

        # Plot for each genotype
        for col_idx, genotype in enumerate(GENOTYPE_ORDER, start=1):
            trajectories, embryo_ids, n_embryos = all_data[genotype]
            color = GENOTYPE_COLORS[genotype]

            if trajectories is None or n_embryos == 0:
                fig.add_annotation(
                    text=f'No data for<br>{genotype.replace("b9d2_", "")}',
                    xref=f'x{col_idx}', yref=f'y{col_idx}',
                    x=(global_time_min + global_time_max) / 2,
                    y=(global_metric_min + global_metric_max) / 2,
                    showarrow=False, font=dict(size=12, color='gray'),
                    row=1, col=col_idx
                )
            else:
                # Plot individual trajectories
                for traj in trajectories:
                    fig.add_trace(go.Scatter(
                        x=traj['times'],
                        y=traj['metrics'],
                        mode='lines',
                        line=dict(color=color, width=1),
                        opacity=0.3,
                        hovertemplate=(
                            '<b>Embryo:</b> %{customdata[0]}<br>'
                            '<b>Time:</b> %{x:.2f} hpf<br>'
                            f'<b>{metric_label}:</b> %{{y:.4f}}<br>'
                            '<b>Genotype:</b> %{customdata[1]}<br>'
                            '<b>Pair:</b> %{customdata[2]}<extra></extra>'
                        ),
                        customdata=np.column_stack((
                            [traj['embryo_id']] * len(traj['times']),
                            [traj['genotype']] * len(traj['times']),
                            [traj['pair']] * len(traj['times']),
                        )),
                        showlegend=False,
                        name=traj['embryo_id'],
                    ), row=1, col=col_idx)

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
                    fig.add_trace(go.Scatter(
                        x=bin_times,
                        y=bin_means,
                        mode='lines',
                        line=dict(color=color, width=3),
                        hovertemplate=(
                            f'<b>Mean {metric_label}</b><br>'
                            '<b>Time:</b> %{x:.2f} hpf<br>'
                            '<b>Value:</b> %{y:.4f}<extra></extra>'
                        ),
                        showlegend=(col_idx == 1),
                        name='Mean',
                    ), row=1, col=col_idx)

            # Update subplot axes
            fig.update_xaxes(title_text='Time (hpf)', row=1, col=col_idx)
            if col_idx == 1:
                fig.update_yaxes(title_text=metric_label, row=1, col=col_idx)

        # Update layout
        fig.update_xaxes(range=[global_time_min, global_time_max])
        fig.update_yaxes(range=[global_metric_min, global_metric_max])

        fig.update_layout(
            title=f'{metric_label} Trajectories - {pair}',
            height=500,
            width=1400,
            hovermode='closest',
            showlegend=True,
            legend=dict(x=1.02, y=1),
        )

        fig.write_html(output_path)
        print(f"Saved HTML: {output_path}")


def plot_all_pairs_overview(df, pairs, metric_name, experiment_id, figures_dir,
                            plotly=False, png=False):
    """Create a comprehensive overview plot with all pairs.

    Supports dual backends: matplotlib (PNG) and plotly (interactive HTML).

    Layout: Rows = pairs, Columns = genotypes
    All axes are aligned for comparison.

    Args:
        df: DataFrame with trajectory data
        pairs: List of pair identifiers
        metric_name: Full metric name
        experiment_id: Experiment identifier
        figures_dir: Output directory for figures
        plotly: If True, generate interactive HTML plot
        png: If True, generate static PNG plot

    Outputs:
        - If png=True: {exp_id}_{metric_abbrev}_all_pairs_overview.png
        - If plotly=True: {exp_id}_{metric_abbrev}_all_pairs_overview.html
    """
    print(f"\nCreating overview plot with all {len(pairs)} pairs...")
    metric_label = get_metric_label(metric_name)

    n_pairs = len(pairs)
    n_genotypes = 3

    # ========================================================================
    # SHARED DATA PREPARATION
    # ========================================================================

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

    # ========================================================================
    # PNG RENDERING
    # ========================================================================

    if png:
        output_path = get_output_path(figures_dir, experiment_id, metric_name,
                                     'overview', 'png')

        fig, axes = plt.subplots(n_pairs, n_genotypes, figsize=(15, 4.5 * n_pairs))

        # Ensure axes is always 2D even with single row
        if n_pairs == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(f'{metric_label} Trajectories - All Pairs Overview',
                     fontsize=16, fontweight='bold', y=0.995)

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
        print(f"Saved PNG: {output_path}")
        plt.close()

    # ========================================================================
    # PLOTLY RENDERING
    # ========================================================================

    if plotly:
        output_path = get_output_path(figures_dir, experiment_id, metric_name,
                                     'overview', 'html')

        fig = make_subplots(
            rows=n_pairs, cols=n_genotypes,
            subplot_titles=[f'{g.replace("b9d2_", "").title()}' for g in GENOTYPE_ORDER] * n_pairs,
            specs=[[{} for _ in range(n_genotypes)] for _ in range(n_pairs)]
        )

        # Plot
        for row_idx, pair in enumerate(pairs, start=1):
            for col_idx, genotype in enumerate(GENOTYPE_ORDER, start=1):
                trajectories, embryo_ids, n_embryos = all_data[(pair, genotype)]
                color = GENOTYPE_COLORS[genotype]

                if trajectories is None or n_embryos == 0:
                    fig.add_annotation(
                        text='No data',
                        xref=f'x{col_idx}', yref=f'y{col_idx}',
                        showarrow=False, font=dict(size=10, color='lightgray'),
                        row=row_idx, col=col_idx
                    )
                else:
                    # Individual trajectories
                    for traj in trajectories:
                        fig.add_trace(go.Scatter(
                            x=traj['times'],
                            y=traj['metrics'],
                            mode='lines',
                            line=dict(color=color, width=0.8),
                            opacity=0.25,
                            hovertemplate=(
                                '<b>Embryo:</b> %{customdata[0]}<br>'
                                '<b>Time:</b> %{x:.2f} hpf<br>'
                                f'<b>{metric_label}:</b> %{{y:.4f}}<br>'
                                '<extra></extra>'
                            ),
                            customdata=np.column_stack((
                                [traj['embryo_id']] * len(traj['times']),
                            )),
                            showlegend=False,
                        ), row=row_idx, col=col_idx)

                    # Mean trajectory
                    all_times = np.concatenate([t['times'] for t in trajectories])
                    all_metrics = np.concatenate([t['metrics'] for t in trajectories])

                    time_bins = np.arange(np.floor(all_times.min()), np.ceil(all_times.max()) + 1, 0.5)
                    bin_means = []
                    bin_times = []
                    for i in range(len(time_bins) - 1):
                        mask = (all_times >= time_bins[i]) & (all_times < time_bins[i+1])
                        if mask.sum() > 0:
                            bin_means.append(all_metrics[mask].mean())
                            bin_times.append((time_bins[i] + time_bins[i+1]) / 2)

                    if bin_times:
                        fig.add_trace(go.Scatter(
                            x=bin_times,
                            y=bin_means,
                            mode='lines',
                            line=dict(color=color, width=2),
                            showlegend=(row_idx == 1 and col_idx == 1),
                            name='Mean',
                        ), row=row_idx, col=col_idx)

                fig.update_xaxes(title_text='Time (hpf)' if row_idx == n_pairs else '', row=row_idx, col=col_idx)
                if col_idx == 1:
                    fig.update_yaxes(title_text=f'{pair}<br>{metric_label}', row=row_idx, col=col_idx)

        # Apply global axes
        for row in range(1, n_pairs + 1):
            for col in range(1, n_genotypes + 1):
                fig.update_xaxes(range=[global_time_min, global_time_max], row=row, col=col)
                fig.update_yaxes(range=[global_metric_min, global_metric_max], row=row, col=col)

        fig.update_layout(
            title=f'{metric_label} Trajectories - All Pairs Overview',
            height=250 * n_pairs,
            width=1400,
            hovermode='closest',
            showlegend=True,
        )

        fig.write_html(output_path)
        print(f"Saved HTML: {output_path}")


def plot_genotypes_by_pair(df, pairs, metric_name, experiment_id, figures_dir,
                           plotly=False, png=False):
    """Create a 1xN plot showing all three genotypes overlaid for each pair.

    Supports dual backends: matplotlib (PNG) and plotly (interactive HTML).

    Layout: 1 row × N columns, where each column is a pair.
    Each subplot contains all three genotypes (WT, Het, Homo) overlaid for direct comparison.

    Args:
        df: DataFrame with trajectory data
        pairs: List of pair identifiers
        metric_name: Full metric name
        experiment_id: Experiment identifier
        figures_dir: Output directory for figures
        plotly: If True, generate interactive HTML plot
        png: If True, generate static PNG plot

    Outputs:
        - If png=True: {exp_id}_{metric_abbrev}_genotypes_by_pair.png
        - If plotly=True: {exp_id}_{metric_abbrev}_genotypes_by_pair.html
    """
    print(f"\nCreating genotypes-by-pair comparison plot for {len(pairs)} pairs...")
    metric_label = get_metric_label(metric_name)

    # ========================================================================
    # SHARED DATA PREPARATION
    # ========================================================================

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

    # ========================================================================
    # PNG RENDERING
    # ========================================================================

    if png:
        output_path = get_output_path(figures_dir, experiment_id, metric_name,
                                     'genotypes_by_pair', 'png')

        fig, axes = plt.subplots(1, len(pairs), figsize=(15, 4.5))

        # Ensure axes is always iterable
        if len(pairs) == 1:
            axes = [axes]

        fig.suptitle(f'{metric_label} Trajectories by Pair - All Genotypes Compared',
                     fontsize=14, fontweight='bold')

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
        print(f"Saved PNG: {output_path}")
        plt.close()

    # ========================================================================
    # PLOTLY RENDERING
    # ========================================================================

    if plotly:
        output_path = get_output_path(figures_dir, experiment_id, metric_name,
                                     'genotypes_by_pair', 'html')

        fig = make_subplots(
            rows=1, cols=len(pairs),
            subplot_titles=[f'{p}' for p in pairs],
        )

        # Second pass: plot
        for col_idx, pair in enumerate(pairs, start=1):
            for genotype in GENOTYPE_ORDER:
                trajectories, embryo_ids, n_embryos = all_pair_genotype_data[(pair, genotype)]
                color = GENOTYPE_COLORS[genotype]

                if trajectories is None or n_embryos == 0:
                    continue

                # Individual trajectories (faded)
                for traj in trajectories:
                    fig.add_trace(go.Scatter(
                        x=traj['times'],
                        y=traj['metrics'],
                        mode='lines',
                        line=dict(color=color, width=0.8),
                        opacity=0.2,
                        hovertemplate=(
                            '<b>Embryo:</b> %{customdata[0]}<br>'
                            '<b>Time:</b> %{x:.2f} hpf<br>'
                            f'<b>{metric_label}:</b> %{{y:.4f}}<br>'
                            '<extra></extra>'
                        ),
                        customdata=np.column_stack((
                            [traj['embryo_id']] * len(traj['times']),
                        )),
                        showlegend=False,
                    ), row=1, col=col_idx)

                # Mean trajectory
                all_times = np.concatenate([t['times'] for t in trajectories])
                all_metrics = np.concatenate([t['metrics'] for t in trajectories])

                time_bins = np.arange(np.floor(all_times.min()), np.ceil(all_times.max()) + 1, 0.5)
                bin_means = []
                bin_times = []
                for i in range(len(time_bins) - 1):
                    mask = (all_times >= time_bins[i]) & (all_times < time_bins[i+1])
                    if mask.sum() > 0:
                        bin_means.append(all_metrics[mask].mean())
                        bin_times.append((time_bins[i] + time_bins[i+1]) / 2)

                if bin_times:
                    fig.add_trace(go.Scatter(
                        x=bin_times,
                        y=bin_means,
                        mode='lines',
                        line=dict(color=color, width=2.5),
                        name=f'{genotype.replace("b9d2_", "").title()} (n={n_embryos})',
                        showlegend=(col_idx == 1),
                    ), row=1, col=col_idx)

            fig.update_xaxes(title_text='Time (hpf)', row=1, col=col_idx)
            if col_idx == 1:
                fig.update_yaxes(title_text=metric_label, row=1, col=col_idx)

        # Apply global axes
        for col in range(1, len(pairs) + 1):
            fig.update_xaxes(range=[global_time_min, global_time_max], row=1, col=col)
            fig.update_yaxes(range=[global_metric_min, global_metric_max], row=1, col=col)

        fig.update_layout(
            title=f'{metric_label} Trajectories by Pair - All Genotypes Compared',
            height=500,
            width=1400,
            hovermode='closest',
            showlegend=True,
            legend=dict(x=1.02, y=1),
        )

        fig.write_html(output_path)
        print(f"Saved HTML: {output_path}")


def plot_homozygous_across_pairs(df, pairs, metric_name, experiment_id, figures_dir,
                                 plotly=False, png=False):
    """Create a 1xN plot showing only homozygous genotype across all pairs.

    Supports dual backends: matplotlib (PNG) and plotly (interactive HTML).

    Args:
        df: DataFrame with trajectory data
        pairs: List of pair identifiers
        metric_name: Full metric name
        experiment_id: Experiment identifier
        figures_dir: Output directory for figures
        plotly: If True, generate interactive HTML plot
        png: If True, generate static PNG plot

    Outputs:
        - If png=True: {exp_id}_{metric_abbrev}_homozygous_across_pairs.png
        - If plotly=True: {exp_id}_{metric_abbrev}_homozygous_across_pairs.html
    """
    print(f"\nCreating homozygous-only plot across {len(pairs)} pairs...")
    metric_label = get_metric_label(metric_name)

    homozygous_genotype = 'b9d2_homozygous'
    color = GENOTYPE_COLORS[homozygous_genotype]

    # ========================================================================
    # SHARED DATA PREPARATION
    # ========================================================================

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

    # ========================================================================
    # PNG RENDERING
    # ========================================================================

    if png:
        output_path = get_output_path(figures_dir, experiment_id, metric_name,
                                     'homozygous', 'png')

        fig, axes = plt.subplots(1, len(pairs), figsize=(15, 4.5))

        # Ensure axes is always iterable
        if len(pairs) == 1:
            axes = [axes]

        fig.suptitle(f'{metric_label} Trajectories - Homozygous (b9d2) Genotype Across Pairs',
                     fontsize=14, fontweight='bold')

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
        print(f"Saved PNG: {output_path}")
        plt.close()

    # ========================================================================
    # PLOTLY RENDERING
    # ========================================================================

    if plotly:
        output_path = get_output_path(figures_dir, experiment_id, metric_name,
                                     'homozygous', 'html')

        fig = make_subplots(
            rows=1, cols=len(pairs),
            subplot_titles=[f'{p}' for p in pairs],
        )

        # Second pass: plot
        for col_idx, pair in enumerate(pairs, start=1):
            trajectories, embryo_ids, n_embryos = all_pair_data[pair]

            if trajectories is None or n_embryos == 0:
                fig.add_annotation(
                    text=f'No data for {pair}',
                    xref=f'x{col_idx}', yref=f'y{col_idx}',
                    showarrow=False, font=dict(size=10, color='gray'),
                    row=1, col=col_idx
                )
            else:
                # Individual trajectories
                for traj in trajectories:
                    fig.add_trace(go.Scatter(
                        x=traj['times'],
                        y=traj['metrics'],
                        mode='lines',
                        line=dict(color=color, width=1),
                        opacity=0.3,
                        hovertemplate=(
                            '<b>Embryo:</b> %{customdata[0]}<br>'
                            '<b>Time:</b> %{x:.2f} hpf<br>'
                            f'<b>{metric_label}:</b> %{{y:.4f}}<br>'
                            '<extra></extra>'
                        ),
                        customdata=np.column_stack((
                            [traj['embryo_id']] * len(traj['times']),
                        )),
                        showlegend=False,
                    ), row=1, col=col_idx)

                # Mean trajectory
                all_times = np.concatenate([t['times'] for t in trajectories])
                all_metrics = np.concatenate([t['metrics'] for t in trajectories])

                time_bins = np.arange(np.floor(all_times.min()), np.ceil(all_times.max()) + 1, 0.5)
                bin_means = []
                bin_times = []
                for i in range(len(time_bins) - 1):
                    mask = (all_times >= time_bins[i]) & (all_times < time_bins[i+1])
                    if mask.sum() > 0:
                        bin_means.append(all_metrics[mask].mean())
                        bin_times.append((time_bins[i] + time_bins[i+1]) / 2)

                if bin_times:
                    fig.add_trace(go.Scatter(
                        x=bin_times,
                        y=bin_means,
                        mode='lines',
                        line=dict(color=color, width=2.5),
                        name='Mean' if col_idx == 1 else '',
                        showlegend=(col_idx == 1),
                    ), row=1, col=col_idx)

            fig.update_xaxes(title_text='Time (hpf)', row=1, col=col_idx)
            if col_idx == 1:
                fig.update_yaxes(title_text=metric_label, row=1, col=col_idx)

        # Apply global axes
        for col in range(1, len(pairs) + 1):
            fig.update_xaxes(range=[global_time_min, global_time_max], row=1, col=col)
            fig.update_yaxes(range=[global_metric_min, global_metric_max], row=1, col=col)

        fig.update_layout(
            title=f'{metric_label} Trajectories - Homozygous (b9d2) Across Pairs',
            height=500,
            width=1400,
            hovermode='closest',
            showlegend=True,
        )

        fig.write_html(output_path)
        print(f"Saved HTML: {output_path}")


def plot_length_vs_curvature_scatter(df, pairs, experiment_id, figures_dir,
                                     plotly=False, png=False):
    """
    Create scatter plot of total_length_um vs baseline_deviation_normalized.

    Layout: Rows = pairs, Columns = genotypes
    Each subplot shows length vs curvature for that pair-genotype combination
    with all timepoints as individual points and a smoothed trend line (binned means).

    Args:
        df: DataFrame with both total_length_um and baseline_deviation_normalized
        pairs: List of pair identifiers
        experiment_id: Experiment identifier
        figures_dir: Output directory for figures
        plotly: If True, generate interactive HTML plot
        png: If True, generate static PNG plot

    Outputs:
        - PNG: {exp_id}_length_vs_curvature_scatter.png
        - HTML: {exp_id}_length_vs_curvature_scatter.html
    """
    print(f"\nCreating length vs curvature scatter plot...")

    n_pairs = len(pairs)
    n_genotypes = 3

    # ========================================================================
    # SHARED DATA PREPARATION
    # ========================================================================

    # First pass: collect all data to determine global axis ranges
    all_data = {}
    global_length_min, global_length_max = float('inf'), float('-inf')
    global_curv_min, global_curv_max = float('inf'), float('-inf')

    for pair in pairs:
        for genotype in GENOTYPE_ORDER:
            data = df[(df[PAIR_COL] == pair) & (df[GENOTYPE_COL] == genotype)].copy()
            data = data.dropna(subset=['total_length_um', 'baseline_deviation_normalized'])
            all_data[(pair, genotype)] = data

            if len(data) > 0:
                global_length_min = min(global_length_min, data['total_length_um'].min())
                global_length_max = max(global_length_max, data['total_length_um'].max())
                global_curv_min = min(global_curv_min, data['baseline_deviation_normalized'].min())
                global_curv_max = max(global_curv_max, data['baseline_deviation_normalized'].max())

    # Add padding to axes
    length_padding = (global_length_max - global_length_min) * 0.1
    curv_padding = (global_curv_max - global_curv_min) * 0.1
    global_length_min -= length_padding
    global_length_max += length_padding
    global_curv_min -= curv_padding
    global_curv_max += curv_padding

    # ========================================================================
    # PNG RENDERING
    # ========================================================================

    if png:
        output_path = get_output_path(figures_dir, experiment_id, 'baseline_deviation_normalized',
                                     'scatter', 'png')

        fig, axes = plt.subplots(n_pairs, n_genotypes, figsize=(15, 4.5 * n_pairs))

        # Ensure axes is always 2D
        if n_pairs == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle('Total Length vs Normalized Baseline Deviation - All Pairs',
                     fontsize=16, fontweight='bold', y=0.995)

        for row_idx, pair in enumerate(pairs):
            for col_idx, genotype in enumerate(GENOTYPE_ORDER):
                ax = axes[row_idx, col_idx]
                data = all_data[(pair, genotype)]

                if len(data) == 0:
                    ax.text(0.5, 0.5, 'No data',
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=10, color='lightgray')
                    ax.set_xlabel('Total Length (µm)', fontsize=9)
                    ax.set_ylabel('Normalized Baseline Deviation', fontsize=9)
                    if row_idx == 0:
                        ax.set_title(f'{genotype.replace("b9d2_", "").title()}', fontweight='bold')
                    if col_idx == 0:
                        ax.set_ylabel(f'{pair}\n\nNorm. Baseline Dev.', fontsize=9)
                    ax.tick_params(labelsize=8)
                    ax.set_xlim(global_length_min, global_length_max)
                    ax.set_ylim(global_curv_min, global_curv_max)
                    continue

                # Plot scatter of all timepoints
                color = GENOTYPE_COLORS[genotype]
                ax.scatter(data['total_length_um'], data['baseline_deviation_normalized'],
                          alpha=0.25, s=15, color=color, zorder=3)

                # Plot individual embryo trajectories
                for embryo_id in data['embryo_id'].unique():
                    embryo_data = data[data['embryo_id'] == embryo_id].copy()
                    # Sort by time to connect points in temporal order
                    embryo_data = embryo_data.sort_values(TIME_COL)

                    # Plot line connecting this embryo's timepoints
                    ax.plot(embryo_data['total_length_um'],
                           embryo_data['baseline_deviation_normalized'],
                           color=color, alpha=0.3, linewidth=1, zorder=2)

                # Set labels and title
                ax.set_xlabel('Total Length (µm)', fontsize=9)
                if col_idx == 0:
                    ax.set_ylabel(f'{pair}\n\nNorm. Baseline Dev.', fontsize=9)
                else:
                    ax.set_ylabel('')

                if row_idx == 0:
                    ax.set_title(f'{genotype.replace("b9d2_", "").title()} (n={len(data)})',
                               fontweight='bold', fontsize=10)
                else:
                    ax.set_title(f'n={len(data)}', fontsize=9)

                ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
                ax.tick_params(labelsize=8)

                # Set aligned axes
                ax.set_xlim(global_length_min, global_length_max)
                ax.set_ylim(global_curv_min, global_curv_max)

                # Add legend only to top-left plot
                if row_idx == 0 and col_idx == 0:
                    ax.legend(fontsize=8, loc='upper right')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved PNG: {output_path}")
        plt.close()

    # ========================================================================
    # PLOTLY RENDERING
    # ========================================================================

    if plotly:
        output_path = get_output_path(figures_dir, experiment_id, 'baseline_deviation_normalized',
                                     'scatter', 'html')

        fig = make_subplots(
            rows=n_pairs, cols=n_genotypes,
            subplot_titles=[f'{g.replace("b9d2_", "").title()}' for g in GENOTYPE_ORDER] * n_pairs,
            specs=[[{} for _ in range(n_genotypes)] for _ in range(n_pairs)]
        )

        for row_idx, pair in enumerate(pairs, start=1):
            for col_idx, genotype in enumerate(GENOTYPE_ORDER, start=1):
                data = all_data[(pair, genotype)]
                color = GENOTYPE_COLORS[genotype]

                if len(data) == 0:
                    fig.add_annotation(
                        text='No data',
                        xref=f'x{col_idx}', yref=f'y{col_idx}',
                        showarrow=False, font=dict(size=10, color='lightgray'),
                        row=row_idx, col=col_idx
                    )
                else:
                    # Scatter traces
                    fig.add_trace(go.Scatter(
                        x=data['total_length_um'],
                        y=data['baseline_deviation_normalized'],
                        mode='markers',
                        marker=dict(color=color, size=5, opacity=0.4),
                        hovertemplate=(
                            '<b>Embryo:</b> %{customdata[0]}<br>'
                            '<b>Length:</b> %{x:.1f} µm<br>'
                            '<b>Curvature:</b> %{y:.4f}<br>'
                            '<extra></extra>'
                        ),
                        customdata=np.column_stack((
                            data['embryo_id'].values,
                        )),
                        showlegend=(row_idx == 1 and col_idx == 1),
                        name='Data',
                    ), row=row_idx, col=col_idx)

                    # Plot individual embryo trajectories
                    for embryo_id in data['embryo_id'].unique():
                        embryo_data = data[data['embryo_id'] == embryo_id].copy()
                        # Sort by time to connect points in temporal order
                        embryo_data = embryo_data.sort_values(TIME_COL)

                        fig.add_trace(go.Scatter(
                            x=embryo_data['total_length_um'],
                            y=embryo_data['baseline_deviation_normalized'],
                            mode='lines',
                            line=dict(color=color, width=1),
                            opacity=0.3,
                            hovertemplate=(
                                '<b>Embryo:</b> ' + embryo_id + '<br>'
                                '<b>Time:</b> %{customdata[0]:.1f} hpf<br>'
                                '<b>Length:</b> %{x:.1f} µm<br>'
                                '<b>Curvature:</b> %{y:.4f}<br>'
                                '<extra></extra>'
                            ),
                            customdata=embryo_data[TIME_COL].values.reshape(-1, 1),
                            showlegend=False,
                            name=embryo_id,
                        ), row=row_idx, col=col_idx)

                fig.update_xaxes(title_text='Total Length (µm)' if row_idx == n_pairs else '', row=row_idx, col=col_idx)
                if col_idx == 1:
                    fig.update_yaxes(title_text=f'{pair}<br>Norm. Baseline Dev.', row=row_idx, col=col_idx)

        # Apply global axes
        for row in range(1, n_pairs + 1):
            for col in range(1, n_genotypes + 1):
                fig.update_xaxes(range=[global_length_min, global_length_max], row=row, col=col)
                fig.update_yaxes(range=[global_curv_min, global_curv_max], row=row, col=col)

        fig.update_layout(
            title='Total Length vs Normalized Baseline Deviation - All Pairs',
            height=250 * n_pairs,
            width=1400,
            hovermode='closest',
            showlegend=True,
        )

        fig.write_html(output_path)
        print(f"Saved HTML: {output_path}")


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


# ============================================================================
# MAIN ANALYSIS FUNCTIONS
# ============================================================================

def analyze_experiment_metric(experiment_id, metric_name, generate_png=True, generate_plotly=True):
    """Run analysis for a single experiment and metric combination.

    Args:
        experiment_id: Experiment identifier (e.g., '20251104')
        metric_name: Name of metric to analyze
        generate_png: If True, generate PNG plots
        generate_plotly: If True, generate interactive HTML plots
    """
    print("\n" + "=" * 80)
    print(f"Analyzing Experiment {experiment_id} - Metric: {metric_name}")
    print(f"Formats: PNG={generate_png}, Plotly={generate_plotly}")
    print("=" * 80)

    # Set up output directories
    output_dir = BASE_OUTPUT_DIR / f'output_{experiment_id}_{get_metric_abbreviation(metric_name)}'
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
        plot_trajectories_for_pair(
            df, pair, metric_name, experiment_id, figures_dir,
            plotly=generate_plotly, png=generate_png
        )

    # Create overview plot with all pairs side-by-side
    plot_all_pairs_overview(
        df, pairs, metric_name, experiment_id, figures_dir,
        plotly=generate_plotly, png=generate_png
    )

    # Create collapsed plot: all genotypes by pair (1xN layout)
    plot_genotypes_by_pair(
        df, pairs, metric_name, experiment_id, figures_dir,
        plotly=generate_plotly, png=generate_png
    )

    # Create homozygous-only plot across pairs
    plot_homozygous_across_pairs(
        df, pairs, metric_name, experiment_id, figures_dir,
        plotly=generate_plotly, png=generate_png
    )

    # Create summary statistics
    summary_df = create_summary_statistics(df, metric_name, tables_dir)

    print("\n" + "=" * 80)
    print(f"Analysis complete for {experiment_id} - {metric_name}!")
    print(f"Figures saved to: {figures_dir}")
    print(f"Tables saved to: {tables_dir}")
    print("=" * 80)


def analyze_length_curvature_relationship(experiment_id, generate_png=True, generate_plotly=True):
    """
    Analyze relationship between length and curvature for all pairs.

    This analysis requires both total_length_um and baseline_deviation_normalized,
    so we load the full dataset and filter for rows with both metrics.

    Args:
        experiment_id: Experiment identifier (e.g., '20251121')
        generate_png: If True, generate PNG plots
        generate_plotly: If True, generate interactive HTML plots
    """
    print("\n" + "=" * 80)
    print(f"Analyzing Length vs Curvature Relationship - Experiment {experiment_id}")
    print("=" * 80)

    # Load data (any metric works since we need the full df with both columns)
    df = load_and_prepare_data(experiment_id, 'baseline_deviation_normalized')

    # Ensure both columns exist and filter for rows with non-null values in both
    df_filtered = df.dropna(subset=['total_length_um', 'baseline_deviation_normalized']).copy()

    print(f"Data shape after filtering for both metrics: {df_filtered.shape}")

    # Set up output directory
    output_dir = BASE_OUTPUT_DIR / f'output_{experiment_id}_length_curvature'
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Get unique pairs
    pairs = sorted(df_filtered[PAIR_COL].unique())
    print(f"Found {len(pairs)} pair groups: {pairs}")

    # Create scatter plot
    plot_length_vs_curvature_scatter(
        df_filtered, pairs, experiment_id, figures_dir,
        plotly=generate_plotly, png=generate_png
    )

    print("\n" + "=" * 80)
    print(f"Length vs Curvature analysis complete for {experiment_id}!")
    print(f"Figures saved to: {figures_dir}")
    print("=" * 80)


def main():
    """Main analysis function - loops over all experiments and metrics."""
    # Configuration: control which formats to generate
    GENERATE_PNG = True
    GENERATE_PLOTLY = True

    print("\n" + "=" * 80)
    print("B9D2 PAIR ANALYSIS - UNIFIED DUAL-BACKEND VERSION")
    print("Analyzing baseline deviation and length/area across genotypes and pairs")
    print("=" * 80)
    print(f"Experiments: {EXPERIMENT_IDS}")
    print(f"Metrics: {METRICS}")
    print(f"Output formats: PNG={GENERATE_PNG}, Plotly={GENERATE_PLOTLY}")
    print("=" * 80)

    # Loop over all combinations
    for experiment_id in EXPERIMENT_IDS:
        for metric_name in METRICS:
            analyze_experiment_metric(
                experiment_id, metric_name,
                generate_png=GENERATE_PNG,
                generate_plotly=GENERATE_PLOTLY
            )

    # Add length vs curvature scatter analysis
    for experiment_id in EXPERIMENT_IDS:
        analyze_length_curvature_relationship(
            experiment_id,
            generate_png=GENERATE_PNG,
            generate_plotly=GENERATE_PLOTLY
        )

    print("\n" + "=" * 80)
    print("ALL ANALYSES COMPLETE!")
    print(f"Results saved to: {BASE_OUTPUT_DIR}")
    print("=" * 80)


if __name__ == '__main__':
    main()
