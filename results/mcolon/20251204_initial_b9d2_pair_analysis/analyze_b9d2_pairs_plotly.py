#!/usr/bin/env python3
"""
Interactive Plotly version of B9D2 pair analysis with hover tooltips showing embryo_id.

This script analyzes b9d2 data from experiments, creating interactive Plotly plots
with hover information showing embryo_id, time, metric value, genotype, and pair.

Output: Interactive HTML files (can zoom, pan, toggle traces, see embryo IDs on hover)
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
from scipy.ndimage import gaussian_filter1d

# Add src to path
sys.path.insert(0, '/net/trapnell/vol1/home/mdcolon/proj/morphseq')

from src.analyze.trajectory_analysis.data_loading import _load_df03_format

# Configuration
EXPERIMENT_IDS = ['20251119']
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
SMOOTH_METHOD = 'gaussian'
SMOOTH_PARAMS = {'sigma': 1.5}


def get_metric_label(metric_name):
    """Get human-readable label for a metric."""
    return METRIC_DISPLAY_NAMES.get(metric_name, metric_name)


def load_and_prepare_data(experiment_id, metric_name):
    """Load data and prepare for analysis."""
    print(f"Loading data for experiment {experiment_id}, metric {metric_name}...")

    df = _load_df03_format(experiment_id)

    # Handle column naming collisions
    if metric_name == 'total_length_um' and 'total_length_um_y' in df.columns:
        df['total_length_um'] = df['total_length_um_y']

    # Filter for valid embryos
    df = df[df['use_embryo_flag'] == 1].copy()

    # Drop rows with missing values
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


def plot_trajectories_for_pair_plotly(df, pair, metric_name, output_path,
                                      global_time_min=None, global_time_max=None,
                                      global_metric_min=None, global_metric_max=None):
    """Create a 1x3 faceted Plotly plot for a pair showing all three genotypes."""
    print(f"Creating Plotly plot for {pair}...")

    metric_label = get_metric_label(metric_name)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[f'{g.replace("b9d2_", "").title()}' for g in GENOTYPE_ORDER],
        specs=[[{}, {}, {}]]
    )

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
    print(f"Saved: {output_path}")


def plot_all_pairs_overview_plotly(df, pairs, metric_name, output_path):
    """Create an interactive overview plot with all pairs (rows) and genotypes (columns)."""
    print(f"Creating Plotly overview plot with all {len(pairs)} pairs...")

    n_pairs = len(pairs)
    n_genotypes = 3
    metric_label = get_metric_label(metric_name)

    fig = make_subplots(
        rows=n_pairs, cols=n_genotypes,
        subplot_titles=[f'{g.replace("b9d2_", "").title()}' for g in GENOTYPE_ORDER] * n_pairs,
        specs=[[{} for _ in range(n_genotypes)] for _ in range(n_pairs)]
    )

    # First pass: determine global axis ranges
    all_data = {}
    global_time_min, global_time_max = float('inf'), float('-inf')
    global_metric_min, global_metric_max = float('inf'), float('-inf')

    for pair in pairs:
        for genotype in GENOTYPE_ORDER:
            trajectories, _, n_embryos = get_trajectories_for_pair_genotype(df, pair, genotype, metric_name)
            all_data[(pair, genotype)] = (trajectories, n_embryos)

            if trajectories is not None and n_embryos > 0:
                for traj in trajectories:
                    global_time_min = min(global_time_min, traj['times'].min())
                    global_time_max = max(global_time_max, traj['times'].max())
                    global_metric_min = min(global_metric_min, traj['metrics'].min())
                    global_metric_max = max(global_metric_max, traj['metrics'].max())

    # Add padding
    metric_padding = (global_metric_max - global_metric_min) * 0.1
    global_metric_min -= metric_padding
    global_metric_max += metric_padding

    # Second pass: plot
    for row_idx, pair in enumerate(pairs, start=1):
        for col_idx, genotype in enumerate(GENOTYPE_ORDER, start=1):
            trajectories, n_embryos = all_data[(pair, genotype)]
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
    print(f"Saved: {output_path}")


def plot_genotypes_by_pair_plotly(df, pairs, metric_name, output_path):
    """Create a 1xN Plotly plot showing all three genotypes overlaid for each pair."""
    print(f"Creating Plotly genotypes-by-pair comparison for {len(pairs)} pairs...")

    metric_label = get_metric_label(metric_name)

    fig = make_subplots(
        rows=1, cols=len(pairs),
        subplot_titles=[f'{p}' for p in pairs],
    )

    # First pass: collect data and determine axis ranges
    all_pair_genotype_data = {}
    global_time_min, global_time_max = float('inf'), float('-inf')
    global_metric_min, global_metric_max = float('inf'), float('-inf')

    for pair in pairs:
        for genotype in GENOTYPE_ORDER:
            trajectories, _, n_embryos = get_trajectories_for_pair_genotype(df, pair, genotype, metric_name)
            all_pair_genotype_data[(pair, genotype)] = (trajectories, n_embryos)

            if trajectories is not None and n_embryos > 0:
                for traj in trajectories:
                    global_time_min = min(global_time_min, traj['times'].min())
                    global_time_max = max(global_time_max, traj['times'].max())
                    global_metric_min = min(global_metric_min, traj['metrics'].min())
                    global_metric_max = max(global_metric_max, traj['metrics'].max())

    # Add padding
    metric_padding = (global_metric_max - global_metric_min) * 0.1
    global_metric_min -= metric_padding
    global_metric_max += metric_padding

    # Second pass: plot
    for col_idx, pair in enumerate(pairs, start=1):
        for genotype in GENOTYPE_ORDER:
            trajectories, n_embryos = all_pair_genotype_data[(pair, genotype)]
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
    print(f"Saved: {output_path}")


def plot_homozygous_across_pairs_plotly(df, pairs, metric_name, output_path):
    """Create a 1xN Plotly plot showing only homozygous genotype across all pairs."""
    print(f"Creating Plotly homozygous-only plot across {len(pairs)} pairs...")

    metric_label = get_metric_label(metric_name)
    homozygous_genotype = 'b9d2_homozygous'
    color = GENOTYPE_COLORS[homozygous_genotype]

    fig = make_subplots(
        rows=1, cols=len(pairs),
        subplot_titles=[f'{p}' for p in pairs],
    )

    # First pass: determine axis ranges
    all_pair_data = {}
    global_time_min, global_time_max = float('inf'), float('-inf')
    global_metric_min, global_metric_max = float('inf'), float('-inf')

    for pair in pairs:
        trajectories, _, n_embryos = get_trajectories_for_pair_genotype(df, pair, homozygous_genotype, metric_name)
        all_pair_data[pair] = (trajectories, n_embryos)

        if trajectories is not None and n_embryos > 0:
            for traj in trajectories:
                global_time_min = min(global_time_min, traj['times'].min())
                global_time_max = max(global_time_max, traj['times'].max())
                global_metric_min = min(global_metric_min, traj['metrics'].min())
                global_metric_max = max(global_metric_max, traj['metrics'].max())

    # Add padding
    metric_padding = (global_metric_max - global_metric_min) * 0.1
    global_metric_min -= metric_padding
    global_metric_max += metric_padding

    # Second pass: plot
    for col_idx, pair in enumerate(pairs, start=1):
        trajectories, n_embryos = all_pair_data[pair]

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
    print(f"Saved: {output_path}")


def analyze_experiment_metric_plotly(experiment_id, metric_name):
    """Run interactive Plotly analysis for a single experiment and metric."""
    print("\n" + "=" * 80)
    print(f"Analyzing (Plotly) Experiment {experiment_id} - Metric: {metric_name}")
    print("=" * 80)

    # Set up output directories
    output_dir = BASE_OUTPUT_DIR / f'output_{experiment_id}_{metric_name}_plotly'
    figures_dir = output_dir / 'figures'

    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_and_prepare_data(experiment_id, metric_name)

    # Get unique pairs
    pairs = sorted(df[PAIR_COL].unique())
    print(f"\nFound {len(pairs)} pair groups: {pairs}")

    # Create per-pair plots
    for pair in pairs:
        output_path = figures_dir / f'pair_{pair.replace("b9d2_", "").replace("_", "")}_all_genotypes.html'
        plot_trajectories_for_pair_plotly(df, pair, metric_name, output_path)

    # Create overview plot
    overview_path = figures_dir / 'all_pairs_overview.html'
    plot_all_pairs_overview_plotly(df, pairs, metric_name, overview_path)

    # Create genotypes-by-pair plot
    genotypes_by_pair_path = figures_dir / 'genotypes_by_pair_comparison.html'
    plot_genotypes_by_pair_plotly(df, pairs, metric_name, genotypes_by_pair_path)

    # Create homozygous-only plot
    homozygous_path = figures_dir / 'homozygous_across_pairs.html'
    plot_homozygous_across_pairs_plotly(df, pairs, metric_name, homozygous_path)

    print("\n" + "=" * 80)
    print(f"Interactive Plotly analysis complete for {experiment_id} - {metric_name}!")
    print(f"HTML plots saved to: {figures_dir}")
    print("=" * 80)


def main():
    """Main analysis function."""
    print("\n" + "=" * 80)
    print("B9D2 PAIR ANALYSIS - INTERACTIVE PLOTLY VERSION")
    print("With hover tooltips showing embryo IDs and metrics")
    print("=" * 80)
    print(f"Experiments: {EXPERIMENT_IDS}")
    print(f"Metrics: {METRICS}")
    print("=" * 80)

    # Loop over all combinations
    for experiment_id in EXPERIMENT_IDS:
        for metric_name in METRICS:
            analyze_experiment_metric_plotly(experiment_id, metric_name)

    print("\n" + "=" * 80)
    print("ALL INTERACTIVE PLOTLY ANALYSES COMPLETE!")
    print(f"Results saved to: {BASE_OUTPUT_DIR}")
    print("=" * 80)


if __name__ == '__main__':
    main()
