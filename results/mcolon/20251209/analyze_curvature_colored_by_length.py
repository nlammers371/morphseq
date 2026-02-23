#!/usr/bin/env python3
"""
Curvature Over Time Colored by Total Length - B9D2 Analysis

This script plots curvature (baseline_deviation_normalized) over time (hpf) with
each trajectory colored by its total_length_um instead of genotype. This visualization
explores the relationship between trajectory length and curvature patterns.

GOAL: Find a way to overlay curvature and length information on one plot.
This is a first approach. If it doesn't work well, we'll try normalizing by
percentage of wildtype average at each timepoint.

DATA SOURCE:
- Uses build04 output (qc_staged) from morphseq_playground/metadata/build04_output/
- Filters for use_embryo_flag == 1

OUTPUT:
- PNG and/or HTML plots showing curvature trajectories colored by length
- Layout: Rows = pairs, Columns = genotypes
- Color scale: continuous from min to max trajectory length

USAGE:
1. Configure EXPERIMENT_IDS at the top
2. Set GENERATE_PNG and GENERATE_PLOTLY flags
3. Run: python analyze_curvature_colored_by_length.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
from scipy.ndimage import gaussian_filter1d

# Add src to path
sys.path.insert(0, '/net/trapnell/vol1/home/mdcolon/proj/morphseq')

from src.analyze.trajectory_analysis.data_loading import _load_qc_staged

# ============================================================================
# CONFIGURATION
# ============================================================================

# Experiments to analyze
EXPERIMENT_IDS = ['20251121']

# Column names
TIME_COL = 'predicted_stage_hpf'
EMBRYO_ID_COL = 'embryo_id'
PAIR_COL = 'pair'
GENOTYPE_COL = 'genotype'
CURVATURE_COL = 'baseline_deviation_normalized'
LENGTH_COL = 'total_length_um'

# Base output directory
BASE_OUTPUT_DIR = Path('/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251209')

# Genotype configuration for b9d2
GENOTYPE_ORDER = ['b9d2_wildtype', 'b9d2_heterozygous', 'b9d2_homozygous']

# Smoothing configuration
SMOOTH_METHOD = 'gaussian'
SMOOTH_PARAMS = {'sigma': 1.5}


# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

def load_and_prepare_data(experiment_id):
    """Load data and prepare for analysis."""
    print(f"Loading data for experiment {experiment_id}...")

    df = _load_qc_staged(experiment_id)

    # Filter for valid embryos
    df = df[df['use_embryo_flag'] == 1].copy()

    # Drop rows with missing values in key columns
    df = df.dropna(subset=[EMBRYO_ID_COL, TIME_COL, CURVATURE_COL, LENGTH_COL, PAIR_COL, GENOTYPE_COL])

    print(f"Data shape: {df.shape}")
    print(f"Unique pairs: {df[PAIR_COL].unique()}")
    print(f"Genotypes: {df[GENOTYPE_COL].unique()}")

    return df


def get_trajectories_with_lengths(df, pair, genotype):
    """
    Extract curvature trajectories for a specific pair and genotype,
    including the length at each timepoint (optionally smoothed).

    Returns:
        list of dicts with keys:
            - embryo_id: str
            - times: array of timepoints
            - curvatures: array of curvature values (smoothed)
            - lengths: array of length values at each timepoint (smoothed)
            - genotype: str
            - pair: str
    """
    filtered = df[(df[PAIR_COL] == pair) & (df[GENOTYPE_COL] == genotype)].copy()

    if len(filtered) == 0:
        return None, 0

    embryo_ids = filtered[EMBRYO_ID_COL].unique()
    trajectories = []

    for embryo_id in embryo_ids:
        embryo_data = filtered[filtered[EMBRYO_ID_COL] == embryo_id].sort_values(TIME_COL)
        if len(embryo_data) > 1:
            times = embryo_data[TIME_COL].values
            curvatures = embryo_data[CURVATURE_COL].values
            lengths = embryo_data[LENGTH_COL].values

            # Apply Gaussian smoothing if enabled
            if SMOOTH_METHOD == 'gaussian':
                sigma = SMOOTH_PARAMS.get('sigma', 1.5)
                curvatures = gaussian_filter1d(curvatures, sigma=sigma)
                # Also smooth the lengths to prevent color jumping
                lengths = gaussian_filter1d(lengths, sigma=sigma)

            trajectories.append({
                'embryo_id': embryo_id,
                'times': times,
                'curvatures': curvatures,
                'lengths': lengths,  # Array of lengths at each timepoint
                'genotype': genotype,
                'pair': pair,
            })

    return trajectories, len(trajectories)


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_curvature_colored_by_length(df, pairs, experiment_id, figures_dir,
                                     plotly=False, png=False):
    """
    Create plot of curvature over time, with trajectories colored by total length.

    Layout: Rows = pairs, Columns = genotypes
    Each trajectory is colored by its mean total_length_um across timepoints.

    Args:
        df: DataFrame with trajectory data
        pairs: List of pair identifiers
        experiment_id: Experiment identifier
        figures_dir: Output directory for figures
        plotly: If True, generate interactive HTML plot
        png: If True, generate static PNG plot
    """
    print(f"\nCreating curvature-vs-time plot colored by length...")

    n_pairs = len(pairs)
    n_genotypes = 3

    # ========================================================================
    # SHARED DATA PREPARATION
    # ========================================================================

    # First pass: collect all data to determine global axis ranges and color scale
    all_data = {}
    global_time_min, global_time_max = float('inf'), float('-inf')
    global_curv_min, global_curv_max = float('inf'), float('-inf')
    global_length_min, global_length_max = float('inf'), float('-inf')

    for pair in pairs:
        for genotype in GENOTYPE_ORDER:
            trajectories, n_embryos = get_trajectories_with_lengths(df, pair, genotype)
            all_data[(pair, genotype)] = (trajectories, n_embryos)

            if trajectories is not None and n_embryos > 0:
                for traj in trajectories:
                    global_time_min = min(global_time_min, traj['times'].min())
                    global_time_max = max(global_time_max, traj['times'].max())
                    global_curv_min = min(global_curv_min, traj['curvatures'].min())
                    global_curv_max = max(global_curv_max, traj['curvatures'].max())
                    global_length_min = min(global_length_min, traj['lengths'].min())
                    global_length_max = max(global_length_max, traj['lengths'].max())

    # Add padding to axes
    curv_padding = (global_curv_max - global_curv_min) * 0.1
    global_curv_min -= curv_padding
    global_curv_max += curv_padding

    print(f"Global length range: {global_length_min:.1f} - {global_length_max:.1f} µm")
    print(f"Global curvature range: {global_curv_min:.4f} - {global_curv_max:.4f}")

    # ========================================================================
    # PNG RENDERING
    # ========================================================================

    if png:
        output_path = figures_dir / f'{experiment_id}_curvature_by_time_colored_by_length.png'

        fig, axes = plt.subplots(n_pairs, n_genotypes, figsize=(15, 4.5 * n_pairs))

        # Ensure axes is always 2D
        if n_pairs == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle('Curvature Over Time - Colored by Trajectory Length',
                     fontsize=16, fontweight='bold', y=0.995)

        # Create colormap
        cmap = plt.colormaps.get_cmap('viridis')
        norm = mcolors.Normalize(vmin=global_length_min, vmax=global_length_max)

        for row_idx, pair in enumerate(pairs):
            for col_idx, genotype in enumerate(GENOTYPE_ORDER):
                ax = axes[row_idx, col_idx]
                trajectories, n_embryos = all_data[(pair, genotype)]

                if trajectories is None or n_embryos == 0:
                    ax.text(0.5, 0.5, 'No data',
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=10, color='lightgray')
                    ax.set_xlabel('Time (hpf)', fontsize=9)
                    ax.set_ylabel('Normalized Baseline Deviation', fontsize=9)
                    if row_idx == 0:
                        ax.set_title(f'{genotype.replace("b9d2_", "").title()}', fontweight='bold')
                    if col_idx == 0:
                        ax.set_ylabel(f'{pair}\n\nNorm. Baseline Dev.', fontsize=9)
                    ax.tick_params(labelsize=8)
                    ax.set_xlim(global_time_min, global_time_max)
                    ax.set_ylim(global_curv_min, global_curv_max)
                    continue

                # Plot trajectories with colors varying by length at each timepoint
                for traj in trajectories:
                    # Create line segments
                    points = np.array([traj['times'], traj['curvatures']]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)

                    # Use the average length of adjacent points for each segment
                    segment_colors = (traj['lengths'][:-1] + traj['lengths'][1:]) / 2

                    # Create LineCollection
                    lc = LineCollection(segments, cmap=cmap, norm=norm, alpha=0.6, linewidths=1.5)
                    lc.set_array(segment_colors)
                    ax.add_collection(lc)

                # Set labels and title
                ax.set_xlabel('Time (hpf)', fontsize=9)
                if col_idx == 0:
                    ax.set_ylabel(f'{pair}\n\nNorm. Baseline Dev.', fontsize=9)
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
                ax.set_ylim(global_curv_min, global_curv_max)

        # Add colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
        cbar.set_label('Mean Total Length (µm)', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved PNG: {output_path}")
        plt.close()

    # ========================================================================
    # PLOTLY RENDERING
    # ========================================================================

    if plotly:
        output_path = figures_dir / f'{experiment_id}_curvature_by_time_colored_by_length.html'

        fig = make_subplots(
            rows=n_pairs, cols=n_genotypes,
            subplot_titles=[f'{g.replace("b9d2_", "").title()}' for g in GENOTYPE_ORDER] * n_pairs,
            specs=[[{} for _ in range(n_genotypes)] for _ in range(n_pairs)]
        )

        for row_idx, pair in enumerate(pairs, start=1):
            for col_idx, genotype in enumerate(GENOTYPE_ORDER, start=1):
                trajectories, n_embryos = all_data[(pair, genotype)]

                if trajectories is None or n_embryos == 0:
                    fig.add_annotation(
                        text='No data',
                        xref=f'x{col_idx}', yref=f'y{col_idx}',
                        showarrow=False, font=dict(size=10, color='lightgray'),
                        row=row_idx, col=col_idx
                    )
                else:
                    # Plot trajectories colored by length at each timepoint
                    for traj in trajectories:
                        fig.add_trace(go.Scatter(
                            x=traj['times'],
                            y=traj['curvatures'],
                            mode='lines+markers',
                            line=dict(width=1.5, color='rgba(0,0,0,0)'),  # Transparent line
                            marker=dict(
                                size=6,
                                color=traj['lengths'],
                                colorscale='Viridis',
                                cmin=global_length_min,
                                cmax=global_length_max,
                                colorbar=dict(
                                    title='Length (µm)',
                                    x=1.02,
                                ) if (row_idx == 1 and col_idx == n_genotypes) else None,
                                showscale=(row_idx == 1 and col_idx == n_genotypes),
                                line=dict(width=0),
                            ),
                            opacity=0.6,
                            hovertemplate=(
                                '<b>Embryo:</b> %{customdata[0]}<br>'
                                '<b>Time:</b> %{x:.2f} hpf<br>'
                                '<b>Curvature:</b> %{y:.4f}<br>'
                                '<b>Length:</b> %{customdata[1]:.1f} µm<br>'
                                '<extra></extra>'
                            ),
                            customdata=np.column_stack((
                                [traj['embryo_id']] * len(traj['times']),
                                traj['lengths'],
                            )),
                            showlegend=False,
                        ), row=row_idx, col=col_idx)

                        # Add connecting lines between markers (colored by average of adjacent points)
                        for i in range(len(traj['times']) - 1):
                            avg_length = (traj['lengths'][i] + traj['lengths'][i+1]) / 2
                            # Map length to color
                            color_val = (avg_length - global_length_min) / (global_length_max - global_length_min)
                            # Viridis colormap approximation
                            color_rgb = plt.colormaps.get_cmap('viridis')(color_val)
                            color_str = f'rgba({int(color_rgb[0]*255)},{int(color_rgb[1]*255)},{int(color_rgb[2]*255)},0.6)'

                            fig.add_trace(go.Scatter(
                                x=[traj['times'][i], traj['times'][i+1]],
                                y=[traj['curvatures'][i], traj['curvatures'][i+1]],
                                mode='lines',
                                line=dict(width=1.5, color=color_str),
                                showlegend=False,
                                hoverinfo='skip',
                            ), row=row_idx, col=col_idx)

                fig.update_xaxes(title_text='Time (hpf)' if row_idx == n_pairs else '', row=row_idx, col=col_idx)
                if col_idx == 1:
                    fig.update_yaxes(title_text=f'{pair}<br>Norm. Baseline Dev.', row=row_idx, col=col_idx)

        # Apply global axes
        for row in range(1, n_pairs + 1):
            for col in range(1, n_genotypes + 1):
                fig.update_xaxes(range=[global_time_min, global_time_max], row=row, col=col)
                fig.update_yaxes(range=[global_curv_min, global_curv_max], row=row, col=col)

        fig.update_layout(
            title='Curvature Over Time - Colored by Trajectory Length',
            height=250 * n_pairs,
            width=1400,
            hovermode='closest',
            showlegend=False,
        )

        fig.write_html(output_path)
        print(f"Saved HTML: {output_path}")


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def main():
    """Main analysis function."""
    # Configuration: control which formats to generate
    GENERATE_PNG = True
    GENERATE_PLOTLY = True

    print("\n" + "=" * 80)
    print("CURVATURE OVER TIME COLORED BY LENGTH - B9D2 ANALYSIS")
    print("=" * 80)
    print(f"Experiments: {EXPERIMENT_IDS}")
    print(f"Output formats: PNG={GENERATE_PNG}, Plotly={GENERATE_PLOTLY}")
    print("=" * 80)

    for experiment_id in EXPERIMENT_IDS:
        print(f"\nAnalyzing experiment {experiment_id}...")

        # Set up output directory
        output_dir = BASE_OUTPUT_DIR / f'output_{experiment_id}_curvature_by_length'
        figures_dir = output_dir / 'figures'
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        df = load_and_prepare_data(experiment_id)

        # Get unique pairs
        pairs = sorted(df[PAIR_COL].unique())
        print(f"Found {len(pairs)} pair groups: {pairs}")

        # Create plot
        plot_curvature_colored_by_length(
            df, pairs, experiment_id, figures_dir,
            plotly=GENERATE_PLOTLY, png=GENERATE_PNG
        )

        print("\n" + "=" * 80)
        print(f"Analysis complete for {experiment_id}!")
        print(f"Figures saved to: {figures_dir}")
        print("=" * 80)


if __name__ == '__main__':
    main()
