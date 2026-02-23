#!/usr/bin/env python3
"""
Trajectory visualization with ranking-based coloring.

This module provides functions to visualize embryo metric trajectories over time,
with embryos colored by their ranking based on average values in a specified time window.

This helps answer questions like:
- How well does a metric at one timepoint predict the same metric at another timepoint?
- Are high/low values at time T1 predictive of high/low values at time T2?
- How does temporal consistency differ across genotypes?

Functions
---------
plot_ranked_trajectories : Main plotting function
get_embryo_rankings : Helper to compute rankings
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path


def get_embryo_rankings(
    df: pd.DataFrame,
    metric: str,
    time_column: str,
    embryo_id_column: str,
    ranking_window: tuple,
    reverse_rank: bool = False
) -> pd.DataFrame:
    """
    Calculate embryo rankings based on average metric values in a time window.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with embryo time-series data
    metric : str
        Column name for the metric to rank
    time_column : str
        Column name for time/stage
    embryo_id_column : str
        Column name for embryo identifiers
    ranking_window : tuple
        (min_time, max_time) for calculating average metrics
    reverse_rank : bool
        If True, reverse ranking so high values get low normalized rank (0)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: embryo_id, avg_metric, rank, normalized_rank
    """
    min_time, max_time = ranking_window

    # Filter to ranking window
    window_df = df[
        (df[time_column] >= min_time) &
        (df[time_column] <= max_time)
    ].copy()

    if window_df.empty:
        raise ValueError(f"No data found in ranking window {min_time}-{max_time}")

    # Calculate average metric per embryo
    rankings = (
        window_df
        .groupby(embryo_id_column)[metric]
        .mean()
        .reset_index()
        .rename(columns={metric: 'avg_metric'})
        .sort_values('avg_metric')
    )

    # Assign ranks
    rankings['rank'] = range(1, len(rankings) + 1)

    # Normalize ranks to [0, 1]
    if len(rankings) > 1:
        rankings['normalized_rank'] = (rankings['rank'] - 1) / (len(rankings) - 1)
    else:
        rankings['normalized_rank'] = 0.5

    # Reverse if requested (high values -> 0, low values -> 1)
    if reverse_rank:
        rankings['normalized_rank'] = 1 - rankings['normalized_rank']

    return rankings


def plot_ranked_trajectories(
    df: pd.DataFrame,
    metric: str = 'normalized_baseline_deviation',
    time_column: str = 'predicted_stage_hpf',
    genotype_column: str = 'genotype',
    embryo_id_column: str = 'embryo_id',
    ranking_window: tuple = (55, 70),
    display_window: tuple = None,
    smooth_window: int = 5,
    cmap: str = 'viridis',
    reverse_cmap: bool = False,
    figsize: tuple = (18, 8),
    save_path: Path = None
) -> tuple:
    """
    Plot embryo trajectories colored by ranking based on average values in a time window.

    Creates two side-by-side plots:
    1. Left: trajectories colored by metric rank (gradient showing low to high)
    2. Right: trajectories colored by genotype

    This visualization helps understand temporal prediction: do embryos with high
    values at time T1 tend to have high values at time T2?

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with embryo time-series data.
        Required columns: embryo_id, time, metric, genotype
    metric : str
        Column name for the metric to visualize
    time_column : str
        Column name for time/developmental stage
    genotype_column : str
        Column name for genotype labels
    embryo_id_column : str
        Column name for embryo identifiers
    ranking_window : tuple
        (min_time, max_time) for calculating average metrics and ranking embryos
    display_window : tuple, optional
        (min_time, max_time) for display. If None, shows all available data.
    smooth_window : int
        Rolling average window size for smoothing trajectories. Set to 1 for no smoothing.
    cmap : str
        Matplotlib colormap name for ranking visualization
    reverse_cmap : bool
        If True, reverses colormap (high values at bottom of color scale)
    figsize : tuple
        Figure size (width, height)
    save_path : Path, optional
        Path to save figure

    Returns
    -------
    tuple
        (fig, rankings_df) where rankings_df contains embryo rankings

    Examples
    --------
    >>> # Basic usage
    >>> fig, rankings = plot_ranked_trajectories(
    ...     df,
    ...     metric='normalized_baseline_deviation',
    ...     ranking_window=(55, 70)
    ... )

    >>> # Rank by early timepoints, display full trajectory
    >>> fig, rankings = plot_ranked_trajectories(
    ...     df,
    ...     ranking_window=(45, 55),
    ...     display_window=(40, 80)
    ... )
    """
    # Validate required columns
    required_cols = [metric, embryo_id_column, time_column, genotype_column]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Make a copy to avoid modifying original
    df = df.copy()

    # Calculate rankings
    rankings = get_embryo_rankings(
        df,
        metric=metric,
        time_column=time_column,
        embryo_id_column=embryo_id_column,
        ranking_window=ranking_window,
        reverse_rank=reverse_cmap
    )

    # Create embryo -> rank mapping
    embryo_to_rank = dict(zip(rankings[embryo_id_column], rankings['normalized_rank']))

    # Get genotype for each embryo
    embryo_to_genotype = (
        df[[embryo_id_column, genotype_column]]
        .drop_duplicates()
        .set_index(embryo_id_column)[genotype_column]
        .to_dict()
    )

    # Determine display window
    embryo_ids = rankings[embryo_id_column].tolist()
    display_df = df[df[embryo_id_column].isin(embryo_ids)].copy()

    if display_window is None:
        min_display = display_df[time_column].min()
        max_display = display_df[time_column].max()
    else:
        min_display, max_display = display_window

    # Filter to display window
    display_df = display_df[
        (display_df[time_column] >= min_display) &
        (display_df[time_column] <= max_display)
    ]

    # Setup genotype colors
    unique_genotypes = sorted(display_df[genotype_column].unique())
    genotype_colors = dict(zip(
        unique_genotypes,
        sns.color_palette("husl", len(unique_genotypes))
    ))

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # Setup colormap for ranking
    cmap_obj = plt.get_cmap(cmap)
    norm = Normalize(vmin=rankings['avg_metric'].min(), vmax=rankings['avg_metric'].max())

    # Plot each embryo
    for _, row in rankings.iterrows():
        embryo_id = row[embryo_id_column]
        normalized_rank = row['normalized_rank']

        # Get embryo data
        embryo_data = display_df[display_df[embryo_id_column] == embryo_id].copy()

        if embryo_data.empty:
            continue

        # Sort by time
        embryo_data = embryo_data.sort_values(time_column)

        # Apply smoothing
        if smooth_window > 1:
            embryo_data[f'smooth_{metric}'] = (
                embryo_data[metric]
                .rolling(window=smooth_window, min_periods=1, center=True)
                .mean()
            )
        else:
            embryo_data[f'smooth_{metric}'] = embryo_data[metric]

        # Get colors
        rank_color = cmap_obj(normalized_rank)
        genotype = embryo_to_genotype[embryo_id]
        genotype_color = genotype_colors[genotype]

        # Plot on both axes
        ax1.plot(
            embryo_data[time_column],
            embryo_data[f'smooth_{metric}'],
            color=rank_color,
            linewidth=1.5,
            alpha=0.7
        )

        ax2.plot(
            embryo_data[time_column],
            embryo_data[f'smooth_{metric}'],
            color=genotype_color,
            linewidth=1.5,
            alpha=0.7
        )

    # Add shaded ranking window
    min_rank, max_rank = ranking_window
    for ax in [ax1, ax2]:
        ax.axvspan(min_rank, max_rank, alpha=0.2, color='gray', zorder=0)

        # Add label for ranking window
        y_range = ax.get_ylim()
        y_pos = y_range[0] + 0.05 * (y_range[1] - y_range[0])
        ax.text(
            (min_rank + max_rank) / 2,
            y_pos,
            f"Ranking window: {min_rank}-{max_rank}",
            horizontalalignment='center',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
            fontsize=10
        )

    # Titles and labels
    ax1.set_title(
        f"All Embryos (n={len(rankings)})\nColored by {metric} Rank",
        fontsize=14
    )
    ax2.set_title(
        f"All Embryos\nColored by Genotype",
        fontsize=14
    )

    ax1.set_xlabel(time_column, fontsize=12)
    ax2.set_xlabel(time_column, fontsize=12)
    ax1.set_ylabel(metric, fontsize=12)

    # Add grids
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax2.grid(True, linestyle='--', alpha=0.3)

    # Add colorbar for ranking plot
    sm = ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label(f'Average {metric} ({min_rank}-{max_rank})', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # Add legend for genotypes
    genotype_lines = [
        plt.Line2D([0], [0], color=color, lw=2)
        for genotype, color in genotype_colors.items()
    ]
    ax2.legend(
        genotype_lines,
        list(genotype_colors.keys()),
        title="Genotypes",
        loc='best',
        fontsize=10
    )

    # Main title
    fig.suptitle(
        f"{metric.replace('_', ' ').title()} Trajectories - Ranked by {min_rank}-{max_rank} {time_column}",
        fontsize=16,
        y=1.00
    )

    # Layout
    plt.tight_layout()

    # Save if requested
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig, rankings


# Example usage
if __name__ == '__main__':
    print("Trajectory visualization module loaded.")
    print("\nExample usage:")
    print("""
    from trajectory_visualization import plot_ranked_trajectories

    # Create ranked trajectory plot
    fig, rankings = plot_ranked_trajectories(
        df,
        metric='normalized_baseline_deviation',
        ranking_window=(55, 70),
        display_window=(40, 80),
        smooth_window=5
    )
    """)
