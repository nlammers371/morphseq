"""Composition and layout functions for multi-panel figures.

High-level functions that compose modular plotting functions into layouts.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from typing import Dict, Optional, Tuple, List
from .plotting_functions import (
    plot_auroc_with_null,
    plot_divergence_timecourse,
    plot_raw_metric_timecourse
)


def create_auroc_only_figure(
    auroc_df: pd.DataFrame,
    color: str = '#2ca02c',
    label: str = 'Classification',
    title: str = 'AUROC Over Time',
    figsize: Tuple[int, int] = (14, 7),
    ylim: Tuple[float, float] = (0.3, 1.05),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create single AUROC plot.

    Simple wrapper for plot_auroc_with_null.

    Parameters
    ----------
    auroc_df : pd.DataFrame
        Pre-processed AUROC data
    color : str
        Line color
    label : str
        Legend label
    title : str
        Figure title
    figsize : tuple
        Figure size
    ylim : tuple
        Y-axis limits
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    plot_auroc_with_null(
        ax=ax,
        auroc_df=auroc_df,
        color=color,
        label=label
    )

    # Reference line at 0.5 (chance)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Chance (0.5)')

    # Formatting
    ax.set_xlabel('Hours Post Fertilization (hpf)', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(ylim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def create_three_panel_comparison(
    auroc_data,
    divergence_data,
    trajectory_data,
    group1_label: str,
    group2_label: str,
    metric_cols: List[str],
    embedding_auroc_data=None,
    metric_labels: Optional[Dict[str, str]] = None,
    time_landmarks: Optional[Dict[float, str]] = None,
    title: str = 'Group Comparison',
    save_path: Optional[str] = None,
    divergence_use_smoothed: bool = True,
    trajectory_use_smoothed: bool = True
) -> plt.Figure:
    """Create three-panel comparison figure.

    Refactored version of create_comparison_figure that uses modular plotting functions.

    Parameters
    ----------
    auroc_data : pd.DataFrame
        Pre-processed AUROC data (metric-based)
    divergence_data : pd.DataFrame
        Pre-processed divergence data (with smoothing applied if desired)
    trajectory_data : pd.DataFrame
        Trajectory data (can be smoothed or raw)
    group1_label : str
        Label for group 1
    group2_label : str
        Label for group 2
    metric_cols : list of str
        Metric columns to plot
    embedding_auroc_data : pd.DataFrame, optional
        Embedding-based AUROC for overlay
    metric_labels : dict, optional
        Display labels for metrics
    time_landmarks : dict, optional
        Time points to annotate {hpf: label}
    title : str
        Figure title
    save_path : str, optional
        Path to save figure
    divergence_use_smoothed : bool
        Use smoothed divergence values
    trajectory_use_smoothed : bool
        Use smoothed trajectory values

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Create figure with 3 panels
    fig = plt.figure(figsize=(18, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)

    ax_auroc = fig.add_subplot(gs[0])
    ax_divergence = fig.add_subplot(gs[1])
    ax_trajectories = fig.add_subplot(gs[2])

    # Panel A: AUROC
    plot_auroc_with_null(
        ax=ax_auroc,
        auroc_df=auroc_data,
        color='#2ca02c',
        label='Metric AUROC'
    )

    if embedding_auroc_data is not None:
        plot_auroc_with_null(
            ax=ax_auroc,
            auroc_df=embedding_auroc_data,
            color='#1f77b4',
            label='Embedding AUROC'
        )

    ax_auroc.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Chance')
    ax_auroc.set_xlabel('Hours Post Fertilization (hpf)', fontsize=11)
    ax_auroc.set_ylabel('AUROC', fontsize=11)
    ax_auroc.set_title('Panel A: Classification Performance', fontsize=12, fontweight='bold')
    ax_auroc.legend(loc='upper left', fontsize=9)
    ax_auroc.set_ylim(0.3, 1.05)
    ax_auroc.grid(True, alpha=0.3)

    # Panel B: Divergence
    plot_divergence_timecourse(
        ax=ax_divergence,
        divergence_df=divergence_data,
        metric_filter=metric_cols,
        use_smoothed=divergence_use_smoothed,
        labels_dict=metric_labels
    )

    ax_divergence.set_xlabel('Hours Post Fertilization (hpf)', fontsize=11)
    ax_divergence.set_ylabel('Absolute Difference (Z-scored)', fontsize=11)
    ax_divergence.set_title('Panel B: Metric Divergence', fontsize=12, fontweight='bold')
    ax_divergence.legend(loc='upper left', fontsize=9)
    ax_divergence.grid(True, alpha=0.3)

    # Panel C: Trajectories (show for primary metric)
    primary_metric = metric_cols[0] if metric_cols else 'baseline_deviation_normalized'

    group_colors = {
        group1_label: '#D32F2F',
        group2_label: '#1976D2'
    }

    plot_raw_metric_timecourse(
        ax=ax_trajectories,
        df_trajectories=trajectory_data,
        metric_col=primary_metric,
        show_individual=True,
        show_group_stats=True,
        colors_dict=group_colors,
        use_smoothed=trajectory_use_smoothed,
        alpha_individual=0.25
    )

    ax_trajectories.set_xlabel('Hours Post Fertilization (hpf)', fontsize=11)
    metric_display = metric_labels.get(primary_metric, primary_metric) if metric_labels else primary_metric
    ax_trajectories.set_ylabel(metric_display, fontsize=11)
    ax_trajectories.set_title('Panel C: Individual Trajectories', fontsize=12, fontweight='bold')
    ax_trajectories.legend(loc='upper left', fontsize=9)
    ax_trajectories.grid(True, alpha=0.3)

    # Add time landmarks if provided
    if time_landmarks:
        for hpf, landmark_label in time_landmarks.items():
            for ax in [ax_auroc, ax_divergence, ax_trajectories]:
                ax.axvline(x=hpf, color='gray', linestyle='--', alpha=0.4)
                ax.text(hpf, ax.get_ylim()[1] * 0.95, landmark_label,
                       ha='center', va='top', fontsize=8, color='gray')

    # Overall title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def create_feature_comparison_panels(
    results_curvature: Dict[str, Dict],
    results_length: Dict[str, Dict],
    results_embedding: Dict[str, Dict],
    colors_dict: Dict[str, str],
    styles_dict: Dict[str, str],
    title: str = 'Feature Comparison: Curvature vs Length vs VAE Embedding',
    figsize: Tuple[int, int] = (18, 5),
    ylim: Tuple[float, float] = (0.3, 1.05),
    time_col: str = 'time_bin_center',
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create 1x3 panel figure comparing AUROC across different feature types.

    Each panel shows all genotype comparisons (Homo vs WT, Homo vs Het, Het vs WT)
    overlaid for a specific feature type.

    Parameters
    ----------
    results_curvature : dict
        Results dict {comparison_label: {'auroc_data': pd.DataFrame}}
        for curvature-based classification
    results_length : dict
        Results dict for length-based classification
    results_embedding : dict
        Results dict for VAE embedding-based classification
    colors_dict : dict
        Color mapping {comparison_label: color}
    styles_dict : dict
        Line style mapping {comparison_label: style}
    title : str
        Overall figure title
    figsize : tuple
        Figure size (width, height)
    ylim : tuple
        Y-axis limits for AUROC
    time_col : str
        Time column to use for x-axis
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    ax_curv, ax_length, ax_emb = axes

    panel_configs = [
        (ax_curv, results_curvature, 'Curvature\n(baseline_deviation_normalized)'),
        (ax_length, results_length, 'Length\n(total_length_um)'),
        (ax_emb, results_embedding, 'VAE Embedding\n(z_mu_b features)')
    ]

    for ax, results_dict, panel_title in panel_configs:
        # Plot each comparison
        for label, result in results_dict.items():
            auroc_df = result['auroc_data']
            if auroc_df.empty:
                continue

            color = colors_dict.get(label, '#000000')
            style = styles_dict.get(label, '-')

            plot_auroc_with_null(
                ax=ax,
                auroc_df=auroc_df,
                color=color,
                label=label,
                style=style,
                time_col=time_col
            )

        # Reference line at 0.5 (chance)
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Chance (0.5)')

        # Formatting
        ax.set_xlabel('Hours Post Fertilization (hpf)', fontsize=11)
        if ax == ax_curv:
            ax.set_ylabel('AUROC', fontsize=11)
        ax.set_title(panel_title, fontsize=12, fontweight='bold')
        ax.set_ylim(ylim)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)

    # Add legend with significance marker to first panel only
    ax_curv.scatter([], [], s=200, facecolors='none', edgecolors='black',
                    linewidths=2.5, label='p < 0.05')
    ax_curv.legend(loc='upper left', fontsize=9)

    # Overall title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig
