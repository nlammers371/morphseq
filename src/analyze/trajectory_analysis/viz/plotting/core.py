"""
Plotting Functions

Visualization for trajectory analysis, clustering, and quality assessment.

This module provides DataFrame-centric plotting functions where time column (hpf)
is used directly from the DataFrame. This eliminates time-axis alignment bugs
that occurred with array-based plotting.

Functions (New API - DataFrame-first, recommended)
--------------------------------------------------
- plot_cluster_trajectories_df: Plot trajectories by cluster (uses DataFrame)
- plot_membership_trajectories_df: Plot trajectories by membership (uses DataFrame)
- plot_posterior_heatmap: Posterior probability heatmap (works with both APIs)
- plot_2d_scatter: 2D scatter plot (works with both APIs)
- plot_membership_vs_k: Plot membership quality trends across k values

Functions (Legacy API - deprecated)
-----------------------------------
- plot_cluster_trajectories: DEPRECATED - use plot_cluster_trajectories_df()
- plot_membership_trajectories: DEPRECATED - use plot_membership_trajectories_df()
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from ...config import DEFAULT_DPI, DEFAULT_FIGSIZE, MEMBERSHIP_COLORS


# ============================================================================
# NEW API: DataFrame-first plotting (recommended)
# ============================================================================

def plot_cluster_trajectories_df(
    df_interpolated: pd.DataFrame,
    cluster_labels: np.ndarray,
    embryo_ids: Optional[List[str]] = None,
    *,
    show_mean: bool = True,
    show_individual: bool = True,
    figsize: tuple = DEFAULT_FIGSIZE,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = DEFAULT_DPI
) -> plt.Figure:
    """
    Plot trajectories colored by cluster - DataFrame version.

    Uses the hpf column directly from the DataFrame, preserving correct time alignment.
    No manual padding or grid index tracking needed.

    Parameters
    ----------
    df_interpolated : pd.DataFrame
        Trajectory data with columns [embryo_id, hpf, metric_value]
        Should come from interpolate_to_common_grid_df()
    cluster_labels : np.ndarray
        Cluster assignment for each unique embryo_id (length = n_unique_embryos)
    embryo_ids : list of str, optional
        Embryo identifiers (for reference, computed from DataFrame if not provided)
    show_mean : bool
        Plot cluster mean trajectories in bold
    show_individual : bool
        Plot individual trajectories faintly in background
    figsize : tuple
        Figure size (width, height)
    save_path : str or Path, optional
        Path to save figure
    dpi : int
        Resolution for saved figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        The resulting figure object

    Examples
    --------
    >>> df_filtered = extract_trajectories_df(df)
    >>> df_interpolated = interpolate_to_common_grid_df(df_filtered)
    >>> trajectories, ids, grid = df_to_trajectories(df_interpolated)
    >>> D = compute_dtw_distance_matrix(trajectories)
    >>> results = run_bootstrap_hierarchical(D, k=3, embryo_ids=ids)
    >>> posteriors = analyze_bootstrap_results(results)
    >>>
    >>> # Plot using DataFrame - time column preserved!
    >>> fig = plot_cluster_trajectories_df(
    ...     df_interpolated,
    ...     posteriors['modal_cluster'],
    ...     embryo_ids=ids
    ... )
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Use provided embryo_ids (aligned with cluster_labels) or extract from DataFrame
    if embryo_ids is not None:
        unique_embryo_ids = np.array(embryo_ids)
    else:
        # Fallback: extract from DataFrame in sorted order for determinism
        # WARNING: cluster_labels must be aligned with this sorted order!
        unique_embryo_ids = np.array(sorted(df_interpolated['embryo_id'].unique()))

    # Validate alignment
    if len(unique_embryo_ids) != len(cluster_labels):
        raise ValueError(
            f"Mismatch between number of embryos ({len(unique_embryo_ids)}) "
            f"and cluster_labels length ({len(cluster_labels)}). "
            f"Ensure cluster_labels is aligned with embryo_ids."
        )

    n_clusters = int(np.max(cluster_labels)) + 1
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_clusters, 10)))
    if n_clusters > 10:
        colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))

    # Plot individual trajectories
    if show_individual:
        for i, embryo_id in enumerate(unique_embryo_ids):
            subset = df_interpolated[df_interpolated['embryo_id'] == embryo_id]
            cluster = cluster_labels[i]
            ax.plot(subset['hpf'], subset['metric_value'],
                   color=colors[cluster], alpha=0.3, linewidth=0.8)

    # Plot cluster means
    if show_mean:
        for c in range(n_clusters):
            mask = cluster_labels == c
            cluster_embryos = unique_embryo_ids[mask]  # Now works with numpy array
            cluster_data = df_interpolated[
                df_interpolated['embryo_id'].isin(cluster_embryos)
            ]
            mean_traj = cluster_data.groupby('hpf')['metric_value'].mean()
            ax.plot(mean_traj.index, mean_traj.values,
                   color=colors[c], linewidth=2.5, label=f'Cluster {c}')

    ax.set_xlabel('HPF', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Trajectories by Cluster', fontsize=14, fontweight='bold')
    if show_mean:
        ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig


def plot_membership_trajectories_df(
    df_interpolated: pd.DataFrame,
    classification: Dict[str, Any],
    *,
    per_cluster: bool = True,
    figsize: tuple = (15, 10),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = DEFAULT_DPI
) -> plt.Figure:
    """
    Plot trajectories colored by membership category - DataFrame version.

    Uses the hpf column directly from DataFrame, preserving correct time alignment.
    Trajectories are colored by membership quality (core/uncertain/outlier).

    Parameters
    ----------
    df_interpolated : pd.DataFrame
        Trajectory data with columns [embryo_id, hpf, metric_value]
    classification : dict
        Output from classify_membership_2d() with keys:
        'embryo_ids' (required), 'category', 'cluster'.
        Note: 'embryo_ids' must be present and aligned with category/cluster arrays
        to ensure correct trajectory-to-category mapping
    per_cluster : bool
        If True, create one subplot per cluster showing membership breakdown
    figsize : tuple
        Figure size (width, height)
    save_path : str or Path, optional
        Path to save figure
    dpi : int
        Resolution for saved figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        The resulting figure object

    Examples
    --------
    >>> classification = classify_membership_2d(...)
    >>> fig = plot_membership_trajectories_df(df_interpolated, classification)
    """
    categories = classification['category']
    clusters = classification['cluster']

    # Get embryo_ids from classification or warn if missing
    if 'embryo_ids' not in classification:
        warnings.warn(
            "classification dict missing 'embryo_ids' key. Falling back to sorted "
            "embryo IDs from DataFrame, but this may not match the classification "
            "order, causing incorrect trajectory-to-category mapping. "
            "Ensure classify_membership_2d() includes 'embryo_ids' in output.",
            UserWarning,
            stacklevel=2
        )
        embryo_ids = np.array(sorted(df_interpolated['embryo_id'].unique()))
    else:
        embryo_ids = np.array(classification['embryo_ids'])

    # Validate alignment
    if len(embryo_ids) != len(categories) or len(embryo_ids) != len(clusters):
        raise ValueError(
            f"Mismatch in classification dict: embryo_ids length ({len(embryo_ids)}) "
            f"!= categories length ({len(categories)}) or clusters length ({len(clusters)}). "
            f"All arrays must be aligned."
        )

    n_clusters = int(np.max(clusters)) + 1

    if per_cluster:
        # Create n_clusters rows × 3 columns (one column per membership category)
        if figsize == (15, 10):  # Default figsize
            figsize = (15, 4 * n_clusters)

        fig, axes = plt.subplots(n_clusters, 3, figsize=figsize, sharey=True)
        if n_clusters == 1:
            axes = axes.reshape(1, -1)  # Ensure 2D shape for consistency

        for c in range(n_clusters):
            for cat_idx, category in enumerate(['core', 'uncertain', 'outlier']):
                ax = axes[c, cat_idx]

                cat_mask = (clusters == c) & (categories == category)
                if np.sum(cat_mask) > 0:
                    cat_embryo_ids = np.array(embryo_ids)[cat_mask]
                    cat_data = df_interpolated[
                        df_interpolated['embryo_id'].isin(cat_embryo_ids)
                    ]

                    color = MEMBERSHIP_COLORS.get(category, 'gray')
                    for embryo_id in cat_embryo_ids:
                        subset = cat_data[cat_data['embryo_id'] == embryo_id]
                        ax.plot(subset['hpf'], subset['metric_value'],
                               color=color, alpha=0.5, linewidth=1.0)

                # Titles and labels
                if c == 0:
                    ax.set_title(f'{category.capitalize()}', fontweight='bold', fontsize=11)
                if cat_idx == 0:
                    ax.set_ylabel(f'Cluster {c}\nMetric Value', fontsize=10, fontweight='bold')
                else:
                    ax.set_ylabel('')

                if c == n_clusters - 1:
                    ax.set_xlabel('HPF', fontsize=10)

                ax.grid(True, alpha=0.3)

        # Add suptitle
        fig.suptitle('Membership Trajectories by Cluster and Category', fontsize=14, fontweight='bold', y=0.995)

    else:
        fig, ax = plt.subplots(figsize=figsize)

        for category in ['outlier', 'uncertain', 'core']:
            cat_mask = categories == category
            if np.sum(cat_mask) > 0:
                cat_embryo_ids = np.array(embryo_ids)[cat_mask]
                cat_data = df_interpolated[
                    df_interpolated['embryo_id'].isin(cat_embryo_ids)
                ]

                color = MEMBERSHIP_COLORS.get(category, 'gray')
                for embryo_id in cat_embryo_ids:
                    subset = cat_data[cat_data['embryo_id'] == embryo_id]
                    label = category if embryo_id == cat_embryo_ids[0] else ''
                    ax.plot(subset['hpf'], subset['metric_value'],
                           color=color, alpha=0.4, linewidth=0.8, label=label)

        ax.set_xlabel('HPF', fontsize=12)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_title('Trajectories by Membership Category', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig


# ============================================================================
# LEGACY API: Array-based plotting (deprecated)
# ============================================================================


def plot_posterior_heatmap(
    posterior_analysis: Dict[str, Any],
    embryo_ids: Optional[List[str]] = None,
    *,
    figsize: tuple = (12, 8),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = DEFAULT_DPI
) -> plt.Figure:
    """
    Plot posterior probability heatmap.

    Rows = embryos, Columns = clusters
    Color intensity = p_i(c)

    Parameters
    ----------
    posterior_analysis : dict
        Output from analyze_bootstrap_results()
    embryo_ids : list of str, optional
        Embryo identifiers for y-axis labels
    figsize : tuple
        Figure size (width, height)
    save_path : str or Path, optional
        Path to save figure
    dpi : int
        Resolution for saved figure

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    p_matrix = posterior_analysis['p_matrix']
    n_embryos, n_clusters = p_matrix.shape

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(p_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    # Labels
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Embryo', fontsize=12)
    ax.set_title('Posterior Probabilities p_i(c)', fontsize=14, fontweight='bold')

    # Ticks
    ax.set_xticks(np.arange(n_clusters))
    ax.set_xticklabels([f'C{i}' for i in range(n_clusters)])

    if embryo_ids is not None and len(embryo_ids) == n_embryos:
        ax.set_yticks(np.arange(n_embryos))
        ax.set_yticklabels(embryo_ids, fontsize=8)
    else:
        ax.set_yticks(np.arange(0, n_embryos, max(1, n_embryos // 20)))

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Probability', fontsize=11)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig


def plot_2d_scatter(
    classification: Dict[str, Any],
    embryo_ids: Optional[List[str]] = None,
    *,
    show_thresholds: bool = True,
    figsize: tuple = (10, 8),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = DEFAULT_DPI
) -> plt.Figure:
    """
    Plot 2D scatter of max_p vs log_odds_gap.

    Points colored by membership category (core/uncertain/outlier).
    Optionally shows threshold lines.

    Parameters
    ----------
    classification : dict
        Output from classify_membership_2d()
    embryo_ids : list of str, optional
        For labeling outlier points
    show_thresholds : bool
        Show threshold lines
    figsize : tuple
        Figure size
    save_path : str or Path, optional
        Path to save
    dpi : int
        Resolution

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    max_p = classification['max_p']
    log_odds_gap = classification['log_odds_gap']
    categories = classification['category']

    fig, ax = plt.subplots(figsize=figsize)

    # Plot by category
    for category in ['outlier', 'uncertain', 'core']:
        mask = categories == category
        if np.sum(mask) > 0:
            color = MEMBERSHIP_COLORS.get(category, 'gray')
            ax.scatter(max_p[mask], log_odds_gap[mask], label=category, alpha=0.6, s=50, color=color)

    # Threshold lines
    if show_thresholds:
        thresholds = classification.get('thresholds', {})
        max_p_thresh = thresholds.get('threshold_max_p', 0.8)
        log_odds_thresh = thresholds.get('threshold_log_odds_gap', 0.7)
        outlier_thresh = thresholds.get('threshold_outlier_max_p', 0.5)

        ax.axvline(max_p_thresh, color='red', linestyle='--', alpha=0.5, label=f'max_p = {max_p_thresh}')
        ax.axvline(outlier_thresh, color='orange', linestyle='--', alpha=0.5, label=f'outlier = {outlier_thresh}')
        ax.axhline(log_odds_thresh, color='green', linestyle='--', alpha=0.5, label=f'log_odds = {log_odds_thresh}')

    ax.set_xlabel('Max Posterior Probability', fontsize=12)
    ax.set_ylabel('Log-Odds Gap', fontsize=12)
    ax.set_title('2D Membership Classification', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig


def plot_membership_vs_k(
    results: Dict[str, Any],
    genotype: Optional[str] = None,
    figsize: tuple = (14, 5.5),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = DEFAULT_DPI
) -> plt.Figure:
    """
    Plot membership category percentages as k varies.

    Shows line plot and stacked area chart tracking how core/uncertain/outlier
    proportions change across different k values.

    Parameters
    ----------
    results : dict
        Multi-k analysis results with structure:
        {
            'results': {
                k1: {'classification': {'category': np.ndarray, ...}},
                k2: {'classification': {'category': np.ndarray, ...}},
                ...
            },
            'genotype': str (optional)
        }
        Each k should have a 'classification' dict from classify_membership_2d()
    genotype : str, optional
        Genotype name for plot title (overrides results['genotype'] if provided)
    figsize : tuple
        Figure size (width, height). Default (14, 5.5)
    save_path : str or Path, optional
        Path to save figure
    dpi : int
        Resolution for saved figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Extract k values and compute percentages
    k_values = sorted(results['results'].keys())
    core_pcts = []
    uncertain_pcts = []
    outlier_pcts = []

    for k in k_values:
        if results['results'][k] is None:
            continue

        classification = results['results'][k]['classification']
        categories = classification['category']

        n_core = np.sum(categories == 'core')
        n_uncertain = np.sum(categories == 'uncertain')
        n_outlier = np.sum(categories == 'outlier')
        n_total = n_core + n_uncertain + n_outlier

        if n_total > 0:
            core_pcts.append(100.0 * n_core / n_total)
            uncertain_pcts.append(100.0 * n_uncertain / n_total)
            outlier_pcts.append(100.0 * n_outlier / n_total)
        else:
            core_pcts.append(0)
            uncertain_pcts.append(0)
            outlier_pcts.append(0)

    # Create single figure with line plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot membership trends across k values
    ax.plot(k_values, core_pcts, 'o-', color='green', linewidth=2.5, markersize=8,
           label='Core', alpha=0.8)
    ax.plot(k_values, uncertain_pcts, 's-', color='orange', linewidth=2.5, markersize=8,
           label='Uncertain', alpha=0.8)
    ax.plot(k_values, outlier_pcts, '^-', color='red', linewidth=2.5, markersize=8,
           label='Outlier', alpha=0.8)

    ax.set_xlabel('k (number of clusters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Membership Category Trends Across k', fontsize=13, fontweight='bold')
    ax.set_xticks(k_values)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])

    # Main title
    title = 'Membership Distribution Across K Values'
    if genotype is None:
        genotype = results.get('genotype')
    if genotype:
        title = f'{genotype}: {title}'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig


def plot_cluster_trajectories(
    trajectories: List[np.ndarray],
    common_grid: np.ndarray,
    cluster_labels: np.ndarray,
    embryo_ids: Optional[List[str]] = None,
    *,
    show_mean: bool = True,
    show_individual: bool = True,
    figsize: tuple = DEFAULT_FIGSIZE,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = DEFAULT_DPI
) -> plt.Figure:
    """
    DEPRECATED: Use plot_cluster_trajectories_df() instead.

    Plot trajectories colored by cluster assignment (legacy array API).

    This function has a time-axis alignment bug: trimmed arrays don't preserve
    each trajectory's start time, causing late-start embryos to be plotted against
    wrong time values. Use plot_cluster_trajectories_df() with DataFrame input
    to preserve correct time alignment.
    """
    warnings.warn(
        "plot_cluster_trajectories() is deprecated. Use plot_cluster_trajectories_df() instead, "
        "which uses the DataFrame's hpf column to preserve correct time alignment. "
        "See migration guide in README.md",
        DeprecationWarning,
        stacklevel=2
    )

    n_clusters = int(np.max(cluster_labels)) + 1
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    fig, ax = plt.subplots(figsize=figsize)

    # Plot individual trajectories
    if show_individual:
        for i, traj in enumerate(trajectories):
            cluster = cluster_labels[i]
            grid_subset = common_grid[:len(traj)]
            ax.plot(grid_subset, traj, color=colors[cluster], alpha=0.3, linewidth=0.8)

    # Plot cluster means
    if show_mean:
        for c in range(n_clusters):
            mask = cluster_labels == c
            if np.sum(mask) > 0:
                cluster_trajs = [trajectories[i] for i in np.where(mask)[0]]
                # Compute mean (handling variable lengths)
                min_len = min([len(t) for t in cluster_trajs])
                mean_traj = np.mean([t[:min_len] for t in cluster_trajs], axis=0)
                grid_subset = common_grid[:min_len]
                ax.plot(grid_subset, mean_traj, color=colors[c], linewidth=2.5, label=f'Cluster {c}')

    ax.set_xlabel('HPF', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Trajectories by Cluster', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig


def plot_membership_trajectories(
    trajectories: List[np.ndarray],
    common_grid: np.ndarray,
    classification: Dict[str, Any],
    *,
    per_cluster: bool = True,
    figsize: tuple = (15, 10),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = DEFAULT_DPI
) -> plt.Figure:
    """
    Plot trajectories colored by membership category (core/uncertain/outlier).

    If per_cluster=True, creates panel grid with one subplot per cluster.

    .. deprecated:: 0.2.0
        Use :func:`plot_membership_trajectories_df` instead. This function has a
        time-axis alignment bug where late-start trajectories are plotted against
        incorrect times. The new version uses the DataFrame's hpf column to preserve
        correct alignment. See migration guide in README.md

    Parameters
    ----------
    trajectories : list of np.ndarray
        Individual trajectories
    common_grid : np.ndarray
        Common time grid
    classification : dict
        Output from classify_membership_2d()
    per_cluster : bool
        Create per-cluster panels
    figsize : tuple
        Figure size
    save_path : str or Path, optional
        Path to save
    dpi : int
        Resolution

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    warnings.warn(
        "plot_membership_trajectories() is deprecated (v0.2.0). Use plot_membership_trajectories_df() instead, "
        "which uses the DataFrame's hpf column to preserve correct time alignment. "
        "See migration guide in README.md",
        DeprecationWarning,
        stacklevel=2
    )
    categories = classification['category']
    clusters = classification['cluster']
    n_clusters = int(np.max(clusters)) + 1

    if per_cluster:
        fig, axes = plt.subplots(1, n_clusters, figsize=figsize, sharey=True)
        if n_clusters == 1:
            axes = [axes]

        for c in range(n_clusters):
            ax = axes[c]
            mask = clusters == c

            for category in ['outlier', 'uncertain', 'core']:
                cat_mask = mask & (categories == category)
                if np.sum(cat_mask) > 0:
                    color = MEMBERSHIP_COLORS.get(category, 'gray')
                    for i in np.where(cat_mask)[0]:
                        traj = trajectories[i]
                        grid_subset = common_grid[:len(traj)]
                        ax.plot(grid_subset, traj, color=color, alpha=0.4, linewidth=0.8)

            ax.set_title(f'Cluster {c}', fontweight='bold')
            ax.set_xlabel('HPF')
            if c == 0:
                ax.set_ylabel('Metric Value')
            ax.grid(True, alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=MEMBERSHIP_COLORS.get(cat, 'gray'), label=cat.capitalize())
                          for cat in ['core', 'uncertain', 'outlier']]
        fig.legend(handles=legend_elements, loc='upper right')

    else:
        fig, ax = plt.subplots(figsize=figsize)

        for category in ['outlier', 'uncertain', 'core']:
            mask = categories == category
            if np.sum(mask) > 0:
                color = MEMBERSHIP_COLORS.get(category, 'gray')
                for i in np.where(mask)[0]:
                    traj = trajectories[i]
                    grid_subset = common_grid[:len(traj)]
                    ax.plot(grid_subset, traj, color=color, alpha=0.4, linewidth=0.8, label=category if i == np.where(mask)[0][0] else '')

        ax.set_xlabel('HPF', fontsize=12)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_title('Trajectories by Membership Category', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig
