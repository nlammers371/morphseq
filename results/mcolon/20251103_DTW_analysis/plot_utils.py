"""Shared plotting utilities for DTW clustering pipeline."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import linregress
from pathlib import Path


# ============================================================================
# PLOTTING PARAMETERS
# ============================================================================

# Color palettes
CLUSTER_CMAP = plt.cm.tab10
INDIVIDUAL_COLOR = 'gray'
MEAN_COLOR = 'black'
SD_COLOR = 'blue'
FIT_COLOR = 'red'

# Transparency/alpha values
INDIVIDUAL_ALPHA = 0.3
SD_ALPHA = 0.18
FIT_ALPHA = 0.8

# Membership-based transparency values
MEMBERSHIP_CORE_ALPHA = 0.8
MEMBERSHIP_UNCERTAIN_ALPHA = 0.8
MEMBERSHIP_OUTLIER_ALPHA = 0.8

# Membership colors
MEMBERSHIP_CORE_COLOR = '#2ecc71'      # Green
MEMBERSHIP_UNCERTAIN_COLOR = '#f39c12'  # Orange
MEMBERSHIP_OUTLIER_COLOR = '#e74c3c'   # Red

# Line widths
INDIVIDUAL_LW = 0.8
MEAN_LW = 2.8
FIT_LW = 2.0
MEMBERSHIP_LW = 1.2

# Default figure parameters
DEFAULT_DPI = 100
COMPARISON_DPI = 200


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_cluster_colors(n_clusters):
    """Get color palette for clusters."""
    colors = CLUSTER_CMAP(np.linspace(0, 1, min(n_clusters, 10)))
    return {i: colors[i] for i in range(n_clusters)}


def compute_trajectory_stats(trajectories):
    """
    Compute mean and standard deviation for trajectories.

    Parameters
    ----------
    trajectories : np.ndarray
        (n_trajectories, n_timepoints) array of pre-padded trajectories.
        Should be uniform length (time-aligned with NaN padding).

    Returns
    -------
    mean : np.ndarray
        Mean trajectory (with NaN for positions with no data)
    std : np.ndarray
        Standard deviation (with NaN for positions with no data)
    """
    # Ensure it's a 2D array
    if isinstance(trajectories, list):
        # Convert list of arrays to 2D array - assumes padding was done elsewhere
        max_len = max([len(t) for t in trajectories]) if trajectories else 0
        padded = np.full((len(trajectories), max_len), np.nan)
        for i, traj in enumerate(trajectories):
            padded[i, :len(traj)] = traj
    else:
        padded = trajectories

    mean = np.nanmean(padded, axis=0)
    std = np.nanstd(padded, axis=0)
    return mean, std


def fit_linear_regression(y, return_r2=True):
    """
    Fit linear regression to y values.

    Parameters
    ----------
    y : np.ndarray
        Y values (may contain NaN)
    return_r2 : bool
        Return R² value

    Returns
    -------
    fit_line : np.ndarray
        Fitted values
    r2 : float
        R-squared value (or np.nan if < 2 valid points)
    """
    valid_mask = ~np.isnan(y)

    if np.sum(valid_mask) < 2:
        fit_line = np.full_like(y, np.nan)
        r2 = np.nan
    else:
        x_valid = np.arange(len(y))[valid_mask]
        y_valid = y[valid_mask]
        slope, intercept, r_value, p_value, std_err = linregress(x_valid, y_valid)

        fit_line = slope * np.arange(len(y)) + intercept
        r2 = r_value ** 2

    return fit_line, r2


def get_shared_ylims(all_trajectories, std_trajectories, margin_fraction=0.1):
    """
    Compute shared y-axis limits for fair comparison.

    Parameters
    ----------
    all_trajectories : list of arrays
        All trajectory arrays (to include in y-limits)
    std_trajectories : list of arrays
        Standard deviation arrays (to include error bands)
    margin_fraction : float
        Add margin as fraction of range

    Returns
    -------
    y_min, y_max : tuple of float
        Y-axis limits with margin
    """
    all_values = []

    # Include trajectories
    for traj in all_trajectories:
        if isinstance(traj, list):
            for t in traj:
                all_values.extend(t[~np.isnan(t)])
        else:
            all_values.extend(traj[~np.isnan(traj)])

    # Include error bands
    for std_vals in std_trajectories:
        if isinstance(std_vals, list):
            for s in std_vals:
                all_values.extend(s[~np.isnan(s)])
        else:
            all_values.extend(std_vals[~np.isnan(std_vals)])

    y_min = np.nanmin(all_values) if all_values else 0
    y_max = np.nanmax(all_values) if all_values else 1

    y_range = y_max - y_min
    if y_range == 0:
        y_range = 1

    margin = y_range * margin_fraction
    return y_min - margin, y_max + margin


# ============================================================================
# MAIN PLOTTING FUNCTIONS
# ============================================================================

def plot_distance_matrix(D, labels=None, title="DTW Distance Matrix", dpi=200):
    """
    Plot DTW distance matrix as heatmap.

    Parameters
    ----------
    D : np.ndarray
        Distance matrix (symmetric)
    labels : np.ndarray, optional
        Cluster labels (if provided, sort by cluster)
    title : str
        Figure title
    dpi : int
        Figure DPI

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)

    # Sort by cluster if labels provided
    if labels is not None:
        sort_idx = np.argsort(labels)
        D_sorted = D[np.ix_(sort_idx, sort_idx)]
        im = ax.imshow(D_sorted, cmap='viridis', aspect='auto')
    else:
        im = ax.imshow(D, cmap='viridis', aspect='auto')

    ax.set_xlabel('Embryo')
    ax.set_ylabel('Embryo')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='DTW Distance')

    plt.tight_layout()
    return fig


def plot_silhouette_scores(silhouette_dict, title="Silhouette Scores by K"):
    """
    Plot silhouette scores for different k values.

    Parameters
    ----------
    silhouette_dict : dict
        k -> silhouette_score mapping
    title : str
        Figure title

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

    k_values = sorted(silhouette_dict.keys())
    scores = [silhouette_dict[k] for k in k_values]

    ax.plot(k_values, scores, 'o-', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)

    plt.tight_layout()
    return fig


def plot_temporal_trends_by_cluster(trajectories_by_cluster, common_grid, k, title=None, dpi=100):
    """
    Plot temporal trends for each cluster in a grid of subplots.

    Ported from old analysis script. Shows individual trajectories, mean, ±SD, and linear fit
    for each cluster in separate subplots.

    Parameters
    ----------
    trajectories_by_cluster : dict
        cluster_id -> list of trajectory arrays
    common_grid : np.ndarray
        Time points
    k : int
        Number of clusters (for title)
    title : str, optional
        Figure title
    dpi : int
        Figure DPI

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    if title is None:
        title = f"Temporal Trends by Cluster (k={k})"

    n_clusters = len(trajectories_by_cluster)
    n_cols = min(3, n_clusters)
    n_rows = int(np.ceil(n_clusters / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), dpi=dpi)
    if n_clusters == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten() if n_clusters > 1 else np.array([axes])

    # Compute shared y-limits across all clusters
    all_trajs = []
    all_stds = []
    for trajs in trajectories_by_cluster.values():
        all_trajs.extend(trajs)
        mean_t, std_t = compute_trajectory_stats(trajs)
        all_stds.append(mean_t + std_t)
        all_stds.append(mean_t - std_t)

    y_min, y_max = get_shared_ylims(all_trajs, all_stds)

    # Plot each cluster
    for i, cluster_id in enumerate(sorted(trajectories_by_cluster.keys())):
        ax = axes[i]
        trajs = trajectories_by_cluster[cluster_id]

        # Compute statistics
        mean_traj, std_traj = compute_trajectory_stats(trajs)
        fit_line, r2 = fit_linear_regression(mean_traj)

        # Plot individual trajectories
        for traj in trajs:
            ax.plot(common_grid, traj, color=INDIVIDUAL_COLOR, alpha=INDIVIDUAL_ALPHA,
                   linewidth=INDIVIDUAL_LW)

        # Plot mean
        ax.plot(common_grid, mean_traj, color=MEAN_COLOR, linewidth=MEAN_LW, zorder=5)

        # Plot ±SD band
        ax.fill_between(common_grid, mean_traj - std_traj, mean_traj + std_traj,
                       color=SD_COLOR, alpha=SD_ALPHA, zorder=3)

        # Plot linear fit
        ax.plot(common_grid, fit_line, color=FIT_COLOR, linestyle='--', linewidth=FIT_LW,
               alpha=FIT_ALPHA, zorder=4, label=f'Linear (R²={r2:.3f})')

        ax.set_xlabel('Time (HPF)', fontsize=10)
        ax.set_ylabel('Metric Value', fontsize=10)
        ax.set_title(f'Cluster {cluster_id} (n={len(trajs)})', fontsize=11, fontweight='bold')
        ax.set_ylim([y_min, y_max])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)

    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_cluster_trajectory_overlay(trajectories_by_cluster, common_grid, k, title=None, dpi=100):
    """
    Plot trajectories from all clusters overlaid together for comparison.

    Ported from old analysis script. Shows colored individual trajectories and colored
    means with ±SD bands in two-panel comparison.

    Parameters
    ----------
    trajectories_by_cluster : dict
        cluster_id -> list of trajectory arrays
    common_grid : np.ndarray
        Time points
    k : int
        Number of clusters (for title)
    title : str, optional
        Figure title
    dpi : int
        Figure DPI

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    if title is None:
        title = f"Cluster Trajectory Overlay (k={k})"

    n_clusters = len(trajectories_by_cluster)
    cluster_colors = get_cluster_colors(n_clusters)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=dpi)

    # Compute shared y-limits
    all_trajs = []
    all_stds = []
    for trajs in trajectories_by_cluster.values():
        all_trajs.extend(trajs)
        mean_t, std_t = compute_trajectory_stats(trajs)
        all_stds.append(mean_t + std_t)
        all_stds.append(mean_t - std_t)

    y_min, y_max = get_shared_ylims(all_trajs, all_stds)

    # ========== LEFT PANEL: Individual trajectories by cluster ==========
    ax = axes[0]
    for cluster_id, trajs in trajectories_by_cluster.items():
        color = cluster_colors[cluster_id]
        for traj in trajs:
            ax.plot(common_grid, traj, color=color, alpha=INDIVIDUAL_ALPHA, linewidth=INDIVIDUAL_LW)

    # Overlay means
    for cluster_id, trajs in trajectories_by_cluster.items():
        mean_traj, _ = compute_trajectory_stats(trajs)
        color = cluster_colors[cluster_id]
        ax.plot(common_grid, mean_traj, color=color, linewidth=MEAN_LW, label=f'Cluster {cluster_id}')

    ax.set_xlabel('Time (HPF)', fontsize=11)
    ax.set_ylabel('Metric Value', fontsize=11)
    ax.set_title('Individual Trajectories', fontsize=12, fontweight='bold')
    ax.set_ylim([y_min, y_max])
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # ========== RIGHT PANEL: Mean trajectories with SD and fits ==========
    ax = axes[1]
    for cluster_id, trajs in trajectories_by_cluster.items():
        mean_traj, std_traj = compute_trajectory_stats(trajs)
        color = cluster_colors[cluster_id]

        # Plot mean
        ax.plot(common_grid, mean_traj, color=color, linewidth=MEAN_LW,
               label=f'Cluster {cluster_id}')

        # Plot ±SD band
        ax.fill_between(common_grid, mean_traj - std_traj, mean_traj + std_traj,
                       color=color, alpha=SD_ALPHA)

        # Plot linear fit
        fit_line, r2 = fit_linear_regression(mean_traj)
        ax.plot(common_grid, fit_line, color=color, linestyle='--', linewidth=FIT_LW,
               alpha=FIT_ALPHA)

    ax.set_xlabel('Time (HPF)', fontsize=11)
    ax.set_ylabel('Metric Value', fontsize=11)
    ax.set_title('Mean Trajectories with ±1 SD', fontsize=12, fontweight='bold')
    ax.set_ylim([y_min, y_max])
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_temporal_trends_with_membership(trajectories_by_cluster, common_grid, k,
                                         membership_classification, cluster_indices_map,
                                         title=None, dpi=100):
    """
    Plot temporal trends for each cluster with trajectories colored by membership status.

    Shows individual trajectories colored and styled by membership category (core/uncertain/outlier),
    mean, ±SD, and linear fit for each cluster in separate subplots.

    Parameters
    ----------
    trajectories_by_cluster : dict
        cluster_id -> list of trajectory arrays (pre-padded to uniform length)
    common_grid : np.ndarray
        Time points
    k : int
        Number of clusters (for title)
    membership_classification : dict
        From analyze_membership(), maps global embryo index → {
            'category': 'core'/'uncertain'/'outlier',
            'cluster': int,
            'intra_coassoc': float,
            'silhouette': float
        }
    cluster_indices_map : dict
        Maps cluster_id → list of global embryo indices in that cluster
    title : str, optional
        Figure title
    dpi : int
        Figure DPI

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    if title is None:
        title = f"Temporal Trends by Cluster with Membership (k={k})"

    n_clusters = len(trajectories_by_cluster)
    # Single row layout for side-by-side comparison
    n_cols = n_clusters
    n_rows = 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5), dpi=dpi)
    if n_clusters == 1:
        axes = np.array([axes])
    else:
        axes = axes if isinstance(axes, np.ndarray) else np.array([axes])

    # Compute shared y-limits across all clusters
    all_trajs = []
    all_stds = []
    for trajs in trajectories_by_cluster.values():
        all_trajs.extend(trajs)
        mean_t, std_t = compute_trajectory_stats(trajs)
        all_stds.append(mean_t + std_t)
        all_stds.append(mean_t - std_t)

    y_min, y_max = get_shared_ylims(all_trajs, all_stds)

    # Plot each cluster
    for i, cluster_id in enumerate(sorted(trajectories_by_cluster.keys())):
        ax = axes[i]
        trajs = trajectories_by_cluster[cluster_id]
        cluster_global_indices = cluster_indices_map[cluster_id]

        # Compute statistics
        mean_traj, std_traj = compute_trajectory_stats(trajs)
        fit_line, r2 = fit_linear_regression(mean_traj)

        # Plot individual trajectories colored by membership
        for local_idx, (traj, global_idx) in enumerate(zip(trajs, cluster_global_indices)):
            member_info = membership_classification.get(global_idx, {})
            category = member_info.get('category', 'uncertain')

            # Set color, alpha, and linestyle based on membership category
            if category == 'core':
                color = MEMBERSHIP_CORE_COLOR
                alpha = MEMBERSHIP_CORE_ALPHA
                linestyle = '-'
            elif category == 'outlier':
                color = MEMBERSHIP_OUTLIER_COLOR
                alpha = MEMBERSHIP_OUTLIER_ALPHA
                linestyle = ':'
            else:  # uncertain
                color = MEMBERSHIP_UNCERTAIN_COLOR
                alpha = MEMBERSHIP_UNCERTAIN_ALPHA
                linestyle = '--'

            ax.plot(common_grid, traj, color=color, alpha=alpha, linewidth=MEMBERSHIP_LW,
                   linestyle=linestyle)

        # Plot mean
        ax.plot(common_grid, mean_traj, color=MEAN_COLOR, linewidth=MEAN_LW, zorder=5)

        # Plot ±SD band
        ax.fill_between(common_grid, mean_traj - std_traj, mean_traj + std_traj,
                       color=SD_COLOR, alpha=SD_ALPHA, zorder=3)

        # Plot linear fit
        ax.plot(common_grid, fit_line, color=FIT_COLOR, linestyle='--', linewidth=FIT_LW,
               alpha=FIT_ALPHA, zorder=4, label=f'Linear (R²={r2:.3f})')

        ax.set_xlabel('Time (HPF)', fontsize=10)
        ax.set_ylabel('Metric Value', fontsize=10)
        ax.set_title(f'Cluster {cluster_id} (n={len(trajs)})', fontsize=11, fontweight='bold')
        ax.set_ylim([y_min, y_max])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)

    # Add membership legend above the plots
    core_patch = mpatches.Patch(color=MEMBERSHIP_CORE_COLOR, label='Core', alpha=MEMBERSHIP_CORE_ALPHA)
    uncertain_patch = mpatches.Patch(color=MEMBERSHIP_UNCERTAIN_COLOR, label='Uncertain', alpha=MEMBERSHIP_UNCERTAIN_ALPHA)
    outlier_patch = mpatches.Patch(color=MEMBERSHIP_OUTLIER_COLOR, label='Outlier', alpha=MEMBERSHIP_OUTLIER_ALPHA)

    fig.legend(handles=[core_patch, uncertain_patch, outlier_patch],
              loc='upper center', ncol=3, fontsize=11,
              bbox_to_anchor=(0.5, 1.02), frameon=True)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    # Test utilities
    print("Plot utilities loaded successfully")
