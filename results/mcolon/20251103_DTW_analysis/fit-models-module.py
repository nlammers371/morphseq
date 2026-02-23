# 4_fit_models.py
"""Fit mixed-effects models to trajectory clusters."""

import sys
from pathlib import Path
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
from scipy.stats import linregress
from typing import Dict, Tuple, List
import warnings

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# Import DBA from src utilities
from src.analyze.dtw_time_trend_analysis import compute_dtw_distance
from src.analyze.dtw_time_trend_analysis.dtw_clustering import dba

# Try to import plot_utils, but don't fail if not available
try:
    from plot_utils import compute_trajectory_stats, fit_linear_regression
except ImportError:
    # Define local versions if plot_utils not available
    def compute_trajectory_stats(trajectories):
        """Compute mean and std for trajectories."""
        if isinstance(trajectories, list):
            max_len = max([len(t) for t in trajectories])
            padded = np.full((len(trajectories), max_len), np.nan)
            for i, traj in enumerate(trajectories):
                padded[i, :len(traj)] = traj
        else:
            padded = trajectories
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        return mean, std

    def fit_linear_regression(y):
        """Fit linear regression to y."""
        valid_mask = ~np.isnan(y)
        if np.sum(valid_mask) < 2:
            return np.full_like(y, np.nan), np.nan
        x_valid = np.arange(len(y))[valid_mask]
        y_valid = y[valid_mask]
        slope, intercept, r_value, _, _ = linregress(x_valid, y_valid)
        fit_line = slope * np.arange(len(y)) + intercept
        return fit_line, r_value ** 2

# ============ CORE FUNCTIONS ============

def fit_spline(t: np.ndarray, y: np.ndarray, smoothing: float = 1.0,
               k: int = 3) -> UnivariateSpline:
    """Fit penalized spline to trajectory."""
    # Sort by time
    sort_idx = np.argsort(t)
    t_sorted = t[sort_idx]
    y_sorted = y[sort_idx]

    # Fit spline
    spline = UnivariateSpline(t_sorted, y_sorted, s=smoothing, k=k)
    return spline


def compute_dba_average(trajectories: List[np.ndarray], max_iter: int = 10,
                        smooth_sigma: float = 0.0, weights: np.ndarray = None):
    """
    Compute DBA (DTW Barycenter Average) of trajectories.

    Uses the efficient implementation from src utilities.

    Parameters
    ----------
    trajectories : List[np.ndarray]
        List of 1D trajectory arrays (possibly different lengths)
    max_iter : int
        Maximum iterations for DBA refinement
    smooth_sigma : float
        Gaussian smoothing sigma for output
    weights : np.ndarray, optional
        Per-trajectory weights for weighted averaging

    Returns
    -------
    np.ndarray
        Barycenter trajectory
    """
    # Define DTW function for DBA
    def dtw_func(seq1, seq2):
        # Return (path, distance) tuple expected by DBA
        # For now, use simple distance computation
        # In full implementation, would return actual alignment path
        dist = compute_dtw_distance(seq1, seq2, window=3)
        # Approximate path (not used in simplified DBA but required by interface)
        path = [(i, i) for i in range(min(len(seq1), len(seq2)))]
        return path, dist

    # Run DBA
    barycenter = dba(
        trajectories,
        dtw_func=dtw_func,
        weights=weights,
        max_iter=max_iter,
        smooth_sigma=smooth_sigma,
        verbose=False
    )

    return barycenter


def fit_random_effects(t: np.ndarray, y: np.ndarray, 
                      mean_curve: callable) -> Dict:
    """Estimate random intercept and slope for individual."""
    # Compute residuals from mean
    y_mean = mean_curve(t)
    residuals = y - y_mean
    
    # Simple linear regression on residuals
    X = np.column_stack([np.ones_like(t), t])
    coeffs = np.linalg.lstsq(X, residuals, rcond=None)[0]
    
    b0 = coeffs[0]  # Random intercept
    b1 = coeffs[1]  # Random slope
    
    # Residual variance
    y_pred = y_mean + b0 + b1 * t
    sigma2 = np.var(y - y_pred)
    
    return {'b0': b0, 'b1': b1, 'sigma2': sigma2}


def estimate_variance_components(random_effects: List[Dict]) -> Dict:
    """Estimate variance-covariance of random effects."""
    b0_vals = [re['b0'] for re in random_effects]
    b1_vals = [re['b1'] for re in random_effects]
    
    var_b0 = np.var(b0_vals)
    var_b1 = np.var(b1_vals)
    cov_b0_b1 = np.cov(b0_vals, b1_vals)[0, 1]
    
    # Pooled residual variance
    sigma2_pooled = np.mean([re['sigma2'] for re in random_effects])
    
    return {
        'var_b0': var_b0,
        'var_b1': var_b1,
        'cov_b0_b1': cov_b0_b1,
        'sigma2': sigma2_pooled,
        'cov_matrix': np.array([[var_b0, cov_b0_b1], 
                                [cov_b0_b1, var_b1]])
    }


# ============ WRAPPER FUNCTIONS ============

def fit_cluster_model(trajectories: List[np.ndarray], common_grid: np.ndarray = None,
                      core_mask: np.ndarray = None, use_dba: bool = False) -> Dict:
    """
    Fit mixed-effects model to a cluster.

    Parameters
    ----------
    trajectories : List[np.ndarray]
        List of trajectory value arrays (already on common grid from precomputation)
    common_grid : np.ndarray, optional
        Time points corresponding to trajectories
    core_mask : np.ndarray, optional
        Boolean mask for core members to use in fitting
    use_dba : bool
        Whether to compute DBA average as mean curve

    Returns
    -------
    dict
        Fitted model with splines, random effects, and statistics
    """
    if common_grid is None:
        # Create default grid if not provided
        common_grid = np.arange(len(trajectories[0]))

    # Center time
    t_mean = np.mean(common_grid)
    t_centered = common_grid - t_mean

    # Get trajectories to use for mean curve fitting (core if available, else all)
    if core_mask is not None and core_mask.any():
        core_trajectories = [traj for i, traj in enumerate(trajectories)
                            if i < len(core_mask) and core_mask[i]]
    else:
        core_trajectories = trajectories

    # Fit mean curve (two methods)
    # Method 1: Spline on pooled data
    if len(core_trajectories) == 0:
        # Fallback if core mask filtered everything out
        core_trajectories = trajectories

    t_pooled = np.concatenate([t_centered for _ in core_trajectories])
    y_pooled = np.concatenate(core_trajectories)

    # Sort for spline fitting
    sort_idx = np.argsort(t_pooled)
    t_pooled_sorted = t_pooled[sort_idx]
    y_pooled_sorted = y_pooled[sort_idx]
    mean_spline = fit_spline(t_pooled_sorted, y_pooled_sorted, smoothing=10.0)

    # Method 2: DBA (if requested)
    if use_dba:
        try:
            y_dba = compute_dba_average(core_trajectories, max_iter=10, smooth_sigma=0.0)
            # Fit spline to DBA curve
            dba_spline = fit_spline(t_centered, y_dba, smoothing=1.0)
        except Exception as e:
            print(f"Warning: DBA computation failed ({e}), using pooled spline only")
            dba_spline = None
    else:
        dba_spline = None

    # Fit random effects for all members
    random_effects = []
    for y in trajectories:
        re = fit_random_effects(t_centered, y, mean_spline)
        random_effects.append(re)

    # Estimate variance components
    var_components = estimate_variance_components(random_effects)

    # Compute fit statistics
    r2_values = []
    for y, re in zip(trajectories, random_effects):
        y_pred = mean_spline(t_centered) + re['b0'] + re['b1'] * t_centered
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        r2_values.append(r2)

    return {
        'mean_spline': mean_spline,
        'dba_spline': dba_spline,
        't_center': t_mean,
        'common_grid': common_grid,
        'random_effects': random_effects,
        'variance_components': var_components,
        'r2_values': np.array(r2_values),
        'mean_r2': np.mean(r2_values)
    }


def predict_trajectory(t_new: np.ndarray, cluster_model: Dict, 
                      b0: float = 0, b1: float = 0) -> np.ndarray:
    """Predict trajectory from fitted model."""
    t_centered = t_new - cluster_model['t_center']
    y_pred = cluster_model['mean_spline'](t_centered) + b0 + b1 * t_centered
    return y_pred


def compute_confidence_bands(t: np.ndarray, cluster_model: Dict, 
                            alpha: float = 0.05) -> Tuple:
    """Compute confidence bands for mean curve."""
    t_centered = t - cluster_model['t_center']
    y_mean = cluster_model['mean_spline'](t_centered)
    
    # Standard error from variance components
    var_b0 = cluster_model['variance_components']['var_b0']
    var_b1 = cluster_model['variance_components']['var_b1']
    cov = cluster_model['variance_components']['cov_b0_b1']
    sigma2 = cluster_model['variance_components']['sigma2']
    
    # Variance at each time point
    var_t = var_b0 + 2*cov*t_centered + var_b1*t_centered**2 + sigma2
    se = np.sqrt(var_t)
    
    # Normal approximation
    from scipy.stats import norm
    z = norm.ppf(1 - alpha/2)
    
    lower = y_mean - z * se
    upper = y_mean + z * se
    
    return y_mean, lower, upper


# ============ MODEL COMPARISON ============

def compare_spline_dba(cluster_model: Dict, t_eval: np.ndarray) -> float:
    """Compare spline and DBA mean curves."""
    if cluster_model['dba_spline'] is None:
        return np.nan

    t_centered = t_eval - cluster_model['t_center']
    y_spline = cluster_model['mean_spline'](t_centered)
    y_dba = cluster_model['dba_spline'](t_centered)

    rmse = np.sqrt(np.mean((y_spline - y_dba)**2))
    return rmse


# ============ PLOTTING FUNCTIONS ============

def plot_cluster_trajectories(trajectories, common_grid, cluster_id=0, cluster_indices=None,
                              membership_classification=None, title=None, dpi=100):
    """
    Plot individual trajectories in a cluster with mean ± SD and linear fit.
    Optionally highlight membership categories (core, uncertain, outlier).

    Parameters
    ----------
    trajectories : list of np.ndarray
        Individual trajectory arrays
    common_grid : np.ndarray
        Time points
    cluster_id : int
        Cluster identifier (for title)
    cluster_indices : array-like, optional
        Global indices of trajectories in this cluster (needed for membership lookup)
    membership_classification : dict, optional
        Classification dict from analyze_membership (index -> classification info)
    title : str, optional
        Figure title
    dpi : int
        Figure DPI

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    import matplotlib.pyplot as plt
    from scipy.stats import linregress

    if title is None:
        title = f"Cluster {cluster_id} Trajectories"

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=dpi)

    # Compute statistics
    mean_traj, std_traj = compute_trajectory_stats(trajectories)
    fit_line, r2 = fit_linear_regression(mean_traj)

    # Plot individual trajectories with membership coloring
    if membership_classification is not None and cluster_indices is not None:
        # Count membership categories for title annotation
        core_count = 0
        uncertain_count = 0
        outlier_count = 0

        for i, (idx, traj) in enumerate(zip(cluster_indices, trajectories)):
            if idx in membership_classification:
                cat = membership_classification[idx]['category']
                if cat == 'core':
                    ax.plot(common_grid, traj, color='green', alpha=0.5, linewidth=1.0)
                    core_count += 1
                elif cat == 'uncertain':
                    ax.plot(common_grid, traj, color='orange', alpha=0.35, linewidth=0.9, linestyle='--')
                    uncertain_count += 1
                elif cat == 'outlier':
                    ax.plot(common_grid, traj, color='red', alpha=0.25, linewidth=0.7, linestyle=':')
                    outlier_count += 1
                else:
                    ax.plot(common_grid, traj, color='gray', alpha=0.2, linewidth=0.8)
            else:
                ax.plot(common_grid, traj, color='gray', alpha=0.2, linewidth=0.8)

        # Add custom legend for membership categories
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', linewidth=2, label=f'Core ({core_count})'),
            Line2D([0], [0], color='orange', linewidth=2, linestyle='--', label=f'Uncertain ({uncertain_count})'),
            Line2D([0], [0], color='red', linewidth=2, linestyle=':', label=f'Outlier ({outlier_count})'),
            Line2D([0], [0], color='black', linewidth=2.5, label='Mean'),
            Line2D([0], [0], color='blue', linewidth=6, alpha=0.2, label='±1 SD'),
            Line2D([0], [0], color='red', linewidth=2, linestyle='--', label=f'Linear fit (R²={r2:.3f})')
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=9)

    else:
        # Original behavior: all gray
        for traj in trajectories:
            ax.plot(common_grid, traj, color='gray', alpha=0.3, linewidth=0.8)

        # Plot mean trajectory
        ax.plot(common_grid, mean_traj, color='black', linewidth=2.8, label='Mean')

        # Plot ±1 SD band
        ax.fill_between(common_grid, mean_traj - std_traj, mean_traj + std_traj,
                        color='blue', alpha=0.18, label='±1 SD')

        # Plot linear fit
        ax.plot(common_grid, fit_line, color='red', linestyle='--', linewidth=2.0,
                label=f'Linear fit (R²={r2:.3f})', alpha=0.8)

        ax.legend(loc='best', fontsize=10)

    # Always plot mean and SD bands
    ax.plot(common_grid, mean_traj, color='black', linewidth=2.8, zorder=10)
    ax.fill_between(common_grid, mean_traj - std_traj, mean_traj + std_traj,
                    color='blue', alpha=0.15, zorder=5)
    ax.plot(common_grid, fit_line, color='red', linestyle='--', linewidth=2.0, alpha=0.8, zorder=8)

    ax.set_xlabel('Time (HPF)', fontsize=11)
    ax.set_ylabel('Metric Value', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_cluster_comparison(trajectories_by_cluster, common_grid, title=None, dpi=100):
    """
    Plot cluster comparison with individual trajectories and means.

    Parameters
    ----------
    trajectories_by_cluster : dict
        cluster_id -> list of trajectory arrays
    common_grid : np.ndarray
        Time points
    title : str, optional
        Figure title
    dpi : int
        Figure DPI

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    import matplotlib.pyplot as plt
    from plot_utils import get_cluster_colors, compute_trajectory_stats, fit_linear_regression

    if title is None:
        title = "Cluster Trajectory Comparison"

    n_clusters = len(trajectories_by_cluster)
    cluster_colors = get_cluster_colors(n_clusters)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=dpi)

    # Compute shared y-limits (two-pass pattern)
    all_y_values = []
    for cluster_id, trajs in trajectories_by_cluster.items():
        mean_traj, std_traj = compute_trajectory_stats(trajs)
        # Include all trajectories in y-range calculation, not just first one
        for traj in trajs:
            all_y_values.extend(traj[~np.isnan(traj)])
        all_y_values.extend(mean_traj[~np.isnan(mean_traj)])
        all_y_values.extend((mean_traj + std_traj)[~np.isnan(mean_traj + std_traj)])
        all_y_values.extend((mean_traj - std_traj)[~np.isnan(mean_traj - std_traj)])

    y_min, y_max = np.nanmin(all_y_values), np.nanmax(all_y_values)
    y_range = y_max - y_min if y_max > y_min else 1
    y_margin = y_range * 0.1
    y_lim = (y_min - y_margin, y_max + y_margin)

    # ========== LEFT PANEL: Individual trajectories by cluster ==========
    ax = axes[0]
    for cluster_id, trajs in trajectories_by_cluster.items():
        color = cluster_colors[cluster_id]
        for traj in trajs:
            ax.plot(common_grid, traj, color=color, alpha=0.3, linewidth=1)

    # Overlay means
    for cluster_id, trajs in trajectories_by_cluster.items():
        mean_traj, _ = compute_trajectory_stats(trajs)
        color = cluster_colors[cluster_id]
        ax.plot(common_grid, mean_traj, color=color, linewidth=2.8,
               label=f'Cluster {cluster_id}')

    ax.set_xlabel('Time (HPF)', fontsize=11)
    ax.set_ylabel('Metric Value', fontsize=11)
    ax.set_title('Individual Trajectories', fontsize=12, fontweight='bold')
    ax.set_ylim(y_lim)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # ========== RIGHT PANEL: Mean trajectories with CI and fits ==========
    ax = axes[1]
    for cluster_id, trajs in trajectories_by_cluster.items():
        mean_traj, std_traj = compute_trajectory_stats(trajs)
        color = cluster_colors[cluster_id]

        # Plot mean
        ax.plot(common_grid, mean_traj, color=color, linewidth=2.8,
               label=f'Cluster {cluster_id}')

        # Plot ±1 SD band
        ax.fill_between(common_grid, mean_traj - std_traj, mean_traj + std_traj,
                       color=color, alpha=0.25)

        # Plot linear fit
        fit_line, r2 = fit_linear_regression(mean_traj)
        ax.plot(common_grid, fit_line, color=color, linestyle='--', linewidth=1.8,
               alpha=0.8)

    ax.set_xlabel('Time (HPF)', fontsize=11)
    ax.set_ylabel('Metric Value', fontsize=11)
    ax.set_title('Mean Trajectories with ±1 SD', fontsize=12, fontweight='bold')
    ax.set_ylim(y_lim)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_random_effects(random_effects, title="Random Effects Distribution"):
    """Scatter plot of random intercepts vs slopes."""
    pass

def plot_residuals(t, y, cluster_model, random_effect, title="Residual Analysis"):
    """Q-Q plot and residuals over time."""
    pass

def plot_dba_trajectory(cluster_trajectories, common_grid, dba_trajectory=None, cluster_id=0,
                       title=None, dpi=100):
    """
    Plot DBA trajectory overlay on individual cluster trajectories.

    Parameters
    ----------
    cluster_trajectories : list of np.ndarray
        Individual trajectory arrays for cluster
    common_grid : np.ndarray
        Time points
    dba_trajectory : np.ndarray, optional
        DBA barycenter trajectory (if None, computes simple mean)
    cluster_id : int
        Cluster identifier for title
    title : str, optional
        Figure title
    dpi : int
        Figure DPI

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    import matplotlib.pyplot as plt

    if title is None:
        title = f"Cluster {cluster_id} DBA Trajectory"

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=dpi)

    # Compute simple mean for comparison
    mean_traj, std_traj = compute_trajectory_stats(cluster_trajectories)

    # Plot individual trajectories
    for traj in cluster_trajectories:
        ax.plot(common_grid, traj, color='lightgray', alpha=0.2, linewidth=0.8)

    # Plot mean
    ax.plot(common_grid, mean_traj, color='blue', linewidth=2.8, label='Simple Mean',
           alpha=0.8, zorder=8)

    # Plot ±1 SD band
    ax.fill_between(common_grid, mean_traj - std_traj, mean_traj + std_traj,
                    color='blue', alpha=0.15, zorder=5, label='±1 SD (Mean)')

    # Plot DBA if provided
    if dba_trajectory is not None:
        ax.plot(common_grid, dba_trajectory, color='red', linewidth=3.0, label='DBA Barycenter',
               alpha=0.85, zorder=9)

        # Compute and display RMSE between mean and DBA
        valid_mask = ~(np.isnan(mean_traj) | np.isnan(dba_trajectory))
        if valid_mask.any():
            rmse = np.sqrt(np.mean((mean_traj[valid_mask] - dba_trajectory[valid_mask])**2))
            max_diff = np.max(np.abs(mean_traj[valid_mask] - dba_trajectory[valid_mask]))

            # Add text annotation
            ax.text(0.98, 0.05, f'RMSE: {rmse:.4f}\nMax diff: {max_diff:.4f}',
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   verticalalignment='bottom', horizontalalignment='right')

    ax.set_xlabel('Time (HPF)', fontsize=11)
    ax.set_ylabel('Metric Value', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_spline_vs_dba(t, cluster_model, title="Spline vs DBA", dpi=100):
    """
    Compare mean curve from spline fitting vs DBA method.

    Parameters
    ----------
    t : np.ndarray
        Time points for evaluation
    cluster_model : dict
        Fitted cluster model with mean_spline and dba_spline
    title : str, optional
        Figure title
    dpi : int
        Figure DPI

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    import matplotlib.pyplot as plt

    # Check if both methods are available
    if cluster_model['dba_spline'] is None:
        # Just plot spline
        fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)

        t_centered = t - cluster_model['t_center']
        y_spline = cluster_model['mean_spline'](t_centered)

        ax.plot(t, y_spline, 'b-', linewidth=2.5, label='Spline (pooled data)')
        ax.set_xlabel('Time (HPF)', fontsize=11)
        ax.set_ylabel('Metric Value', fontsize=11)
        ax.set_title(title + ' [DBA not computed]', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    # Both methods available - compare them
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=dpi)

    t_centered = t - cluster_model['t_center']
    y_spline = cluster_model['mean_spline'](t_centered)
    y_dba = cluster_model['dba_spline'](t_centered)

    # Left panel: Both curves
    ax = axes[0]
    ax.plot(t, y_spline, 'b-', linewidth=2.5, label='Spline (pooled)', alpha=0.8)
    ax.plot(t, y_dba, 'r--', linewidth=2.5, label='DBA (barycenter)', alpha=0.8)
    ax.fill_between(t, y_spline, y_dba, alpha=0.2, color='gray')
    ax.set_xlabel('Time (HPF)', fontsize=11)
    ax.set_ylabel('Metric Value', fontsize=11)
    ax.set_title('Mean Curves Comparison', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right panel: Difference
    ax = axes[1]
    difference = y_spline - y_dba
    rmse = np.sqrt(np.nanmean(difference**2))
    max_diff = np.nanmax(np.abs(difference))

    ax.plot(t, difference, 'purple', linewidth=2.0)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.fill_between(t, difference, 0, where=(difference >= 0), alpha=0.3, color='green', label='Spline > DBA')
    ax.fill_between(t, difference, 0, where=(difference < 0), alpha=0.3, color='red', label='DBA > Spline')

    ax.set_xlabel('Time (HPF)', fontsize=11)
    ax.set_ylabel('Difference (Spline - DBA)', fontsize=11)
    ax.set_title(f'Difference (RMSE={rmse:.4f}, Max={max_diff:.4f})', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig