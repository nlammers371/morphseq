"""
Generic data processing utilities for trajectory and time-series data.

Domain-agnostic functions for extracting and computing statistics on grouped time-series data.
These were extracted from pair_analysis to make them reusable across analysis modules.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy.ndimage import gaussian_filter1d


def get_trajectories_for_group(
    df: pd.DataFrame,
    filter_dict: Dict[str, Any],
    time_col: str = 'predicted_stage_hpf',
    metric_col: str = 'baseline_deviation_normalized',
    embryo_id_col: str = 'embryo_id',
    smooth_method: Optional[str] = 'gaussian',
    smooth_params: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[List[Dict]], Optional[np.ndarray], int]:
    """Extract trajectories for a specific group defined by filter conditions.

    Generic function that works with any DataFrame and column names. Groups rows by a
    user-specified ID column and extracts time-metric pairs for each group.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with time-series data.
    filter_dict : Dict[str, Any]
        Dictionary of {column_name: value} pairs to filter rows.
        All filters are AND-ed together.
    time_col : str, default='predicted_stage_hpf'
        Column name containing time values.
    metric_col : str, default='baseline_deviation_normalized'
        Column name containing metric/dependent variable values.
    embryo_id_col : str, default='embryo_id'
        Column name for grouping rows (typically embryo or sample ID).
    smooth_method : str or None, default='gaussian'
        Smoothing method for individual trajectories:
        - 'gaussian': Apply Gaussian filter (recommended)
        - None: No smoothing (raw data)
    smooth_params : Dict or None, default=None
        Parameters for smoothing. Defaults:
        - gaussian: {'sigma': 1.5}
        - None: no smoothing applied

    Returns
    -------
    trajectories : List[Dict] or None
        List of trajectory dictionaries with keys:
        - 'embryo_id': ID of this trajectory
        - 'times': np.ndarray of time values
        - 'metrics': np.ndarray of metric values
        Returns None if no data matches filters.
    embryo_ids : np.ndarray or None
        Array of unique embryo IDs in filtered data.
        Returns None if no data matches filters.
    n_embryos : int
        Number of embryos/trajectories extracted.

    Examples
    --------
    >>> trajectories, ids, n = get_trajectories_for_group(
    ...     df,
    ...     {'genotype': 'wildtype', 'treatment': 'control'},
    ...     time_col='hpf',
    ...     metric_col='expression_level',
    ...     embryo_id_col='embryo_id'
    ... )
    >>> print(f"Found {n} trajectories")
    >>> for traj in trajectories:
    ...     print(traj['embryo_id'], len(traj['times']), 'timepoints')
    """
    # Apply filters
    mask = pd.Series([True] * len(df), index=df.index)
    for col, val in filter_dict.items():
        mask &= (df[col] == val)

    filtered = df[mask].copy()

    if len(filtered) == 0:
        return None, None, 0

    # Set default smoothing parameters
    if smooth_params is None:
        if smooth_method == 'gaussian':
            smooth_params = {'sigma': 1.5}
        else:
            smooth_params = {}

    # Group by embryo and get trajectories
    embryo_ids = filtered[embryo_id_col].unique()
    trajectories = []

    for embryo_id in embryo_ids:
        embryo_data = filtered[filtered[embryo_id_col] == embryo_id].sort_values(time_col)
        embryo_data = embryo_data[[time_col, metric_col]].dropna()
        if len(embryo_data) > 1:
            times = embryo_data[time_col].values
            metrics = embryo_data[metric_col].values

            # Apply Gaussian smoothing if requested
            if smooth_method == 'gaussian':
                sigma = smooth_params.get('sigma', 1.5)
                metrics = gaussian_filter1d(metrics, sigma=sigma)
            # else: use raw data (no smoothing)

            trajectories.append({
                'embryo_id': embryo_id,
                'times': times,
                'metrics': metrics,
            })

    return trajectories, embryo_ids, len(trajectories)


def compute_binned_mean(
    times: np.ndarray,
    values: np.ndarray,
    bin_width: float = 0.5,
) -> Tuple[List[float], List[float]]:
    """Compute binned mean of values over time (deprecated wrapper).

    .. deprecated::
        Use :func:`compute_trend_line` from
        :mod:`src.analyze.trajectory_analysis.trajectory_utils` instead.
        This function is kept for backward compatibility.

    Parameters
    ----------
    times : np.ndarray
        Array of time values.
    values : np.ndarray
        Array of metric values.
    bin_width : float, default=0.5
        Width of time bins.

    Returns
    -------
    bin_times : List[float]
        Center time of each bin.
    bin_means : List[float]
        Mean value in each bin.

    Notes
    -----
    Prefer compute_trend_line() which supports median aggregation and
    Gaussian smoothing for more robust trend lines.
    """
    import warnings
    warnings.warn(
        "compute_binned_mean() is deprecated. Use compute_trend_line() from "
        "trajectory_utils instead, which supports median aggregation and "
        "Gaussian smoothing for cleaner trend lines.",
        DeprecationWarning,
        stacklevel=2
    )
    from ..trajectory_analysis.trajectory_utils import compute_trend_line
    return compute_trend_line(times, values, bin_width, statistic='mean', smooth_sigma=None)


def get_global_axis_ranges(
    all_trajectories: List[List[Dict]],
    padding_fraction: float = 0.1,
) -> Tuple[float, float, float, float]:
    """Compute global axis ranges from multiple trajectory lists.

    Useful for ensuring all subplots have the same axis scales for easy comparison.
    Examines all trajectories across all groups and computes min/max values with
    symmetric padding.

    Parameters
    ----------
    all_trajectories : List[List[Dict]]
        List of trajectory lists. Each inner list comes from
        :func:`get_trajectories_for_group`.
        Each trajectory dict has 'times' and 'metrics' keys.
    padding_fraction : float, default=0.1
        Fraction of range to add as symmetric padding (e.g., 0.1 = 10% padding).

    Returns
    -------
    time_min : float
        Minimum time value (with padding).
    time_max : float
        Maximum time value (with padding).
    metric_min : float
        Minimum metric value (with padding).
    metric_max : float
        Maximum metric value (with padding).

    Examples
    --------
    >>> groups = []
    >>> for genotype in ['wildtype', 'mutant']:
    ...     traj, _, _ = get_trajectories_for_group(df, {'genotype': genotype})
    ...     groups.append(traj)
    >>> t_min, t_max, m_min, m_max = get_global_axis_ranges(groups)
    >>> print(f"Time range: {t_min:.1f} - {t_max:.1f}")
    >>> print(f"Metric range: {m_min:.2f} - {m_max:.2f}")
    """
    time_min, time_max = float('inf'), float('-inf')
    metric_min, metric_max = float('inf'), float('-inf')

    for trajectories in all_trajectories:
        if trajectories is None:
            continue
        for traj in trajectories:
            time_min = min(time_min, traj['times'].min())
            time_max = max(time_max, traj['times'].max())
            metric_min = min(metric_min, traj['metrics'].min())
            metric_max = max(metric_max, traj['metrics'].max())

    # Add padding
    if metric_max > metric_min:
        padding = (metric_max - metric_min) * padding_fraction
        metric_min -= padding
        metric_max += padding

    return time_min, time_max, metric_min, metric_max


__all__ = [
    'get_trajectories_for_group',
    'get_global_axis_ranges',
    'compute_binned_mean',  # deprecated
]
