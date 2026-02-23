"""
Time Series Interpolation Utilities

Data extraction, interpolation, and preprocessing for temporal trajectories.

These functions provide a DataFrame-centric API where time column travels
with data through the entire pipeline, eliminating time-axis alignment bugs.

Functions
---------
- interpolate_to_common_grid : Interpolate trajectories to a common time grid
- pad_trajectories : Pad trajectories to uniform length with NaN
"""

import warnings
import numpy as np
import pandas as pd
from scipy import interpolate as scipy_interp
from typing import Tuple, List, Dict, Optional


def interpolate_to_common_grid(
    df_long: pd.DataFrame,
    grid_step: float = 0.5,
    time_col: str = 'hpf',
    value_col: str = 'metric_value',
    id_col: str = 'embryo_id',
    verbose: bool = False
) -> Tuple[List[np.ndarray], List[str], Dict[str, int], np.ndarray]:
    """
    Interpolate all trajectories to a common timepoint grid.

    Takes variable-length trajectories and interpolates them to a regular
    time grid. Trajectories are trimmed to their observed range (no extrapolation).

    Parameters
    ----------
    df_long : pd.DataFrame
        Long-format dataframe with columns for time, value, and ID
    grid_step : float, default=0.5
        Step size for common time grid (in time units)
    time_col : str, default='hpf'
        Name of time column
    value_col : str, default='metric_value'
        Name of value column
    id_col : str, default='embryo_id'
        Name of ID column
    verbose : bool, default=False
        If True, print progress information

    Returns
    -------
    interpolated_trajectories : list of np.ndarray
        List of interpolated trajectories (trimmed to observed ranges)
    ids_ordered : list of str
        IDs in same order as trajectories
    original_lengths : dict
        Mapping of ID to original trajectory length
    common_grid : np.ndarray
        The full common time grid used for interpolation

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'embryo_id': ['A', 'A', 'A', 'B', 'B'],
    ...     'hpf': [1.0, 2.0, 3.0, 1.5, 2.5],
    ...     'metric_value': [0.1, 0.2, 0.3, 0.15, 0.25]
    ... })
    >>> trajs, ids, lens, grid = interpolate_to_common_grid(df, grid_step=0.5)
    >>> print(f"Grid: {grid}")
    >>> print(f"Trajectories: {len(trajs)}")
    """
    if verbose:
        print(f"\n{'='*80}")
        print("TRAJECTORY INTERPOLATION TO COMMON TIMEPOINTS")
        print(f"{'='*80}")

    # Find min/max time across all trajectories
    min_time = df_long.groupby(id_col)[time_col].min().min()
    max_time = df_long.groupby(id_col)[time_col].max().max()

    if verbose:
        print(f"\n  Time range: {min_time:.1f} to {max_time:.1f}")

    # Create common timepoint grid
    common_grid = np.arange(min_time, max_time + grid_step, grid_step)
    if verbose:
        print(f"  Common grid: {len(common_grid)} timepoints (step={grid_step})")

    # Interpolate each trajectory
    interpolated_trajectories = []
    original_lengths = {}
    ids_ordered = []

    for entity_id, group in df_long.groupby(id_col):
        group_sorted = group.sort_values(time_col)
        time_vals = group_sorted[time_col].values
        value_vals = group_sorted[value_col].values

        original_lengths[entity_id] = len(value_vals)

        # Linear interpolation to common grid
        f = scipy_interp.interp1d(
            time_vals,
            value_vals,
            kind='linear',
            bounds_error=False,
            fill_value=np.nan
        )

        # Interpolate - will have NaN where outside observed range
        interpolated = f(common_grid)

        # Trim to observed range (remove NaN padding)
        # This keeps trajectories variable-length but NaN-free
        valid_mask = ~np.isnan(interpolated)
        interpolated_trimmed = interpolated[valid_mask]

        if len(interpolated_trimmed) > 0:  # Only keep if we have data
            interpolated_trajectories.append(interpolated_trimmed)
            ids_ordered.append(entity_id)

    if verbose:
        print(f"\n  Interpolated: {len(interpolated_trajectories)} trajectories")
        if interpolated_trajectories:
            lengths = [len(t) for t in interpolated_trajectories]
            print(f"  Trajectory lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
            print(f"  Variable lengths (trimmed to observed ranges, no NaN padding)")

    return interpolated_trajectories, ids_ordered, original_lengths, common_grid


def pad_trajectories(
    trajectories: List[np.ndarray],
    common_grid: np.ndarray,
    df_long: pd.DataFrame,
    ids: List[str],
    time_col: str = 'hpf',
    id_col: str = 'embryo_id',
    verbose: bool = False
) -> List[np.ndarray]:
    """
    Pad variable-length trajectories to uniform length with NaN.

    Takes trimmed trajectories and pads them with NaN to align with a common
    time grid, enabling matrix operations (e.g., mean/std computation).

    Parameters
    ----------
    trajectories : list of np.ndarray
        Variable-length trajectories from interpolate_to_common_grid
    common_grid : np.ndarray
        The common time grid
    df_long : pd.DataFrame
        Original long-format data (used to determine each trajectory's time range)
    ids : list of str
        IDs in same order as trajectories
    time_col : str, default='hpf'
        Name of time column in df_long
    id_col : str, default='embryo_id'
        Name of ID column in df_long
    verbose : bool, default=False
        If True, print progress information

    Returns
    -------
    padded_trajectories : list of np.ndarray
        All trajectories padded to len(common_grid) with NaN for missing values

    Notes
    -----
    This function is primarily used internally for computing group statistics
    (mean, std) across trajectories with different time ranges.

    Examples
    --------
    >>> trajs, ids, lens, grid = interpolate_to_common_grid(df, grid_step=0.5)
    >>> padded = pad_trajectories(trajs, grid, df, ids)
    >>> # Now all trajectories have same length
    >>> np.array(padded).shape  # (n_trajectories, len(grid))
    """
    if verbose:
        print(f"\n  Padding {len(trajectories)} trajectories to common grid...")

    padded_trajectories = []

    for entity_id, traj in zip(ids, trajectories):
        # Find this entity's observed time range from original data
        entity_data = df_long[df_long[id_col] == entity_id].sort_values(time_col)
        if len(entity_data) == 0:
            # No data for this entity, return all NaN
            padded_trajectories.append(np.full(len(common_grid), np.nan))
            continue

        min_time = entity_data[time_col].min()
        max_time = entity_data[time_col].max()

        # Find start and end indices in common grid
        start_idx = np.searchsorted(common_grid, min_time, side='left')
        end_idx = np.searchsorted(common_grid, max_time, side='right')

        # Handle edge case where trajectory extends beyond grid
        traj_len = len(traj)
        grid_span = end_idx - start_idx

        if grid_span != traj_len:
            # Adjust end_idx if mismatch (can happen due to interpolation grid steps)
            end_idx = start_idx + traj_len

        # Create padded array
        padded_traj = np.full(len(common_grid), np.nan)

        # Insert trajectory data at correct position
        if end_idx <= len(common_grid):
            padded_traj[start_idx:end_idx] = traj
        else:
            # Trajectory extends beyond grid, truncate
            available = len(common_grid) - start_idx
            padded_traj[start_idx:] = traj[:available]

        padded_trajectories.append(padded_traj)

    if verbose:
        # Verify all same length
        lengths = [len(t) for t in padded_trajectories]
        if len(set(lengths)) == 1:
            print(f"  All {len(padded_trajectories)} trajectories padded to uniform length: {lengths[0]}")
        else:
            print(f"  WARNING: Inconsistent lengths after padding: {set(lengths)}")

    return padded_trajectories


# ==============================================================================
# Backward Compatibility (Deprecated Aliases)
# ==============================================================================

def pad_trajectories_for_plotting(*args, **kwargs):
    """Deprecated: Use pad_trajectories instead."""
    warnings.warn(
        "pad_trajectories_for_plotting is deprecated. Use pad_trajectories instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return pad_trajectories(*args, **kwargs)
