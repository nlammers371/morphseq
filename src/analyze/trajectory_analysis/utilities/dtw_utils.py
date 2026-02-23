"""
DTW Utility Functions for Trajectory Analysis

Domain-specific helper functions for preparing trajectory data and computing
DTW distance matrices. These functions bridge the gap between trajectory
DataFrames and the generic DTW algorithms in utils.timeseries.

Functions
=========
- prepare_multivariate_array : Convert DataFrame to 3D array for MD-DTW
- compute_trajectory_distances : High-level function to compute trajectory distances
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

from analyze.utils.timeseries.dtw import compute_md_dtw_distance_matrix


def prepare_multivariate_array(
    df: pd.DataFrame,
    metrics: List[str],
    time_col: str = 'predicted_stage_hpf',
    embryo_id_col: str = 'embryo_id',
    time_grid: Optional[np.ndarray] = None,
    normalize: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Convert long-format DataFrame to 3D array for multivariate DTW.

    Takes a DataFrame with multiple metric columns and converts to a 3D numpy array
    suitable for multivariate DTW computation. Handles interpolation to common time
    grid and optional Z-score normalization.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame with columns [embryo_id, time_col, metric1, metric2, ...]
    metrics : List[str]
        List of metric column names (e.g., ['baseline_deviation_normalized', 'total_length_um'])
    time_col : str, default='predicted_stage_hpf'
        Name of time column
    embryo_id_col : str, default='embryo_id'
        Name of embryo ID column
    time_grid : np.ndarray, optional
        Optional pre-defined time grid. If None, auto-computed from data
    normalize : bool, default=True
        Whether to Z-score normalize each metric globally
    verbose : bool, default=True
        Print progress information

    Returns
    -------
    X : np.ndarray
        Array with shape (n_embryos, n_timepoints, n_metrics)
    embryo_ids : List[str]
        List of embryo identifiers (same order as X rows)
    time_grid : np.ndarray
        Time values (same for all embryos)

    Examples
    --------
    >>> df = load_experiment_dataframe('20251121')
    >>> X, embryo_ids, time_grid = prepare_multivariate_array(
    ...     df,
    ...     metrics=['baseline_deviation_normalized', 'total_length_um']
    ... )
    >>> print(X.shape)  # (n_embryos, n_timepoints, 2)

    Notes
    -----
    - All embryos are interpolated to the same common time grid
    - Missing values (NaN) are handled by linear interpolation for interior gaps
    - Edge NaNs are preserved for DTW to handle with NaN-aware costs
    - Z-score normalization ignores NaNs (per-metric)
    """
    from ..utilities.trajectory_utils import interpolate_to_common_grid_multi_df
    from ..config import GRID_STEP

    if verbose:
        print(f"Preparing multivariate array for {len(metrics)} metrics...")
        print(f"  Metrics: {metrics}")
        print(f"  Normalization: {normalize}")

    # Step 1: Get embryo IDs in sorted order (for consistency)
    embryo_ids = sorted(df[embryo_id_col].unique())
    n_embryos = len(embryo_ids)
    n_metrics = len(metrics)

    if verbose:
        print(f"  Embryos: {n_embryos}")

    # Step 2: Interpolate each metric for all embryos using the trajectory utility
    # If time_grid is provided, pass it through; otherwise let the utility derive the grid
    provided_time_grid = None
    if time_grid is not None:
        provided_time_grid = np.asarray(time_grid, dtype=float)
        if provided_time_grid.ndim != 1:
            raise ValueError(f"time_grid must be 1D, got shape {provided_time_grid.shape}")
        if len(provided_time_grid) == 0:
            raise ValueError("time_grid must be non-empty")
        # Ensure strictly increasing, unique grid (dedupe protects against float replication)
        provided_time_grid = np.unique(provided_time_grid)
        if len(provided_time_grid) > 1 and not np.all(np.diff(provided_time_grid) > 0):
            provided_time_grid = np.sort(provided_time_grid)

    df_interp = interpolate_to_common_grid_multi_df(
        df,
        metrics,
        grid_step=(provided_time_grid[1] - provided_time_grid[0]) if provided_time_grid is not None and len(provided_time_grid) > 1 else GRID_STEP,
        time_col=time_col,
        time_grid=provided_time_grid,
        fill_edges=False,
        verbose=verbose,
    )

    # If a grid was provided, keep it exactly (critical for cross-dataset comparisons).
    # Otherwise derive grid from interpolation output.
    if provided_time_grid is not None:
        time_grid = provided_time_grid
    else:
        time_grid = np.sort(df_interp[time_col].unique())
    n_timepoints = len(time_grid)

    if verbose:
        print(f"  Time points: {n_timepoints} ({time_grid.min():.1f} - {time_grid.max():.1f} hpf)")

    # Step 3: Initialize 3D array
    X = np.zeros((n_embryos, n_timepoints, n_metrics))

    for i, embryo_id in enumerate(embryo_ids):
        emb_df = df_interp[df_interp[embryo_id_col] == embryo_id].set_index(time_col)

        if emb_df.empty:
            if verbose:
                print(f"  WARNING: Embryo {embryo_id} has no interpolated rows, using zeros")
            continue

        for j, metric in enumerate(metrics):
            # Reindex onto full grid and fill interior gaps only; keep edge NaNs.
            ser = emb_df[metric].reindex(time_grid)
            ser = ser.interpolate(limit_area='inside')
            X[i, :, j] = ser.values

    # Step 5: Handle remaining NaNs (e.g., at edges due to interpolation bounds)
    mask = np.isnan(X)
    if mask.any():
        for i in range(n_embryos):
            for j in range(n_metrics):
                series = X[i, :, j]
                nans = np.isnan(series)

                if nans.all():
                    # All NaNs - leave as NaN
                    continue
                # Fill interior NaNs only; preserve edge NaNs
                filled = pd.Series(series).interpolate(limit_area='inside')
                X[i, :, j] = filled.values

    if verbose:
        print(f"  Array shape: {X.shape}")
        print(f"  Before normalization:")
        for j, metric in enumerate(metrics):
            print(f"    {metric}: mean={np.nanmean(X[:, :, j]):.3f}, std={np.nanstd(X[:, :, j]):.3f}")

    # Step 6: Global Z-score normalization (if enabled)
    if normalize:
        means = np.nanmean(X, axis=(0, 1))
        stds = np.nanstd(X, axis=(0, 1))
        stds = np.where(stds == 0, 1.0, stds)
        for j in range(n_metrics):
            X[:, :, j] = (X[:, :, j] - means[j]) / stds[j]

        if verbose:
            print(f"  After normalization:")
            for j, metric in enumerate(metrics):
                print(f"    {metric}: mean={np.nanmean(X[:, :, j]):.6f}, std={np.nanstd(X[:, :, j]):.6f}")

    if verbose:
        print(f"  Multivariate array prepared successfully")

    return X, embryo_ids, time_grid


def compute_trajectory_distances(
    df: pd.DataFrame,
    metrics: List[str],
    time_col: str = 'predicted_stage_hpf',
    embryo_id_col: str = 'embryo_id',
    time_window: Optional[Tuple[float, float]] = None,
    normalize: bool = True,
    sakoe_chiba_radius: int = 3,
    n_jobs: int = -1,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Compute MD-DTW distance matrix from trajectory DataFrame.

    This is the PRIMARY convenience function for converting trajectory data into
    a distance matrix for clustering analysis. Handles time filtering, array
    preparation, and distance computation in one step.

    This function combines three steps:
    1. Optional time window filtering
    2. Multivariate array preparation (prepare_multivariate_array)
    3. MD-DTW distance computation (compute_md_dtw_distance_matrix)

    Parameters
    ----------
    df : pd.DataFrame
        Long-format trajectory data with columns for time, embryo_id, and metrics.
    metrics : List[str]
        Names of columns to use as features (e.g., ['curvature', 'length']).
    time_col : str, default='predicted_stage_hpf'
        Column name for time values.
    embryo_id_col : str, default='embryo_id'
        Column identifying unique embryos/trajectories.
    time_window : Optional[Tuple[float, float]], default=None
        If provided, filters to (min_time, max_time) before computing distances.
        Example: (30, 60) to analyze only 30-60 hpf.
    normalize : bool, default=True
        If True, z-score normalize each metric across all embryos.
    sakoe_chiba_radius : int, default=3
        Warping window constraint for DTW (3 is a good default).
    n_jobs : int, default=-1
        Number of parallel jobs for distance computation.
        -1 means use all available CPUs (auto-detect).
        1 means single-threaded (no parallelization).
    verbose : bool, default=True
        Print progress information.

    Returns
    -------
    D : np.ndarray
        Distance matrix (n_embryos x n_embryos), symmetric.
    embryo_ids : List[str]
        Ordered list of embryo IDs corresponding to distance matrix rows/cols.
    time_grid : np.ndarray
        Common time grid used for interpolation.

    Examples
    --------
    >>> # Compute distances on full time range
    >>> D, embryo_ids, time_grid = compute_trajectory_distances(
    ...     df,
    ...     metrics=['baseline_deviation_normalized', 'total_length_um'],
    ... )

    >>> # Compute distances on specific time window (e.g., 30-60 hpf)
    >>> D, embryo_ids, time_grid = compute_trajectory_distances(
    ...     df,
    ...     metrics=['baseline_deviation_normalized', 'total_length_um'],
    ...     time_window=(30, 60),
    ... )

    >>> # Then cluster
    >>> from src.analyze.trajectory_analysis import run_k_selection_with_plots
    >>> results = run_k_selection_with_plots(
    ...     df=df,
    ...     D=D,
    ...     embryo_ids=embryo_ids,
    ...     output_dir=Path('results/clustering'),
    ...     k_range=[2, 3, 4, 5],
    ... )

    Notes
    -----
    - This is a convenience wrapper that simplifies the common workflow
    - For advanced use cases, you can still use prepare_multivariate_array()
      and compute_md_dtw_distance_matrix() separately
    - Time filtering happens BEFORE interpolation, which may result in fewer
      embryos if some have no data in the specified window
    """
    if verbose:
        print("="*70)
        print("COMPUTE TRAJECTORY DISTANCES")
        print("="*70)

    # Step 1: Filter by time window if specified
    if time_window is not None:
        min_time, max_time = time_window
        if verbose:
            print(f"\n1. Filtering to time window: [{min_time}, {max_time}] {time_col}")

        df_filtered = df[
            (df[time_col] >= min_time) &
            (df[time_col] <= max_time)
        ].copy()

        if verbose:
            n_embryos_before = df[embryo_id_col].nunique()
            n_embryos_after = df_filtered[embryo_id_col].nunique()
            print(f"   Embryos before: {n_embryos_before}")
            print(f"   Embryos after: {n_embryos_after}")

            if n_embryos_after < n_embryos_before:
                print(f"   WARNING: Lost {n_embryos_before - n_embryos_after} embryos")
                print("            (no data in time window)")
    else:
        df_filtered = df.copy()
        if verbose:
            print("\n1. Using full time range")

    # Step 2: Prepare multivariate array
    if verbose:
        print(f"\n2. Preparing multivariate array")
        print(f"   Metrics: {metrics}")
        print(f"   Normalize: {normalize}")

    X, embryo_ids, time_grid = prepare_multivariate_array(
        df_filtered,
        metrics=metrics,
        time_col=time_col,
        embryo_id_col=embryo_id_col,
        normalize=normalize,
        verbose=verbose,
    )

    if verbose:
        print(f"\n   Array shape: {X.shape} (embryos x timepoints x metrics)")
        print(f"   Time grid: [{time_grid[0]:.1f}, {time_grid[-1]:.1f}] ({len(time_grid)} points)")

    # Step 3: Compute MD-DTW distance matrix
    if verbose:
        print(f"\n3. Computing MD-DTW distances")
        print(f"   Sakoe-Chiba radius: {sakoe_chiba_radius}")

    D = compute_md_dtw_distance_matrix(
        X,
        sakoe_chiba_radius=sakoe_chiba_radius,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    if verbose:
        print(f"\nDistance matrix: {D.shape}")
        print(f"  Range: [{D[D > 0].min():.3f}, {D.max():.3f}]")
        print("="*70)

    return D, embryo_ids, time_grid


__all__ = [
    'prepare_multivariate_array',
    'compute_trajectory_distances',
]
