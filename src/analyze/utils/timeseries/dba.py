"""
DTW Barycenter Averaging (DBA)

Computes consensus trajectory from multiple time series using DTW alignment.

DBA iteratively refines a barycenter trajectory by aligning all input sequences
to it via DTW and averaging the aligned positions.

Functions
---------
- dba: Compute consensus trajectory via DTW Barycenter Averaging

References
----------
Petitjean, F., Ketterlin, A., & Gancarski, P. (2011). A global averaging method
for dynamic time warping, with applications to clustering. Pattern Recognition.
"""

import numpy as np
from typing import Callable, Optional, List
from scipy.ndimage import gaussian_filter1d


def dba(
    series_list: List[np.ndarray],
    dtw_func: Callable,
    weights: Optional[np.ndarray] = None,
    max_iter: int = 10,
    smooth_sigma: float = 0.0,
    verbose: bool = False
) -> np.ndarray:
    """
    Compute DTW Barycenter Averaging (numerically stable).

    Iteratively refines a consensus trajectory by aligning all input sequences
    to it via DTW and averaging the aligned positions.

    Parameters
    ----------
    series_list : list of np.ndarray
        Input time series to average. Each 1-D array represents a trajectory
        (lengths may differ).
    dtw_func : callable
        Function returning (path, dist) given two series.
        Signature: dtw_func(series1, series2) -> (path, distance)
    weights : list or np.ndarray, optional
        Per-series weights. Defaults to uniform.
        Will be normalized to sum to 1.
    max_iter : int, default=10
        Number of refinement iterations.
    smooth_sigma : float, default=0.0
        Gaussian smoothing sigma (0 disables smoothing).
    verbose : bool, default=False
        Print per-iteration statistics.

    Returns
    -------
    np.ndarray
        Barycenter (consensus) trajectory.

    Examples
    --------
    >>> from src.analyze.utils.timeseries.dtw import compute_dtw_distance
    >>>
    >>> # Define DTW function for DBA
    >>> def dtw_func(seq1, seq2):
    ...     dist = compute_dtw_distance(seq1, seq2)
    ...     # DBA needs (path, dist) format
    ...     # Approximate path as diagonal alignment
    ...     min_len = min(len(seq1), len(seq2))
    ...     path = [(i, i) for i in range(min_len)]
    ...     return path, dist
    >>>
    >>> # Series to average
    >>> series = [
    ...     np.array([1.0, 2.0, 3.0, 4.0]),
    ...     np.array([1.5, 2.5, 3.5, 4.5]),
    ...     np.array([0.9, 2.1, 3.1, 3.9])
    ... ]
    >>>
    >>> # Compute consensus trajectory
    >>> consensus = dba(series, dtw_func, max_iter=10, verbose=True)
    """
    n = len(series_list)
    if n == 0:
        raise ValueError("series_list is empty")

    series_list = [np.asarray(s, dtype=np.float64) for s in series_list]

    # Initialize weights
    if weights is None:
        weights = np.ones(n, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        weights /= np.maximum(weights.sum(), 1e-8)

    # Initialize barycenter as copy of first series
    bary = series_list[0].copy()

    # Iterative refinement
    for it in range(max_iter):
        accum = np.zeros_like(bary)
        counts = np.zeros_like(bary)
        total_cost = 0.0

        # Align each series to current barycenter
        for s, w in zip(series_list, weights):
            path, dist = dtw_func(s, bary)
            total_cost += dist * w

            # Accumulate weighted values along alignment path
            for i, j in path:
                if j < len(accum) and i < len(s):
                    accum[j] += w * s[i]
                    counts[j] += w

        # Update barycenter: weighted average of aligned positions
        counts = np.maximum(counts, 1e-8)  # Avoid division by zero
        bary = np.nan_to_num(accum / counts, nan=0.0, posinf=0.0, neginf=0.0)

        # Optional smoothing
        if smooth_sigma > 0:
            bary = gaussian_filter1d(bary, sigma=smooth_sigma)

        if verbose:
            print(f"Iter {it+1}/{max_iter} | mean cost: {total_cost/n:.6f}")

    return bary
