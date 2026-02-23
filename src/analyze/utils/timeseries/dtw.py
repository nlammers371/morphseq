"""
DTW Distance Computation

Dynamic Time Warping (DTW) implementation with Sakoe-Chiba band constraint.

This module provides functions for computing pairwise DTW distances between
variable-length temporal sequences, suitable for clustering and comparison of
time-series data.

Functions
=========
Univariate DTW:
- compute_dtw_distance : Compute DTW distance between two 1D sequences
- compute_dtw_distance_matrix : Compute pairwise DTW distances for multiple 1D sequences

Multivariate DTW (MD-DTW):
- _dtw_multivariate_pair : Helper for pairwise multivariate DTW (internal)
- compute_md_dtw_distance_matrix : Compute pairwise MD-DTW distances
"""

import numpy as np
from typing import List, Optional


def compute_dtw_distance(
    seq1: np.ndarray,
    seq2: np.ndarray,
    window: int = 3,
    normalize: bool = False
) -> float:
    """
    Compute Dynamic Time Warping distance between two sequences.

    Uses the Sakoe-Chiba band constraint to limit the warping path and
    improve computational efficiency. Band width is automatically expanded
    to handle sequences of different lengths.

    Parameters
    ----------
    seq1 : array-like
        First input sequence (1D array)
    seq2 : array-like
        Second input sequence (1D array)
    window : int, default=3
        Sakoe-Chiba band width constraint. Window is automatically expanded
        to max(window, abs(len(seq1) - len(seq2))) to handle length differences.
    normalize : bool, default=False
        If True, return per-step normalized distance (length-independent metric).
        Normalization divides by path length (n + m).

    Returns
    -------
    float
        DTW distance between sequences. If normalize=True, returns per-step distance.
        Returns inf if computation fails (e.g., all NaN values).

    Notes
    -----
    The DTW distance represents the minimum cumulative cost of aligning two sequences,
    where cost is defined as the absolute difference between values at each pair of
    timepoints. The Sakoe-Chiba band constraint restricts the warping path to stay
    within a diagonal band, improving speed while limiting pathological alignments.

    References
    ----------
    - Sakoe, H., & Chiba, S. (1978). Dynamic programming algorithm optimization
      for spoken word recognition. IEEE Transactions on Acoustics, Speech, and
      Signal Processing, 26(1), 43-49.

    Examples
    --------
    >>> seq1 = np.array([1.0, 2.0, 3.0, 4.0])
    >>> seq2 = np.array([1.5, 2.5, 3.5])
    >>> dist = compute_dtw_distance(seq1, seq2)
    >>> print(f"DTW distance: {dist:.3f}")

    >>> # Normalized distance (length-independent)
    >>> dist_norm = compute_dtw_distance(seq1, seq2, normalize=True)
    >>> print(f"Normalized DTW: {dist_norm:.3f}")
    """
    # Convert to numeric arrays (prevents LAPACK errors with object dtypes)
    seq1 = np.asarray(seq1, dtype=float)
    seq2 = np.asarray(seq2, dtype=float)
    n, m = len(seq1), len(seq2)

    # Expand window to handle sequence length differences
    w = max(window, abs(n - m))

    # Initialize cost matrix with infinity
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    # Fill the cost matrix with band constraint
    for i in range(1, n + 1):
        j_start = max(1, i - w)
        j_end = min(m + 1, i + w + 1)
        for j in range(j_start, j_end):
            cost = abs(seq1[i - 1] - seq2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # insertion
                dtw_matrix[i, j - 1],      # deletion
                dtw_matrix[i - 1, j - 1]   # match
            )

    distance = dtw_matrix[n, m]

    # Normalize by path length if requested (length-independent metric)
    if normalize and not np.isinf(distance):
        distance = distance / (n + m)

    return float(distance)


def compute_dtw_distance_matrix(
    trajectories: list,
    window: int = 3,
    verbose: bool = False
) -> np.ndarray:
    """
    Compute pairwise DTW distances for multiple trajectories.

    Computes a symmetric distance matrix where each element (i, j) represents
    the DTW distance between trajectories i and j. Diagonal elements are zero.

    Parameters
    ----------
    trajectories : list of array-like
        List of trajectories (variable-length 1D arrays)
    window : int, default=3
        Sakoe-Chiba band width for DTW computation
    verbose : bool, default=False
        If True, print progress updates every 10 trajectories

    Returns
    -------
    distance_matrix : np.ndarray
        Symmetric (n_trajectories x n_trajectories) distance matrix.
        Distance matrix properties:
        - Symmetric: distance_matrix[i,j] == distance_matrix[j,i]
        - Zero diagonal: np.allclose(np.diag(distance_matrix), 0)
        - Non-negative: all values >= 0 (unless inf from computation errors)

    Raises
    ------
    Warning (printed, not raised)
        If DTW computation fails for a pair, that pair receives inf distance
        and a warning is printed (if verbose=True).

    Examples
    --------
    >>> trajectories = [
    ...     np.array([1.0, 2.0, 3.0, 4.0]),
    ...     np.array([1.5, 2.5, 3.5]),
    ...     np.array([0.9, 2.1, 3.1, 3.9, 4.1])
    ... ]
    >>> dist_matrix = compute_dtw_distance_matrix(trajectories, window=3)
    >>> print(dist_matrix.shape)
    (3, 3)
    >>> print(np.diag(dist_matrix))  # Should be [0, 0, 0]
    [0. 0. 0.]
    >>> print(dist_matrix[0, 1], dist_matrix[1, 0])  # Should be symmetric
    """
    n_trajectories = len(trajectories)
    distance_matrix = np.zeros((n_trajectories, n_trajectories))

    if verbose:
        print(f"\n  Computing pairwise DTW distances (window={window})...")

    for i in range(n_trajectories):
        if verbose and (i + 1) % 10 == 0:
            print(f"    Progress: {i + 1}/{n_trajectories}")

        for j in range(i + 1, n_trajectories):
            try:
                dist = compute_dtw_distance(
                    trajectories[i],
                    trajectories[j],
                    window=window
                )
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
            except Exception as e:
                if verbose:
                    print(f"    Warning: DTW computation failed for pair ({i}, {j}): {e}")
                distance_matrix[i, j] = np.inf
                distance_matrix[j, i] = np.inf

    if verbose:
        # Validation stats
        nan_count = np.isnan(distance_matrix).sum()
        inf_count = np.isinf(distance_matrix).sum()
        diag_check = np.allclose(np.diag(distance_matrix), 0)

        print(f"\n  Validation:")
        print(f"    NaN count: {nan_count}")
        print(f"    Inf count: {inf_count}")
        print(f"    Diagonal ~ 0: {diag_check}")
        print(f"    Distance stats: min={np.nanmin(distance_matrix):.3f}, "
              f"max={np.nanmax(distance_matrix):.3f}, "
              f"mean={np.nanmean(distance_matrix):.3f}")
        print(f"    Matrix shape: {distance_matrix.shape}")

    return distance_matrix


# ============================================================================
# Multivariate DTW (MD-DTW) Functions
# ============================================================================


def _nan_aware_cost_matrix(ts_a: np.ndarray, ts_b: np.ndarray) -> np.ndarray:
    """Compute NaN-aware Euclidean cost matrix between two multivariate series.

    For each (i, j), distance is computed using only feature dimensions where
    both ts_a[i] and ts_b[j] are finite. If there is no overlap, cost is inf.
    Distances are scaled by total_features / valid_features to keep scale
    comparable across pairs with different valid counts.
    """
    A = np.asarray(ts_a, dtype=float)
    B = np.asarray(ts_b, dtype=float)

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("ts_a and ts_b must be 2D arrays (timepoints x features)")
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"Feature mismatch: {A.shape[1]} vs {B.shape[1]}")

    n, n_features = A.shape
    m = B.shape[0]

    sse = np.zeros((n, m), dtype=float)
    valid_counts = np.zeros((n, m), dtype=np.int32)

    for k in range(n_features):
        a = A[:, k][:, None]
        b = B[:, k][None, :]
        valid = np.isfinite(a) & np.isfinite(b)
        if not valid.any():
            continue
        diff = a - b
        sse += np.where(valid, diff * diff, 0.0)
        valid_counts += valid

    cost = np.full((n, m), np.inf, dtype=float)
    has_overlap = valid_counts > 0
    if np.any(has_overlap):
        scaling = n_features / valid_counts[has_overlap]
        cost[has_overlap] = np.sqrt(sse[has_overlap] * scaling)

    return cost


def _trim_nan_edges(ts: np.ndarray) -> np.ndarray:
    """Trim leading/trailing timepoints with all-NaN features."""
    if ts.ndim != 2:
        raise ValueError("ts must be 2D (timepoints x features)")
    valid = np.isfinite(ts).any(axis=1)
    if not valid.any():
        return ts[:0]
    start = int(np.argmax(valid))
    end = len(valid) - int(np.argmax(valid[::-1]))
    return ts[start:end]


def _dtw_multivariate_pair(
    ts_a: np.ndarray,
    ts_b: np.ndarray,
    window: Optional[int] = 3,
) -> float:
    """
    Compute DTW distance between two multivariate time series.

    Uses NaN-aware Euclidean distance in feature space as the local distance metric.
    Implements dynamic programming with optional Sakoe-Chiba band constraint.

    Parameters
    ----------
    ts_a : np.ndarray
        2D array with shape (T_a, n_features) - time series A
    ts_b : np.ndarray
        2D array with shape (T_b, n_features) - time series B
    window : int, optional
        Sakoe-Chiba band width (None for unconstrained DTW)

    Returns
    -------
    float
        DTW distance between ts_a and ts_b

    Notes
    -----
    - The "multivariate" part is handled by computing Euclidean distance
      between feature vectors at each timepoint pair
    - NaNs are ignored per-feature; if no feature overlaps at a pair, cost=inf
    - ts_a[i] and ts_b[j] are vectors in feature space
    - Distance is computed as sqrt(sum((ts_a[i] - ts_b[j])^2))
    """
    ts_a = _trim_nan_edges(ts_a)
    ts_b = _trim_nan_edges(ts_b)
    if ts_a.size == 0 or ts_b.size == 0:
        return np.inf

    # Step 1: Compute local cost matrix (NaN-aware)
    dist_matrix = _nan_aware_cost_matrix(ts_a, ts_b)

    n, m = dist_matrix.shape

    # Step 2: Initialize accumulated cost matrix
    # Set all to infinity initially
    acc_cost = np.full((n + 1, m + 1), np.inf)
    acc_cost[0, 0] = 0

    # Step 3: Dynamic Programming with Sakoe-Chiba Constraint
    # The window parameter limits how far we can warp in time
    if window is None:
        # Unconstrained DTW - compute all pairs
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = dist_matrix[i - 1, j - 1]
                acc_cost[i, j] = cost + min(
                    acc_cost[i - 1, j],      # Insertion (skip in ts_a)
                    acc_cost[i, j - 1],      # Deletion (skip in ts_b)
                    acc_cost[i - 1, j - 1]   # Match
                )
    else:
        # Constrained DTW - only compute within band
        # The band width is determined by the window parameter
        w = max(window, abs(n - m))

        for i in range(1, n + 1):
            # Determine valid range for j
            start_j = max(1, i - w)
            end_j = min(m + 1, i + w + 1)

            for j in range(start_j, end_j):
                cost = dist_matrix[i - 1, j - 1]
                acc_cost[i, j] = cost + min(
                    acc_cost[i - 1, j],
                    acc_cost[i, j - 1],
                    acc_cost[i - 1, j - 1]
                )

    return acc_cost[n, m]


def compute_md_dtw_distance_matrix(
    X: np.ndarray,
    sakoe_chiba_radius: Optional[int] = 3,
    n_jobs: int = -1,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute multivariate DTW distance matrix.

    Computes pairwise multivariate DTW distances between all samples using
    pure Python/NumPy implementation with parallel processing.

    Parameters
    ----------
    X : np.ndarray
        3D array with shape (n_samples, n_timepoints, n_features)
    sakoe_chiba_radius : int, optional
        Sakoe-Chiba band constraint width.
        Limits warping to within +/- radius timepoints.
        None for unconstrained DTW (slower).
        Default: 3 (good balance of flexibility and speed)
    n_jobs : int, default=-1
        Number of parallel jobs. -1 means use all available CPUs.
        1 means single-threaded (no parallelization).
    verbose : bool, default=True
        Print progress and diagnostics

    Returns
    -------
    distance_matrix : np.ndarray
        Array with shape (n_samples, n_samples).
        Symmetric matrix where D[i,j] = DTW distance between sample i and j

    Examples
    --------
    >>> # Create random multivariate time series
    >>> X = np.random.randn(10, 50, 3)  # 10 samples, 50 timepoints, 3 features
    >>> D = compute_md_dtw_distance_matrix(X, sakoe_chiba_radius=3)
    >>> print(D.shape)  # (10, 10)

    Notes
    -----
    - This is a pure Python implementation using NumPy/SciPy
    - Parallelized with joblib for multi-core speedup
    - Time complexity: O(N^2 * T^2 / n_jobs) where N=samples, T=timepoints
    - Output is symmetric by construction: D[i,j] == D[j,i]
    - Diagonal is zero: D[i,i] == 0
    - NaNs in input are handled by ignoring missing features per timepoint pair
    """
    from joblib import Parallel, delayed, cpu_count

    n_samples = X.shape[0]

    # Determine actual number of jobs
    if n_jobs == -1:
        actual_jobs = cpu_count()
    else:
        actual_jobs = min(n_jobs, cpu_count())

    if verbose:
        print(f"Computing MD-DTW distance matrix...")
        print(f"  Samples: {n_samples}")
        print(f"  Array shape: {X.shape}")
        print(f"  Sakoe-Chiba radius: {sakoe_chiba_radius}")
        print(f"  Parallel jobs: {actual_jobs} (of {cpu_count()} CPUs available)")

    # Generate all unique pairs (i, j) where i < j
    pairs = [(i, j) for i in range(n_samples) for j in range(i + 1, n_samples)]
    total_pairs = len(pairs)

    if verbose:
        print(f"  Computing {total_pairs} pairwise distances...")

    # Parallel computation of all pairwise distances
    if actual_jobs == 1:
        # Single-threaded fallback
        results = []
        for idx, (i, j) in enumerate(pairs):
            dist = _dtw_multivariate_pair(X[i], X[j], window=sakoe_chiba_radius)
            results.append(dist)
            if verbose and (idx + 1) % max(1, total_pairs // 10) == 0:
                print(f"  Progress: {idx + 1}/{total_pairs} ({100*(idx+1)//total_pairs}%)", end='\r')
    else:
        # Parallel computation
        results = Parallel(n_jobs=actual_jobs, verbose=0)(
            delayed(_dtw_multivariate_pair)(X[i], X[j], window=sakoe_chiba_radius)
            for i, j in pairs
        )

    # Build symmetric distance matrix from results
    D = np.zeros((n_samples, n_samples))
    for (i, j), dist in zip(pairs, results):
        D[i, j] = dist
        D[j, i] = dist

    if verbose:
        # Verify properties
        diagonal_max = np.max(np.abs(np.diag(D)))
        asymmetry = np.max(np.abs(D - D.T))

        print(f"\n  Distance matrix computed")
        print(f"  Shape: {D.shape}")
        print(f"  Distance range: [{D[D > 0].min():.4f}, {D.max():.4f}]")
        print(f"  Max diagonal value: {diagonal_max:.2e} (should be ~0)")
        print(f"  Max asymmetry: {asymmetry:.2e} (should be ~0)")

        if diagonal_max > 1e-10:
            print(f"  WARNING: Diagonal not zero (max={diagonal_max:.2e})")
        if asymmetry > 1e-10:
            print(f"  WARNING: Matrix not symmetric (max asymmetry={asymmetry:.2e})")

    return D
