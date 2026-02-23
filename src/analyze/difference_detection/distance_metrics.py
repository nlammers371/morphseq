"""
Test statistics for permutation testing.

This module provides pluggable test statistic functions that can be used with
the permutation testing framework. Each function takes samples and returns a
scalar statistic value.

These statistics can be used for:
- Distribution comparison (energy distance, MMD, centroid distance)
- Classification (AUROC, accuracy)
- Regression (coefficient magnitude)
- Future additions (curvature difference, trajectory divergence, etc.)
"""

import numpy as np
from typing import Optional
from scipy.spatial.distance import pdist, squareform


def compute_energy_distance(X1: np.ndarray, X2: np.ndarray) -> float:
    """
    Compute energy distance between two multivariate distributions.

    Energy distance is a statistical metric for measuring distance between
    probability distributions. It equals zero if and only if the two
    distributions are identical.

    The energy distance is defined as:
        E(P, Q) = E[|X - Y|] - 0.5*E[|X - X'|] - 0.5*E[|Y - Y'|]
    where:
    - X, X' are iid samples from P
    - Y, Y' are iid samples from Q
    - |.| denotes the Euclidean distance

    Parameters
    ----------
    X1 : np.ndarray, shape (n1, p)
        Samples from first distribution
    X2 : np.ndarray, shape (n2, p)
        Samples from second distribution

    Returns
    -------
    float
        Energy distance (non-negative)

    References
    ----------
    Szekely, G. J., & Rizzo, M. L. (2013). Energy statistics: a new approach.
    Journal of Statistical Planning and Inference, 143(8), 1249-1272.

    Examples
    --------
    >>> X1 = np.random.randn(100, 10)
    >>> X2 = np.random.randn(100, 10) + 0.5  # Shifted distribution
    >>> energy = compute_energy_distance(X1, X2)
    """
    X1 = np.asarray(X1)
    X2 = np.asarray(X2)

    if X1.ndim == 1:
        X1 = X1.reshape(-1, 1)
    if X2.ndim == 1:
        X2 = X2.reshape(-1, 1)

    n1 = X1.shape[0]
    n2 = X2.shape[0]

    # Compute pairwise distances within X1
    if n1 > 1:
        dist_X1 = squareform(pdist(X1))
        mean_X1_X1 = np.mean(dist_X1[np.triu_indices_from(dist_X1, k=1)])
    else:
        mean_X1_X1 = 0

    # Compute pairwise distances within X2
    if n2 > 1:
        dist_X2 = squareform(pdist(X2))
        mean_X2_X2 = np.mean(dist_X2[np.triu_indices_from(dist_X2, k=1)])
    else:
        mean_X2_X2 = 0

    # Compute pairwise distances between X1 and X2
    diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)
    mean_X1_X2 = np.mean(distances)

    # Compute energy distance
    energy = mean_X1_X2 - 0.5 * mean_X1_X1 - 0.5 * mean_X2_X2

    return max(0.0, energy)  # Ensure non-negative


def compute_rbf_kernel(
    X1: np.ndarray,
    X2: np.ndarray,
    bandwidth: Optional[float] = None
) -> np.ndarray:
    """
    Compute RBF (Gaussian) kernel matrix between two point sets.

    Parameters
    ----------
    X1 : np.ndarray, shape (n1, p)
        First set of points
    X2 : np.ndarray, shape (n2, p)
        Second set of points
    bandwidth : float, optional
        Kernel bandwidth (sigma). If None, uses median heuristic.

    Returns
    -------
    np.ndarray, shape (n1, n2)
        Kernel matrix K(x, y) = exp(-||x-y||^2/(2*bandwidth^2))
    """
    # Compute pairwise distances
    n1 = X1.shape[0]
    n2 = X2.shape[0]

    # Squared distances: ||x - y||^2
    sq_norms_1 = np.sum(X1**2, axis=1, keepdims=True)  # (n1, 1)
    sq_norms_2 = np.sum(X2**2, axis=1, keepdims=True)  # (n2, 1)

    sq_distances = sq_norms_1 + sq_norms_2.T - 2 * X1 @ X2.T

    # Ensure non-negative (numerical stability)
    sq_distances = np.maximum(sq_distances, 0)

    # Determine bandwidth if not provided
    if bandwidth is None:
        bandwidth = estimate_bandwidth_median(np.vstack([X1, X2]))

    # Compute kernel
    kernel = np.exp(-sq_distances / (2 * bandwidth**2))

    return kernel


def estimate_bandwidth_median(X: np.ndarray) -> float:
    """
    Estimate kernel bandwidth using median heuristic.

    The median heuristic sets bandwidth to the median pairwise distance
    in the pooled sample.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Data points

    Returns
    -------
    float
        Estimated bandwidth
    """
    n = X.shape[0]

    if n <= 1:
        return 1.0

    # Compute pairwise distances
    sq_distances = np.sum(X**2, axis=1, keepdims=True) + np.sum(X**2, axis=1) - 2 * X @ X.T
    sq_distances = np.maximum(sq_distances, 0)
    distances = np.sqrt(sq_distances)

    # Return median of upper triangle (excluding diagonal)
    triu_idx = np.triu_indices(n, k=1)
    distances_upper = distances[triu_idx]

    if len(distances_upper) > 0:
        bandwidth = np.median(distances_upper)
    else:
        bandwidth = 1.0

    return max(bandwidth, 1e-6)  # Avoid zero bandwidth


def compute_mmd(
    X1: np.ndarray,
    X2: np.ndarray,
    bandwidth: Optional[float] = None
) -> float:
    """
    Compute Maximum Mean Discrepancy between two distributions.

    MMD is a distance metric between probability distributions that equals
    zero if and only if the two distributions are identical. It can be
    estimated from finite samples and is computationally efficient.

    The squared MMD is:
        MMD^2(X, Y) = E_x,x'[k(x,x')] + E_y,y'[k(y,y')] - 2*E_x,y[k(x,y)]
    where k is a kernel function (here, RBF).

    Parameters
    ----------
    X1 : np.ndarray, shape (n1, p)
        Samples from first distribution
    X2 : np.ndarray, shape (n2, p)
        Samples from second distribution
    bandwidth : float, optional
        Kernel bandwidth. If None, uses median heuristic.

    Returns
    -------
    float
        Maximum Mean Discrepancy value (non-negative)

    References
    ----------
    Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A.
    (2012). A kernel two-sample test. The journal of machine learning research,
    13(1), 723-773.

    Examples
    --------
    >>> X1 = np.random.randn(100, 10)
    >>> X2 = np.random.randn(100, 10) + 0.3
    >>> mmd = compute_mmd(X1, X2)
    """
    X1 = np.asarray(X1)
    X2 = np.asarray(X2)

    if X1.ndim == 1:
        X1 = X1.reshape(-1, 1)
    if X2.ndim == 1:
        X2 = X2.reshape(-1, 1)

    n1 = X1.shape[0]
    n2 = X2.shape[0]

    # Compute kernels
    K11 = compute_rbf_kernel(X1, X1, bandwidth=bandwidth)
    K22 = compute_rbf_kernel(X2, X2, bandwidth=bandwidth)
    K12 = compute_rbf_kernel(X1, X2, bandwidth=bandwidth)

    # Compute MMD^2 (unbiased estimator)
    # E[k(X,X')] using upper triangle (excluding diagonal)
    mean_K11 = np.sum(np.triu(K11, k=1)) / (n1 * (n1 - 1)) if n1 > 1 else 0
    mean_K22 = np.sum(np.triu(K22, k=1)) / (n2 * (n2 - 1)) if n2 > 1 else 0

    # E[k(X,Y)]
    mean_K12 = np.mean(K12)

    # Compute squared MMD
    mmd_squared = mean_K11 + mean_K22 - 2 * mean_K12

    # Return MMD (ensure non-negative)
    mmd = np.sqrt(max(0.0, mmd_squared))

    return mmd


def compute_mean_distance(X1: np.ndarray, X2: np.ndarray) -> float:
    """
    Compute Euclidean distance between centroids.

    Simple statistic that measures the distance between the means of two
    distributions. Useful as a baseline or for quick checks.

    Parameters
    ----------
    X1 : np.ndarray, shape (n1, p)
        First sample
    X2 : np.ndarray, shape (n2, p)
        Second sample

    Returns
    -------
    float
        Euclidean distance between means

    Examples
    --------
    >>> X1 = np.array([[1, 2], [2, 3], [3, 4]])
    >>> X2 = np.array([[5, 6], [6, 7], [7, 8]])
    >>> compute_mean_distance(X1, X2)
    5.656854249492381
    """
    X1 = np.asarray(X1)
    X2 = np.asarray(X2)

    if X1.ndim == 1:
        X1 = X1.reshape(-1, 1)
    if X2.ndim == 1:
        X2 = X2.reshape(-1, 1)

    mean1 = X1.mean(axis=0)
    mean2 = X2.mean(axis=0)

    return np.linalg.norm(mean1 - mean2)


__all__ = [
    'compute_energy_distance',
    'compute_mmd',
    'compute_rbf_kernel',
    'estimate_bandwidth_median',
    'compute_mean_distance'
]
