"""
Distribution-based permutation tests.

Tests whether two samples come from the same distribution using
energy distance, Maximum Mean Discrepancy (MMD), and other distribution metrics.

These tests use pool-shuffle strategy: combine both samples, permute indices,
then redistribute. This tests the null hypothesis that both groups were drawn
from the same underlying distribution.
"""

import numpy as np
from typing import Optional, Callable, Dict, Any

from .permutation_utils import compute_pvalue, pool_shuffle, PermutationResult
from .distance_metrics import compute_energy_distance, compute_mmd, compute_mean_distance


def permutation_test_distribution(
    X1: np.ndarray,
    X2: np.ndarray,
    statistic: str = "energy",
    n_permutations: int = 1000,
    random_state: Optional[int] = None,
    pseudo_count: bool = True,
    **kwargs
) -> PermutationResult:
    """
    Generic distribution-based permutation test.

    Tests null hypothesis: X1 and X2 come from the same distribution.

    Parameters
    ----------
    X1 : np.ndarray, shape (n1, p)
        Samples from first distribution
    X2 : np.ndarray, shape (n2, p)
        Samples from second distribution
    statistic : str, default="energy"
        Test statistic to use:
        - "energy": Energy distance (Szekely-Rizzo)
        - "mmd": Maximum Mean Discrepancy with RBF kernel
        - "mean": Simple Euclidean distance between centroids
    n_permutations : int, default=1000
        Number of permutations for null distribution
    random_state : int, optional
        Random seed for reproducibility
    pseudo_count : bool, default=True
        Use (k+1)/(n+1) p-value formula to avoid exact zeros
    **kwargs
        Additional arguments passed to statistic function
        (e.g., bandwidth for MMD)

    Returns
    -------
    PermutationResult
        Test result with observed statistic, p-value, and null distribution

    Examples
    --------
    >>> X1 = np.random.randn(50, 10)
    >>> X2 = np.random.randn(50, 10) + 0.3
    >>> result = permutation_test_distribution(X1, X2, statistic="energy")
    >>> print(result)
    PermutationResult(energy=0.2134, p=0.0099)
    >>> result.is_significant(alpha=0.05)
    True

    >>> # With MMD and custom bandwidth
    >>> result_mmd = permutation_test_distribution(
    ...     X1, X2, statistic="mmd", bandwidth=1.5
    ... )
    """
    rng = np.random.default_rng(random_state)

    # Select statistic function
    if statistic == "energy":
        compute_stat = compute_energy_distance
    elif statistic == "mmd":
        # Wrap to pass kwargs
        compute_stat = lambda a, b: compute_mmd(a, b, **kwargs)
    elif statistic == "mean":
        compute_stat = compute_mean_distance
    else:
        raise ValueError(
            f"Unknown statistic: '{statistic}'. "
            "Choose from: 'energy', 'mmd', 'mean'"
        )

    # Observed statistic
    observed = compute_stat(X1, X2)

    # Null distribution via pool shuffle
    null_dist = []
    for _ in range(n_permutations):
        X1_perm, X2_perm = pool_shuffle(X1, X2, rng)
        null_dist.append(compute_stat(X1_perm, X2_perm))

    null_dist = np.array(null_dist)

    # Compute p-value
    pvalue = compute_pvalue(
        observed, null_dist,
        alternative="greater",
        pseudo_count=pseudo_count
    )

    # Build result
    metadata = {
        'n_permutations': n_permutations,
        'n1': len(X1),
        'n2': len(X2),
        'statistic_type': statistic
    }

    # Add bandwidth info for MMD
    if statistic == "mmd" and 'bandwidth' in kwargs:
        metadata['bandwidth'] = kwargs['bandwidth']

    return PermutationResult(
        statistic_name=statistic,
        observed=observed,
        pvalue=pvalue,
        null_distribution=null_dist,
        **metadata
    )


def permutation_test_energy(
    X1: np.ndarray,
    X2: np.ndarray,
    n_permutations: int = 1000,
    random_state: Optional[int] = None
) -> PermutationResult:
    """
    Permutation test for energy distance.

    Convenience wrapper for permutation_test_distribution with
    statistic="energy".

    Parameters
    ----------
    X1 : np.ndarray, shape (n1, p)
        Samples from first distribution
    X2 : np.ndarray, shape (n2, p)
        Samples from second distribution
    n_permutations : int, default=1000
        Number of permutations to perform
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    PermutationResult
        Test result with energy distance, p-value, and null distribution

    Examples
    --------
    >>> X1 = np.random.randn(50, 10)
    >>> X2 = np.random.randn(50, 10) + 0.3
    >>> result = permutation_test_energy(X1, X2, n_permutations=100)
    >>> print(f"Energy: {result.observed:.4f}, p-value: {result.pvalue:.4f}")
    """
    return permutation_test_distribution(
        X1, X2,
        statistic="energy",
        n_permutations=n_permutations,
        random_state=random_state
    )


def permutation_test_mmd(
    X1: np.ndarray,
    X2: np.ndarray,
    n_permutations: int = 1000,
    bandwidth: Optional[float] = None,
    random_state: Optional[int] = None
) -> PermutationResult:
    """
    Permutation test for Maximum Mean Discrepancy.

    Convenience wrapper for permutation_test_distribution with
    statistic="mmd".

    Parameters
    ----------
    X1 : np.ndarray, shape (n1, p)
        Samples from first distribution
    X2 : np.ndarray, shape (n2, p)
        Samples from second distribution
    n_permutations : int, default=1000
        Number of permutations to perform
    bandwidth : float, optional
        RBF kernel bandwidth. If None, uses median heuristic.
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    PermutationResult
        Test result with MMD value, p-value, and null distribution

    Examples
    --------
    >>> X1 = np.random.randn(50, 10)
    >>> X2 = np.random.randn(50, 10) + 0.3
    >>> result = permutation_test_mmd(X1, X2, n_permutations=100)
    >>> print(f"MMD: {result.observed:.4f}, p-value: {result.pvalue:.4f}")

    >>> # With custom bandwidth
    >>> result = permutation_test_mmd(X1, X2, bandwidth=1.5)
    """
    return permutation_test_distribution(
        X1, X2,
        statistic="mmd",
        n_permutations=n_permutations,
        random_state=random_state,
        bandwidth=bandwidth
    )


__all__ = [
    'permutation_test_distribution',
    'permutation_test_energy',
    'permutation_test_mmd'
]
