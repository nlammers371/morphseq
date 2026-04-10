"""
Shared utilities for permutation testing.

This module provides the common infrastructure for all permutation tests:
- P-value calculation with consistent formulas
- Shuffle strategies (pool vs label)
- Standardized result format

These utilities enable DRY code and consistent statistical methodology across
classification-based, distribution-based, and future test types.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional


def compute_pvalue(
    observed: float,
    null_distribution: np.ndarray,
    alternative: str = "greater",
    pseudo_count: bool = True
) -> float:
    """
    Compute permutation p-value with consistent formula.

    Parameters
    ----------
    observed : float
        Observed test statistic
    null_distribution : np.ndarray
        Null distribution from permutations
    alternative : str, default="greater"
        Type of test:
        - "greater": tests if observed is significantly greater than null
        - "less": tests if observed is significantly less than null
        - "two-sided": tests if observed is significantly different from null
    pseudo_count : bool, default=True
        If True, use (k+1)/(n+1) formula to avoid zero p-values.
        This provides a conservative estimate: minimum p-value is 1/(n+1).
        If False, use k/n formula (can yield exact 0 or 1).

    Returns
    -------
    float
        Permutation p-value

    Notes
    -----
    The pseudo-count approach is statistically more conservative and is
    recommended for most applications. It prevents reporting p=0 which
    technically means "the observed value is more extreme than ALL permutations"
    but doesn't account for the finite number of permutations tested.

    Examples
    --------
    >>> null_dist = np.array([0.5, 0.6, 0.55, 0.52, 0.58])
    >>> compute_pvalue(0.75, null_dist, alternative="greater")
    0.16666666666666666  # (0+1)/(5+1)
    >>> compute_pvalue(0.75, null_dist, alternative="greater", pseudo_count=False)
    0.0  # 0/5
    """
    null = np.asarray(null_distribution)
    n = len(null)

    if alternative == "greater":
        k = np.sum(null >= observed)
    elif alternative == "less":
        k = np.sum(null <= observed)
    elif alternative == "two-sided":
        k = np.sum(np.abs(null - np.mean(null)) >= np.abs(observed - np.mean(null)))
    else:
        raise ValueError(f"Unknown alternative: {alternative}. Use 'greater', 'less', or 'two-sided'.")

    if pseudo_count:
        return (k + 1) / (n + 1)
    return k / n


def pool_shuffle(
    X1: np.ndarray,
    X2: np.ndarray,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pool-and-redistribute shuffle for distribution tests.

    Combines both samples, shuffles, then redistributes while maintaining
    original sample sizes. Tests the null hypothesis that X1 and X2 come
    from the same distribution.

    Parameters
    ----------
    X1 : np.ndarray, shape (n1, p)
        First sample
    X2 : np.ndarray, shape (n2, p)
        Second sample
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    X1_perm, X2_perm : tuple of np.ndarray
        Shuffled samples with original sizes preserved

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> X1 = np.array([[1, 2], [3, 4]])
    >>> X2 = np.array([[5, 6], [7, 8], [9, 10]])
    >>> X1_perm, X2_perm = pool_shuffle(X1, X2, rng)
    >>> X1_perm.shape
    (2, 2)
    >>> X2_perm.shape
    (3, 2)
    """
    combined = np.vstack([X1, X2])
    n1 = len(X1)
    perm_idx = rng.permutation(len(combined))
    return combined[perm_idx[:n1]], combined[perm_idx[n1:]]


def label_shuffle(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Label shuffle for classification tests.

    Shuffles class labels to test the null hypothesis that labels are
    independent of features.

    Parameters
    ----------
    y : np.ndarray
        Class labels
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    np.ndarray
        Shuffled labels

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> y = np.array(['A', 'A', 'B', 'B', 'B'])
    >>> y_shuffled = label_shuffle(y, rng)
    """
    return rng.permutation(y)


class PermutationResult:
    """
    Standardized return format for all permutation tests.

    This class ensures consistent output across different test types
    (classification, distribution, etc.) and provides convenient access
    to test results and diagnostics.

    Attributes
    ----------
    statistic_name : str
        Name of the test statistic (e.g., 'AUROC', 'energy', 'mmd')
    observed : float
        Observed value of the test statistic
    pvalue : float
        Permutation p-value
    null_distribution : np.ndarray
        Full null distribution from permutations
    null_mean : float
        Mean of null distribution
    null_std : float
        Standard deviation of null distribution
    metadata : dict
        Additional test-specific information

    Examples
    --------
    >>> null_dist = np.array([0.5, 0.52, 0.48, 0.51, 0.49])
    >>> result = PermutationResult(
    ...     statistic_name='AUROC',
    ...     observed=0.75,
    ...     pvalue=0.01,
    ...     null_distribution=null_dist,
    ...     n_permutations=100
    ... )
    >>> print(result)
    PermutationResult(AUROC=0.7500, p=0.0100)
    >>> result.to_dict()
    {'AUROC': 0.75, 'pvalue': 0.01, 'null_mean': 0.5, ...}
    """

    def __init__(
        self,
        statistic_name: str,
        observed: float,
        pvalue: float,
        null_distribution: np.ndarray,
        **metadata
    ):
        self.statistic_name = statistic_name
        self.observed = observed
        self.pvalue = pvalue
        self.null_distribution = np.asarray(null_distribution)
        self.null_mean = np.mean(self.null_distribution)
        self.null_std = np.std(self.null_distribution)
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            self.statistic_name: self.observed,
            'pvalue': self.pvalue,
            'null_mean': self.null_mean,
            'null_std': self.null_std,
            'null_distribution': self.null_distribution,
            **self.metadata
        }

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Test if result is significant at given alpha level."""
        return self.pvalue < alpha

    def __repr__(self):
        return f"PermutationResult({self.statistic_name}={self.observed:.4f}, p={self.pvalue:.4f})"

    def __str__(self):
        sig_marker = "***" if self.is_significant(0.001) else \
                     "**" if self.is_significant(0.01) else \
                     "*" if self.is_significant(0.05) else ""
        return (f"{self.statistic_name}: {self.observed:.4f} "
                f"(p={self.pvalue:.4f}{sig_marker}, "
                f"null: {self.null_mean:.4f}±{self.null_std:.4f})")


__all__ = [
    'compute_pvalue',
    'pool_shuffle',
    'label_shuffle',
    'PermutationResult'
]
