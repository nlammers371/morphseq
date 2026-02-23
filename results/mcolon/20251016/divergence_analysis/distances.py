"""
Distance computation functions for morphological divergence analysis.

Implements multiple distance metrics for measuring how far embryos are
from a reference distribution (typically wild-type).
"""

import numpy as np
from scipy.spatial.distance import mahalanobis as scipy_mahalanobis
from scipy.spatial.distance import euclidean, cosine
from typing import Optional


def compute_mahalanobis_distance(
    X: np.ndarray,
    mu_ref: np.ndarray,
    cov_ref: np.ndarray,
    robust: bool = True,
    reg_param: float = 1e-6
) -> np.ndarray:
    """
    Compute Mahalanobis distance from reference distribution.
    
    The Mahalanobis distance accounts for correlations between features
    and is scale-invariant. It measures how many "standard deviations"
    away each point is from the reference centroid.
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Data points to compute distance for
    mu_ref : np.ndarray, shape (n_features,)
        Reference distribution mean (centroid)
    cov_ref : np.ndarray, shape (n_features, n_features)
        Reference distribution covariance matrix
    robust : bool, default=True
        If True, add regularization to handle near-singular covariance
    reg_param : float, default=1e-6
        Regularization parameter added to diagonal
    
    Returns
    -------
    np.ndarray, shape (n_samples,)
        Mahalanobis distance for each sample
    
    Notes
    -----
    Under multivariate normality, squared Mahalanobis distance follows
    a chi-squared distribution with df = n_features.
    
    Examples
    --------
    >>> X = np.random.randn(100, 10)
    >>> mu = X.mean(axis=0)
    >>> cov = np.cov(X.T)
    >>> distances = compute_mahalanobis_distance(X, mu, cov)
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    # Add regularization for numerical stability
    if robust:
        cov_ref = cov_ref + reg_param * np.eye(cov_ref.shape[0])
    
    # Compute inverse covariance
    try:
        cov_inv = np.linalg.inv(cov_ref)
    except np.linalg.LinAlgError:
        # If still singular, use pseudo-inverse
        cov_inv = np.linalg.pinv(cov_ref)
    
    # Compute distance for each sample
    distances = np.array([
        scipy_mahalanobis(x, mu_ref, cov_inv)
        for x in X
    ])
    
    return distances


def compute_euclidean_distance(
    X: np.ndarray,
    mu_ref: np.ndarray
) -> np.ndarray:
    """
    Compute Euclidean (L2) distance from reference centroid.
    
    Simple distance metric that doesn't account for correlations
    or feature scales. Good baseline for comparison.
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Data points to compute distance for
    mu_ref : np.ndarray, shape (n_features,)
        Reference distribution mean (centroid)
    
    Returns
    -------
    np.ndarray, shape (n_samples,)
        Euclidean distance for each sample
    
    Examples
    --------
    >>> X = np.random.randn(100, 10)
    >>> mu = X.mean(axis=0)
    >>> distances = compute_euclidean_distance(X, mu)
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    # Compute L2 distance from centroid
    distances = np.linalg.norm(X - mu_ref, axis=1)
    
    return distances


def compute_standardized_distance(
    X: np.ndarray,
    mu_ref: np.ndarray,
    std_ref: np.ndarray
) -> np.ndarray:
    """
    Compute standardized Euclidean distance.
    
    Like Euclidean but after z-scoring each feature by reference std.
    Accounts for different feature scales but not correlations.
    Middle ground between Euclidean and Mahalanobis.
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Data points to compute distance for
    mu_ref : np.ndarray, shape (n_features,)
        Reference distribution mean
    std_ref : np.ndarray, shape (n_features,)
        Reference distribution standard deviations
    
    Returns
    -------
    np.ndarray, shape (n_samples,)
        Standardized Euclidean distance for each sample
    
    Examples
    --------
    >>> X = np.random.randn(100, 10)
    >>> mu = X.mean(axis=0)
    >>> std = X.std(axis=0)
    >>> distances = compute_standardized_distance(X, mu, std)
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    # Avoid division by zero
    std_ref = np.where(std_ref < 1e-10, 1.0, std_ref)
    
    # Standardize and compute distance
    X_std = (X - mu_ref) / std_ref
    distances = np.linalg.norm(X_std, axis=1)
    
    return distances


def compute_cosine_distance(
    X: np.ndarray,
    mu_ref: np.ndarray
) -> np.ndarray:
    """
    Compute cosine distance from reference centroid.
    
    Measures the angle between vectors, ignoring magnitude.
    Good for detecting directional changes in morphological space.
    
    Distance is 1 - cosine_similarity, so:
    - 0 = same direction
    - 1 = orthogonal
    - 2 = opposite direction
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Data points to compute distance for
    mu_ref : np.ndarray, shape (n_features,)
        Reference distribution mean (centroid)
    
    Returns
    -------
    np.ndarray, shape (n_samples,)
        Cosine distance for each sample
    
    Examples
    --------
    >>> X = np.random.randn(100, 10)
    >>> mu = X.mean(axis=0)
    >>> distances = compute_cosine_distance(X, mu)
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    # Compute cosine distance for each sample
    distances = np.array([
        cosine(x, mu_ref) for x in X
    ])
    
    return distances


def compute_all_distances(
    X: np.ndarray,
    mu_ref: np.ndarray,
    cov_ref: Optional[np.ndarray] = None,
    std_ref: Optional[np.ndarray] = None
) -> dict:
    """
    Compute all distance metrics at once.
    
    Parameters
    ----------
    X : np.ndarray
        Data points
    mu_ref : np.ndarray
        Reference mean
    cov_ref : np.ndarray, optional
        Reference covariance (for Mahalanobis)
    std_ref : np.ndarray, optional
        Reference std deviations (for standardized distance)
    
    Returns
    -------
    dict
        Dictionary with keys: 'mahalanobis', 'euclidean', 'standardized', 'cosine'
    """
    results = {
        'euclidean': compute_euclidean_distance(X, mu_ref),
        'cosine': compute_cosine_distance(X, mu_ref)
    }
    
    if cov_ref is not None:
        results['mahalanobis'] = compute_mahalanobis_distance(X, mu_ref, cov_ref)
    
    if std_ref is not None:
        results['standardized'] = compute_standardized_distance(X, mu_ref, std_ref)
    
    return results


def detect_outliers_mahalanobis(
    distances: np.ndarray,
    n_features: int,
    alpha: float = 0.001
) -> np.ndarray:
    """
    Detect statistical outliers using Mahalanobis distance.
    
    Under normality, squared Mahalanobis distance follows chi-squared
    distribution. We can use this to detect extreme outliers.
    
    Parameters
    ----------
    distances : np.ndarray
        Mahalanobis distances
    n_features : int
        Number of features (degrees of freedom for chi-squared)
    alpha : float, default=0.001
        Significance level for outlier detection
    
    Returns
    -------
    np.ndarray
        Boolean array, True for outliers
    
    Examples
    --------
    >>> distances = compute_mahalanobis_distance(X, mu, cov)
    >>> outliers = detect_outliers_mahalanobis(distances, n_features=10)
    >>> print(f"Found {outliers.sum()} outliers")
    """
    from scipy.stats import chi2
    
    # Critical value from chi-squared distribution
    threshold = np.sqrt(chi2.ppf(1 - alpha, df=n_features))
    
    # Detect outliers
    is_outlier = distances > threshold
    
    return is_outlier
