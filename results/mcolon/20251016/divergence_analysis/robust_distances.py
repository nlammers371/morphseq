"""
Robust distance computation methods that handle high-dimensional, low-sample scenarios.

These methods address the "curse of dimensionality" problem where:
- n_samples << n_features (e.g., 15-20 samples, 80 dimensions)
- Standard covariance estimation fails (near-singular matrices)
- Mahalanobis distances become unstable
"""

import numpy as np
from typing import Optional, Tuple
from sklearn.covariance import LedoitWolf, OAS, ShrunkCovariance
from sklearn.decomposition import PCA


def compute_shrinkage_mahalanobis(
    X: np.ndarray,
    mu_ref: np.ndarray,
    X_ref: np.ndarray,
    method: str = 'ledoit_wolf'
) -> np.ndarray:
    """
    Compute Mahalanobis distance with shrinkage-regularized covariance.

    Uses covariance shrinkage to stabilize estimation when n_samples < n_features.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Test points
    mu_ref : np.ndarray, shape (n_features,)
        Reference mean
    X_ref : np.ndarray, shape (n_ref_samples, n_features)
        Reference samples (needed for shrinkage estimation)
    method : str, default='ledoit_wolf'
        Shrinkage method: 'ledoit_wolf', 'oas', or 'basic'

    Returns
    -------
    np.ndarray
        Mahalanobis distances

    Notes
    -----
    Shrinkage methods interpolate between sample covariance and a
    well-conditioned target (usually identity or diagonal):

    Σ_shrink = (1-α)·Σ_sample + α·Target

    where α is chosen optimally (Ledoit-Wolf, OAS) or fixed (basic).
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)

    # Choose shrinkage estimator
    if method == 'ledoit_wolf':
        estimator = LedoitWolf(assume_centered=False)
    elif method == 'oas':
        estimator = OAS(assume_centered=False)
    elif method == 'basic':
        estimator = ShrunkCovariance(shrinkage=0.5, assume_centered=False)
    else:
        raise ValueError(f"Unknown shrinkage method: {method}")

    # Fit to reference data
    estimator.fit(X_ref)
    cov_shrunk = estimator.covariance_

    # Compute distances
    try:
        cov_inv = np.linalg.inv(cov_shrunk)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_shrunk)

    diff = X - mu_ref
    distances = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

    return distances


def compute_pca_mahalanobis(
    X: np.ndarray,
    mu_ref: np.ndarray,
    X_ref: np.ndarray,
    n_components: Optional[int] = None,
    explained_variance_threshold: float = 0.95
) -> Tuple[np.ndarray, PCA]:
    """
    Compute Mahalanobis distance in PCA-reduced space.

    Projects data to lower-dimensional PCA space before computing distance.
    This is robust when n_samples < n_features.

    Parameters
    ----------
    X : np.ndarray
        Test points
    mu_ref : np.ndarray
        Reference mean
    X_ref : np.ndarray
        Reference samples
    n_components : int, optional
        Number of PCA components. If None, uses explained_variance_threshold
    explained_variance_threshold : float, default=0.95
        Keep enough components to explain this fraction of variance

    Returns
    -------
    distances : np.ndarray
        Mahalanobis distances in PCA space
    pca : PCA
        Fitted PCA object
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)

    # Fit PCA to reference
    if n_components is None:
        # Determine components by explained variance
        pca = PCA(n_components=explained_variance_threshold, svd_solver='full')
    else:
        pca = PCA(n_components=n_components)

    pca.fit(X_ref)

    # Transform to PCA space
    X_pca = pca.transform(X)
    mu_ref_pca = pca.transform(mu_ref.reshape(1, -1))[0]
    X_ref_pca = pca.transform(X_ref)

    # Compute covariance in PCA space (should be well-conditioned)
    cov_pca = np.cov(X_ref_pca.T)

    # Add small regularization just in case
    cov_pca += 1e-6 * np.eye(cov_pca.shape[0])

    # Compute distances
    try:
        cov_inv = np.linalg.inv(cov_pca)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_pca)

    diff = X_pca - mu_ref_pca
    distances = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

    return distances, pca


def compute_diagonal_mahalanobis(
    X: np.ndarray,
    mu_ref: np.ndarray,
    std_ref: np.ndarray
) -> np.ndarray:
    """
    Compute Mahalanobis distance with diagonal covariance (no correlations).

    Equivalent to standardized Euclidean distance. More stable than full
    Mahalanobis when sample size is small.

    Parameters
    ----------
    X : np.ndarray
        Test points
    mu_ref : np.ndarray
        Reference mean
    std_ref : np.ndarray
        Reference standard deviations

    Returns
    -------
    np.ndarray
        Diagonal Mahalanobis distances
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)

    # Avoid division by zero
    std_ref = np.where(std_ref < 1e-10, 1.0, std_ref)

    # Standardize and compute distance
    X_std = (X - mu_ref) / std_ref
    distances = np.linalg.norm(X_std, axis=1)

    return distances


def compute_robust_mahalanobis_suite(
    X: np.ndarray,
    mu_ref: np.ndarray,
    cov_ref: np.ndarray,
    std_ref: np.ndarray,
    X_ref: np.ndarray
) -> dict:
    """
    Compute multiple robust Mahalanobis variants for comparison.

    Parameters
    ----------
    X : np.ndarray
        Test points
    mu_ref : np.ndarray
        Reference mean
    cov_ref : np.ndarray
        Sample covariance (may be singular)
    std_ref : np.ndarray
        Reference standard deviations
    X_ref : np.ndarray
        Reference samples

    Returns
    -------
    dict
        Dictionary with keys:
        - 'diagonal': Diagonal Mahalanobis (no correlations)
        - 'shrinkage_lw': Ledoit-Wolf shrinkage
        - 'shrinkage_oas': Oracle Approximating Shrinkage
        - 'pca_95': PCA with 95% variance explained
        - 'pca_50': PCA with 50% variance explained
    """
    results = {}

    # 1. Diagonal (fastest, most stable)
    results['diagonal'] = compute_diagonal_mahalanobis(X, mu_ref, std_ref)

    # 2. Shrinkage methods
    try:
        results['shrinkage_lw'] = compute_shrinkage_mahalanobis(
            X, mu_ref, X_ref, method='ledoit_wolf'
        )
    except Exception as e:
        print(f"Warning: Ledoit-Wolf failed: {e}")
        results['shrinkage_lw'] = np.full(len(X), np.nan)

    try:
        results['shrinkage_oas'] = compute_shrinkage_mahalanobis(
            X, mu_ref, X_ref, method='oas'
        )
    except Exception as e:
        print(f"Warning: OAS failed: {e}")
        results['shrinkage_oas'] = np.full(len(X), np.nan)

    # 3. PCA-based methods
    try:
        results['pca_95'], _ = compute_pca_mahalanobis(
            X, mu_ref, X_ref, explained_variance_threshold=0.95
        )
    except Exception as e:
        print(f"Warning: PCA 95% failed: {e}")
        results['pca_95'] = np.full(len(X), np.nan)

    try:
        results['pca_50'], _ = compute_pca_mahalanobis(
            X, mu_ref, X_ref, explained_variance_threshold=0.50
        )
    except Exception as e:
        print(f"Warning: PCA 50% failed: {e}")
        results['pca_50'] = np.full(len(X), np.nan)

    return results


def recommend_distance_method(n_samples: int, n_features: int) -> str:
    """
    Recommend appropriate distance method based on data dimensions.

    Parameters
    ----------
    n_samples : int
        Number of reference samples
    n_features : int
        Number of features (dimensions)

    Returns
    -------
    str
        Recommended method name
    """
    ratio = n_samples / n_features

    if ratio < 0.5:
        return "pca_50"  # Severe undersampling
    elif ratio < 1.0:
        return "pca_95"  # Moderate undersampling
    elif ratio < 3.0:
        return "shrinkage_lw"  # Mild undersampling
    else:
        return "standard"  # Standard Mahalanobis should work
