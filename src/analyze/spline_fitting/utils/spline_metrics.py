"""Quantitative metrics for comparing and analyzing splines.

This module provides metrics for:
- Error/distance measurements (RMSE, RMSD)
- Direction consistency between splines
- Dispersion along trajectories

Example:
    >>> from src.analyze.spline_fitting.utils import segment_direction_consistency
    >>>
    >>> # Compare direction consistency between splines
    >>> consistency = segment_direction_consistency(spline1, spline2, k=10)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# =============================================================================
# Basic Distance Metrics
# =============================================================================

def rmse(a, b):
    """Compute Root Mean Square Error between two arrays.

    Parameters
    ----------
    a, b : ndarray
        Arrays of same shape.

    Returns
    -------
    error : float
        RMSE value.
    """
    return np.sqrt(np.mean((a - b)**2))


def rmsd(X, Y):
    """Compute Root Mean Square Deviation between point sets.

    Parameters
    ----------
    X, Y : ndarray, shape (n_points, n_dims)
        Point clouds of same shape.

    Returns
    -------
    deviation : float
        RMSD value.
    """
    return np.sqrt(np.mean(np.sum((X - Y)**2, axis=1)))


def mean_l1_error(a, b):
    """Compute mean L1 (Manhattan) distance between point sets.

    Parameters
    ----------
    a, b : ndarray, shape (n_points, n_dims)
        Point clouds of same shape.

    Returns
    -------
    error : float
        Mean L1 distance.
    """
    return np.mean(np.sum(np.abs(a - b), axis=1))


# =============================================================================
# Direction Consistency Metrics
# =============================================================================

def _cosine_similarity(a, b):
    """Compute cosine similarity between two vectors.

    Parameters
    ----------
    a, b : ndarray
        Vectors (1D or 2D).

    Returns
    -------
    similarity : float
        Cosine similarity (-1 to 1).
    """
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0
    return np.dot(a.flatten(), b.flatten()) / (a_norm * b_norm)


def _segment_direction_metrics(data_a, data_b, k=10):
    """Compute segment-wise direction consistency between two curves.

    Divides curves into k segments and compares direction vectors using
    cosine similarity and covariance.

    Parameters
    ----------
    data_a, data_b : ndarray, shape (n_points, 3)
        Two curves to compare.
    k : int, default=10
        Number of segments.

    Returns
    -------
    avg_cosine_sim : float
        Average cosine similarity across segments.
    avg_cov : float
        Average covariance across segment directions.

    Notes
    -----
    Returns (np.nan, np.nan) if insufficient points for k segments.
    """
    min_len = min(len(data_a), len(data_b))
    data_a = data_a[:min_len]
    data_b = data_b[:min_len]

    if min_len < k + 1 or min_len == 0:
        return (np.nan, np.nan)

    # Define segments using data_b
    segment_indices = np.linspace(0, min_len - 1, k + 1, dtype=int)

    aligned_segment_vecs = []
    all_segment_vecs = []

    for i in range(k):
        start_idx = segment_indices[i]
        end_idx = segment_indices[i + 1]

        start_b = data_b[start_idx]
        end_b = data_b[end_idx]

        # Find closest points in data_a
        start_dists = np.linalg.norm(data_a - start_b, axis=1)
        closest_start_idx = np.argmin(start_dists)
        closest_start_a = data_a[closest_start_idx]

        end_dists = np.linalg.norm(data_a - end_b, axis=1)
        closest_end_idx = np.argmin(end_dists)
        closest_end_a = data_a[closest_end_idx]

        # Construct direction vectors
        vec_a = closest_end_a - closest_start_a
        vec_b = end_b - start_b

        # Normalize
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a > 0:
            vec_a = vec_a / norm_a
        else:
            vec_a = np.zeros(3)
        if norm_b > 0:
            vec_b = vec_b / norm_b
        else:
            vec_b = np.zeros(3)

        aligned_segment_vecs.append(vec_a)
        all_segment_vecs.append(vec_b)

    aligned_segment_vecs = np.array(aligned_segment_vecs)
    all_segment_vecs = np.array(all_segment_vecs)

    # Cosine similarities
    cos_sims = []
    for i in range(len(aligned_segment_vecs)):
        va = aligned_segment_vecs[i]
        vb = all_segment_vecs[i]
        sim = _cosine_similarity(va, vb)
        cos_sims.append(sim)

    avg_cosine_sim = np.mean(cos_sims) if len(cos_sims) > 0 else np.nan

    # Covariances per dimension
    covariances = []
    for dim_idx in range(3):
        dim_a = aligned_segment_vecs[:, dim_idx]
        dim_b = all_segment_vecs[:, dim_idx]
        if len(dim_a) > 1:
            cov = np.cov(dim_a, dim_b, bias=True)[0, 1]
        else:
            cov = np.nan
        covariances.append(cov)
    avg_cov = np.nanmean(covariances) if len(covariances) > 0 else np.nan

    return (avg_cosine_sim, avg_cov)


def segment_direction_consistency(spline_a, spline_b, k=10, coord_cols=None):
    """Compute direction consistency metrics between two splines.

    Parameters
    ----------
    spline_a, spline_b : ndarray or pd.DataFrame
        Spline coordinates. If DataFrame, must have coord_cols.
    k : int, default=10
        Number of segments for comparison.
    coord_cols : list of str, optional
        Column names if inputs are DataFrames.
        Defaults to ['PCA_1', 'PCA_2', 'PCA_3'].

    Returns
    -------
    avg_cosine_sim : float
        Average cosine similarity of segment directions.
    avg_cov : float
        Average covariance of segment directions.

    Examples
    --------
    >>> sim, cov = segment_direction_consistency(wt_spline, mut_spline, k=10)
    >>> print(f"Direction consistency: {sim:.3f}")
    """
    # Convert to numpy if DataFrame
    if isinstance(spline_a, pd.DataFrame):
        if coord_cols is None:
            coord_cols = ['PCA_1', 'PCA_2', 'PCA_3']
        spline_a = spline_a[coord_cols].values

    if isinstance(spline_b, pd.DataFrame):
        if coord_cols is None:
            coord_cols = ['PCA_1', 'PCA_2', 'PCA_3']
        spline_b = spline_b[coord_cols].values

    return _segment_direction_metrics(spline_a, spline_b, k=k)


# =============================================================================
# Dispersion Metrics
# =============================================================================

def compute_dispersion(df, pca_columns):
    """Compute average distance from centroid (dispersion).

    Measures how spread out points are around their center of mass.

    Parameters
    ----------
    df : pd.DataFrame
        Points with coordinate columns.
    pca_columns : list of str
        Column names for coordinates.

    Returns
    -------
    dispersion : float
        Average Euclidean distance from centroid.
    """
    if df.empty:
        return np.nan

    centroid = df[pca_columns].mean().values
    distances = np.linalg.norm(df[pca_columns].values - centroid, axis=1)
    return distances.mean()


def calculate_dispersion_metrics(
    splines_df,
    n=5,
    group_by_col="dataset",
    point_index_col="point_index",
    coord_cols=None
):
    """Calculate dispersion metrics along splines.

    Computes how trajectory spread changes along the curve:
    - Dispersion coefficient: slope of dispersion vs position
    - Initial/final dispersion: spread at start/end of trajectory

    Parameters
    ----------
    splines_df : pd.DataFrame
        Spline data with grouping column and point indices.
    n : int, default=5
        Number of points for initial/final dispersion.
    group_by_col : str, default='dataset'
        Column to group by.
    point_index_col : str, default='point_index'
        Column with point indices along spline.
    coord_cols : list of str, optional
        Coordinate columns. Defaults to ['PCA_1', 'PCA_2', 'PCA_3'].

    Returns
    -------
    metrics_df : pd.DataFrame
        Dispersion metrics per group with columns:
        - {group_by_col}: group identifier
        - disp_coefficient: normalized slope of dispersion
        - dispersion_first_n: avg dispersion in first n points
        - dispersion_last_n: avg dispersion in last n points
    """
    if coord_cols is None:
        coord_cols = ["PCA_1", "PCA_2", "PCA_3"]

    # Validate columns
    for col in coord_cols:
        if col not in splines_df.columns:
            raise ValueError(f"Missing required coordinate column: {col}")

    groups = splines_df[group_by_col].unique()
    results = []

    for group in groups:
        group_df = splines_df[splines_df[group_by_col] == group]
        point_indices = sorted(group_df[point_index_col].unique())

        dispersion_list = []
        point_index_list = []
        initial_dispersions = []
        last_dispersions = []

        for pid in point_indices:
            point_df = group_df[group_df[point_index_col] == pid]
            dispersion = compute_dispersion(point_df, coord_cols)

            dispersion_list.append(dispersion)
            point_index_list.append(pid)

            if pid < n:
                initial_dispersions.append(dispersion)
            if pid >= max(point_indices) - n + 1:
                last_dispersions.append(dispersion)

        # Linear regression for dispersion coefficient
        if len(point_index_list) < 2:
            disp_coefficient = np.nan
        else:
            X = np.array(point_index_list).reshape(-1, 1)
            y = np.array(dispersion_list)
            reg = LinearRegression().fit(X, y)
            disp_coefficient = reg.coef_[0] * len(point_indices)

        dispersion_first_n = np.mean(initial_dispersions) if initial_dispersions else np.nan
        dispersion_last_n = np.mean(last_dispersions) if last_dispersions else np.nan

        results.append({
            group_by_col: group,
            "disp_coefficient": disp_coefficient,
            "dispersion_first_n": dispersion_first_n,
            "dispersion_last_n": dispersion_last_n
        })

    return pd.DataFrame(results)
