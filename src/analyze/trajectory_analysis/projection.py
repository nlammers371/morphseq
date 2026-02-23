"""
Trajectory Projection Utilities

Functions for projecting new trajectories onto existing cluster assignments
using DTW-based nearest neighbor matching. Handles automatic time window
detection to avoid extrapolation bias.

Key Functions
-------------
- project_onto_reference_clusters: High-level projection API (recommended)
- prepare_projection_arrays: Precompute aligned arrays (no DTW yet)
- project_onto_reference_clusters_from_distance: Assign using precomputed D_cross
- compute_cross_dtw_distance_matrix: Low-level cross-dataset DTW
- assign_clusters_nearest_neighbor: NN cluster assignment
- assign_clusters_knn_posterior: KNN with posteriors

Example
-------
>>> from src.analyze.trajectory_analysis import (
...     prepare_projection_arrays,
...     compute_cross_dtw_distance_matrix,
...     project_onto_reference_clusters_from_distance,
... )
>>>
>>> # Compute D_cross once, then reuse for assignments
>>> arrays = prepare_projection_arrays(
...     source_df=df_new_experiment,
...     reference_df=df_reference,
...     reference_cluster_map=cluster_map,
...     metrics=['baseline_deviation_normalized'],
... )
>>> D_cross = compute_cross_dtw_distance_matrix(
...     arrays.X_source, arrays.X_ref, sakoe_chiba_radius=20
... )
>>> assignments = project_onto_reference_clusters_from_distance(
...     D_cross=D_cross,
...     source_ids=arrays.source_ids,
...     ref_ids=arrays.ref_ids,
...     reference_cluster_map=cluster_map,
...     reference_category_map=category_map,
... )

IMPORTANT LIMITATIONS
---------------------
**Normalization and Time Window Dependency**

When projecting experiments with different temporal coverage, be aware that
global Z-score normalization can produce DIFFERENT normalized values for the
same raw measurement, depending on the time window used. This affects DTW
distance calculations and can lead to different cluster assignments.

Why this happens:
1. Normalization computes mean/std from ALL data in the time window
2. Different time windows → different data → different mean/std
3. Same raw value gets normalized differently
4. Different normalized values → different DTW distances → different assignments

Example:
  - Experiment A (12-47 hpf): Normalized using data from 12-47 hpf window
  - Experiment B (27-77 hpf): Normalized using data from 27-77 hpf window
  - Combined (12-77 hpf): Normalized using data from full 12-77 hpf window
  → Same embryo can get different distances to same reference!

Impact:
  - ~25% of embryos near cluster boundaries may be assigned differently
  - Correlation of distances for same pairs: r ≈ 0.65-0.94 (not 1.0)
  - NOT a bug in NaN-aware DTW - it's a fundamental normalization issue

Recommendations:
  1. **Acknowledge this limitation** - it's not easily solvable without a
     stable reference dataset for normalization
  2. **Project experiments separately** when they have very different temporal
     coverage (<50% overlap)
  3. **Use consistent time windows** when possible (batch experiments together)
  4. **Focus on high-confidence assignments** - embryos far from cluster
     boundaries will be stable regardless of time window
  5. **Consider K-NN with posteriors** to identify boundary cases (low posterior)

See Also:
  - Tutorial 04d: Direct distance comparison test
  - src/analyze/trajectory_analysis/KNOWN_ISSUES.md
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from joblib import Parallel, delayed, cpu_count


@dataclass
class ProjectionArrays:
    """Container for precomputed projection arrays."""
    X_source: np.ndarray
    source_ids: List[str]
    X_ref: np.ndarray
    ref_ids: List[str]
    time_grid: np.ndarray


def prepare_projection_arrays(
    source_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    reference_cluster_map: Dict[str, int],
    *,
    metrics: List[str] = None,
    time_col: str = 'predicted_stage_hpf',
    embryo_id_col: str = 'embryo_id',
    normalize: bool = True,
    verbose: bool = True,
) -> ProjectionArrays:
    """
    Prepare aligned arrays for projection without computing DTW.

    This performs steps 1-3 of the projection pipeline:
      1) Find temporal overlap
      2) Filter both datasets to overlap window
      3) Interpolate to common grid and return arrays

    Use this when you want to compute the cross-DTW matrix once and
    reuse it across different assignment methods or downstream analyses.
    """
    from .distance import prepare_multivariate_array

    if metrics is None:
        metrics = ['baseline_deviation_normalized']

    # Filter reference to only embryos with cluster assignments
    valid_ref_ids = list(reference_cluster_map.keys())
    reference_df = reference_df[reference_df[embryo_id_col].isin(valid_ref_ids)].copy()

    if verbose:
        print("="*80)
        print("PREPARE PROJECTION ARRAYS")
        print("="*80)
        print(f"\nSource embryos: {source_df[embryo_id_col].nunique()}")
        print(f"Reference embryos: {reference_df[embryo_id_col].nunique()}")
        print(f"Metrics: {metrics}")

    # Step 1: Auto-detect shared time window
    source_min = source_df[time_col].min()
    source_max = source_df[time_col].max()
    ref_min = reference_df[time_col].min()
    ref_max = reference_df[time_col].max()

    window_start = max(source_min, ref_min)
    window_end = min(source_max, ref_max)

    if window_start >= window_end:
        raise ValueError(
            f"No temporal overlap between source and reference:\n"
            f"  Source: {source_min:.1f} - {source_max:.1f} hpf\n"
            f"  Reference: {ref_min:.1f} - {ref_max:.1f} hpf"
        )

    if verbose:
        print(f"\nProjection time window: {window_start:.1f} - {window_end:.1f} hpf")
        print(f"  (intersection of source and reference coverage)")

    # Step 2: Filter to shared window
    source_filtered = source_df[
        (source_df[time_col] >= window_start) &
        (source_df[time_col] <= window_end)
    ].copy()

    ref_filtered = reference_df[
        (reference_df[time_col] >= window_start) &
        (reference_df[time_col] <= window_end)
    ].copy()

    if verbose:
        print(f"\nFiltered datasets:")
        print(f"  Source: {source_filtered[embryo_id_col].nunique()} embryos")
        print(f"  Reference: {ref_filtered[embryo_id_col].nunique()} embryos")

    # Step 3: Prepare arrays on common grid
    if verbose:
        print(f"\nPreparing reference array...")

    X_ref, ref_ids, time_grid = prepare_multivariate_array(
        ref_filtered,
        metrics=metrics,
        time_col=time_col,
        embryo_id_col=embryo_id_col,
        normalize=normalize,
        verbose=False
    )

    if verbose:
        print(f"  Shape: {X_ref.shape}")
        print(f"  Time grid: {len(time_grid)} points ({time_grid.min():.1f} - {time_grid.max():.1f} hpf)")
        print(f"\nPreparing source array...")

    X_source, source_ids, _ = prepare_multivariate_array(
        source_filtered,
        metrics=metrics,
        time_col=time_col,
        embryo_id_col=embryo_id_col,
        time_grid=time_grid,  # Use same grid as reference
        normalize=normalize,
        verbose=False
    )

    if verbose:
        print(f"  Shape: {X_source.shape}")

    return ProjectionArrays(
        X_source=X_source,
        source_ids=source_ids,
        X_ref=X_ref,
        ref_ids=ref_ids,
        time_grid=time_grid
    )


def project_onto_reference_clusters_from_distance(
    D_cross: np.ndarray,
    source_ids: List[str],
    ref_ids: List[str],
    reference_cluster_map: Dict[str, int],
    reference_category_map: Optional[Dict[str, str]] = None,
    *,
    method: str = 'nearest_neighbor',
    k: int = 5,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Assign clusters using a precomputed cross-DTW distance matrix.

    This is the fast path: compute D_cross once, then reuse this function
    for different assignment methods without recomputing DTW.
    """
    if verbose:
        print(f"\nAssigning clusters via {method}...")

    if method == 'nearest_neighbor':
        assignments = assign_clusters_nearest_neighbor(
            D_cross,
            source_ids,
            ref_ids,
            reference_cluster_map,
            reference_category_map
        )
    elif method == 'knn':
        assignments = assign_clusters_knn_posterior(
            D_cross,
            source_ids,
            ref_ids,
            reference_cluster_map,
            k=k,
            target_category_map=reference_category_map
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'nearest_neighbor' or 'knn'")

    if verbose:
        print(f"\n✓ Projected {len(assignments)} embryos")
        if 'cluster_category' in assignments.columns:
            print(f"\nCluster distribution:")
            for cat, count in assignments['cluster_category'].value_counts().items():
                pct = count / len(assignments) * 100
                print(f"  {cat}: {count} ({pct:.1f}%)")

    return assignments


def project_onto_reference_clusters(
    source_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    reference_cluster_map: Dict[str, int],
    reference_category_map: Optional[Dict[str, str]] = None,
    metrics: List[str] = None,
    time_col: str = 'predicted_stage_hpf',
    embryo_id_col: str = 'embryo_id',
    sakoe_chiba_radius: int = 20,
    method: str = 'nearest_neighbor',
    k: int = 5,
    normalize: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Project source embryos onto reference cluster assignments using DTW.

    Automatically determines the temporal overlap window between source and
    reference datasets to avoid extrapolation bias. This is the recommended
    high-level API for cluster projection.

    Workflow:
    ---------
    1. Find temporal intersection (shared time window)
    2. Filter both datasets to intersection
    3. Interpolate to common grid (within shared window only)
    4. Compute cross-DTW distances
    5. Assign clusters via nearest neighbor or KNN

    Parameters
    ----------
    source_df : pd.DataFrame
        Source trajectories to project (long format with time_col and metrics)
    reference_df : pd.DataFrame
        Reference trajectories with known cluster assignments
    reference_cluster_map : Dict[str, int]
        Mapping from reference embryo_id to cluster number
    reference_category_map : Dict[str, str], optional
        Mapping from reference embryo_id to cluster category name
    metrics : List[str], optional
        Metric columns to use for DTW (default: ['baseline_deviation_normalized'])
    time_col : str, default='predicted_stage_hpf'
        Name of time column
    embryo_id_col : str, default='embryo_id'
        Name of embryo ID column
    sakoe_chiba_radius : int, default=20
        DTW warping window constraint
    method : str, default='nearest_neighbor'
        Assignment method: 'nearest_neighbor' or 'knn'
    k : int, default=5
        Number of neighbors for KNN method
    normalize : bool, default=True
        Whether to Z-score normalize metrics
    verbose : bool, default=True
        Print progress information

    Returns
    -------
    assignments : pd.DataFrame
        Cluster assignments with columns:
        - embryo_id: source embryo ID
        - cluster: assigned cluster number
        - cluster_category: assigned category (if category_map provided)
        - nearest_distance: DTW distance to nearest reference (NN method)
        - nearest_embryo_id: ID of nearest reference (NN method)
        - max_posterior: probability of modal cluster (KNN method)
        - p_cluster_X: posterior for each cluster (KNN method)
    time_grid : np.ndarray
        Time grid used for projection (for reference)

    Notes
    -----
    - Automatically uses only the temporal overlap between source and reference
    - No extrapolation is performed - only real data is compared
    - Reference embryos are filtered to only those in reference_cluster_map

    Example
    -------
    >>> assignments, time_grid = project_onto_reference_clusters(
    ...     source_df=df_new_experiment,
    ...     reference_df=df_reference,
    ...     reference_cluster_map=cluster_map,
    ...     reference_category_map=category_map,
    ... )
    >>> print(assignments['cluster_category'].value_counts())
    """
    arrays = prepare_projection_arrays(
        source_df=source_df,
        reference_df=reference_df,
        reference_cluster_map=reference_cluster_map,
        metrics=metrics,
        time_col=time_col,
        embryo_id_col=embryo_id_col,
        normalize=normalize,
        verbose=verbose
    )

    # Step 4: Compute cross-DTW distances
    if verbose:
        print(f"\nComputing cross-DTW distances...")
        print(f"  Pairs: {len(arrays.source_ids)} × {len(arrays.ref_ids)} = {len(arrays.source_ids) * len(arrays.ref_ids)}")

    D_cross = compute_cross_dtw_distance_matrix(
        arrays.X_source,
        arrays.X_ref,
        sakoe_chiba_radius=sakoe_chiba_radius,
        n_jobs=-1,
        verbose=verbose
    )

    # Step 5: Assign clusters
    assignments = project_onto_reference_clusters_from_distance(
        D_cross=D_cross,
        source_ids=arrays.source_ids,
        ref_ids=arrays.ref_ids,
        reference_cluster_map=reference_cluster_map,
        reference_category_map=reference_category_map,
        method=method,
        k=k,
        verbose=verbose
    )

    return assignments, arrays.time_grid


def compute_cross_dtw_distance_matrix(
    X_source: np.ndarray,
    X_target: np.ndarray,
    sakoe_chiba_radius: int = 3,
    n_jobs: int = -1,
    verbose: bool = True
) -> np.ndarray:
    """
    Compute DTW distances from source embryos to target embryos.

    This computes a cross-dataset distance matrix where each element D[i, j]
    represents the DTW distance from source embryo i to target embryo j.

    Parameters
    ----------
    X_source : np.ndarray
        Source trajectories with shape (N_source, T, M)
        - N_source: number of source embryos
        - T: number of timepoints
        - M: number of metrics
    X_target : np.ndarray
        Target reference trajectories with shape (N_target, T, M)
        Must have same T and M as X_source
    sakoe_chiba_radius : int, default=3
        DTW warping window (Sakoe-Chiba band constraint)
    n_jobs : int, default=-1
        Number of parallel jobs (-1 for all CPUs)
    verbose : bool, default=True
        Print progress information

    Returns
    -------
    D_cross : np.ndarray
        Distance matrix with shape (N_source, N_target)
        D_cross[i, j] = DTW distance from source[i] to target[j]

    Notes
    -----
    - Both X_source and X_target must use the SAME time grid
    - Uses multivariate DTW (Euclidean distance in feature space)
    - Parallelized using joblib for efficiency
    """
    from analyze.utils.timeseries import _dtw_multivariate_pair

    n_source = X_source.shape[0]
    n_target = X_target.shape[0]

    # Validate inputs
    if X_source.shape[2] != X_target.shape[2]:
        raise ValueError(
            f"Metric dimension mismatch: source has {X_source.shape[2]} metrics, "
            f"target has {X_target.shape[2]}"
        )

    # Handle time dimension mismatch by padding shorter array
    if X_source.shape[1] != X_target.shape[1]:
        if verbose:
            print("  WARNING: Time dimension mismatch detected!")
            print(f"    Source: {X_source.shape[1]} timepoints")
            print(f"    Target: {X_target.shape[1]} timepoints")
            print(f"    Padding shorter array with zeros to match longer one...")

        max_time = max(X_source.shape[1], X_target.shape[1])

        if X_source.shape[1] < max_time:
            # Pad source
            pad_width = ((0, 0), (0, max_time - X_source.shape[1]), (0, 0))
            X_source = np.pad(X_source, pad_width, mode='constant', constant_values=np.nan)
            if verbose:
                print(f"    Padded source to shape: {X_source.shape}")

        if X_target.shape[1] < max_time:
            # Pad target
            pad_width = ((0, 0), (0, max_time - X_target.shape[1]), (0, 0))
            X_target = np.pad(X_target, pad_width, mode='constant', constant_values=np.nan)
            if verbose:
                print(f"    Padded target to shape: {X_target.shape}")

    if n_jobs == -1:
        actual_jobs = cpu_count()
    else:
        actual_jobs = min(n_jobs, cpu_count())

    if verbose:
        print(f"  Computing DTW distances...")
        print(f"    Source: {n_source} embryos")
        print(f"    Target: {n_target} embryos")
        print(f"    Total pairs: {n_source * n_target}")

    # Generate all (source, target) pairs
    pairs = [(i, j) for i in range(n_source) for j in range(n_target)]

    # Parallel computation
    results = Parallel(n_jobs=actual_jobs, verbose=0)(
        delayed(_dtw_multivariate_pair)(
            X_source[i], X_target[j], window=sakoe_chiba_radius
        )
        for i, j in pairs
    )

    # Build distance matrix
    D_cross = np.zeros((n_source, n_target))
    for (i, j), dist in zip(pairs, results):
        D_cross[i, j] = dist

    if verbose:
        print(f"\n  Distance statistics:")
        print(f"    Min: {D_cross.min():.4f}")
        print(f"    Max: {D_cross.max():.4f}")
        print(f"    Mean: {D_cross.mean():.4f}")
        print(f"    Median: {np.median(D_cross):.4f}")

    return D_cross


def assign_clusters_nearest_neighbor(
    D_cross: np.ndarray,
    source_embryo_ids: List[str],
    target_embryo_ids: List[str],
    target_cluster_map: Dict[str, int],
    target_category_map: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Assign source embryos to clusters via nearest neighbor method.

    For each source embryo, finds the closest target embryo in DTW space
    and assigns it to the same cluster.

    Parameters
    ----------
    D_cross : np.ndarray
        Cross-dataset distance matrix (N_source, N_target)
    source_embryo_ids : List[str]
        Source embryo IDs (length N_source)
    target_embryo_ids : List[str]
        Target embryo IDs (length N_target)
    target_cluster_map : Dict[str, int]
        Mapping from target embryo_id to cluster number
    target_category_map : Dict[str, str], optional
        Mapping from target embryo_id to cluster category name

    Returns
    -------
    df_assignments : pd.DataFrame
        DataFrame with columns:
        - embryo_id: source embryo ID
        - nearest_embryo_id: closest target embryo
        - nearest_distance: DTW distance to nearest target
        - cluster: assigned cluster number
        - cluster_category: assigned cluster category (if target_category_map provided)
    """
    assignments = []

    for i, source_id in enumerate(source_embryo_ids):
        # Find nearest target embryo
        distances = D_cross[i, :]
        nearest_idx = np.argmin(distances)
        nearest_id = target_embryo_ids[nearest_idx]
        nearest_dist = distances[nearest_idx]

        # Get cluster of nearest target
        cluster = target_cluster_map.get(nearest_id, np.nan)

        record = {
            'embryo_id': source_id,
            'nearest_embryo_id': nearest_id,
            'nearest_distance': nearest_dist,
            'cluster': cluster
        }

        # Add category if available
        if target_category_map is not None:
            category = target_category_map.get(nearest_id, None)
            record['cluster_category'] = category

        assignments.append(record)

    return pd.DataFrame(assignments)


def assign_clusters_knn_posterior(
    D_cross: np.ndarray,
    source_embryo_ids: List[str],
    target_embryo_ids: List[str],
    target_cluster_map: Dict[str, int],
    k: int = 5,
    target_category_map: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Assign source embryos to clusters via K-NN with posterior probabilities.

    For each source embryo, finds k nearest target embryos and computes
    a probability distribution over clusters based on their votes.

    Parameters
    ----------
    D_cross : np.ndarray
        Cross-dataset distance matrix (N_source, N_target)
    source_embryo_ids : List[str]
        Source embryo IDs (length N_source)
    target_embryo_ids : List[str]
        Target embryo IDs (length N_target)
    target_cluster_map : Dict[str, int]
        Mapping from target embryo_id to cluster number
    k : int, default=5
        Number of nearest neighbors to use
    target_category_map : Dict[str, str], optional
        Mapping from target embryo_id to cluster category name

    Returns
    -------
    df_assignments : pd.DataFrame
        DataFrame with columns:
        - embryo_id: source embryo ID
        - cluster: modal (most common) cluster among k neighbors
        - max_posterior: probability of modal cluster
        - p_cluster_X: posterior probability for each cluster X
        - cluster_category: category of modal cluster (if provided)
    """
    # Get unique clusters (filter out NaN)
    all_clusters = [c for c in target_cluster_map.values() if pd.notna(c)]
    unique_clusters = sorted(set(all_clusters))

    assignments = []

    for i, source_id in enumerate(source_embryo_ids):
        distances = D_cross[i, :]

        # Get k nearest neighbors
        knn_indices = np.argsort(distances)[:k]
        knn_ids = [target_embryo_ids[idx] for idx in knn_indices]
        knn_clusters = [target_cluster_map.get(eid, np.nan) for eid in knn_ids]

        # Compute cluster votes (simple voting)
        cluster_counts = {}
        for c in knn_clusters:
            if pd.notna(c):
                cluster_counts[c] = cluster_counts.get(c, 0) + 1

        # Compute posteriors
        total = sum(cluster_counts.values())
        if total > 0:
            posteriors = {c: cluster_counts.get(c, 0) / total for c in unique_clusters}
        else:
            posteriors = {c: 0.0 for c in unique_clusters}

        # Modal cluster (most common)
        if cluster_counts:
            modal_cluster = max(cluster_counts.keys(), key=lambda c: cluster_counts[c])
            max_posterior = posteriors.get(modal_cluster, 0)
        else:
            modal_cluster = np.nan
            max_posterior = 0

        record = {
            'embryo_id': source_id,
            'cluster': modal_cluster,
            'max_posterior': max_posterior,
            'cluster_category': None,
        }

        # Add posterior for each cluster
        for c in unique_clusters:
            record[f'p_cluster_{int(c)}'] = posteriors.get(c, 0)

        # Add category if available
        if target_category_map is not None and pd.notna(modal_cluster):
            # Get category from one of the knn embryos with this cluster
            for eid in knn_ids:
                if target_cluster_map.get(eid) == modal_cluster:
                    record['cluster_category'] = target_category_map.get(eid)
                    break

        assignments.append(record)

    return pd.DataFrame(assignments)


__all__ = [
    'project_onto_reference_clusters',
    'compute_cross_dtw_distance_matrix',
    'assign_clusters_nearest_neighbor',
    'assign_clusters_knn_posterior',
]
