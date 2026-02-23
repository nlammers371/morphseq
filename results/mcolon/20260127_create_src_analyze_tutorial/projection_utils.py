"""
Cluster Projection Utilities

Functions for projecting new trajectories onto existing cluster assignments
using Dynamic Time Warping (DTW) distance.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from joblib import Parallel, delayed, cpu_count


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
    from src.analyze.utils.timeseries import _dtw_multivariate_pair

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
            X_source = np.pad(X_source, pad_width, mode='constant', constant_values=0)
            if verbose:
                print(f"    Padded source to shape: {X_source.shape}")

        if X_target.shape[1] < max_time:
            # Pad target
            pad_width = ((0, 0), (0, max_time - X_target.shape[1]), (0, 0))
            X_target = np.pad(X_target, pad_width, mode='constant', constant_values=0)
            if verbose:
                print(f"    Padded target to shape: {X_target.shape}")

    if n_jobs == -1:
        actual_jobs = cpu_count()
    else:
        actual_jobs = min(n_jobs, cpu_count())

    if verbose:
        print("\nComputing cross-dataset DTW distances...")
        print(f"  Source: {n_source} embryos")
        print(f"  Target: {n_target} embryos")
        print(f"  Total pairs: {n_source * n_target}")
        print(f"  Time points: {X_source.shape[1]}")

    # Generate all (source, target) pairs
    pairs = [(i, j) for i in range(n_source) for j in range(n_target)]

    # Parallel computation
    # Avoid joblib's per-batch logging; keep output readable and consistent.
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


def compare_cluster_frequencies(
    df_ref: pd.DataFrame,
    df_projected: pd.DataFrame,
    category_col: str = 'cluster_category',
    ref_label: str = 'Reference',
    projected_label: str = 'Projected'
) -> Tuple[pd.DataFrame, Dict]:
    """
    Compare cluster frequencies between reference and projected datasets.

    Parameters
    ----------
    df_ref : pd.DataFrame
        Reference dataset with cluster assignments
    df_projected : pd.DataFrame
        Projected dataset with cluster assignments
    category_col : str, default='cluster_category'
        Column containing cluster categories
    ref_label : str, default='Reference'
        Label for reference dataset in output
    projected_label : str, default='Projected'
        Label for projected dataset in output

    Returns
    -------
    freq_comparison : pd.DataFrame
        DataFrame with frequencies (%) for each cluster in both datasets
    stats : Dict
        Dictionary containing:
        - contingency_table: 2D array for statistical tests
        - chi2_statistic: Chi-square test statistic
        - chi2_pvalue: Chi-square p-value
        - fisher_odds_ratio: Fisher's exact odds ratio (for 2x2 only)
        - fisher_pvalue: Fisher's exact p-value (for 2x2 only)
    """
    from scipy.stats import chi2_contingency, fisher_exact

    # Compute frequencies
    ref_counts = df_ref[category_col].value_counts()
    ref_freqs = (ref_counts / ref_counts.sum() * 100).round(2)

    proj_counts = df_projected[category_col].value_counts()
    proj_freqs = (proj_counts / proj_counts.sum() * 100).round(2)

    # Combine into comparison DataFrame
    freq_comparison = pd.DataFrame({
        ref_label: ref_freqs,
        projected_label: proj_freqs
    }).fillna(0)

    # Build contingency table for all categories
    all_categories = sorted(set(ref_counts.index) | set(proj_counts.index))
    contingency = np.array([
        [ref_counts.get(cat, 0) for cat in all_categories],
        [proj_counts.get(cat, 0) for cat in all_categories]
    ])

    # Chi-square test
    chi2, p_chi2, dof, expected = chi2_contingency(contingency)

    stats = {
        'contingency_table': contingency,
        'categories': all_categories,
        'chi2_statistic': chi2,
        'chi2_pvalue': p_chi2,
        'chi2_dof': dof
    }

    # Fisher's exact test (only for 2x2 tables)
    if contingency.shape == (2, 2):
        odds_ratio, p_fisher = fisher_exact(contingency)
        stats['fisher_odds_ratio'] = odds_ratio
        stats['fisher_pvalue'] = p_fisher

    return freq_comparison, stats
