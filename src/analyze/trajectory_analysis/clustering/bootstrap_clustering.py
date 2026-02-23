"""
Bootstrap Clustering

Bootstrap resampling methods for consensus clustering with quality assessment.

Functions
---------
- run_bootstrap_hierarchical: Bootstrap hierarchical clustering with consensus labels
- run_bootstrap_kmedoids: Bootstrap k-medoids clustering
- run_bootstrap_projection: Bootstrap projection for uncertainty quantification
- compute_consensus_labels: Compute consensus cluster labels from bootstrap iterations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from ..config import N_BOOTSTRAP, BOOTSTRAP_FRAC, RANDOM_SEED
from analyze.utils import resampling as resample

try:
    from sklearn_extra.cluster import KMedoids
    KMEDOIDS_AVAILABLE = True
except ImportError:
    KMEDOIDS_AVAILABLE = False


def run_bootstrap_hierarchical(
    D: np.ndarray,
    k: int,
    embryo_ids: List[str],
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    frac: float = BOOTSTRAP_FRAC,
    random_state: int = RANDOM_SEED,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Bootstrap hierarchical clustering with label alignment.

    Performs repeated hierarchical clustering on random subsamples of the data,
    storing the resulting cluster labels for posterior probability computation.

    Parameters
    ----------
    D : np.ndarray
        Distance matrix (n × n)
    k : int
        Number of clusters
    embryo_ids : list of str
        Embryo identifiers (required for tracking). Should encode experiment/run info.
        Example: ['cep290_wt_run1_emb_01', 'cep290_wt_run1_emb_02', ...]
    n_bootstrap : int, default=100
        Number of bootstrap iterations
    frac : float, default=0.8
        Fraction of samples per bootstrap
    random_state : int, default=42
        Random seed for reproducibility
    verbose : bool, default=False
        Print progress

    Returns
    -------
    bootstrap_results_dict : dict
        - 'embryo_ids': list of str (copy of input)
        - 'reference_labels': np.ndarray, consensus labels from full data
        - 'bootstrap_results': list of dicts
            - 'labels': np.ndarray (-1 for unsampled)
            - 'indices': np.ndarray of sampled indices
            - 'silhouette': float
        - 'n_clusters': int

    Examples
    --------
    >>> D = compute_dtw_distance_matrix(trajectories)
    >>> embryo_ids = ['emb_01', 'emb_02', 'emb_03', ...]
    >>> results = run_bootstrap_hierarchical(D, k=3, embryo_ids=embryo_ids, n_bootstrap=100)
    >>> reference_labels = results['reference_labels']
    >>> # Lookup embryo by ID
    >>> idx = results['embryo_ids'].index('emb_02')
    >>> label = results['reference_labels'][idx]
    """
    n_samples = len(D)

    # Compute reference labels from full data
    clusterer = AgglomerativeClustering(
        n_clusters=k,
        linkage='average',
        metric='precomputed'
    )
    reference_labels = clusterer.fit_predict(D)

    # Subsample size — preserve original ceil semantics
    n_to_sample = max(int(np.ceil(frac * n_samples)), 1)

    if verbose:
        print(f"Running {n_bootstrap} bootstrap iterations...")
        print(f"  Sampling {n_to_sample}/{n_samples} samples per iteration")

    # Scorer for the resampling framework
    def _hierarchical_scorer(data, rng):
        n = data["n"]
        k_val = data["k"]
        idx = np.sort(data["indices"]) if "indices" in data else np.arange(n)
        D_sub = data["D"][np.ix_(idx, idx)]

        clusterer_boot = AgglomerativeClustering(
            n_clusters=k_val,
            linkage='average',
            metric='precomputed'
        )
        labels_subset = clusterer_boot.fit_predict(D_sub)

        labels_full = np.full(n, -1, dtype=int)
        labels_full[idx] = labels_subset

        try:
            sil = float(silhouette_score(D_sub, labels_subset, metric='precomputed'))
        except Exception:
            sil = np.nan

        return {
            'labels': labels_full,
            'indices': idx,
            'silhouette': sil
        }

    data_bundle = {"n": n_samples, "D": D, "k": k, "embryo_ids": embryo_ids}
    spec = resample.subsample(size=n_to_sample)

    out = resample.run(
        data_bundle,
        spec,
        _hierarchical_scorer,
        n_iters=n_bootstrap,
        seed=random_state,
        store="all",
        max_retries_per_iter=0,
        verbose=verbose,
    )

    bootstrap_results = out.samples if out.samples is not None else []

    if verbose:
        print(f"\nCompleted {out.n_success} successful bootstrap iterations")

    return {
        'embryo_ids': list(embryo_ids),
        'reference_labels': reference_labels,
        'bootstrap_results': bootstrap_results,
        'n_clusters': k,
        'distance_matrix': D,
        'n_samples': n_samples
    }


def run_bootstrap_kmedoids(
    D: np.ndarray,
    k: int,
    embryo_ids: List[str],
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    frac: float = BOOTSTRAP_FRAC,
    random_state: int = RANDOM_SEED,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Bootstrap K-medoids clustering with label alignment.

    Performs repeated K-medoids clustering on random subsamples of the data,
    storing the resulting cluster labels for posterior probability computation.
    K-medoids is more robust to outliers than hierarchical clustering.

    Parameters
    ----------
    D : np.ndarray
        Distance matrix (n × n)
    k : int
        Number of clusters
    embryo_ids : list of str
        Embryo identifiers (required for tracking). Should encode experiment/run info.
        Example: ['cep290_wt_run1_emb_01', 'cep290_wt_run1_emb_02', ...]
    n_bootstrap : int, default=100
        Number of bootstrap iterations
    frac : float, default=0.8
        Fraction of samples per bootstrap
    random_state : int, default=42
        Random seed for reproducibility
    verbose : bool, default=False
        Print progress

    Returns
    -------
    bootstrap_results_dict : dict
        - 'embryo_ids': list of str (copy of input)
        - 'reference_labels': np.ndarray, consensus labels from full data
        - 'bootstrap_results': list of dicts
            - 'labels': np.ndarray (-1 for unsampled)
            - 'indices': np.ndarray of sampled indices
            - 'silhouette': float
        - 'n_clusters': int
        - 'medoid_indices': np.ndarray, medoid indices from reference clustering
        - 'distance_matrix': np.ndarray
        - 'n_samples': int

    Raises
    ------
    ImportError
        If sklearn_extra is not installed

    Examples
    --------
    >>> D = compute_dtw_distance_matrix(trajectories)
    >>> embryo_ids = ['emb_01', 'emb_02', 'emb_03', ...]
    >>> results = run_bootstrap_kmedoids(D, k=3, embryo_ids=embryo_ids, n_bootstrap=100)
    >>> reference_labels = results['reference_labels']
    >>> medoid_indices = results['medoid_indices']  # Representative embryos
    >>> # Lookup embryo by ID
    >>> idx = results['embryo_ids'].index('emb_02')
    >>> label = results['reference_labels'][idx]

    Notes
    -----
    - K-medoids uses actual data points as cluster centers (medoids)
    - More robust to outliers than hierarchical clustering
    - Output contract matches run_bootstrap_hierarchical() for compatibility
    - Requires sklearn-extra package: pip install scikit-learn-extra
    """
    if not KMEDOIDS_AVAILABLE:
        raise ImportError(
            "K-medoids clustering requires scikit-learn-extra. "
            "Install with: pip install scikit-learn-extra"
        )

    n_samples = len(D)

    # Compute reference labels from full data
    clusterer = KMedoids(
        n_clusters=k,
        metric='precomputed',
        random_state=random_state,
        init='k-medoids++',
        max_iter=300
    )
    reference_labels = clusterer.fit_predict(D)
    reference_medoid_indices = clusterer.medoid_indices_

    # Subsample size — preserve original ceil semantics
    n_to_sample = max(int(np.ceil(frac * n_samples)), 1)

    if verbose:
        print(f"Running {n_bootstrap} bootstrap iterations (K-medoids)...")
        print(f"  Sampling {n_to_sample}/{n_samples} samples per iteration")

    # Scorer for the resampling framework
    def _kmedoids_scorer(data, rng):
        n = data["n"]
        k_val = data["k"]
        idx = np.sort(data["indices"]) if "indices" in data else np.arange(n)
        D_sub = data["D"][np.ix_(idx, idx)]

        # Derive a deterministic per-iteration seed from the rng
        iter_seed = int(rng.integers(0, 2**31))

        clusterer_boot = KMedoids(
            n_clusters=k_val,
            metric='precomputed',
            random_state=iter_seed,
            init='k-medoids++',
            max_iter=300
        )
        labels_subset = clusterer_boot.fit_predict(D_sub)

        labels_full = np.full(n, -1, dtype=int)
        labels_full[idx] = labels_subset

        try:
            sil = float(silhouette_score(D_sub, labels_subset, metric='precomputed'))
        except Exception:
            sil = np.nan

        return {
            'labels': labels_full,
            'indices': idx,
            'silhouette': sil
        }

    data_bundle = {"n": n_samples, "D": D, "k": k, "embryo_ids": embryo_ids}
    spec = resample.subsample(size=n_to_sample)

    out = resample.run(
        data_bundle,
        spec,
        _kmedoids_scorer,
        n_iters=n_bootstrap,
        seed=random_state,
        store="all",
        max_retries_per_iter=0,
        verbose=verbose,
    )

    bootstrap_results = out.samples if out.samples is not None else []

    if verbose:
        print(f"\nCompleted {out.n_success} successful bootstrap iterations")

    return {
        'embryo_ids': list(embryo_ids),
        'reference_labels': reference_labels,
        'bootstrap_results': bootstrap_results,
        'n_clusters': k,
        'medoid_indices': reference_medoid_indices,
        'distance_matrix': D,
        'n_samples': n_samples
    }


def compute_consensus_labels(
    bootstrap_results: Dict[str, Any]
) -> np.ndarray:
    """
    Compute consensus labels from bootstrap results.

    Uses majority voting (argmax of assignment frequency matrix) to determine
    the most frequently assigned cluster for each sample across all bootstrap
    iterations.

    Parameters
    ----------
    bootstrap_results : dict
        Output from run_bootstrap_hierarchical()

    Returns
    -------
    consensus_labels : np.ndarray
        Consensus cluster assignments
    """
    n_samples = bootstrap_results['n_samples']
    bootstrap_iter_results = bootstrap_results['bootstrap_results']

    # Count cluster assignments
    assignment_matrix = np.full((n_samples, bootstrap_results['n_clusters']), 0, dtype=int)

    for boot_result in bootstrap_iter_results:
        labels = boot_result['labels']
        for i in range(n_samples):
            if labels[i] >= 0:
                assignment_matrix[i, labels[i]] += 1

    # Compute consensus: most frequent cluster per sample
    consensus_labels = np.argmax(assignment_matrix, axis=1)

    return consensus_labels


def get_cluster_assignments(
    distance_matrix: np.ndarray,
    embryo_ids: List[str],
    k_values: List[int] = [3, 4, 5, 6],
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    bootstrap_frac: float = BOOTSTRAP_FRAC,
    random_seed: int = RANDOM_SEED,
    verbose: bool = False
) -> tuple:
    """
    Run clustering for multiple k values and return consolidated assignments.

    Wraps run_bootstrap_hierarchical() + analyze_bootstrap_results() for multiple k.
    This is a convenience function for testing multiple cluster counts.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Pairwise distance matrix (n x n)
    embryo_ids : list of str
        List of embryo identifiers
    k_values : list of int, default=[3, 4, 5, 6]
        K values to test
    n_bootstrap : int, default=100
        Number of bootstrap iterations
    bootstrap_frac : float, default=0.8
        Fraction to sample per iteration
    random_seed : int, default=42
        Random seed for reproducibility
    verbose : bool, default=False
        Print progress

    Returns
    -------
    df_assignments : DataFrame
        Columns: embryo_id, cluster_k3, cluster_k4, ..., max_p_k3, max_p_k4, ...
    all_results : dict
        Full results keyed by k: {3: {'bootstrap_results': ..., 'posteriors': ...}, ...}

    Examples
    --------
    >>> D = compute_dtw_distance_matrix(trajectories)
    >>> df_assignments, all_results = get_cluster_assignments(D, embryo_ids, k_values=[3,4,5])
    >>> # Access cluster assignments for k=3
    >>> cluster_3_labels = df_assignments['cluster_k3']
    >>> # Access full posterior analysis for k=3
    >>> posteriors_k3 = all_results[3]['posteriors']
    """
    import pandas as pd
    from .cluster_posteriors import analyze_bootstrap_results

    all_results = {}
    assignment_dfs = []

    for k in k_values:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running clustering for k={k}")
            print(f"{'='*60}")

        # Run bootstrap clustering
        bootstrap_results = run_bootstrap_hierarchical(
            distance_matrix, k, embryo_ids,
            n_bootstrap=n_bootstrap,
            frac=bootstrap_frac,
            random_state=random_seed,
            verbose=verbose
        )

        # Compute posteriors
        posteriors = analyze_bootstrap_results(bootstrap_results)

        # Store full results
        all_results[k] = {
            'bootstrap_results': bootstrap_results,
            'posteriors': posteriors
        }

        # Extract assignments for this k
        df_k = pd.DataFrame({
            'embryo_id': posteriors['embryo_ids'],
            f'cluster_k{k}': posteriors['modal_cluster'],
            f'max_p_k{k}': posteriors['max_p'],
            f'entropy_k{k}': posteriors['entropy']
        })
        assignment_dfs.append(df_k)

    # Merge all k values
    df_assignments = assignment_dfs[0]
    for df_k in assignment_dfs[1:]:
        df_assignments = df_assignments.merge(df_k, on='embryo_id')

    if verbose:
        print(f"\n{'='*60}")
        print(f"Clustering complete for k={k_values}")
        print(f"{'='*60}")

    return df_assignments, all_results


def compute_coassociation_matrix(
    bootstrap_results_dict: Dict[str, Any],
    *,
    verbose: bool = True
) -> np.ndarray:
    """
    Evidence Accumulation: Compute co-association matrix from bootstrap results.

    M[i,j] = fraction of bootstrap iterations where embryos i and j
             were assigned to the same cluster

    Uses RAW bootstrap labels (no Hungarian alignment needed) - this is simpler
    and alignment-agnostic. The co-clustering frequency is invariant to cluster
    numbering.

    This matrix can be used to build a consensus dendrogram where merge heights
    reflect clustering stability across bootstrap iterations.

    Parameters
    ----------
    bootstrap_results_dict : Dict[str, Any]
        Output from run_bootstrap_hierarchical()
        Must contain 'embryo_ids' and 'bootstrap_results'
    verbose : bool, default=True
        Print diagnostic information

    Returns
    -------
    M : np.ndarray
        Co-association matrix (n_embryos × n_embryos)
        - M[i,j] ∈ [0, 1]: fraction of co-clustering
        - Symmetric: M[i,j] = M[j,i]
        - Diagonal = 1: M[i,i] = 1.0
        - M[i,j] = 0.5 if i and j were never co-sampled (neutral)

    Examples
    --------
    >>> # After bootstrap clustering
    >>> bootstrap_results = run_bootstrap_hierarchical(D, k=3, embryo_ids=ids)
    >>> M = compute_coassociation_matrix(bootstrap_results)
    >>>
    >>> # High co-association means stable clustering
    >>> print(f"Mean co-association: {M[np.triu_indices(len(M), k=1)].mean():.3f}")
    >>>
    >>> # Build consensus dendrogram
    >>> D_consensus = coassociation_to_distance(M)
    >>> fig, info = generate_dendrograms(D, ids, coassociation_matrix=M)

    Notes
    -----
    - Uses raw bootstrap labels (no Hungarian alignment)
    - Handles unsampled embryos (labels = -1) correctly
    - Co-association = co-cluster count / co-sample count
    - If i and j never co-sampled, M[i,j] = 0.5 (neutral prior)
    - Consensus distance = 1 - M (use coassociation_to_distance())

    References
    ----------
    Fred, A.L.N., & Jain, A.K. (2005). Combining multiple clusterings using
    evidence accumulation. IEEE TPAMI, 27(6), 835-850.
    """
    n_embryos = len(bootstrap_results_dict['embryo_ids'])
    bootstrap_results = bootstrap_results_dict['bootstrap_results']

    # Count co-clustering and co-sampling
    coassoc_count = np.zeros((n_embryos, n_embryos), dtype=int)
    cosample_count = np.zeros((n_embryos, n_embryos), dtype=int)

    for boot_result in bootstrap_results:
        labels = boot_result['labels']  # RAW labels (no alignment!)

        for i in range(n_embryos):
            for j in range(i, n_embryos):
                # Both sampled? (labels >= 0)
                if labels[i] >= 0 and labels[j] >= 0:
                    cosample_count[i, j] += 1
                    cosample_count[j, i] += 1

                    # Same cluster?
                    if labels[i] == labels[j]:
                        coassoc_count[i, j] += 1
                        coassoc_count[j, i] += 1

    # Compute co-association frequency
    M = np.zeros((n_embryos, n_embryos), dtype=float)
    for i in range(n_embryos):
        for j in range(n_embryos):
            if cosample_count[i, j] > 0:
                M[i, j] = coassoc_count[i, j] / cosample_count[i, j]
            else:
                # Never co-sampled: use neutral prior (0.5)
                M[i, j] = 0.5 if i != j else 1.0

    # Ensure diagonal = 1.0
    np.fill_diagonal(M, 1.0)

    if verbose:
        # Compute statistics on upper triangle (exclude diagonal)
        upper_tri_indices = np.triu_indices(n_embryos, k=1)
        upper_tri_values = M[upper_tri_indices]

        print(f"\nCo-association matrix computed (Evidence Accumulation):")
        print(f"  Size: {n_embryos} × {n_embryos}")
        print(f"  Bootstrap iterations: {len(bootstrap_results)}")
        print(f"  Mean co-association: {upper_tri_values.mean():.3f}")
        print(f"  Std co-association: {upper_tri_values.std():.3f}")
        print(f"  Range: [{upper_tri_values.min():.3f}, {upper_tri_values.max():.3f}]")

    return M


def coassociation_to_distance(M: np.ndarray) -> np.ndarray:
    """
    Convert co-association matrix to distance matrix for consensus dendrogram.

    Consensus distance = 1 - co-association frequency

    This transformation allows using the co-association matrix with hierarchical
    clustering algorithms that expect distance matrices.

    Interpretation of consensus distances:
    - D[i,j] = 0.0: always clustered together (100% co-clustering)
    - D[i,j] = 0.5: neutral (50% co-clustering, or never co-sampled)
    - D[i,j] = 1.0: never clustered together (0% co-clustering)

    Parameters
    ----------
    M : np.ndarray
        Co-association matrix (n × n), values in [0, 1]
        Output from compute_coassociation_matrix()

    Returns
    -------
    D : np.ndarray
        Consensus distance matrix (n × n)
        - D[i,j] = 1 - M[i,j]
        - Symmetric, zero diagonal
        - Values in [0, 1]

    Examples
    --------
    >>> M = compute_coassociation_matrix(bootstrap_results)
    >>> D_consensus = coassociation_to_distance(M)
    >>> fig, info = generate_dendrograms(D, ids, coassociation_matrix=M)

    Notes
    -----
    - This is a simple transformation: D = 1 - M
    - The resulting distance matrix is suitable for hierarchical clustering
    - Merge heights in dendrogram = 1 - co-clustering frequency
    """
    D = 1.0 - M
    np.fill_diagonal(D, 0.0)
    return D


def run_bootstrap_projection(
    source_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    reference_cluster_map: Dict[str, int],
    reference_category_map: Optional[Dict[str, str]] = None,
    metrics: List[str] = None,
    time_col: str = 'predicted_stage_hpf',
    embryo_id_col: str = 'embryo_id',
    sakoe_chiba_radius: int = 20,
    normalize: bool = True,
    n_bootstrap: int = N_BOOTSTRAP,
    frac: float = BOOTSTRAP_FRAC,
    random_state: int = RANDOM_SEED,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Bootstrap resampling for cluster projection uncertainty quantification.

    Follows the same pattern as run_bootstrap_hierarchical() but for projection
    instead of clustering. Subsamples source embryos and projects onto reference.

    Parameters
    ----------
    source_df : pd.DataFrame
        Source trajectories to project
    reference_df : pd.DataFrame
        Reference trajectories with known clusters
    reference_cluster_map : Dict[str, int]
        embryo_id -> cluster assignments for reference
    reference_category_map : Optional[Dict[str, str]]
        embryo_id -> category name mapping (not used in bootstrap, but passed through)
    metrics : List[str]
        Trajectory metrics to use
    time_col : str, default='predicted_stage_hpf'
        Time column name
    embryo_id_col : str, default='embryo_id'
        Embryo ID column name
    sakoe_chiba_radius : int, default=20
        DTW warping constraint
    normalize : bool, default=True
        Whether to Z-score normalize
    n_bootstrap : int, default=100
        Number of bootstrap iterations
    frac : float, default=0.8
        Fraction of source embryos to subsample per iteration
    random_state : int, default=42
        Random seed for reproducibility
    verbose : bool, default=False
        Print progress

    Returns
    -------
    bootstrap_results_dict : dict
        - 'embryo_ids': list of str (all source embryo IDs)
        - 'reference_labels': np.ndarray, original projection cluster labels
        - 'bootstrap_results': list of dicts
            - 'labels': np.ndarray (-1 for unsampled embryos)
            - 'indices': np.ndarray of sampled embryo indices
        - 'n_clusters': int (from reference)
        - 'n_samples': int (number of source embryos)

    Examples
    --------
    >>> bootstrap_results = run_bootstrap_projection(
    ...     source_df=df_20260122,
    ...     reference_df=df_cep290,
    ...     reference_cluster_map=cluster_map,
    ...     metrics=['baseline_deviation_normalized'],
    ...     n_bootstrap=100
    ... )
    >>> # Analyze with existing posterior infrastructure
    >>> from .cluster_posteriors import analyze_bootstrap_results
    >>> posteriors = analyze_bootstrap_results(bootstrap_results)
    >>> print(posteriors['max_p'])  # Confidence per embryo
    """
    from ..projection import project_onto_reference_clusters

    if metrics is None:
        metrics = ['baseline_deviation_normalized']

    # Get baseline projection (reference labels)
    if verbose:
        print("Computing reference projection (full dataset)...")

    full_projection, _ = project_onto_reference_clusters(
        source_df=source_df,
        reference_df=reference_df,
        reference_cluster_map=reference_cluster_map,
        reference_category_map=reference_category_map,
        metrics=metrics,
        time_col=time_col,
        embryo_id_col=embryo_id_col,
        sakoe_chiba_radius=sakoe_chiba_radius,
        normalize=normalize,
        verbose=False
    )

    # Extract embryo IDs and labels
    source_embryo_ids = full_projection[embryo_id_col].tolist()
    reference_labels = full_projection['cluster'].values
    n_clusters = len(set(reference_cluster_map.values()))
    n_samples = len(source_embryo_ids)
    n_to_sample = max(int(np.ceil(frac * n_samples)), 1)

    if verbose:
        print(f"\nBootstrap projection setup:")
        print(f"  Source embryos: {n_samples}")
        print(f"  Reference clusters: {n_clusters}")
        print(f"  Bootstrap iterations: {n_bootstrap}")
        print(f"  Subsample size: {n_to_sample} ({frac*100:.0f}%)")

    # Scorer for the resampling framework
    def _projection_scorer(data, rng):
        n = data["n"]
        src_ids = data["source_embryo_ids"]
        idx = np.sort(data["indices"]) if "indices" in data else np.arange(n)
        sampled_ids = [src_ids[i] for i in idx]

        source_subset = data["source_df"][
            data["source_df"][data["embryo_id_col"]].isin(sampled_ids)
        ].copy()

        projection_subset, _ = project_onto_reference_clusters(
            source_df=source_subset,
            reference_df=data["reference_df"],
            reference_cluster_map=data["reference_cluster_map"],
            reference_category_map=data["reference_category_map"],
            metrics=data["metrics"],
            time_col=data["time_col"],
            embryo_id_col=data["embryo_id_col"],
            sakoe_chiba_radius=data["sakoe_chiba_radius"],
            normalize=data["normalize"],
            verbose=False
        )

        labels_full = np.full(n, -1, dtype=int)
        for eid, cluster in zip(
            projection_subset[data["embryo_id_col"]],
            projection_subset['cluster']
        ):
            labels_full[src_ids.index(eid)] = int(cluster)

        return {
            'labels': labels_full,
            'indices': idx
        }

    data_bundle = {
        "n": n_samples,
        "source_df": source_df,
        "reference_df": reference_df,
        "reference_cluster_map": reference_cluster_map,
        "reference_category_map": reference_category_map,
        "metrics": metrics,
        "time_col": time_col,
        "embryo_id_col": embryo_id_col,
        "sakoe_chiba_radius": sakoe_chiba_radius,
        "normalize": normalize,
        "source_embryo_ids": source_embryo_ids,
    }
    spec = resample.subsample(size=n_to_sample)

    if verbose:
        print(f"\nRunning bootstrap iterations...")

    out = resample.run(
        data_bundle,
        spec,
        _projection_scorer,
        n_iters=n_bootstrap,
        seed=random_state,
        store="all",
        max_retries_per_iter=1,
        verbose=verbose,
    )

    bootstrap_results = out.samples if out.samples is not None else []

    if verbose:
        print(f"\nCompleted {out.n_success} successful bootstrap iterations")

    return {
        'embryo_ids': source_embryo_ids,
        'reference_labels': reference_labels,
        'bootstrap_results': bootstrap_results,
        'n_clusters': n_clusters,
        'n_samples': n_samples
    }


def _detect_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    matches = [c for c in df.columns if c.lower() in candidates]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous columns found: {matches}. Please specify explicitly.")
    return None


def _resolve_time_col(df: pd.DataFrame, time_col: Optional[str]) -> str:
    if time_col is not None:
        return time_col
    for candidate in ['predicted_stage_hpf', 'time', 'hpf']:
        if candidate in df.columns:
            return candidate
    raise ValueError("No time column found. Please set time_col.")


def _resolve_cluster_columns(
    reference_df: pd.DataFrame,
    labels_df: Optional[pd.DataFrame],
    id_col: str,
    cluster_col: Optional[str],
    category_col: Optional[str],
) -> Tuple[pd.DataFrame, str, Optional[str]]:
    ref = reference_df.copy()
    cat_candidates = ['cluster_category', 'cluster_categories', 'category', 'categories']

    # If cluster_col not provided, try reference_df first
    if cluster_col is None:
        cluster_col = _detect_column(ref, ['cluster', 'clusters', 'label', 'labels'])

    # If still missing, try labels_df
    if cluster_col is None and labels_df is not None:
        cluster_col = _detect_column(labels_df, ['cluster', 'clusters', 'label', 'labels'])

    if cluster_col is None:
        raise ValueError("No cluster column found. Provide labels_df or cluster_col.")

    # Resolve category column if not provided (prefer reference_df)
    if category_col is None:
        category_col = _detect_column(ref, cat_candidates)
        if category_col is None and labels_df is not None:
            category_col = _detect_column(labels_df, cat_candidates)

    # Merge labels_df if needed for missing columns
    if labels_df is not None:
        label_cols = [id_col]
        merge_needed = False
        if cluster_col not in ref.columns and cluster_col in labels_df.columns:
            label_cols.append(cluster_col)
            merge_needed = True
        if category_col and category_col not in ref.columns and category_col in labels_df.columns:
            label_cols.append(category_col)
            merge_needed = True
        if merge_needed:
            labels_unique = labels_df[label_cols].drop_duplicates(subset=[id_col])
            ref = ref.merge(labels_unique, on=id_col, how='left')

    if cluster_col not in ref.columns:
        raise ValueError("Cluster column not found in reference_df after merge.")

    return ref, cluster_col, category_col


def _make_label_codes(labels: pd.Series) -> Tuple[Dict[Any, int], Dict[int, Any], List[Any]]:
    """Map arbitrary cluster labels to contiguous integer codes."""
    unique_labels = list(pd.unique(labels.dropna()))
    # If labels are numeric-like, sort by numeric value for stability
    numeric_labels = []
    for label in unique_labels:
        try:
            numeric_labels.append(float(label))
        except (TypeError, ValueError):
            numeric_labels = None
            break
    if numeric_labels is not None:
        unique_labels = [x for _, x in sorted(zip(numeric_labels, unique_labels), key=lambda t: t[0])]
    label_to_code = {label: idx for idx, label in enumerate(unique_labels)}
    code_to_label = {idx: label for label, idx in label_to_code.items()}
    return label_to_code, code_to_label, unique_labels


def _label_to_key(label: Any) -> str:
    """Convert label to a safe column suffix."""
    label_str = str(label).strip().lower()
    cleaned = []
    prev_underscore = False
    for ch in label_str:
        if ch.isalnum():
            cleaned.append(ch)
            prev_underscore = False
        else:
            if not prev_underscore:
                cleaned.append('_')
                prev_underscore = True
    key = ''.join(cleaned).strip('_')
    return key or "label"


def bootstrap_projection_assignments_from_distance(
    D_cross: np.ndarray,
    source_ids: List[str],
    ref_ids: List[str],
    reference_cluster_map: Dict[str, int],
    *,
    reference_category_map: Optional[Dict[str, str]] = None,
    method: str = 'nearest_neighbor',
    k: int = 5,
    n_bootstrap: int = N_BOOTSTRAP,
    frac: float = BOOTSTRAP_FRAC,
    bootstrap_on: str = "reference",
    random_state: int = RANDOM_SEED,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Bootstrap projection using a precomputed cross-DTW distance matrix.

    bootstrap_on:
      - "reference": subsample reference embryos (columns of D_cross)
      - "source": subsample source embryos (rows of D_cross)
    """
    from ..projection import project_onto_reference_clusters_from_distance

    n_samples = len(source_ids)
    n_ref = len(ref_ids)
    n_clusters = len(set(reference_cluster_map.values()))
    if bootstrap_on not in {"reference", "source"}:
        raise ValueError("bootstrap_on must be 'reference' or 'source'")
    pop_size = n_ref if bootstrap_on == "reference" else n_samples
    n_to_sample = max(int(np.ceil(frac * pop_size)), 1)

    # Full projection (reference labels)
    full_assignments = project_onto_reference_clusters_from_distance(
        D_cross=D_cross,
        source_ids=source_ids,
        ref_ids=ref_ids,
        reference_cluster_map=reference_cluster_map,
        reference_category_map=reference_category_map,
        method=method,
        k=k,
        verbose=False
    )
    reference_labels = full_assignments['cluster'].values

    id_to_idx = {embryo_id: i for i, embryo_id in enumerate(source_ids)}

    if verbose:
        print(f"\nRunning bootstrap iterations...")

    if bootstrap_on == "source":
        def _source_scorer(data, rng):
            n = data["n_samples"]
            idx = np.sort(data["indices"]) if "indices" in data else np.arange(data["n"])
            sampled_ids = [data["source_ids"][i] for i in idx]
            D_sub = data["D_cross"][idx, :]

            subset_assignments = project_onto_reference_clusters_from_distance(
                D_cross=D_sub,
                source_ids=sampled_ids,
                ref_ids=data["ref_ids"],
                reference_cluster_map=data["reference_cluster_map"],
                reference_category_map=data["reference_category_map"],
                method=data["method"],
                k=data["k_proj"],
                verbose=False
            )

            labels_full = np.full(n, -1, dtype=int)
            for eid, cluster in zip(
                subset_assignments['embryo_id'],
                subset_assignments['cluster']
            ):
                labels_full[data["id_to_idx"][eid]] = int(cluster)

            return {
                'labels': labels_full,
                'indices': idx
            }

        data_bundle = {
            "n": n_samples,
            "D_cross": D_cross,
            "source_ids": source_ids,
            "ref_ids": ref_ids,
            "reference_cluster_map": reference_cluster_map,
            "reference_category_map": reference_category_map,
            "method": method,
            "k_proj": k,
            "id_to_idx": id_to_idx,
            "n_samples": n_samples,
        }
        spec = resample.subsample(size=n_to_sample)

        out = resample.run(
            data_bundle,
            spec,
            _source_scorer,
            n_iters=n_bootstrap,
            seed=random_state,
            store="all",
            max_retries_per_iter=1,
            verbose=verbose,
        )

    else:  # bootstrap_on == "reference"
        def _reference_scorer(data, rng):
            n = data["n_samples"]
            idx = np.sort(data["indices"]) if "indices" in data else np.arange(data["n"])
            sampled_ref_ids = [data["ref_ids"][i] for i in idx]

            ref_cluster_sub = {
                rid: data["reference_cluster_map"][rid]
                for rid in sampled_ref_ids
                if rid in data["reference_cluster_map"]
            }
            if len(ref_cluster_sub) == 0:
                raise ValueError("No reference clusters after subsample")

            ref_category_sub = None
            if data["reference_category_map"] is not None:
                ref_category_sub = {
                    rid: data["reference_category_map"][rid]
                    for rid in sampled_ref_ids
                    if rid in data["reference_category_map"]
                }

            D_sub = data["D_cross"][:, idx]

            subset_assignments = project_onto_reference_clusters_from_distance(
                D_cross=D_sub,
                source_ids=data["source_ids"],
                ref_ids=sampled_ref_ids,
                reference_cluster_map=ref_cluster_sub,
                reference_category_map=ref_category_sub,
                method=data["method"],
                k=data["k_proj"],
                verbose=False
            )

            labels_full = subset_assignments['cluster'].astype(int).values

            return {
                'labels': labels_full,
                'indices': np.arange(n)
            }

        data_bundle = {
            "n": n_ref,
            "D_cross": D_cross,
            "source_ids": source_ids,
            "ref_ids": ref_ids,
            "reference_cluster_map": reference_cluster_map,
            "reference_category_map": reference_category_map,
            "method": method,
            "k_proj": k,
            "n_samples": n_samples,
        }
        spec = resample.subsample(size=n_to_sample)

        out = resample.run(
            data_bundle,
            spec,
            _reference_scorer,
            n_iters=n_bootstrap,
            seed=random_state,
            store="all",
            max_retries_per_iter=1,
            verbose=verbose,
        )

    bootstrap_results = out.samples if out.samples is not None else []

    if verbose:
        print(f"\nCompleted {out.n_success} successful bootstrap iterations")

    return {
        'embryo_ids': source_ids,
        'reference_labels': reference_labels,
        'bootstrap_results': bootstrap_results,
        'n_clusters': n_clusters,
        'n_samples': n_samples
    }


def run_bootstrap_projection_with_plots(
    source_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    output_dir: Path,
    run_name: str,
    labels_df: Optional[pd.DataFrame] = None,
    *,
    id_col: str = "embryo_id",
    time_col: Optional[str] = "predicted_stage_hpf",
    cluster_col: Optional[str] = None,
    category_col: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    plotting_metrics: Optional[List[str]] = None,
    sakoe_chiba_radius: int = 20,
    n_bootstrap: int = N_BOOTSTRAP,
    frac: float = BOOTSTRAP_FRAC,
    bootstrap_on: str = "reference",
    method: str = "nearest_neighbor",
    k: int = 5,
    classification: str = "2d",
    normalize: bool = True,
    verbose: bool = True,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """
    High-level wrapper for bootstrapped projection with plots.

    Computes cross-DTW once, bootstraps assignments using matrix slicing,
    classifies membership, and saves plots/CSVs in an orderly format.
    """
    from ..projection import (
        prepare_projection_arrays,
        compute_cross_dtw_distance_matrix,
        project_onto_reference_clusters_from_distance,
    )
    from .cluster_posteriors import analyze_bootstrap_results
    from .cluster_classification import (
        classify_membership_2d,
        classify_membership_adaptive,
        get_classification_summary,
    )
    from analyze.viz.plotting import plot_feature_over_time
    from analyze.viz.plotting import plot_proportions

    output_dir = Path(output_dir)
    if save_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)

    time_col = _resolve_time_col(source_df, time_col)
    ref_labeled, cluster_col, category_col = _resolve_cluster_columns(
        reference_df=reference_df,
        labels_df=labels_df,
        id_col=id_col,
        cluster_col=cluster_col,
        category_col=category_col,
    )

    if id_col not in source_df.columns:
        raise ValueError(f"Source DataFrame missing id_col: '{id_col}'")
    if id_col not in ref_labeled.columns:
        raise ValueError(f"Reference DataFrame missing id_col: '{id_col}'")
    if time_col not in source_df.columns:
        raise ValueError(f"Source DataFrame missing time_col: '{time_col}'")
    if time_col not in ref_labeled.columns:
        raise ValueError(f"Reference DataFrame missing time_col: '{time_col}'")

    if metrics is None:
        for candidate in ['baseline_deviation_normalized', 'total_length_um']:
            if candidate in source_df.columns:
                metrics = [candidate]
                break
        if metrics is None:
            raise ValueError("No default metric found. Please provide metrics.")
    if plotting_metrics is None:
        plotting_metrics = metrics

    if verbose:
        print("="*80)
        print("BOOTSTRAP PROJECTION WITH PLOTS")
        print("="*80)
        print(f"Run name: {run_name}")
        print(f"ID column: {id_col}")
        print(f"Time column: {time_col}")
        print(f"Cluster column: {cluster_col}")
        if category_col:
            print(f"Category column: {category_col}")
        print(f"Metrics: {metrics}")
        print(f"Bootstrap: n={n_bootstrap}, frac={frac}")
        print(f"Method: {method}")

    # Build cluster/category maps (encode labels to contiguous integer codes)
    ref_unique = ref_labeled[[id_col, cluster_col]].drop_duplicates(subset=[id_col])
    ref_unique = ref_unique[ref_unique[cluster_col].notna()]
    label_to_code, code_to_label, label_order = _make_label_codes(ref_unique[cluster_col])
    reference_cluster_map = {
        row[id_col]: label_to_code[row[cluster_col]]
        for _, row in ref_unique.iterrows()
    }

    reference_category_map = None
    if category_col and category_col in ref_labeled.columns:
        ref_cat = ref_labeled[[id_col, category_col]].drop_duplicates(subset=[id_col])
        reference_category_map = dict(zip(ref_cat[id_col], ref_cat[category_col]))

    # Step 1-3: Prepare arrays (no DTW yet)
    arrays = prepare_projection_arrays(
        source_df=source_df,
        reference_df=ref_labeled,
        reference_cluster_map=reference_cluster_map,
        metrics=metrics,
        time_col=time_col,
        embryo_id_col=id_col,
        normalize=normalize,
        verbose=verbose
    )

    # Step 4: Compute cross-DTW once
    if verbose:
        print(f"\nComputing cross-DTW once...")
    D_cross = compute_cross_dtw_distance_matrix(
        arrays.X_source,
        arrays.X_ref,
        sakoe_chiba_radius=sakoe_chiba_radius,
        n_jobs=-1,
        verbose=verbose
    )

    # Precompute min distances to each cluster (for diagnostics)
    cluster_to_ref_indices = {}
    for j, ref_id in enumerate(arrays.ref_ids):
        if ref_id in reference_cluster_map:
            cluster_id = reference_cluster_map[ref_id]
            cluster_to_ref_indices.setdefault(cluster_id, []).append(j)

    cluster_codes_sorted = sorted(cluster_to_ref_indices.keys())
    cluster_label_keys = {
        code: _label_to_key(code_to_label.get(code, code))
        for code in cluster_codes_sorted
    }

    cluster_min_distances = {}
    for code, idxs in cluster_to_ref_indices.items():
        if len(idxs) == 0:
            continue
        cluster_min_distances[code] = np.nanmin(D_cross[:, idxs], axis=1)

    # Full projection (baseline assignments)
    full_assignments = project_onto_reference_clusters_from_distance(
        D_cross=D_cross,
        source_ids=arrays.source_ids,
        ref_ids=arrays.ref_ids,
        reference_cluster_map=reference_cluster_map,
        reference_category_map=reference_category_map,
        method=method,
        k=k,
        verbose=verbose
    )
    full_assignments = full_assignments.rename(columns={'embryo_id': id_col})
    cluster_labels_full = full_assignments['cluster'].values
    cluster_label_strings = [code_to_label.get(int(c), c) for c in cluster_labels_full]

    # Step 5: Bootstrap using D_cross
    bootstrap_results = bootstrap_projection_assignments_from_distance(
        D_cross=D_cross,
        source_ids=arrays.source_ids,
        ref_ids=arrays.ref_ids,
        reference_cluster_map=reference_cluster_map,
        reference_category_map=reference_category_map,
        method=method,
        k=k,
        n_bootstrap=n_bootstrap,
        frac=frac,
        bootstrap_on=bootstrap_on,
        verbose=verbose
    )

    # Step 6: Posterior analysis + classification
    posteriors = analyze_bootstrap_results(bootstrap_results)
    if classification == "adaptive":
        classification_result = classify_membership_adaptive(
            posteriors['max_p'],
            posteriors['log_odds_gap'],
            posteriors['modal_cluster'],
        )
    else:
        classification_result = classify_membership_2d(
            posteriors['max_p'],
            posteriors['log_odds_gap'],
            posteriors['modal_cluster'],
            embryo_ids=arrays.source_ids
        )

    # Ensure embryo_ids and baseline cluster labels are present
    classification_result['embryo_ids'] = arrays.source_ids
    classification_result['cluster'] = cluster_labels_full.copy()

    # Add "unclassified" for never-sampled embryos
    categories = classification_result['category'].copy()
    sample_counts = posteriors['sample_counts']
    categories[sample_counts == 0] = 'unclassified'
    classification_result['category'] = categories

    # Step 7: Assemble assignments DataFrame
    assignments_df = pd.DataFrame({
        id_col: arrays.source_ids,
        'cluster': cluster_labels_full,
        'cluster_label': cluster_label_strings,
        'membership': categories,
        'max_p': posteriors['max_p'],
        'entropy': posteriors['entropy'],
        'log_odds_gap': posteriors['log_odds_gap'],
        'modal_cluster': posteriors['modal_cluster'],
        'second_best_cluster': posteriors['second_best_cluster'],
        'sample_counts': sample_counts,
        'nearest_distance': D_cross.min(axis=1),
    })

    # Add per-cluster min distances and margin diagnostics
    dist_matrix = []
    for code in cluster_codes_sorted:
        dist_matrix.append(cluster_min_distances[code])
        key = cluster_label_keys[code]
        assignments_df[f'dist_to_{key}'] = cluster_min_distances[code]
    if dist_matrix:
        dist_matrix = np.vstack(dist_matrix).T  # shape (n_source, n_clusters)
        order = np.argsort(dist_matrix, axis=1)
        best_idx = order[:, 0]
        second_idx = order[:, 1] if dist_matrix.shape[1] > 1 else order[:, 0]
        best_dist = dist_matrix[np.arange(dist_matrix.shape[0]), best_idx]
        second_dist = dist_matrix[np.arange(dist_matrix.shape[0]), second_idx]
        best_cluster_code = [cluster_codes_sorted[i] for i in best_idx]
        second_cluster_code = [cluster_codes_sorted[i] for i in second_idx]
        best_label = [code_to_label.get(int(c), c) for c in best_cluster_code]
        second_label = [code_to_label.get(int(c), c) for c in second_cluster_code]
        assignments_df['best_cluster_by_distance'] = best_label
        assignments_df['second_best_cluster_by_distance'] = second_label
        assignments_df['distance_margin'] = second_dist - best_dist

    if 'nearest_distance' in full_assignments.columns:
        assignments_df['nearest_distance'] = full_assignments['nearest_distance'].values

    # Include category map if available
    if reference_category_map is not None and 'cluster_category' in full_assignments.columns:
        assignments_df['cluster_category'] = full_assignments['cluster_category'].values

    # Summary table
    summary = get_classification_summary({
        'category': categories,
        'cluster': cluster_labels_full
    })
    summary_rows = []
    for cluster_id, stats in summary['per_cluster'].items():
        mask = cluster_labels_full == cluster_id
        label = code_to_label.get(int(cluster_id), cluster_id)
        n_total = int(np.sum(mask))
        n_unclassified = int(np.sum(categories[mask] == 'unclassified'))
        summary_rows.append({
            'cluster': cluster_id,
            'cluster_label': label,
            'n_total': n_total,
            'n_core': stats['n_core'],
            'n_uncertain': stats['n_uncertain'],
            'n_outlier': stats['n_outlier'],
            'n_unclassified': n_unclassified,
            'core_fraction': stats['n_core'] / n_total if n_total > 0 else 0.0,
            'uncertain_fraction': stats['n_uncertain'] / n_total if n_total > 0 else 0.0,
            'outlier_fraction': stats['n_outlier'] / n_total if n_total > 0 else 0.0,
            'unclassified_fraction': n_unclassified / n_total if n_total > 0 else 0.0,
            'mean_max_p': float(np.mean(posteriors['max_p'][mask])) if n_total > 0 else np.nan,
            'mean_entropy': float(np.mean(posteriors['entropy'][mask])) if n_total > 0 else np.nan,
            'mean_log_odds_gap': float(np.mean(posteriors['log_odds_gap'][mask])) if n_total > 0 else np.nan,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df['run_name'] = run_name
    summary_df['n_bootstrap'] = n_bootstrap
    summary_df['frac'] = frac
    summary_df['bootstrap_on'] = bootstrap_on
    summary_df['sakoe_chiba_radius'] = sakoe_chiba_radius

    # Step 8: Plots
    plot_paths = {}
    prefix = f"{run_name}_" if run_name else ""

    if save_outputs:
        # Plot A: trajectories (reference vs source, colored by membership)
        df_source_plot = source_df[source_df[id_col].isin(arrays.source_ids)].copy()
        df_source_plot['cluster'] = df_source_plot[id_col].map(
            dict(zip(arrays.source_ids, cluster_labels_full))
        )
        df_source_plot['cluster_label'] = df_source_plot[id_col].map(
            dict(zip(arrays.source_ids, cluster_label_strings))
        )
        df_source_plot['membership'] = df_source_plot[id_col].map(
            dict(zip(arrays.source_ids, categories))
        )
        df_source_plot['dataset'] = 'source'

        df_ref_plot = ref_labeled[ref_labeled[id_col].isin(arrays.ref_ids)].copy()
        df_ref_plot['cluster'] = df_ref_plot[id_col].map(reference_cluster_map)
        df_ref_plot['cluster_label'] = df_ref_plot[id_col].map(
            {k: code_to_label.get(v, v) for k, v in reference_cluster_map.items()}
        )
        df_ref_plot['membership'] = 'reference'
        df_ref_plot['dataset'] = 'reference'

        df_plot = pd.concat([df_ref_plot, df_source_plot], ignore_index=True)

        fig = plot_feature_over_time(
            df_plot,
            features=plotting_metrics,
            time_col=time_col,
            id_col=id_col,
            color_by='membership',
            facet_row='dataset',
            facet_col='cluster_label',
            backend='matplotlib',
            bin_width=2.0,
            title=f"{run_name}: Membership by Cluster (Reference vs Source)"
        )
        traj_path = output_dir / f"{prefix}projection_membership_trajectories.png"
        plt.savefig(traj_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        plot_paths['membership_trajectories'] = str(traj_path)

        # Plot A2: trajectories faceted by membership (rows) and cluster (cols)
        fig_mem = plot_feature_over_time(
            df_source_plot,
            features=plotting_metrics,
            time_col=time_col,
            id_col=id_col,
            color_by='membership',
            facet_row='membership',
            facet_col='cluster_label',
            backend='matplotlib',
            bin_width=2.0,
            title=f"{run_name}: Trajectories by Membership and Cluster"
        )
        mem_path = output_dir / f"{prefix}projection_membership_by_cluster.png"
        plt.savefig(mem_path, dpi=150, bbox_inches='tight')
        plt.close(fig_mem)
        plot_paths['membership_by_cluster'] = str(mem_path)

        # Plot B: proportions (source only)
        df_prop = df_source_plot.copy()
        prop_path = output_dir / f"{prefix}projection_cluster_proportions.png"
        fig_prop = plot_proportions(
            df=df_prop,
            color_by_grouping='membership',
            col_by='cluster_label',
            count_by=id_col,
            normalize=True,
            bar_mode='grouped',
            output_path=prop_path,
            title=f"{run_name}: Membership Proportions by Cluster"
        )
        plt.close(fig_prop)
        plot_paths['membership_proportions'] = str(prop_path)

        # Plot C: confidence scatter
        color_map = {
            'core': '#2ca02c',
            'uncertain': '#ff7f0e',
            'outlier': '#d62728',
            'unclassified': '#7f7f7f',
        }
        fig_scatter, ax = plt.subplots(figsize=(6, 4.5))
        for label in ['core', 'uncertain', 'outlier', 'unclassified']:
            mask = categories == label
            if np.any(mask):
                ax.scatter(
                    posteriors['max_p'][mask],
                    posteriors['log_odds_gap'][mask],
                    s=18,
                    alpha=0.7,
                    label=label,
                    color=color_map.get(label, '#333333')
                )
        ax.set_xlabel('max_p')
        ax.set_ylabel('log_odds_gap')
        ax.set_title(f"{run_name}: Confidence Scatter")
        ax.legend(frameon=False, fontsize=8)
        scatter_path = output_dir / f"{prefix}projection_confidence_scatter.png"
        fig_scatter.tight_layout()
        fig_scatter.savefig(scatter_path, dpi=150, bbox_inches='tight')
        plt.close(fig_scatter)
        plot_paths['confidence_scatter'] = str(scatter_path)

        # Save CSVs + pickle
        assignments_csv = output_dir / f"{prefix}projection_assignments.csv"
        assignments_df.to_csv(assignments_csv, index=False)

        summary_csv = output_dir / f"{prefix}projection_summary.csv"
        summary_df.to_csv(summary_csv, index=False)

        pkl_path = output_dir / f"{prefix}projection_results.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump({
                'assignments_df': assignments_df,
                'summary_df': summary_df,
                'bootstrap_results': bootstrap_results,
                'posteriors': posteriors,
                'classification': classification_result,
                'cluster_label_order': label_order,
                'cluster_label_map': code_to_label,
                'cluster_label_key_map': {
                    code_to_label.get(code, code): cluster_label_keys[code]
                    for code in cluster_codes_sorted
                },
                'plot_paths': plot_paths,
                'params': {
                    'run_name': run_name,
                    'n_bootstrap': n_bootstrap,
                    'frac': frac,
                    'sakoe_chiba_radius': sakoe_chiba_radius,
                    'method': method,
                    'k': k,
                    'metrics': metrics,
                },
            }, f)

    return {
        'assignments_df': assignments_df,
        'summary_df': summary_df,
        'bootstrap_results': bootstrap_results,
        'posteriors': posteriors,
        'classification': classification_result,
        'cluster_label_order': label_order,
        'cluster_label_map': code_to_label,
        'cluster_label_key_map': {
            code_to_label.get(code, code): cluster_label_keys[code]
            for code in cluster_codes_sorted
        },
        'plot_paths': plot_paths,
    }
