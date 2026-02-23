"""
Consensus Clustering Pipeline with Two-Stage Outlier Filtering

Provides an end-to-end pipeline for robust trajectory analysis combining:
1. Stage 1 outlier filtering (IQR distance) before clustering
2. Bootstrap consensus clustering
3. Evidence accumulation consensus dendrograms
4. Posterior probability analysis
5. Stage 2 outlier filtering (cluster IQR + posterior) after clustering
6. Final consensus dendrogram
7. Comprehensive filtering log (chain of custody)

This pipeline is designed for publication-ready trajectory analysis with
transparency about outlier removal and cluster stability.

NOTE (2025-12-22): Switched from k-NN to IQR distance filtering for Stage 1.
Empirical testing confirmed k-NN kept problematic embryos that formed stable
clusters together, while IQR distance filtering properly removes global outliers.
k-NN method retained as 'knn' option for backwards compatibility.

Functions
=========
- run_consensus_pipeline : Complete two-stage filtering + consensus clustering
- create_filtering_log : Generate chain-of-custody DataFrame

Created: 2025-12-22
Purpose: Production-ready consensus clustering with robust outlier filtering
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

from ..qc import (
    identify_outliers,
    identify_embryo_outliers_iqr,
    filter_data_and_ids,
    identify_cluster_outliers_combined
)
from .bootstrap_clustering import (
    run_bootstrap_hierarchical,
    compute_coassociation_matrix
)
from .cluster_posteriors import analyze_bootstrap_results
from ..viz import generate_dendrograms
from ..config import (
    N_BOOTSTRAP,
    BOOTSTRAP_FRAC,
    RANDOM_SEED,
    IQR_MULTIPLIER,
    KNN_K,
    POSTERIOR_OUTLIER_THRESHOLD
)


def run_consensus_pipeline(
    D: np.ndarray,
    embryo_ids: List[str],
    k: int,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    bootstrap_frac: float = BOOTSTRAP_FRAC,
    random_state: int = RANDOM_SEED,
    enable_stage1_filtering: bool = True,
    enable_stage2_filtering: bool = True,
    stage1_method: str = 'iqr',
    iqr_multiplier: float = IQR_MULTIPLIER,
    k_neighbors: int = KNN_K,
    posterior_threshold: float = POSTERIOR_OUTLIER_THRESHOLD,
    k_highlight: Optional[List[int]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Complete two-stage outlier filtering + consensus clustering pipeline.

    Pipeline flow:
    1. Stage 1: IQR distance filtering (remove global outliers)
    2. Bootstrap clustering on filtered data
    3. Build initial consensus dendrogram (evidence accumulation)
    4. Posterior analysis (Core/Uncertain/Outlier classification)
    5. Stage 2: Within-cluster IQR + posterior filtering (remove cluster outliers)
    6. Build final consensus dendrogram
    7. Return comprehensive results + filtering log

    This pipeline provides:
    - Robust outlier filtering using global distance statistics (IQR approach)
    - Consensus dendrograms reflecting clustering stability
    - Chain-of-custody log for every embryo (transparency)
    - Backwards compatibility (can disable filtering or use k-NN method)

    Parameters
    ----------
    D : np.ndarray
        Distance matrix (n_embryos × n_embryos)
        Output from compute_dtw_distance_matrix() or compute_md_dtw_distance_matrix()
    embryo_ids : List[str]
        Embryo identifiers (same order as D rows/columns)
    k : int
        Number of clusters
    n_bootstrap : int, default=100
        Number of bootstrap iterations
    bootstrap_frac : float, default=0.8
        Fraction of samples per bootstrap iteration
    random_state : int, default=42
        Random seed for reproducibility
    enable_stage1_filtering : bool, default=True
        Enable Stage 1 outlier filtering before clustering
    enable_stage2_filtering : bool, default=True
        Enable Stage 2 cluster + posterior filtering after clustering
    stage1_method : str, default='iqr'
        Stage 1 filtering method: 'iqr' (distance-based) or 'knn' (k-nearest neighbors)
        'iqr' removes global outliers based on mean distance to all embryos
        'knn' uses local neighborhood distances (legacy, can keep problematic clusters)
    iqr_multiplier : float, default=2
        IQR multiplier for outlier thresholds (both stages, both methods)
    k_neighbors : int, default=5
        Number of nearest neighbors for Stage 1 k-NN filtering (only used if stage1_method='knn')
    posterior_threshold : float, default=0.5
        Minimum max_p for Stage 2 filtering
    k_highlight : List[int], optional
        K values to highlight in dendrograms (e.g., [2, 3, 4])
    verbose : bool, default=True
        Print pipeline progress

    Returns
    -------
    results : Dict[str, Any]
        Comprehensive results dictionary:

        Stage 1 Filtering:
        - 'stage1_filter_results': Dict from identify_embryo_outliers_iqr()
          (empty dict if filtering disabled)
        - 'D_after_stage1': Distance matrix after Stage 1 filtering
        - 'embryo_ids_after_stage1': Embryo IDs after Stage 1 filtering

        Bootstrap Clustering:
        - 'bootstrap_results': Dict from run_bootstrap_hierarchical()
        - 'consensus_matrix': Co-association matrix (n_after_stage1 × n_after_stage1)
        - 'posterior_results': Dict from analyze_bootstrap_results()

        Stage 2 Filtering:
        - 'stage2_filter_results': Dict from identify_cluster_outliers_combined()
          (empty dict if filtering disabled)
        - 'final_D': Distance matrix after all filtering
        - 'final_embryo_ids': Embryo IDs after all filtering
        - 'final_consensus_matrix': Co-association matrix after Stage 2 filtering

        Dendrograms:
        - 'dendrogram_info_initial': Dict from generate_dendrograms() (after Stage 1)
        - 'dendrogram_info_final': Dict from generate_dendrograms() (after Stage 2)

        Filtering Log:
        - 'filtering_log': DataFrame with columns:
          - embryo_id: Embryo identifier
          - stage1_status: 'kept' or 'removed'
          - stage2_status: 'kept', 'removed_iqr', 'removed_posterior', 'removed_both', or '-'
          - final_status: 'kept', 'removed_stage1', or 'removed_stage2'
          - stage1_knn_distance: k-NN distance (if Stage 1 enabled)
          - stage2_within_cluster_distance: Within-cluster distance (if Stage 2 enabled)
          - stage2_max_p: Posterior max_p (if Stage 2 enabled)

    Examples
    --------
    >>> # Simple usage with defaults
    >>> results = run_consensus_pipeline(D, embryo_ids, k=3)
    >>> final_clusters = results['posterior_results']['modal_cluster']
    >>> final_embryo_ids = results['final_embryo_ids']
    >>> filtering_log = results['filtering_log']
    >>>
    >>> # Check filtering log for bias
    >>> print(filtering_log.groupby('final_status').size())
    >>>
    >>> # Custom parameters (more aggressive filtering)
    >>> results = run_consensus_pipeline(
    ...     D, embryo_ids, k=3,
    ...     iqr_multiplier=3.0,          # More aggressive
    ...     posterior_threshold=0.6,      # Higher confidence
    ...     k_neighbors=10                # Larger neighborhood
    ... )
    >>>
    >>> # Disable filtering stages (standard bootstrap clustering)
    >>> results = run_consensus_pipeline(
    ...     D, embryo_ids, k=3,
    ...     enable_stage1_filtering=False,
    ...     enable_stage2_filtering=False
    ... )

    Notes
    -----
    - Stage 1 k-NN filtering protects rare mutant clusters
    - Stage 2 removes within-cluster outliers and low-confidence assignments
    - Consensus dendrograms use evidence accumulation (raw bootstrap labels)
    - Filtering log provides chain of custody for transparency
    - All filtering stages can be disabled for comparison with standard approach
    """
    if verbose:
        print("="*70)
        print("CONSENSUS CLUSTERING PIPELINE WITH TWO-STAGE OUTLIER FILTERING")
        print("="*70)
        print(f"Initial embryos: {len(embryo_ids)}")
        print(f"Clusters (k): {k}")
        print(f"Bootstrap iterations: {n_bootstrap}")
        print(f"Stage 1 filtering: {'Enabled' if enable_stage1_filtering else 'Disabled'}")
        print(f"Stage 2 filtering: {'Enabled' if enable_stage2_filtering else 'Disabled'}")
        print("="*70)

    # =========================================================================
    # STAGE 1: k-NN IQR FILTERING (Before Clustering)
    # =========================================================================

    if enable_stage1_filtering:
        if verbose:
            print("\n" + "="*70)
            method_name = "IQR Distance Filtering" if stage1_method == 'iqr' else "k-NN IQR Filtering"
            print(f"STAGE 1: {method_name} (Global Outliers)")
            print("="*70)

        if stage1_method == 'iqr':
            # IQR distance filtering: Remove embryos far from all others
            outlier_ids, inlier_ids, stage1_info = identify_outliers(
                D, embryo_ids,
                method='iqr',
                threshold=iqr_multiplier,
                verbose=verbose
            )
            
            # Convert to indices
            inlier_indices = stage1_info['inlier_indices']
            outlier_indices = stage1_info['outlier_indices']
            
            stage1_results = {
                'outlier_indices': outlier_indices,
                'outlier_ids': outlier_ids,
                'kept_indices': inlier_indices,
                'kept_ids': inlier_ids,
                'method': 'iqr',
                'threshold': stage1_info['threshold'],
                'median_distances': stage1_info['median_distances']
            }
            
        elif stage1_method == 'knn':
            # k-NN IQR filtering: Legacy method using local neighborhoods
            stage1_results = identify_embryo_outliers_iqr(
                D, embryo_ids,
                iqr_multiplier=iqr_multiplier,
                k_neighbors=k_neighbors,
                verbose=verbose
            )
            stage1_results['method'] = 'knn'
        else:
            raise ValueError(f"Unknown stage1_method: {stage1_method}. Use 'iqr' or 'knn'")

        # Filter data
        D_after_stage1, embryo_ids_after_stage1 = filter_data_and_ids(
            D, embryo_ids, stage1_results['kept_indices']
        )
    else:
        if verbose:
            print("\n[Stage 1 filtering disabled]")

        stage1_results = {}
        D_after_stage1 = D
        embryo_ids_after_stage1 = embryo_ids

    # =========================================================================
    # BOOTSTRAP CLUSTERING + CONSENSUS MATRIX
    # =========================================================================

    if verbose:
        print("\n" + "="*70)
        print("BOOTSTRAP CLUSTERING")
        print("="*70)

    bootstrap_results = run_bootstrap_hierarchical(
        D_after_stage1,
        k=k,
        embryo_ids=embryo_ids_after_stage1,
        n_bootstrap=n_bootstrap,
        frac=bootstrap_frac,
        random_state=random_state,
        verbose=verbose
    )

    # Compute co-association matrix (evidence accumulation)
    consensus_matrix = compute_coassociation_matrix(
        bootstrap_results,
        verbose=verbose
    )

    # =========================================================================
    # POSTERIOR ANALYSIS
    # =========================================================================

    if verbose:
        print("\n" + "="*70)
        print("POSTERIOR PROBABILITY ANALYSIS")
        print("="*70)

    posterior_results = analyze_bootstrap_results(bootstrap_results)

    if verbose:
        max_p_mean = posterior_results['max_p'].mean()
        entropy_mean = posterior_results['entropy'].mean()
        print(f"  Mean max_p: {max_p_mean:.3f}")
        print(f"  Mean entropy: {entropy_mean:.3f} bits")

    # =========================================================================
    # INITIAL CONSENSUS DENDROGRAM
    # =========================================================================

    if verbose:
        print("\n" + "="*70)
        print("INITIAL CONSENSUS DENDROGRAM (After Stage 1)")
        print("="*70)

    fig_initial, dendrogram_info_initial = generate_dendrograms(
        D_after_stage1,
        embryo_ids_after_stage1,
        coassociation_matrix=consensus_matrix,
        k_highlight=k_highlight if k_highlight else [k],
        verbose=verbose
    )
    plt.close(fig_initial)  # Close figure to save memory

    # =========================================================================
    # STAGE 2: CLUSTER IQR + POSTERIOR FILTERING
    # =========================================================================

    if enable_stage2_filtering:
        if verbose:
            print("\n" + "="*70)
            print("STAGE 2: CLUSTER IQR + POSTERIOR FILTERING")
            print("="*70)

        stage2_results = identify_cluster_outliers_combined(
            D_after_stage1,
            cluster_labels=posterior_results['modal_cluster'],
            posterior_results=posterior_results,
            embryo_ids=embryo_ids_after_stage1,
            iqr_multiplier=iqr_multiplier,
            posterior_threshold=posterior_threshold,
            verbose=verbose
        )

        # Filter data
        final_D, final_embryo_ids = filter_data_and_ids(
            D_after_stage1, embryo_ids_after_stage1, stage2_results['kept_indices']
        )

        # Re-run bootstrap and consensus on final data
        if verbose:
            print("\n[Re-running bootstrap clustering on final filtered data]")

        final_bootstrap_results = run_bootstrap_hierarchical(
            final_D,
            k=k,
            embryo_ids=final_embryo_ids,
            n_bootstrap=n_bootstrap,
            frac=bootstrap_frac,
            random_state=random_state,
            verbose=False  # Less verbose for final round
        )

        final_consensus_matrix = compute_coassociation_matrix(
            final_bootstrap_results,
            verbose=False
        )

        # Update posterior results to final version
        final_posterior_results = analyze_bootstrap_results(final_bootstrap_results)

    else:
        if verbose:
            print("\n[Stage 2 filtering disabled]")

        stage2_results = {}
        final_D = D_after_stage1
        final_embryo_ids = embryo_ids_after_stage1
        final_consensus_matrix = consensus_matrix
        final_posterior_results = posterior_results

    # =========================================================================
    # FINAL CONSENSUS DENDROGRAM
    # =========================================================================

    if verbose:
        print("\n" + "="*70)
        print("FINAL CONSENSUS DENDROGRAM (After Stage 2)")
        print("="*70)
        print(f"  Final embryos: {len(final_embryo_ids)}")

    fig_final, dendrogram_info_final = generate_dendrograms(
        final_D,
        final_embryo_ids,
        coassociation_matrix=final_consensus_matrix,
        k_highlight=k_highlight if k_highlight else [k],
        verbose=verbose
    )
    plt.close(fig_final)  # Close figure to save memory

    # =========================================================================
    # FILTERING LOG (Chain of Custody)
    # =========================================================================

    if verbose:
        print("\n" + "="*70)
        print("FILTERING LOG (Chain of Custody)")
        print("="*70)

    filtering_log = create_filtering_log(
        embryo_ids,
        stage1_results,
        stage2_results,
        embryo_ids_after_stage1,
        final_embryo_ids,
        verbose=verbose
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    if verbose:
        print("\n" + "="*70)
        print("PIPELINE SUMMARY")
        print("="*70)
        print(f"  Initial embryos: {len(embryo_ids)}")
        if enable_stage1_filtering and stage1_results:
            print(f"  After Stage 1: {len(embryo_ids_after_stage1)} "
                  f"({len(stage1_results['outlier_ids'])} removed)")
        if enable_stage2_filtering and stage2_results:
            print(f"  After Stage 2: {len(final_embryo_ids)} "
                  f"({len(stage2_results['outlier_ids'])} removed)")
        print(f"  Final embryos: {len(final_embryo_ids)}")
        print(f"  Total removed: {len(embryo_ids) - len(final_embryo_ids)}")
        print("="*70)

    # =========================================================================
    # RETURN COMPREHENSIVE RESULTS
    # =========================================================================

    return {
        # Stage 1 filtering
        'stage1_filter_results': stage1_results,
        'D_after_stage1': D_after_stage1,
        'embryo_ids_after_stage1': embryo_ids_after_stage1,

        # Bootstrap clustering (on Stage 1 filtered data)
        'bootstrap_results': bootstrap_results,
        'consensus_matrix': consensus_matrix,
        'posterior_results': posterior_results,

        # Stage 2 filtering
        'stage2_filter_results': stage2_results,
        'final_D': final_D,
        'final_embryo_ids': final_embryo_ids,
        'final_consensus_matrix': final_consensus_matrix,
        'final_posterior_results': final_posterior_results,

        # Dendrograms
        'dendrogram_info_initial': dendrogram_info_initial,
        'dendrogram_info_final': dendrogram_info_final,

        # Filtering log
        'filtering_log': filtering_log,
    }


def create_filtering_log(
    embryo_ids: List[str],
    stage1_results: Dict[str, Any],
    stage2_results: Dict[str, Any],
    embryo_ids_after_stage1: List[str],
    final_embryo_ids: List[str],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create chain-of-custody DataFrame tracking every embryo through filtering.

    This log provides transparency about outlier removal, enabling detection of
    filtering bias (e.g., "are we removing all mutants?").

    Parameters
    ----------
    embryo_ids : List[str]
        Initial embryo IDs (before any filtering)
    stage1_results : Dict[str, Any]
        Output from identify_embryo_outliers_iqr()
        Empty dict if Stage 1 disabled
    stage2_results : Dict[str, Any]
        Output from identify_cluster_outliers_combined()
        Empty dict if Stage 2 disabled
    embryo_ids_after_stage1 : List[str]
        Embryo IDs after Stage 1 filtering
    final_embryo_ids : List[str]
        Embryo IDs after all filtering
    verbose : bool, default=True
        Print summary statistics

    Returns
    -------
    log_df : pd.DataFrame
        Chain-of-custody DataFrame with columns:
        - embryo_id: Embryo identifier
        - stage1_status: 'kept' or 'removed'
        - stage2_status: 'kept', 'removed_iqr', 'removed_posterior', 'removed_both', or '-'
        - final_status: 'kept', 'removed_stage1', or 'removed_stage2'
        - stage1_distance: Distance metric from Stage 1 (k-NN or median distance, else NaN)
        - stage2_within_cluster_distance: Within-cluster distance (if Stage 2 enabled, else NaN)
        - stage2_max_p: Posterior max_p (if Stage 2 enabled, else NaN)

    Examples
    --------
    >>> log_df = create_filtering_log(embryo_ids, stage1_res, stage2_res, ids1, ids_final)
    >>> # Check if filtering is biased by genotype
    >>> df_with_genotype = log_df.merge(metadata[['embryo_id', 'genotype']], on='embryo_id')
    >>> print(df_with_genotype.groupby(['genotype', 'final_status']).size())

    Notes
    -----
    - One row per initial embryo (complete history)
    - Stage 1/2 disabled → all embryos marked 'kept' for that stage
    - Use for publication transparency and bias detection
    """
    import matplotlib.pyplot as plt

    # Initialize log
    log_data = []

    for emb_id in embryo_ids:
        # Stage 1 status
        if stage1_results:
            idx = embryo_ids.index(emb_id)
            if emb_id in stage1_results['outlier_ids']:
                stage1_status = 'removed'
            else:
                stage1_status = 'kept'
            
            # Get distance metric (depends on method used)
            if 'knn_distances' in stage1_results:
                # k-NN method
                stage1_knn_dist = stage1_results['knn_distances'][idx]
            elif 'median_distances' in stage1_results:
                # IQR distance method
                stage1_knn_dist = stage1_results['median_distances'][idx]
            else:
                stage1_knn_dist = np.nan
        else:
            stage1_status = 'kept'
            stage1_knn_dist = np.nan

        # Stage 2 status (only if embryo made it through Stage 1)
        if emb_id in embryo_ids_after_stage1:
            if stage2_results:
                if emb_id in stage2_results['outlier_ids']:
                    reason = stage2_results['outlier_reason'][emb_id]
                    stage2_status = f'removed_{reason}'

                    idx_stage2 = embryo_ids_after_stage1.index(emb_id)
                    stage2_within_cluster_dist = stage2_results['within_cluster_mean_distances'][idx_stage2]
                    # Note: max_p is from the bootstrap_results that were input to stage2
                    # We'd need to pass that in separately, for now set to NaN
                    stage2_max_p = np.nan
                else:
                    stage2_status = 'kept'
                    idx_stage2 = embryo_ids_after_stage1.index(emb_id)
                    stage2_within_cluster_dist = stage2_results['within_cluster_mean_distances'][idx_stage2]
                    stage2_max_p = np.nan
            else:
                stage2_status = 'kept'
                stage2_within_cluster_dist = np.nan
                stage2_max_p = np.nan
        else:
            stage2_status = '-'  # Didn't make it to Stage 2
            stage2_within_cluster_dist = np.nan
            stage2_max_p = np.nan

        # Final status
        if emb_id in final_embryo_ids:
            final_status = 'kept'
        elif stage1_status == 'removed':
            final_status = 'removed_stage1'
        else:
            final_status = 'removed_stage2'

        log_data.append({
            'embryo_id': emb_id,
            'stage1_status': stage1_status,
            'stage2_status': stage2_status,
            'final_status': final_status,
            'stage1_distance': stage1_knn_dist,
            'stage2_within_cluster_distance': stage2_within_cluster_dist,
            'stage2_max_p': stage2_max_p
        })

    log_df = pd.DataFrame(log_data)

    if verbose:
        print(f"  Total embryos tracked: {len(log_df)}")
        print(f"  Final status breakdown:")
        for status, count in log_df['final_status'].value_counts().items():
            print(f"    {status}: {count}")

    return log_df
