"""
Cluster extraction utilities for k-selection results.

Provides functions to extract embryo IDs from clusters, summarize cluster
composition, and map clusters to known phenotypes.

Example
-------
>>> from morphseq.trajectory_analysis import extract_cluster_embryos, get_cluster_summary
>>>
>>> # Get all embryos from cluster 2 at k=4
>>> embryo_ids = extract_cluster_embryos(k_results, k=4, cluster=2)
>>>
>>> # Get only core members
>>> core_ids = extract_cluster_embryos(k_results, k=4, cluster=2, membership='core')
>>>
>>> # Get summary of all clusters at k=4
>>> summary_df = get_cluster_summary(k_results, k=4)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union


def extract_cluster_embryos(
    k_results: Dict[str, Any],
    k: int,
    cluster: Union[int, str],
    cluster_names: Optional[Dict[int, str]] = None,
    membership: Optional[str] = None,
) -> List[str]:
    """
    Extract embryo IDs from a specific cluster.

    Parameters
    ----------
    k_results : Dict
        Output from run_k_selection_with_plots() or evaluate_k_range()
    k : int
        Which k value to use
    cluster : int or str
        Cluster identifier. If int, uses cluster index directly.
        If str, looks up in cluster_names dict.
    cluster_names : Dict[int, str], optional
        Mapping of cluster_id -> name. Required if cluster is a string.
    membership : str, optional
        Filter by membership quality: 'core', 'uncertain', or None for all.

    Returns
    -------
    embryo_ids : List[str]
        List of embryo IDs in the specified cluster

    Examples
    --------
    >>> # Get all embryos from cluster 2
    >>> ids = extract_cluster_embryos(k_results, k=4, cluster=2)

    >>> # Get only core members from cluster 2
    >>> core_ids = extract_cluster_embryos(k_results, k=4, cluster=2, membership='core')

    >>> # Use cluster names
    >>> cluster_names = {0: 'wildtype', 1: 'phenotype', 2: 'severe'}
    >>> pheno_ids = extract_cluster_embryos(k_results, k=4, cluster='phenotype',
    ...                                      cluster_names=cluster_names)
    """
    # Validate k_results structure
    if 'clustering_by_k' not in k_results:
        raise ValueError(
            "k_results must contain 'clustering_by_k'. "
            "Expected output from run_k_selection_with_plots() or evaluate_k_range()"
        )

    if k not in k_results['clustering_by_k']:
        available_k = list(k_results['clustering_by_k'].keys())
        raise ValueError(f"k={k} not found in k_results. Available: {available_k}")

    k_data = k_results['clustering_by_k'][k]

    # Resolve cluster identifier
    if isinstance(cluster, str):
        if cluster_names is None:
            raise ValueError(
                "cluster_names dict required when cluster is a string. "
                "Pass cluster_names={0: 'name', 1: 'name', ...}"
            )
        # Reverse lookup: find cluster_id for the given name
        name_to_id = {v: k for k, v in cluster_names.items()}
        if cluster not in name_to_id:
            raise ValueError(
                f"Cluster name '{cluster}' not found in cluster_names. "
                f"Available: {list(cluster_names.values())}"
            )
        cluster_id = name_to_id[cluster]
    else:
        cluster_id = int(cluster)

    # Get embryo IDs in this cluster
    cluster_to_embryos = k_data['assignments']['cluster_to_embryos']

    if cluster_id not in cluster_to_embryos:
        raise ValueError(
            f"Cluster {cluster_id} not found. Available: {list(cluster_to_embryos.keys())}"
        )

    embryo_ids = list(cluster_to_embryos[cluster_id])

    # Filter by membership if requested
    if membership is not None:
        if membership not in ['core', 'uncertain', 'outlier']:
            raise ValueError(
                f"membership must be 'core', 'uncertain', or 'outlier', got '{membership}'"
            )

        embryo_to_membership = k_data['membership']['embryo_to_membership_quality']
        embryo_ids = [
            eid for eid in embryo_ids
            if embryo_to_membership.get(eid) == membership
        ]

    return embryo_ids


def get_cluster_summary(
    k_results: Dict[str, Any],
    k: int,
    cluster_names: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """
    Get summary table of all clusters at given k.

    Parameters
    ----------
    k_results : Dict
        Output from run_k_selection_with_plots() or evaluate_k_range()
    k : int
        Which k value to use
    cluster_names : Dict[int, str], optional
        Mapping of cluster_id -> name for display

    Returns
    -------
    summary_df : pd.DataFrame
        Summary with columns:
        - cluster: int
        - name: str (if cluster_names provided)
        - n_embryos: total count
        - n_core: core members
        - n_uncertain: uncertain members
        - n_outlier: outlier members
        - pct_core: percentage core
        - mean_posterior: mean max posterior probability

    Examples
    --------
    >>> summary = get_cluster_summary(k_results, k=4)
    >>> print(summary)
    >>>
    >>> # With cluster names
    >>> cluster_names = {0: 'wildtype', 1: 'phenotype'}
    >>> summary = get_cluster_summary(k_results, k=4, cluster_names=cluster_names)
    """
    # Validate
    if 'clustering_by_k' not in k_results:
        raise ValueError("k_results must contain 'clustering_by_k'")

    if k not in k_results['clustering_by_k']:
        available_k = list(k_results['clustering_by_k'].keys())
        raise ValueError(f"k={k} not found. Available: {available_k}")

    k_data = k_results['clustering_by_k'][k]
    cluster_to_embryos = k_data['assignments']['cluster_to_embryos']
    embryo_to_membership = k_data['membership']['embryo_to_membership_quality']
    posteriors = k_data['posteriors']

    # Build embryo_id to max_p mapping
    embryo_ids_list = posteriors['embryo_ids']
    max_p_values = posteriors['max_p']
    embryo_to_max_p = dict(zip(embryo_ids_list, max_p_values))

    # Compute summary for each cluster
    records = []
    for cluster_id in sorted(cluster_to_embryos.keys()):
        embryos = cluster_to_embryos[cluster_id]
        n_total = len(embryos)

        # Count by membership
        memberships = [embryo_to_membership.get(e, 'unknown') for e in embryos]
        n_core = sum(1 for m in memberships if m == 'core')
        n_uncertain = sum(1 for m in memberships if m == 'uncertain')
        n_outlier = sum(1 for m in memberships if m == 'outlier')

        # Mean posterior
        max_ps = [embryo_to_max_p.get(e, np.nan) for e in embryos]
        mean_posterior = np.nanmean(max_ps) if max_ps else np.nan

        record = {
            'cluster': cluster_id,
            'n_embryos': n_total,
            'n_core': n_core,
            'n_uncertain': n_uncertain,
            'n_outlier': n_outlier,
            'pct_core': 100.0 * n_core / n_total if n_total > 0 else 0,
            'mean_posterior': mean_posterior,
        }

        # Add name if provided
        if cluster_names and cluster_id in cluster_names:
            record['name'] = cluster_names[cluster_id]

        records.append(record)

    df = pd.DataFrame(records)

    # Reorder columns if names present
    if cluster_names:
        cols = ['cluster', 'name'] + [c for c in df.columns if c not in ['cluster', 'name']]
        df = df[cols]

    return df


def map_clusters_to_phenotypes(
    k_results: Dict[str, Any],
    k: int,
    phenotype_dict: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Show how known phenotypes distribute across clusters.

    Useful for understanding which cluster corresponds to which phenotype
    when you have ground truth labels.

    Parameters
    ----------
    k_results : Dict
        Output from run_k_selection_with_plots() or evaluate_k_range()
    k : int
        Which k value to use
    phenotype_dict : Dict[str, List[str]]
        Mapping of phenotype_name -> list of embryo IDs
        Example: {'CE': [ce_ids], 'HTA': [hta_ids]}

    Returns
    -------
    mapping_df : pd.DataFrame
        Columns:
        - phenotype: name
        - n_embryos: total in phenotype
        - main_cluster: cluster with most members
        - purity: fraction in main cluster
        - distribution: dict of cluster -> count

    Examples
    --------
    >>> phenotypes = {
    ...     'CE': load_phenotype_file('ce_embryos.txt'),
    ...     'HTA': load_phenotype_file('hta_embryos.txt'),
    ... }
    >>> mapping = map_clusters_to_phenotypes(k_results, k=4, phenotype_dict=phenotypes)
    >>> print(mapping)
    """
    # Validate
    if 'clustering_by_k' not in k_results:
        raise ValueError("k_results must contain 'clustering_by_k'")

    if k not in k_results['clustering_by_k']:
        available_k = list(k_results['clustering_by_k'].keys())
        raise ValueError(f"k={k} not found. Available: {available_k}")

    k_data = k_results['clustering_by_k'][k]
    embryo_to_cluster = k_data['assignments']['embryo_to_cluster']

    records = []
    for phenotype_name, embryo_ids in phenotype_dict.items():
        # Count how many of each phenotype end up in each cluster
        cluster_counts = {}
        for eid in embryo_ids:
            if eid in embryo_to_cluster:
                c = embryo_to_cluster[eid]
                cluster_counts[c] = cluster_counts.get(c, 0) + 1

        if not cluster_counts:
            records.append({
                'phenotype': phenotype_name,
                'n_embryos': len(embryo_ids),
                'n_in_clusters': 0,
                'main_cluster': None,
                'purity': 0.0,
                'distribution': {},
            })
            continue

        # Find main cluster (most members)
        main_cluster = max(cluster_counts.keys(), key=lambda c: cluster_counts[c])
        n_in_main = cluster_counts[main_cluster]
        n_in_clusters = sum(cluster_counts.values())
        purity = n_in_main / n_in_clusters if n_in_clusters > 0 else 0

        records.append({
            'phenotype': phenotype_name,
            'n_embryos': len(embryo_ids),
            'n_in_clusters': n_in_clusters,
            'main_cluster': main_cluster,
            'purity': purity,
            'distribution': cluster_counts,
        })

    return pd.DataFrame(records)
