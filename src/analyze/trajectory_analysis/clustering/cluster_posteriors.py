"""
Cluster Posteriors

Posterior probability computation from bootstrap clustering results.

This module computes per-embryo cluster assignment probabilities p_i(c) from
bootstrap resampling, along with information-theoretic quality metrics.

Functions
---------
- analyze_bootstrap_results: Complete posterior analysis pipeline
- compute_assignment_posteriors: Calculate p_i(c) from bootstrap labels
- compute_quality_metrics: Calculate entropy, log_odds_gap, max_p
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Any, Tuple, Optional


def align_bootstrap_labels(
    labels_bootstrap: np.ndarray,
    labels_reference: np.ndarray,
    sampled_indices: np.ndarray,
    n_clusters: int
) -> np.ndarray:
    """
    Align bootstrap cluster labels to reference using Hungarian algorithm.

    Parameters
    ----------
    labels_bootstrap : np.ndarray
        Cluster labels from bootstrap iteration (with -1 for unsampled)
    labels_reference : np.ndarray
        Reference consensus labels
    sampled_indices : np.ndarray
        Indices of embryos sampled in this iteration
    n_clusters : int
        Number of clusters

    Returns
    -------
    aligned_labels : np.ndarray
        Bootstrap labels aligned to reference numbering
    """
    # Build contingency table for sampled elements only
    sampled_ref = labels_reference[sampled_indices]
    sampled_boot = labels_bootstrap[sampled_indices]

    # Create cost matrix (negative agreement for maximization)
    cost_matrix = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            cost_matrix[i, j] = -np.sum((sampled_boot == i) & (sampled_ref == j))

    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Create mapping from bootstrap clusters to reference clusters
    mapping = {}
    for boot_cluster, ref_cluster in zip(row_ind, col_ind):
        mapping[boot_cluster] = ref_cluster

    # Apply mapping
    aligned_labels = np.full_like(labels_bootstrap, -1, dtype=int)
    for i in range(len(labels_bootstrap)):
        if labels_bootstrap[i] >= 0:
            aligned_labels[i] = mapping.get(labels_bootstrap[i], labels_bootstrap[i])

    return aligned_labels


def compute_assignment_posteriors(
    bootstrap_results_dict: Dict[str, Any],
    return_aligned_labels: bool = False
) -> Dict[str, Any]:
    """
    Compute posterior probabilities p_i(c) from bootstrap results.

    For each embryo i and cluster c:
    p_i(c) = (# times i assigned to c) / (# times i was sampled)

    Parameters
    ----------
    bootstrap_results_dict : dict
        Output from run_bootstrap_hierarchical()
    return_aligned_labels : bool
        If True, return aligned labels from each iteration

    Returns
    -------
    posteriors : dict
        - 'p_matrix': np.ndarray, shape (n_embryos, n_clusters)
        - 'sample_counts': np.ndarray, shape (n_embryos,)
        - 'aligned_labels': list of np.ndarray (optional)
    """
    n_embryos = bootstrap_results_dict['n_samples']
    n_clusters = bootstrap_results_dict['n_clusters']
    reference_labels = bootstrap_results_dict['reference_labels']
    bootstrap_results = bootstrap_results_dict['bootstrap_results']

    # Count matrix
    count_matrix = np.zeros((n_embryos, n_clusters), dtype=int)
    sample_counts = np.zeros(n_embryos, dtype=int)
    aligned_labels_list = []

    for boot_result in bootstrap_results:
        labels_boot = boot_result['labels']
        indices = boot_result['indices']

        # Align labels
        aligned_boot = align_bootstrap_labels(
            labels_boot, reference_labels, indices, n_clusters
        )

        if return_aligned_labels:
            aligned_labels_list.append(aligned_boot)

        # Update counts
        for i in range(n_embryos):
            if aligned_boot[i] >= 0:
                count_matrix[i, aligned_boot[i]] += 1
                sample_counts[i] += 1

    # Compute posteriors (avoid division by zero)
    p_matrix = np.zeros((n_embryos, n_clusters), dtype=float)
    for i in range(n_embryos):
        if sample_counts[i] > 0:
            p_matrix[i, :] = count_matrix[i, :] / sample_counts[i]

    result = {
        'p_matrix': p_matrix,
        'sample_counts': sample_counts,
    }
    if return_aligned_labels:
        result['aligned_labels'] = aligned_labels_list

    return result


def compute_quality_metrics(p_matrix: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute quality metrics from posterior probability matrix.

    Parameters
    ----------
    p_matrix : np.ndarray
        Posterior probability matrix (n_embryos × n_clusters)

    Returns
    -------
    metrics : dict
        - 'max_p': Maximum probability per embryo
        - 'entropy': Shannon entropy per embryo
        - 'log_odds_gap': Log-odds gap between top 2 clusters
        - 'modal_cluster': Most likely cluster per embryo
        - 'second_best_cluster': Second most likely cluster per embryo
    """
    n_embryos = p_matrix.shape[0]

    # Max probability
    max_p = np.max(p_matrix, axis=1)

    # Modal cluster
    modal_cluster = np.argmax(p_matrix, axis=1)

    # Entropy
    entropy = -np.sum(p_matrix * np.log2(p_matrix + 1e-10), axis=1)

    # Log-odds gap (top 2 clusters)
    sorted_probs = np.sort(p_matrix, axis=1)[:, ::-1]
    p_top1 = sorted_probs[:, 0]
    p_top2 = sorted_probs[:, 1] if p_matrix.shape[1] > 1 else np.zeros(n_embryos)

    log_odds_gap = np.log2((p_top1 + 1e-10) / (p_top2 + 1e-10))

    # Second-best cluster (guard for k=1 case)
    if p_matrix.shape[1] > 1:
        second_best_cluster = np.argsort(p_matrix, axis=1)[:, -2]
    else:
        # Single cluster case: no second-best cluster, use -1 to indicate undefined
        second_best_cluster = np.full(n_embryos, -1, dtype=int)

    return {
        'max_p': max_p,
        'entropy': entropy,
        'log_odds_gap': log_odds_gap,
        'modal_cluster': modal_cluster,
        'second_best_cluster': second_best_cluster
    }


def analyze_bootstrap_results(
    bootstrap_results_dict: Dict[str, Any],
    return_aligned_labels: bool = False
) -> Dict[str, Any]:
    """
    Complete posterior analysis from bootstrap clustering results.

    Parameters
    ----------
    bootstrap_results_dict : dict
        Output from run_bootstrap_hierarchical() - MUST include 'embryo_ids' key
    return_aligned_labels : bool, default=False
        If True, include aligned label history in output

    Returns
    -------
    posterior_analysis : dict
        - 'embryo_ids': list of str (copied from input)
        - 'p_matrix': np.ndarray, shape (n_embryos, n_clusters)
        - 'sample_counts': np.ndarray, shape (n_embryos,)
        - 'max_p': np.ndarray, shape (n_embryos,)
        - 'entropy': np.ndarray, shape (n_embryos,)
        - 'log_odds_gap': np.ndarray, shape (n_embryos,)
        - 'modal_cluster': np.ndarray, shape (n_embryos,), dtype=int
        - 'second_best_cluster': np.ndarray, shape (n_embryos,), dtype=int
        - 'aligned_labels': list (optional, if return_aligned_labels=True)

    Examples
    --------
    >>> embryo_ids = ['emb_01', 'emb_02', 'emb_03']
    >>> bootstrap_results = run_bootstrap_hierarchical(D, k=3, embryo_ids=embryo_ids)
    >>> posteriors = analyze_bootstrap_results(bootstrap_results)
    >>>
    >>> # Lookup by embryo ID
    >>> idx = posteriors['embryo_ids'].index('emb_02')
    >>> print(f"Max probability: {posteriors['max_p'][idx]:.3f}")
    >>> print(f"Entropy: {posteriors['entropy'][idx]:.3f}")
    """
    # Compute posteriors
    posteriors = compute_assignment_posteriors(
        bootstrap_results_dict,
        return_aligned_labels=return_aligned_labels
    )

    # Compute quality metrics
    metrics = compute_quality_metrics(posteriors['p_matrix'])

    # Combine results
    result = {
        'embryo_ids': bootstrap_results_dict['embryo_ids'],
        'p_matrix': posteriors['p_matrix'],
        'sample_counts': posteriors['sample_counts'],
        'max_p': metrics['max_p'],
        'entropy': metrics['entropy'],
        'log_odds_gap': metrics['log_odds_gap'],
        'modal_cluster': metrics['modal_cluster'],
        'second_best_cluster': metrics['second_best_cluster'],
        'n_clusters': bootstrap_results_dict['n_clusters']
    }

    if return_aligned_labels:
        result['aligned_labels'] = posteriors['aligned_labels']

    return result
