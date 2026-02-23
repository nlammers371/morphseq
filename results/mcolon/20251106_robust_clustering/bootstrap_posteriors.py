"""
Bootstrap Assignment Posteriors Module

Computes per-embryo cluster assignment posteriors from bootstrap iterations.
Implements label alignment using the Hungarian algorithm to ensure cluster IDs
are consistent across bootstrap iterations.

Key Functions:
- align_bootstrap_labels(): Align cluster IDs to reference labels
- compute_assignment_posteriors(): Calculate p_i(c) for each embryo
- compute_quality_metrics(): Calculate entropy, max_p, log_odds_gap
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy as scipy_entropy
from typing import Dict, List, Tuple, Optional
import warnings


def build_contingency_table(labels_bootstrap: np.ndarray,
                            labels_reference: np.ndarray,
                            sampled_indices: np.ndarray) -> np.ndarray:
    """
    Build contingency table between bootstrap and reference labels.

    Parameters
    ----------
    labels_bootstrap : np.ndarray
        Cluster labels from bootstrap iteration (full length, -1 for non-sampled)
    labels_reference : np.ndarray
        Reference consensus cluster labels (full length)
    sampled_indices : np.ndarray
        Indices of embryos sampled in this bootstrap iteration

    Returns
    -------
    contingency : np.ndarray, shape (n_clusters_bootstrap, n_clusters_reference)
        Count matrix where contingency[i, j] = # embryos with bootstrap label i
        and reference label j
    """
    # Get sampled labels only
    boot_sampled = labels_bootstrap[sampled_indices]
    ref_sampled = labels_reference[sampled_indices]

    # Get unique cluster IDs
    boot_clusters = np.unique(boot_sampled)
    ref_clusters = np.unique(ref_sampled)

    # Build contingency table
    n_boot = len(boot_clusters)
    n_ref = len(ref_clusters)
    contingency = np.zeros((n_boot, n_ref), dtype=int)

    for i, boot_id in enumerate(boot_clusters):
        for j, ref_id in enumerate(ref_clusters):
            mask = (boot_sampled == boot_id) & (ref_sampled == ref_id)
            contingency[i, j] = np.sum(mask)

    return contingency, boot_clusters, ref_clusters


def align_bootstrap_labels(labels_bootstrap: np.ndarray,
                           labels_reference: np.ndarray,
                           sampled_indices: np.ndarray) -> np.ndarray:
    """
    Align bootstrap cluster labels to reference labels using Hungarian algorithm.

    The Hungarian algorithm finds the optimal 1-to-1 mapping between bootstrap
    cluster IDs and reference cluster IDs by maximizing overlap.

    Parameters
    ----------
    labels_bootstrap : np.ndarray
        Cluster labels from bootstrap iteration (full length, -1 for non-sampled)
    labels_reference : np.ndarray
        Reference consensus cluster labels (full length)
    sampled_indices : np.ndarray
        Indices of embryos sampled in this bootstrap iteration

    Returns
    -------
    labels_aligned : np.ndarray
        Bootstrap labels remapped to reference cluster IDs
    """
    contingency, boot_clusters, ref_clusters = build_contingency_table(
        labels_bootstrap, labels_reference, sampled_indices
    )

    # Hungarian algorithm minimizes cost, so use negative counts
    cost_matrix = -contingency

    # Find optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Build mapping: boot_id -> ref_id
    mapping = {}
    for boot_idx, ref_idx in zip(row_ind, col_ind):
        boot_id = boot_clusters[boot_idx]
        ref_id = ref_clusters[ref_idx]
        mapping[boot_id] = ref_id

    # Apply mapping
    labels_aligned = labels_bootstrap.copy()
    for boot_id, ref_id in mapping.items():
        labels_aligned[labels_bootstrap == boot_id] = ref_id

    return labels_aligned


def compute_assignment_posteriors(bootstrap_results: List[Dict],
                                  labels_reference: np.ndarray,
                                  n_embryos: int) -> Dict[str, np.ndarray]:
    """
    Compute per-embryo cluster assignment posteriors from bootstrap iterations.

    For each embryo i and cluster c:
        p_i(c) = (# times embryo i assigned to cluster c) / (# times embryo i sampled)

    Parameters
    ----------
    bootstrap_results : list of dict
        List of bootstrap iteration results, each containing:
        - 'labels': np.ndarray with cluster assignments (-1 for non-sampled)
        - 'indices': np.ndarray with sampled indices
    labels_reference : np.ndarray
        Reference consensus cluster labels
    n_embryos : int
        Total number of embryos

    Returns
    -------
    posteriors : dict with keys:
        - 'p_matrix': np.ndarray, shape (n_embryos, n_clusters)
            Posterior probabilities p_i(c)
        - 'sample_counts': np.ndarray, shape (n_embryos,)
            Number of times each embryo was sampled
        - 'aligned_labels': list of np.ndarray
            Aligned cluster labels for each bootstrap iteration
    """
    n_clusters = len(np.unique(labels_reference))
    n_iterations = len(bootstrap_results)

    # Initialize accumulator matrices
    assignment_counts = np.zeros((n_embryos, n_clusters), dtype=int)
    sample_counts = np.zeros(n_embryos, dtype=int)
    aligned_labels_all = []

    # Process each bootstrap iteration
    for boot_result in bootstrap_results:
        labels_boot = boot_result['labels']
        indices_sampled = boot_result['indices']

        # Align labels
        labels_aligned = align_bootstrap_labels(
            labels_boot, labels_reference, indices_sampled
        )
        aligned_labels_all.append(labels_aligned)

        # Accumulate counts for sampled embryos
        for idx in indices_sampled:
            cluster_id = labels_aligned[idx]
            if cluster_id >= 0:  # Valid cluster assignment
                assignment_counts[idx, cluster_id] += 1
                sample_counts[idx] += 1

    # Compute posteriors with normalization
    p_matrix = np.zeros((n_embryos, n_clusters), dtype=float)
    for i in range(n_embryos):
        if sample_counts[i] > 0:
            p_matrix[i, :] = assignment_counts[i, :] / sample_counts[i]
        else:
            # Embryo never sampled - uniform prior
            warnings.warn(f"Embryo {i} was never sampled in {n_iterations} iterations")
            p_matrix[i, :] = 1.0 / n_clusters

    return {
        'p_matrix': p_matrix,
        'sample_counts': sample_counts,
        'aligned_labels': aligned_labels_all
    }


def compute_quality_metrics(p_matrix: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute quality metrics from assignment posterior probabilities.

    Metrics:
    - max_p: Maximum posterior probability (confidence in top assignment)
    - entropy: Shannon entropy of posterior distribution (overall uncertainty)
    - log_odds_gap: log(p_top1) - log(p_top2) (disambiguation between top 2)
    - modal_cluster: Cluster with highest posterior probability

    Parameters
    ----------
    p_matrix : np.ndarray, shape (n_embryos, n_clusters)
        Posterior probability matrix

    Returns
    -------
    metrics : dict with keys:
        - 'max_p': np.ndarray, shape (n_embryos,)
        - 'entropy': np.ndarray, shape (n_embryos,)
        - 'log_odds_gap': np.ndarray, shape (n_embryos,)
        - 'modal_cluster': np.ndarray, shape (n_embryos,), dtype=int
        - 'second_best_cluster': np.ndarray, shape (n_embryos,), dtype=int
    """
    n_embryos, n_clusters = p_matrix.shape

    # Max probability (confidence)
    max_p = np.max(p_matrix, axis=1)

    # Modal cluster
    modal_cluster = np.argmax(p_matrix, axis=1)

    # Entropy (overall uncertainty)
    entropy = np.array([scipy_entropy(p_matrix[i, :], base=2) for i in range(n_embryos)])

    # Log-odds gap (top1 vs top2)
    log_odds_gap = np.zeros(n_embryos)
    second_best_cluster = np.zeros(n_embryos, dtype=int)

    for i in range(n_embryos):
        # Get top 2 probabilities
        sorted_indices = np.argsort(p_matrix[i, :])[::-1]
        p_top1 = p_matrix[i, sorted_indices[0]]
        p_top2 = p_matrix[i, sorted_indices[1]] if n_clusters > 1 else 0.0

        second_best_cluster[i] = sorted_indices[1] if n_clusters > 1 else sorted_indices[0]

        # Compute log-odds gap with numerical stability
        if p_top1 > 0 and p_top2 > 0:
            log_odds_gap[i] = np.log(p_top1) - np.log(p_top2)
        elif p_top1 > 0:
            log_odds_gap[i] = np.inf  # Top cluster is certain
        else:
            log_odds_gap[i] = 0.0  # Both zero (shouldn't happen)

    return {
        'max_p': max_p,
        'entropy': entropy,
        'log_odds_gap': log_odds_gap,
        'modal_cluster': modal_cluster,
        'second_best_cluster': second_best_cluster
    }


def analyze_bootstrap_results(bootstrap_results_dict: Dict,
                              return_aligned_labels: bool = False) -> Dict:
    """
    Complete analysis pipeline: compute posteriors and quality metrics.

    Parameters
    ----------
    bootstrap_results_dict : dict
        Output from run_bootstrap(), containing:
        - 'reference_labels': Reference cluster assignments
        - 'bootstrap_results': List of bootstrap iteration results
    return_aligned_labels : bool
        Whether to include aligned labels in output (can be memory-intensive)

    Returns
    -------
    analysis : dict containing:
        - 'p_matrix': Posterior probability matrix
        - 'sample_counts': Per-embryo sample counts
        - 'max_p': Maximum posterior probability per embryo
        - 'entropy': Shannon entropy per embryo
        - 'log_odds_gap': Log-odds gap per embryo
        - 'modal_cluster': Most likely cluster per embryo
        - 'second_best_cluster': Second most likely cluster
        - 'aligned_labels': (optional) List of aligned label arrays
    """
    labels_reference = bootstrap_results_dict['reference_labels']
    bootstrap_results = bootstrap_results_dict['bootstrap_results']
    n_embryos = len(labels_reference)

    # Compute posteriors
    posterior_results = compute_assignment_posteriors(
        bootstrap_results, labels_reference, n_embryos
    )

    # Compute quality metrics
    quality_metrics = compute_quality_metrics(posterior_results['p_matrix'])

    # Combine results
    analysis = {
        'p_matrix': posterior_results['p_matrix'],
        'sample_counts': posterior_results['sample_counts'],
        **quality_metrics
    }

    if return_aligned_labels:
        analysis['aligned_labels'] = posterior_results['aligned_labels']

    return analysis


if __name__ == '__main__':
    # Example usage
    import pickle

    # Load bootstrap results
    with open('../20251103_DTW_analysis/output/2_select_k/data/bootstrap_k3.pkl', 'rb') as f:
        bootstrap_data = pickle.load(f)

    # Analyze
    results = analyze_bootstrap_results(bootstrap_data)

    # Print summary
    print(f"Analyzed {len(bootstrap_data['bootstrap_results'])} bootstrap iterations")
    print(f"Mean max_p: {results['max_p'].mean():.3f}")
    print(f"Mean entropy: {results['entropy'].mean():.3f}")
    print(f"Mean log-odds gap: {results['log_odds_gap'].mean():.3f}")
