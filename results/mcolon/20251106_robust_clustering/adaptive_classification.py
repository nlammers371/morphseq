"""
Adaptive Classification Module

Classifies embryos as core/uncertain/outlier based on bootstrap assignment posteriors.
Uses 2D gating with max_p (confidence) and log_odds_gap (disambiguation) metrics.

Key Functions:
- classify_embryos_2d(): Two-dimensional gating classifier
- classify_embryos_adaptive(): Adaptive per-cluster thresholds
- get_classification_summary(): Aggregate statistics
"""

import numpy as np
from typing import Dict, Optional, Tuple


def classify_embryos_2d(max_p: np.ndarray,
                       log_odds_gap: np.ndarray,
                       modal_cluster: np.ndarray,
                       threshold_max_p: float = 0.8,
                       threshold_log_odds: float = 0.7,
                       threshold_outlier_max_p: float = 0.5) -> Dict:
    """
    Classify embryos using 2D gating with max_p and log_odds_gap.

    Classification logic:
    - Core: high confidence (max_p) AND unambiguous (log_odds_gap)
    - Outlier: very low confidence (max_p below outlier threshold)
    - Uncertain: everything else (either ambiguous or moderate confidence)

    Parameters
    ----------
    max_p : np.ndarray, shape (n_embryos,)
        Maximum posterior probability per embryo
    log_odds_gap : np.ndarray, shape (n_embryos,)
        Log-odds gap between top 2 clusters
    modal_cluster : np.ndarray, shape (n_embryos,)
        Most likely cluster assignment per embryo
    threshold_max_p : float, default=0.8
        Minimum max_p for core membership
    threshold_log_odds : float, default=0.7
        Minimum log_odds_gap for core membership
    threshold_outlier_max_p : float, default=0.5
        Maximum max_p for outlier classification

    Returns
    -------
    classification : dict with keys:
        - 'category': np.ndarray of strings ('core'/'uncertain'/'outlier')
        - 'cluster': np.ndarray of cluster assignments
        - 'max_p': np.ndarray of max_p values
        - 'log_odds_gap': np.ndarray of log_odds_gap values
        - 'thresholds': dict of threshold values used
    """
    n_embryos = len(max_p)
    category = np.empty(n_embryos, dtype=object)

    # Apply 2D gating logic
    for i in range(n_embryos):
        if max_p[i] < threshold_outlier_max_p:
            # Very low confidence -> outlier
            category[i] = 'outlier'
        elif max_p[i] >= threshold_max_p and log_odds_gap[i] >= threshold_log_odds:
            # High confidence AND unambiguous -> core
            category[i] = 'core'
        else:
            # Moderate confidence OR ambiguous -> uncertain
            category[i] = 'uncertain'

    return {
        'category': category,
        'cluster': modal_cluster,
        'max_p': max_p,
        'log_odds_gap': log_odds_gap,
        'thresholds': {
            'max_p': threshold_max_p,
            'log_odds_gap': threshold_log_odds,
            'outlier_max_p': threshold_outlier_max_p
        }
    }


def classify_embryos_adaptive(max_p: np.ndarray,
                              log_odds_gap: np.ndarray,
                              modal_cluster: np.ndarray,
                              base_threshold_max_p: float = 0.8,
                              base_threshold_log_odds: float = 0.7,
                              threshold_outlier_max_p: float = 0.5,
                              adaptive_percentile: float = 0.75) -> Dict:
    """
    Classify embryos with adaptive per-cluster thresholds.

    For clusters with high overall confidence, uses stricter thresholds.
    For clusters with lower confidence, relaxes thresholds to avoid
    over-penalizing naturally more variable clusters.

    Parameters
    ----------
    max_p : np.ndarray, shape (n_embryos,)
        Maximum posterior probability per embryo
    log_odds_gap : np.ndarray, shape (n_embryos,)
        Log-odds gap between top 2 clusters
    modal_cluster : np.ndarray, shape (n_embryos,)
        Most likely cluster assignment per embryo
    base_threshold_max_p : float, default=0.8
        Base threshold for max_p (adjusted per cluster)
    base_threshold_log_odds : float, default=0.7
        Base threshold for log_odds_gap (adjusted per cluster)
    threshold_outlier_max_p : float, default=0.5
        Global outlier threshold (not adapted)
    adaptive_percentile : float, default=0.75
        Percentile for computing cluster-specific thresholds

    Returns
    -------
    classification : dict (same structure as classify_embryos_2d)
        Additional key 'cluster_thresholds' contains per-cluster thresholds
    """
    n_embryos = len(max_p)
    n_clusters = len(np.unique(modal_cluster))
    category = np.empty(n_embryos, dtype=object)

    # Compute per-cluster thresholds
    cluster_thresholds = {}
    for c in range(n_clusters):
        mask = (modal_cluster == c)
        if np.sum(mask) > 0:
            # Use percentile of within-cluster distributions
            max_p_threshold = max(
                np.percentile(max_p[mask], adaptive_percentile * 100),
                base_threshold_max_p * 0.7  # Don't go below 70% of base
            )
            log_odds_threshold = max(
                np.percentile(log_odds_gap[mask], adaptive_percentile * 100),
                base_threshold_log_odds * 0.7
            )
        else:
            # Fallback to base thresholds
            max_p_threshold = base_threshold_max_p
            log_odds_threshold = base_threshold_log_odds

        cluster_thresholds[c] = {
            'max_p': max_p_threshold,
            'log_odds_gap': log_odds_threshold
        }

    # Apply adaptive classification
    for i in range(n_embryos):
        cluster_id = modal_cluster[i]
        thresh_max_p = cluster_thresholds[cluster_id]['max_p']
        thresh_log_odds = cluster_thresholds[cluster_id]['log_odds_gap']

        if max_p[i] < threshold_outlier_max_p:
            category[i] = 'outlier'
        elif max_p[i] >= thresh_max_p and log_odds_gap[i] >= thresh_log_odds:
            category[i] = 'core'
        else:
            category[i] = 'uncertain'

    return {
        'category': category,
        'cluster': modal_cluster,
        'max_p': max_p,
        'log_odds_gap': log_odds_gap,
        'thresholds': {
            'base_max_p': base_threshold_max_p,
            'base_log_odds': base_threshold_log_odds,
            'outlier_max_p': threshold_outlier_max_p,
            'adaptive_percentile': adaptive_percentile
        },
        'cluster_thresholds': cluster_thresholds
    }


def get_classification_summary(classification: Dict) -> Dict:
    """
    Compute summary statistics for classification results.

    Parameters
    ----------
    classification : dict
        Output from classify_embryos_2d() or classify_embryos_adaptive()

    Returns
    -------
    summary : dict with keys:
        - 'n_core': Total core members
        - 'n_uncertain': Total uncertain members
        - 'n_outlier': Total outliers
        - 'core_fraction': Fraction of embryos classified as core
        - 'uncertain_fraction': Fraction classified as uncertain
        - 'outlier_fraction': Fraction classified as outlier
        - 'per_cluster': dict with per-cluster breakdowns
    """
    category = classification['category']
    cluster = classification['cluster']
    n_embryos = len(category)

    # Global counts
    n_core = np.sum(category == 'core')
    n_uncertain = np.sum(category == 'uncertain')
    n_outlier = np.sum(category == 'outlier')

    # Per-cluster breakdown
    n_clusters = len(np.unique(cluster))
    per_cluster = {}

    for c in range(n_clusters):
        mask = (cluster == c)
        per_cluster[c] = {
            'total': np.sum(mask),
            'n_core': np.sum((category == 'core') & mask),
            'n_uncertain': np.sum((category == 'uncertain') & mask),
            'n_outlier': np.sum((category == 'outlier') & mask),
            'core_fraction': np.sum((category == 'core') & mask) / np.sum(mask) if np.sum(mask) > 0 else 0.0,
            'mean_max_p': np.mean(classification['max_p'][mask]) if np.sum(mask) > 0 else 0.0,
            'mean_log_odds_gap': np.mean(classification['log_odds_gap'][mask]) if np.sum(mask) > 0 else 0.0
        }

    summary = {
        'n_core': n_core,
        'n_uncertain': n_uncertain,
        'n_outlier': n_outlier,
        'core_fraction': n_core / n_embryos if n_embryos > 0 else 0.0,
        'uncertain_fraction': n_uncertain / n_embryos if n_embryos > 0 else 0.0,
        'outlier_fraction': n_outlier / n_embryos if n_embryos > 0 else 0.0,
        'per_cluster': per_cluster
    }

    return summary


def get_core_indices(classification: Dict) -> np.ndarray:
    """Get indices of core members."""
    return np.where(classification['category'] == 'core')[0]


def get_uncertain_indices(classification: Dict) -> np.ndarray:
    """Get indices of uncertain members."""
    return np.where(classification['category'] == 'uncertain')[0]


def get_outlier_indices(classification: Dict) -> np.ndarray:
    """Get indices of outlier members."""
    return np.where(classification['category'] == 'outlier')[0]


def format_classification_report(classification: Dict,
                                 summary: Optional[Dict] = None,
                                 embryo_ids: Optional[np.ndarray] = None) -> str:
    """
    Generate a human-readable classification report.

    Parameters
    ----------
    classification : dict
        Classification results
    summary : dict, optional
        Summary statistics (computed if not provided)
    embryo_ids : np.ndarray, optional
        Embryo identifiers for display

    Returns
    -------
    report : str
        Formatted text report
    """
    if summary is None:
        summary = get_classification_summary(classification)

    if embryo_ids is None:
        embryo_ids = np.arange(len(classification['category']))

    # Global summary
    lines = [
        "=" * 60,
        "CLASSIFICATION SUMMARY",
        "=" * 60,
        f"Total embryos: {len(classification['category'])}",
        f"Core members: {summary['n_core']} ({summary['core_fraction']:.1%})",
        f"Uncertain: {summary['n_uncertain']} ({summary['uncertain_fraction']:.1%})",
        f"Outliers: {summary['n_outlier']} ({summary['outlier_fraction']:.1%})",
        "",
        "Thresholds used:",
    ]

    for key, value in classification['thresholds'].items():
        if key != 'adaptive_percentile' and key != 'cluster_thresholds':
            lines.append(f"  {key}: {value:.3f}")

    lines.append("")
    lines.append("Per-Cluster Breakdown:")
    lines.append("-" * 60)

    for cluster_id, stats in summary['per_cluster'].items():
        lines.append(f"\nCluster {cluster_id} (n={stats['total']}):")
        lines.append(f"  Core: {stats['n_core']} ({stats['core_fraction']:.1%})")
        lines.append(f"  Uncertain: {stats['n_uncertain']}")
        lines.append(f"  Outliers: {stats['n_outlier']}")
        lines.append(f"  Mean max_p: {stats['mean_max_p']:.3f}")
        lines.append(f"  Mean log-odds gap: {stats['mean_log_odds_gap']:.3f}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


if __name__ == '__main__':
    # Example usage
    import pickle
    import sys
    sys.path.append('..')
    from bootstrap_posteriors import analyze_bootstrap_results

    # Load and analyze bootstrap results
    with open('../20251103_DTW_analysis/output/2_select_k/data/bootstrap_k3.pkl', 'rb') as f:
        bootstrap_data = pickle.load(f)

    posterior_results = analyze_bootstrap_results(bootstrap_data)

    # Classify using 2D gating
    classification = classify_embryos_2d(
        max_p=posterior_results['max_p'],
        log_odds_gap=posterior_results['log_odds_gap'],
        modal_cluster=posterior_results['modal_cluster'],
        threshold_max_p=0.8,
        threshold_log_odds=0.7
    )

    # Get summary
    summary = get_classification_summary(classification)

    # Print report
    print(format_classification_report(classification, summary))
