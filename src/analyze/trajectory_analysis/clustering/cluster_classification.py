"""
Cluster Classification

Membership quality classification using posterior probabilities.

Classifies embryos as core/uncertain/outlier based on confidence metrics
using 2D gating (max_p × log_odds_gap) or adaptive per-cluster thresholds.

Functions
---------
- classify_membership_2d: Two-dimensional gating classification
- classify_membership_adaptive: Adaptive per-cluster thresholds
- get_classification_summary: Summary statistics
"""

import numpy as np
from typing import Dict, Any, Optional, List
from ..config import THRESHOLD_MAX_P, THRESHOLD_LOG_ODDS_GAP, THRESHOLD_OUTLIER_MAX_P, ADAPTIVE_PERCENTILE


def classify_membership_2d(
    max_p: np.ndarray,
    log_odds_gap: np.ndarray,
    modal_cluster: np.ndarray,
    embryo_ids: Optional[List[str]] = None,
    *,
    threshold_max_p: float = THRESHOLD_MAX_P,
    threshold_log_odds_gap: float = THRESHOLD_LOG_ODDS_GAP,
    threshold_outlier_max_p: float = THRESHOLD_OUTLIER_MAX_P
) -> Dict[str, Any]:
    """
    Classify membership quality using 2D gating.

    Uses two metrics jointly for classification:
    - max_p: Overall confidence in assigned cluster
    - log_odds_gap: Disambiguation between top 2 clusters

    Classification logic:
    - Outlier: max_p < threshold_outlier_max_p (very low confidence)
    - Core: max_p >= threshold_max_p AND log_odds_gap >= threshold_log_odds
    - Uncertain: Everything else (high confidence but contested, or low gap)

    Parameters
    ----------
    max_p : np.ndarray
        Maximum posterior probability per embryo
    log_odds_gap : np.ndarray
        Log-odds gap between top 2 clusters per embryo
    modal_cluster : np.ndarray
        Most likely cluster assignment per embryo
    embryo_ids : list of str, optional
        Embryo identifiers. If provided, included in output for tracking.
    threshold_max_p : float, default=0.8
        Minimum probability for core membership (80% confidence)
    threshold_log_odds : float, default=0.7
        Minimum log-odds gap for core membership (unambiguous)
    threshold_outlier_max_p : float, default=0.5
        Maximum probability before being outlier (50% confidence)

    Returns
    -------
    classification : dict
        - 'embryo_ids': list of str (if provided as input)
        - 'category': np.ndarray of str ('core'/'uncertain'/'outlier')
        - 'cluster': np.ndarray of int (cluster assignments)
        - 'max_p': np.ndarray (copy of input)
        - 'log_odds_gap': np.ndarray (copy of input)
        - 'thresholds': dict with threshold values used

    Examples
    --------
    >>> embryo_ids = ['emb_01', 'emb_02', 'emb_03']
    >>> posteriors = analyze_bootstrap_results(bootstrap_results)
    >>> classification = classify_membership_2d(
    ...     posteriors['max_p'],
    ...     posteriors['log_odds_gap'],
    ...     posteriors['modal_cluster'],
    ...     embryo_ids=posteriors['embryo_ids']
    ... )
    """
    n_embryos = len(max_p)
    categories = np.full(n_embryos, 'uncertain', dtype=object)

    for i in range(n_embryos):
        if max_p[i] < threshold_outlier_max_p:
            categories[i] = 'outlier'
        elif max_p[i] >= threshold_max_p and log_odds_gap[i] >= threshold_log_odds_gap:
            categories[i] = 'core'

    result = {
        'category': categories,
        'cluster': modal_cluster.copy(),
        'max_p': max_p.copy(),
        'log_odds_gap': log_odds_gap.copy(),
        'thresholds': {
            'threshold_max_p': threshold_max_p,
            'threshold_log_odds_gap': threshold_log_odds_gap,
            'threshold_outlier_max_p': threshold_outlier_max_p
        }
    }

    if embryo_ids is not None:
        result['embryo_ids'] = embryo_ids

    return result


def classify_membership_adaptive(
    max_p: np.ndarray,
    log_odds_gap: np.ndarray,
    modal_cluster: np.ndarray,
    *,
    base_threshold_max_p: float = THRESHOLD_MAX_P,
    base_threshold_log_odds: float = THRESHOLD_LOG_ODDS_GAP,
    threshold_outlier_max_p: float = THRESHOLD_OUTLIER_MAX_P,
    adaptive_percentile: float = ADAPTIVE_PERCENTILE
) -> Dict[str, Any]:
    """
    Classify membership quality using adaptive per-cluster thresholds.

    Computes thresholds as percentiles of within-cluster distributions,
    allowing looser/tighter standards based on cluster tightness.

    Parameters
    ----------
    max_p : np.ndarray
        Maximum posterior probability per embryo
    log_odds_gap : np.ndarray
        Log-odds gap per embryo
    modal_cluster : np.ndarray
        Cluster assignments
    base_threshold_max_p : float
        Minimum threshold (fallback)
    base_threshold_log_odds : float
        Minimum threshold (fallback)
    threshold_outlier_max_p : float
        Global outlier threshold
    adaptive_percentile : float, default=0.75
        Percentile for computing adaptive thresholds

    Returns
    -------
    classification : dict
        Same as classify_membership_2d() plus:
        - 'cluster_thresholds': dict mapping cluster_id to thresholds
    """
    n_embryos = len(max_p)
    n_clusters = int(np.max(modal_cluster)) + 1
    categories = np.full(n_embryos, 'uncertain', dtype=object)

    # Compute per-cluster thresholds
    cluster_thresholds = {}
    for c in range(n_clusters):
        mask = modal_cluster == c
        if np.sum(mask) > 0:
            p_thresh = max(base_threshold_max_p,
                          np.percentile(max_p[mask], adaptive_percentile))
            gap_thresh = max(base_threshold_log_odds,
                            np.percentile(log_odds_gap[mask], adaptive_percentile))
            cluster_thresholds[c] = {
                'max_p': p_thresh,
                'log_odds_gap': gap_thresh
            }
        else:
            cluster_thresholds[c] = {
                'max_p': base_threshold_max_p,
                'log_odds_gap': base_threshold_log_odds
            }

    # Classify using adaptive thresholds
    for i in range(n_embryos):
        c = modal_cluster[i]
        thresh = cluster_thresholds[c]

        if max_p[i] < threshold_outlier_max_p:
            categories[i] = 'outlier'
        elif max_p[i] >= thresh['max_p'] and log_odds_gap[i] >= thresh['log_odds_gap']:
            categories[i] = 'core'

    result = {
        'category': categories,
        'cluster': modal_cluster.copy(),
        'max_p': max_p.copy(),
        'log_odds_gap': log_odds_gap.copy(),
        'cluster_thresholds': cluster_thresholds,
        'thresholds': {
            'base_threshold_max_p': base_threshold_max_p,
            'base_threshold_log_odds': base_threshold_log_odds,
            'threshold_outlier_max_p': threshold_outlier_max_p,
            'adaptive_percentile': adaptive_percentile
        }
    }

    return result


def get_classification_summary(classification: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute summary statistics for classification results.

    Parameters
    ----------
    classification : dict
        Output from classify_membership_2d() or classify_membership_adaptive()

    Returns
    -------
    summary : dict
        - 'n_core': int
        - 'n_uncertain': int
        - 'n_outlier': int
        - 'core_fraction': float
        - 'uncertain_fraction': float
        - 'outlier_fraction': float
        - 'per_cluster': dict mapping cluster_id to per-cluster stats

    Examples
    --------
    >>> classification = classify_membership_2d(...)
    >>> summary = get_classification_summary(classification)
    >>> print(f"Core: {summary['core_fraction']:.1%}")
    """
    categories = classification['category']
    clusters = classification['cluster']

    n_total = len(categories)
    n_core = np.sum(categories == 'core')
    n_uncertain = np.sum(categories == 'uncertain')
    n_outlier = np.sum(categories == 'outlier')

    summary = {
        'n_core': int(n_core),
        'n_uncertain': int(n_uncertain),
        'n_outlier': int(n_outlier),
        'n_total': int(n_total),
        'core_fraction': float(n_core / n_total) if n_total > 0 else 0.0,
        'uncertain_fraction': float(n_uncertain / n_total) if n_total > 0 else 0.0,
        'outlier_fraction': float(n_outlier / n_total) if n_total > 0 else 0.0,
        'per_cluster': {}
    }

    # Per-cluster statistics
    for c in np.unique(clusters):
        mask = clusters == c
        cat_subset = categories[mask]

        summary['per_cluster'][int(c)] = {
            'n_total': int(np.sum(mask)),
            'n_core': int(np.sum(cat_subset == 'core')),
            'n_uncertain': int(np.sum(cat_subset == 'uncertain')),
            'n_outlier': int(np.sum(cat_subset == 'outlier')),
        }

    return summary
