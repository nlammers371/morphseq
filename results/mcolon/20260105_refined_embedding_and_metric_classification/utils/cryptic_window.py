"""Cryptic window detection - identifies embedding-before-metric signal gaps."""
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np


def detect_cryptic_window(
    embedding_auroc: pd.DataFrame,
    metric_divergence: pd.DataFrame,
    auroc_threshold: float = 0.6,
    auroc_pval_threshold: float = 0.05,
    divergence_zscore_threshold: float = 1.0,
    time_col: str = 'time_bin',
) -> Dict[str, Any]:
    """
    Detect time window where embedding signal precedes metric divergence.

    A "cryptic window" exists when embeddings detect a phenotype significantly
    BEFORE standard metrics (curvature/length) show divergence. This indicates
    subtle morphological changes that are captured by VAE embeddings but not
    by simple measurements.

    Parameters
    ----------
    embedding_auroc : pd.DataFrame
        Output from compare_groups()['classification'] using embedding features.
        Must have columns: time_col, 'auroc_observed', 'pval'
    metric_divergence : pd.DataFrame
        Output from compute_multi_metric_divergence() with zscore normalization.
        Must have columns: 'hpf', 'abs_difference_zscore', 'metric'
    auroc_threshold : float
        AUROC value to consider "significant signal" (default: 0.6)
    auroc_pval_threshold : float
        p-value threshold for AUROC significance (default: 0.05)
    divergence_zscore_threshold : float
        Z-score threshold for metric divergence (default: 1.0 = 1 SD above mean)
    time_col : str
        Time column name in embedding_auroc (default: 'time_bin')

    Returns
    -------
    Dict with keys:
        - has_cryptic_window: bool - Whether embedding detects before metrics
        - embedding_first_signal_hpf: float or None - First significant embedding time
        - metric_first_divergence_hpf: float or None - First significant metric time
        - cryptic_window_duration_hours: float or None - Gap between signals
        - thresholds: Dict - The thresholds used
        - per_metric_details: Dict - Per-metric breakdown of first detection times

    Example
    -------
    >>> cryptic = detect_cryptic_window(
    ...     embedding_auroc,
    ...     divergence,
    ...     auroc_threshold=0.6,
    ...     divergence_zscore_threshold=1.0,
    ... )
    >>> if cryptic['has_cryptic_window']:
    ...     print(f"Cryptic window: {cryptic['cryptic_window_duration_hours']} hours")
    """
    # Find first significant embedding signal
    sig_embedding = embedding_auroc[
        (embedding_auroc['auroc_observed'] > auroc_threshold) &
        (embedding_auroc['pval'] < auroc_pval_threshold)
    ]
    emb_first = sig_embedding[time_col].min() if len(sig_embedding) > 0 else None

    # Find first significant metric divergence (any metric)
    sig_metric = metric_divergence[
        metric_divergence['abs_difference_zscore'] > divergence_zscore_threshold
    ]
    metric_first = sig_metric['hpf'].min() if len(sig_metric) > 0 else None

    # Per-metric breakdown
    metric_details = {}
    for metric in metric_divergence['metric'].unique():
        metric_data = metric_divergence[metric_divergence['metric'] == metric]
        sig = metric_data[metric_data['abs_difference_zscore'] > divergence_zscore_threshold]
        metric_details[metric] = {
            'first_signal_hpf': float(sig['hpf'].min()) if len(sig) > 0 else None
        }

    # Determine cryptic window
    has_window = False
    duration = None
    if emb_first is not None and metric_first is not None:
        has_window = emb_first < metric_first
        duration = float(metric_first - emb_first) if has_window else 0.0

    return {
        'has_cryptic_window': has_window,
        'embedding_first_signal_hpf': float(emb_first) if emb_first is not None else None,
        'metric_first_divergence_hpf': float(metric_first) if metric_first is not None else None,
        'cryptic_window_duration_hours': duration,
        'thresholds': {
            'auroc': auroc_threshold,
            'auroc_pval': auroc_pval_threshold,
            'divergence_zscore': divergence_zscore_threshold,
        },
        'per_metric_details': metric_details,
    }


def detect_cryptic_window_from_aurocs(
    embedding_auroc: pd.DataFrame,
    metric_auroc: pd.DataFrame,
    auroc_threshold: float = 0.6,
    auroc_pval_threshold: float = 0.05,
    time_col: str = 'time_bin',
) -> Dict[str, Any]:
    """
    Detect cryptic window by comparing embedding vs metric AUROC directly.

    Alternative to detect_cryptic_window() when you have AUROC for both
    embeddings and metrics (instead of divergence z-scores).

    Parameters
    ----------
    embedding_auroc : pd.DataFrame
        AUROC from compare_groups() using embedding features
    metric_auroc : pd.DataFrame
        AUROC from compare_groups() using metric features
    auroc_threshold : float
        AUROC threshold for significance (default: 0.6)
    auroc_pval_threshold : float
        p-value threshold (default: 0.05)
    time_col : str
        Time column name (default: 'time_bin')

    Returns
    -------
    Dict with same structure as detect_cryptic_window()
    """
    # Find first significant embedding signal
    sig_embedding = embedding_auroc[
        (embedding_auroc['auroc_observed'] > auroc_threshold) &
        (embedding_auroc['pval'] < auroc_pval_threshold)
    ]
    emb_first = sig_embedding[time_col].min() if len(sig_embedding) > 0 else None

    # Find first significant metric signal
    sig_metric = metric_auroc[
        (metric_auroc['auroc_observed'] > auroc_threshold) &
        (metric_auroc['pval'] < auroc_pval_threshold)
    ]
    metric_first = sig_metric[time_col].min() if len(sig_metric) > 0 else None

    # Determine cryptic window
    has_window = False
    duration = None
    if emb_first is not None and metric_first is not None:
        has_window = emb_first < metric_first
        duration = float(metric_first - emb_first) if has_window else 0.0

    return {
        'has_cryptic_window': has_window,
        'embedding_first_signal_hpf': float(emb_first) if emb_first is not None else None,
        'metric_first_divergence_hpf': float(metric_first) if metric_first is not None else None,
        'cryptic_window_duration_hours': duration,
        'thresholds': {
            'auroc': auroc_threshold,
            'auroc_pval': auroc_pval_threshold,
        },
        'per_metric_details': {},
    }


def summarize_cryptic_windows(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Summarize cryptic window results across multiple comparisons.

    Parameters
    ----------
    results : Dict[str, Dict]
        Mapping of comparison name to detect_cryptic_window() output

    Returns
    -------
    pd.DataFrame
        Summary table with one row per comparison:
        - comparison: Comparison name
        - has_cryptic_window: bool
        - embedding_first_hpf: First significant embedding time
        - metric_first_hpf: First significant metric time
        - window_hours: Duration of cryptic window

    Example
    -------
    >>> results = {
    ...     'CE_vs_WT': detect_cryptic_window(emb_auroc, divergence),
    ...     'CE_vs_hets': detect_cryptic_window(emb_auroc2, divergence2),
    ... }
    >>> summary = summarize_cryptic_windows(results)
    >>> summary.to_csv('cryptic_window_summary.csv', index=False)
    """
    rows = []
    for name, cw in results.items():
        rows.append({
            'comparison': name,
            'has_cryptic_window': cw['has_cryptic_window'],
            'embedding_first_hpf': cw['embedding_first_signal_hpf'],
            'metric_first_hpf': cw['metric_first_divergence_hpf'],
            'window_hours': cw['cryptic_window_duration_hours'],
        })
    return pd.DataFrame(rows)
