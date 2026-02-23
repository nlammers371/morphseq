"""
AUROC-based comparison of distance metrics for WT/mutant separation.

This module treats distance-to-WT as a 1D binary classifier:
- Small distance → WT-like (class 0)
- Large distance → mutant-like (class 1)

Then compares multiple distance metrics (Diagonal Mahalanobis, Euclidean)
using standard classification metrics (AUROC, PR-AUC, balanced accuracy).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_curve,
    precision_recall_curve
)
from multiprocessing import Pool, cpu_count


def compute_distance_auroc(
    distances: np.ndarray,
    labels: np.ndarray,
    positive_class: str = 'mutant'
) -> Dict[str, float]:
    """
    Compute classification metrics treating distance as a 1D classifier score.

    Parameters
    ----------
    distances : np.ndarray
        Distance values (higher = more divergent from WT)
    labels : np.ndarray
        Binary labels: 0 for WT, 1 for mutant (or genotype strings)
    positive_class : str, default='mutant'
        Which class should have higher distance (usually mutant)

    Returns
    -------
    dict
        Metrics including:
        - 'auroc': Area under ROC curve
        - 'ap': Average precision (PR-AUC)
        - 'balanced_accuracy': Balanced accuracy at optimal threshold
        - 'optimal_threshold': Distance threshold that maximizes balanced accuracy
        - 'n_wt': Number of WT samples
        - 'n_mutant': Number of mutant samples

    Notes
    -----
    AUROC interpretation:
    - 0.5: Random (no separation)
    - 0.7-0.8: Fair separation
    - 0.8-0.9: Good separation
    - 0.9+: Excellent separation
    """
    # Convert to numpy array if pandas Series
    if hasattr(labels, 'values'):
        labels = labels.values

    # Convert string labels to binary if needed
    unique_labels = np.unique(labels)
    if len(unique_labels) != 2:
        raise ValueError(f"Expected exactly 2 classes, got {len(unique_labels)}: {unique_labels}")

    # Create binary labels: 1 for mutant (positive class), 0 for WT
    if labels.dtype == object or isinstance(labels[0], str):
        # Assume WT contains 'wildtype' or 'wik' or 'ab'
        is_wt = np.array([
            'wildtype' in str(lbl).lower() or
            str(lbl).lower() in ['wik', 'ab', 'wik-ab']
            for lbl in labels
        ])
        y_true = (~is_wt).astype(int)  # 1 for mutant, 0 for WT
    else:
        y_true = labels.astype(int)

    # Distances are the "scores" - higher distance = more likely mutant
    y_score = distances

    # Compute AUROC
    auroc = roc_auc_score(y_true, y_score)

    # Compute PR-AUC (precision-recall)
    ap = average_precision_score(y_true, y_score)

    # Find optimal threshold by maximizing balanced accuracy
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    balanced_accs = []
    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)
        balanced_accs.append(balanced_accuracy_score(y_true, y_pred))

    optimal_idx = np.argmax(balanced_accs)
    optimal_threshold = thresholds[optimal_idx]
    optimal_balanced_acc = balanced_accs[optimal_idx]

    return {
        'auroc': auroc,
        'ap': ap,
        'balanced_accuracy': optimal_balanced_acc,
        'optimal_threshold': optimal_threshold,
        'n_wt': np.sum(y_true == 0),
        'n_mutant': np.sum(y_true == 1)
    }


def compare_distance_metrics(
    distance_dict: Dict[str, np.ndarray],
    labels: np.ndarray
) -> pd.DataFrame:
    """
    Compare multiple distance metrics head-to-head.

    Parameters
    ----------
    distance_dict : dict
        Dictionary mapping metric name -> distance array
        Example: {'diagonal_mahalanobis': array(...), 'euclidean': array(...)}
    labels : np.ndarray
        Binary labels (0=WT, 1=mutant)

    Returns
    -------
    pd.DataFrame
        Comparison table with columns:
        - metric: Distance metric name
        - auroc, ap, balanced_accuracy, optimal_threshold
        - n_wt, n_mutant
    """
    results = []

    for metric_name, distances in distance_dict.items():
        metrics = compute_distance_auroc(distances, labels)
        metrics['metric'] = metric_name
        results.append(metrics)

    df_comparison = pd.DataFrame(results)

    # Sort by AUROC descending
    df_comparison = df_comparison.sort_values('auroc', ascending=False).reset_index(drop=True)

    return df_comparison


def _bootstrap_single_iteration(args):
    """Helper function for parallel bootstrap - single iteration."""
    distances_a, distances_b, labels, metric, seed = args
    rng = np.random.default_rng(seed)
    n_samples = len(labels)

    # Resample with replacement
    indices = rng.choice(n_samples, size=n_samples, replace=True)

    dist_a_boot = distances_a[indices]
    dist_b_boot = distances_b[indices]
    labels_boot = labels[indices]

    # Ensure we have both classes
    if len(np.unique(labels_boot)) != 2:
        return None

    # Compute metrics
    try:
        metrics_a = compute_distance_auroc(dist_a_boot, labels_boot)
        metrics_b = compute_distance_auroc(dist_b_boot, labels_boot)
        diff = metrics_a[metric] - metrics_b[metric]
        return diff
    except:
        return None


def bootstrap_paired_difference(
    distances_a: np.ndarray,
    distances_b: np.ndarray,
    labels: np.ndarray,
    metric: str = 'auroc',
    n_bootstrap: int = 100,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None,
    verbose: bool = True,
    n_jobs: int = -1
) -> Dict[str, float]:
    """
    Bootstrap confidence interval for paired difference in classification metrics.

    Tests: metric(distances_a) - metric(distances_b)

    Parameters
    ----------
    distances_a : np.ndarray
        First distance metric
    distances_b : np.ndarray
        Second distance metric
    labels : np.ndarray
        Binary labels
    metric : str, default='auroc'
        Metric to compare: 'auroc', 'ap', or 'balanced_accuracy'
    n_bootstrap : int, default=100
        Number of bootstrap samples
    confidence_level : float, default=0.95
        Confidence level for CI
    random_state : int, optional
        Random seed
    verbose : bool, default=True
        Print progress indicators
    n_jobs : int, default=-1
        Number of parallel jobs. -1 uses all available CPUs

    Returns
    -------
    dict
        - 'mean_diff': Mean difference across bootstraps
        - 'ci_lower': Lower bound of CI
        - 'ci_upper': Upper bound of CI
        - 'p_value': Two-sided p-value (fraction of bootstraps where diff crosses 0)
    """
    # Convert to numpy arrays if needed
    if hasattr(distances_a, 'values'):
        distances_a = distances_a.values
    if hasattr(distances_b, 'values'):
        distances_b = distances_b.values
    if hasattr(labels, 'values'):
        labels = labels.values

    # Determine number of CPUs to use
    if n_jobs == -1:
        n_jobs = cpu_count()
    n_jobs = min(n_jobs, cpu_count())

    if verbose:
        print(f"  Running {n_bootstrap} bootstrap iterations using {n_jobs} CPUs...")

    # Generate random seeds for each bootstrap iteration
    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, 2**32, size=n_bootstrap)

    # Prepare arguments for parallel processing
    args_list = [
        (distances_a, distances_b, labels, metric, seed)
        for seed in seeds
    ]

    # Run bootstrap in parallel
    with Pool(processes=n_jobs) as pool:
        results = pool.map(_bootstrap_single_iteration, args_list)

    # Filter out None results
    differences = np.array([r for r in results if r is not None])

    if verbose:
        print(f"    Bootstrap: {n_bootstrap}/{n_bootstrap} (100%) - Done! ({len(differences)} successful iterations)")

    # Compute statistics
    mean_diff = np.mean(differences)
    alpha = 1 - confidence_level
    ci_lower = np.percentile(differences, 100 * alpha / 2)
    ci_upper = np.percentile(differences, 100 * (1 - alpha / 2))

    # Two-sided p-value: fraction of bootstraps where CI crosses 0
    p_value = np.sum(np.sign(differences) != np.sign(mean_diff)) / len(differences)

    return {
        'mean_diff': mean_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value,
        'n_bootstrap': len(differences)
    }


def compute_all_classification_metrics(
    distance_dict: Dict[str, np.ndarray],
    labels: np.ndarray
) -> pd.DataFrame:
    """
    Compute comprehensive classification metrics for all distance types.

    Parameters
    ----------
    distance_dict : dict
        Dictionary mapping metric name -> distance array
    labels : np.ndarray
        Binary labels

    Returns
    -------
    pd.DataFrame
        Metrics table with columns:
        - metric, auroc, ap, balanced_accuracy, optimal_threshold, n_wt, n_mutant
    """
    return compare_distance_metrics(distance_dict, labels)


def compute_roc_curves(
    distance_dict: Dict[str, np.ndarray],
    labels: np.ndarray
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute ROC curves for all distance metrics.

    Parameters
    ----------
    distance_dict : dict
        Dictionary mapping metric name -> distance array
    labels : np.ndarray
        Binary labels

    Returns
    -------
    dict
        Dictionary mapping metric name -> (fpr, tpr, thresholds)
    """
    # Convert to numpy array if pandas Series
    if hasattr(labels, 'values'):
        labels = labels.values

    # Convert to binary labels
    unique_labels = np.unique(labels)
    if labels.dtype == object or isinstance(labels[0], str):
        is_wt = np.array([
            'wildtype' in str(lbl).lower() or
            str(lbl).lower() in ['wik', 'ab', 'wik-ab']
            for lbl in labels
        ])
        y_true = (~is_wt).astype(int)
    else:
        y_true = labels.astype(int)

    roc_curves = {}
    for metric_name, distances in distance_dict.items():
        fpr, tpr, thresholds = roc_curve(y_true, distances)
        roc_curves[metric_name] = (fpr, tpr, thresholds)

    return roc_curves


def aggregate_across_time_bins(
    results_per_bin: pd.DataFrame,
    metric_col: str = 'auroc',
    aggregation: str = 'median'
) -> Dict[str, float]:
    """
    Aggregate metric across time bins.

    Parameters
    ----------
    results_per_bin : pd.DataFrame
        Results per time bin (from per-bin analysis)
    metric_col : str, default='auroc'
        Column to aggregate
    aggregation : str, default='median'
        Aggregation method: 'median', 'mean', 'max'

    Returns
    -------
    dict
        Aggregated statistics:
        - '{aggregation}': Central tendency
        - 'std': Standard deviation
        - 'min': Minimum
        - 'max': Maximum
        - 'n_bins': Number of time bins
    """
    values = results_per_bin[metric_col].values

    if aggregation == 'median':
        central = np.median(values)
    elif aggregation == 'mean':
        central = np.mean(values)
    elif aggregation == 'max':
        central = np.max(values)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    return {
        aggregation: central,
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'n_bins': len(values)
    }
