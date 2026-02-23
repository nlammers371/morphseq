"""
Correlation analysis for incomplete penetrance quantification.

This module computes correlations between morphological distance (from WT)
and classifier-based mutant probability to assess penetrance in homozygous mutants.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from typing import Dict, Tuple, Optional


def compute_per_embryo_metrics(
    df_distances: pd.DataFrame,
    df_predictions: pd.DataFrame,
    genotype: str,
    distance_col: str = 'euclidean_distance',
    prob_col: str = 'pred_proba'
) -> pd.DataFrame:
    """
    Aggregate per-embryo metrics: mean distance and mean predicted probability.

    Parameters
    ----------
    df_distances : pd.DataFrame
        Distance data with columns: embryo_id, time_bin, genotype, euclidean_distance
    df_predictions : pd.DataFrame
        Classifier predictions with columns: embryo_id, time_bin, pred_proba
    genotype : str
        Genotype to filter (e.g., 'cep290_homozygous')
    distance_col : str
        Column name for distance metric
    prob_col : str
        Column name for predicted probability

    Returns
    -------
    pd.DataFrame
        Per-embryo aggregated metrics with columns:
        - embryo_id
        - mean_distance: Mean distance across all timepoints
        - mean_prob: Mean predicted mutant probability
        - n_timepoints: Number of timepoints per embryo
        - std_distance: Std of distance
        - std_prob: Std of probability
    """
    # Filter to specific genotype
    df_dist_filtered = df_distances[df_distances['genotype'] == genotype].copy()

    # Merge distances with predictions
    df_merged = df_dist_filtered.merge(
        df_predictions[['embryo_id', 'time_bin', prob_col]],
        on=['embryo_id', 'time_bin'],
        how='inner'
    )

    # Aggregate per embryo
    embryo_metrics = df_merged.groupby('embryo_id').agg({
        distance_col: ['mean', 'std', 'count'],
        prob_col: ['mean', 'std']
    }).reset_index()

    # Flatten column names
    embryo_metrics.columns = [
        'embryo_id',
        'mean_distance',
        'std_distance',
        'n_timepoints',
        'mean_prob',
        'std_prob'
    ]

    return embryo_metrics


def compute_correlation_statistics(
    embryo_metrics: pd.DataFrame,
    distance_col: str = 'mean_distance',
    prob_col: str = 'mean_prob'
) -> Dict[str, float]:
    """
    Compute correlation statistics between distance and probability.

    Parameters
    ----------
    embryo_metrics : pd.DataFrame
        Per-embryo metrics from compute_per_embryo_metrics()
    distance_col : str
        Column name for distance
    prob_col : str
        Column name for probability

    Returns
    -------
    dict
        Statistics including:
        - n_embryos: Number of embryos
        - pearson_r: Pearson correlation coefficient
        - pearson_p: Pearson p-value
        - spearman_rho: Spearman rank correlation
        - spearman_p: Spearman p-value
        - mean_distance: Mean of distances
        - mean_prob: Mean of probabilities
        - std_distance: Std of distances
        - std_prob: Std of probabilities
    """
    distances = embryo_metrics[distance_col].values
    probs = embryo_metrics[prob_col].values

    # Remove any NaN values
    valid_mask = ~(np.isnan(distances) | np.isnan(probs))
    distances = distances[valid_mask]
    probs = probs[valid_mask]

    # Compute correlations
    pearson_r, pearson_p = pearsonr(distances, probs)
    spearman_rho, spearman_p = spearmanr(distances, probs)

    return {
        'n_embryos': len(distances),
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_rho': spearman_rho,
        'spearman_p': spearman_p,
        'mean_distance': np.mean(distances),
        'mean_prob': np.mean(probs),
        'std_distance': np.std(distances),
        'std_prob': np.std(probs)
    }


def bootstrap_correlation_ci(
    embryo_metrics: pd.DataFrame,
    distance_col: str = 'mean_distance',
    prob_col: str = 'mean_prob',
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None
) -> Dict[str, Tuple[float, float]]:
    """
    Compute bootstrap confidence intervals for correlation coefficients.

    Parameters
    ----------
    embryo_metrics : pd.DataFrame
        Per-embryo metrics
    distance_col : str
        Column name for distance
    prob_col : str
        Column name for probability
    n_bootstrap : int
        Number of bootstrap iterations
    confidence_level : float
        Confidence level for CI
    random_state : int, optional
        Random seed

    Returns
    -------
    dict
        Bootstrap CIs:
        - pearson_ci: (lower, upper)
        - spearman_ci: (lower, upper)
    """
    rng = np.random.default_rng(random_state)

    distances = embryo_metrics[distance_col].values
    probs = embryo_metrics[prob_col].values

    # Remove NaNs
    valid_mask = ~(np.isnan(distances) | np.isnan(probs))
    distances = distances[valid_mask]
    probs = probs[valid_mask]

    n_embryos = len(distances)

    pearson_rs = []
    spearman_rhos = []

    for _ in range(n_bootstrap):
        # Resample embryos with replacement
        indices = rng.choice(n_embryos, size=n_embryos, replace=True)
        boot_dist = distances[indices]
        boot_prob = probs[indices]

        # Compute correlations
        r, _ = pearsonr(boot_dist, boot_prob)
        rho, _ = spearmanr(boot_dist, boot_prob)

        pearson_rs.append(r)
        spearman_rhos.append(rho)

    # Compute CIs
    alpha = 1 - confidence_level
    pearson_ci = (
        np.percentile(pearson_rs, 100 * alpha / 2),
        np.percentile(pearson_rs, 100 * (1 - alpha / 2))
    )
    spearman_ci = (
        np.percentile(spearman_rhos, 100 * alpha / 2),
        np.percentile(spearman_rhos, 100 * (1 - alpha / 2))
    )

    return {
        'pearson_ci': pearson_ci,
        'spearman_ci': spearman_ci
    }


def correlation_by_time_bin(
    df_distances: pd.DataFrame,
    df_predictions: pd.DataFrame,
    genotype: str,
    distance_col: str = 'euclidean_distance',
    prob_col: str = 'pred_proba',
    min_samples: int = 10
) -> pd.DataFrame:
    """
    Compute correlation separately for each time bin.

    Useful for assessing temporal dynamics of correlation strength.

    Parameters
    ----------
    df_distances : pd.DataFrame
        Distance data
    df_predictions : pd.DataFrame
        Classifier predictions
    genotype : str
        Genotype to filter
    distance_col : str
        Distance column name
    prob_col : str
        Probability column name
    min_samples : int
        Minimum samples per time bin

    Returns
    -------
    pd.DataFrame
        Correlation per time bin with columns:
        - time_bin
        - n_embryos
        - pearson_r, pearson_p
        - spearman_rho, spearman_p
    """
    # Filter to genotype
    df_dist_filtered = df_distances[df_distances['genotype'] == genotype].copy()

    # Merge
    df_merged = df_dist_filtered.merge(
        df_predictions[['embryo_id', 'time_bin', prob_col]],
        on=['embryo_id', 'time_bin'],
        how='inner'
    )

    results = []

    for time_bin, group in df_merged.groupby('time_bin'):
        if len(group) < min_samples:
            continue

        distances = group[distance_col].values
        probs = group[prob_col].values

        # Remove NaNs
        valid_mask = ~(np.isnan(distances) | np.isnan(probs))
        distances = distances[valid_mask]
        probs = probs[valid_mask]

        if len(distances) < min_samples:
            continue

        # Compute correlations
        pearson_r, pearson_p = pearsonr(distances, probs)
        spearman_rho, spearman_p = spearmanr(distances, probs)

        results.append({
            'time_bin': time_bin,
            'n_embryos': len(distances),
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_rho': spearman_rho,
            'spearman_p': spearman_p
        })

    return pd.DataFrame(results)
