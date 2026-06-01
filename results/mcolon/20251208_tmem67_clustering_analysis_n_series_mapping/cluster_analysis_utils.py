"""
Utilities for mutant cluster identification.
"""
import numpy as np
import pandas as pd


def compute_cluster_mean_trajectory(df_cluster, time_col='hpf', metric_col='metric_value'):
    """
    Compute mean trajectory for a cluster using binned averaging.

    Parameters
    ----------
    df_cluster : DataFrame
        DataFrame with columns [embryo_id, hpf, metric_value]
    time_col : str
        Name of time column
    metric_col : str
        Name of metric column

    Returns
    -------
    mean_trajectory : dict
        Dictionary with keys 'hpf' and 'mean_value'
    """
    # Group by hpf and compute mean
    grouped = df_cluster.groupby(time_col)[metric_col].mean().reset_index()

    return {
        'hpf': grouped[time_col].values,
        'mean_value': grouped[metric_col].values
    }


def compute_trajectory_average(mean_trajectory):
    """
    Compute average value across all timepoints in a trajectory.

    Parameters
    ----------
    mean_trajectory : dict
        Dictionary with 'hpf' and 'mean_value' arrays

    Returns
    -------
    avg : float
        Mean of mean_value array
    """
    return np.mean(mean_trajectory['mean_value'])


def compute_composite_k_scores(comparison_df):
    """
    Compute composite scores for optimal k selection.

    Parameters
    ----------
    comparison_df : DataFrame
        DataFrame with columns [k, avg_max_p, avg_entropy,
                       core_fraction, silhouette, ...]

    Returns
    -------
    scores : dict
        Dictionary mapping k -> composite_score
    """
    scores = {}

    for _, row in comparison_df.iterrows():
        k = row['k']

        # Normalize metrics to 0-1 scale
        max_p_norm = (row['avg_max_p'] - 0.5) / 0.5  # Assume 0.5-1.0 range
        max_p_norm = np.clip(max_p_norm, 0, 1)

        entropy_norm = 1 - (row['avg_entropy'] / np.log2(k))  # Invert (lower is better)
        entropy_norm = np.clip(entropy_norm, 0, 1)

        core_frac_norm = row['core_fraction']

        silhouette_norm = (row['silhouette'] + 1) / 2  # Map -1,1 to 0,1

        # Weighted combination
        weights = {
            'max_p': 0.30,
            'entropy': 0.25,
            'core_fraction': 0.25,
            'silhouette': 0.20
        }

        score = (
            weights['max_p'] * max_p_norm +
            weights['entropy'] * entropy_norm +
            weights['core_fraction'] * core_frac_norm +
            weights['silhouette'] * silhouette_norm
        )

        scores[k] = score

    return scores
