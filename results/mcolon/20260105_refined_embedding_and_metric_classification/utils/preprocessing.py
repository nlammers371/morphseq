"""Data preprocessing functions for plotting.

Move all data transformations out of plotting functions.
Plotting functions should receive "plot-ready" data.
"""

import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import List, Optional


def prepare_auroc_data(
    classification_df: pd.DataFrame,
    significance_threshold: float = 0.01
) -> pd.DataFrame:
    """Prepare AUROC data for plotting.

    Adds pre-computed significance column and ensures time_bin_center exists.

    Parameters
    ----------
    classification_df : pd.DataFrame
        Raw classification output from compare_groups()
        Must contain: time_bin, auroc_observed, pval
    significance_threshold : float
        P-value threshold used to mark significance (default: 0.01)

    Returns
    -------
    pd.DataFrame
        Plot-ready AUROC data with added columns:
        - is_significant: bool, pval <= significance_threshold
        - time_bin_center: float, preferred x-axis value
    """
    df = classification_df.copy()

    # Add significance flags
    df['is_significant'] = df['pval'] <= significance_threshold

    # Ensure time_bin_center exists (prefer it over time_bin for plotting)
    if 'time_bin_center' not in df.columns:
        if 'bin_width' in df.columns:
            df['time_bin_center'] = df['time_bin'] + df['bin_width'] / 2
        else:
            df['time_bin_center'] = df['time_bin']

    return df


def smooth_divergence(
    divergence_df: pd.DataFrame,
    sigma: float = 1.5,
    time_col: str = 'hpf',
    value_col: str = 'abs_difference',
    groupby_col: str = 'metric'
) -> pd.DataFrame:
    """Apply Gaussian smoothing to divergence data.

    Parameters
    ----------
    divergence_df : pd.DataFrame
        Divergence data with columns: {time_col}, {value_col}, {groupby_col}
    sigma : float
        Gaussian filter sigma (larger = more smoothing)
    time_col : str
        Column name for time axis
    value_col : str
        Column name to smooth (e.g., 'abs_difference', 'abs_difference_zscore')
    groupby_col : str
        Column to group by before smoothing (typically 'metric')

    Returns
    -------
    pd.DataFrame
        Original data with added column: {value_col}_smoothed
    """
    df = divergence_df.copy()
    smoothed_values = []

    for metric_name, group in df.groupby(groupby_col):
        # Sort by time to ensure proper smoothing
        group = group.sort_values(time_col)

        # Apply Gaussian filter
        smoothed = gaussian_filter1d(group[value_col].values, sigma=sigma)

        # Store smoothed values with original index
        for idx, val in zip(group.index, smoothed):
            smoothed_values.append((idx, val))

    # Add smoothed column
    smoothed_col = f'{value_col}_smoothed'
    df[smoothed_col] = np.nan
    for idx, val in smoothed_values:
        df.loc[idx, smoothed_col] = val

    return df


def smooth_trajectories(
    df_trajectories: pd.DataFrame,
    metric_cols: List[str],
    time_col: str = 'predicted_stage_hpf',
    groupby_col: str = 'embryo_id',
    sigma: float = 1.5
) -> pd.DataFrame:
    """Apply Gaussian smoothing to trajectory metrics.

    Parameters
    ----------
    df_trajectories : pd.DataFrame
        Trajectory data with embryo timeseries
    metric_cols : list of str
        Metric columns to smooth
    time_col : str
        Time column for sorting
    groupby_col : str
        Column to group by (typically 'embryo_id')
    sigma : float
        Gaussian filter sigma

    Returns
    -------
    pd.DataFrame
        Original data with added columns: {metric}_smoothed for each metric
    """
    df = df_trajectories.copy()

    for metric in metric_cols:
        if metric not in df.columns:
            continue

        smoothed_col = f'{metric}_smoothed'
        df[smoothed_col] = np.nan

        for embryo_id, group in df.groupby(groupby_col):
            # Sort by time
            group = group.sort_values(time_col)

            # Apply Gaussian filter
            smoothed = gaussian_filter1d(group[metric].values, sigma=sigma)

            # Store smoothed values
            df.loc[group.index, smoothed_col] = smoothed

    return df


def compute_trajectory_statistics(
    df_trajectories: pd.DataFrame,
    metric_col: str,
    time_col: str = 'predicted_stage_hpf',
    groupby_col: str = 'group'
) -> pd.DataFrame:
    """Compute mean and SEM for trajectories per group and time point.

    Parameters
    ----------
    df_trajectories : pd.DataFrame
        Trajectory data
    metric_col : str
        Metric column to compute statistics for
    time_col : str
        Time column
    groupby_col : str
        Group column

    Returns
    -------
    pd.DataFrame
        Statistics with columns: {time_col}, {groupby_col}, mean, sem, count
    """
    stats = df_trajectories.groupby([time_col, groupby_col])[metric_col].agg([
        ('mean', 'mean'),
        ('sem', 'sem'),
        ('count', 'count')
    ]).reset_index()

    return stats


def limit_trajectories_per_group(
    df_trajectories: pd.DataFrame,
    max_embryos_per_group: int = 20,
    group_col: str = 'group',
    embryo_col: str = 'embryo_id',
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """Limit number of embryos per group (for cleaner visualization).

    Parameters
    ----------
    df_trajectories : pd.DataFrame
        Trajectory data
    max_embryos_per_group : int
        Maximum embryos to keep per group
    group_col : str
        Group column
    embryo_col : str
        Embryo ID column
    seed : int, optional
        Random seed for reproducible sampling

    Returns
    -------
    pd.DataFrame
        Filtered trajectory data
    """
    selected_embryos = []

    for group_name, group in df_trajectories.groupby(group_col):
        unique_embryos = group[embryo_col].unique()

        if len(unique_embryos) > max_embryos_per_group:
            # Randomly sample
            rng = np.random.RandomState(seed)
            selected = rng.choice(unique_embryos, size=max_embryos_per_group, replace=False)
        else:
            selected = unique_embryos

        selected_embryos.extend(selected)

    return df_trajectories[df_trajectories[embryo_col].isin(selected_embryos)].copy()


# =============================================================================
# Multiclass Preprocessing Functions
# =============================================================================

def prepare_multiclass_auroc_data(
    ovr_results: dict,
    significance_threshold: float = 0.01
) -> pd.DataFrame:
    """
    Stack per-class OvR results into a single long-format DataFrame.

    Parameters
    ----------
    ovr_results : Dict[str, pd.DataFrame]
        Dictionary mapping class labels to AUROC DataFrames
        Each DataFrame should have: time_bin, auroc_observed, pval, etc.
    significance_threshold : float
        P-value threshold used to mark significance (default: 0.01)

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with added columns:
        - class_label: which class this row represents
        - is_significant: significance flag
    """
    dfs = []

    for class_label, df in ovr_results.items():
        if df is None or df.empty:
            continue

        df_copy = df.copy()
        df_copy['class_label'] = class_label

        # Add significance flags
        df_copy['is_significant'] = df_copy['pval'] <= significance_threshold

        # Ensure time_bin_center exists
        if 'time_bin_center' not in df_copy.columns:
            if 'bin_width' in df_copy.columns:
                df_copy['time_bin_center'] = df_copy['time_bin'] + df_copy['bin_width'] / 2
            else:
                df_copy['time_bin_center'] = df_copy['time_bin']

        dfs.append(df_copy)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def extract_temporal_confusion_profile(
    confusion_matrices: dict,
    class_labels: list
) -> pd.DataFrame:
    """
    Extract temporal confusion profile from confusion matrices.

    For each class and time bin, shows the proportion classified as each class.
    This reveals when phenotypes become distinguishable and which classes they
    get confused with at different times.

    Parameters
    ----------
    confusion_matrices : Dict[int, pd.DataFrame]
        Confusion matrices keyed by time bin
        Each DataFrame has class labels as index (true) and columns (predicted)
    class_labels : List[str]
        List of class labels in order

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        - time_bin: time bin value
        - true_class: the actual class
        - predicted_class: what it was classified as
        - proportion: fraction of true_class classified as predicted_class
        - count: raw count
        - is_correct: whether true_class == predicted_class

    Example
    -------
    >>> profile = extract_temporal_confusion_profile(results['confusion_matrices'], class_labels)
    >>> # Get profile for CE class
    >>> ce_profile = profile[profile['true_class'] == 'CE']
    >>> # See what CE gets confused with at 24 hpf
    >>> ce_24 = ce_profile[ce_profile['time_bin'] == 24]
    """
    records = []

    for time_bin, cm_df in sorted(confusion_matrices.items()):
        for true_class in class_labels:
            if true_class not in cm_df.index:
                continue

            row_sum = cm_df.loc[true_class].sum()

            if not np.isfinite(row_sum) or row_sum == 0:
                continue

            for pred_class in class_labels:
                if pred_class not in cm_df.columns:
                    continue

                count = cm_df.loc[true_class, pred_class]
                proportion = count / row_sum

                records.append({
                    'time_bin': time_bin,
                    'true_class': true_class,
                    'predicted_class': pred_class,
                    'proportion': proportion,
                    'count': count,
                    'is_correct': true_class == pred_class,
                })

    return pd.DataFrame(records)


def compute_per_embryo_accuracy(
    embryo_predictions: pd.DataFrame,
    embryo_id_col: str = 'embryo_id'
) -> pd.DataFrame:
    """
    Compute per-embryo classification accuracy across all time bins.

    Identifies embryos that are systematically misclassified (potentially
    ambiguous phenotypes or labeling errors).

    Parameters
    ----------
    embryo_predictions : pd.DataFrame
        Output from compare_groups_multiclass()['embryo_predictions']
        Must have: embryo_id, is_correct, true_class
    embryo_id_col : str
        Embryo ID column name

    Returns
    -------
    pd.DataFrame
        Per-embryo statistics with columns:
        - embryo_id: unique identifier
        - true_class: the embryo's assigned class
        - n_timepoints: number of time bins with predictions
        - n_correct: number correctly classified
        - accuracy: proportion correct
        - is_ambiguous: True if accuracy < 0.5
    """
    stats = embryo_predictions.groupby([embryo_id_col, 'true_class']).agg({
        'is_correct': ['sum', 'count']
    }).reset_index()

    stats.columns = [embryo_id_col, 'true_class', 'n_correct', 'n_timepoints']
    stats['accuracy'] = stats['n_correct'] / stats['n_timepoints']
    stats['is_ambiguous'] = stats['accuracy'] < 0.5

    return stats.sort_values('accuracy')
