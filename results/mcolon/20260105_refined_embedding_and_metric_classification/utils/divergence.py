"""
Multi-metric divergence computation - wraps comparison.compute_metric_divergence().

CAUTION
-------
This module should remain a thin loop over the public divergence API. Avoid
adding layers whose main function is renaming/reshaping outputs; that tends to
drift and create maintenance burden across analyses.
"""
import sys
from pathlib import Path
from typing import List
import pandas as pd

# Add src to path for analyze.* imports
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from analyze.difference_detection.comparison import compute_metric_divergence


def compute_multi_metric_divergence(
    df: pd.DataFrame,
    group_col: str,
    group1_label: str,
    group2_label: str,
    metric_cols: List[str],
    time_col: str = 'predicted_stage_hpf',
    embryo_id_col: str = 'embryo_id',
) -> pd.DataFrame:
    """
    Compute trajectory divergence for multiple metrics.

    Loops compute_metric_divergence() for each metric and concatenates results
    into a single long-format DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Data with group column and metric columns
    group_col : str
        Column containing group labels
    group1_label : str
        Label for group1 (positive/phenotype class)
    group2_label : str
        Label for group2 (negative/reference class)
    metric_cols : List[str]
        Metrics to compute divergence for (e.g., ['baseline_deviation_normalized', 'total_length_um'])
    time_col : str
        Time column name (default: 'predicted_stage_hpf')
    embryo_id_col : str
        Embryo ID column name (default: 'embryo_id')

    Returns
    -------
    pd.DataFrame
        Long-format divergence with columns:
        - hpf: Time point
        - group1_mean, group1_sem: Group1 statistics
        - group2_mean, group2_sem: Group2 statistics
        - abs_difference: |group2_mean - group1_mean|
        - n_group1, n_group2: Sample sizes
        - metric: Name of the metric

    Example
    -------
    >>> divergence = compute_multi_metric_divergence(
    ...     df_prep,
    ...     group_col='group',
    ...     group1_label='CE',
    ...     group2_label='WT',
    ...     metric_cols=['baseline_deviation_normalized', 'total_length_um'],
    ... )
    """
    dfs = []
    for metric in metric_cols:
        # Check if metric exists in dataframe
        if metric not in df.columns:
            print(f"Warning: metric '{metric}' not found in dataframe, skipping")
            continue

        div = compute_metric_divergence(
            df,
            group_col,
            group1_label,
            group2_label,
            metric,
            time_col,
            embryo_id_col
        )
        div['metric'] = metric
        dfs.append(div)

    if not dfs:
        raise ValueError(f"No valid metrics found. Available columns: {df.columns.tolist()}")

    return pd.concat(dfs, ignore_index=True)


def zscore_divergence(
    divergence_df: pd.DataFrame,
    value_col: str = 'abs_difference'
) -> pd.DataFrame:
    """
    Z-score normalize divergence within each metric for multi-metric comparison.

    This allows plotting multiple metrics with different scales on the same axis.

    Parameters
    ----------
    divergence_df : pd.DataFrame
        Output from compute_multi_metric_divergence()
    value_col : str
        Column to normalize (default: 'abs_difference')

    Returns
    -------
    pd.DataFrame
        Same DataFrame with '{value_col}_zscore' column added

    Example
    -------
    >>> divergence = compute_multi_metric_divergence(...)
    >>> divergence = zscore_divergence(divergence)
    >>> # Now plot 'abs_difference_zscore' for all metrics on same axis
    """
    def zscore(x):
        std = x.std()
        if std == 0 or pd.isna(std):
            return x - x.mean()  # Return centered values if no variance
        return (x - x.mean()) / std

    divergence_df = divergence_df.copy()
    divergence_df[f'{value_col}_zscore'] = divergence_df.groupby('metric')[value_col].transform(zscore)
    return divergence_df
