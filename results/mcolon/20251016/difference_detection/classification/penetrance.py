"""
Embryo-level penetrance analysis.

This module computes per-embryo penetrance metrics that quantify how consistently
each embryo expresses a classifiable phenotype across developmental time.
"""

import numpy as np
import pandas as pd
from typing import Optional, List


def compute_embryo_penetrance(
    df_embryo_probs: pd.DataFrame,
    confidence_threshold: float = 0.1,
    penetrance_bins: Optional[List[float]] = None,
    penetrance_labels: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute per-embryo penetrance metrics from prediction probabilities.

    This quantifies how consistently each embryo expresses a classifiable
    phenotype across developmental time.

    Parameters
    ----------
    df_embryo_probs : pd.DataFrame
        Per-embryo predictions from predictive_signal_test().
        Must have columns: embryo_id, time_bin, true_label, confidence,
        predicted_label, support_true, signed_margin, mutant_class, and
        per-class probability columns (e.g., pred_proba_wildtype, pred_proba_mutant)
    confidence_threshold : float, default=0.1
        Minimum confidence (|p - 0.5|) to consider a prediction "confident"
    penetrance_bins : list of float, optional
        Bin edges for categorizing penetrance levels.
        Default: [0, 0.1, 0.2, 0.5]
    penetrance_labels : list of str, optional
        Labels for penetrance categories.
        Default: ['low', 'medium', 'high']

    Returns
    -------
    pd.DataFrame
        One row per embryo with penetrance metrics:

        - embryo_id : str
            Unique embryo identifier
        - true_label : str
            True genotype label
        - mean_confidence : float
            Average prediction confidence magnitude across time
        - mean_support_true : float
            Average probability assigned to the true class
        - mean_signed_margin : float
            Average signed margin relative to 0.5 decision boundary
        - temporal_consistency : float
            Fraction of time bins correctly classified
        - max_confidence : float
            Peak confidence magnitude across development
        - min_support_true : float
            Lowest probability assigned to the true class
        - min_signed_margin : float
            Most negative margin (worst wrong-side confidence)
        - first_confident_time : float
            First time bin with confidence > threshold
        - n_time_bins : int
            Number of time bins embryo was observed
        - mean_pred_prob : float
            Average predicted probability for positive class
        - penetrance_category : str
            Categorical penetrance level (low/medium/high)

    Examples
    --------
    >>> df_pen = compute_embryo_penetrance(df_embryo_probs)
    >>> df_pen['penetrance_category'].value_counts()
    high      25
    medium    15
    low       10
    """
    if df_embryo_probs.empty:
        return pd.DataFrame()

    # Set default bins and labels if not provided
    if penetrance_bins is None:
        penetrance_bins = [0, 0.1, 0.2, 0.5]
    if penetrance_labels is None:
        penetrance_labels = ['low', 'medium', 'high']

    penetrance_metrics = []

    for embryo_id, grp in df_embryo_probs.groupby('embryo_id'):
        # Sort by time
        grp = grp.sort_values('time_bin')

        # Compute metrics (guard against missing columns during experimentation)
        mean_conf = grp['confidence'].mean()
        max_conf = grp['confidence'].max()
        n_bins = len(grp)

        # Handle optional columns
        mean_support_true = grp['support_true'].mean() if 'support_true' in grp.columns else np.nan
        min_support_true = grp['support_true'].min() if 'support_true' in grp.columns else np.nan
        mean_signed_margin = grp['signed_margin'].mean() if 'signed_margin' in grp.columns else np.nan
        min_signed_margin = grp['signed_margin'].min() if 'signed_margin' in grp.columns else np.nan

        # Temporal consistency: fraction correctly classified
        correct = (grp['true_label'] == grp['predicted_label']).sum()
        temporal_consistency = correct / n_bins if n_bins > 0 else 0.0

        # First confident prediction time
        confident_bins = grp[grp['confidence'] > confidence_threshold]
        first_confident_time = confident_bins['time_bin'].min() if len(confident_bins) > 0 else np.nan

        # Get true label (should be constant per embryo)
        true_label = grp['true_label'].iloc[0]

        prob_cols = [c for c in grp.columns if c.startswith('pred_proba_')]

        mutant_prob_col = None
        if 'mutant_class' in grp.columns:
            mutant_vals = grp['mutant_class'].dropna().unique()
            if len(mutant_vals) == 1:
                candidate = f"pred_proba_{str(mutant_vals[0])}"
                if candidate in grp.columns:
                    mutant_prob_col = candidate

        if mutant_prob_col is None and len(prob_cols) == 1:
            mutant_prob_col = prob_cols[0]

        mean_pred_prob = grp[mutant_prob_col].mean() if mutant_prob_col else np.nan

        penetrance_metrics.append({
            'embryo_id': embryo_id,
            'true_label': true_label,
            'mean_confidence': mean_conf,
            'mean_support_true': mean_support_true,
            'mean_signed_margin': mean_signed_margin,
            'temporal_consistency': temporal_consistency,
            'max_confidence': max_conf,
            'min_support_true': min_support_true,
            'min_signed_margin': min_signed_margin,
            'first_confident_time': first_confident_time,
            'n_time_bins': n_bins,
            'mean_pred_prob': mean_pred_prob
        })

    df_penetrance = pd.DataFrame(penetrance_metrics)

    # Classify embryos by penetrance level
    df_penetrance['penetrance_category'] = pd.cut(
        df_penetrance['mean_confidence'],
        bins=penetrance_bins,
        labels=penetrance_labels,
        include_lowest=True
    )

    return df_penetrance


def summarize_penetrance(
    df_penetrance: pd.DataFrame,
    group_col: str = 'true_label'
) -> pd.DataFrame:
    """
    Generate summary statistics for penetrance metrics by group.

    Parameters
    ----------
    df_penetrance : pd.DataFrame
        Output from compute_embryo_penetrance()
    group_col : str, default='true_label'
        Column to group by (e.g., genotype)

    Returns
    -------
    pd.DataFrame
        Summary statistics per group

    Examples
    --------
    >>> summary = summarize_penetrance(df_penetrance)
    >>> print(summary)
    """
    if df_penetrance.empty:
        return pd.DataFrame()

    metrics = [
        'mean_confidence',
        'mean_support_true',
        'mean_signed_margin',
        'temporal_consistency'
    ]

    # Filter to available metrics
    available_metrics = [m for m in metrics if m in df_penetrance.columns]

    summary = (
        df_penetrance.groupby(group_col)[available_metrics]
        .agg(['mean', 'std', 'median', 'min', 'max'])
    )

    return summary


def get_high_penetrance_embryos(
    df_penetrance: pd.DataFrame,
    threshold: float = 0.2,
    metric: str = 'mean_confidence'
) -> pd.DataFrame:
    """
    Filter to high-penetrance embryos above a threshold.

    Parameters
    ----------
    df_penetrance : pd.DataFrame
        Output from compute_embryo_penetrance()
    threshold : float, default=0.2
        Minimum value for penetrance metric
    metric : str, default='mean_confidence'
        Metric to filter on

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with high-penetrance embryos
    """
    if metric not in df_penetrance.columns:
        raise ValueError(f"Metric '{metric}' not found in penetrance data")

    return df_penetrance[df_penetrance[metric] >= threshold].copy()
