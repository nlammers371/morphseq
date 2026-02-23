"""
Correlation Analysis

Statistical tests for temporal patterns in trajectories.

Functions
---------
- test_anticorrelation: Test for anticorrelation between early and late patterns
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from typing import Tuple, Dict, Any


def test_anticorrelation(
    df: pd.DataFrame,
    early_window: Tuple[float, float],
    late_window: Tuple[float, float],
    metric_col: str = 'normalized_baseline_deviation',
    time_col: str = 'predicted_stage_hpf',
    embryo_id_col: str = 'embryo_id',
    n_permutations: int = 10000,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Test for anticorrelation between early and late temporal patterns.

    Uses Pearson correlation + permutation testing.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format data with embryo trajectories
    early_window : tuple of float
        (start, end) time window for early pattern
    late_window : tuple of float
        (start, end) time window for late pattern
    metric_col : str
        Column name for metric values
    time_col : str
        Column name for time values
    embryo_id_col : str
        Column name for embryo IDs
    n_permutations : int
        Number of permutations for null distribution
    alpha : float
        Significance level

    Returns
    -------
    results : dict
        - 'correlation': Observed Pearson r
        - 'p_value': Permutation p-value
        - 'classification': 'Anti-correlated' / 'Correlated' / 'Uncorrelated'
    """
    # Extract early and late means per embryo
    embryo_ids = df[embryo_id_col].unique()
    early_means = []
    late_means = []

    for embryo_id in embryo_ids:
        embryo_data = df[df[embryo_id_col] == embryo_id]

        # Early window
        early_data = embryo_data[
            (embryo_data[time_col] >= early_window[0]) &
            (embryo_data[time_col] <= early_window[1])
        ][metric_col]

        # Late window
        late_data = embryo_data[
            (embryo_data[time_col] >= late_window[0]) &
            (embryo_data[time_col] <= late_window[1])
        ][metric_col]

        if len(early_data) > 0 and len(late_data) > 0:
            early_means.append(early_data.mean())
            late_means.append(late_data.mean())

    early_means = np.array(early_means)
    late_means = np.array(late_means)

    # Observed correlation
    if len(early_means) < 3:
        return {
            'correlation': np.nan,
            'p_value': np.nan,
            'classification': 'Insufficient data',
            'n_embryos': len(early_means)
        }

    r_obs, _ = pearsonr(early_means, late_means)

    # Permutation test
    perm_correlations = []
    np.random.seed(42)
    for _ in range(n_permutations):
        perm_late = np.random.permutation(late_means)
        r_perm, _ = pearsonr(early_means, perm_late)
        perm_correlations.append(r_perm)

    perm_correlations = np.array(perm_correlations)

    # Two-tailed p-value
    p_value = np.mean(np.abs(perm_correlations) >= np.abs(r_obs))

    # Classification
    if p_value > alpha:
        classification = 'Uncorrelated'
    elif r_obs < -0.3:
        classification = 'Anti-correlated'
    else:
        classification = 'Correlated'

    return {
        'correlation': r_obs,
        'p_value': p_value,
        'classification': classification,
        'n_embryos': len(early_means),
        'early_means': early_means,
        'late_means': late_means
    }
