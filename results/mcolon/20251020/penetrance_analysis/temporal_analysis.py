"""
Temporal analysis for incomplete penetrance quantification.

This module analyzes how the distance-probability relationship changes over
developmental time to identify when phenotypes emerge and whether cutoffs
should be time-dependent.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr
from typing import Dict, Tuple, Optional, List
import warnings


def compute_per_bin_regression(
    df_distances: pd.DataFrame,
    df_predictions: pd.DataFrame,
    genotype: str,
    distance_col: str = 'euclidean_distance',
    prob_col: str = 'pred_prob_mutant',
    time_col: str = 'time_bin',
    min_samples: int = 10
) -> pd.DataFrame:
    """
    Compute regression metrics separately for each time bin.

    Parameters
    ----------
    df_distances : pd.DataFrame
        Distance data with columns: embryo_id, time_bin, genotype, distance
    df_predictions : pd.DataFrame
        Classifier predictions with columns: embryo_id, time_bin, pred_prob_mutant
    genotype : str
        Genotype to filter (e.g., 'cep290_homozygous')
    distance_col : str
        Distance column name
    prob_col : str
        Probability column name
    time_col : str
        Time bin column name
    min_samples : int
        Minimum samples per time bin

    Returns
    -------
    pd.DataFrame
        Temporal regression metrics with columns:
        - time_bin
        - n_embryos: Number of unique embryos
        - n_samples: Number of observations
        - pearson_r, pearson_p
        - spearman_rho, spearman_p
        - r_squared: OLS R²
        - r_squared_adj: Adjusted R²
        - beta0, beta0_se: Intercept
        - beta1, beta1_se: Slope
        - beta1_pval: Slope p-value
        - beta1_ci_lower, beta1_ci_upper: 95% CI for slope
        - d_star: Penetrance cutoff at prob=0.5
        - aic, bic: Model quality
        - residual_std: Residual standard error
    """
    # Filter to genotype
    df_dist_filtered = df_distances[df_distances['genotype'] == genotype].copy()

    # Merge distances with predictions
    df_merged = df_dist_filtered.merge(
        df_predictions[['embryo_id', time_col, prob_col]],
        on=['embryo_id', time_col],
        how='inner'
    )

    results = []

    for time_bin in sorted(df_merged[time_col].unique()):
        group = df_merged[df_merged[time_col] == time_bin]

        # Check minimum samples
        n_samples = len(group)
        n_embryos = group['embryo_id'].nunique()

        if n_samples < min_samples:
            continue

        # Extract data
        distances = group[distance_col].values
        probs = group[prob_col].values

        # Remove NaNs
        valid_mask = ~(np.isnan(distances) | np.isnan(probs))
        distances_clean = distances[valid_mask]
        probs_clean = probs[valid_mask]

        if len(distances_clean) < min_samples:
            continue

        # Compute correlations
        try:
            pearson_r, pearson_p = pearsonr(distances_clean, probs_clean)
            spearman_rho, spearman_p = spearmanr(distances_clean, probs_clean)
        except:
            pearson_r = pearson_p = spearman_rho = spearman_p = np.nan

        # Fit OLS regression
        try:
            X = sm.add_constant(distances_clean.reshape(-1, 1))
            y = probs_clean
            model = sm.OLS(y, X)
            ols_results = model.fit()

            beta0 = ols_results.params[0]
            beta1 = ols_results.params[1]
            beta0_se = ols_results.bse[0]
            beta1_se = ols_results.bse[1]
            beta1_pval = ols_results.pvalues[1]

            # Confidence intervals
            conf_int = ols_results.conf_int(alpha=0.05)
            beta1_ci_lower = conf_int[1, 0]
            beta1_ci_upper = conf_int[1, 1]

            # Model quality
            r_squared = ols_results.rsquared
            r_squared_adj = ols_results.rsquared_adj
            aic = ols_results.aic
            bic = ols_results.bic
            residual_std = np.sqrt(ols_results.mse_resid)

            # Compute cutoff: d* = (0.5 - beta0) / beta1
            if beta1 != 0:
                d_star = (0.5 - beta0) / beta1
            else:
                d_star = np.nan

        except Exception as e:
            # If regression fails, fill with NaNs
            beta0 = beta1 = beta0_se = beta1_se = beta1_pval = np.nan
            beta1_ci_lower = beta1_ci_upper = np.nan
            r_squared = r_squared_adj = aic = bic = residual_std = d_star = np.nan

        results.append({
            'time_bin': time_bin,
            'n_embryos': n_embryos,
            'n_samples': n_samples,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_rho': spearman_rho,
            'spearman_p': spearman_p,
            'r_squared': r_squared,
            'r_squared_adj': r_squared_adj,
            'beta0': beta0,
            'beta0_se': beta0_se,
            'beta1': beta1,
            'beta1_se': beta1_se,
            'beta1_pval': beta1_pval,
            'beta1_ci_lower': beta1_ci_lower,
            'beta1_ci_upper': beta1_ci_upper,
            'd_star': d_star,
            'aic': aic,
            'bic': bic,
            'residual_std': residual_std
        })

    return pd.DataFrame(results)


def compute_sliding_window_regression(
    df_distances: pd.DataFrame,
    df_predictions: pd.DataFrame,
    genotype: str,
    distance_col: str = 'euclidean_distance',
    prob_col: str = 'pred_prob_mutant',
    time_col: str = 'time_bin',
    window_size: float = 15.0,
    step_size: float = 5.0,
    min_samples: int = 15
) -> pd.DataFrame:
    """
    Compute regression metrics using sliding time windows.

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
    time_col : str
        Time column name
    window_size : float
        Window width in time units (e.g., 15 hpf)
    step_size : float
        Sliding step in time units (e.g., 5 hpf)
    min_samples : int
        Minimum samples per window

    Returns
    -------
    pd.DataFrame
        Windowed regression metrics with columns similar to per_bin_regression
        plus 'window_start' and 'window_end'
    """
    # Filter and merge
    df_dist_filtered = df_distances[df_distances['genotype'] == genotype].copy()
    df_merged = df_dist_filtered.merge(
        df_predictions[['embryo_id', time_col, prob_col]],
        on=['embryo_id', time_col],
        how='inner'
    )

    # Get time range
    time_min = df_merged[time_col].min()
    time_max = df_merged[time_col].max()

    # Generate window starts
    window_starts = np.arange(time_min, time_max - window_size + step_size, step_size)

    results = []

    for window_start in window_starts:
        window_end = window_start + window_size

        # Filter to window
        window_mask = (df_merged[time_col] >= window_start) & (df_merged[time_col] < window_end)
        window_data = df_merged[window_mask]

        if len(window_data) < min_samples:
            continue

        # Extract data
        distances = window_data[distance_col].values
        probs = window_data[prob_col].values

        # Remove NaNs
        valid_mask = ~(np.isnan(distances) | np.isnan(probs))
        distances_clean = distances[valid_mask]
        probs_clean = probs[valid_mask]

        if len(distances_clean) < min_samples:
            continue

        # Compute metrics (same as per_bin)
        try:
            pearson_r, pearson_p = pearsonr(distances_clean, probs_clean)
            spearman_rho, spearman_p = spearmanr(distances_clean, probs_clean)
        except:
            pearson_r = pearson_p = spearman_rho = spearman_p = np.nan

        try:
            X = sm.add_constant(distances_clean.reshape(-1, 1))
            y = probs_clean
            model = sm.OLS(y, X)
            ols_results = model.fit()

            beta0 = ols_results.params[0]
            beta1 = ols_results.params[1]
            beta0_se = ols_results.bse[0]
            beta1_se = ols_results.bse[1]
            r_squared = ols_results.rsquared
            d_star = (0.5 - beta0) / beta1 if beta1 != 0 else np.nan

        except:
            beta0 = beta1 = beta0_se = beta1_se = r_squared = d_star = np.nan

        results.append({
            'window_start': window_start,
            'window_end': window_end,
            'window_center': (window_start + window_end) / 2,
            'n_samples': len(distances_clean),
            'n_embryos': window_data['embryo_id'].nunique(),
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_rho': spearman_rho,
            'spearman_p': spearman_p,
            'r_squared': r_squared,
            'beta0': beta0,
            'beta1': beta1,
            'beta0_se': beta0_se,
            'beta1_se': beta1_se,
            'd_star': d_star
        })

    return pd.DataFrame(results)


def fit_interaction_model(
    df_distances: pd.DataFrame,
    df_predictions: pd.DataFrame,
    genotype: str,
    distance_col: str = 'euclidean_distance',
    prob_col: str = 'pred_prob_mutant',
    time_col: str = 'time_bin'
) -> Tuple[sm.regression.linear_model.RegressionResultsWrapper, Dict[str, float]]:
    """
    Fit interaction model to test if distance effect changes with time.

    Model: prob = β₀ + β₁·distance + β₂·time + β₃·(distance × time) + ε

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
    time_col : str
        Time column name

    Returns
    -------
    results : RegressionResultsWrapper
        Fitted statsmodels regression
    summary_dict : dict
        Summary statistics including:
        - beta0, beta1, beta2, beta3: Coefficients
        - beta3_pval: Interaction p-value (KEY!)
        - r_squared, r_squared_adj
        - aic, bic
    """
    # Filter and merge
    df_dist_filtered = df_distances[df_distances['genotype'] == genotype].copy()
    df_merged = df_dist_filtered.merge(
        df_predictions[['embryo_id', time_col, prob_col]],
        on=['embryo_id', time_col],
        how='inner'
    )

    # Extract data
    distances = df_merged[distance_col].values
    times = df_merged[time_col].values
    probs = df_merged[prob_col].values

    # Remove NaNs
    valid_mask = ~(np.isnan(distances) | np.isnan(times) | np.isnan(probs))
    distances = distances[valid_mask]
    times = times[valid_mask]
    probs = probs[valid_mask]

    # Normalize time to improve numerical stability
    time_mean = times.mean()
    time_std = times.std()
    times_normalized = (times - time_mean) / time_std

    # Create design matrix: [1, distance, time, distance×time]
    interaction_term = distances * times_normalized

    X = np.column_stack([
        np.ones(len(distances)),  # Intercept
        distances,                 # Distance
        times_normalized,          # Time
        interaction_term           # Distance × Time interaction
    ])

    y = probs

    # Fit OLS
    model = sm.OLS(y, X)
    results = model.fit()

    # Extract coefficients
    beta0 = results.params[0]  # Intercept
    beta1 = results.params[1]  # Distance main effect
    beta2 = results.params[2]  # Time main effect
    beta3 = results.params[3]  # Interaction (KEY!)

    beta3_pval = results.pvalues[3]  # Is interaction significant?

    summary_dict = {
        'beta0': beta0,
        'beta1': beta1,
        'beta2': beta2,
        'beta3': beta3,
        'beta0_se': results.bse[0],
        'beta1_se': results.bse[1],
        'beta2_se': results.bse[2],
        'beta3_se': results.bse[3],
        'beta0_pval': results.pvalues[0],
        'beta1_pval': results.pvalues[1],
        'beta2_pval': results.pvalues[2],
        'beta3_pval': beta3_pval,
        'r_squared': results.rsquared,
        'r_squared_adj': results.rsquared_adj,
        'aic': results.aic,
        'bic': results.bic,
        'f_statistic': results.fvalue,
        'f_pvalue': results.f_pvalue,
        'n_obs': int(results.nobs),
        'time_mean': time_mean,
        'time_std': time_std
    }

    return results, summary_dict


def identify_penetrance_onset(
    temporal_results: pd.DataFrame,
    r_squared_threshold: float = 0.3,
    slope_threshold: float = 0.05,
    pval_threshold: float = 0.05,
    time_col: str = 'time_bin'
) -> Dict[str, float]:
    """
    Identify when penetrance becomes detectable based on temporal trends.

    Penetrance onset is defined as the first time bin where:
    1. R² exceeds threshold (relationship is strong enough)
    2. Slope (β₁) exceeds threshold (effect size is meaningful)
    3. Slope is statistically significant

    Parameters
    ----------
    temporal_results : pd.DataFrame
        Results from compute_per_bin_regression()
    r_squared_threshold : float
        Minimum R² for detectable penetrance
    slope_threshold : float
        Minimum slope for meaningful effect
    pval_threshold : float
        Maximum p-value for significance
    time_col : str
        Time column name

    Returns
    -------
    dict
        Onset statistics:
        - onset_time: Estimated onset time (or NaN if never detected)
        - onset_r_squared: R² at onset
        - onset_slope: β₁ at onset
        - n_bins_detected: Number of bins meeting criteria
    """
    # Sort by time
    df_sorted = temporal_results.sort_values(time_col).copy()

    # Criteria mask
    criteria_mask = (
        (df_sorted['r_squared'] >= r_squared_threshold) &
        (df_sorted['beta1'] >= slope_threshold) &
        (df_sorted['beta1_pval'] <= pval_threshold)
    )

    bins_meeting_criteria = df_sorted[criteria_mask]

    if len(bins_meeting_criteria) == 0:
        # No onset detected
        return {
            'onset_time': np.nan,
            'onset_r_squared': np.nan,
            'onset_slope': np.nan,
            'n_bins_detected': 0
        }

    # Onset is first bin meeting criteria
    onset_row = bins_meeting_criteria.iloc[0]

    return {
        'onset_time': onset_row[time_col],
        'onset_r_squared': onset_row['r_squared'],
        'onset_slope': onset_row['beta1'],
        'onset_pearson_r': onset_row['pearson_r'],
        'n_bins_detected': len(bins_meeting_criteria),
        'n_bins_total': len(df_sorted)
    }


def compute_temporal_cutoffs(
    temporal_results: pd.DataFrame,
    prob_threshold: float = 0.5,
    time_col: str = 'time_bin'
) -> pd.DataFrame:
    """
    Extract time-dependent penetrance cutoffs from temporal regression.

    d*(t) = (prob_threshold - β₀(t)) / β₁(t)

    Parameters
    ----------
    temporal_results : pd.DataFrame
        Results from compute_per_bin_regression()
    prob_threshold : float
        Probability threshold for penetrance
    time_col : str
        Time column name

    Returns
    -------
    pd.DataFrame
        Time-dependent cutoffs with columns:
        - time_bin
        - d_star: Cutoff distance
        - beta0, beta1: Regression coefficients used
        - r_squared: Model quality at that time
    """
    cutoffs = temporal_results[[
        time_col, 'd_star', 'beta0', 'beta1', 'r_squared', 'n_samples'
    ]].copy()

    # Filter out invalid cutoffs
    cutoffs = cutoffs[~cutoffs['d_star'].isna()]

    return cutoffs


def compare_early_vs_late(
    df_distances: pd.DataFrame,
    df_predictions: pd.DataFrame,
    genotype: str,
    time_cutoff: float,
    distance_col: str = 'euclidean_distance',
    prob_col: str = 'pred_prob_mutant',
    time_col: str = 'time_bin'
) -> Dict[str, Dict[str, float]]:
    """
    Compare regression metrics between early and late timepoints.

    Useful for testing hypothesis that early data has weak phenotype.

    Parameters
    ----------
    df_distances : pd.DataFrame
        Distance data
    df_predictions : pd.DataFrame
        Classifier predictions
    genotype : str
        Genotype to filter
    time_cutoff : float
        Time threshold to split early vs late (e.g., 30 hpf)
    distance_col : str
        Distance column name
    prob_col : str
        Probability column name
    time_col : str
        Time column name

    Returns
    -------
    dict
        Nested dict with keys 'early' and 'late', each containing:
        - n_samples, n_embryos
        - pearson_r, spearman_rho
        - r_squared, beta0, beta1
        - aic, bic
    """
    # Filter and merge
    df_dist_filtered = df_distances[df_distances['genotype'] == genotype].copy()
    df_merged = df_dist_filtered.merge(
        df_predictions[['embryo_id', time_col, prob_col]],
        on=['embryo_id', time_col],
        how='inner'
    )

    results = {}

    for period_name, mask in [
        ('early', df_merged[time_col] < time_cutoff),
        ('late', df_merged[time_col] >= time_cutoff)
    ]:
        period_data = df_merged[mask]

        distances = period_data[distance_col].values
        probs = period_data[prob_col].values

        # Remove NaNs
        valid_mask = ~(np.isnan(distances) | np.isnan(probs))
        distances_clean = distances[valid_mask]
        probs_clean = probs[valid_mask]

        # Compute metrics
        try:
            pearson_r, pearson_p = pearsonr(distances_clean, probs_clean)
            spearman_rho, spearman_p = spearmanr(distances_clean, probs_clean)
        except:
            pearson_r = pearson_p = spearman_rho = spearman_p = np.nan

        try:
            X = sm.add_constant(distances_clean.reshape(-1, 1))
            y = probs_clean
            model = sm.OLS(y, X)
            ols_results = model.fit()

            beta0 = ols_results.params[0]
            beta1 = ols_results.params[1]
            r_squared = ols_results.rsquared
            r_squared_adj = ols_results.rsquared_adj
            aic = ols_results.aic
            bic = ols_results.bic

        except:
            beta0 = beta1 = r_squared = r_squared_adj = aic = bic = np.nan

        results[period_name] = {
            'n_samples': len(distances_clean),
            'n_embryos': period_data['embryo_id'].nunique(),
            'time_range': (period_data[time_col].min(), period_data[time_col].max()),
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_rho': spearman_rho,
            'spearman_p': spearman_p,
            'r_squared': r_squared,
            'r_squared_adj': r_squared_adj,
            'beta0': beta0,
            'beta1': beta1,
            'aic': aic,
            'bic': bic
        }

    return results
