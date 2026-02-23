"""
Regression analysis for incomplete penetrance quantification.

This module fits regression models to quantify how much of the classifier's
predicted mutant probability can be explained by morphological distance from WT.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from typing import Dict, Tuple, Optional
import warnings


def logit_transform(p: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Apply logit transformation to probabilities.

    logit(p) = log(p / (1 - p))

    Parameters
    ----------
    p : np.ndarray
        Probabilities in [0, 1]
    epsilon : float
        Small value to avoid log(0) or log(inf)

    Returns
    -------
    np.ndarray
        Logit-transformed values
    """
    # Clip to avoid numerical issues
    p_clipped = np.clip(p, epsilon, 1 - epsilon)
    return np.log(p_clipped / (1 - p_clipped))


def fit_ols_regression(
    distances: np.ndarray,
    probabilities: np.ndarray,
    add_constant: bool = True
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Fit ordinary least squares regression: prob = β0 + β1 * distance + ε

    Parameters
    ----------
    distances : np.ndarray
        Morphological distances (independent variable)
    probabilities : np.ndarray
        Predicted mutant probabilities (dependent variable)
    add_constant : bool
        Whether to add intercept term

    Returns
    -------
    RegressionResultsWrapper
        Fitted OLS model
    """
    # Remove NaNs
    valid_mask = ~(np.isnan(distances) | np.isnan(probabilities))
    X = distances[valid_mask].reshape(-1, 1)
    y = probabilities[valid_mask]

    # Add constant if requested
    if add_constant:
        X = sm.add_constant(X)

    # Fit model
    model = sm.OLS(y, X)
    results = model.fit()

    return results


def fit_logit_regression(
    distances: np.ndarray,
    probabilities: np.ndarray,
    add_constant: bool = True,
    epsilon: float = 1e-6
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Fit regression on logit-transformed probabilities.

    logit(prob) = β0 + β1 * distance + ε

    Parameters
    ----------
    distances : np.ndarray
        Morphological distances
    probabilities : np.ndarray
        Predicted mutant probabilities
    add_constant : bool
        Whether to add intercept
    epsilon : float
        Small value for logit transform stability

    Returns
    -------
    RegressionResultsWrapper
        Fitted OLS model on logit scale
    """
    # Remove NaNs
    valid_mask = ~(np.isnan(distances) | np.isnan(probabilities))
    X = distances[valid_mask].reshape(-1, 1)
    y_prob = probabilities[valid_mask]

    # Logit transform
    y_logit = logit_transform(y_prob, epsilon=epsilon)

    # Add constant
    if add_constant:
        X = sm.add_constant(X)

    # Fit model
    model = sm.OLS(y_logit, X)
    results = model.fit()

    return results


def compute_regression_metrics(
    results: sm.regression.linear_model.RegressionResultsWrapper,
    model_type: str = "ols"
) -> Dict[str, float]:
    """
    Extract key regression metrics from fitted model.

    Parameters
    ----------
    results : RegressionResultsWrapper
        Fitted statsmodels regression
    model_type : str
        Model type ('ols' or 'logit')

    Returns
    -------
    dict
        Regression metrics including:
        - r_squared: R²
        - r_squared_adj: Adjusted R²
        - beta0: Intercept coefficient
        - beta1: Slope coefficient
        - beta0_se: Standard error of intercept
        - beta1_se: Standard error of slope
        - beta0_pval: p-value for intercept
        - beta1_pval: p-value for slope
        - beta1_ci_lower: Lower 95% CI for slope
        - beta1_ci_upper: Upper 95% CI for slope
        - aic: Akaike Information Criterion
        - bic: Bayesian Information Criterion
        - f_statistic: F-statistic
        - f_pvalue: p-value for F-test
        - n_obs: Number of observations
        - residual_std: Residual standard error
    """
    # Extract coefficients
    beta0 = results.params[0]  # Intercept
    beta1 = results.params[1]  # Slope

    # Standard errors
    beta0_se = results.bse[0]
    beta1_se = results.bse[1]

    # p-values
    beta0_pval = results.pvalues[0]
    beta1_pval = results.pvalues[1]

    # Confidence intervals (95%)
    conf_int = results.conf_int(alpha=0.05)
    beta1_ci_lower = conf_int[1, 0]
    beta1_ci_upper = conf_int[1, 1]

    return {
        'model_type': model_type,
        'r_squared': results.rsquared,
        'r_squared_adj': results.rsquared_adj,
        'beta0': beta0,
        'beta1': beta1,
        'beta0_se': beta0_se,
        'beta1_se': beta1_se,
        'beta0_pval': beta0_pval,
        'beta1_pval': beta1_pval,
        'beta1_ci_lower': beta1_ci_lower,
        'beta1_ci_upper': beta1_ci_upper,
        'aic': results.aic,
        'bic': results.bic,
        'f_statistic': results.fvalue,
        'f_pvalue': results.f_pvalue,
        'n_obs': int(results.nobs),
        'residual_std': np.sqrt(results.mse_resid)
    }


def compute_predicted_values(
    results: sm.regression.linear_model.RegressionResultsWrapper,
    distances: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute predicted values and confidence intervals from fitted model.

    Parameters
    ----------
    results : RegressionResultsWrapper
        Fitted regression model
    distances : np.ndarray
        Distance values for prediction

    Returns
    -------
    y_pred : np.ndarray
        Predicted values
    ci_lower : np.ndarray
        Lower 95% confidence interval
    ci_upper : np.ndarray
        Upper 95% confidence interval
    """
    # Prepare X matrix
    X = sm.add_constant(distances.reshape(-1, 1))

    # Predictions
    predictions = results.get_prediction(X)
    y_pred = predictions.predicted_mean

    # Confidence intervals
    ci = predictions.conf_int(alpha=0.05)
    ci_lower = ci[:, 0]
    ci_upper = ci[:, 1]

    return y_pred, ci_lower, ci_upper


def compute_residual_diagnostics(
    results: sm.regression.linear_model.RegressionResultsWrapper
) -> Dict[str, np.ndarray]:
    """
    Compute residual diagnostics for regression model.

    Parameters
    ----------
    results : RegressionResultsWrapper
        Fitted regression model

    Returns
    -------
    dict
        Residual diagnostics including:
        - residuals: Raw residuals
        - standardized_residuals: Standardized residuals
        - fitted_values: Fitted values
        - leverage: Leverage values
        - cooks_d: Cook's distance
    """
    # Get influence measures
    influence = results.get_influence()

    # Standardized residuals
    standardized_residuals = influence.resid_studentized_internal

    # Leverage
    leverage = influence.hat_matrix_diag

    # Cook's distance
    cooks_d = influence.cooks_distance[0]

    return {
        'residuals': results.resid,
        'standardized_residuals': standardized_residuals,
        'fitted_values': results.fittedvalues,
        'leverage': leverage,
        'cooks_d': cooks_d
    }


def test_normality(residuals: np.ndarray) -> Dict[str, float]:
    """
    Test normality of residuals using Shapiro-Wilk test.

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals

    Returns
    -------
    dict
        Test statistics:
        - shapiro_statistic: Shapiro-Wilk W statistic
        - shapiro_pvalue: p-value
    """
    statistic, pvalue = stats.shapiro(residuals)

    return {
        'shapiro_statistic': statistic,
        'shapiro_pvalue': pvalue
    }


def test_heteroskedasticity(
    results: sm.regression.linear_model.RegressionResultsWrapper
) -> Dict[str, float]:
    """
    Test heteroskedasticity using Breusch-Pagan test.

    Parameters
    ----------
    results : RegressionResultsWrapper
        Fitted regression model

    Returns
    -------
    dict
        Test statistics:
        - bp_statistic: Breusch-Pagan LM statistic
        - bp_pvalue: p-value
    """
    from statsmodels.stats.diagnostic import het_breuschpagan

    # Get residuals and fitted values
    resid = results.resid
    exog = results.model.exog

    # Breusch-Pagan test
    bp_stat, bp_pval, _, _ = het_breuschpagan(resid, exog)

    return {
        'bp_statistic': bp_stat,
        'bp_pvalue': bp_pval
    }


def compute_penetrance_cutoff(
    beta0: float,
    beta1: float,
    threshold_prob: float = 0.5,
    model_type: str = "ols",
    epsilon: float = 1e-6
) -> float:
    """
    Compute distance cutoff for penetrance classification.

    For OLS model: d* = (threshold - β0) / β1
    For logit model: d* = (logit(threshold) - β0) / β1

    Parameters
    ----------
    beta0 : float
        Intercept coefficient
    beta1 : float
        Slope coefficient
    threshold_prob : float
        Probability threshold (default 0.5)
    model_type : str
        Model type ('ols' or 'logit')
    epsilon : float
        Small value for numerical stability

    Returns
    -------
    float
        Distance cutoff d*
    """
    if model_type == "ols":
        # Direct calculation
        d_star = (threshold_prob - beta0) / beta1
    elif model_type == "logit":
        # Transform threshold to logit scale
        threshold_logit = logit_transform(np.array([threshold_prob]), epsilon=epsilon)[0]
        d_star = (threshold_logit - beta0) / beta1
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return d_star


def bootstrap_regression_ci(
    distances: np.ndarray,
    probabilities: np.ndarray,
    model_type: str = "ols",
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None
) -> Dict[str, Tuple[float, float]]:
    """
    Compute bootstrap confidence intervals for regression coefficients.

    Parameters
    ----------
    distances : np.ndarray
        Morphological distances
    probabilities : np.ndarray
        Predicted probabilities
    model_type : str
        Model type ('ols' or 'logit')
    n_bootstrap : int
        Number of bootstrap iterations
    confidence_level : float
        Confidence level
    random_state : int, optional
        Random seed

    Returns
    -------
    dict
        Bootstrap CIs:
        - beta0_ci: (lower, upper)
        - beta1_ci: (lower, upper)
        - r_squared_ci: (lower, upper)
    """
    rng = np.random.default_rng(random_state)

    # Remove NaNs
    valid_mask = ~(np.isnan(distances) | np.isnan(probabilities))
    distances = distances[valid_mask]
    probabilities = probabilities[valid_mask]

    n_samples = len(distances)

    beta0_boot = []
    beta1_boot = []
    r2_boot = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        boot_dist = distances[indices]
        boot_prob = probabilities[indices]

        # Fit model
        try:
            if model_type == "ols":
                results = fit_ols_regression(boot_dist, boot_prob)
            elif model_type == "logit":
                results = fit_logit_regression(boot_dist, boot_prob)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            beta0_boot.append(results.params[0])
            beta1_boot.append(results.params[1])
            r2_boot.append(results.rsquared)
        except:
            # Skip failed iterations
            continue

    # Compute CIs
    alpha = 1 - confidence_level
    beta0_ci = (
        np.percentile(beta0_boot, 100 * alpha / 2),
        np.percentile(beta0_boot, 100 * (1 - alpha / 2))
    )
    beta1_ci = (
        np.percentile(beta1_boot, 100 * alpha / 2),
        np.percentile(beta1_boot, 100 * (1 - alpha / 2))
    )
    r2_ci = (
        np.percentile(r2_boot, 100 * alpha / 2),
        np.percentile(r2_boot, 100 * (1 - alpha / 2))
    )

    return {
        'beta0_ci': beta0_ci,
        'beta1_ci': beta1_ci,
        'r_squared_ci': r2_ci
    }
