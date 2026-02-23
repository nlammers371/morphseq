"""
Visualization utilities for penetrance correlation analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, Optional, List
import statsmodels.api as sm


def plot_distance_vs_probability(
    embryo_metrics: pd.DataFrame,
    correlation_stats: Dict[str, float],
    genotype: str,
    distance_col: str = 'mean_distance',
    prob_col: str = 'mean_prob',
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create scatter plot of distance vs predicted probability with regression line.

    Parameters
    ----------
    embryo_metrics : pd.DataFrame
        Per-embryo metrics
    correlation_stats : dict
        Correlation statistics from compute_correlation_statistics()
    genotype : str
        Genotype label for title
    distance_col : str
        Column name for distance (x-axis)
    prob_col : str
        Column name for probability (y-axis)
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    distances = embryo_metrics[distance_col].values
    probs = embryo_metrics[prob_col].values

    # Scatter plot
    ax.scatter(distances, probs, alpha=0.6, s=100, edgecolors='k', linewidths=0.5)

    # Regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(distances, probs)
    x_range = np.linspace(distances.min(), distances.max(), 100)
    y_pred = slope * x_range + intercept
    ax.plot(x_range, y_pred, 'r--', linewidth=2, label=f'Linear fit (R²={r_value**2:.3f})')

    # Add confidence interval for regression
    predict_std = np.sqrt(np.sum((probs - (slope * distances + intercept))**2) / (len(distances) - 2))
    margin = 1.96 * predict_std  # 95% CI
    ax.fill_between(x_range, y_pred - margin, y_pred + margin, alpha=0.2, color='red')

    # Labels and title
    ax.set_xlabel('Mean Distance from WT (Euclidean)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Predicted Mutant Probability', fontsize=14, fontweight='bold')
    ax.set_title(f'{genotype}: Distance vs Classifier Probability', fontsize=16, fontweight='bold')

    # Add statistics text box
    stats_text = (
        f"Pearson r = {correlation_stats['pearson_r']:.3f}\n"
        f"Pearson p = {correlation_stats['pearson_p']:.3e}\n"
        f"Spearman ρ = {correlation_stats['spearman_rho']:.3f}\n"
        f"Spearman p = {correlation_stats['spearman_p']:.3e}\n"
        f"N = {correlation_stats['n_embryos']} embryos"
    )
    ax.text(0.05, 0.95, stats_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='lower right')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_correlation_summary(
    all_results: pd.DataFrame,
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create summary bar plot comparing correlations across genotypes.

    Parameters
    ----------
    all_results : pd.DataFrame
        Combined results with columns: genotype, pearson_r, spearman_rho, etc.
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Pearson correlation
    ax = axes[0]
    genotypes = all_results['genotype'].values
    x_pos = np.arange(len(genotypes))

    bars = ax.bar(x_pos, all_results['pearson_r'].values,
                   color='steelblue', alpha=0.8, edgecolor='black')

    # Add error bars if CI available
    if 'pearson_ci_lower' in all_results.columns:
        yerr = [
            all_results['pearson_r'].values - all_results['pearson_ci_lower'].values,
            all_results['pearson_ci_upper'].values - all_results['pearson_r'].values
        ]
        ax.errorbar(x_pos, all_results['pearson_r'].values,
                   yerr=yerr, fmt='none', ecolor='black', capsize=5, linewidth=2)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(genotypes, rotation=45, ha='right')
    ax.set_ylabel('Pearson r', fontsize=12, fontweight='bold')
    ax.set_title('Pearson Correlation', fontsize=14, fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='r=0.5 threshold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()

    # Add value labels on bars
    for i, (bar, val, p) in enumerate(zip(bars, all_results['pearson_r'].values, all_results['pearson_p'].values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}\np={p:.3f}',
               ha='center', va='bottom', fontsize=9)

    # Spearman correlation
    ax = axes[1]
    bars = ax.bar(x_pos, all_results['spearman_rho'].values,
                   color='coral', alpha=0.8, edgecolor='black')

    # Add error bars if CI available
    if 'spearman_ci_lower' in all_results.columns:
        yerr = [
            all_results['spearman_rho'].values - all_results['spearman_ci_lower'].values,
            all_results['spearman_ci_upper'].values - all_results['spearman_rho'].values
        ]
        ax.errorbar(x_pos, all_results['spearman_rho'].values,
                   yerr=yerr, fmt='none', ecolor='black', capsize=5, linewidth=2)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(genotypes, rotation=45, ha='right')
    ax.set_ylabel('Spearman ρ', fontsize=12, fontweight='bold')
    ax.set_title('Spearman Rank Correlation', fontsize=14, fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='ρ=0.5 threshold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()

    # Add value labels
    for i, (bar, val, p) in enumerate(zip(bars, all_results['spearman_rho'].values, all_results['spearman_p'].values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}\np={p:.3f}',
               ha='center', va='bottom', fontsize=9)

    plt.suptitle('Correlation: Distance vs Predicted Mutant Probability', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_correlation_over_time(
    df_time_correlation: pd.DataFrame,
    genotype: str,
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot how correlation strength changes over developmental time.

    Parameters
    ----------
    df_time_correlation : pd.DataFrame
        Results from correlation_by_time_bin()
    genotype : str
        Genotype label
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save

    Returns
    -------
    matplotlib.Figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Pearson over time
    ax = axes[0]
    ax.plot(df_time_correlation['time_bin'], df_time_correlation['pearson_r'],
            marker='o', linewidth=2, markersize=8, color='steelblue', label='Pearson r')
    ax.fill_between(df_time_correlation['time_bin'],
                    df_time_correlation['pearson_r'] - 0.1,
                    df_time_correlation['pearson_r'] + 0.1,
                    alpha=0.2, color='steelblue')
    ax.axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='r=0.5')
    ax.set_ylabel('Pearson r', fontsize=12, fontweight='bold')
    ax.set_title(f'{genotype}: Correlation Strength Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Spearman over time
    ax = axes[1]
    ax.plot(df_time_correlation['time_bin'], df_time_correlation['spearman_rho'],
            marker='s', linewidth=2, markersize=8, color='coral', label='Spearman ρ')
    ax.fill_between(df_time_correlation['time_bin'],
                    df_time_correlation['spearman_rho'] - 0.1,
                    df_time_correlation['spearman_rho'] + 0.1,
                    alpha=0.2, color='coral')
    ax.axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='ρ=0.5')
    ax.set_xlabel('Developmental Time (hpf)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Spearman ρ', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig

def plot_regression_fit(
    embryo_metrics: pd.DataFrame,
    results: sm.regression.linear_model.RegressionResultsWrapper,
    regression_metrics: Dict[str, float],
    genotype: str,
    distance_col: str = 'mean_distance',
    prob_col: str = 'mean_prob',
    model_type: str = 'ols',
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot regression fit with confidence intervals.

    Parameters
    ----------
    embryo_metrics : pd.DataFrame
        Per-embryo metrics
    results : RegressionResultsWrapper
        Fitted statsmodels regression
    regression_metrics : dict
        Regression metrics from compute_regression_metrics()
    genotype : str
        Genotype label
    distance_col : str
        Distance column name
    prob_col : str
        Probability column name
    model_type : str
        Model type ('ols' or 'logit')
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save

    Returns
    -------
    matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    distances = embryo_metrics[distance_col].values
    probs = embryo_metrics[prob_col].values

    # For logit model, transform y-axis
    if model_type == 'logit':
        from .regression import logit_transform
        probs_plot = logit_transform(probs)
        ylabel = 'Logit(Predicted Mutant Probability)'
    else:
        probs_plot = probs
        ylabel = 'Predicted Mutant Probability'

    # Scatter plot
    ax.scatter(distances, probs_plot, alpha=0.6, s=100, edgecolors='k', linewidths=0.5)

    # Regression line with CI
    x_range = np.linspace(distances.min(), distances.max(), 100)
    X_pred = sm.add_constant(x_range.reshape(-1, 1))
    predictions = results.get_prediction(X_pred)
    y_pred = predictions.predicted_mean
    ci = predictions.conf_int(alpha=0.05)

    ax.plot(x_range, y_pred, 'r-', linewidth=2,
            label=f'Regression fit (R²={regression_metrics["r_squared"]:.3f})')
    ax.fill_between(x_range, ci[:, 0], ci[:, 1], alpha=0.2, color='red',
                     label='95% CI')

    # Labels
    ax.set_xlabel('Mean Distance from WT (Euclidean)', fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')

    title = f'{genotype}: Regression Analysis ({model_type.upper()} Model)'
    ax.set_title(title, fontsize=16, fontweight='bold')

    # Statistics text box
    stats_text = (
        f"R² = {regression_metrics['r_squared']:.3f}\n"
        f"Adj. R² = {regression_metrics['r_squared_adj']:.3f}\n"
        f"β₀ = {regression_metrics['beta0']:.3f} ± {regression_metrics['beta0_se']:.3f}\n"
        f"β₁ = {regression_metrics['beta1']:.3f} ± {regression_metrics['beta1_se']:.3f}\n"
        f"F = {regression_metrics['f_statistic']:.2f}\n"
        f"p = {regression_metrics['f_pvalue']:.3e}\n"
        f"N = {regression_metrics['n_obs']} embryos"
    )
    ax.text(0.05, 0.95, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='lower right')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_regression_diagnostics(
    embryo_metrics: pd.DataFrame,
    results: sm.regression.linear_model.RegressionResultsWrapper,
    diagnostics: Dict[str, np.ndarray],
    genotype: str,
    distance_col: str = 'mean_distance',
    prob_col: str = 'mean_prob',
    figsize: tuple = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comprehensive regression diagnostic plots.

    Includes:
    1. Residuals vs Fitted
    2. Q-Q plot
    3. Scale-Location plot
    4. Residuals vs Leverage (Cook's distance)

    Parameters
    ----------
    embryo_metrics : pd.DataFrame
        Per-embryo metrics
    results : RegressionResultsWrapper
        Fitted regression model
    diagnostics : dict
        Diagnostics from compute_residual_diagnostics()
    genotype : str
        Genotype label
    distance_col : str
        Distance column
    prob_col : str
        Probability column
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save

    Returns
    -------
    matplotlib.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Extract diagnostics
    residuals = diagnostics['residuals']
    standardized_resid = diagnostics['standardized_residuals']
    fitted = diagnostics['fitted_values']
    leverage = diagnostics['leverage']
    cooks_d = diagnostics['cooks_d']

    # 1. Residuals vs Fitted
    ax = axes[0, 0]
    ax.scatter(fitted, residuals, alpha=0.6, edgecolors='k', linewidths=0.5)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)

    # Add LOWESS smoothing
    from statsmodels.nonparametric.smoothers_lowess import lowess
    lowess_result = lowess(residuals, fitted, frac=0.6)
    ax.plot(lowess_result[:, 0], lowess_result[:, 1], 'b-', linewidth=2, label='LOWESS')

    ax.set_xlabel('Fitted Values', fontsize=12, fontweight='bold')
    ax.set_ylabel('Residuals', fontsize=12, fontweight='bold')
    ax.set_title('Residuals vs Fitted', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. Q-Q Plot
    ax = axes[0, 1]
    stats.probplot(standardized_resid, dist="norm", plot=ax)
    ax.set_title('Normal Q-Q Plot', fontsize=14, fontweight='bold')
    ax.set_xlabel('Theoretical Quantiles', fontsize=12, fontweight='bold')
    ax.set_ylabel('Standardized Residuals', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 3. Scale-Location (sqrt of standardized residuals vs fitted)
    ax = axes[1, 0]
    sqrt_std_resid = np.sqrt(np.abs(standardized_resid))
    ax.scatter(fitted, sqrt_std_resid, alpha=0.6, edgecolors='k', linewidths=0.5)

    # LOWESS smoothing
    lowess_result = lowess(sqrt_std_resid, fitted, frac=0.6)
    ax.plot(lowess_result[:, 0], lowess_result[:, 1], 'r-', linewidth=2, label='LOWESS')

    ax.set_xlabel('Fitted Values', fontsize=12, fontweight='bold')
    ax.set_ylabel('√|Standardized Residuals|', fontsize=12, fontweight='bold')
    ax.set_title('Scale-Location', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4. Residuals vs Leverage (with Cook's distance contours)
    ax = axes[1, 1]
    ax.scatter(leverage, standardized_resid, alpha=0.6, edgecolors='k', linewidths=0.5)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)

    # Highlight high Cook's distance points
    high_influence = cooks_d > 4 / len(residuals)  # Threshold: 4/n
    if np.any(high_influence):
        ax.scatter(leverage[high_influence], standardized_resid[high_influence],
                  s=200, facecolors='none', edgecolors='r', linewidths=2,
                  label=f"High Cook's D (n={np.sum(high_influence)})")

    # Add Cook's distance contours
    x_range = np.linspace(0.001, leverage.max(), 50)
    for d in [0.5, 1.0]:
        y_pos = np.sqrt(d * len(residuals) * x_range / (1 - x_range))
        y_neg = -y_pos
        ax.plot(x_range, y_pos, '--', color='orange', alpha=0.5, linewidth=1)
        ax.plot(x_range, y_neg, '--', color='orange', alpha=0.5, linewidth=1)
        ax.text(x_range[-1], y_pos[-1], f"Cook's d={d}", fontsize=8, color='orange')

    ax.set_xlabel('Leverage', fontsize=12, fontweight='bold')
    ax.set_ylabel('Standardized Residuals', fontsize=12, fontweight='bold')
    ax.set_title('Residuals vs Leverage', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if np.any(high_influence):
        ax.legend()

    fig.suptitle(f'{genotype}: Regression Diagnostics', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_regression_comparison(
    all_results: pd.DataFrame,
    figsize: tuple = (14, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare regression results across genotypes and models.

    Parameters
    ----------
    all_results : pd.DataFrame
        Combined regression results with columns:
        genotype, model_type, r_squared, beta0, beta1, etc.
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save

    Returns
    -------
    matplotlib.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Group by genotype and model_type
    genotypes = all_results['genotype'].unique()
    model_types = all_results['model_type'].unique()

    x_pos = np.arange(len(genotypes))
    width = 0.35

    # 1. R² comparison
    ax = axes[0]
    for i, model in enumerate(model_types):
        data = all_results[all_results['model_type'] == model]
        offset = width * (i - len(model_types)/2 + 0.5)

        bars = ax.bar(x_pos + offset, data['r_squared'].values, width,
                     label=model.upper(), alpha=0.8, edgecolor='black')

        # Add value labels
        for bar, val in zip(bars, data['r_squared'].values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(genotypes, rotation=45, ha='right')
    ax.set_ylabel('R²', fontsize=12, fontweight='bold')
    ax.set_title('Variance Explained', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)

    # 2. Slope (β₁) comparison
    ax = axes[1]
    for i, model in enumerate(model_types):
        data = all_results[all_results['model_type'] == model]
        offset = width * (i - len(model_types)/2 + 0.5)

        bars = ax.bar(x_pos + offset, data['beta1'].values, width,
                     label=model.upper(), alpha=0.8, edgecolor='black')

        # Add error bars if CI available
        if 'beta1_ci_lower' in data.columns:
            yerr = [
                data['beta1'].values - data['beta1_ci_lower'].values,
                data['beta1_ci_upper'].values - data['beta1'].values
            ]
            ax.errorbar(x_pos + offset, data['beta1'].values,
                       yerr=yerr, fmt='none', ecolor='black', capsize=5, linewidth=1.5)

        # Add value labels
        for bar, val in zip(bars, data['beta1'].values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(genotypes, rotation=45, ha='right')
    ax.set_ylabel('β₁ (Slope)', fontsize=12, fontweight='bold')
    ax.set_title('Distance Effect Size', fontsize=14, fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Model comparison (AIC)
    ax = axes[2]
    for i, model in enumerate(model_types):
        data = all_results[all_results['model_type'] == model]
        offset = width * (i - len(model_types)/2 + 0.5)

        bars = ax.bar(x_pos + offset, data['aic'].values, width,
                     label=model.upper(), alpha=0.8, edgecolor='black')

        # Add value labels
        for bar, val in zip(bars, data['aic'].values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(genotypes, rotation=45, ha='right')
    ax.set_ylabel('AIC', fontsize=12, fontweight='bold')
    ax.set_title('Model Fit (lower is better)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Regression Model Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


# ============================================================================
# TEMPORAL ANALYSIS VISUALIZATIONS
# ============================================================================

def plot_correlation_over_time(
    temporal_results: pd.DataFrame,
    genotype: str,
    time_col: str = 'time_bin',
    onset_time: Optional[float] = None,
    figsize: tuple = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot how correlation strength changes over developmental time.

    Parameters
    ----------
    temporal_results : pd.DataFrame
        Results from compute_per_bin_regression()
    genotype : str
        Genotype label
    time_col : str
        Time column name
    onset_time : float, optional
        Penetrance onset time to mark
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save

    Returns
    -------
    matplotlib.Figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Pearson correlation
    ax = axes[0]
    ax.plot(temporal_results[time_col], temporal_results['pearson_r'],
            'o-', linewidth=2, markersize=8, color='steelblue', label='Pearson r')

    # Shade weak correlation zone
    ax.axhspan(-0.3, 0.3, alpha=0.1, color='gray', label='Weak correlation')
    ax.axhline(0.5, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Moderate (r=0.5)')
    ax.axhline(0.7, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Strong (r=0.7)')

    # Mark onset if provided
    if onset_time is not None:
        ax.axvline(onset_time, color='red', linestyle=':', linewidth=2, alpha=0.7, label=f'Onset ({onset_time} hpf)')

    ax.set_ylabel('Pearson r', fontsize=12, fontweight='bold')
    ax.set_title(f'{genotype}: Correlation Strength Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, 1.0)

    # Spearman correlation
    ax = axes[1]
    ax.plot(temporal_results[time_col], temporal_results['spearman_rho'],
            's-', linewidth=2, markersize=8, color='coral', label='Spearman ρ')

    ax.axhspan(-0.3, 0.3, alpha=0.1, color='gray')
    ax.axhline(0.5, color='orange', linestyle='--', alpha=0.7, linewidth=2)
    ax.axhline(0.7, color='green', linestyle='--', alpha=0.7, linewidth=2)

    if onset_time is not None:
        ax.axvline(onset_time, color='red', linestyle=':', linewidth=2, alpha=0.7)

    ax.set_xlabel('Developmental Time (hpf)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Spearman ρ', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, 1.0)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_r_squared_evolution(
    temporal_results_dict: Dict[str, pd.DataFrame],
    time_col: str = 'time_bin',
    onset_times: Optional[Dict[str, float]] = None,
    figsize: tuple = (12, 7),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot R² evolution over time, comparing genotypes.

    Parameters
    ----------
    temporal_results_dict : dict
        Dict mapping genotype name to temporal results DataFrame
    time_col : str
        Time column name
    onset_times : dict, optional
        Dict mapping genotype to onset time
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save

    Returns
    -------
    matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = {'cep290_homozygous': 'steelblue', 'tmem67_homozygous': 'coral'}
    markers = {'cep290_homozygous': 'o', 'tmem67_homozygous': 's'}

    for genotype, df in temporal_results_dict.items():
        color = colors.get(genotype, 'gray')
        marker = markers.get(genotype, 'o')

        # Plot R² with confidence band (using sample size as weight)
        ax.plot(df[time_col], df['r_squared'],
                marker=marker, linestyle='-', linewidth=2, markersize=8,
                color=color, label=genotype, alpha=0.8)

        # Mark onset if provided
        if onset_times and genotype in onset_times:
            onset = onset_times[genotype]
            ax.axvline(onset, color=color, linestyle=':', linewidth=2, alpha=0.5)

    # Reference lines
    ax.axhline(0.3, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='R²=0.3 (moderate)')
    ax.axhline(0.5, color='green', linestyle='--', alpha=0.5, linewidth=2, label='R²=0.5 (strong)')

    ax.set_xlabel('Developmental Time (hpf)', fontsize=12, fontweight='bold')
    ax.set_ylabel('R² (Variance Explained)', fontsize=12, fontweight='bold')
    ax.set_title('Variance Explained Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_slope_evolution(
    temporal_results: pd.DataFrame,
    genotype: str,
    time_col: str = 'time_bin',
    onset_time: Optional[float] = None,
    figsize: tuple = (12, 7),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot slope (β₁) evolution over time with confidence intervals.

    Parameters
    ----------
    temporal_results : pd.DataFrame
        Results from compute_per_bin_regression()
    genotype : str
        Genotype label
    time_col : str
        Time column name
    onset_time : float, optional
        Penetrance onset time
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save

    Returns
    -------
    matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot slope with error bars
    ax.errorbar(temporal_results[time_col], temporal_results['beta1'],
                yerr=temporal_results['beta1_se'],
                fmt='o-', linewidth=2, markersize=8, capsize=5, capthick=2,
                color='darkgreen', label='Slope (β₁)')

    # Fill between CI
    if 'beta1_ci_lower' in temporal_results.columns:
        ax.fill_between(temporal_results[time_col],
                        temporal_results['beta1_ci_lower'],
                        temporal_results['beta1_ci_upper'],
                        alpha=0.2, color='darkgreen', label='95% CI')

    # Zero line
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=2)

    # Mark onset
    if onset_time is not None:
        ax.axvline(onset_time, color='red', linestyle=':', linewidth=2, alpha=0.7,
                  label=f'Onset ({onset_time} hpf)')

    ax.set_xlabel('Developmental Time (hpf)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Slope (β₁) - Effect Size', fontsize=12, fontweight='bold')
    ax.set_title(f'{genotype}: Distance Effect Size Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_temporal_cutoffs(
    temporal_cutoffs_dict: Dict[str, pd.DataFrame],
    time_col: str = 'time_bin',
    figsize: tuple = (12, 7),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot time-dependent penetrance cutoffs d*(t).

    Parameters
    ----------
    temporal_cutoffs_dict : dict
        Dict mapping genotype to cutoff DataFrame
    time_col : str
        Time column name
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save

    Returns
    -------
    matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = {'cep290_homozygous': 'steelblue', 'tmem67_homozygous': 'coral'}
    markers = {'cep290_homozygous': 'o', 'tmem67_homozygous': 's'}

    for genotype, df in temporal_cutoffs_dict.items():
        color = colors.get(genotype, 'gray')
        marker = markers.get(genotype, 'o')

        ax.plot(df[time_col], df['d_star'],
                marker=marker, linestyle='-', linewidth=2, markersize=8,
                color=color, label=genotype, alpha=0.8)

    ax.set_xlabel('Developmental Time (hpf)', fontsize=12, fontweight='bold')
    ax.set_ylabel('d* (Penetrance Cutoff Distance)', fontsize=12, fontweight='bold')
    ax.set_title('Time-Dependent Penetrance Cutoffs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_scatter_by_timebin(
    df_distances: pd.DataFrame,
    df_predictions: pd.DataFrame,
    temporal_results: pd.DataFrame,
    genotype: str,
    distance_col: str = 'euclidean_distance',
    prob_col: str = 'pred_prob_mutant',
    time_col: str = 'time_bin',
    bins_to_plot: Optional[List[float]] = None,
    ncols: int = 4,
    figsize: tuple = (16, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create multi-panel scatter plots of distance vs probability by time bin.

    Parameters
    ----------
    df_distances : pd.DataFrame
        Distance data
    df_predictions : pd.DataFrame
        Classifier predictions
    temporal_results : pd.DataFrame
        Temporal regression results
    genotype : str
        Genotype to filter
    distance_col : str
        Distance column name
    prob_col : str
        Probability column name
    time_col : str
        Time column name
    bins_to_plot : list, optional
        Specific time bins to plot (if None, plot all)
    ncols : int
        Number of columns in grid
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save

    Returns
    -------
    matplotlib.Figure
    """
    # Filter and merge
    df_dist_filtered = df_distances[df_distances['genotype'] == genotype].copy()
    df_merged = df_dist_filtered.merge(
        df_predictions[['embryo_id', time_col, prob_col]],
        on=['embryo_id', time_col],
        how='inner'
    )

    # Determine bins to plot
    if bins_to_plot is None:
        bins_to_plot = sorted(temporal_results[time_col].unique())

    n_bins = len(bins_to_plot)
    nrows = int(np.ceil(n_bins / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten() if n_bins > 1 else [axes]

    for idx, time_bin in enumerate(bins_to_plot):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Get data for this bin
        bin_data = df_merged[df_merged[time_col] == time_bin]

        if len(bin_data) == 0:
            ax.text(0.5, 0.5, f'{time_bin} hpf\nNo data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_xlim(0, 4)
            ax.set_ylim(0, 1)
            continue

        distances = bin_data[distance_col].values
        probs = bin_data[prob_col].values

        # Scatter
        ax.scatter(distances, probs, alpha=0.6, s=50, edgecolors='k', linewidths=0.5)

        # Get regression for this bin
        bin_regression = temporal_results[temporal_results[time_col] == time_bin]

        if len(bin_regression) > 0:
            row = bin_regression.iloc[0]
            beta0 = row['beta0']
            beta1 = row['beta1']
            r_squared = row['r_squared']

            # Plot regression line
            x_range = np.linspace(distances.min(), distances.max(), 100)
            y_pred = beta0 + beta1 * x_range
            ax.plot(x_range, y_pred, 'r-', linewidth=2, alpha=0.8)

            # Add stats text
            stats_text = f'R²={r_squared:.2f}\nβ₁={beta1:.3f}\nn={len(bin_data)}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        ax.set_title(f'{time_bin} hpf', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Remove extra axes
    for idx in range(n_bins, len(axes)):
        fig.delaxes(axes[idx])

    # Common labels
    fig.text(0.5, 0.02, 'Distance from WT', ha='center', fontsize=14, fontweight='bold')
    fig.text(0.02, 0.5, 'Predicted Mutant Probability', va='center', rotation='vertical',
            fontsize=14, fontweight='bold')
    fig.suptitle(f'{genotype}: Distance vs Probability Over Time', fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_distance_time_heatmap(
    df_distances: pd.DataFrame,
    df_predictions: pd.DataFrame,
    genotype: str,
    distance_col: str = 'euclidean_distance',
    prob_col: str = 'pred_prob_mutant',
    time_col: str = 'time_bin',
    distance_bins: int = 20,
    figsize: tuple = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create heatmap: Distance × Time → Mean Probability.

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
    distance_bins : int
        Number of distance bins
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save

    Returns
    -------
    matplotlib.Figure
    """
    # Filter and merge
    df_dist_filtered = df_distances[df_distances['genotype'] == genotype].copy()
    df_merged = df_dist_filtered.merge(
        df_predictions[['embryo_id', time_col, prob_col]],
        on=['embryo_id', time_col],
        how='inner'
    )

    # Create distance bins
    df_merged['distance_bin'] = pd.cut(df_merged[distance_col], bins=distance_bins)

    # Compute mean probability for each distance bin × time bin
    heatmap_data = df_merged.groupby([time_col, 'distance_bin'])[prob_col].mean().unstack()

    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(heatmap_data.T, aspect='auto', cmap='RdYlGn', origin='lower',
                   vmin=0, vmax=1, interpolation='bilinear')

    # Contour for 50% threshold
    X, Y = np.meshgrid(range(heatmap_data.shape[0]), range(heatmap_data.shape[1]))
    CS = ax.contour(X, Y, heatmap_data.T.values, levels=[0.5], colors='blue',
                    linewidths=2, linestyles='--')
    ax.clabel(CS, inline=True, fontsize=10, fmt='%.1f')

    # Labels
    ax.set_xlabel('Developmental Time (hpf)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Distance from WT (binned)', fontsize=12, fontweight='bold')
    ax.set_title(f'{genotype}: Phenotype Emergence Heatmap', fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Predicted Mutant Probability', fontsize=11, fontweight='bold')

    # Set ticks
    time_ticks = np.linspace(0, len(heatmap_data) - 1, min(10, len(heatmap_data)))
    ax.set_xticks(time_ticks)
    ax.set_xticklabels([f'{heatmap_data.index[int(t)]:.0f}' for t in time_ticks])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_interaction_model(
    df_distances: pd.DataFrame,
    df_predictions: pd.DataFrame,
    interaction_results: sm.regression.linear_model.RegressionResultsWrapper,
    summary_dict: Dict[str, float],
    genotype: str,
    distance_col: str = 'euclidean_distance',
    prob_col: str = 'pred_prob_mutant',
    time_col: str = 'time_bin',
    figsize: tuple = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize interaction model: Does slope change with time?

    Parameters
    ----------
    df_distances : pd.DataFrame
        Distance data
    df_predictions : pd.DataFrame
        Predictions
    interaction_results : RegressionResultsWrapper
        Fitted interaction model
    summary_dict : dict
        Summary statistics
    genotype : str
        Genotype label
    distance_col : str
        Distance column
    prob_col : str
        Probability column
    time_col : str
        Time column
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save

    Returns
    -------
    matplotlib.Figure
    """
    # Filter and merge
    df_dist_filtered = df_distances[df_distances['genotype'] == genotype].copy()
    df_merged = df_dist_filtered.merge(
        df_predictions[['embryo_id', time_col, prob_col]],
        on=['embryo_id', time_col],
        how='inner'
    )

    fig, ax = plt.subplots(figsize=figsize)

    # Divide time into quartiles
    time_quartiles = df_merged[time_col].quantile([0, 0.25, 0.5, 0.75, 1.0])
    colors = ['blue', 'green', 'orange', 'red']
    labels = ['Early', 'Mid-Early', 'Mid-Late', 'Late']

    for i in range(4):
        q_low = time_quartiles.iloc[i]
        q_high = time_quartiles.iloc[i + 1]

        mask = (df_merged[time_col] >= q_low) & (df_merged[time_col] < q_high)
        subset = df_merged[mask]

        # Scatter
        ax.scatter(subset[distance_col], subset[prob_col],
                  alpha=0.4, s=30, color=colors[i], label=f'{labels[i]} ({q_low:.0f}-{q_high:.0f} hpf)')

        # Fit line for this quartile
        if len(subset) > 10:
            X = sm.add_constant(subset[distance_col].values.reshape(-1, 1))
            y = subset[prob_col].values
            model = sm.OLS(y, X).fit()

            x_range = np.linspace(subset[distance_col].min(), subset[distance_col].max(), 100)
            y_pred = model.params[0] + model.params[1] * x_range
            ax.plot(x_range, y_pred, color=colors[i], linewidth=2, alpha=0.8)

    ax.set_xlabel('Distance from WT', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Mutant Probability', fontsize=12, fontweight='bold')
    ax.set_title(f'{genotype}: Interaction Model (Distance × Time)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    # Add interaction statistics
    beta3 = summary_dict['beta3']
    beta3_pval = summary_dict['beta3_pval']
    r_squared = summary_dict['r_squared']

    stats_text = (
        f"Interaction Model Results:\n"
        f"β₃ (interaction) = {beta3:.4f}\n"
        f"p-value = {beta3_pval:.3e}\n"
        f"R² = {r_squared:.3f}\n"
        f"{'Significant!' if beta3_pval < 0.05 else 'Not significant'}"
    )

    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_penetrance_onset(
    temporal_results: pd.DataFrame,
    onset_dict: Dict[str, float],
    genotype: str,
    time_col: str = 'time_bin',
    figsize: tuple = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Multi-panel visualization of penetrance onset detection.

    Parameters
    ----------
    temporal_results : pd.DataFrame
        Temporal regression results
    onset_dict : dict
        Onset statistics from identify_penetrance_onset()
    genotype : str
        Genotype label
    time_col : str
        Time column
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save

    Returns
    -------
    matplotlib.Figure
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    onset_time = onset_dict['onset_time']

    # Panel 1: R² over time
    ax = axes[0]
    ax.plot(temporal_results[time_col], temporal_results['r_squared'],
           'o-', linewidth=2, markersize=8, color='steelblue')
    ax.axhline(0.3, color='orange', linestyle='--', alpha=0.7, label='Threshold (R²=0.3)')

    if not np.isnan(onset_time):
        ax.axvline(onset_time, color='red', linestyle=':', linewidth=2, alpha=0.7,
                  label=f'Onset ({onset_time} hpf)')

    ax.set_ylabel('R² (Variance Explained)', fontsize=12, fontweight='bold')
    ax.set_title(f'{genotype}: Penetrance Onset Detection', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Panel 2: Slope over time
    ax = axes[1]
    ax.errorbar(temporal_results[time_col], temporal_results['beta1'],
               yerr=temporal_results['beta1_se'],
               fmt='o-', linewidth=2, markersize=8, capsize=5, color='darkgreen')
    ax.axhline(0.05, color='orange', linestyle='--', alpha=0.7, label='Threshold (β₁=0.05)')

    if not np.isnan(onset_time):
        ax.axvline(onset_time, color='red', linestyle=':', linewidth=2, alpha=0.7)

    # Mark significant slopes
    sig_mask = temporal_results['beta1_pval'] < 0.05
    ax.scatter(temporal_results.loc[sig_mask, time_col],
              temporal_results.loc[sig_mask, 'beta1'],
              s=100, facecolors='none', edgecolors='red', linewidths=2,
              label='Significant (p<0.05)')

    ax.set_ylabel('Slope (β₁)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 3: Sample size
    ax = axes[2]
    ax.bar(temporal_results[time_col], temporal_results['n_samples'],
          alpha=0.7, color='gray', edgecolor='black')

    if not np.isnan(onset_time):
        ax.axvline(onset_time, color='red', linestyle=':', linewidth=2, alpha=0.7)

    ax.set_xlabel('Developmental Time (hpf)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sample Size', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


# ============================================================================
# TRAJECTORY PREDICTION VISUALIZATIONS
# ============================================================================

def plot_dual_prediction_heatmaps(
    homo_predictions: pd.DataFrame,
    wt_predictions: pd.DataFrame,
    genotype: str,
    figsize: tuple = (18, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create dual heatmaps showing prediction errors from both models.

    Parameters
    ----------
    homo_predictions : pd.DataFrame
        Predictions from homozygous model
    wt_predictions : pd.DataFrame
        Predictions from WT model
    genotype : str
        Genotype label
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save

    Returns
    -------
    matplotlib.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Prepare heatmap data
    def make_heatmap_data(pred_df):
        # Pivot: rows=embryos, columns=time_i
        heatmap = pred_df.pivot_table(
            index='embryo_id',
            columns='time_i',
            values='prediction_error',
            aggfunc='mean'
        )
        # Sort rows by first available timepoint
        first_time = heatmap.apply(lambda row: row.first_valid_index(), axis=1)
        heatmap = heatmap.loc[first_time.sort_values().index]
        return heatmap

    homo_heatmap = make_heatmap_data(homo_predictions)
    wt_heatmap = make_heatmap_data(wt_predictions)

    # Compute error ratio
    ratio_heatmap = wt_heatmap / homo_heatmap

    # Panel 1: Homozygous model errors
    ax = axes[0]
    im1 = ax.imshow(homo_heatmap.values, aspect='auto', cmap='Reds', vmin=0)
    ax.set_title('Homozygous Model\nPrediction Error', fontsize=12, fontweight='bold')
    ax.set_xlabel('Starting Time (hpf)', fontsize=11)
    ax.set_ylabel('Embryo ID', fontsize=11)
    ax.set_yticks([])
    plt.colorbar(im1, ax=ax, label='Error (distance units)')

    # Panel 2: WT model errors
    ax = axes[1]
    im2 = ax.imshow(wt_heatmap.values, aspect='auto', cmap='Blues', vmin=0)
    ax.set_title('WT Model\nPrediction Error', fontsize=12, fontweight='bold')
    ax.set_xlabel('Starting Time (hpf)', fontsize=11)
    ax.set_ylabel('Embryo ID', fontsize=11)
    ax.set_yticks([])
    plt.colorbar(im2, ax=ax, label='Error (distance units)')

    # Panel 3: Error ratio
    ax = axes[2]
    im3 = ax.imshow(ratio_heatmap.values, aspect='auto', cmap='RdBu_r', norm=plt.Normalize(vmin=0.5, vmax=2))
    ax.set_title('Error Ratio\n(WT / Homo)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Starting Time (hpf)', fontsize=11)
    ax.set_ylabel('Embryo ID', fontsize=11)
    ax.set_yticks([])

    # Add contour at ratio=1.5 (penetrance threshold)
    X, Y = np.meshgrid(range(ratio_heatmap.shape[1]), range(ratio_heatmap.shape[0]))
    CS = ax.contour(X, Y, ratio_heatmap.values, levels=[1.5], colors='green', linewidths=2)
    ax.clabel(CS, inline=True, fontsize=10)

    plt.colorbar(im3, ax=ax, label='Ratio (>1.5 = penetrant)')

    fig.suptitle(f'{genotype}: Trajectory Prediction Errors', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_trajectory_examples(
    df_binned: pd.DataFrame,
    homo_predictions: pd.DataFrame,
    wt_predictions: pd.DataFrame,
    classification: pd.DataFrame,
    genotype: str,
    n_examples: int = 6,
    distance_col: str = 'euclidean_distance',
    time_col: str = 'time_bin',
    figsize: tuple = (16, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot example trajectories showing actual vs predicted distances.

    Parameters
    ----------
    df_binned : pd.DataFrame
        Binned data with actual distances
    homo_predictions : pd.DataFrame
        Homozygous model predictions
    wt_predictions : pd.DataFrame
        WT model predictions
    classification : pd.DataFrame
        Penetrance classification
    genotype : str
        Genotype label
    n_examples : int
        Number of examples to show
    distance_col : str
        Distance column name
    time_col : str
        Time column name
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save

    Returns
    -------
    matplotlib.Figure
    """
    # Select examples: 2 penetrant, 2 non-penetrant, 2 intermediate
    penetrant = classification[classification['penetrance_status'] == 'penetrant'].sort_values('confidence', ascending=False)
    non_penetrant = classification[classification['penetrance_status'] == 'non-penetrant'].sort_values('confidence', ascending=False)
    intermediate = classification[classification['penetrance_status'] == 'intermediate']

    n_per_class = n_examples // 3
    selected_embryos = []

    if len(penetrant) > 0:
        selected_embryos.extend(penetrant['embryo_id'].iloc[:n_per_class].tolist())
    if len(non_penetrant) > 0:
        selected_embryos.extend(non_penetrant['embryo_id'].iloc[:n_per_class].tolist())
    if len(intermediate) > 0:
        selected_embryos.extend(intermediate['embryo_id'].iloc[:n_per_class].tolist())

    n_plots = len(selected_embryos)
    ncols = 3
    nrows = int(np.ceil(n_plots / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten() if n_plots > 1 else [axes]

    for idx, embryo_id in enumerate(selected_embryos):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Get actual trajectory
        embryo_data = df_binned[(df_binned['embryo_id'] == embryo_id) & (df_binned['genotype'] == genotype)]
        actual_times = embryo_data[time_col].values
        actual_distances = embryo_data[distance_col].values

        # Get predictions
        homo_pred = homo_predictions[homo_predictions['embryo_id'] == embryo_id]
        wt_pred = wt_predictions[wt_predictions['embryo_id'] == embryo_id]

        # Plot actual
        ax.plot(actual_times, actual_distances, 'o-', color='black', linewidth=2, markersize=6, label='Actual')

        # Plot predictions (connect starting time to predicted future time)
        for _, row in homo_pred.iterrows():
            ax.plot([row['time_i'], row['actual_time_future']],
                   [actual_distances[actual_times == row['time_i']][0] if np.any(actual_times == row['time_i']) else np.nan, row['predicted_distance']],
                   '--', color='red', alpha=0.3)

        for _, row in wt_pred.iterrows():
            ax.plot([row['time_i'], row['actual_time_future']],
                   [actual_distances[actual_times == row['time_i']][0] if np.any(actual_times == row['time_i']) else np.nan, row['predicted_distance']],
                   '--', color='blue', alpha=0.3)

        # Get classification
        embryo_class = classification[classification['embryo_id'] == embryo_id]
        if len(embryo_class) > 0:
            status = embryo_class['penetrance_status'].iloc[0]
            error_ratio = embryo_class['error_ratio'].iloc[0]
            title = f"{embryo_id}\n{status} (ratio={error_ratio:.2f})"
        else:
            title = embryo_id

        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(['Actual', 'Homo pred', 'WT pred'], fontsize=8)

    # Remove empty axes
    for idx in range(n_plots, len(axes)):
        fig.delaxes(axes[idx])

    fig.text(0.5, 0.02, 'Developmental Time (hpf)', ha='center', fontsize=12, fontweight='bold')
    fig.text(0.02, 0.5, 'Distance from WT', va='center', rotation='vertical', fontsize=12, fontweight='bold')
    fig.suptitle(f'{genotype}: Trajectory Prediction Examples', fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_prediction_error_scatter(
    classification: pd.DataFrame,
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Scatter plot of homo vs WT model errors.

    Parameters
    ----------
    classification : pd.DataFrame
        Classification results
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save

    Returns
    -------
    matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Color by penetrance status
    colors = {'penetrant': 'red', 'non-penetrant': 'blue', 'intermediate': 'gray'}

    for status, color in colors.items():
        subset = classification[classification['penetrance_status'] == status]
        ax.scatter(subset['mean_error_homo'], subset['mean_error_wt'],
                  alpha=0.6, s=100, color=color, edgecolors='k', linewidths=0.5,
                  label=f"{status} (n={len(subset)})")

    # Diagonal line (ratio = 1)
    max_error = max(classification['mean_error_homo'].max(), classification['mean_error_wt'].max())
    ax.plot([0, max_error], [0, max_error], 'k--', alpha=0.5, label='Ratio=1')

    # Threshold lines (ratio = 1.5 and 0.67)
    x_range = np.linspace(0, max_error, 100)
    ax.plot(x_range, 1.5 * x_range, 'r:', alpha=0.5, label='Ratio=1.5 (penetrant)')
    ax.plot(x_range, 0.67 * x_range, 'b:', alpha=0.5, label='Ratio=0.67 (non-penetrant)')

    ax.set_xlabel('Homozygous Model Error', fontsize=12, fontweight='bold')
    ax.set_ylabel('WT Model Error', fontsize=12, fontweight='bold')
    ax.set_title('Trajectory Prediction Error Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_penetrance_distribution(
    classification: pd.DataFrame,
    genotype: str,
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot distribution of penetrance classifications.

    Parameters
    ----------
    classification : pd.DataFrame
        Classification results
    genotype : str
        Genotype label
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save

    Returns
    -------
    matplotlib.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Panel 1: Bar chart of classifications
    ax = axes[0]
    status_counts = classification['penetrance_status'].value_counts()
    colors_map = {'penetrant': 'red', 'non-penetrant': 'blue', 'intermediate': 'gray'}
    colors_list = [colors_map.get(s, 'gray') for s in status_counts.index]

    bars = ax.bar(range(len(status_counts)), status_counts.values, color=colors_list, edgecolor='black', alpha=0.7)
    ax.set_xticks(range(len(status_counts)))
    ax.set_xticklabels(status_counts.index, rotation=45, ha='right')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Penetrance Classification', fontsize=14, fontweight='bold')

    # Add percentages
    total = status_counts.sum()
    for bar, count in zip(bars, status_counts.values):
        pct = 100 * count / total
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10)

    ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: Histogram of error ratios
    ax = axes[1]
    ax.hist(classification['error_ratio'], bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(1.5, color='red', linestyle='--', linewidth=2, label='Penetrant threshold')
    ax.axvline(0.67, color='blue', linestyle='--', linewidth=2, label='Non-penetrant threshold')
    ax.axvline(1.0, color='gray', linestyle=':', linewidth=2, label='Equal error')

    ax.set_xlabel('Error Ratio (WT / Homo)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Error Ratio Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'{genotype}: Penetrance Classification Summary', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig
