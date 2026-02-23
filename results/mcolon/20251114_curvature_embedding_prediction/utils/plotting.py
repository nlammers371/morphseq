"""
Plotting utilities for regression analysis visualization.

Creates diagnostic plots for model evaluation, feature importance,
and prediction accuracy.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path


def plot_predictions_vs_actual(
    predictions_df: pd.DataFrame,
    title: str = 'Predictions vs Actual',
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[Path] = None,
    show_genotype: bool = True,
    genotype_col: str = 'genotype',
    n_test_samples: int = None,
    n_test_embryos: int = None
) -> plt.Figure:
    """
    Create scatter plot of predicted vs actual target values (holdout test set).

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Must contain 'actual' and 'predicted' columns
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : Path, optional
        Save plot to this path
    show_genotype : bool
        Color by genotype if available
    genotype_col : str
        Column name for genotype
    n_test_samples : int, optional
        Number of test samples for subtitle
    n_test_embryos : int, optional
        Number of test embryos for subtitle

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if show_genotype and genotype_col in predictions_df.columns:
        # Color by genotype
        genotypes = predictions_df[genotype_col].unique()
        colors = sns.color_palette('husl', len(genotypes))
        genotype_colors = dict(zip(genotypes, colors))

        for genotype in genotypes:
            mask = predictions_df[genotype_col] == genotype
            subset = predictions_df[mask]

            ax.scatter(
                subset['actual'],
                subset['predicted'],
                alpha=0.6,
                s=50,
                label=genotype,
                color=genotype_colors[genotype]
            )
    else:
        ax.scatter(
            predictions_df['actual'],
            predictions_df['predicted'],
            alpha=0.6,
            s=50
        )

    # Perfect prediction line
    all_vals = np.concatenate([predictions_df['actual'].values, predictions_df['predicted'].values])
    lims = [all_vals.min(), all_vals.max()]
    ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=2)

    ax.set_xlabel('Actual', fontsize=11)
    ax.set_ylabel('Predicted', fontsize=11)

    # Build title with holdout info and R²
    from sklearn.metrics import r2_score
    n_samples = len(predictions_df)
    n_embryos = predictions_df['embryo_id'].nunique() if 'embryo_id' in predictions_df.columns else None
    r2 = r2_score(predictions_df['actual'], predictions_df['predicted'])

    subtitle = f'{n_embryos} held-out embryos ({n_samples} samples) | R² = {r2:.3f}'

    ax.set_title(f'{title}\n{subtitle}', fontweight='bold', fontsize=12)
    ax.grid(alpha=0.3)

    if show_genotype and genotype_col in predictions_df.columns:
        ax.legend(fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {save_path}")

    return fig


def plot_residuals(
    predictions_df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create residual diagnostic plots.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        With 'actual' and 'predicted' columns
    figsize : tuple
    save_path : Path, optional

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    residuals = predictions_df['residual'].values

    # 1. Histogram
    ax = axes[0, 0]
    ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Residual', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Residual Distribution', fontweight='bold')
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)

    # 2. Q-Q plot
    ax = axes[0, 1]
    sorted_residuals = np.sort(residuals)
    theoretical_quantiles = np.sort(np.random.normal(0, np.std(residuals), len(residuals)))
    ax.scatter(theoretical_quantiles, sorted_residuals, alpha=0.6)
    ax.set_xlabel('Theoretical Quantiles', fontsize=11)
    ax.set_ylabel('Sample Quantiles', fontsize=11)
    ax.set_title('Q-Q Plot', fontweight='bold')
    ax.grid(alpha=0.3)

    # 3. Residuals vs index
    ax = axes[1, 0]
    ax.scatter(np.arange(len(residuals)), residuals, alpha=0.6, s=20)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Sample Index', fontsize=11)
    ax.set_ylabel('Residual', fontsize=11)
    ax.set_title('Residuals vs Index', fontweight='bold')
    ax.grid(alpha=0.3)

    # 4. Box plot by quartile of prediction
    ax = axes[1, 1]
    pred_quartiles = pd.qcut(predictions_df['predicted'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    residual_by_quartile = [residuals[pred_quartiles == q] for q in ['Q1', 'Q2', 'Q3', 'Q4']]

    bp = ax.boxplot(residual_by_quartile, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Prediction Quartile', fontsize=11)
    ax.set_ylabel('Residual', fontsize=11)
    ax.set_title('Residuals by Prediction Quartile', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Residual Diagnostics', fontweight='bold', fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {save_path}")

    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create bar plot of feature importance.

    Parameters
    ----------
    importance_df : pd.DataFrame
        With 'feature' and 'importance' columns
    top_n : int
        Show top N features
    figsize : tuple
    save_path : Path, optional

    Returns
    -------
    plt.Figure
    """
    df_plot = importance_df.nlargest(top_n, 'importance').copy()

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(range(len(df_plot)), df_plot['importance'].values)
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(df_plot['feature'].values)
    ax.set_xlabel('Importance', fontsize=11)
    ax.set_title(f'Top {top_n} Feature Importance', fontweight='bold', fontsize=12)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {save_path}")

    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metrics: List[str] = ['r2', 'mae', 'rmse'],
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create multi-panel comparison of model performance.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        With 'model' column and metric columns
    metrics : list of str
        Metrics to plot
    figsize : tuple
    save_path : Path, optional

    Returns
    -------
    plt.Figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        df_sorted = comparison_df.sort_values(metric)

        colors = sns.color_palette('RdYlGn_r', len(df_sorted))
        ax.barh(range(len(df_sorted)), df_sorted[metric].values, color=colors)

        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted['model'].values)
        ax.set_xlabel(metric.upper(), fontsize=11)
        ax.set_title(f'{metric.upper()}', fontweight='bold', fontsize=12)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

    plt.suptitle('Model Comparison', fontweight='bold', fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {save_path}")

    return fig


def plot_metrics_table(
    metrics_dict: Dict[str, float],
    figsize: Tuple[int, int] = (8, 4),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create a text-based table of metrics as a plot.

    Parameters
    ----------
    metrics_dict : dict
        {metric_name: value, ...}
    figsize : tuple
    save_path : Path, optional

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    # Format metrics
    cell_text = []
    for key, val in metrics_dict.items():
        if isinstance(val, float):
            cell_text.append([key, f'{val:.4f}'])
        else:
            cell_text.append([key, str(val)])

    table = ax.table(
        cellText=cell_text,
        colLabels=['Metric', 'Value'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(cell_text) + 1):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('white')

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {save_path}")

    return fig
