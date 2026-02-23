"""
Visualization functions for LOEO trajectory prediction analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def create_aggregated_heatmap(
    predictions: pd.DataFrame,
    error_metric: str = 'absolute_error',
    aggregation: str = 'mean'
) -> pd.DataFrame:
    """
    Create aggregated (FROM_time × TO_time) heatmap from predictions.

    Parameters
    ----------
    predictions : pd.DataFrame
        Predictions with from_time, to_time, and error columns
    error_metric : str
        Column name for error metric ('absolute_error', 'relative_error')
    aggregation : str
        Aggregation method ('mean', 'median', 'std')

    Returns
    -------
    pd.DataFrame
        Heatmap with:
        - Index: FROM times (rows)
        - Columns: TO times (columns)
        - Values: aggregated error
        - Lower triangle: NaN (TO must be > FROM)
    """
    # Get unique times
    from_times = sorted(predictions['from_time'].unique())
    to_times = sorted(predictions['to_time'].unique())

    # Initialize heatmap
    heatmap = pd.DataFrame(index=from_times, columns=to_times, dtype=float)

    # Fill heatmap
    for from_t in from_times:
        for to_t in to_times:
            if to_t <= from_t:
                # Lower triangle (invalid predictions)
                heatmap.loc[from_t, to_t] = np.nan
            else:
                # Get predictions for this cell
                mask = (predictions['from_time'] == from_t) & (predictions['to_time'] == to_t)
                cell_data = predictions.loc[mask, error_metric]

                if len(cell_data) > 0:
                    if aggregation == 'mean':
                        heatmap.loc[from_t, to_t] = cell_data.mean()
                    elif aggregation == 'median':
                        heatmap.loc[from_t, to_t] = cell_data.median()
                    elif aggregation == 'std':
                        heatmap.loc[from_t, to_t] = cell_data.std()
                else:
                    heatmap.loc[from_t, to_t] = np.nan

    return heatmap


def create_per_embryo_heatmaps(
    predictions: pd.DataFrame,
    error_metric: str = 'absolute_error'
) -> Dict[str, pd.DataFrame]:
    """
    Create individual heatmap for each embryo.

    Parameters
    ----------
    predictions : pd.DataFrame
        Predictions DataFrame
    error_metric : str
        Error column name

    Returns
    -------
    dict
        {embryo_id: heatmap_df}
    """
    embryo_heatmaps = {}

    for embryo_id in predictions['embryo_id'].unique():
        embryo_preds = predictions[predictions['embryo_id'] == embryo_id]
        heatmap = create_aggregated_heatmap(embryo_preds, error_metric, aggregation='mean')
        embryo_heatmaps[embryo_id] = heatmap

    return embryo_heatmaps


def compute_r2_per_cell(
    predictions: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute R² for each (from_time, to_time) cell.

    Parameters
    ----------
    predictions : pd.DataFrame
        Predictions DataFrame

    Returns
    -------
    pd.DataFrame
        Heatmap with R² values
    """
    from sklearn.metrics import r2_score

    from_times = sorted(predictions['from_time'].unique())
    to_times = sorted(predictions['to_time'].unique())

    heatmap = pd.DataFrame(index=from_times, columns=to_times, dtype=float)

    for from_t in from_times:
        for to_t in to_times:
            if to_t <= from_t:
                heatmap.loc[from_t, to_t] = np.nan
            else:
                mask = (predictions['from_time'] == from_t) & (predictions['to_time'] == to_t)
                cell_data = predictions[mask]

                if len(cell_data) >= 3:  # Need at least 3 points for R²
                    y_true = cell_data['actual_distance'].values
                    y_pred = cell_data['predicted_distance'].values

                    if np.var(y_true) > 1e-8:
                        r2 = r2_score(y_true, y_pred)
                        heatmap.loc[from_t, to_t] = r2
                    else:
                        heatmap.loc[from_t, to_t] = np.nan
                else:
                    heatmap.loc[from_t, to_t] = np.nan

    return heatmap


def plot_aggregated_heatmap(
    heatmap_df: pd.DataFrame,
    genotype: str,
    model_name: str,
    test_genotype: str,
    error_metric: str,
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = 'viridis',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot triangular heatmap showing prediction error landscape.

    Parameters
    ----------
    heatmap_df : pd.DataFrame
        Heatmap from create_aggregated_heatmap() or compute_r2_per_cell()
    genotype : str
        Genotype name for title
    model_name : str
        Model name (e.g., 'WT model')
    test_genotype : str
        Test genotype name
    error_metric : str
        Metric name for labeling ('absolute_error', 'relative_error', 'r2')
    figsize : tuple
        Figure size
    cmap : str
        Colormap name
    save_path : str, optional
        Path to save figure

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create mask for lower triangle
    mask = np.tril(np.ones_like(heatmap_df, dtype=bool), k=0)

    # Plot heatmap
    sns.heatmap(
        heatmap_df,
        mask=mask,
        cmap=cmap,
        cbar_kws={'label': error_metric.replace('_', ' ').title()},
        square=False,
        ax=ax,
        fmt='.3f'
    )

    # Labels
    ax.set_xlabel('TO time (hpf)', fontsize=12, fontweight='bold')
    ax.set_ylabel('FROM time (hpf)', fontsize=12, fontweight='bold')

    # Title
    title = f'{model_name} tested on {test_genotype}\n'
    title += f'Prediction Error Landscape ({error_metric.replace("_", " ").title()})'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_per_embryo_grid(
    embryo_heatmaps: Dict[str, pd.DataFrame],
    genotype: str,
    model_name: str,
    test_genotype: str,
    error_metric: str,
    n_cols: int = 4,
    figsize_per_subplot: Tuple[float, float] = (3, 3),
    cmap: str = 'viridis',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Grid of small heatmaps, one per embryo.

    Parameters
    ----------
    embryo_heatmaps : dict
        {embryo_id: heatmap_df} from create_per_embryo_heatmaps()
    genotype : str
        Genotype name
    model_name : str
        Model name
    test_genotype : str
        Test genotype name
    error_metric : str
        Error metric name
    n_cols : int
        Number of columns in grid
    figsize_per_subplot : tuple
        Size of each subplot
    cmap : str
        Colormap
    save_path : str, optional
        Save path

    Returns
    -------
    plt.Figure
    """
    n_embryos = len(embryo_heatmaps)
    n_rows = int(np.ceil(n_embryos / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_subplot[0] * n_cols, figsize_per_subplot[1] * n_rows)
    )

    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Get global vmin/vmax for consistent color scale
    all_values = []
    for heatmap in embryo_heatmaps.values():
        all_values.extend(heatmap.values.flatten())
    all_values = [v for v in all_values if not np.isnan(v)]

    if len(all_values) > 0:
        vmin, vmax = np.percentile(all_values, [5, 95])
    else:
        vmin, vmax = 0, 1

    # Plot each embryo
    for idx, (embryo_id, heatmap) in enumerate(embryo_heatmaps.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Create mask
        mask = np.tril(np.ones_like(heatmap, dtype=bool), k=0)

        # Plot
        sns.heatmap(
            heatmap,
            mask=mask,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            square=False,
            ax=ax,
            xticklabels=False,
            yticklabels=False
        )

        ax.set_title(embryo_id, fontsize=8)
        ax.set_xlabel('')
        ax.set_ylabel('')

    # Hide empty subplots
    for idx in range(n_embryos, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    # Overall title
    fig.suptitle(
        f'Per-Embryo Heatmaps: {model_name} on {test_genotype}',
        fontsize=14,
        fontweight='bold'
    )

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label(error_metric.replace('_', ' ').title(), fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_error_vs_horizon(
    error_vs_dt: pd.DataFrame,
    genotype: str,
    model_name: str,
    test_genotype: str,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Line plot: prediction horizon (delta_t) vs error.

    Parameters
    ----------
    error_vs_dt : pd.DataFrame
        From compute_error_vs_horizon()
    genotype : str
        Genotype name
    model_name : str
        Model name
    test_genotype : str
        Test genotype
    figsize : tuple
        Figure size
    save_path : str, optional
        Save path

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Panel 1: Absolute error
    ax = axes[0]
    ax.errorbar(
        error_vs_dt['delta_t'],
        error_vs_dt['mean_abs_error'],
        yerr=error_vs_dt['std_abs_error'],
        marker='o',
        linewidth=2,
        capsize=5,
        label='Mean ± Std'
    )
    ax.set_xlabel('Prediction Horizon Δt (hpf)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Absolute Error', fontsize=11, fontweight='bold')
    ax.set_title('Absolute Error vs Horizon', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()

    # Panel 2: Relative error
    ax = axes[1]
    ax.errorbar(
        error_vs_dt['delta_t'],
        error_vs_dt['mean_rel_error'],
        yerr=error_vs_dt['std_rel_error'],
        marker='o',
        linewidth=2,
        capsize=5,
        label='Mean ± Std',
        color='orange'
    )
    ax.set_xlabel('Prediction Horizon Δt (hpf)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Relative Error', fontsize=11, fontweight='bold')
    ax.set_title('Relative Error vs Horizon', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()

    fig.suptitle(
        f'{model_name} on {test_genotype}',
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_temporal_breakdown(
    predictions: pd.DataFrame,
    genotype: str,
    model_name: str,
    test_genotype: str,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    For each FROM time, show error distribution.

    Parameters
    ----------
    predictions : pd.DataFrame
        Predictions DataFrame
    genotype : str
        Genotype name
    model_name : str
        Model name
    test_genotype : str
        Test genotype
    figsize : tuple
        Figure size
    save_path : str, optional
        Save path

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Box plot by FROM time
    from_times = sorted(predictions['from_time'].unique())

    data_to_plot = [predictions[predictions['from_time'] == t]['absolute_error'].values
                    for t in from_times]

    bp = ax.boxplot(data_to_plot, positions=from_times, widths=1.5, patch_artist=True)

    # Color boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    ax.set_xlabel('FROM Time (hpf)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Absolute Error', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Error Distribution by Starting Time: {model_name} on {test_genotype}',
        fontsize=14,
        fontweight='bold'
    )
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_per_embryo_error_distribution(
    per_embryo_metrics: pd.DataFrame,
    genotype: str,
    model_name: str,
    test_genotype: str,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Bar plot of per-embryo mean errors.

    Parameters
    ----------
    per_embryo_metrics : pd.DataFrame
        From compute_per_embryo_metrics()
    genotype : str
        Genotype name
    model_name : str
        Model name
    test_genotype : str
        Test genotype
    figsize : tuple
        Figure size
    save_path : str, optional
        Save path

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Sort by mean error
    df_sorted = per_embryo_metrics.sort_values('mean_abs_error')

    # Bar plot
    x = np.arange(len(df_sorted))
    ax.bar(x, df_sorted['mean_abs_error'], alpha=0.7, color='steelblue')

    # Population mean line
    mean_error = df_sorted['mean_abs_error'].mean()
    std_error = df_sorted['mean_abs_error'].std()

    ax.axhline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.3f}')
    ax.axhspan(mean_error - std_error, mean_error + std_error, alpha=0.2, color='red', label='±1 Std')

    ax.set_xlabel('Embryo (sorted by error)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error', fontsize=11, fontweight='bold')
    ax.set_title(
        f'Per-Embryo Error Distribution: {model_name} on {test_genotype}',
        fontsize=13,
        fontweight='bold'
    )
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Remove x-ticks (too many embryos)
    ax.set_xticks([])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_model_comparison_3x3(
    predictions_dict: Dict[str, pd.DataFrame],
    gene: str,
    figsize: Tuple[int, int] = (20, 16),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    3×3 grid comparing all model-test combinations for one gene.

    Rows: Model trained on (WT, Het, Homo)
    Cols: Tested on (WT, Het, Homo)

    Each cell: Aggregated heatmap (FROM × TO)
    Diagonal cells highlighted (LOEO).

    Parameters
    ----------
    predictions_dict : dict
        {combo_key: predictions_df}
        combo_key format: '{model_geno}_model_on_{test_geno}'
    gene : str
        Gene name (e.g., 'cep290')
    figsize : tuple
        Figure size
    save_path : str, optional
        Save path

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(3, 3, figsize=figsize)

    model_genos = ['wt', 'het', 'homo']
    test_genos = ['wt', 'het', 'homo']

    # Find global min/max for unified color scale
    all_errors = []
    for mg in model_genos:
        for tg in test_genos:
            combo_key = f'{mg}_model_on_{tg}'
            if combo_key in predictions_dict:
                preds = predictions_dict[combo_key]
                all_errors.extend(preds['absolute_error'].dropna().values)

    if len(all_errors) > 0:
        vmin, vmax = np.percentile(all_errors, [5, 95])
    else:
        vmin, vmax = 0, 1

    # Plot each combination
    for i, model_geno in enumerate(model_genos):
        for j, test_geno in enumerate(test_genos):
            ax = axes[i, j]

            combo_key = f'{model_geno}_model_on_{test_geno}'

            if combo_key not in predictions_dict:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            preds = predictions_dict[combo_key]

            # Create heatmap
            heatmap = create_aggregated_heatmap(preds, 'absolute_error')

            # Create mask
            mask = np.tril(np.ones_like(heatmap, dtype=bool), k=0)

            # Plot
            sns.heatmap(
                heatmap,
                mask=mask,
                cmap='viridis',
                vmin=vmin,
                vmax=vmax,
                cbar=False,
                square=False,
                ax=ax,
                xticklabels=False,
                yticklabels=False
            )

            # Title
            title = f'{model_geno.upper()} model\n→ {test_geno.upper()} test'
            ax.set_title(title, fontweight='bold' if model_geno == test_geno else 'normal', fontsize=10)

            # Highlight diagonal
            if model_geno == test_geno:
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(4)

            # Axis labels (only on edges)
            if j == 0:
                ax.set_ylabel('FROM time', fontsize=9)
            if i == 2:
                ax.set_xlabel('TO time', fontsize=9)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, label='Mean Absolute Error', shrink=0.6, pad=0.02)

    # Overall title
    fig.suptitle(
        f'{gene.upper()} Model Comparison\n(Red border = LOEO, Others = Cross-genotype)',
        fontsize=18,
        fontweight='bold',
        y=0.98
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_penetrance_classification(
    classification: pd.DataFrame,
    gene: str,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize penetrance classification results.

    Parameters
    ----------
    classification : pd.DataFrame
        From classify_penetrance_dual_model()
    gene : str
        Gene name
    figsize : tuple
        Figure size
    save_path : str, optional
        Save path

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel 1: Bar chart of classification counts
    ax = axes[0]
    status_counts = classification['penetrance_status'].value_counts()
    colors = {'penetrant': 'red', 'non-penetrant': 'green', 'intermediate': 'orange'}
    status_order = ['penetrant', 'non-penetrant', 'intermediate']

    bars = ax.bar(
        range(len(status_order)),
        [status_counts.get(s, 0) for s in status_order],
        color=[colors[s] for s in status_order],
        alpha=0.7
    )

    ax.set_xticks(range(len(status_order)))
    ax.set_xticklabels([s.capitalize() for s in status_order], rotation=45, ha='right')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title('Penetrance Classification', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add counts on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')

    # Panel 2: Histogram of error ratios
    ax = axes[1]
    ax.hist(classification['error_ratio'], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(1.5, color='red', linestyle='--', linewidth=2, label='Threshold (1.5)')
    ax.axvline(1/1.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.67)')
    ax.set_xlabel('Error Ratio (WT / Homo)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax.set_title('Error Ratio Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Panel 3: Scatter plot of WT vs Homo errors
    ax = axes[2]
    for status, color in colors.items():
        subset = classification[classification['penetrance_status'] == status]
        ax.scatter(
            subset['mean_error_homo_model'],
            subset['mean_error_wt_model'],
            c=color,
            label=status.capitalize(),
            alpha=0.7,
            s=50
        )

    # Diagonal line (equal errors)
    max_val = max(classification['mean_error_homo_model'].max(),
                  classification['mean_error_wt_model'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Equal Error')

    ax.set_xlabel('Homo Model Error', fontsize=11, fontweight='bold')
    ax.set_ylabel('WT Model Error', fontsize=11, fontweight='bold')
    ax.set_title('Model Error Comparison', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(
        f'{gene.upper()} Penetrance Classification',
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
