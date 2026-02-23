"""
Visualization functions for simple trajectory prediction analysis.

Updated to work with per-horizon models (separate model for each target time).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def plot_model_comparison_curves(
    results_dict: Dict[str, Dict[float, Dict]],
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot error vs prediction horizon curves for all models.

    Parameters
    ----------
    results_dict : dict
        {
            'linear': {32.0: result, 34.0: result, ...},
            'ridge': {32.0: result, 34.0: result, ...},
            ...
        }
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    plt.Figure
    """
    from .simple_trajectory import aggregate_metrics_across_horizons

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Panel 1: MAE vs horizon
    ax = axes[0]
    for model_type, results_by_horizon in results_dict.items():
        df_metrics = aggregate_metrics_across_horizons(results_by_horizon)

        ax.plot(
            df_metrics['horizon'],
            df_metrics['mae'],
            marker='o',
            linewidth=2,
            label=model_type,
            alpha=0.8
        )

    ax.set_xlabel('Prediction Horizon (hours ahead)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison: Error vs Prediction Horizon', fontsize=13, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: R² vs horizon
    ax = axes[1]
    for model_type, results_by_horizon in results_dict.items():
        df_metrics = aggregate_metrics_across_horizons(results_by_horizon)

        ax.plot(
            df_metrics['horizon'],
            df_metrics['r2'],
            marker='o',
            linewidth=2,
            label=model_type,
            alpha=0.8
        )

    ax.set_xlabel('Prediction Horizon (hours ahead)', fontsize=12, fontweight='bold')
    ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison: R² vs Prediction Horizon', fontsize=13, fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_trajectory_examples(
    results_dict: Dict[str, Dict[float, Dict]],
    top_models: List[str],
    n_examples: int = 6,
    figsize: Tuple[int, int] = (18, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot example trajectories for top models.

    Assembles predictions from separate per-horizon models to show
    full trajectory curves.

    Parameters
    ----------
    results_dict : dict
        {model_type: {target_time: result}}
    top_models : list
        List of top model names to plot
    n_examples : int
        Number of example embryos to show
    figsize : tuple
        Figure size
    save_path : str, optional
        Save path

    Returns
    -------
    plt.Figure
    """
    # Get example embryos from first model
    first_model_type = top_models[0]
    first_horizon_results = list(results_dict[first_model_type].values())[0]
    all_embryo_ids = first_horizon_results['predictions']['embryo_id'].unique()

    # Sample embryos
    if len(all_embryo_ids) > n_examples:
        np.random.seed(42)
        selected_embryos = np.random.choice(all_embryo_ids, n_examples, replace=False)
    else:
        selected_embryos = all_embryo_ids[:n_examples]

    # Create subplots
    n_rows = 2
    n_cols = int(np.ceil(n_examples / n_rows))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    colors = plt.cm.tab10(range(len(top_models)))

    for idx, embryo_id in enumerate(selected_embryos):
        ax = axes[idx]

        # Assemble actual trajectory across all horizons
        actual_times = []
        actual_distances = []

        for target_time in sorted(results_dict[first_model_type].keys()):
            preds = results_dict[first_model_type][target_time]['predictions']
            embryo_data = preds[preds['embryo_id'] == embryo_id]

            if len(embryo_data) > 0:
                actual_times.append(embryo_data['target_time'].iloc[0])
                actual_distances.append(embryo_data['target_distance'].iloc[0])

        # Plot actual trajectory
        ax.plot(
            actual_times,
            actual_distances,
            color='black',
            linewidth=3,
            label='Actual',
            marker='o',
            markersize=6
        )

        # Plot predictions from each top model
        for i, model_type in enumerate(top_models):
            pred_times = []
            pred_distances = []

            for target_time in sorted(results_dict[model_type].keys()):
                preds = results_dict[model_type][target_time]['predictions']
                embryo_data = preds[preds['embryo_id'] == embryo_id]

                if len(embryo_data) > 0:
                    pred_times.append(embryo_data['target_time'].iloc[0])
                    pred_distances.append(embryo_data['predicted_distance'].iloc[0])

            if len(pred_times) > 0:
                ax.plot(
                    pred_times,
                    pred_distances,
                    color=colors[i],
                    linewidth=2,
                    label=model_type,
                    linestyle='--',
                    alpha=0.7
                )

        # Mark start time (30 hpf)
        if len(actual_times) > 0:
            start_time = actual_times[0] - (actual_times[1] - actual_times[0]) if len(actual_times) > 1 else 30.0
            ax.axvline(start_time, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Start (30 hpf)')

        ax.set_xlabel('Time (hpf)', fontsize=10)
        ax.set_ylabel('Distance from WT', fontsize=10)
        ax.set_title(f'{embryo_id}', fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3)

        if idx == 0:
            ax.legend(fontsize=8, loc='best')

    # Hide empty subplots
    for idx in range(len(selected_embryos), len(axes)):
        axes[idx].axis('off')

    fig.suptitle(
        f'Example Trajectories: Actual vs Predicted (Top {len(top_models)} Models)',
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_model_performance_heatmap(
    results_dict: Dict[str, Dict[float, Dict]],
    figsize: Tuple[int, int] = (14, 8),
    cmap: str = 'RdBu_r',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Heatmap: Model × Horizon showing prediction error.

    Parameters
    ----------
    results_dict : dict
        {model_type: {target_time: result}}
    figsize : tuple
        Figure size
    cmap : str
        Colormap (RED=bad, BLUE=good)
    save_path : str, optional
        Save path

    Returns
    -------
    plt.Figure
    """
    from .simple_trajectory import aggregate_metrics_across_horizons

    # Collect MAE by horizon for each model
    model_types = list(results_dict.keys())
    all_horizons = set()

    mae_data = {}
    for model_type, results_by_horizon in results_dict.items():
        df_metrics = aggregate_metrics_across_horizons(results_by_horizon)
        mae_data[model_type] = df_metrics.set_index('horizon')['mae']
        all_horizons.update(df_metrics['horizon'].values)

    # Create matrix
    all_horizons = sorted(all_horizons)
    heatmap_data = pd.DataFrame(index=model_types, columns=all_horizons, dtype=float)

    for model_type in model_types:
        for horizon in all_horizons:
            if horizon in mae_data[model_type].index:
                heatmap_data.loc[model_type, horizon] = mae_data[model_type].loc[horizon]

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        heatmap_data.astype(float),
        cmap=cmap,
        cbar_kws={'label': 'Mean Absolute Error (BLUE=good, RED=bad)'},
        fmt='.2f',
        linewidths=0.5,
        ax=ax,
        vmin=0,
        annot=True,
        annot_kws={'fontsize': 8}
    )

    ax.set_xlabel('Prediction Horizon (hours ahead)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model Type', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Heatmap: MAE Across Prediction Horizons', fontsize=13, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_error_distributions(
    results_dict: Dict[str, Dict[float, Dict]],
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Boxplots of prediction errors per model (aggregated across all horizons).

    Parameters
    ----------
    results_dict : dict
        {model_type: {target_time: result}}
    figsize : tuple
        Figure size
    save_path : str, optional
        Save path

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data for boxplot
    error_data = []
    labels = []

    for model_type, results_by_horizon in results_dict.items():
        # Aggregate errors across all horizons
        all_errors = []
        for target_time, result in results_by_horizon.items():
            if len(result['predictions']) > 0:
                all_errors.extend(result['predictions']['absolute_error'].values)

        if len(all_errors) > 0:
            error_data.append(all_errors)
            labels.append(model_type)

    # Create boxplot
    bp = ax.boxplot(
        error_data,
        labels=labels,
        patch_artist=True,
        widths=0.6
    )

    # Color boxes
    colors = plt.cm.Set3(range(len(labels)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Model Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Absolute Error', fontsize=12, fontweight='bold')
    ax.set_title('Error Distribution by Model (All Horizons)', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_model_ranking_table(
    comparison_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Styled table showing model rankings.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        From compare_models_across_horizons() - long form with all model×horizon combos
    figsize : tuple
        Figure size
    save_path : str, optional
        Save path

    Returns
    -------
    plt.Figure
    """
    # Compute average performance per model
    avg_metrics = comparison_df.groupby('model_type').agg({
        'mae': 'mean',
        'rmse': 'mean',
        'r2': 'mean',
        'n_predictions': 'sum'
    }).reset_index()

    avg_metrics = avg_metrics.sort_values('mae')
    avg_metrics['rank'] = range(1, len(avg_metrics) + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    # Prepare data for display
    display_df = avg_metrics[['rank', 'model_type', 'mae', 'rmse', 'r2', 'n_predictions']].copy()
    display_df.columns = ['Rank', 'Model', 'Avg MAE', 'Avg RMSE', 'Avg R²', 'Total Predictions']

    # Format numbers
    display_df['Avg MAE'] = display_df['Avg MAE'].map('{:.4f}'.format)
    display_df['Avg RMSE'] = display_df['Avg RMSE'].map('{:.4f}'.format)
    display_df['Avg R²'] = display_df['Avg R²'].map('{:.4f}'.format)

    # Create table
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.1, 0.25, 0.15, 0.15, 0.15, 0.2]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(display_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Highlight top 3
    colors = ['#FFD700', '#C0C0C0', '#CD7F32']  # Gold, Silver, Bronze
    for i in range(min(3, len(display_df))):
        for j in range(len(display_df.columns)):
            table[(i+1, j)].set_facecolor(colors[i])
            table[(i+1, j)].set_alpha(0.3)

    ax.set_title('Model Performance Ranking (Averaged Across All Horizons)', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_best_model_per_horizon(
    best_per_horizon_df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot which model performs best at each prediction horizon.

    Parameters
    ----------
    best_per_horizon_df : pd.DataFrame
        From get_best_model_per_horizon()
    figsize : tuple
        Figure size
    save_path : str, optional
        Save path

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get unique models and assign colors
    unique_models = best_per_horizon_df['best_model'].unique()
    color_map = {model: plt.cm.tab10(i) for i, model in enumerate(unique_models)}

    # Plot bars colored by best model
    for i, row in best_per_horizon_df.iterrows():
        ax.bar(
            row['horizon'],
            row['mae'],
            width=1.5,
            color=color_map[row['best_model']],
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )

    # Create legend
    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color_map[model], alpha=0.7, label=model)
                      for model in unique_models]
    ax.legend(handles=legend_elements, title='Best Model', loc='upper left', fontsize=9)

    ax.set_xlabel('Prediction Horizon (hours ahead)', fontsize=12, fontweight='bold')
    ax.set_ylabel('MAE (Best Model)', fontsize=12, fontweight='bold')
    ax.set_title('Best Model at Each Prediction Horizon', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
