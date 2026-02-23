"""Modular plotting functions.

Pure visualization functions with no data processing logic.
All functions expect pre-processed, plot-ready data.

Following the pattern of plot_pooled_auroc_with_pvalues as the gold standard.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from .plotting_schemas import validate_auroc_data, validate_divergence_data


def plot_auroc_with_null(
    ax: plt.Axes,
    auroc_df: pd.DataFrame,
    color: str,
    label: str,
    style: str = '-',
    time_col: str = 'time_bin_center',
    show_null_band: bool = True,
    show_significance: bool = True,
    sig_05_marker_size: int = 200
):
    """Plot single AUROC curve with null distribution and significance markers.

    This is the core plotting function for AUROC data. Pure visualization only.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    auroc_df : pd.DataFrame
        Pre-processed AUROC data with required columns:
        - time_bin or time_bin_center
        - auroc_observed
        - auroc_null_mean, auroc_null_std (for null band)
        - is_significant (optional, pre-computed in preprocessing)
    color : str
        Line and marker color
    label : str
        Legend label
    style : str
        Line style ('-', '--', ':', etc.)
    time_col : str
        Column to use for x-axis (default: time_bin_center)
    show_null_band : bool
        Whether to show null distribution band
    show_significance : bool
        Whether to show significance markers
    sig_05_marker_size : int
        Size of circle for significant markers (p <= 0.01 by default)
    """
    # Validate data
    validate_auroc_data(auroc_df)

    # Prefer time_bin_center if available
    if time_col not in auroc_df.columns:
        time_col = 'time_bin'

    # Plot AUROC line
    ax.plot(
        auroc_df[time_col],
        auroc_df['auroc_observed'],
        f'o{style}',
        label=label,
        color=color,
        linewidth=2,
        markersize=5
    )

    # Null distribution band (mean ± 1 SD)
    if show_null_band and 'auroc_null_mean' in auroc_df.columns and 'auroc_null_std' in auroc_df.columns:
        ax.fill_between(
            auroc_df[time_col],
            auroc_df['auroc_null_mean'] - auroc_df['auroc_null_std'],
            auroc_df['auroc_null_mean'] + auroc_df['auroc_null_std'],
            color=color,
            alpha=0.10,
            linewidth=0
        )

    # Significance markers (circles for p <= 0.01 by default)
    if show_significance:
        if 'is_significant' in auroc_df.columns:
            sig_mask = auroc_df['is_significant']
        elif 'is_significant_01' in auroc_df.columns:
            sig_mask = auroc_df['is_significant_01']
        else:
            sig_mask = auroc_df['pval'] <= 0.01

        if sig_mask.any():
            ax.scatter(
                auroc_df.loc[sig_mask, time_col],
                auroc_df.loc[sig_mask, 'auroc_observed'],
                s=sig_05_marker_size,
                facecolors='none',
                edgecolors=color,
                linewidths=2.5,
                zorder=5
            )


def plot_multiple_aurocs(
    auroc_dfs_dict: Dict[str, pd.DataFrame],
    colors_dict: Dict[str, str],
    styles_dict: Optional[Dict[str, str]] = None,
    title: str = 'AUROC Comparison',
    figsize: Tuple[int, int] = (14, 7),
    ylim: Tuple[float, float] = (0.3, 1.05),
    time_col: str = 'time_bin_center',
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot multiple AUROC curves overlaid on a single axis.

    Replicates plot_pooled_auroc_with_pvalues pattern.

    Parameters
    ----------
    auroc_dfs_dict : dict of {str: pd.DataFrame}
        Dictionary mapping comparison labels to AUROC DataFrames
    colors_dict : dict of {str: str}
        Color for each comparison
    styles_dict : dict of {str: str}, optional
        Line style for each comparison (default: all solid lines)
    title : str
        Figure title
    figsize : tuple of (int, int)
        Figure size
    ylim : tuple of (float, float)
        Y-axis limits
    time_col : str
        Time column to use for x-axis
    save_path : str, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if styles_dict is None:
        styles_dict = {label: '-' for label in auroc_dfs_dict.keys()}

    # Plot each comparison
    for label, auroc_df in auroc_dfs_dict.items():
        if auroc_df.empty:
            continue

        color = colors_dict.get(label, '#000000')
        style = styles_dict.get(label, '-')

        plot_auroc_with_null(
            ax=ax,
            auroc_df=auroc_df,
            color=color,
            label=label,
            style=style,
            time_col=time_col
        )

    # Reference line at 0.5 (chance)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Chance (0.5)')

    # Add significance marker to legend (black circle for p <= 0.01)
    ax.scatter([], [], s=200, facecolors='none', edgecolors='black',
               linewidths=2.5, label='p <= 0.01')

    # Formatting
    ax.set_xlabel('Hours Post Fertilization (hpf)', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(ylim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_divergence_timecourse(
    ax: plt.Axes,
    divergence_df: pd.DataFrame,
    metric_filter: Optional[List[str]] = None,
    use_smoothed: bool = True,
    colors_dict: Optional[Dict[str, str]] = None,
    labels_dict: Optional[Dict[str, str]] = None,
    time_col: str = 'hpf',
    show_sem_bands: bool = True
):
    """Plot divergence (or metrics) over time.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    divergence_df : pd.DataFrame
        Pre-processed divergence data
    metric_filter : list of str, optional
        Which metrics to plot (None = all)
    use_smoothed : bool
        If True, use 'abs_difference_smoothed' column
    colors_dict : dict, optional
        Color mapping {metric_name: color}
    labels_dict : dict, optional
        Label mapping {metric_name: display_label}
    time_col : str
        Time column name
    show_sem_bands : bool
        Show SEM bands if available
    """
    validate_divergence_data(divergence_df)

    # Determine which metrics to plot
    if metric_filter is None:
        metrics = divergence_df['metric'].unique()
    else:
        metrics = metric_filter

    # Select value column
    if use_smoothed and 'abs_difference_smoothed' in divergence_df.columns:
        value_col = 'abs_difference_smoothed'
    else:
        value_col = 'abs_difference'

    # Plot each metric
    for i, metric_name in enumerate(metrics):
        metric_data = divergence_df[divergence_df['metric'] == metric_name].sort_values(time_col)

        if metric_data.empty:
            continue

        # Determine color and label
        if colors_dict:
            color = colors_dict.get(metric_name, f'C{i}')
        else:
            color = f'C{i}'

        if labels_dict:
            label = labels_dict.get(metric_name, metric_name)
        else:
            label = metric_name

        # Plot line
        ax.plot(
            metric_data[time_col],
            metric_data[value_col],
            label=label,
            color=color,
            linewidth=2
        )

        # SEM bands if available
        if show_sem_bands and 'group1_sem' in metric_data.columns and 'group2_sem' in metric_data.columns:
            # Compute combined SEM for difference
            combined_sem = np.sqrt(metric_data['group1_sem']**2 + metric_data['group2_sem']**2)

            ax.fill_between(
                metric_data[time_col],
                metric_data[value_col] - combined_sem,
                metric_data[value_col] + combined_sem,
                color=color,
                alpha=0.2,
                linewidth=0
            )


def plot_raw_metric_timecourse(
    ax: plt.Axes,
    df_trajectories: pd.DataFrame,
    metric_col: str,
    group_col: str = 'group',
    time_col: str = 'predicted_stage_hpf',
    show_individual: bool = False,
    show_group_stats: bool = True,
    colors_dict: Optional[Dict[str, str]] = None,
    use_smoothed: bool = False,
    alpha_individual: float = 0.3
):
    """Plot raw metric values over time (no AUROC, no classification).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    df_trajectories : pd.DataFrame
        Trajectory data with metric values
    metric_col : str
        Metric column to plot
    group_col : str
        Group column
    time_col : str
        Time column
    show_individual : bool
        Show individual embryo trajectories
    show_group_stats : bool
        Show group mean + SEM bands
    colors_dict : dict, optional
        Color mapping {group_name: color}
    use_smoothed : bool
        Use smoothed metric if available
    alpha_individual : float
        Transparency for individual trajectories
    """
    # Select metric column
    if use_smoothed and f'{metric_col}_smoothed' in df_trajectories.columns:
        value_col = f'{metric_col}_smoothed'
    else:
        value_col = metric_col

    groups = df_trajectories[group_col].unique()

    for i, group_name in enumerate(groups):
        group_data = df_trajectories[df_trajectories[group_col] == group_name]

        # Determine color
        if colors_dict:
            color = colors_dict.get(group_name, f'C{i}')
        else:
            color = f'C{i}'

        # Individual trajectories
        if show_individual:
            for embryo_id in group_data['embryo_id'].unique():
                embryo_data = group_data[group_data['embryo_id'] == embryo_id].sort_values(time_col)
                ax.plot(
                    embryo_data[time_col],
                    embryo_data[value_col],
                    color=color,
                    alpha=alpha_individual,
                    linewidth=0.8
                )

        # Group statistics
        if show_group_stats:
            stats = group_data.groupby(time_col)[value_col].agg(['mean', 'sem']).reset_index()

            ax.plot(
                stats[time_col],
                stats['mean'],
                label=group_name,
                color=color,
                linewidth=2.5
            )

            # SEM band
            ax.fill_between(
                stats[time_col],
                stats['mean'] - stats['sem'],
                stats['mean'] + stats['sem'],
                color=color,
                alpha=0.25,
                linewidth=0
            )


def plot_zscore_metrics(
    ax: plt.Axes,
    zscore_df: pd.DataFrame,
    metric_filter: Optional[List[str]] = None,
    use_smoothed: bool = True,
    colors_dict: Optional[Dict[str, str]] = None,
    labels_dict: Optional[Dict[str, str]] = None,
    time_col: str = 'hpf'
):
    """Plot z-scored metrics over time.

    Similar to plot_divergence_timecourse but expects z-score column.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    zscore_df : pd.DataFrame
        DataFrame with z-scored metrics
    metric_filter : list of str, optional
        Which metrics to plot
    use_smoothed : bool
        Use smoothed z-scores if available
    colors_dict : dict, optional
        Color mapping
    labels_dict : dict, optional
        Label mapping
    time_col : str
        Time column
    """
    # Determine which metrics to plot
    if metric_filter is None and 'metric' in zscore_df.columns:
        metrics = zscore_df['metric'].unique()
    else:
        metrics = metric_filter if metric_filter else []

    # Select value column
    if use_smoothed and 'abs_difference_zscore_smoothed' in zscore_df.columns:
        value_col = 'abs_difference_zscore_smoothed'
    elif 'abs_difference_zscore' in zscore_df.columns:
        value_col = 'abs_difference_zscore'
    else:
        raise ValueError("No z-score column found (expected 'abs_difference_zscore')")

    # Plot each metric
    for i, metric_name in enumerate(metrics):
        metric_data = zscore_df[zscore_df['metric'] == metric_name].sort_values(time_col)

        if metric_data.empty:
            continue

        # Determine color and label
        if colors_dict:
            color = colors_dict.get(metric_name, f'C{i}')
        else:
            color = f'C{i}'

        if labels_dict:
            label = labels_dict.get(metric_name, metric_name)
        else:
            label = metric_name

        # Plot line
        ax.plot(
            metric_data[time_col],
            metric_data[value_col],
            label=label,
            color=color,
            linewidth=2
        )

    # Reference line at z=0
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)


# =============================================================================
# Multiclass Plotting Functions
# =============================================================================

def plot_multiclass_ovr_aurocs(
    ovr_results: Dict[str, pd.DataFrame],
    colors_dict: Dict[str, str],
    title: str = 'Per-Class OvR AUROC',
    figsize: Tuple[int, int] = (12, 7),
    ylim: Tuple[float, float] = (0.3, 1.05),
    time_col: str = 'time_bin_center',
    show_null_band: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot One-vs-Rest AUROC curves for multiclass classification.

    Each class gets its own curve showing how well it separates from all other
    classes combined, with its own null distribution band.

    Parameters
    ----------
    ovr_results : Dict[str, pd.DataFrame]
        Dictionary mapping class labels to AUROC DataFrames
        Each DataFrame should have: time_bin_center, auroc_observed,
        auroc_null_mean, auroc_null_std, pval
    colors_dict : Dict[str, str]
        Color for each class
    title : str
        Figure title
    figsize : Tuple[int, int]
        Figure size
    ylim : Tuple[float, float]
        Y-axis limits
    time_col : str
        Time column to use for x-axis
    show_null_band : bool
        Whether to show null distribution bands
    save_path : str, optional
        Path to save figure

    Returns
    -------
    plt.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each class
    for class_label, auroc_df in ovr_results.items():
        if auroc_df is None or auroc_df.empty:
            continue

        color = colors_dict.get(class_label, f'C{list(ovr_results.keys()).index(class_label)}')

        plot_auroc_with_null(
            ax=ax,
            auroc_df=auroc_df,
            color=color,
            label=f'{class_label} vs Rest',
            style='-',
            time_col=time_col,
            show_null_band=show_null_band,
            show_significance=True
        )

    # Reference line at 0.5 (chance)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Chance (0.5)')

    # Add significance marker to legend
    ax.scatter([], [], s=200, facecolors='none', edgecolors='black',
               linewidths=2.5, label='p <= 0.01')

    # Formatting
    ax.set_xlabel('Hours Post Fertilization (hpf)', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(ylim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_confusion_matrix_heatmap(
    confusion_matrix_df: pd.DataFrame,
    title: str = 'Confusion Matrix',
    figsize: Tuple[int, int] = (8, 6),
    normalize: bool = True,
    cmap: str = 'Blues',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix as a heatmap.

    Parameters
    ----------
    confusion_matrix_df : pd.DataFrame
        Confusion matrix with class labels as index and columns
        Index = true class, columns = predicted class
    title : str
        Figure title
    figsize : Tuple[int, int]
        Figure size
    normalize : bool
        If True, normalize by row (shows recall per class)
    cmap : str
        Colormap name
    save_path : str, optional
        Path to save figure

    Returns
    -------
    plt.Figure
        The created figure
    """
    import seaborn as sns

    fig, ax = plt.subplots(figsize=figsize)

    # Normalize if requested
    if normalize:
        cm_plot = confusion_matrix_df.div(confusion_matrix_df.sum(axis=1), axis=0)
        fmt = '.2f'
        vmin, vmax = 0, 1
    else:
        cm_plot = confusion_matrix_df
        fmt = 'd'
        vmin, vmax = None, None

    # Create heatmap
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'},
        linewidths=0.5,
        linecolor='white'
    )

    ax.set_xlabel('Predicted Class', fontsize=12)
    ax.set_ylabel('True Class', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Rotate labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_temporal_confusion_profile(
    temporal_profile_df: pd.DataFrame,
    class_label: str,
    colors_dict: Dict[str, str],
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    time_col: str = 'time_bin',
    plot_style: str = 'stacked_area',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot how classification of a single class breaks down over time.

    Shows the proportion classified as each class at each time point,
    revealing when the class becomes distinguishable and which classes
    it gets confused with.

    Parameters
    ----------
    temporal_profile_df : pd.DataFrame
        Output from extract_temporal_confusion_profile()
        Must have: time_bin, true_class, predicted_class, proportion
    class_label : str
        Which class to show the profile for
    colors_dict : Dict[str, str]
        Color for each class
    title : str, optional
        Figure title (default: auto-generated)
    figsize : Tuple[int, int]
        Figure size
    time_col : str
        Time column name
    plot_style : str
        'stacked_area' or 'lines'
    save_path : str, optional
        Path to save figure

    Returns
    -------
    plt.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Filter to the target class
    class_data = temporal_profile_df[
        temporal_profile_df['true_class'] == class_label
    ].copy()

    if class_data.empty:
        ax.text(0.5, 0.5, f'No data for class: {class_label}',
                ha='center', va='center', transform=ax.transAxes)
        return fig

    # Pivot to get time x predicted_class
    pivot = class_data.pivot(
        index=time_col,
        columns='predicted_class',
        values='proportion'
    ).fillna(0)

    # Sort columns: correct class first, then others
    pred_classes = list(pivot.columns)
    if class_label in pred_classes:
        pred_classes.remove(class_label)
        pred_classes = [class_label] + sorted(pred_classes)
    pivot = pivot[pred_classes]

    time_values = pivot.index.values

    if plot_style == 'stacked_area':
        # Stacked area plot
        colors = [colors_dict.get(c, f'C{i}') for i, c in enumerate(pred_classes)]
        ax.stackplot(
            time_values,
            [pivot[c].values for c in pred_classes],
            labels=pred_classes,
            colors=colors,
            alpha=0.8
        )
    else:
        # Line plot
        for pred_class in pred_classes:
            color = colors_dict.get(pred_class, f'C{list(pred_classes).index(pred_class)}')
            ax.plot(
                time_values,
                pivot[pred_class].values,
                label=pred_class,
                color=color,
                linewidth=2,
                marker='o',
                markersize=4
            )

    # Formatting
    ax.set_xlabel('Hours Post Fertilization (hpf)', fontsize=12)
    ax.set_ylabel('Proportion', fontsize=12)
    ax.set_ylim(0, 1)

    if title is None:
        title = f'Classification Profile: {class_label}'
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.legend(loc='upper right', fontsize=10, framealpha=0.9, title='Predicted as')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_all_temporal_confusion_profiles(
    temporal_profile_df: pd.DataFrame,
    class_labels: List[str],
    colors_dict: Dict[str, str],
    figsize: Tuple[int, int] = (14, 10),
    time_col: str = 'time_bin',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot temporal confusion profiles for all classes in a grid.

    Parameters
    ----------
    temporal_profile_df : pd.DataFrame
        Output from extract_temporal_confusion_profile()
    class_labels : List[str]
        List of class labels to plot
    colors_dict : Dict[str, str]
        Color for each class
    figsize : Tuple[int, int]
        Figure size
    time_col : str
        Time column name
    save_path : str, optional
        Path to save figure

    Returns
    -------
    plt.Figure
        The created figure
    """
    n_classes = len(class_labels)
    n_cols = min(2, n_classes)
    n_rows = (n_classes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes)

    for idx, class_label in enumerate(class_labels):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Filter to the target class
        class_data = temporal_profile_df[
            temporal_profile_df['true_class'] == class_label
        ].copy()

        if class_data.empty:
            ax.text(0.5, 0.5, f'No data for: {class_label}',
                    ha='center', va='center', transform=ax.transAxes)
            continue

        # Pivot to get time x predicted_class
        pivot = class_data.pivot(
            index=time_col,
            columns='predicted_class',
            values='proportion'
        ).fillna(0)

        # Sort columns: correct class first
        pred_classes = list(pivot.columns)
        if class_label in pred_classes:
            pred_classes.remove(class_label)
            pred_classes = [class_label] + sorted(pred_classes)
        pivot = pivot[pred_classes]

        time_values = pivot.index.values

        # Stacked area plot
        colors = [colors_dict.get(c, f'C{i}') for i, c in enumerate(pred_classes)]
        ax.stackplot(
            time_values,
            [pivot[c].values for c in pred_classes],
            labels=pred_classes,
            colors=colors,
            alpha=0.8
        )

        ax.set_title(f'{class_label}', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_xlabel('hpf', fontsize=10)
        ax.set_ylabel('Proportion', fontsize=10)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide unused axes
    for idx in range(n_classes, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    plt.suptitle('Temporal Classification Profiles', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig
