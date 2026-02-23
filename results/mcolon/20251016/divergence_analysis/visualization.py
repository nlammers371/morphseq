"""
Visualization functions for morphological divergence analysis.

Provides plotting functions for:
- Divergence trajectories over time
- Distribution of divergence scores
- Heatmaps of embryo divergence
- Metric comparisons
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
from matplotlib.figure import Figure


def plot_divergence_over_time(
    df_divergence: pd.DataFrame,
    metric: str = "mahalanobis_distance",
    by_genotype: bool = True,
    show_individuals: bool = False,
    figsize: tuple = (12, 6)
) -> Figure:
    """
    Plot divergence trajectories over time.
    
    Parameters
    ----------
    df_divergence : pd.DataFrame
        Output from compute_divergence_scores()
    metric : str, default="mahalanobis_distance"
        Distance metric to plot
    by_genotype : bool, default=True
        If True, separate lines by genotype
    show_individuals : bool, default=False
        If True, show individual embryo trajectories (lighter)
    figsize : tuple
        Figure size
    
    Returns
    -------
    Figure
        Matplotlib figure
    """
    if metric not in df_divergence.columns:
        raise ValueError(f"Metric '{metric}' not found in data")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if by_genotype and 'genotype' in df_divergence.columns:
        genotypes = df_divergence['genotype'].unique()
        colors = sns.color_palette("husl", len(genotypes))
        
        for genotype, color in zip(genotypes, colors):
            geno_data = df_divergence[df_divergence['genotype'] == genotype]
            
            # Show individuals if requested
            if show_individuals:
                for embryo_id in geno_data['embryo_id'].unique():
                    emb_data = geno_data[geno_data['embryo_id'] == embryo_id]
                    ax.plot(emb_data['time_bin'], emb_data[metric], 
                           alpha=0.1, color=color, linewidth=0.5)
            
            # Plot mean trajectory
            time_means = geno_data.groupby('time_bin')[metric].agg(['mean', 'std', 'count'])
            ax.plot(time_means.index, time_means['mean'], 
                   label=genotype, color=color, linewidth=2)
            
            # Add error band (SEM)
            sem = time_means['std'] / np.sqrt(time_means['count'])
            ax.fill_between(time_means.index, 
                           time_means['mean'] - sem,
                           time_means['mean'] + sem,
                           alpha=0.3, color=color)
    else:
        # Plot overall trajectory
        time_means = df_divergence.groupby('time_bin')[metric].agg(['mean', 'std', 'count'])
        ax.plot(time_means.index, time_means['mean'], linewidth=2)
        sem = time_means['std'] / np.sqrt(time_means['count'])
        ax.fill_between(time_means.index, 
                       time_means['mean'] - sem,
                       time_means['mean'] + sem,
                       alpha=0.3)
    
    ax.set_xlabel('Developmental time (hpf)', fontsize=12)
    ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=12)
    ax.set_title('Morphological Divergence Over Time', fontsize=14, fontweight='bold')
    
    if by_genotype:
        ax.legend()
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig


def plot_divergence_distribution(
    df_divergence: pd.DataFrame,
    metric: str = "mahalanobis_distance",
    by_genotype: bool = True,
    plot_type: str = "violin",
    figsize: tuple = (10, 6)
) -> Figure:
    """
    Plot distribution of divergence scores.
    
    Parameters
    ----------
    df_divergence : pd.DataFrame
        Output from compute_divergence_scores()
    metric : str
        Distance metric to plot
    by_genotype : bool, default=True
        Separate by genotype
    plot_type : str, default="violin"
        "violin", "box", or "hist"
    figsize : tuple
        Figure size
    
    Returns
    -------
    Figure
    """
    if metric not in df_divergence.columns:
        raise ValueError(f"Metric '{metric}' not found in data")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if by_genotype and 'genotype' in df_divergence.columns:
        if plot_type == "violin":
            sns.violinplot(data=df_divergence, x='genotype', y=metric, ax=ax)
        elif plot_type == "box":
            sns.boxplot(data=df_divergence, x='genotype', y=metric, ax=ax)
        elif plot_type == "hist":
            for genotype in df_divergence['genotype'].unique():
                data = df_divergence[df_divergence['genotype'] == genotype][metric]
                ax.hist(data, alpha=0.5, label=genotype, bins=30)
            ax.legend()
        
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    else:
        ax.hist(df_divergence[metric], bins=30, edgecolor='black')
    
    ax.set_xlabel('Genotype' if by_genotype else '', fontsize=12)
    ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=12)
    ax.set_title('Distribution of Divergence Scores', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_divergence_heatmap(
    df_divergence: pd.DataFrame,
    metric: str = "mahalanobis_distance",
    max_embryos: int = 50,
    figsize: tuple = (14, 10)
) -> Figure:
    """
    Create heatmap of embryo divergence over time.
    
    Parameters
    ----------
    df_divergence : pd.DataFrame
        Output from compute_divergence_scores()
    metric : str
        Distance metric to plot
    max_embryos : int, default=50
        Maximum embryos to show
    figsize : tuple
        Figure size
    
    Returns
    -------
    Figure
    """
    if metric not in df_divergence.columns:
        raise ValueError(f"Metric '{metric}' not found in data")
    
    # Pivot to embryo x time matrix
    pivot = df_divergence.pivot_table(
        values=metric,
        index='embryo_id',
        columns='time_bin',
        aggfunc='mean'
    )
    
    # Sample if too many embryos
    if len(pivot) > max_embryos:
        # Sample those with highest mean divergence
        mean_div = pivot.mean(axis=1).sort_values(ascending=False)
        pivot = pivot.loc[mean_div.head(max_embryos).index]
    
    # Sort by mean divergence
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        pivot,
        cmap='YlOrRd',
        cbar_kws={'label': metric.replace('_', ' ').title()},
        ax=ax,
        xticklabels=5,  # Show every 5th time label
        yticklabels=True
    )
    
    ax.set_xlabel('Developmental time (hpf)', fontsize=12)
    ax.set_ylabel('Embryo ID', fontsize=12)
    ax.set_title(f'Embryo Divergence Heatmap ({max_embryos} embryos)', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_metric_comparison(
    df_divergence: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    figsize: tuple = (12, 10)
) -> Figure:
    """
    Compare different distance metrics.
    
    Parameters
    ----------
    df_divergence : pd.DataFrame
        Output from compute_divergence_scores()
    metrics : list of str, optional
        Metrics to compare. If None, uses all available.
    figsize : tuple
        Figure size
    
    Returns
    -------
    Figure
    """
    # Auto-detect metrics if not specified
    if metrics is None:
        metrics = [c for c in df_divergence.columns if c.endswith('_distance')]
    
    if len(metrics) < 2:
        raise ValueError("Need at least 2 metrics to compare")
    
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(n_metrics, n_metrics, figsize=figsize)
    
    for i, metric_y in enumerate(metrics):
        for j, metric_x in enumerate(metrics):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: distribution
                ax.hist(df_divergence[metric_x], bins=30, edgecolor='black', alpha=0.7)
                ax.set_ylabel('Count')
                if i == len(metrics) - 1:
                    ax.set_xlabel(metric_x.replace('_', ' ').title())
            else:
                # Off-diagonal: scatter
                ax.scatter(df_divergence[metric_x], df_divergence[metric_y], 
                          alpha=0.3, s=10)
                
                # Compute correlation
                corr = df_divergence[[metric_x, metric_y]].corr().iloc[0, 1]
                ax.text(0.05, 0.95, f'r = {corr:.3f}', 
                       transform=ax.transAxes, 
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                if j == 0:
                    ax.set_ylabel(metric_y.replace('_', ' ').title())
                if i == len(metrics) - 1:
                    ax.set_xlabel(metric_x.replace('_', ' ').title())
            
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Distance Metric Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig


def plot_outliers(
    df_divergence: pd.DataFrame,
    metric: str = "mahalanobis_distance",
    outlier_col: str = "is_outlier",
    figsize: tuple = (12, 6)
) -> Figure:
    """
    Visualize outlier embryos.
    
    Parameters
    ----------
    df_divergence : pd.DataFrame
        Output from compute_divergence_scores()
    metric : str
        Distance metric to plot
    outlier_col : str, default="is_outlier"
        Column indicating outlier status
    figsize : tuple
        Figure size
    
    Returns
    -------
    Figure
    """
    if outlier_col not in df_divergence.columns:
        raise ValueError(f"Outlier column '{outlier_col}' not found")
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left: distribution with outliers highlighted
    ax = axes[0]
    non_outliers = df_divergence[~df_divergence[outlier_col]][metric]
    outliers = df_divergence[df_divergence[outlier_col]][metric]
    
    ax.hist(non_outliers, bins=30, alpha=0.7, label='Normal', edgecolor='black')
    ax.hist(outliers, bins=30, alpha=0.7, label='Outliers', edgecolor='black', color='red')
    ax.set_xlabel(metric.replace('_', ' ').title())
    ax.set_ylabel('Count')
    ax.set_title('Distribution with Outliers')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Right: time course with outliers
    ax = axes[1]
    
    # Plot non-outliers
    non_out_data = df_divergence[~df_divergence[outlier_col]]
    time_means = non_out_data.groupby('time_bin')[metric].mean()
    ax.plot(time_means.index, time_means.values, 
           label='Mean (non-outliers)', linewidth=2, color='blue')
    
    # Scatter outliers
    out_data = df_divergence[df_divergence[outlier_col]]
    ax.scatter(out_data['time_bin'], out_data[metric], 
              c='red', alpha=0.5, s=20, label='Outliers')
    
    ax.set_xlabel('Developmental time (hpf)')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title('Outliers Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Outlier Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig
