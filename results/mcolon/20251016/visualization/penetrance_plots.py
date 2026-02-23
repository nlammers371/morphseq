"""
Penetrance distribution visualization.

This module provides functions for visualizing embryo-level penetrance
metric distributions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple


def plot_penetrance_distribution(
    df_penetrance: pd.DataFrame,
    group1: str,
    group2: str,
    output_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """
    Histogram and summary of penetrance distribution split by genotype.

    Displays distributions for multiple metrics including confidence, support
    for true class, signed margin, and temporal consistency. Each metric is
    shown with overlaid histograms colored by genotype for easy comparison.

    Parameters
    ----------
    df_penetrance : pd.DataFrame
        Penetrance metrics from compute_embryo_penetrance()
    group1, group2 : str
        Comparison groups (genotype labels)
    output_path : str or None
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure or None
        The generated figure, or None if insufficient data
    """
    if df_penetrance.empty:
        print("  Skipping penetrance distribution: no data")
        return None

    # Detect which metrics are available
    metrics_config = [
        ('mean_signed_margin', 'Mean Signed Margin', 'coral', (-0.5, 0.5)),
        ('mean_support_true', 'Mean Support (True Class)', 'forestgreen', (0, 1.0)),
        ('mean_confidence', 'Mean Confidence (|p - 0.5|)', 'steelblue', (0, 0.5)),
        ('temporal_consistency', 'Temporal Consistency', 'darkorange', (0, 1.0))
    ]
    available_metrics = [
        (col, title, color, xlim) for col, title, color, xlim in metrics_config
        if col in df_penetrance.columns
    ]

    if len(available_metrics) == 0:
        print("  Skipping penetrance distribution: no metrics found")
        return None

    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=(7 * ((n_metrics + 1) // 2), 10))
    axes = np.atleast_2d(axes).flatten()

    # Get unique genotypes
    genotypes = [g for g in [group1, group2] if g in df_penetrance['true_label'].unique()]
    palette = ['dodgerblue', 'orangered', 'mediumseagreen', 'mediumpurple']
    colors_by_genotype = {genotype: palette[idx % len(palette)] for idx, genotype in enumerate(genotypes)}

    for idx, (col, title, base_color, (xmin, xmax)) in enumerate(available_metrics):
        ax = axes[idx]

        # Plot histograms for each genotype
        for genotype in genotypes:
            subset = df_penetrance[df_penetrance['true_label'] == genotype][col].dropna()
            if len(subset) == 0:
                continue

            ax.hist(subset, bins=15, alpha=0.6, color=colors_by_genotype.get(genotype, 'gray'),
                   edgecolor='black', linewidth=0.5, label=f'{genotype} (n={len(subset)})')

            # Add median line for this genotype
            median_val = subset.median()
            ax.axvline(median_val, color=colors_by_genotype.get(genotype, 'gray'),
                      linestyle='--', linewidth=2, alpha=0.8)

        ax.set_xlabel(title, fontsize=11)
        ax.set_ylabel('Number of Embryos', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(xmin, xmax)
        ax.legend(fontsize=9, loc='best')
        ax.grid(alpha=0.3)

    # Hide unused axes
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(f'Penetrance Distributions by Genotype: {group1} vs {group2}',
                fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    return fig


def plot_penetrance_summary_by_genotype(
    df_penetrance: pd.DataFrame,
    metric: str = 'mean_confidence',
    genotypes: Optional[List[str]] = None,
    output_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """
    Box plot or violin plot showing penetrance metric distribution by genotype.

    Parameters
    ----------
    df_penetrance : pd.DataFrame
        Penetrance metrics from compute_embryo_penetrance()
    metric : str, default='mean_confidence'
        Metric to plot
    genotypes : list of str or None
        Genotypes to include (None = all)
    output_path : str or None
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure or None
        The generated figure, or None if insufficient data
    """
    if df_penetrance.empty:
        print("  Skipping penetrance summary: no data")
        return None

    if metric not in df_penetrance.columns:
        print(f"  Skipping penetrance summary: metric '{metric}' not found")
        return None

    df_plot = df_penetrance.copy()
    if genotypes:
        df_plot = df_plot[df_plot['true_label'].isin(genotypes)]

    if df_plot.empty:
        print("  Skipping penetrance summary: no data after filtering")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create box plot
    genotype_order = sorted(df_plot['true_label'].unique())
    positions = np.arange(len(genotype_order))

    box_data = [df_plot[df_plot['true_label'] == g][metric].dropna().values
                for g in genotype_order]

    bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                    patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

    # Color boxes
    colors = ['dodgerblue', 'orangered', 'mediumseagreen', 'mediumpurple']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels([g.split('_')[-1] for g in genotype_order], rotation=45, ha='right')
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'{metric.replace("_", " ").title()} by Genotype',
                fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    return fig
