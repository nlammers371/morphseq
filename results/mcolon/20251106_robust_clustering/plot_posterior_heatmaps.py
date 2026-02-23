"""
Plot Posterior Heatmaps

Visualizes posterior probability matrices p_i(c) as heatmaps,
with embryos sorted by modal cluster assignment.

Usage:
    python plot_posterior_heatmaps.py --k 3
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
import argparse
from typing import Dict


def load_results(k: int, data_dir: str = 'output/data') -> Dict:
    """Load posterior analysis results for given k."""
    filepath = Path(data_dir) / f'posteriors_k{k}.pkl'
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    return results


def plot_posterior_heatmap(k: int,
                           data_dir: str = 'output/data',
                           output_dir: str = 'output/figures',
                           show_classification: bool = True):
    """
    Plot heatmap of posterior probabilities p_i(c).

    Parameters
    ----------
    k : int
        Number of clusters
    data_dir : str
        Directory containing results
    output_dir : str
        Directory to save figure
    show_classification : bool
        Whether to show classification annotations
    """
    # Load results
    results = load_results(k, data_dir)
    posterior_data = results['posterior_method']['posterior_data']
    classification = results['posterior_method']['classification']

    p_matrix = posterior_data['p_matrix']
    modal_cluster = posterior_data['modal_cluster']
    category = classification['category']

    # Sort embryos by modal cluster, then by max_p within cluster
    sort_idx = np.lexsort((posterior_data['max_p'], modal_cluster))
    p_matrix_sorted = p_matrix[sort_idx, :]
    category_sorted = category[sort_idx]
    modal_cluster_sorted = modal_cluster[sort_idx]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 12))

    # Plot heatmap
    im = ax.imshow(p_matrix_sorted, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Posterior Probability p(cluster | embryo)', fontsize=11)

    # Add cluster boundaries
    cluster_boundaries = []
    for c in range(k):
        if c < k - 1:
            boundary = np.where(modal_cluster_sorted == c)[0][-1] + 0.5
            cluster_boundaries.append(boundary)
            ax.axhline(y=boundary, color='black', linewidth=2, alpha=0.7)

    # Add classification annotations if requested
    if show_classification:
        # Add colored bar on the left showing classification
        category_colors = {'core': 'green', 'uncertain': 'orange', 'outlier': 'red'}
        category_numeric = np.array([
            0 if cat == 'core' else 1 if cat == 'uncertain' else 2
            for cat in category_sorted
        ])

        # Create a narrow axis for the classification bar
        divider_ax = fig.add_axes([0.08, 0.11, 0.02, 0.77])  # [left, bottom, width, height]
        divider_ax.imshow(category_numeric[:, np.newaxis], aspect='auto',
                         cmap=plt.cm.colors.ListedColormap(['green', 'orange', 'red']),
                         vmin=0, vmax=2)
        divider_ax.set_xticks([])
        divider_ax.set_yticks([])
        divider_ax.set_ylabel('Classification', fontsize=10, rotation=0, ha='right', va='center')

    # Labels
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Embryo (sorted by modal cluster)', fontsize=12)
    ax.set_title(f'Assignment Posterior Probabilities (k={k})', fontsize=14, fontweight='bold')

    # Set ticks
    ax.set_xticks(range(k))
    ax.set_xticklabels(range(k))

    # Adjust y-axis
    ax.set_yticks([])  # Too many embryos to show individual labels

    # Add cluster labels on the right
    for c in range(k):
        mask = (modal_cluster_sorted == c)
        y_center = np.where(mask)[0].mean()
        n_in_cluster = np.sum(mask)
        ax.text(k + 0.2, y_center, f'C{c}\n(n={n_in_cluster})',
               ha='left', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / f'posterior_heatmap_k{k}.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {filepath}")

    plt.close()


def plot_posterior_heatmap_with_metrics(k: int,
                                       data_dir: str = 'output/data',
                                       output_dir: str = 'output/figures'):
    """
    Plot heatmap with accompanying quality metrics panel.

    Parameters
    ----------
    k : int
        Number of clusters
    data_dir : str
        Directory containing results
    output_dir : str
        Directory to save figure
    """
    # Load results
    results = load_results(k, data_dir)
    posterior_data = results['posterior_method']['posterior_data']
    classification = results['posterior_method']['classification']

    p_matrix = posterior_data['p_matrix']
    modal_cluster = posterior_data['modal_cluster']
    max_p = posterior_data['max_p']
    entropy = posterior_data['entropy']
    category = classification['category']

    # Sort embryos
    sort_idx = np.lexsort((max_p, modal_cluster))
    p_matrix_sorted = p_matrix[sort_idx, :]
    max_p_sorted = max_p[sort_idx]
    entropy_sorted = entropy[sort_idx]
    category_sorted = category[sort_idx]
    modal_cluster_sorted = modal_cluster[sort_idx]

    # Create figure with GridSpec for better control
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.15, 0.15, 1, 0.05], wspace=0.05)

    # Axis for classification bar
    ax_cat = fig.add_subplot(gs[0, 0])
    # Axis for metrics (max_p, entropy)
    ax_metrics = fig.add_subplot(gs[0, 1])
    # Axis for heatmap
    ax_heat = fig.add_subplot(gs[0, 2])
    # Axis for colorbar
    ax_cbar = fig.add_subplot(gs[0, 3])

    # Plot classification bar
    category_colors = {'core': 0, 'uncertain': 1, 'outlier': 2}
    category_numeric = np.array([category_colors[cat] for cat in category_sorted])
    ax_cat.imshow(category_numeric[:, np.newaxis], aspect='auto',
                  cmap=plt.cm.colors.ListedColormap(['green', 'orange', 'red']),
                  vmin=0, vmax=2)
    ax_cat.set_xticks([])
    ax_cat.set_yticks([])
    ax_cat.set_xlabel('Class', fontsize=9)

    # Plot metrics
    # Normalize metrics to [0, 1] for visualization
    max_p_norm = max_p_sorted
    entropy_norm = entropy_sorted / np.log2(k)  # Normalize by max entropy

    metrics_matrix = np.column_stack([max_p_norm, entropy_norm])
    im_metrics = ax_metrics.imshow(metrics_matrix, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax_metrics.set_xticks([0, 1])
    ax_metrics.set_xticklabels(['max_p', 'entropy'], rotation=45, ha='right', fontsize=9)
    ax_metrics.set_yticks([])

    # Plot heatmap
    im_heat = ax_heat.imshow(p_matrix_sorted, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)

    # Add cluster boundaries
    for c in range(k - 1):
        boundary = np.where(modal_cluster_sorted == c)[0][-1] + 0.5
        ax_cat.axhline(y=boundary, color='black', linewidth=1.5, alpha=0.5)
        ax_metrics.axhline(y=boundary, color='black', linewidth=1.5, alpha=0.5)
        ax_heat.axhline(y=boundary, color='black', linewidth=1.5, alpha=0.5)

    # Labels for heatmap
    ax_heat.set_xlabel('Cluster', fontsize=12)
    ax_heat.set_ylabel('Embryo (sorted by modal cluster)', fontsize=12)
    ax_heat.set_xticks(range(k))
    ax_heat.set_xticklabels(range(k))
    ax_heat.set_yticks([])

    # Add cluster annotations
    for c in range(k):
        mask = (modal_cluster_sorted == c)
        y_center = np.where(mask)[0].mean()
        n_in_cluster = np.sum(mask)
        ax_heat.text(k + 0.3, y_center, f'Cluster {c}\n(n={n_in_cluster})',
                    ha='left', va='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Colorbar for heatmap
    cbar = plt.colorbar(im_heat, cax=ax_cbar)
    cbar.set_label('p(cluster | embryo)', fontsize=11)

    # Overall title
    fig.suptitle(f'Assignment Posteriors with Quality Metrics (k={k})',
                fontsize=14, fontweight='bold', y=0.98)

    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / f'posterior_heatmap_with_metrics_k{k}.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {filepath}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate posterior probability heatmaps')
    parser.add_argument('--k', type=int, default=3, help='Number of clusters to plot')
    parser.add_argument('--data_dir', type=str, default='output/data',
                       help='Directory containing results')
    parser.add_argument('--output_dir', type=str, default='output/figures',
                       help='Output directory for figures')
    parser.add_argument('--with_metrics', action='store_true',
                       help='Generate extended heatmap with metrics panel')

    args = parser.parse_args()

    print("="*60)
    print("GENERATING POSTERIOR HEATMAPS")
    print("="*60)
    print(f"k = {args.k}")

    # Basic heatmap
    print(f"\nGenerating heatmap for k={args.k}...")
    plot_posterior_heatmap(args.k, args.data_dir, args.output_dir)

    # Extended heatmap with metrics
    if args.with_metrics:
        print(f"\nGenerating extended heatmap with metrics...")
        plot_posterior_heatmap_with_metrics(args.k, args.data_dir, args.output_dir)

    print("\n✓ Heatmaps generated successfully!")
    print(f"Figures saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
