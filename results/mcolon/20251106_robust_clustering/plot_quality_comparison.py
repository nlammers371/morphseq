"""
Plot Quality Comparison

Generates comparison plots for posterior-based vs. co-association classification.

Plots:
1. Method comparison across k (3 panels: core %, mean entropy, mean max_p)
2. 2D scatter of max_p vs log_odds_gap (decision boundary visualization)
3. Per-cluster quality metrics

Usage:
    python plot_quality_comparison.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
from pathlib import Path
import argparse
from typing import Dict


def load_comparison_results(data_dir: str = 'output/data') -> Dict:
    """Load comparison results from disk."""
    filepath = Path(data_dir) / 'all_results.pkl'
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    return results


def plot_method_comparison_vs_k(all_results: Dict, output_dir: str = 'output/figures'):
    """
    Plot 3-panel comparison: core %, mean entropy, mean max_p vs. k.

    Parameters
    ----------
    all_results : dict
        Results from compare_methods_v2.py
    output_dir : str
        Directory to save figure
    """
    k_values = sorted(all_results.keys())

    # Extract metrics for posterior method
    posterior_core_frac = []
    posterior_entropy = []
    posterior_max_p = []

    # Extract metrics for co-association method (if available)
    coassoc_core_frac = []

    for k in k_values:
        results = all_results[k]

        # Posterior method
        post_summary = results['posterior_method']['summary']
        post_data = results['posterior_method']['posterior_data']
        posterior_core_frac.append(post_summary['core_fraction'])
        posterior_entropy.append(np.mean(post_data['entropy']))
        posterior_max_p.append(np.mean(post_data['max_p']))

        # Co-association method
        if results['coassoc_method'] is not None:
            coassoc_summary = results['coassoc_method'].get('summary', {})
            coassoc_core_frac.append(coassoc_summary.get('core_fraction', np.nan))
        else:
            coassoc_core_frac.append(np.nan)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel A: Core membership fraction
    axes[0].plot(k_values, posterior_core_frac, 'o-', label='Posterior', linewidth=2, markersize=8)
    if not all(np.isnan(coassoc_core_frac)):
        axes[0].plot(k_values, coassoc_core_frac, 's--', label='Co-association', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[0].set_ylabel('Core Membership Fraction', fontsize=12)
    axes[0].set_title('A. Core Membership vs. k', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])

    # Panel B: Mean entropy
    axes[1].plot(k_values, posterior_entropy, 'o-', color='C1', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[1].set_ylabel('Mean Assignment Entropy (bits)', fontsize=12)
    axes[1].set_title('B. Assignment Uncertainty vs. k', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Panel C: Mean max_p
    axes[2].plot(k_values, posterior_max_p, 'o-', color='C2', linewidth=2, markersize=8)
    axes[2].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[2].set_ylabel('Mean Max Probability', fontsize=12)
    axes[2].set_title('C. Mean Confidence vs. k', fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 1.05])

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / 'quality_comparison_vs_k.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {filepath}")

    plt.close()


def plot_2d_scatter(k: int,
                   all_results: Dict,
                   output_dir: str = 'output/figures'):
    """
    Plot 2D scatter of max_p vs. log_odds_gap with classification overlay.

    Parameters
    ----------
    k : int
        Number of clusters to plot
    all_results : dict
        Results from compare_methods_v2.py
    output_dir : str
        Directory to save figure
    """
    if k not in all_results:
        print(f"Warning: k={k} not found in results")
        return

    results = all_results[k]
    classification = results['posterior_method']['classification']
    posterior_data = results['posterior_method']['posterior_data']

    max_p = classification['max_p']
    log_odds_gap = classification['log_odds_gap']
    category = classification['category']
    thresholds = classification['thresholds']

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Color map
    colors = {'core': 'C2', 'uncertain': 'C1', 'outlier': 'C3'}
    labels_map = {'core': 'Core', 'uncertain': 'Uncertain', 'outlier': 'Outlier'}

    # Plot points by category
    for cat in ['outlier', 'uncertain', 'core']:  # Plot outliers first
        mask = (category == cat)
        ax.scatter(log_odds_gap[mask], max_p[mask],
                  c=colors[cat], label=labels_map[cat],
                  alpha=0.6, s=30, edgecolors='black', linewidths=0.5)

    # Draw decision boundaries
    thresh_max_p = thresholds.get('max_p', 0.8)
    thresh_log_odds = thresholds.get('log_odds_gap', 0.7)
    thresh_outlier = thresholds.get('outlier_max_p', 0.5)

    # Horizontal lines
    ax.axhline(y=thresh_max_p, color='black', linestyle='--', linewidth=1.5,
               alpha=0.7, label=f'max_p threshold ({thresh_max_p:.2f})')
    ax.axhline(y=thresh_outlier, color='red', linestyle='--', linewidth=1.5,
               alpha=0.7, label=f'Outlier threshold ({thresh_outlier:.2f})')

    # Vertical line
    ax.axvline(x=thresh_log_odds, color='black', linestyle=':', linewidth=1.5,
               alpha=0.7, label=f'log-odds threshold ({thresh_log_odds:.2f})')

    ax.set_xlabel('Log-Odds Gap (log p₁ - log p₂)', fontsize=12)
    ax.set_ylabel('Max Posterior Probability (max_p)', fontsize=12)
    ax.set_title(f'2D Classification Gating (k={k})', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Set limits
    ax.set_xlim([max(log_odds_gap[np.isfinite(log_odds_gap)].min() - 0.5, 0),
                 min(log_odds_gap[np.isfinite(log_odds_gap)].max() + 0.5, 10)])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / f'posterior_scatter_k{k}.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {filepath}")

    plt.close()


def plot_per_cluster_quality(k: int,
                             all_results: Dict,
                             output_dir: str = 'output/figures'):
    """
    Plot per-cluster quality metrics (violin plots).

    Parameters
    ----------
    k : int
        Number of clusters to plot
    all_results : dict
        Results from compare_methods_v2.py
    output_dir : str
        Directory to save figure
    """
    if k not in all_results:
        print(f"Warning: k={k} not found in results")
        return

    results = all_results[k]
    classification = results['posterior_method']['classification']
    posterior_data = results['posterior_method']['posterior_data']

    modal_cluster = posterior_data['modal_cluster']
    max_p = posterior_data['max_p']
    entropy = posterior_data['entropy']
    log_odds_gap = posterior_data['log_odds_gap']

    # Build DataFrame for plotting
    df = pd.DataFrame({
        'cluster': modal_cluster,
        'max_p': max_p,
        'entropy': entropy,
        'log_odds_gap': log_odds_gap
    })

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel A: max_p by cluster
    sns.violinplot(data=df, x='cluster', y='max_p', ax=axes[0], inner='quartile')
    axes[0].set_xlabel('Cluster', fontsize=12)
    axes[0].set_ylabel('Max Probability', fontsize=12)
    axes[0].set_title('A. Assignment Confidence by Cluster', fontsize=13, fontweight='bold')
    axes[0].set_ylim([0, 1.05])
    axes[0].grid(True, alpha=0.3, axis='y')

    # Panel B: entropy by cluster
    sns.violinplot(data=df, x='cluster', y='entropy', ax=axes[1], inner='quartile', color='C1')
    axes[1].set_xlabel('Cluster', fontsize=12)
    axes[1].set_ylabel('Entropy (bits)', fontsize=12)
    axes[1].set_title('B. Assignment Uncertainty by Cluster', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Panel C: log_odds_gap by cluster
    log_odds_finite = df['log_odds_gap'].replace([np.inf, -np.inf], np.nan)
    df_plot = df.copy()
    df_plot['log_odds_gap'] = log_odds_finite
    sns.violinplot(data=df_plot, x='cluster', y='log_odds_gap', ax=axes[2], inner='quartile', color='C2')
    axes[2].set_xlabel('Cluster', fontsize=12)
    axes[2].set_ylabel('Log-Odds Gap', fontsize=12)
    axes[2].set_title('C. Top-2 Disambiguation by Cluster', fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / f'per_cluster_quality_k{k}.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {filepath}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate quality comparison plots')
    parser.add_argument('--data_dir', type=str, default='output/data',
                       help='Directory containing comparison results')
    parser.add_argument('--output_dir', type=str, default='output/figures',
                       help='Output directory for figures')
    parser.add_argument('--k_scatter', type=int, default=3,
                       help='k value for 2D scatter plot (default: 3)')
    parser.add_argument('--k_per_cluster', type=int, default=3,
                       help='k value for per-cluster quality plot (default: 3)')

    args = parser.parse_args()

    print("="*60)
    print("GENERATING QUALITY COMPARISON PLOTS")
    print("="*60)

    # Load results
    print(f"Loading results from {args.data_dir}...")
    all_results = load_comparison_results(args.data_dir)
    print(f"Loaded results for k values: {sorted(all_results.keys())}")

    # Plot 1: Method comparison across k
    print("\nGenerating method comparison plot...")
    plot_method_comparison_vs_k(all_results, args.output_dir)

    # Plot 2: 2D scatter for selected k
    print(f"\nGenerating 2D scatter plot for k={args.k_scatter}...")
    plot_2d_scatter(args.k_scatter, all_results, args.output_dir)

    # Plot 3: Per-cluster quality for selected k
    print(f"\nGenerating per-cluster quality plot for k={args.k_per_cluster}...")
    plot_per_cluster_quality(args.k_per_cluster, all_results, args.output_dir)

    print("\n✓ All plots generated successfully!")
    print(f"Figures saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
