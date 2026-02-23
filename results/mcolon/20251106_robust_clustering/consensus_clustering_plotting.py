#!/usr/bin/env python3
"""
Consensus Clustering Plotting Module

Generates visualizations for hierarchical and k-medoids posterior analysis.
Modular design allows easy addition of new plot types.

Available plot types:
1. Posterior heatmaps (p_i(c) matrix)
2. 2D scatter plots (max_p vs log_odds_gap)
3. Trajectory plots with continuous alpha (posterior-weighted)
4. Trajectory plots with category colors (core/uncertain/outlier)
5. Genotype comparison overlays
6. Method comparison (hierarchical vs k-medoids)

Usage:
    # Generate all plots for a genotype
    python consensus_clustering_plotting.py --genotype cep290_homozygous --method hierarchical

    # Generate specific plot type
    python consensus_clustering_plotting.py --plot_type trajectories_posterior --k 4
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analyze.utils.plotting import plot_embryos_metric_over_time


# ============================================================================
# DATA LOADING
# ============================================================================

def load_analysis_results(genotype, method='hierarchical', data_dir='output/data'):
    """
    Load saved analysis results.

    Parameters
    ----------
    genotype : str
        Genotype identifier
    method : str
        'hierarchical' or 'kmedoids'
    data_dir : str
        Base data directory

    Returns
    -------
    dict with all analysis results
    """
    data_dir = Path(data_dir)

    if method == 'hierarchical':
        filepath = data_dir / 'hierarchical' / f'{genotype}_all_k.pkl'
    elif method == 'kmedoids':
        # K-medoids stores results differently (per-k files)
        filepath = data_dir / 'kmedoids' / f'{genotype}_all_k.pkl'
        if not filepath.exists():
            # Try loading from compare_methods_v2 output
            filepath = data_dir / 'all_results.pkl'
    else:
        raise ValueError(f"Unknown method: {method}")

    if not filepath.exists():
        raise FileNotFoundError(f"Results not found: {filepath}")

    with open(filepath, 'rb') as f:
        return pickle.load(f)


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_trajectory_dataframe(trajectories, embryo_ids, common_grid,
                                 posterior_analysis, classification, k):
    """
    Convert trajectories to long-format dataframe with posterior annotations.

    Parameters
    ----------
    trajectories : list of np.ndarray
        Interpolated trajectories
    embryo_ids : list
        Embryo identifiers
    common_grid : np.ndarray
        Time points (HPF)
    posterior_analysis : dict
        Output from analyze_bootstrap_results()
    classification : dict
        Output from classify_embryos_2d()
    k : int
        Number of clusters

    Returns
    -------
    pd.DataFrame with columns:
        embryo_id, time, metric_value, cluster, category,
        posterior_prob, max_p, entropy, posterior_alpha
    """
    rows = []

    for idx, embryo_id in enumerate(embryo_ids):
        traj = trajectories[idx]
        cluster = posterior_analysis['modal_cluster'][idx]
        category = classification['category'][idx]
        posterior_prob = posterior_analysis['p_matrix'][idx, cluster]
        max_p = posterior_analysis['max_p'][idx]
        entropy = posterior_analysis['entropy'][idx]

        # Add data points
        for t, val in zip(common_grid, traj):
            if not np.isnan(val):
                rows.append({
                    'embryo_id': embryo_id,
                    'time': t,
                    'metric_value': val,
                    'cluster': cluster,
                    'category': category,
                    'posterior_prob': posterior_prob,
                    'max_p': max_p,
                    'entropy': entropy,
                    'posterior_alpha': 0.2 + 0.8 * posterior_prob  # Scale to [0.2, 1.0]
                })

    return pd.DataFrame(rows)


# ============================================================================
# PLOT FUNCTIONS
# ============================================================================

def plot_posterior_heatmap(results, k, output_path, show_classification=True):
    """
    Plot posterior probability heatmap (p_i(c) matrix).

    Parameters
    ----------
    results : dict
        Full analysis results
    k : int
        Number of clusters
    output_path : Path or str
        Output file path
    show_classification : bool
        Show classification annotations (core/uncertain/outlier)
    """
    posterior_data = results['results'][k]['posterior_analysis']
    classification = results['results'][k]['classification']

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
    for c in range(k - 1):
        boundary = np.where(modal_cluster_sorted == c)[0][-1] + 0.5
        ax.axhline(y=boundary, color='black', linewidth=2, alpha=0.7)

    # Labels
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Embryo (sorted by modal cluster)', fontsize=12)
    ax.set_title(f'Assignment Posterior Probabilities (k={k})', fontsize=14, fontweight='bold')

    # Set ticks
    ax.set_xticks(range(k))
    ax.set_xticklabels(range(k))
    ax.set_yticks([])

    # Add cluster labels on the right
    for c in range(k):
        mask = (modal_cluster_sorted == c)
        y_center = np.where(mask)[0].mean()
        n_in_cluster = np.sum(mask)
        ax.text(k + 0.2, y_center, f'C{c}\n(n={n_in_cluster})',
               ha='left', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    Saved: {output_path}")


def plot_2d_scatter(results, k, output_path):
    """
    Plot 2D scatter: max_p vs log_odds_gap with classification overlay.

    This validates the 2D gating approach.
    """
    posterior = results['results'][k]['posterior_analysis']
    classification = results['results'][k]['classification']

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot points by category
    colors = {'core': 'C2', 'uncertain': 'C1', 'outlier': 'C3'}
    labels_map = {'core': 'Core', 'uncertain': 'Uncertain', 'outlier': 'Outlier'}

    for cat in ['outlier', 'uncertain', 'core']:  # Plot outliers first (background)
        mask = (classification['category'] == cat)
        log_odds = classification['log_odds_gap'][mask]
        max_p = classification['max_p'][mask]

        # Handle infinite log_odds
        log_odds_finite = np.where(np.isinf(log_odds), 10.0, log_odds)

        ax.scatter(log_odds_finite, max_p,
                  c=colors[cat], label=labels_map[cat],
                  alpha=0.6, s=30, edgecolors='black', linewidths=0.5)

    # Draw decision boundaries
    thresh = classification['thresholds']
    ax.axhline(y=thresh['max_p'], color='black', linestyle='--', linewidth=1.5,
               alpha=0.7, label=f'max_p threshold ({thresh["max_p"]:.2f})')
    ax.axhline(y=thresh['outlier_max_p'], color='red', linestyle='--', linewidth=1.5,
               alpha=0.7, label=f'Outlier threshold ({thresh["outlier_max_p"]:.2f})')
    ax.axvline(x=thresh['log_odds_gap'], color='black', linestyle=':', linewidth=1.5,
               alpha=0.7, label=f'log-odds threshold ({thresh["log_odds_gap"]:.2f})')

    ax.set_xlabel('Log-Odds Gap (log p₁ - log p₂)', fontsize=12)
    ax.set_ylabel('Max Posterior Probability (max_p)', fontsize=12)
    ax.set_title(f'2D Classification Gating (k={k})', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Set reasonable x-axis limits
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    Saved: {output_path}")


def plot_trajectories_posterior(results, k, genotype, output_path):
    """
    Trajectory plot with continuous alpha (posterior-weighted).

    Trajectories are colored by cluster, alpha by posterior probability.
    """
    df = prepare_trajectory_dataframe(
        trajectories=results['trajectories'],
        embryo_ids=results['embryo_ids'],
        common_grid=results['common_grid'],
        posterior_analysis=results['results'][k]['posterior_analysis'],
        classification=results['results'][k]['classification'],
        k=k
    )

    fig = plot_embryos_metric_over_time(
        df=df,
        embryo_col='embryo_id',
        time_col='time',
        metric='metric_value',
        color_by='cluster',
        show_individual=True,
        show_mean=True,
        show_sd_band=True,
        alpha_individual=0.5,
        title=f'{genotype} - k={k} (Posterior-Weighted)',
        figsize=(12, 8)
    )

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    Saved: {output_path}")


def plot_trajectories_category(results, k, genotype, output_path):
    """
    Trajectory plot with category colors (core/uncertain/outlier).

    Colors show assignment quality: green=core, orange=uncertain, red=outlier.
    """
    df = prepare_trajectory_dataframe(
        trajectories=results['trajectories'],
        embryo_ids=results['embryo_ids'],
        common_grid=results['common_grid'],
        posterior_analysis=results['results'][k]['posterior_analysis'],
        classification=results['results'][k]['classification'],
        k=k
    )

    fig = plot_embryos_metric_over_time(
        df=df,
        embryo_col='embryo_id',
        time_col='time',
        metric='metric_value',
        color_by='category',
        show_individual=True,
        show_mean=True,
        show_sd_band=True,
        alpha_individual=0.5,
        title=f'{genotype} - k={k} (Assignment Quality)',
        figsize=(12, 8)
    )

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    Saved: {output_path}")


def plot_membership_vs_k(results, output_path, genotype=None):
    """
    Plot membership percentages vs k (core/uncertain/outlier trends).

    Shows how assignment quality changes with number of clusters.

    Parameters
    ----------
    results : dict
        Full analysis results with multiple k values
    output_path : Path or str
        Output file path
    genotype : str, optional
        Genotype name for title
    """
    k_values = []
    core_pct = []
    uncertain_pct = []
    outlier_pct = []

    # Extract membership percentages for each k
    for k in sorted(results['results'].keys()):
        if results['results'][k] is None:
            continue

        classification = results['results'][k]['classification']
        categories = classification['category']

        n_total = len(categories)
        n_core = np.sum(categories == 'core')
        n_uncertain = np.sum(categories == 'uncertain')
        n_outlier = np.sum(categories == 'outlier')

        k_values.append(k)
        core_pct.append(100 * n_core / n_total)
        uncertain_pct.append(100 * n_uncertain / n_total)
        outlier_pct.append(100 * n_outlier / n_total)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot lines
    ax.plot(k_values, core_pct, 'o-', color='C2', linewidth=2.5,
            markersize=8, label='Core', alpha=0.8)
    ax.plot(k_values, uncertain_pct, 's-', color='C1', linewidth=2.5,
            markersize=8, label='Uncertain', alpha=0.8)
    ax.plot(k_values, outlier_pct, '^-', color='C3', linewidth=2.5,
            markersize=8, label='Outlier', alpha=0.8)

    # Formatting
    ax.set_xlabel('Number of Clusters (k)', fontsize=13)
    ax.set_ylabel('Membership Percentage (%)', fontsize=13)

    title = 'Assignment Quality vs Number of Clusters'
    if genotype:
        title = f'{genotype}: {title}'
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.legend(loc='best', framealpha=0.9, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)
    ax.set_ylim([0, 105])

    # Add percentage labels on points
    for k_val, core, uncertain, outlier in zip(k_values, core_pct, uncertain_pct, outlier_pct):
        ax.text(k_val, core + 2, f'{core:.0f}%', ha='center', va='bottom',
                fontsize=9, color='C2', fontweight='bold')
        ax.text(k_val, uncertain + 2, f'{uncertain:.0f}%', ha='center', va='bottom',
                fontsize=9, color='C1', fontweight='bold')
        ax.text(k_val, outlier + 2, f'{outlier:.0f}%', ha='center', va='bottom',
                fontsize=9, color='C3', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"    Saved: {output_path}")


# ============================================================================
# BATCH PLOTTING
# ============================================================================

def generate_all_plots_for_genotype(genotype, method='hierarchical',
                                    k_values=range(2, 8), output_base='output/figures'):
    """
    Generate all plot types for a single genotype.

    Parameters
    ----------
    genotype : str
        Genotype identifier
    method : str
        'hierarchical' or 'kmedoids'
    k_values : range or list
        k values to plot
    output_base : str
        Base output directory
    """
    print(f"\nGenerating plots for {genotype} ({method})...")

    # Load results
    try:
        results = load_analysis_results(genotype, method)
    except FileNotFoundError as e:
        print(f"  Skipping: {e}")
        return

    # Setup output directories
    output_base = Path(output_base) / method / genotype
    dirs = {
        'heatmaps': output_base / 'posterior_heatmaps',
        'scatters': output_base / 'posterior_scatters',
        'traj_posterior': output_base / 'temporal_trends_posterior',
        'traj_category': output_base / 'temporal_trends_category',
        'trends': output_base / 'membership_trends'
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Generate membership vs k trend plot (once per genotype)
    print(f"  Generating membership vs k trend plot...")
    try:
        plot_membership_vs_k(
            results,
            output_path=dirs['trends'] / 'membership_vs_k.png',
            genotype=genotype
        )
    except Exception as e:
        print(f"    ERROR: {e}")

    # Generate plots for each k
    for k in k_values:
        if k not in results['results'] or results['results'][k] is None:
            print(f"  k={k}: No data, skipping")
            continue

        print(f"  k={k}:")

        try:
            # Posterior heatmap
            plot_posterior_heatmap(
                results, k,
                output_path=dirs['heatmaps'] / f'heatmap_k{k}.png'
            )

            # 2D scatter
            plot_2d_scatter(
                results, k,
                output_path=dirs['scatters'] / f'scatter_k{k}.png'
            )

            # Trajectory - posterior weighted
            plot_trajectories_posterior(
                results, k, genotype,
                output_path=dirs['traj_posterior'] / f'trends_k{k}.png'
            )

            # Trajectory - category colored
            plot_trajectories_category(
                results, k, genotype,
                output_path=dirs['traj_category'] / f'trends_k{k}.png'
            )

        except Exception as e:
            print(f"    ERROR: {e}")

    print(f"  ✓ All plots generated for {genotype}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point for plotting."""
    parser = argparse.ArgumentParser(
        description='Generate consensus clustering plots'
    )
    parser.add_argument(
        '--genotype', '-g',
        type=str,
        default=None,
        help='Genotype to plot (default: all)'
    )
    parser.add_argument(
        '--method', '-m',
        type=str,
        choices=['hierarchical', 'kmedoids', 'both'],
        default='hierarchical',
        help='Clustering method (default: hierarchical)'
    )
    parser.add_argument(
        '--k_min',
        type=int,
        default=2,
        help='Minimum k value (default: 2)'
    )
    parser.add_argument(
        '--k_max',
        type=int,
        default=7,
        help='Maximum k value (default: 7)'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='output/figures',
        help='Output directory (default: output/figures)'
    )

    args = parser.parse_args()

    # Define genotypes
    if args.genotype:
        genotypes = [args.genotype]
    else:
        genotypes = [
            'cep290_wildtype',
            'cep290_heterozygous',
            'cep290_homozygous',
            'cep290_unknown'
        ]

    # Define k range
    k_values = range(args.k_min, args.k_max + 1)

    # Define methods
    if args.method == 'both':
        methods = ['hierarchical', 'kmedoids']
    else:
        methods = [args.method]

    # Generate plots
    print(f"{'='*70}")
    print(f"CONSENSUS CLUSTERING PLOTTING")
    print(f"{'='*70}")
    print(f"Genotypes: {genotypes}")
    print(f"Methods: {methods}")
    print(f"k range: {list(k_values)}")
    print(f"{'='*70}")

    for method in methods:
        for genotype in genotypes:
            try:
                generate_all_plots_for_genotype(
                    genotype=genotype,
                    method=method,
                    k_values=k_values,
                    output_base=args.output_dir
                )
            except Exception as e:
                print(f"\nERROR processing {genotype} ({method}): {e}\n")
                continue

    print(f"\n{'='*70}")
    print(f"✓ PLOTTING COMPLETE")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
