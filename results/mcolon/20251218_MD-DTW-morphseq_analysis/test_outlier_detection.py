#!/usr/bin/env python3
"""
Test Script for Outlier Detection in MD-DTW Analysis

This script tests different outlier detection parameters on the b9d2 dataset
to find the optimal settings that remove singleton outliers while retaining
biological clusters.

Usage:
    python test_outlier_detection.py [--experiment EXPERIMENT_ID]

Purpose:
    - Load pre-computed distance matrix
    - Test multiple outlier detection methods and thresholds
    - Compare clustering results with/without outlier removal
    - Generate comparison visualizations

Created: 2025-12-18
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from md_dtw_prototype import (
    identify_outliers,
    remove_outliers_from_distance_matrix,
    plot_dendrogram,
)
from src.analyze.trajectory_analysis.bootstrap_clustering import get_cluster_assignments


def plot_median_distance_distribution(
    median_distances: np.ndarray,
    embryo_ids: List[str],
    threshold: float,
    outlier_mask: np.ndarray,
    method_name: str,
    save_path: Path,
):
    """
    Plot distribution of median distances with threshold line.

    Shows which embryos are flagged as outliers.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Sort by median distance for clarity
    sorted_idx = np.argsort(median_distances)
    sorted_dists = median_distances[sorted_idx]
    sorted_outliers = outlier_mask[sorted_idx]

    # Plot all embryos
    colors = ['red' if is_outlier else 'steelblue' for is_outlier in sorted_outliers]
    ax.bar(range(len(sorted_dists)), sorted_dists, color=colors, edgecolor='black', alpha=0.7)

    # Threshold line
    ax.axhline(threshold, color='darkred', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.3f}')

    # Labels
    ax.set_xlabel('Embryo (sorted by median distance)', fontsize=12)
    ax.set_ylabel('Median Distance to Other Embryos', fontsize=12)
    ax.set_title(f'Outlier Detection: {method_name}\n{outlier_mask.sum()} outliers detected', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def test_outlier_methods(
    D: np.ndarray,
    embryo_ids: List[str],
    output_dir: Path,
    verbose: bool = True,
):
    """
    Test multiple outlier detection methods and parameters.

    Generates comparison plots and reports for each method.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("Testing Outlier Detection Methods")
    print("=" * 70)

    methods_to_test = [
        {'method': 'percentile', 'percentile': 90, 'name': 'Percentile 90'},
        {'method': 'percentile', 'percentile': 95, 'name': 'Percentile 95'},
        {'method': 'percentile', 'percentile': 98, 'name': 'Percentile 98'},
        {'method': 'mad', 'name': 'MAD (3x)'},
    ]

    results = []

    for i, params in enumerate(methods_to_test):
        print(f"\n{'─' * 70}")
        print(f"Method {i+1}/{len(methods_to_test)}: {params['name']}")
        print(f"{'─' * 70}")

        # Identify outliers
        if params['method'] == 'percentile':
            outlier_ids, inlier_ids, info = identify_outliers(
                D, embryo_ids, method='percentile', percentile=params['percentile'], verbose=verbose
            )
        elif params['method'] == 'mad':
            outlier_ids, inlier_ids, info = identify_outliers(
                D, embryo_ids, method='mad', verbose=verbose
            )

        # Plot distribution
        outlier_mask = np.zeros(len(embryo_ids), dtype=bool)
        outlier_mask[info['outlier_indices']] = True

        plot_median_distance_distribution(
            info['median_distances'],
            embryo_ids,
            info['threshold'],
            outlier_mask,
            params['name'],
            output_dir / f"outlier_detection_{params['name'].replace(' ', '_').lower()}.png"
        )

        # Store results
        results.append({
            'method': params['name'],
            'n_outliers': len(outlier_ids),
            'n_inliers': len(inlier_ids),
            'threshold': info['threshold'],
            'outlier_ids': outlier_ids,
            'inlier_ids': inlier_ids,
            'info': info,
        })

    return results


def compare_clustering_with_without_outliers(
    D: np.ndarray,
    embryo_ids: List[str],
    outlier_method: str = 'percentile',
    outlier_percentile: float = 95,
    k_values: List[int] = [2, 3, 4, 5],
    output_dir: Path = None,
    verbose: bool = True,
):
    """
    Compare clustering results before and after outlier removal.

    Generates side-by-side dendrograms and cluster size comparisons.
    """
    print("\n" + "=" * 70)
    print("Comparing Clustering: With vs Without Outlier Removal")
    print("=" * 70)

    # 1. Cluster WITHOUT outlier removal
    print("\n1. Clustering on FULL dataset (with outliers):")
    print("-" * 50)

    fig_full, _ = plot_dendrogram(
        D,
        embryo_ids,
        k_highlight=k_values,
        title='Full Dataset (With Outliers)',
        save_path=output_dir / 'dendrogram_with_outliers.png',
        verbose=verbose,
    )
    plt.close(fig_full)

    df_assignments_full, _ = get_cluster_assignments(
        distance_matrix=D,
        embryo_ids=embryo_ids,
        k_values=k_values,
        n_bootstrap=100,
        bootstrap_frac=0.8,
        verbose=verbose,
    )

    print("\nCluster sizes (with outliers):")
    for k in k_values:
        cluster_counts = df_assignments_full[f'cluster_k{k}'].value_counts().sort_index()
        print(f"  k={k}: {dict(cluster_counts)}")

    # 2. Remove outliers
    print("\n2. Removing outliers:")
    print("-" * 50)

    D_clean, embryo_ids_clean, outlier_info = remove_outliers_from_distance_matrix(
        D,
        embryo_ids,
        outlier_detection_method=outlier_method,
        outlier_percentile=outlier_percentile,
        verbose=verbose,
    )

    # 3. Cluster WITHOUT outliers
    print("\n3. Clustering on CLEAN dataset (outliers removed):")
    print("-" * 50)

    fig_clean, _ = plot_dendrogram(
        D_clean,
        embryo_ids_clean,
        k_highlight=k_values,
        title='Clean Dataset (Outliers Removed)',
        save_path=output_dir / 'dendrogram_without_outliers.png',
        verbose=verbose,
    )
    plt.close(fig_clean)

    df_assignments_clean, _ = get_cluster_assignments(
        distance_matrix=D_clean,
        embryo_ids=embryo_ids_clean,
        k_values=k_values,
        n_bootstrap=100,
        bootstrap_frac=0.8,
        verbose=verbose,
    )

    print("\nCluster sizes (without outliers):")
    for k in k_values:
        cluster_counts = df_assignments_clean[f'cluster_k{k}'].value_counts().sort_index()
        print(f"  k={k}: {dict(cluster_counts)}")

    # 4. Generate comparison plot
    print("\n4. Generating comparison visualization:")
    print("-" * 50)

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Cluster size comparisons for different k
    for i, k in enumerate(k_values[:4]):  # Show up to 4 k values
        ax = fig.add_subplot(gs[i // 2, i % 2])

        # Get cluster sizes
        counts_full = df_assignments_full[f'cluster_k{k}'].value_counts().sort_index()
        counts_clean = df_assignments_clean[f'cluster_k{k}'].value_counts().sort_index()

        x = np.arange(max(len(counts_full), len(counts_clean)))
        width = 0.35

        # Pad with zeros if needed
        full_vals = [counts_full.get(j, 0) for j in range(len(x))]
        clean_vals = [counts_clean.get(j, 0) for j in range(len(x))]

        ax.bar(x - width/2, full_vals, width, label='With Outliers', color='coral', edgecolor='black')
        ax.bar(x + width/2, clean_vals, width, label='Outliers Removed', color='steelblue', edgecolor='black')

        ax.set_xlabel('Cluster', fontsize=11)
        ax.set_ylabel('Number of Embryos', fontsize=11)
        ax.set_title(f'k={k}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    # Summary text
    ax_text = fig.add_subplot(gs[2, :])
    ax_text.axis('off')

    summary_text = f"""
    OUTLIER REMOVAL SUMMARY
    {'─' * 60}
    Method: {outlier_method} (percentile={outlier_percentile})
    Outliers detected: {len(outlier_info['outlier_indices'])}
    Inliers retained: {len(outlier_info['inlier_indices'])}

    Outlier embryos: {', '.join([embryo_ids[i] for i in outlier_info['outlier_indices']])}

    KEY INSIGHT:
    Removing outliers eliminates singleton clusters and reveals the true
    biological cluster structure in the data.
    """

    ax_text.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('Clustering Comparison: With vs Without Outliers', fontsize=16, fontweight='bold')

    fig.savefig(output_dir / 'clustering_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: clustering_comparison.png")

    print("\n✓ Comparison complete")

    return {
        'df_assignments_full': df_assignments_full,
        'df_assignments_clean': df_assignments_clean,
        'outlier_info': outlier_info,
        'D_clean': D_clean,
        'embryo_ids_clean': embryo_ids_clean,
    }


def main(
    experiment_id: str = '20251121',
    distance_matrix_path: Path = None,
    embryo_ids_path: Path = None,
    output_dir: Path = None,
):
    """
    Main test workflow.
    """
    print("=" * 70)
    print("MD-DTW Outlier Detection Testing")
    print("=" * 70)
    print(f"Experiment: {experiment_id}")

    # Setup paths
    if distance_matrix_path is None:
        # Try to find most recent output
        analysis_dir = Path(__file__).parent
        output_dirs = sorted(analysis_dir.glob(f'output/{experiment_id}_*'))
        if not output_dirs:
            raise FileNotFoundError(f"No output directory found for experiment {experiment_id}")
        distance_matrix_path = output_dirs[-1] / 'distance_matrix.npy'
        embryo_ids_path = output_dirs[-1] / 'embryo_ids.txt'
        print(f"Using output from: {output_dirs[-1].name}")

    if output_dir is None:
        output_dir = Path(__file__).parent / 'output' / 'outlier_testing'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load data
    print("\nLoading distance matrix and embryo IDs...")
    D = np.load(distance_matrix_path)
    with open(embryo_ids_path, 'r') as f:
        embryo_ids = [line.strip() for line in f]

    print(f"  Distance matrix shape: {D.shape}")
    print(f"  Number of embryos: {len(embryo_ids)}")

    # Test different outlier detection methods
    print("\n" + "=" * 70)
    print("PART 1: Testing Different Outlier Detection Methods")
    print("=" * 70)

    test_results = test_outlier_methods(D, embryo_ids, output_dir, verbose=True)

    # Compare clustering with/without outliers
    print("\n" + "=" * 70)
    print("PART 2: Clustering Comparison")
    print("=" * 70)

    comparison_results = compare_clustering_with_without_outliers(
        D,
        embryo_ids,
        outlier_method='percentile',
        outlier_percentile=95,
        k_values=[2, 3, 4, 5],
        output_dir=output_dir,
        verbose=True,
    )

    # Summary
    print("\n" + "=" * 70)
    print("TESTING COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Review outlier detection plots to choose optimal method/threshold")
    print("  2. Inspect dendrograms (with vs without outliers)")
    print("  3. Update run_analysis.py with chosen outlier removal parameters")
    print("  4. Re-run full analysis with outlier removal enabled")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test outlier detection methods for MD-DTW analysis'
    )
    parser.add_argument(
        '--experiment', '-e',
        type=str,
        default='20251121',
        help='Experiment ID (default: 20251121)'
    )
    parser.add_argument(
        '--distance-matrix',
        type=str,
        default=None,
        help='Path to distance matrix .npy file (auto-detected if not specified)'
    )
    parser.add_argument(
        '--embryo-ids',
        type=str,
        default=None,
        help='Path to embryo IDs .txt file (auto-detected if not specified)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory for test results'
    )

    args = parser.parse_args()

    main(
        experiment_id=args.experiment,
        distance_matrix_path=Path(args.distance_matrix) if args.distance_matrix else None,
        embryo_ids_path=Path(args.embryo_ids) if args.embryo_ids else None,
        output_dir=Path(args.output) if args.output else None,
    )
