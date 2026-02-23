"""
Plot Cluster Trajectories

Integrates posterior-based cluster assignments with trajectory visualization.
Uses plot_embryos_metric_over_time() from src/analyze/utils/plotting.py.

Usage:
    python plot_cluster_trajectories.py --k 3 --metric normalized_baseline_deviation
"""

import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pickle
from pathlib import Path
import argparse
from typing import Dict, Optional

# Add src to path for plotting utilities
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analyze.utils.plotting import plot_embryos_metric_over_time, get_membership_category_colors


def load_analysis_results(genotype: str, method: str = 'hierarchical', data_dir: str = 'output/data') -> Dict:
    """Load posterior analysis results for a genotype."""
    data_dir = Path(data_dir)

    if method == 'hierarchical':
        filepath = data_dir / 'hierarchical' / f'{genotype}_all_k.pkl'
    else:
        filepath = data_dir / 'kmedoids' / f'{genotype}_all_k.pkl'

    if not filepath.exists():
        raise FileNotFoundError(f"Results not found: {filepath}")

    with open(filepath, 'rb') as f:
        return pickle.load(f)


def load_embryo_data(metadata_file: Optional[str] = None) -> pd.DataFrame:
    """
    Load embryo metadata and metrics for trajectory plotting.

    Parameters
    ----------
    metadata_file : str, optional
        Path to metadata file. If None, uses default location.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns: embryo_id, predicted_stage_hpf, metric values
    """
    if metadata_file is None:
        # Try default locations
        default_paths = [
            '/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251103_DTW_analysis/embryo_data.csv',
            '/net/trapnell/vol1/home/mdcolon/proj/morphseq/test_data/embryo_metadata.csv'
        ]
        for path in default_paths:
            if Path(path).exists():
                metadata_file = path
                break

    if metadata_file is None or not Path(metadata_file).exists():
        raise FileNotFoundError(
            "Could not find embryo metadata file. Please specify with --metadata_file"
        )

    print(f"Loading embryo data from {metadata_file}")
    df = pd.read_csv(metadata_file)
    return df


def prepare_trajectory_dataframe(results: Dict, k: int) -> pd.DataFrame:
    """
    Convert trajectories to long-format dataframe with posterior annotations.

    Parameters
    ----------
    results : dict
        Full analysis results with all k values
    k : int
        Number of clusters to use

    Returns
    -------
    pd.DataFrame with columns:
        embryo_id, time, metric_value, cluster, category
    """
    if k not in results['results'] or results['results'][k] is None:
        raise ValueError(f"No results for k={k}")

    trajectories = results['trajectories']
    common_grid = results['common_grid']
    embryo_ids = results['embryo_ids']
    posterior_analysis = results['results'][k]['posterior_analysis']
    classification = results['results'][k]['classification']

    # Build dataframe directly from aligned arrays
    data_list = []

    for idx, embryo_id in enumerate(embryo_ids):
        # Get trajectory (handle if it's a list or array)
        if isinstance(trajectories, list):
            traj = np.asarray(trajectories[idx], dtype=float)
        else:
            traj = np.asarray(trajectories[idx], dtype=float)

        # Get cluster and classification
        cluster_id = int(posterior_analysis['modal_cluster'][idx])
        category = str(classification['category'][idx])

        # Create rows for each time point
        n_points = len(traj)
        for i in range(n_points):
            val = traj[i]
            if not np.isnan(val):
                data_list.append({
                    'embryo_id': str(embryo_id),
                    'time': float(common_grid[i]),
                    'metric_value': float(val),
                    'cluster': cluster_id,
                    'category': category
                })

    if not data_list:
        raise ValueError(f"No valid data points for k={k}")

    return pd.DataFrame(data_list)


def plot_trajectories_by_cluster(genotype: str, k: int, method: str = 'hierarchical',
                                 data_dir: str = 'output/data',
                                 output_dir: str = 'output/figures'):
    """
    Plot embryo trajectories colored by cluster assignment.

    Parameters
    ----------
    genotype : str
        Genotype identifier
    k : int
        Number of clusters
    method : str
        'hierarchical' or 'kmedoids'
    data_dir : str
        Directory containing posterior results
    output_dir : str
        Directory to save figures
    """
    # Load results
    print(f"  k={k}: Loading results...")
    try:
        results = load_analysis_results(genotype, method, data_dir)
    except FileNotFoundError as e:
        print(f"    Skipping: {e}")
        return

    # Prepare trajectory dataframe
    try:
        df = prepare_trajectory_dataframe(results, k)
    except ValueError as e:
        print(f"    Skipping: {e}")
        return

    # Create output directory
    output_dir = Path(output_dir) / method / genotype / 'cluster_overlays'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot trajectories colored by cluster
    print(f"    Plotting cluster overlay...")
    fig = plot_embryos_metric_over_time(
        df=df,
        embryo_col='embryo_id',
        time_col='time',
        metric='metric_value',
        color_by='cluster',
        show_individual=True,
        show_mean=True,
        show_sd_band=True,
        alpha_individual=0.4,
        title=f'{genotype} - Trajectories by Cluster (k={k})',
        figsize=(12, 6)
    )

    # Save figure
    filepath = output_dir / f'cluster_overlay_k{k}.png'
    fig.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {filepath}")


def plot_trajectories_by_membership(genotype: str, k: int, method: str = 'hierarchical',
                                    data_dir: str = 'output/data',
                                    output_dir: str = 'output/figures'):
    """
    Plot per-cluster panels with trajectories colored by membership classification.
    """
    # Load results
    print(f"  k={k}: Loading results...")
    try:
        results = load_analysis_results(genotype, method, data_dir)
    except FileNotFoundError as e:
        print(f"    Skipping: {e}")
        return

    # Prepare trajectory dataframe
    try:
        df = prepare_trajectory_dataframe(results, k)
    except ValueError as e:
        print(f"    Skipping: {e}")
        return

    clusters = sorted(df['cluster'].unique())
    if not clusters:
        print(f"    Skipping: no clusters found for k={k}")
        return

    # Determine membership categories with preferred ordering
    category_values = df['category'].dropna().astype(str).unique().tolist()
    preferred_order = ['core', 'uncertain', 'outlier']
    ordered_categories = [c for c in preferred_order if c in category_values]
    ordered_categories += [c for c in category_values if c not in ordered_categories]
    if not ordered_categories:
        ordered_categories = ['unlabeled']
        df['category'] = 'unlabeled'

    # Use standardized colors for membership categories
    category_colors = get_membership_category_colors(ordered_categories)

    n_clusters = len(clusters)
    ncols = min(3, max(1, n_clusters))
    nrows = math.ceil(n_clusters / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * 5.0, nrows * 4.0),
        sharex=True,
        sharey=True,
        squeeze=False
    )
    axes_flat = axes.flatten()

    for ax_idx, cluster_id in enumerate(clusters):
        ax = axes_flat[ax_idx]
        cluster_df = df[df['cluster'] == cluster_id].copy()
        cluster_df = cluster_df.sort_values('time')
        n_embryos = cluster_df['embryo_id'].nunique()

        for embryo_id, embryo_df in cluster_df.groupby('embryo_id'):
            embryo_df = embryo_df.sort_values('time')
            category = str(embryo_df['category'].iloc[0])
            color = category_colors.get(category, '#999999')
            ax.plot(
                embryo_df['time'],
                embryo_df['metric_value'],
                color=color,
                alpha=0.8,
                linewidth=1.1
            )

        ax.set_title(f'Cluster {cluster_id} (n={n_embryos})', fontsize=11)
        if ax_idx % ncols == 0:
            ax.set_ylabel('Metric value', fontsize=10)
        if ax_idx >= (nrows - 1) * ncols:
            ax.set_xlabel('Time (hpf)', fontsize=10)
        ax.grid(True, alpha=0.2)

    # Hide unused axes, if any
    for ax in axes_flat[len(clusters):]:
        ax.axis('off')

    legend_handles = [
        Line2D([0], [0], color=color, lw=2, label=category.title())
        for category, color in category_colors.items()
    ]

    fig.legend(
        handles=legend_handles,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(legend_handles),
        frameon=False
    )

    fig.suptitle(
        f'{genotype} - Membership Detail by Cluster (k={k})',
        fontsize=14,
        fontweight='bold',
        y=0.97
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))

    # Create output directory
    output_dir = Path(output_dir) / method / genotype / 'cluster_membership_panels'
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / f'membership_panels_k{k}.png'
    fig.savefig(filepath, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {filepath}")


def main():
    """Generate cluster overlay and temporal trends plots for posterior analysis."""
    parser = argparse.ArgumentParser(
        description='Generate cluster trajectory plots from posterior analysis results'
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
        '--data_dir',
        type=str,
        default='output/data',
        help='Directory containing posterior results'
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

    print(f"{'='*70}")
    print(f"CLUSTER TRAJECTORY PLOTTING")
    print(f"{'='*70}")
    print(f"Genotypes: {genotypes}")
    print(f"Methods: {methods}")
    print(f"k range: {list(k_values)}")
    print(f"{'='*70}")

    # Generate plots
    for method in methods:
        for genotype in genotypes:
            print(f"\nGenerating plots for {genotype} ({method})...")
            for k in k_values:
                try:
                    # Cluster overlay plot
                    plot_trajectories_by_cluster(genotype, k, method, args.data_dir, args.output_dir)

                    # Temporal trends with membership plot
                    plot_trajectories_by_membership(genotype, k, method, args.data_dir, args.output_dir)
                except Exception as e:
                    import traceback
                    print(f"    ERROR at k={k}: {e}")
                    traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"✓ PLOTTING COMPLETE")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
