"""
B9D2 Trajectory Clustering - Phase 1

DTW clustering on total_length_um trajectories to identify penetrant vs non-penetrant
phenotype groups in b9d2_pair_7 and b9d2_pair_8.

Generates cluster inspection plots for multiple k values (2-5) with genotype
composition to help identify which clusters represent penetrant phenotypes.

Usage:
    python b9d2_trajectory_clustering.py

Output:
    - cluster_inspection_k{k}.png for k = 2, 3, 4, 5
    - cluster_inspection_k{k}_by_genotype.png (genotype-colored)
    - cluster_assignments_k{k}.csv
    - cluster_genotype_summary_k{k}.csv

Author: Generated via Claude Code
Date: 2025-12-10
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Add src to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

from src.analyze.trajectory_analysis.data_loading import _load_qc_staged
from src.analyze.trajectory_analysis import (
    compute_dtw_distance_matrix,
    run_bootstrap_hierarchical,
    analyze_bootstrap_results,
)

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_IDS = ['20251119', '20251125']
PAIRS = ['b9d2_pair_7', 'b9d2_pair_8']
K_VALUES = [2, 3, 4, 5]

OUTPUT_DIR = Path(__file__).parent / 'output'
FIGURES_DIR = OUTPUT_DIR / 'figures'

# Trajectory parameters
TIME_COL = 'predicted_stage_hpf'
METRIC_COL = 'total_length_um'
EMBRYO_ID_COL = 'embryo_id'
GENOTYPE_COL = 'genotype'
PAIR_COL = 'pair'

MIN_TIMEPOINTS = 5
GRID_STEP = 0.5  # HPF for interpolation

# Clustering parameters
N_BOOTSTRAP = 100
BOOTSTRAP_FRAC = 0.8
RANDOM_SEED = 42
DTW_WINDOW = 3

# Genotype colors
GENOTYPE_COLORS = {
    'b9d2_wildtype': '#2ca02c',       # Green
    'b9d2_heterozygous': '#1f77b4',   # Blue
    'b9d2_homozygous': '#d62728',     # Red
}

# Cluster colors (for trend-based coloring)
CLUSTER_COLORS = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
]


# =============================================================================
# Data Loading
# =============================================================================

def load_and_prepare_data():
    """
    Load data from multiple experiments and filter for target pairs.

    Returns
    -------
    df : pd.DataFrame
        Combined dataframe with all experiments and pairs
    """
    print("Loading data from experiments...")

    dfs = []
    for exp_id in EXPERIMENT_IDS:
        print(f"  Loading {exp_id}...")
        df = _load_qc_staged(exp_id)
        df['experiment_id'] = exp_id
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print(f"  Combined: {len(df)} rows")

    # Filter for valid embryos
    if 'use_embryo_flag' in df.columns:
        df = df[df['use_embryo_flag'] == 1].copy()
        print(f"  After use_embryo_flag filter: {len(df)} rows")

    # Filter for target pairs
    df = df[df[PAIR_COL].isin(PAIRS)].copy()
    print(f"  After pair filter: {len(df)} rows")

    # Drop rows with missing values
    df = df.dropna(subset=[EMBRYO_ID_COL, TIME_COL, METRIC_COL, GENOTYPE_COL])
    print(f"  After dropna: {len(df)} rows")

    print(f"\nData summary:")
    print(f"  Unique embryos: {df[EMBRYO_ID_COL].nunique()}")
    print(f"  Pairs: {df[PAIR_COL].unique().tolist()}")
    print(f"  Genotypes: {df[GENOTYPE_COL].unique().tolist()}")
    print(f"  Experiments: {df['experiment_id'].unique().tolist()}")

    return df


def extract_trajectories(df):
    """
    Extract trajectories and interpolate to common grid.

    Returns
    -------
    trajectories : list of ndarray
        Interpolated trajectory arrays
    embryo_ids : list
        Corresponding embryo IDs
    common_grid : ndarray
        Time grid (hpf)
    df_interpolated : pd.DataFrame
        Interpolated data in long format with genotype
    """
    print("\nExtracting trajectories...")

    # Filter embryos with enough timepoints
    embryo_counts = df.groupby(EMBRYO_ID_COL).size()
    valid_embryos = embryo_counts[embryo_counts >= MIN_TIMEPOINTS].index.tolist()
    df_filtered = df[df[EMBRYO_ID_COL].isin(valid_embryos)]

    print(f"  {len(valid_embryos)} embryos with >= {MIN_TIMEPOINTS} timepoints")

    # Create common time grid
    time_min = np.floor(df_filtered[TIME_COL].min() / GRID_STEP) * GRID_STEP
    time_max = np.ceil(df_filtered[TIME_COL].max() / GRID_STEP) * GRID_STEP
    common_grid = np.arange(time_min, time_max + GRID_STEP, GRID_STEP)

    print(f"  Time grid: {time_min:.1f} to {time_max:.1f} hpf ({len(common_grid)} points)")

    # Interpolate each embryo
    trajectories = []
    embryo_ids = []
    interpolated_records = []

    # Get genotype mapping
    genotype_map = df_filtered[[EMBRYO_ID_COL, GENOTYPE_COL, PAIR_COL, 'experiment_id']].drop_duplicates(subset=[EMBRYO_ID_COL])

    for embryo_id in valid_embryos:
        embryo_data = df_filtered[df_filtered[EMBRYO_ID_COL] == embryo_id].sort_values(TIME_COL)

        if len(embryo_data) < 2:
            continue

        # Get genotype for this embryo
        genotype = embryo_data[GENOTYPE_COL].iloc[0]
        pair = embryo_data[PAIR_COL].iloc[0]
        exp_id = embryo_data['experiment_id'].iloc[0]

        # Interpolate
        interp_values = np.interp(
            common_grid,
            embryo_data[TIME_COL].values,
            embryo_data[METRIC_COL].values
        )

        trajectories.append(interp_values)
        embryo_ids.append(embryo_id)

        for t, v in zip(common_grid, interp_values):
            interpolated_records.append({
                'embryo_id': embryo_id,
                'hpf': t,
                'metric_value': v,
                'genotype': genotype,
                'pair': pair,
                'experiment_id': exp_id,
            })

    df_interpolated = pd.DataFrame(interpolated_records)

    print(f"  Interpolated {len(trajectories)} trajectories")

    # Print genotype breakdown
    genotype_counts = df_interpolated.groupby('genotype')['embryo_id'].nunique()
    print(f"\n  Genotype breakdown:")
    for geno, count in genotype_counts.items():
        print(f"    {geno}: {count} embryos")

    return trajectories, embryo_ids, common_grid, df_interpolated


# =============================================================================
# Clustering
# =============================================================================

def run_clustering(trajectories, embryo_ids, k):
    """
    Run DTW clustering for a specific k value.

    Returns
    -------
    cluster_assignments : dict
        embryo_id -> cluster_id
    posteriors : dict
        Bootstrap posteriors
    D : ndarray
        Distance matrix (only computed once, reused)
    """
    print(f"\nRunning clustering for k={k}...")

    # Compute DTW distance matrix
    print(f"  Computing DTW distance matrix (window={DTW_WINDOW})...")
    D = compute_dtw_distance_matrix(trajectories, window=DTW_WINDOW, verbose=False)

    # Bootstrap clustering
    print(f"  Running bootstrap (n={N_BOOTSTRAP})...")
    bootstrap_results = run_bootstrap_hierarchical(
        D, k, embryo_ids,
        n_bootstrap=N_BOOTSTRAP,
        frac=BOOTSTRAP_FRAC,
        random_state=RANDOM_SEED,
        verbose=False
    )

    posteriors = analyze_bootstrap_results(bootstrap_results)

    # Create assignment dict
    cluster_assignments = {
        eid: cluster
        for eid, cluster in zip(posteriors['embryo_ids'], posteriors['modal_cluster'])
    }

    return cluster_assignments, posteriors, D


def compute_cluster_stats(df_interpolated, cluster_assignments):
    """
    Compute statistics for each cluster.

    Returns
    -------
    cluster_stats : dict
        cluster_id -> {'mean': float, 'n_embryos': int, 'genotype_counts': dict}
    """
    cluster_ids = sorted(set(cluster_assignments.values()))
    cluster_stats = {}

    for cluster_id in cluster_ids:
        # Get embryos in this cluster
        cluster_embryos = [eid for eid, cid in cluster_assignments.items() if cid == cluster_id]
        df_cluster = df_interpolated[df_interpolated['embryo_id'].isin(cluster_embryos)]

        # Compute mean trajectory value
        mean_value = df_cluster['metric_value'].mean()

        # Get genotype counts
        genotype_counts = df_cluster.groupby('genotype')['embryo_id'].nunique().to_dict()

        cluster_stats[cluster_id] = {
            'mean': mean_value,
            'n_embryos': len(cluster_embryos),
            'genotype_counts': genotype_counts,
        }

    return cluster_stats


def generate_genotype_summary(cluster_assignments, df_interpolated, k):
    """
    Generate cluster × genotype cross-tabulation.
    """
    # Get unique embryo info
    embryo_info = df_interpolated[['embryo_id', 'genotype']].drop_duplicates()

    # Add cluster assignments
    embryo_info = embryo_info.copy()
    embryo_info['cluster_id'] = embryo_info['embryo_id'].map(cluster_assignments)

    # Cross-tabulation
    summary = embryo_info.groupby(['cluster_id', 'genotype']).size().reset_index(name='count')

    return summary


# =============================================================================
# Plotting
# =============================================================================

def plot_cluster_inspection(df_interpolated, cluster_assignments, cluster_stats, k, save_path):
    """
    Generate cluster inspection plot with trajectories colored by cluster.
    """
    print(f"  Generating cluster inspection plot (k={k})...")

    n_clusters = len(cluster_stats)
    fig, axes = plt.subplots(1, n_clusters, figsize=(5 * n_clusters, 5), sharey=True)

    if n_clusters == 1:
        axes = [axes]

    for idx, (cluster_id, stats) in enumerate(sorted(cluster_stats.items())):
        ax = axes[idx]
        color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]

        # Get embryos in this cluster
        cluster_embryos = [eid for eid, cid in cluster_assignments.items() if cid == cluster_id]
        df_cluster = df_interpolated[df_interpolated['embryo_id'].isin(cluster_embryos)]

        # Plot individual trajectories
        for embryo_id in cluster_embryos:
            embryo_data = df_cluster[df_cluster['embryo_id'] == embryo_id]
            ax.plot(embryo_data['hpf'], embryo_data['metric_value'],
                   alpha=0.3, linewidth=0.8, color=color)

        # Plot mean trajectory
        mean_traj = df_cluster.groupby('hpf')['metric_value'].mean()
        ax.plot(mean_traj.index, mean_traj.values,
               linewidth=3, color='black', label='Mean')

        # Title with stats
        geno_str = ", ".join([f"{g.replace('b9d2_', '')[:3]}={c}"
                              for g, c in sorted(stats['genotype_counts'].items())])
        ax.set_title(f"Cluster {cluster_id}\n"
                    f"n={stats['n_embryos']}, mean={stats['mean']:.1f}\n"
                    f"({geno_str})",
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('Time (hpf)', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Total Length (µm)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle(f'Cluster Inspection (k={k}) - B9D2 pair_7 + pair_8\n'
                 f'{len(cluster_assignments)} embryos',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"    Saved: {save_path}")
    plt.close(fig)


def plot_cluster_by_genotype(df_interpolated, cluster_assignments, cluster_stats, k, save_path):
    """
    Generate cluster inspection plot with trajectories colored by GENOTYPE.
    """
    print(f"  Generating genotype-colored plot (k={k})...")

    n_clusters = len(cluster_stats)
    fig, axes = plt.subplots(1, n_clusters, figsize=(5 * n_clusters, 5), sharey=True)

    if n_clusters == 1:
        axes = [axes]

    for idx, (cluster_id, stats) in enumerate(sorted(cluster_stats.items())):
        ax = axes[idx]

        # Get embryos in this cluster
        cluster_embryos = [eid for eid, cid in cluster_assignments.items() if cid == cluster_id]
        df_cluster = df_interpolated[df_interpolated['embryo_id'].isin(cluster_embryos)]

        # Plot individual trajectories colored by genotype
        for embryo_id in cluster_embryos:
            embryo_data = df_cluster[df_cluster['embryo_id'] == embryo_id]
            if len(embryo_data) == 0:
                continue

            genotype = embryo_data['genotype'].iloc[0]
            color = GENOTYPE_COLORS.get(genotype, '#808080')

            ax.plot(embryo_data['hpf'], embryo_data['metric_value'],
                   alpha=0.4, linewidth=0.8, color=color)

        # Plot mean trajectory
        mean_traj = df_cluster.groupby('hpf')['metric_value'].mean()
        ax.plot(mean_traj.index, mean_traj.values,
               linewidth=3, color='black', label='Mean', zorder=100)

        # Title with stats
        geno_str = ", ".join([f"{g.replace('b9d2_', '')[:3]}={c}"
                              for g, c in sorted(stats['genotype_counts'].items())])
        ax.set_title(f"Cluster {cluster_id}\n"
                    f"n={stats['n_embryos']}, mean={stats['mean']:.1f}\n"
                    f"({geno_str})",
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('Time (hpf)', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Total Length (µm)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Global genotype legend
    legend_elements = [
        Patch(facecolor=GENOTYPE_COLORS['b9d2_wildtype'], label='Wildtype'),
        Patch(facecolor=GENOTYPE_COLORS['b9d2_heterozygous'], label='Heterozygous'),
        Patch(facecolor=GENOTYPE_COLORS['b9d2_homozygous'], label='Homozygous'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10, title='Genotype')

    plt.suptitle(f'Cluster Inspection by Genotype (k={k}) - B9D2 pair_7 + pair_8\n'
                 f'{len(cluster_assignments)} embryos',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"    Saved: {save_path}")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("B9D2 TRAJECTORY CLUSTERING - PHASE 1")
    print("=" * 80)
    print(f"Experiments: {EXPERIMENT_IDS}")
    print(f"Pairs: {PAIRS}")
    print(f"K values: {K_VALUES}")
    print("=" * 80)

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_and_prepare_data()

    # Extract trajectories
    trajectories, embryo_ids, common_grid, df_interpolated = extract_trajectories(df)

    # Compute distance matrix once
    print("\nComputing DTW distance matrix...")
    D = compute_dtw_distance_matrix(trajectories, window=DTW_WINDOW, verbose=False)
    print(f"  Distance matrix shape: {D.shape}")

    # Run clustering for each k
    all_results = {}

    for k in K_VALUES:
        print(f"\n{'='*60}")
        print(f"K = {k}")
        print(f"{'='*60}")

        # Run bootstrap clustering
        print(f"  Running bootstrap clustering (n={N_BOOTSTRAP})...")
        bootstrap_results = run_bootstrap_hierarchical(
            D, k, embryo_ids,
            n_bootstrap=N_BOOTSTRAP,
            frac=BOOTSTRAP_FRAC,
            random_state=RANDOM_SEED,
            verbose=False
        )

        posteriors = analyze_bootstrap_results(bootstrap_results)

        # Create assignment dict
        cluster_assignments = {
            eid: cluster
            for eid, cluster in zip(posteriors['embryo_ids'], posteriors['modal_cluster'])
        }

        # Compute cluster stats
        cluster_stats = compute_cluster_stats(df_interpolated, cluster_assignments)

        # Print summary
        print(f"\n  Cluster summary:")
        for cid, stats in sorted(cluster_stats.items()):
            geno_str = ", ".join([f"{g.replace('b9d2_', '')}={c}"
                                  for g, c in sorted(stats['genotype_counts'].items())])
            print(f"    Cluster {cid}: n={stats['n_embryos']}, mean={stats['mean']:.1f}µm ({geno_str})")

        # Generate plots
        plot_cluster_inspection(
            df_interpolated, cluster_assignments, cluster_stats, k,
            save_path=OUTPUT_DIR / f'cluster_inspection_k{k}.png'
        )

        plot_cluster_by_genotype(
            df_interpolated, cluster_assignments, cluster_stats, k,
            save_path=OUTPUT_DIR / f'cluster_inspection_k{k}_by_genotype.png'
        )

        # Save cluster assignments
        assignments_df = pd.DataFrame([
            {'embryo_id': eid, 'cluster_id': cid}
            for eid, cid in cluster_assignments.items()
        ])
        # Add genotype
        geno_map = df_interpolated[['embryo_id', 'genotype', 'pair', 'experiment_id']].drop_duplicates()
        assignments_df = assignments_df.merge(geno_map, on='embryo_id', how='left')
        assignments_df.to_csv(OUTPUT_DIR / f'cluster_assignments_k{k}.csv', index=False)

        # Save genotype summary
        geno_summary = generate_genotype_summary(cluster_assignments, df_interpolated, k)
        geno_summary.to_csv(OUTPUT_DIR / f'cluster_genotype_summary_k{k}.csv', index=False)

        # Print genotype × cluster table
        print(f"\n  Genotype × Cluster table:")
        pivot = geno_summary.pivot(index='cluster_id', columns='genotype', values='count').fillna(0).astype(int)
        print(pivot.to_string())

        all_results[k] = {
            'cluster_assignments': cluster_assignments,
            'cluster_stats': cluster_stats,
            'posteriors': posteriors,
        }

    # Final summary
    print("\n" + "=" * 80)
    print("CLUSTERING COMPLETE - REVIEW RESULTS")
    print("=" * 80)
    print(f"\nOutput saved to: {OUTPUT_DIR}")
    print(f"\nCluster inspection plots:")
    for k in K_VALUES:
        print(f"  k={k}: cluster_inspection_k{k}.png, cluster_inspection_k{k}_by_genotype.png")

    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Review cluster_inspection_k{k}_by_genotype.png for each k")
    print("2. Identify which k best separates penetrant (shorter, more homo) vs non-penetrant")
    print("3. Update b9d2_trajectory_classifier.py with:")
    print("   - SELECTED_K = <chosen k>")
    print("   - PENETRANT_CLUSTERS = [<cluster ids with lower mean length>]")
    print("4. Run b9d2_trajectory_classifier.py to get F1-score vs time bin")
    print("=" * 80)


if __name__ == '__main__':
    main()
