#!/usr/bin/env python
"""
CEP290 Cluster Validation via Random Split Control

PRIORITY 1: Validates that per-cluster AUROC increases are not sample size artifacts.

Method:
-------
1. Take all homozygous embryos (N=18)
2. Randomly split into 3 "pseudo-clusters" matching real cluster sizes
3. Run classification on each pseudo-cluster vs WT
4. Compare AUROC to real clusters
5. Repeat random splits many times to get distribution

Expected Outcome:
-----------------
- Real clusters should have significantly higher AUROC than random splits
- Random splits should perform near chance (AUROC ~ 0.5-0.6)
- Real bumpy, high_to_low, low_to_high should show AUROC > 0.7-0.8

Author: Generated for CEP290 lab meeting analysis
Date: 2026-01-02
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.analyze.difference_detection.comparison import compare_groups

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = Path(__file__).parent / "validation_results"
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251229_cep290_phenotype_extraction")

CLUSTERING_DATA_PATH = DATA_DIR / "data" / "clustering_data__early_homo.pkl"
K_RESULTS_PATH = DATA_DIR / "kmedoids_k_selection_early_timepoints_cep290_data" / "k_results.pkl"

# Analysis parameters
METRIC_COL = 'baseline_deviation_normalized'
K_VALUE = 5
TIME_BIN_WIDTH = 2.0  # hours
N_PERMUTATIONS_CLASS = 100  # permutations per classification (increase for stable p-values)
N_JOBS = -1  # parallel jobs for permutation testing (-1 = all cores, 1 = serial)
N_RANDOM_SPLITS = 20  # number of random splits to test
MIN_SAMPLES_PER_BIN = 3
RANDOM_SEED = 42

# Cluster definitions
CLUSTER_NAMES_K5 = {
    0: 'outlier',
    1: 'bumpy',
    2: 'low_to_high',
    3: 'low_to_high',
    4: 'high_to_low',
}

CLUSTERS_TO_VALIDATE = ['bumpy', 'high_to_low', 'low_to_high']

# Colors
COLORS_REAL = {
    'bumpy': '#9467BD',
    'low_to_high': '#17BECF',
    'high_to_low': '#E377C2',
}

COLORS_RANDOM = {
    'random_group_0': '#CCCCCC',
    'random_group_1': '#AAAAAA',
    'random_group_2': '#888888',
}


# =============================================================================
# Data Loading
# =============================================================================

def load_data():
    """Load clustering data and assign real clusters."""
    print("Loading data...")

    with open(CLUSTERING_DATA_PATH, 'rb') as f:
        clustering_data = pickle.load(f)

    with open(K_RESULTS_PATH, 'rb') as f:
        k_results = pickle.load(f)

    df = clustering_data['df_cep290_earyltimepoints'].copy()
    print(f"  Loaded {len(df)} rows, {df['embryo_id'].nunique()} embryos")

    # Add real cluster assignments
    cluster_labels = k_results['clustering_by_k'][K_VALUE]['assignments']['cluster_labels']
    embryo_ids_clustered = k_results['embryo_ids']

    embryo_to_cluster = dict(zip(embryo_ids_clustered, cluster_labels))
    embryo_to_cluster_name = {
        eid: CLUSTER_NAMES_K5[cid] for eid, cid in embryo_to_cluster.items()
    }

    df['real_cluster'] = df['embryo_id'].map(embryo_to_cluster_name)

    # Get homozygous embryos only
    df_homo = df[df['genotype'] == 'cep290_homozygous'].copy()
    homo_embryo_ids = df_homo['embryo_id'].unique()

    print(f"\nHomozygous embryos: {len(homo_embryo_ids)}")
    print("\nReal cluster distribution:")
    for cluster in CLUSTERS_TO_VALIDATE:
        count = df_homo[df_homo['real_cluster'] == cluster]['embryo_id'].nunique()
        print(f"  {cluster}: {count} embryos")

    return df, df_homo, homo_embryo_ids


# =============================================================================
# Real Cluster Classification
# =============================================================================

def run_real_cluster_classification(df):
    """Run classification on real clusters (baseline for comparison)."""
    print("\n" + "="*70)
    print("STEP 1: Real Cluster Classification (Ground Truth)")
    print("="*70)

    real_results = {}

    for cluster in CLUSTERS_TO_VALIDATE:
        print(f"\n--- Real Cluster: {cluster} vs WT ---")

        # Create subset
        df_subset = df[
            (df['real_cluster'] == cluster) | (df['genotype'] == 'cep290_wildtype')
        ].copy()

        df_subset['comparison_group'] = df_subset.apply(
            lambda r: cluster if r['real_cluster'] == cluster else 'WT', axis=1
        )

        n_cluster = df_subset[df_subset['comparison_group'] == cluster]['embryo_id'].nunique()
        n_wt = df_subset[df_subset['comparison_group'] == 'WT']['embryo_id'].nunique()
        print(f"  {cluster}: {n_cluster} embryos, WT: {n_wt} embryos")

        results = compare_groups(
            df_subset,
            group_col='comparison_group',
            group1=cluster,
            group2='WT',
            features='z_mu_b',
            morphology_metric=METRIC_COL,
            bin_width=TIME_BIN_WIDTH,
            n_permutations=N_PERMUTATIONS_CLASS,
            n_jobs=N_JOBS,
            min_samples_per_bin=MIN_SAMPLES_PER_BIN,
            random_state=RANDOM_SEED,
            verbose=True
        )

        real_results[cluster] = results

        summary = results['summary']
        max_auroc = summary['max_auroc'] if summary['max_auroc'] is not None else 0.0
        print(f"  Max AUROC: {max_auroc:.3f}")

    return real_results


# =============================================================================
# Random Split Classification
# =============================================================================

def create_random_split(homo_embryo_ids, cluster_sizes, random_state):
    """
    Randomly split homozygous embryos into pseudo-clusters matching real sizes.
    
    Parameters
    ----------
    homo_embryo_ids : list
        All homozygous embryo IDs
    cluster_sizes : dict
        Real cluster sizes {cluster_name: n_embryos}
    random_state : int
        Random seed
        
    Returns
    -------
    dict : {embryo_id: pseudo_cluster_name}
    """
    rng = np.random.RandomState(random_state)
    shuffled_ids = rng.permutation(homo_embryo_ids)
    
    assignments = {}
    idx = 0
    for i, (cluster_name, size) in enumerate(cluster_sizes.items()):
        for eid in shuffled_ids[idx:idx+size]:
            assignments[eid] = f"random_group_{i}"
        idx += size
    
    return assignments


def run_random_split_classification(df, homo_embryo_ids, cluster_sizes, split_id):
    """Run classification on one random split."""
    # Create random assignments
    random_assignments = create_random_split(
        homo_embryo_ids, cluster_sizes, random_state=RANDOM_SEED + split_id
    )
    
    df_split = df.copy()
    df_split['random_cluster'] = df_split['embryo_id'].map(random_assignments)
    
    split_results = {}
    
    for group_i in range(len(cluster_sizes)):
        group_name = f"random_group_{group_i}"
        
        # Create subset
        df_subset = df_split[
            (df_split['random_cluster'] == group_name) | (df_split['genotype'] == 'cep290_wildtype')
        ].copy()
        
        df_subset['comparison_group'] = df_subset.apply(
            lambda r: 'random' if r['random_cluster'] == group_name else 'WT', axis=1
        )
        
        results = compare_groups(
            df_subset,
            group_col='comparison_group',
            group1='random',
            group2='WT',
            features='z_mu_b',
            morphology_metric=METRIC_COL,
            bin_width=TIME_BIN_WIDTH,
            n_permutations=N_PERMUTATIONS_CLASS,
            n_jobs=N_JOBS,
            min_samples_per_bin=MIN_SAMPLES_PER_BIN,
            random_state=RANDOM_SEED + split_id,
            verbose=False  # Suppress output for multiple runs
        )
        
        split_results[group_name] = results
    
    return split_results


def run_all_random_splits(df, homo_embryo_ids, cluster_sizes, n_splits):
    """Run classification on N random splits."""
    print("\n" + "="*70)
    print(f"STEP 2: Random Split Control ({n_splits} iterations)")
    print("="*70)
    print(f"\nCluster sizes to match: {cluster_sizes}")
    print(f"Total homozygous embryos: {len(homo_embryo_ids)}")
    
    all_random_results = []
    all_timeseries_results = []  # NEW: store per-time-bin results
    
    for split_i in range(n_splits):
        if (split_i + 1) % 5 == 0:
            print(f"  Running split {split_i + 1}/{n_splits}...")
        
        split_results = run_random_split_classification(
            df, homo_embryo_ids, cluster_sizes, split_i
        )
        
        # Extract max AUROC for each group
        split_aurocs = {}
        for group_name, results in split_results.items():
            summary = results['summary']
            split_aurocs[group_name] = summary['max_auroc'] if summary['max_auroc'] is not None else 0.5
            
            # NEW: Extract per-time-bin AUROC
            df_class = results['classification'].copy()
            df_class['split_id'] = split_i
            df_class['random_group'] = group_name
            all_timeseries_results.append(df_class)
        
        split_aurocs['split_id'] = split_i
        all_random_results.append(split_aurocs)
    
    df_random = pd.DataFrame(all_random_results)
    df_random_timeseries = pd.concat(all_timeseries_results, ignore_index=True)
    
    print(f"\n✓ Completed {n_splits} random splits")
    print(f"\nRandom split AUROC summary:")
    print(df_random[[c for c in df_random.columns if 'random_group' in c]].describe())
    
    return df_random, df_random_timeseries


# =============================================================================
# Comparison and Visualization
# =============================================================================

def compare_real_vs_random(real_results, df_random, cluster_sizes):
    """Compare real cluster AUROC to random split distribution."""
    print("\n" + "="*70)
    print("STEP 3: Real vs Random Comparison")
    print("="*70)
    
    comparison_results = []
    
    for i, cluster_name in enumerate(CLUSTERS_TO_VALIDATE):
        if cluster_name not in real_results:
            continue
        
        # Real cluster max AUROC
        real_auroc = real_results[cluster_name]['summary']['max_auroc']
        if real_auroc is None:
            real_auroc = 0.5
        
        # Random split distribution for this group size
        group_col = f"random_group_{i}"
        random_aurocs = df_random[group_col].values
        
        # Compute percentile
        percentile = (random_aurocs < real_auroc).mean() * 100
        
        comparison_results.append({
            'cluster': cluster_name,
            'real_auroc': real_auroc,
            'random_mean': random_aurocs.mean(),
            'random_std': random_aurocs.std(),
            'percentile': percentile,
            'n_embryos': cluster_sizes[cluster_name],
        })
        
        print(f"\n{cluster_name} (n={cluster_sizes[cluster_name]}):")
        print(f"  Real AUROC: {real_auroc:.3f}")
        print(f"  Random mean: {random_aurocs.mean():.3f} ± {random_aurocs.std():.3f}")
        print(f"  Percentile: {percentile:.1f}% (real > random)")
        if percentile > 95:
            print(f"  ✓ SIGNIFICANT: Real cluster outperforms random (p < 0.05)")
        elif percentile > 90:
            print(f"  ✓ TRENDING: Real cluster outperforms random (p < 0.10)")
        else:
            print(f"  ✗ NOT SIGNIFICANT: Real cluster similar to random")
    
    df_comparison = pd.DataFrame(comparison_results)
    return df_comparison


def plot_validation_results(real_results, df_random, df_comparison):
    """Create validation plots."""
    
    # Plot 1: Box plot comparison
    fig, ax = plt.subplots(figsize=(12, 7))
    
    positions = []
    box_data = []
    colors = []
    labels = []
    
    for i, cluster_name in enumerate(CLUSTERS_TO_VALIDATE):
        # Random distribution
        group_col = f"random_group_{i}"
        random_aurocs = df_random[group_col].values
        
        positions.append(i * 2)
        box_data.append(random_aurocs)
        colors.append(COLORS_RANDOM[group_col])
        labels.append(f"{cluster_name}\n(random)")
        
        # Real value
        real_auroc = real_results[cluster_name]['summary']['max_auroc']
        if real_auroc is None:
            real_auroc = 0.5
        
        positions.append(i * 2 + 0.8)
        box_data.append([real_auroc])  # Single value
        colors.append(COLORS_REAL[cluster_name])
        labels.append(f"{cluster_name}\n(real)")
    
    bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                    showfliers=True, notch=False)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=9)
    ax.set_ylabel('Max AUROC', fontsize=12)
    ax.set_title('Real Clusters vs Random Splits: AUROC Validation\n(Box = random distribution, Single point = real cluster)',
                fontsize=13, fontweight='bold')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Chance')
    ax.set_ylim(0.3, 1.0)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    return fig


def plot_detailed_comparison(df_comparison):
    """Create detailed comparison plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: AUROC comparison
    x = np.arange(len(df_comparison))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, df_comparison['real_auroc'], width,
                    label='Real Cluster', color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, df_comparison['random_mean'], width,
                    label='Random Mean', color='gray', alpha=0.6)
    
    # Add error bars for random
    ax1.errorbar(x + width/2, df_comparison['random_mean'],
                yerr=df_comparison['random_std'], fmt='none', color='black',
                capsize=5, alpha=0.5)
    
    ax1.set_ylabel('Max AUROC', fontsize=12)
    ax1.set_title('AUROC: Real vs Random', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_comparison['cluster'])
    ax1.legend()
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax1.set_ylim(0.3, 1.0)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Percentile ranking
    bars = ax2.bar(x, df_comparison['percentile'], color='green', alpha=0.7)
    ax2.axhline(y=95, color='red', linestyle='--', linewidth=2, label='p<0.05 threshold')
    ax2.axhline(y=90, color='orange', linestyle='--', linewidth=1.5, label='p<0.10 threshold')
    
    # Color bars by significance
    for i, (bar, pct) in enumerate(zip(bars, df_comparison['percentile'])):
        if pct > 95:
            bar.set_color('darkgreen')
        elif pct > 90:
            bar.set_color('orange')
        else:
            bar.set_color('gray')
    
    ax2.set_ylabel('Percentile (%)', fontsize=12)
    ax2.set_title('Real Cluster Performance Percentile\n(vs random splits)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_comparison['cluster'])
    ax2.legend(loc='lower right')
    ax2.set_ylim(0, 105)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_temporal_comparison(real_results, df_random_timeseries, cluster_sizes):
    """
    Plot AUROC over time: real clusters vs random split mean ± std.
    
    Parameters
    ----------
    real_results : dict
        Real cluster classification results
    df_random_timeseries : pd.DataFrame
        Per-time-bin AUROC for all random splits
    cluster_sizes : dict
        Cluster sizes for matching
    """
    n_clusters = len(CLUSTERS_TO_VALIDATE)
    fig, axes = plt.subplots(1, n_clusters, figsize=(6*n_clusters, 5), sharey=True)
    if n_clusters == 1:
        axes = [axes]
    
    for i, (ax, cluster_name) in enumerate(zip(axes, CLUSTERS_TO_VALIDATE)):
        if cluster_name not in real_results:
            continue
        
        # Real cluster AUROC over time
        df_real = real_results[cluster_name]['classification']
        real_color = COLORS_REAL[cluster_name]
        
        # Random splits AUROC over time (for matching group size)
        random_group = f"random_group_{i}"
        df_random_subset = df_random_timeseries[
            df_random_timeseries['random_group'] == random_group
        ].copy()
        
        # Compute mean and std per time bin
        random_stats = df_random_subset.groupby('time_bin')['auroc_observed'].agg(['mean', 'std', 'count']).reset_index()
        
        # Plot random mean with shaded std band
        ax.plot(random_stats['time_bin'], random_stats['mean'],
               '--', color='gray', linewidth=2, alpha=0.7, label='Random Mean')
        ax.fill_between(random_stats['time_bin'],
                        random_stats['mean'] - random_stats['std'],
                        random_stats['mean'] + random_stats['std'],
                        color='gray', alpha=0.2, label='Random ±1 SD')
        
        # Plot real cluster
        ax.plot(df_real['time_bin'], df_real['auroc_observed'],
               'o-', color=real_color, linewidth=2.5, markersize=6,
               label=f'{cluster_name} (real)', zorder=10)
        
        # Mark significant points in real
        sig_mask = df_real['pval'] < 0.05
        if sig_mask.any():
            ax.scatter(df_real.loc[sig_mask, 'time_bin'],
                      df_real.loc[sig_mask, 'auroc_observed'],
                      s=200, facecolors='none', edgecolors=real_color,
                      linewidths=2.5, zorder=11)
        
        # Formatting
        ax.axhline(y=0.5, color='black', linestyle=':', alpha=0.3)
        ax.set_xlabel('Hours Post Fertilization (hpf)', fontsize=11)
        ax.set_title(f'{cluster_name} (n={cluster_sizes[cluster_name]})\nvs Random Splits',
                    fontsize=12, fontweight='bold')
        ax.set_ylim(0.3, 1.05)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    axes[0].set_ylabel('AUROC', fontsize=12)
    fig.suptitle('Real Clusters vs Random Splits: AUROC Over Time\n(Shaded band = mean ± 1 SD of random splits)',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*70)
    print("CEP290 CLUSTER VALIDATION VIA RANDOM SPLIT CONTROL")
    print("="*70)
    print("\nObjective: Validate that per-cluster AUROC is not a sample size artifact")
    print(f"Method: {N_RANDOM_SPLITS} random splits of homozygous embryos")
    
    # Load data
    df, df_homo, homo_embryo_ids = load_data()
    
    # Get real cluster sizes
    cluster_sizes = {}
    for cluster in CLUSTERS_TO_VALIDATE:
        cluster_sizes[cluster] = df_homo[df_homo['real_cluster'] == cluster]['embryo_id'].nunique()
    
    # Step 1: Real cluster classification
    real_results = run_real_cluster_classification(df)
    
    # Step 2: Random split classification
    df_random, df_random_timeseries = run_all_random_splits(df, homo_embryo_ids, cluster_sizes, N_RANDOM_SPLITS)
    
    # Step 3: Compare
    df_comparison = compare_real_vs_random(real_results, df_random, cluster_sizes)
    
    # Save results
    csv_path = OUTPUT_DIR / "validation_comparison.csv"
    df_comparison.to_csv(csv_path, index=False)
    print(f"\n✓ Saved comparison to: {csv_path}")
    
    random_csv_path = OUTPUT_DIR / "random_splits_auroc.csv"
    df_random.to_csv(random_csv_path, index=False)
    print(f"✓ Saved random splits to: {random_csv_path}")
    
    # Save timeseries data
    timeseries_csv_path = OUTPUT_DIR / "random_splits_timeseries.csv"
    df_random_timeseries.to_csv(timeseries_csv_path, index=False)
    print(f"✓ Saved random splits timeseries to: {timeseries_csv_path}")
    
    # Plot 1: Box plot (max AUROC)
    fig1 = plot_validation_results(real_results, df_random, df_comparison)
    plot1_path = OUTPUT_DIR / "validation_boxplot.png"
    fig1.savefig(plot1_path, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"✓ Saved validation boxplot to: {plot1_path}")
    
    # Plot 2: Detailed comparison
    fig2 = plot_detailed_comparison(df_comparison)
    plot2_path = OUTPUT_DIR / "validation_detailed.png"
    fig2.savefig(plot2_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"✓ Saved detailed comparison to: {plot2_path}")
    
    # Plot 3: Temporal comparison (NEW!)
    fig3 = plot_temporal_comparison(real_results, df_random_timeseries, cluster_sizes)
    plot3_path = OUTPUT_DIR / "validation_temporal.png"
    fig3.savefig(plot3_path, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"✓ Saved temporal comparison to: {plot3_path}")
    
    # Final summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print("\nInterpretation Guide:")
    print("  - Percentile > 95% → Real cluster significantly better than random (p < 0.05) ✓")
    print("  - Percentile 90-95% → Trending significance (p < 0.10)")
    print("  - Percentile < 90% → No evidence that real clustering improves classification")
    
    print("\nResults:")
    for _, row in df_comparison.iterrows():
        status = "✓ VALIDATED" if row['percentile'] > 95 else ("~ TRENDING" if row['percentile'] > 90 else "✗ NOT SIGNIFICANT")
        print(f"  {row['cluster']}: {row['real_auroc']:.3f} real vs {row['random_mean']:.3f}±{row['random_std']:.3f} random → {status}")
    
    print("\n⭐ NEW: Check validation_temporal.png for AUROC trajectories over time")
    print("  - Shows mean ± SD of random splits at each time bin")
    print("  - Compare temporal dynamics, not just peak values")
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
