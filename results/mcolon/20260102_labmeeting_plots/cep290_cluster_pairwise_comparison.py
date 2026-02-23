#!/usr/bin/env python
"""
CEP290 Cluster Pairwise Comparison

Simpler validation: Compare the 3 phenotype clusters AGAINST EACH OTHER
(not vs WT, which they're all different from).

Tests:
- bumpy vs high_to_low
- bumpy vs low_to_high  
- high_to_low vs low_to_high

If clusters are truly distinct phenotypes → High AUROC (>0.7)
If clusters are random subdivisions → Low AUROC (~0.5)

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
from comparison_plotting_utils import create_full_comparison

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = Path(__file__).parent / "cluster_pairwise_results"
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251229_cep290_phenotype_extraction")

CLUSTERING_DATA_PATH = DATA_DIR / "data" / "clustering_data__early_homo.pkl"
K_RESULTS_PATH = DATA_DIR / "kmedoids_k_selection_early_timepoints_cep290_data" / "k_results.pkl"

# Analysis parameters
METRIC_COL = 'baseline_deviation_normalized'
METRIC_LABEL = 'Baseline Deviation (normalized)'
K_VALUE = 5
TIME_BIN_WIDTH = 2.0  # hours
N_PERMUTATIONS_CLASS = 100  # permutations per classification (increase for stable p-values)
N_JOBS = -1  # parallel jobs for permutation testing (-1 = all cores, 1 = serial)
MIN_SAMPLES_PER_BIN = 3

# Cluster definitions
CLUSTER_NAMES_K5 = {
    0: 'outlier',
    1: 'bumpy',
    2: 'low_to_high',
    3: 'low_to_high',
    4: 'high_to_low',
}

CLUSTERS = ['bumpy', 'high_to_low', 'low_to_high']

# Colors
COLORS = {
    'bumpy': '#9467BD',
    'low_to_high': '#17BECF',
    'high_to_low': '#E377C2',
    'bumpy_vs_high_to_low': '#9467BD',
    'bumpy_vs_low_to_high': '#9467BD',
    'high_to_low_vs_low_to_high': '#E377C2',
}


# =============================================================================
# Data Loading
# =============================================================================

def load_data():
    """Load clustering data and assign clusters."""
    print("Loading data...")

    with open(CLUSTERING_DATA_PATH, 'rb') as f:
        clustering_data = pickle.load(f)

    with open(K_RESULTS_PATH, 'rb') as f:
        k_results = pickle.load(f)

    df = clustering_data['df_cep290_earyltimepoints'].copy()
    print(f"  Loaded {len(df)} rows, {df['embryo_id'].nunique()} embryos")

    # Add cluster assignments
    cluster_labels = k_results['clustering_by_k'][K_VALUE]['assignments']['cluster_labels']
    embryo_ids_clustered = k_results['embryo_ids']

    embryo_to_cluster = dict(zip(embryo_ids_clustered, cluster_labels))
    embryo_to_cluster_name = {
        eid: CLUSTER_NAMES_K5[cid] for eid, cid in embryo_to_cluster.items()
    }

    df['cluster'] = df['embryo_id'].map(embryo_to_cluster_name)

    # Keep only homozygous embryos with cluster assignments
    df_homo = df[(df['genotype'] == 'cep290_homozygous') & (df['cluster'].notna())].copy()

    print("\nCluster distribution (homozygous only):")
    for cluster in CLUSTERS:
        count = df_homo[df_homo['cluster'] == cluster]['embryo_id'].nunique()
        print(f"  {cluster}: {count} embryos")

    return df_homo


# =============================================================================
# Pairwise Comparisons
# =============================================================================

def run_pairwise_comparisons(df):
    """Run classification for all pairs of clusters."""
    print("\n" + "="*70)
    print("PAIRWISE CLUSTER COMPARISON (Homozygous Only)")
    print("="*70)
    print("\nTesting if phenotype clusters are distinguishable from each other")
    
    # Define all pairs
    pairs = [
        ('bumpy', 'high_to_low'),
        ('bumpy', 'low_to_high'),
        ('high_to_low', 'low_to_high'),
    ]
    
    all_results = {}
    
    for cluster1, cluster2 in pairs:
        label = f"{cluster1}_vs_{cluster2}"
        print(f"\n{'='*60}")
        print(f"Testing: {cluster1} vs {cluster2}")
        print(f"{'='*60}")
        
        # Create subset with only these two clusters
        df_subset = df[df['cluster'].isin([cluster1, cluster2])].copy()
        
        n1 = df_subset[df_subset['cluster'] == cluster1]['embryo_id'].nunique()
        n2 = df_subset[df_subset['cluster'] == cluster2]['embryo_id'].nunique()
        print(f"{cluster1}: {n1} embryos")
        print(f"{cluster2}: {n2} embryos")
        
        # Run comparison
        results = compare_groups(
            df_subset,
            group_col='cluster',
            group1=cluster1,
            group2=cluster2,
            features='z_mu_b',
            morphology_metric=METRIC_COL,
            bin_width=TIME_BIN_WIDTH,
            n_permutations=N_PERMUTATIONS_CLASS,
            n_jobs=N_JOBS,
            min_samples_per_bin=MIN_SAMPLES_PER_BIN,
            random_state=42,
            verbose=True
        )
        
        all_results[label] = results
        
        # Print summary
        summary = results['summary']
        max_auroc = summary['max_auroc'] if summary['max_auroc'] is not None else 0.5
        max_hpf = summary['max_auroc_hpf'] if summary['max_auroc_hpf'] is not None else 'N/A'
        earliest_sig = summary['earliest_significant_hpf']
        
        print(f"\nSummary:")
        print(f"  Max AUROC: {max_auroc:.3f} at {max_hpf} hpf")
        print(f"  Earliest significant: {earliest_sig} hpf")
        print(f"  Significant bins: {summary['n_significant_bins']}")
        
        # Interpretation
        if max_auroc > 0.8:
            print(f"  ✓ HIGHLY DISTINGUISHABLE: {cluster1} and {cluster2} are very different")
        elif max_auroc > 0.7:
            print(f"  ✓ DISTINGUISHABLE: {cluster1} and {cluster2} are somewhat different")
        elif max_auroc > 0.6:
            print(f"  ~ WEAKLY DISTINGUISHABLE: Small difference between {cluster1} and {cluster2}")
        else:
            print(f"  ✗ NOT DISTINGUISHABLE: {cluster1} and {cluster2} are similar")
    
    return all_results


# =============================================================================
# Visualization
# =============================================================================

def plot_pairwise_auroc(all_results):
    """Plot AUROC over time for all pairwise comparisons."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot each comparison
    for label, results in all_results.items():
        df_class = results['classification']
        if df_class.empty:
            continue
        
        color = COLORS.get(label, '#666666')
        
        # Extract cluster names for legend
        parts = label.split('_vs_')
        legend_label = f"{parts[0]} vs {parts[1]}"
        
        # Plot AUROC line
        ax.plot(df_class['time_bin'], df_class['auroc_observed'],
               'o-', color=color, linewidth=2, markersize=5,
               label=legend_label)
        
        # Mark significant points
        sig_mask = df_class['pval'] < 0.05
        if sig_mask.any():
            ax.scatter(df_class.loc[sig_mask, 'time_bin'],
                      df_class.loc[sig_mask, 'auroc_observed'],
                      s=200, facecolors='none', edgecolors=color,
                      linewidths=2.5, zorder=5)
        
        # Mark highly significant
        very_sig_mask = df_class['pval'] < 0.01
        if very_sig_mask.any():
            for _, row in df_class[very_sig_mask].iterrows():
                ax.annotate('*', (row['time_bin'], row['auroc_observed'] + 0.03),
                           ha='center', fontsize=14, fontweight='bold', color=color)
    
    # Reference lines
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Chance (0.5)')
    ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.3, label='Good separation (0.7)')
    ax.axhline(y=0.8, color='darkgreen', linestyle='--', alpha=0.3, label='Strong separation (0.8)')
    
    ax.set_xlabel('Hours Post Fertilization (hpf)', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title('Pairwise Cluster Classification (Homozygous Only)\n(VAE Latent Features, 2-hour bins, circles = p<0.05, * = p<0.01)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, ncol=2)
    ax.set_ylim(0.3, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_pairwise_heatmap(all_results):
    """Create heatmap of max AUROC between cluster pairs."""
    # Extract max AUROC for each pair
    pairs_matrix = np.ones((3, 3)) * 0.5  # Initialize with 0.5 (self vs self)
    cluster_order = CLUSTERS
    
    for label, results in all_results.items():
        parts = label.split('_vs_')
        c1, c2 = parts[0], parts[1]
        
        i = cluster_order.index(c1)
        j = cluster_order.index(c2)
        
        max_auroc = results['summary']['max_auroc']
        if max_auroc is None:
            max_auroc = 0.5
        
        pairs_matrix[i, j] = max_auroc
        pairs_matrix[j, i] = max_auroc  # Symmetric
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(pairs_matrix, cmap='RdYlGn', vmin=0.5, vmax=1.0)
    
    # Set ticks
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(cluster_order)
    ax.set_yticklabels(cluster_order)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(3):
        for j in range(3):
            text = ax.text(j, i, f'{pairs_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=12,
                          fontweight='bold')
    
    ax.set_title('Max AUROC: Cluster Pairwise Distinguishability\n(Higher = more distinct phenotypes)',
                fontsize=13, fontweight='bold')
    fig.colorbar(im, ax=ax, label='Max AUROC')
    
    plt.tight_layout()
    return fig


def plot_temporal_separation(all_results):
    """Create individual subplots for each pairwise comparison."""
    n_pairs = len(all_results)
    fig, axes = plt.subplots(1, n_pairs, figsize=(6*n_pairs, 5), sharey=True)
    if n_pairs == 1:
        axes = [axes]
    
    for ax, (label, results) in zip(axes, all_results.items()):
        df_class = results['classification']
        if df_class.empty:
            continue
        
        parts = label.split('_vs_')
        c1, c2 = parts[0], parts[1]
        color = COLORS.get(label, '#666666')
        
        # Bar plot
        bars = ax.bar(df_class['time_bin'], df_class['auroc_observed'],
                     width=TIME_BIN_WIDTH * 0.8, color=color, alpha=0.6)
        
        # Highlight significant bins
        for i, (idx, row) in enumerate(df_class.iterrows()):
            if row['pval'] < 0.05:
                bars[i].set_alpha(1.0)
                bars[i].set_edgecolor('black')
                bars[i].set_linewidth(2)
            if row['pval'] < 0.01:
                ax.annotate('*', (row['time_bin'], row['auroc_observed'] + 0.02),
                           ha='center', fontsize=12, fontweight='bold')
        
        # Reference lines
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.3)
        
        # Mark earliest significant
        earliest_sig = results['summary']['earliest_significant_hpf']
        if earliest_sig is not None:
            ax.axvline(x=earliest_sig, color='green', linestyle='-',
                      alpha=0.7, linewidth=2)
        
        ax.set_xlabel('Hours Post Fertilization (hpf)', fontsize=11)
        ax.set_title(f'{c1}\nvs\n{c2}',
                    fontsize=12, fontweight='bold')
        ax.set_ylim(0.3, 1.05)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3)
    
    axes[0].set_ylabel('AUROC', fontsize=12)
    fig.suptitle('Temporal Emergence of Phenotypic Differences Between Clusters',
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*70)
    print("CEP290 CLUSTER PAIRWISE COMPARISON")
    print("="*70)
    print("\nObjective: Test if phenotype clusters are distinguishable from each other")
    print("(not vs WT, which they're all different from)")
    
    # Load data
    df_homo = load_data()
    
    # Run pairwise comparisons
    all_results = run_pairwise_comparisons(df_homo)
    
    # Save classification results
    classification_records = []
    for label, results in all_results.items():
        df_class = results['classification'].copy()
        df_class['comparison'] = label
        classification_records.append(df_class)
    
    df_classification = pd.concat(classification_records, ignore_index=True)
    csv_path = OUTPUT_DIR / "pairwise_classification_results.csv"
    df_classification.to_csv(csv_path, index=False)
    print(f"\n✓ Saved classification results to: {csv_path}")
    
    # Plot 1: AUROC over time (all pairs)
    fig1 = plot_pairwise_auroc(all_results)
    plot1_path = OUTPUT_DIR / "pairwise_auroc_timeseries.png"
    fig1.savefig(plot1_path, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"✓ Saved pairwise AUROC plot to: {plot1_path}")
    
    # Plot 2: Heatmap
    fig2 = plot_pairwise_heatmap(all_results)
    plot2_path = OUTPUT_DIR / "pairwise_auroc_heatmap.png"
    fig2.savefig(plot2_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"✓ Saved heatmap to: {plot2_path}")
    
    # Plot 3: Temporal separation (subplots)
    fig3 = plot_temporal_separation(all_results)
    plot3_path = OUTPUT_DIR / "pairwise_temporal_separation.png"
    fig3.savefig(plot3_path, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"✓ Saved temporal separation to: {plot3_path}")
    
    # Plot 4-6: Comprehensive figures for each comparison
    print("\n" + "="*70)
    print("GENERATING COMPREHENSIVE FIGURES (Individual Comparisons)")
    print("="*70)
    
    pairs = [
        ('bumpy', 'high_to_low'),
        ('bumpy', 'low_to_high'),
        ('high_to_low', 'low_to_high'),
    ]
    
    for cluster1, cluster2 in pairs:
        label = f"{cluster1}_vs_{cluster2}"
        print(f"\n  Creating comprehensive figure for {label}...")
        
        # Get embryo IDs for each cluster
        cluster1_ids = df_homo[df_homo['cluster'] == cluster1]['embryo_id'].unique().tolist()
        cluster2_ids = df_homo[df_homo['cluster'] == cluster2]['embryo_id'].unique().tolist()
        
        comprehensive_path = OUTPUT_DIR / f"{label}_comprehensive.png"
        fig, divergence_df = create_full_comparison(
            df=df_homo,
            df_results=all_results[label]['classification'],
            group1_ids=cluster1_ids,
            group2_ids=cluster2_ids,
            group1_label=cluster1,
            group2_label=cluster2,
            metric_col=METRIC_COL,
            metric_label=METRIC_LABEL,
            save_path=comprehensive_path,
            time_col='predicted_stage_hpf',
            embryo_id_col='embryo_id',
            colors=COLORS
        )
        
        if fig is not None:
            plt.close(fig)  # Prevent memory buildup
    
    # Summary
    print("\n" + "="*70)
    print("PAIRWISE COMPARISON SUMMARY")
    print("="*70)
    
    for label, results in all_results.items():
        parts = label.split('_vs_')
        summary = results['summary']
        max_auroc = summary['max_auroc'] if summary['max_auroc'] is not None else 0.5
        max_hpf = summary['max_auroc_hpf'] if summary['max_auroc_hpf'] is not None else 'N/A'
        
        print(f"\n{parts[0]} vs {parts[1]}:")
        print(f"  Max AUROC: {max_auroc:.3f} at {max_hpf} hpf")
        print(f"  Earliest significant: {summary['earliest_significant_hpf']} hpf")
        
        if max_auroc > 0.8:
            print(f"  → STRONG SEPARATION ✓")
        elif max_auroc > 0.7:
            print(f"  → MODERATE SEPARATION ~")
        elif max_auroc > 0.6:
            print(f"  → WEAK SEPARATION")
        else:
            print(f"  → NO SEPARATION ✗")
    
    print("\n" + "="*70)
    print("Interpretation:")
    print("  - AUROC > 0.8: Clusters are highly distinct phenotypes")
    print("  - AUROC 0.7-0.8: Clusters are moderately distinct")
    print("  - AUROC < 0.7: Clusters may be subdivisions of same phenotype")
    print("="*70)


if __name__ == "__main__":
    main()
