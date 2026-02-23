#!/usr/bin/env python
"""
CEP290 Phenotype Statistical Analysis - 1-Hour Bins

PRIORITY 2: Time Resolution Sensitivity Test

This is identical to cep290_statistical_analysis.py but uses:
- 1-hour time bins instead of 2-hour bins
- Tests if finer temporal resolution reveals additional dynamics

Compare results to 2-hour version to check if binning hides important patterns.

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
# Configuration - UPDATED FOR 1-HOUR BINS
# =============================================================================

OUTPUT_DIR = Path(__file__).parent / "results_1hr_bins"
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251229_cep290_phenotype_extraction")

CLUSTERING_DATA_PATH = DATA_DIR / "data" / "clustering_data__early_homo.pkl"
K_RESULTS_PATH = DATA_DIR / "kmedoids_k_selection_early_timepoints_cep290_data" / "k_results.pkl"

# Analysis parameters - CHANGED: 1-hour bins instead of 2-hour
METRIC_COL = 'baseline_deviation_normalized'
METRIC_LABEL = 'Baseline Deviation (normalized)'
K_VALUE = 5
TIME_BIN_WIDTH = 1.0  # ⭐ CHANGED from 2.0 to 1.0 hours
N_PERMUTATIONS_CLASS = 100  # for classification (increase for stable p-values)
N_JOBS = -1  # parallel jobs for permutation testing (-1 = all cores, 1 = serial)
MIN_SAMPLES_PER_BIN = 3

# Cluster definitions (excluding outlier as per user request)
CLUSTER_NAMES_K5 = {
    0: 'outlier',
    1: 'bumpy',
    2: 'low_to_high',
    3: 'low_to_high',  # merged with cluster 2
    4: 'high_to_low',
}

CLUSTERS_TO_ANALYZE = ['bumpy', 'high_to_low', 'low_to_high']

# Colors
COLORS = {
    'cep290_wildtype': '#2E7D32',
    'cep290_heterozygous': '#FFA500',
    'cep290_homozygous': '#D32F2F',
    'Het_vs_WT': '#888888',
    'bumpy': '#9467BD',
    'low_to_high': '#17BECF',
    'high_to_low': '#E377C2',
    'bumpy_vs_WT': '#9467BD',
    'high_to_low_vs_WT': '#E377C2',
    'low_to_high_vs_WT': '#17BECF',
    'Homo_vs_WT': '#D32F2F',
    'Homo_vs_Het': '#9467BD',
}

# =============================================================================
# Data Loading
# =============================================================================

def load_data():
    """Load clustering data and k-results."""
    print("Loading data...")

    with open(CLUSTERING_DATA_PATH, 'rb') as f:
        clustering_data = pickle.load(f)

    with open(K_RESULTS_PATH, 'rb') as f:
        k_results = pickle.load(f)

    df = clustering_data['df_cep290_earyltimepoints'].copy()
    print(f"  Loaded {len(df)} rows, {df['embryo_id'].nunique()} embryos")

    # Add cluster assignments for homozygous embryos
    cluster_labels = k_results['clustering_by_k'][K_VALUE]['assignments']['cluster_labels']
    embryo_ids_clustered = k_results['embryo_ids']

    embryo_to_cluster = dict(zip(embryo_ids_clustered, cluster_labels))
    embryo_to_cluster_name = {
        eid: CLUSTER_NAMES_K5[cid] for eid, cid in embryo_to_cluster.items()
    }

    df['cluster'] = df['embryo_id'].map(embryo_to_cluster_name)

    # Print genotype counts
    print("\nGenotype distribution:")
    for geno, count in df.groupby('genotype')['embryo_id'].nunique().items():
        print(f"  {geno}: {count} embryos")

    print("\nCluster distribution (homozygous only):")
    df_homo = df[df['genotype'] == 'cep290_homozygous']
    for cluster, count in df_homo.groupby('cluster')['embryo_id'].nunique().items():
        print(f"  {cluster}: {count} embryos")

    return df, k_results


# =============================================================================
# Part 1: Per-Cluster Classification
# =============================================================================

def run_cluster_classification(df):
    """Run AUROC-based classification for each cluster vs WT."""
    print("\n" + "="*60)
    print("PART 1: Per-Cluster Classification (1-hour bins)")
    print("="*60)

    all_results = {}

    # First, run Het vs WT as baseline
    print("\n--- Het_vs_WT (Baseline) ---")
    results_baseline = compare_groups(
        df,
        group_col='genotype',
        group1='cep290_heterozygous',
        group2='cep290_wildtype',
        features='z_mu_b',
        morphology_metric=METRIC_COL,
        bin_width=TIME_BIN_WIDTH,
        n_permutations=N_PERMUTATIONS_CLASS,
        n_jobs=N_JOBS,
        min_samples_per_bin=MIN_SAMPLES_PER_BIN,
        random_state=42,
        verbose=True
    )
    all_results['Het_vs_WT'] = results_baseline
    summary = results_baseline['summary']
    max_auroc = summary['max_auroc'] if summary['max_auroc'] is not None else 0.0
    print(f"\n  Summary: Max AUROC={max_auroc:.3f}, Earliest sig={summary['earliest_significant_hpf']}")

    # Now run each cluster vs WT
    for cluster in CLUSTERS_TO_ANALYZE:
        label = f"{cluster}_vs_WT"
        print(f"\n--- {label} ---")

        # Create subset: this cluster + all WT
        df_subset = df[
            (df['cluster'] == cluster) | (df['genotype'] == 'cep290_wildtype')
        ].copy()

        # Create comparison group column
        df_subset['comparison_group'] = df_subset.apply(
            lambda r: cluster if r['cluster'] == cluster else 'WT', axis=1
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
            random_state=42,
            verbose=True
        )

        all_results[label] = results

        summary = results['summary']
        max_auroc = summary['max_auroc'] if summary['max_auroc'] is not None else 0.0
        print(f"\n  Summary: Max AUROC={max_auroc:.3f}, Earliest sig={summary['earliest_significant_hpf']}")

    return all_results


# =============================================================================
# Part 2: Pooled Classification
# =============================================================================

def run_pooled_classification(df):
    """Run AUROC-based classification comparing pooled groups."""
    print("\n" + "="*60)
    print("PART 2: Pooled Classification (1-hour bins)")
    print("="*60)

    comparisons = [
        ('cep290_homozygous', 'cep290_wildtype', 'Homo_vs_WT'),
        ('cep290_homozygous', 'cep290_heterozygous', 'Homo_vs_Het'),
        ('cep290_heterozygous', 'cep290_wildtype', 'Het_vs_WT'),
    ]

    all_results = {}

    for group1, group2, label in comparisons:
        print(f"\n--- {label}: {group1} vs {group2} ---")

        results = compare_groups(
            df,
            group_col='genotype',
            group1=group1,
            group2=group2,
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

        summary = results['summary']
        max_auroc = summary['max_auroc'] if summary['max_auroc'] is not None else 0.0
        max_auroc_hpf = summary['max_auroc_hpf'] if summary['max_auroc_hpf'] is not None else 'N/A'
        print(f"\n  Summary:")
        print(f"    Earliest significant: {summary['earliest_significant_hpf']} hpf")
        print(f"    Max AUROC: {max_auroc:.3f} at {max_auroc_hpf} hpf")
        print(f"    Significant bins: {summary['n_significant_bins']}")

    return all_results


# =============================================================================
# Plotting Functions (identical to 2-hour version)
# =============================================================================

def plot_cluster_vs_wt_auroc(cluster_results):
    """Plot AUROC for each cluster vs WT with p-value annotations."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot Het vs WT baseline first (dashed gray)
    if 'Het_vs_WT' in cluster_results:
        df_baseline = cluster_results['Het_vs_WT']['classification']
        ax.plot(df_baseline['time_bin'], df_baseline['auroc_observed'],
                '--', color='#888888', linewidth=2, alpha=0.7, label='Het vs WT (baseline)')
        if 'auroc_null_mean' in df_baseline.columns and 'auroc_null_std' in df_baseline.columns:
            ax.fill_between(
                df_baseline['time_bin'],
                df_baseline['auroc_null_mean'] - df_baseline['auroc_null_std'],
                df_baseline['auroc_null_mean'] + df_baseline['auroc_null_std'],
                color='#888888',
                alpha=0.10,
                linewidth=0
            )

    # Plot each cluster vs WT
    for cluster in CLUSTERS_TO_ANALYZE:
        label = f"{cluster}_vs_WT"
        if label not in cluster_results:
            continue

        results = cluster_results[label]
        df_class = results['classification']
        color = COLORS[label]

        # Plot AUROC line
        line, = ax.plot(df_class['time_bin'], df_class['auroc_observed'],
                       'o-', color=color, linewidth=2, markersize=5,
                       label=f"{cluster} vs WT")

        # Null distribution band (mean ± 1 SD)
        if 'auroc_null_mean' in df_class.columns and 'auroc_null_std' in df_class.columns:
            ax.fill_between(
                df_class['time_bin'],
                df_class['auroc_null_mean'] - df_class['auroc_null_std'],
                df_class['auroc_null_mean'] + df_class['auroc_null_std'],
                color=color,
                alpha=0.10,
                linewidth=0
            )

        # Add significance markers
        sig_mask = df_class['pval'] < 0.05
        if sig_mask.any():
            ax.scatter(df_class.loc[sig_mask, 'time_bin'],
                      df_class.loc[sig_mask, 'auroc_observed'],
                      s=200, facecolors='none', edgecolors=color, linewidths=2.5,
                      zorder=5)

        # Add stars for highly significant
        very_sig_mask = df_class['pval'] < 0.01
        if very_sig_mask.any():
            for _, row in df_class[very_sig_mask].iterrows():
                ax.annotate('*', (row['time_bin'], row['auroc_observed'] + 0.03),
                           ha='center', fontsize=14, fontweight='bold', color=color)

    # Reference line at 0.5 (chance)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Chance (0.5)')

    ax.set_xlabel('Hours Post Fertilization (hpf)', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title('Per-Cluster Classification vs WT\n(VAE Latent Features, 1-hour bins, shaded = null mean ± 1 SD, circles = p<0.05, * = p<0.01)',
                fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(0.3, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_pooled_auroc_with_pvalues(all_results):
    """Plot AUROC over time for pooled comparisons."""
    fig, ax = plt.subplots(figsize=(14, 7))

    comparison_colors = {
        'Homo_vs_WT': '#D32F2F',
        'Homo_vs_Het': '#9467BD',
        'Het_vs_WT': '#888888',
    }

    comparison_styles = {
        'Homo_vs_WT': '-',
        'Homo_vs_Het': '-',
        'Het_vs_WT': '--',
    }

    for label, results in all_results.items():
        df_class = results['classification']
        if df_class.empty:
            continue

        color = comparison_colors.get(label, '#000000')
        style = comparison_styles.get(label, '-')

        # Plot AUROC line
        ax.plot(df_class['time_bin'], df_class['auroc_observed'],
                f'o{style}', label=label, color=color, linewidth=2, markersize=5)

        # Null distribution band (mean ± 1 SD)
        if 'auroc_null_mean' in df_class.columns and 'auroc_null_std' in df_class.columns:
            ax.fill_between(
                df_class['time_bin'],
                df_class['auroc_null_mean'] - df_class['auroc_null_std'],
                df_class['auroc_null_mean'] + df_class['auroc_null_std'],
                color=color,
                alpha=0.10,
                linewidth=0
            )

        # Add significance markers
        sig_mask = df_class['pval'] < 0.05
        if sig_mask.any():
            ax.scatter(df_class.loc[sig_mask, 'time_bin'],
                      df_class.loc[sig_mask, 'auroc_observed'],
                      s=200, facecolors='none', edgecolors=color, linewidths=2.5,
                      zorder=5)

        # Add stars for highly significant
        very_sig_mask = df_class['pval'] < 0.01
        if very_sig_mask.any():
            for _, row in df_class[very_sig_mask].iterrows():
                ax.annotate('*', (row['time_bin'], row['auroc_observed'] + 0.03),
                           ha='center', fontsize=14, fontweight='bold', color=color)

    # Reference line at 0.5 (chance)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Chance (0.5)')

    ax.set_xlabel('Hours Post Fertilization (hpf)', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title('Pooled Classification Performance Over Time\n(VAE Latent Features, 1-hour bins, shaded = null mean ± 1 SD, circles = p<0.05, * = p<0.01)',
                fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(0.3, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_temporal_emergence(all_results, title_prefix=""):
    """Plot when differences emerge with significance markers."""
    n_comparisons = len(all_results)
    fig, axes = plt.subplots(1, n_comparisons, figsize=(5*n_comparisons, 5), sharey=True)
    if n_comparisons == 1:
        axes = [axes]

    for ax, (label, results) in zip(axes, all_results.items()):
        df_class = results['classification']
        summary = results['summary']
        color = COLORS.get(label, '#666666')

        if df_class.empty:
            ax.set_title(f"{label}\n(No data)")
            continue

        # Bar plot of AUROC per time bin
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

        # Reference line
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

        # Mark earliest significant
        if summary['earliest_significant_hpf'] is not None:
            ax.axvline(x=summary['earliest_significant_hpf'],
                      color='green', linestyle='-', alpha=0.7, linewidth=2)
            ax.annotate(f"First sig:\n{summary['earliest_significant_hpf']}h",
                       xy=(summary['earliest_significant_hpf'], 0.95),
                       ha='center', fontsize=9, color='green')

        ax.set_xlabel('Hours Post Fertilization')
        ax.set_title(f"{label}\nMax AUROC: {summary['max_auroc']:.2f}")
        ax.set_ylim(0.3, 1.05)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].set_ylabel('AUROC')
    fig.suptitle(f'{title_prefix}Temporal Emergence of Phenotypic Differences (1-hour bins)', fontsize=14, y=1.02)

    plt.tight_layout()
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*70)
    print("CEP290 Statistical Analysis - 1-HOUR BINS (Priority 2)")
    print("="*70)
    print(f"\n⭐ CHANGED: Using {TIME_BIN_WIDTH}-hour bins (was 2.0 hours)")
    print("Testing if finer temporal resolution reveals additional dynamics\n")

    # Load data
    df, k_results = load_data()

    # Part 1: Per-cluster classification
    cluster_results = run_cluster_classification(df)

    # Part 2: Pooled classification
    pooled_results = run_pooled_classification(df)

    # Combine all results
    all_results = {**cluster_results, **pooled_results}

    # Save CSV
    classification_records = []
    for label, results in all_results.items():
        df_class = results['classification'].copy()
        df_class['comparison'] = label
        classification_records.append(df_class)

    df_classification = pd.concat(classification_records, ignore_index=True)
    class_csv_path = OUTPUT_DIR / "classification_results_1hr.csv"
    df_classification.to_csv(class_csv_path, index=False)
    print(f"\n✓ Saved classification results to: {class_csv_path}")

    # Plot 1: Per-cluster AUROC
    fig_cluster = plot_cluster_vs_wt_auroc(cluster_results)
    cluster_plot_path = OUTPUT_DIR / "cep290_cluster_vs_wt_auroc_1hr.png"
    fig_cluster.savefig(cluster_plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig_cluster)
    print(f"✓ Saved cluster AUROC plot (1hr): {cluster_plot_path}")

    # Plot 2: Pooled AUROC
    fig_pooled = plot_pooled_auroc_with_pvalues(pooled_results)
    pooled_plot_path = OUTPUT_DIR / "cep290_classification_auroc_1hr.png"
    fig_pooled.savefig(pooled_plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig_pooled)
    print(f"✓ Saved pooled AUROC plot (1hr): {pooled_plot_path}")

    # Plot 3: Temporal emergence for clusters
    fig_temporal_cluster = plot_temporal_emergence(cluster_results, "Per-Cluster: ")
    temporal_cluster_path = OUTPUT_DIR / "cep290_temporal_emergence_clusters_1hr.png"
    fig_temporal_cluster.savefig(temporal_cluster_path, dpi=150, bbox_inches='tight')
    plt.close(fig_temporal_cluster)
    print(f"✓ Saved cluster temporal emergence (1hr): {temporal_cluster_path}")

    # Plot 4: Temporal emergence for pooled
    fig_temporal_pooled = plot_temporal_emergence(pooled_results, "Pooled: ")
    temporal_pooled_path = OUTPUT_DIR / "cep290_temporal_emergence_pooled_1hr.png"
    fig_temporal_pooled.savefig(temporal_pooled_path, dpi=150, bbox_inches='tight')
    plt.close(fig_temporal_pooled)
    print(f"✓ Saved pooled temporal emergence (1hr): {temporal_pooled_path}")

    # Plot 5-7: Comprehensive figures for each cluster vs WT (1hr bins)
    print("\n" + "="*70)
    print("GENERATING COMPREHENSIVE FIGURES (Cluster vs WT, 1-hour bins)")
    print("="*70)
    
    for cluster in CLUSTERS_TO_ANALYZE:
        label = f"{cluster}_vs_WT"
        if label not in cluster_results:
            continue
            
        print(f"\n  Creating comprehensive figure for {label}...")
        
        # Get embryo IDs for cluster and WT
        cluster_ids = df[df['cluster'] == cluster]['embryo_id'].unique().tolist()
        wt_ids = df[df['genotype'] == 'wildtype']['embryo_id'].unique().tolist()
        
        comprehensive_path = OUTPUT_DIR / f"{label}_comprehensive_1hr.png"
        fig, divergence_df = create_full_comparison(
            df=df,
            df_results=cluster_results[label]['classification'],
            group1_ids=cluster_ids,
            group2_ids=wt_ids,
            group1_label=cluster,
            group2_label='WT',
            metric_col=METRIC_COL,
            metric_label=METRIC_LABEL,
            save_path=comprehensive_path,
            time_col='predicted_stage_hpf',
            embryo_id_col='embryo_id'
        )
        
        if fig is not None:
            plt.close(fig)  # Prevent memory buildup

    # Summary
    print("\n" + "="*70)
    print("SUMMARY (1-hour bins)")
    print("="*70)

    print("\nPer-Cluster Classification (vs WT):")
    for cluster in CLUSTERS_TO_ANALYZE:
        label = f"{cluster}_vs_WT"
        if label in cluster_results:
            s = cluster_results[label]['summary']
            max_auroc = s['max_auroc'] if s['max_auroc'] is not None else 0.0
            max_hpf = s['max_auroc_hpf'] if s['max_auroc_hpf'] is not None else 'N/A'
            print(f"  {label}:")
            print(f"    Max AUROC: {max_auroc:.3f} at {max_hpf} hpf")
            print(f"    Earliest significant: {s['earliest_significant_hpf']} hpf")

    print("\nPooled Classification:")
    for label, results in pooled_results.items():
        s = results['summary']
        max_auroc = s['max_auroc'] if s['max_auroc'] is not None else 0.0
        max_hpf = s['max_auroc_hpf'] if s['max_auroc_hpf'] is not None else 'N/A'
        print(f"  {label}:")
        print(f"    Max AUROC: {max_auroc:.3f} at {max_hpf} hpf")
        print(f"    Earliest significant: {s['earliest_significant_hpf']} hpf")

    print("\n⭐ TO COMPARE WITH 2-HOUR BINS:")
    print("  - Load quick_results/classification_results.csv (2hr)")
    print(f"  - Load {class_csv_path} (1hr)")
    print("  - Check if earliest_significant_hpf or max_auroc changed")
    print("  - Look for new significant time bins in 1hr version")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
