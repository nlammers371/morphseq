#!/usr/bin/env python
"""
CEP290 Analysis Using Curvature Alone (No Embeddings)

Tests if raw curvature (baseline_deviation_normalized) is sufficient to:
1. Distinguish each cluster from WT
2. Distinguish homozygous from WT/Het

This replicates cep290_statistical_analysis_1hr.py but uses curvature instead of VAE embeddings.

Key question: Can we predict phenotype from shape alone, without deep learning features?

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

OUTPUT_DIR = Path(__file__).parent / "curvature_only_results_2hr"
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251229_cep290_phenotype_extraction")

CLUSTERING_DATA_PATH = DATA_DIR / "data" / "clustering_data__early_homo.pkl"
K_RESULTS_PATH = DATA_DIR / "kmedoids_k_selection_early_timepoints_cep290_data" / "k_results.pkl"

# Analysis parameters
METRIC_COL = 'baseline_deviation_normalized'
METRIC_LABEL = 'Baseline Deviation (normalized)'
K_VALUE = 5
TIME_BIN_WIDTH = 2.0  # 2-hour bins for direct comparison to standard analysis
N_PERMUTATIONS_CLASS = 100
N_JOBS = -1
MIN_SAMPLES_PER_BIN = 3

# Cluster definitions
CLUSTER_NAMES_K5 = {
    0: 'outlier',
    1: 'bumpy',
    2: 'low_to_high',
    3: 'low_to_high',
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

    print("\nGenotype distribution:")
    for geno, count in df.groupby('genotype')['embryo_id'].nunique().items():
        print(f"  {geno}: {count} embryos")

    print("\nCluster distribution (homozygous only):")
    df_homo = df[df['genotype'] == 'cep290_homozygous']
    for cluster, count in df_homo.groupby('cluster')['embryo_id'].nunique().items():
        print(f"  {cluster}: {count} embryos")

    return df, k_results


# =============================================================================
# Part 1: Per-Cluster Classification (Curvature Only)
# =============================================================================

def run_cluster_classification(df):
    """Run classification for each cluster vs WT using CURVATURE ONLY."""
    print("\n" + "="*70)
    print("PART 1: Per-Cluster Classification (Curvature Only, 1-hour bins)")
    print("="*70)
    print("\n⭐ KEY DIFFERENCE: Using baseline_deviation_normalized instead of VAE embeddings")

    all_results = {}

    # Baseline: Het vs WT
    print("\n--- Het_vs_WT (Baseline) ---")
    results_baseline = compare_groups(
        df,
        group_col='genotype',
        group1='cep290_heterozygous',
        group2='cep290_wildtype',
        features=METRIC_COL,  # ← CURVATURE ONLY (not 'z_mu_b')
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

    # Each cluster vs WT
    for cluster in CLUSTERS_TO_ANALYZE:
        label = f"{cluster}_vs_WT"
        print(f"\n--- {label} ---")

        df_subset = df[
            (df['cluster'] == cluster) | (df['genotype'] == 'cep290_wildtype')
        ].copy()

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
            features=METRIC_COL,  # ← CURVATURE ONLY
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
# Part 2: Pooled Classification (Curvature Only)
# =============================================================================

def run_pooled_classification(df):
    """Run pooled classification using CURVATURE ONLY."""
    print("\n" + "="*70)
    print("PART 2: Pooled Classification (Curvature Only, 1-hour bins)")
    print("="*70)

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
            features=METRIC_COL,  # ← CURVATURE ONLY
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
# Plotting Functions (identical to embedding version)
# =============================================================================

def plot_cluster_vs_wt_auroc(cluster_results):
    """Plot AUROC for each cluster vs WT."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Baseline
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

    # Each cluster
    for cluster in CLUSTERS_TO_ANALYZE:
        label = f"{cluster}_vs_WT"
        if label not in cluster_results:
            continue

        results = cluster_results[label]
        df_class = results['classification']
        color = COLORS[label]

        ax.plot(df_class['time_bin'], df_class['auroc_observed'],
               'o-', color=color, linewidth=2, markersize=5,
               label=f"{cluster} vs WT")

        if 'auroc_null_mean' in df_class.columns and 'auroc_null_std' in df_class.columns:
            ax.fill_between(
                df_class['time_bin'],
                df_class['auroc_null_mean'] - df_class['auroc_null_std'],
                df_class['auroc_null_mean'] + df_class['auroc_null_std'],
                color=color,
                alpha=0.10,
                linewidth=0
            )

        sig_mask = df_class['pval'] < 0.05
        if sig_mask.any():
            ax.scatter(df_class.loc[sig_mask, 'time_bin'],
                      df_class.loc[sig_mask, 'auroc_observed'],
                      s=200, facecolors='none', edgecolors=color, linewidths=2.5,
                      zorder=5)

        very_sig_mask = df_class['pval'] < 0.01
        if very_sig_mask.any():
            for _, row in df_class[very_sig_mask].iterrows():
                ax.annotate('*', (row['time_bin'], row['auroc_observed'] + 0.03),
                           ha='center', fontsize=14, fontweight='bold', color=color)

    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Chance (0.5)')

    ax.set_xlabel('Hours Post Fertilization (hpf)', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title('Per-Cluster Classification vs WT (CURVATURE ONLY)\n(baseline_deviation_normalized, 1-hour bins, circles = p<0.05, * = p<0.01)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(0.3, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_pooled_auroc_with_pvalues(all_results):
    """Plot pooled AUROC."""
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

        ax.plot(df_class['time_bin'], df_class['auroc_observed'],
                f'o{style}', label=label, color=color, linewidth=2, markersize=5)

        if 'auroc_null_mean' in df_class.columns and 'auroc_null_std' in df_class.columns:
            ax.fill_between(
                df_class['time_bin'],
                df_class['auroc_null_mean'] - df_class['auroc_null_std'],
                df_class['auroc_null_mean'] + df_class['auroc_null_std'],
                color=color,
                alpha=0.10,
                linewidth=0
            )

        sig_mask = df_class['pval'] < 0.05
        if sig_mask.any():
            ax.scatter(df_class.loc[sig_mask, 'time_bin'],
                      df_class.loc[sig_mask, 'auroc_observed'],
                      s=200, facecolors='none', edgecolors=color, linewidths=2.5,
                      zorder=5)

        very_sig_mask = df_class['pval'] < 0.01
        if very_sig_mask.any():
            for _, row in df_class[very_sig_mask].iterrows():
                ax.annotate('*', (row['time_bin'], row['auroc_observed'] + 0.03),
                           ha='center', fontsize=14, fontweight='bold', color=color)

    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Chance (0.5)')

    ax.set_xlabel('Hours Post Fertilization (hpf)', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title('Pooled Classification Performance (CURVATURE ONLY)\n(baseline_deviation_normalized, 1-hour bins, circles = p<0.05, * = p<0.01)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(0.3, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_temporal_emergence(all_results, title_prefix=""):
    """Plot temporal emergence."""
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

        bars = ax.bar(df_class['time_bin'], df_class['auroc_observed'],
                     width=TIME_BIN_WIDTH * 0.8, color=color, alpha=0.6)

        for i, (idx, row) in enumerate(df_class.iterrows()):
            if row['pval'] < 0.05:
                bars[i].set_alpha(1.0)
                bars[i].set_edgecolor('black')
                bars[i].set_linewidth(2)
            if row['pval'] < 0.01:
                ax.annotate('*', (row['time_bin'], row['auroc_observed'] + 0.02),
                           ha='center', fontsize=12, fontweight='bold')

        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

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
    fig.suptitle(f'{title_prefix}Temporal Emergence (Curvature Only)', fontsize=14, y=1.02)

    plt.tight_layout()
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*70)
    print("CEP290 Analysis: Curvature Only (No VAE Embeddings)")
    print("="*70)
    print("\n⭐ KEY QUESTION: Is raw curvature sufficient to predict phenotype?")
    print("  - Using baseline_deviation_normalized as single feature")
    print("  - Compare to VAE embedding results\n")

    # Load data
    df, k_results = load_data()

    # Part 1: Per-cluster classification
    cluster_results = run_cluster_classification(df)

    # Part 2: Pooled classification
    pooled_results = run_pooled_classification(df)

    # Combine
    all_results = {**cluster_results, **pooled_results}

    # Save CSV
    classification_records = []
    for label, results in all_results.items():
        df_class = results['classification'].copy()
        df_class['comparison'] = label
        classification_records.append(df_class)

    df_classification = pd.concat(classification_records, ignore_index=True)
    class_csv_path = OUTPUT_DIR / "curvature_only_classification_results.csv"
    df_classification.to_csv(class_csv_path, index=False)
    print(f"\n✓ Saved classification results to: {class_csv_path}")

    # Plot 1: Per-cluster AUROC
    fig_cluster = plot_cluster_vs_wt_auroc(cluster_results)
    cluster_plot_path = OUTPUT_DIR / "curvature_cluster_vs_wt_auroc.png"
    fig_cluster.savefig(cluster_plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig_cluster)
    print(f"✓ Saved cluster AUROC plot: {cluster_plot_path}")

    # Plot 2: Pooled AUROC
    fig_pooled = plot_pooled_auroc_with_pvalues(pooled_results)
    pooled_plot_path = OUTPUT_DIR / "curvature_pooled_auroc.png"
    fig_pooled.savefig(pooled_plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig_pooled)
    print(f"✓ Saved pooled AUROC plot: {pooled_plot_path}")

    # Plot 3: Temporal emergence for clusters
    fig_temporal_cluster = plot_temporal_emergence(cluster_results, "Per-Cluster: ")
    temporal_cluster_path = OUTPUT_DIR / "curvature_temporal_emergence_clusters.png"
    fig_temporal_cluster.savefig(temporal_cluster_path, dpi=150, bbox_inches='tight')
    plt.close(fig_temporal_cluster)
    print(f"✓ Saved cluster temporal emergence: {temporal_cluster_path}")

    # Plot 4: Temporal emergence for pooled
    fig_temporal_pooled = plot_temporal_emergence(pooled_results, "Pooled: ")
    temporal_pooled_path = OUTPUT_DIR / "curvature_temporal_emergence_pooled.png"
    fig_temporal_pooled.savefig(temporal_pooled_path, dpi=150, bbox_inches='tight')
    plt.close(fig_temporal_pooled)
    print(f"✓ Saved pooled temporal emergence: {temporal_pooled_path}")

    # Plot 5-7: Comprehensive figures
    print("\n" + "="*70)
    print("GENERATING COMPREHENSIVE FIGURES (Curvature Only)")
    print("="*70)
    
    for cluster in CLUSTERS_TO_ANALYZE:
        label = f"{cluster}_vs_WT"
        if label not in cluster_results:
            continue
            
        print(f"\n  Creating comprehensive figure for {label}...")
        
        cluster_ids = df[df['cluster'] == cluster]['embryo_id'].unique().tolist()
        wt_ids = df[df['genotype'] == 'wildtype']['embryo_id'].unique().tolist()
        
        comprehensive_path = OUTPUT_DIR / f"curvature_{label}_comprehensive.png"
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
            plt.close(fig)

    # =============================================================================
    # Part 3: Batch Control - Compare Within Genotypes Across Experiments
    # =============================================================================
    
    print("\n" + "="*70)
    print("PART 3: Batch Control - Within-Genotype Cross-Experiment Comparison")
    print("="*70)
    print("\n⚠️  CONTROL: If batch differences are larger than genotype differences,")
    print("   the phenotype signal may be a technical artifact!")
    
    batch_control_results = {}
    
    # Get experiment IDs
    experiments = sorted(df['experiment_date'].unique())
    print(f"\nExperiments: {experiments}")
    
    # Control 1: WT between experiments
    print("\n--- WT_exp1_vs_WT_exp2 (Batch Control) ---")
    df_wt = df[df['genotype'] == 'cep290_wildtype'].copy()
    
    if len(experiments) >= 2:
        df_wt['exp_group'] = df_wt['experiment_date'].apply(
            lambda x: f'WT_{experiments[0]}' if x == experiments[0] else f'WT_{experiments[1]}'
        )
        
        n_exp1 = df_wt[df_wt['experiment_date'] == experiments[0]]['embryo_id'].nunique()
        n_exp2 = df_wt[df_wt['experiment_date'] == experiments[1]]['embryo_id'].nunique()
        print(f"  WT from {experiments[0]}: {n_exp1} embryos")
        print(f"  WT from {experiments[1]}: {n_exp2} embryos")
        
        if n_exp1 >= MIN_SAMPLES_PER_BIN and n_exp2 >= MIN_SAMPLES_PER_BIN:
            results_wt_batch = compare_groups(
                df_wt,
                group_col='exp_group',
                group1=f'WT_{experiments[0]}',
                group2=f'WT_{experiments[1]}',
                features=METRIC_COL,
                morphology_metric=METRIC_COL,
                bin_width=TIME_BIN_WIDTH,
                n_permutations=N_PERMUTATIONS_CLASS,
                n_jobs=N_JOBS,
                min_samples_per_bin=MIN_SAMPLES_PER_BIN,
                random_state=42,
                verbose=True
            )
            batch_control_results['WT_batch'] = results_wt_batch
            
            summary = results_wt_batch['summary']
            max_auroc = summary['max_auroc'] if summary['max_auroc'] is not None else 0.0
            print(f"\n  ⚠️  Batch effect AUROC: {max_auroc:.3f}")
            if max_auroc > 0.7:
                print(f"  ⚠️  WARNING: Strong batch effect detected! (AUROC > 0.7)")
            elif max_auroc > 0.6:
                print(f"  ⚠️  Moderate batch effect (AUROC > 0.6)")
            else:
                print(f"  ✓ Low batch effect (AUROC ≤ 0.6)")
        else:
            print(f"  Skipped: Insufficient samples in one experiment")
    
    # Control 2: Het between experiments
    print("\n--- Het_exp1_vs_Het_exp2 (Batch Control) ---")
    df_het = df[df['genotype'] == 'cep290_heterozygous'].copy()
    
    if len(experiments) >= 2:
        df_het['exp_group'] = df_het['experiment_date'].apply(
            lambda x: f'Het_{experiments[0]}' if x == experiments[0] else f'Het_{experiments[1]}'
        )
        
        n_exp1 = df_het[df_het['experiment_date'] == experiments[0]]['embryo_id'].nunique()
        n_exp2 = df_het[df_het['experiment_date'] == experiments[1]]['embryo_id'].nunique()
        print(f"  Het from {experiments[0]}: {n_exp1} embryos")
        print(f"  Het from {experiments[1]}: {n_exp2} embryos")
        
        if n_exp1 >= MIN_SAMPLES_PER_BIN and n_exp2 >= MIN_SAMPLES_PER_BIN:
            results_het_batch = compare_groups(
                df_het,
                group_col='exp_group',
                group1=f'Het_{experiments[0]}',
                group2=f'Het_{experiments[1]}',
                features=METRIC_COL,
                morphology_metric=METRIC_COL,
                bin_width=TIME_BIN_WIDTH,
                n_permutations=N_PERMUTATIONS_CLASS,
                n_jobs=N_JOBS,
                min_samples_per_bin=MIN_SAMPLES_PER_BIN,
                random_state=42,
                verbose=True
            )
            batch_control_results['Het_batch'] = results_het_batch
            
            summary = results_het_batch['summary']
            max_auroc = summary['max_auroc'] if summary['max_auroc'] is not None else 0.0
            print(f"\n  ⚠️  Batch effect AUROC: {max_auroc:.3f}")
            if max_auroc > 0.7:
                print(f"  ⚠️  WARNING: Strong batch effect detected! (AUROC > 0.7)")
            elif max_auroc > 0.6:
                print(f"  ⚠️  Moderate batch effect (AUROC > 0.6)")
            else:
                print(f"  ✓ Low batch effect (AUROC ≤ 0.6)")
        else:
            print(f"  Skipped: Insufficient samples in one experiment")
    
    # Save batch control results
    if batch_control_results:
        batch_records = []
        for label, results in batch_control_results.items():
            df_class = results['classification'].copy()
            df_class['comparison'] = label
            batch_records.append(df_class)
        
        df_batch = pd.concat(batch_records, ignore_index=True)
        batch_csv_path = OUTPUT_DIR / "batch_control_results.csv"
        df_batch.to_csv(batch_csv_path, index=False)
        print(f"\n✓ Saved batch control results: {batch_csv_path}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY (Curvature Only)")
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

    print("\n⚠️  BATCH CONTROL (Within-Genotype Across Experiments):")
    for label, results in batch_control_results.items():
        s = results['summary']
        max_auroc = s['max_auroc'] if s['max_auroc'] is not None else 0.0
        max_hpf = s['max_auroc_hpf'] if s['max_auroc_hpf'] is not None else 'N/A'
        print(f"  {label}:")
        print(f"    Max AUROC: {max_auroc:.3f} at {max_hpf} hpf")
        print(f"    Earliest significant: {s['earliest_significant_hpf']} hpf")
        if max_auroc > 0.7:
            print(f"    ⚠️  WARNING: Batch effect comparable to phenotype signal!")
        elif max_auroc > 0.6:
            print(f"    ⚠️  Moderate batch effect detected")
        else:
            print(f"    ✓ Batch effect appears minimal")

    print("\n⭐ TO COMPARE WITH VAE EMBEDDINGS:")
    print("  - Load results_1hr_bins/classification_results_1hr.csv (VAE embeddings)")
    print(f"  - Load {class_csv_path} (curvature only)")
    print("  - Compare Max AUROC and earliest_significant_hpf")
    print("  - If curvature AUROC is similar → Deep learning not necessary")
    print("  - If curvature AUROC is much lower → VAE captures important info")

    print("\n⭐ BATCH CONTROL INTERPRETATION:")
    print("  - If batch AUROC > phenotype AUROC → Technical artifact, not biology!")
    print("  - If batch AUROC ≪ phenotype AUROC → Biological signal is robust")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
