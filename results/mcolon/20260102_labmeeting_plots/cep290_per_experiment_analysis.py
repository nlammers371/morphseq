#!/usr/bin/env python
"""
CEP290 Per-Experiment Analysis (Batch Effect Control)

This script runs the same CEP290 classification analyses as cep290_statistical_analysis.py,
but separately within each experiment to control for batch effects.

If phenotype signals are robust, they should appear consistently across experiments.
If signals only appear in pooled data but not per-experiment, this suggests batch artifacts.

Analyses performed per experiment:
1. Per-cluster classification: Each phenotype cluster vs WT (within experiment)
2. Pooled classification: Homo vs WT, Homo vs Het (within experiment)

Author: Generated for CEP290 batch effect control
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
from comparison_plotting_utils import (
    create_full_comparison, 
    plot_auroc_overlay, 
    plot_temporal_emergence
)

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = Path(__file__).parent / "per_experiment_results"
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251229_cep290_phenotype_extraction")

CLUSTERING_DATA_PATH = DATA_DIR / "data" / "clustering_data__early_homo.pkl"
K_RESULTS_PATH = DATA_DIR / "kmedoids_k_selection_early_timepoints_cep290_data" / "k_results.pkl"

# Analysis parameters
METRIC_COL = 'baseline_deviation_normalized'
K_VALUE = 5
TIME_BIN_WIDTH = 2.0  # hours
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

    print("\nGenotype distribution (all experiments):")
    for geno, count in df.groupby('genotype')['embryo_id'].nunique().items():
        print(f"  {geno}: {count} embryos")

    print("\nExperiment distribution:")
    for exp, count in df.groupby('experiment_date')['embryo_id'].nunique().items():
        print(f"  Experiment {exp}: {count} embryos")
        # Per-genotype breakdown
        df_exp = df[df['experiment_date'] == exp]
        for geno, geno_count in df_exp.groupby('genotype')['embryo_id'].nunique().items():
            print(f"    {geno}: {geno_count} embryos")

    return df


# =============================================================================
# Per-Experiment Analysis
# =============================================================================

def run_per_experiment_analysis(df, experiment_id):
    """Run all analyses for a single experiment."""
    print("\n" + "="*70)
    print(f"EXPERIMENT: {experiment_id}")
    print("="*70)
    
    df_exp = df[df['experiment_date'] == experiment_id].copy()
    
    print(f"\nData for experiment {experiment_id}:")
    print(f"  Total rows: {len(df_exp)}")
    print(f"  Embryos: {df_exp['embryo_id'].nunique()}")
    print(f"  Genotypes:")
    for geno, count in df_exp.groupby('genotype')['embryo_id'].nunique().items():
        print(f"    {geno}: {count} embryos")
    
    # Create experiment-specific output directory
    exp_output_dir = OUTPUT_DIR / f"experiment_{experiment_id}"
    exp_output_dir.mkdir(exist_ok=True)
    
    exp_results = {
        'experiment_id': experiment_id,
        'cluster_vs_wt': {},
        'pooled': {},
        'output_dir': exp_output_dir
    }
    
    # -------------------------------------------------------------------------
    # Part 1: Per-Cluster vs WT (within this experiment)
    # -------------------------------------------------------------------------
    print("\n--- Part 1: Per-Cluster Classification (vs WT) ---")
    
    for cluster in CLUSTERS_TO_ANALYZE:
        label = f"{cluster}_vs_WT"
        print(f"\n{label} (Experiment {experiment_id}):")
        
        df_subset = df_exp[
            (df_exp['cluster'] == cluster) | (df_exp['genotype'] == 'cep290_wildtype')
        ].copy()
        
        df_subset['comparison_group'] = df_subset.apply(
            lambda r: cluster if r['cluster'] == cluster else 'WT', axis=1
        )
        
        n_cluster = df_subset[df_subset['comparison_group'] == cluster]['embryo_id'].nunique()
        n_wt = df_subset[df_subset['comparison_group'] == 'WT']['embryo_id'].nunique()
        
        print(f"  {cluster}: {n_cluster} embryos, WT: {n_wt} embryos")
        
        if n_cluster < MIN_SAMPLES_PER_BIN or n_wt < MIN_SAMPLES_PER_BIN:
            print(f"  ⚠️  Skipped: Insufficient samples")
            exp_results['cluster_vs_wt'][label] = None
            continue
        
        try:
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
                verbose=False
            )
            exp_results['cluster_vs_wt'][label] = results
            
            summary = results['summary']
            max_auroc = summary['max_auroc'] if summary['max_auroc'] is not None else 0.0
            print(f"  Max AUROC: {max_auroc:.3f}, Earliest sig: {summary['earliest_significant_hpf']}")
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            exp_results['cluster_vs_wt'][label] = None
    
    # -------------------------------------------------------------------------
    # Part 2: Pooled Classification (within this experiment)
    # -------------------------------------------------------------------------
    print("\n--- Part 2: Pooled Classification ---")
    
    # Homo vs WT
    print(f"\nHomo_vs_WT (Experiment {experiment_id}):")
    df_homo_wt = df_exp[df_exp['genotype'].isin(['cep290_homozygous', 'cep290_wildtype'])].copy()
    
    n_homo = df_homo_wt[df_homo_wt['genotype'] == 'cep290_homozygous']['embryo_id'].nunique()
    n_wt = df_homo_wt[df_homo_wt['genotype'] == 'cep290_wildtype']['embryo_id'].nunique()
    print(f"  Homo: {n_homo} embryos, WT: {n_wt} embryos")
    
    if n_homo >= MIN_SAMPLES_PER_BIN and n_wt >= MIN_SAMPLES_PER_BIN:
        try:
            results_homo_wt = compare_groups(
                df_homo_wt,
                group_col='genotype',
                group1='cep290_homozygous',
                group2='cep290_wildtype',
                features='z_mu_b',
                morphology_metric=METRIC_COL,
                bin_width=TIME_BIN_WIDTH,
                n_permutations=N_PERMUTATIONS_CLASS,
                n_jobs=N_JOBS,
                min_samples_per_bin=MIN_SAMPLES_PER_BIN,
                random_state=42,
                verbose=False
            )
            exp_results['pooled']['Homo_vs_WT'] = results_homo_wt
            
            summary = results_homo_wt['summary']
            max_auroc = summary['max_auroc'] if summary['max_auroc'] is not None else 0.0
            print(f"  Max AUROC: {max_auroc:.3f}, Earliest sig: {summary['earliest_significant_hpf']}")
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            exp_results['pooled']['Homo_vs_WT'] = None
    else:
        print(f"  ⚠️  Skipped: Insufficient samples")
        exp_results['pooled']['Homo_vs_WT'] = None
    
    # Homo vs Het
    print(f"\nHomo_vs_Het (Experiment {experiment_id}):")
    df_homo_het = df_exp[df_exp['genotype'].isin(['cep290_homozygous', 'cep290_heterozygous'])].copy()
    
    n_homo = df_homo_het[df_homo_het['genotype'] == 'cep290_homozygous']['embryo_id'].nunique()
    n_het = df_homo_het[df_homo_het['genotype'] == 'cep290_heterozygous']['embryo_id'].nunique()
    print(f"  Homo: {n_homo} embryos, Het: {n_het} embryos")
    
    if n_homo >= MIN_SAMPLES_PER_BIN and n_het >= MIN_SAMPLES_PER_BIN:
        try:
            results_homo_het = compare_groups(
                df_homo_het,
                group_col='genotype',
                group1='cep290_homozygous',
                group2='cep290_heterozygous',
                features='z_mu_b',
                morphology_metric=METRIC_COL,
                bin_width=TIME_BIN_WIDTH,
                n_permutations=N_PERMUTATIONS_CLASS,
                n_jobs=N_JOBS,
                min_samples_per_bin=MIN_SAMPLES_PER_BIN,
                random_state=42,
                verbose=False
            )
            exp_results['pooled']['Homo_vs_Het'] = results_homo_het
            
            summary = results_homo_het['summary']
            max_auroc = summary['max_auroc'] if summary['max_auroc'] is not None else 0.0
            print(f"  Max AUROC: {max_auroc:.3f}, Earliest sig: {summary['earliest_significant_hpf']}")
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            exp_results['pooled']['Homo_vs_Het'] = None
    else:
        print(f"  ⚠️  Skipped: Insufficient samples")
        exp_results['pooled']['Homo_vs_Het'] = None
    
    # -------------------------------------------------------------------------
    # Part 3: Generate Comprehensive Figures
    # -------------------------------------------------------------------------
    print("\n--- Part 3: Generating Comprehensive Figures ---")
    
    # Cluster vs WT comprehensive figures
    for cluster in CLUSTERS_TO_ANALYZE:
        label = f"{cluster}_vs_WT"
        
        if label in exp_results['cluster_vs_wt'] and exp_results['cluster_vs_wt'][label] is not None:
            print(f"\n  Creating comprehensive figure: {label}")
            
            results = exp_results['cluster_vs_wt'][label]
            
            # Get embryo IDs
            cluster_ids = df_exp[df_exp['cluster'] == cluster]['embryo_id'].unique().tolist()
            wt_ids = df_exp[df_exp['genotype'] == 'cep290_wildtype']['embryo_id'].unique().tolist()
            
            comprehensive_path = exp_output_dir / f"{label}_comprehensive.png"
            
            try:
                fig, divergence_df = create_full_comparison(
                    df=df_exp,
                    df_results=results['classification'],
                    group1_ids=cluster_ids,
                    group2_ids=wt_ids,
                    group1_label=cluster,
                    group2_label='WT',
                    metric_col=METRIC_COL,
                    metric_label='Baseline Deviation (normalized)',
                    save_path=comprehensive_path,
                    time_col='predicted_stage_hpf',
                    embryo_id_col='embryo_id'
                )
                
                if fig is not None:
                    plt.close(fig)
                print(f"    ✓ Saved: {comprehensive_path.name}")
            except Exception as e:
                print(f"    ⚠️  Error generating figure: {e}")
    
    # Pooled comprehensive figures
    for label in ['Homo_vs_WT', 'Homo_vs_Het']:
        if label in exp_results['pooled'] and exp_results['pooled'][label] is not None:
            print(f"\n  Creating comprehensive figure: {label}")
            
            results = exp_results['pooled'][label]
            
            # Get embryo IDs
            if label == 'Homo_vs_WT':
                group1_ids = df_exp[df_exp['genotype'] == 'cep290_homozygous']['embryo_id'].unique().tolist()
                group2_ids = df_exp[df_exp['genotype'] == 'cep290_wildtype']['embryo_id'].unique().tolist()
                group1_label = 'Homo'
                group2_label = 'WT'
            else:  # Homo_vs_Het
                group1_ids = df_exp[df_exp['genotype'] == 'cep290_homozygous']['embryo_id'].unique().tolist()
                group2_ids = df_exp[df_exp['genotype'] == 'cep290_heterozygous']['embryo_id'].unique().tolist()
                group1_label = 'Homo'
                group2_label = 'Het'
            
            comprehensive_path = exp_output_dir / f"{label}_comprehensive.png"
            
            try:
                fig, divergence_df = create_full_comparison(
                    df=df_exp,
                    df_results=results['classification'],
                    group1_ids=group1_ids,
                    group2_ids=group2_ids,
                    group1_label=group1_label,
                    group2_label=group2_label,
                    metric_col=METRIC_COL,
                    metric_label='Baseline Deviation (normalized)',
                    save_path=comprehensive_path,
                    time_col='predicted_stage_hpf',
                    embryo_id_col='embryo_id'
                )
                
                if fig is not None:
                    plt.close(fig)
                print(f"    ✓ Saved: {comprehensive_path.name}")
            except Exception as e:
                print(f"    ⚠️  Error generating figure: {e}")
    
    # -------------------------------------------------------------------------
    # Part 4: Generate Overlay Plots
    # -------------------------------------------------------------------------
    print("\n--- Part 4: Generating Overlay Plots ---")
    
    # Cluster vs WT overlay
    cluster_results_valid = {k: v for k, v in exp_results['cluster_vs_wt'].items() if v is not None}
    if cluster_results_valid:
        print("\n  Creating cluster vs WT overlay plot...")
        fig_cluster = plot_auroc_overlay(
            results_dict=cluster_results_valid,
            colors=COLORS,
            title=f'Per-Cluster Classification vs WT (Experiment {experiment_id})',
            time_bin_width=TIME_BIN_WIDTH
        )
        cluster_overlay_path = exp_output_dir / "cluster_vs_wt_overlay.png"
        fig_cluster.savefig(cluster_overlay_path, dpi=300, bbox_inches='tight')
        plt.close(fig_cluster)
        print(f"    ✓ Saved: {cluster_overlay_path.name}")
    
    # Pooled overlay
    pooled_results_valid = {k: v for k, v in exp_results['pooled'].items() if v is not None}
    if pooled_results_valid:
        print("\n  Creating pooled overlay plot...")
        fig_pooled = plot_auroc_overlay(
            results_dict=pooled_results_valid,
            colors=COLORS,
            title=f'Pooled Classification (Experiment {experiment_id})',
            time_bin_width=TIME_BIN_WIDTH
        )
        pooled_overlay_path = exp_output_dir / "pooled_overlay.png"
        fig_pooled.savefig(pooled_overlay_path, dpi=300, bbox_inches='tight')
        plt.close(fig_pooled)
        print(f"    ✓ Saved: {pooled_overlay_path.name}")
    
    # -------------------------------------------------------------------------
    # Part 5: Generate Temporal Emergence Plots
    # -------------------------------------------------------------------------
    print("\n--- Part 5: Generating Temporal Emergence Plots ---")
    
    # Cluster temporal emergence
    if cluster_results_valid:
        print("\n  Creating cluster temporal emergence plot...")
        fig_temporal_cluster = plot_temporal_emergence(
            results_dict=cluster_results_valid,
            colors=COLORS,
            time_bin_width=TIME_BIN_WIDTH,
            title_prefix=f"Experiment {experiment_id} - "
        )
        temporal_cluster_path = exp_output_dir / "temporal_emergence_clusters.png"
        fig_temporal_cluster.savefig(temporal_cluster_path, dpi=300, bbox_inches='tight')
        plt.close(fig_temporal_cluster)
        print(f"    ✓ Saved: {temporal_cluster_path.name}")
    
    # Pooled temporal emergence
    if pooled_results_valid:
        print("\n  Creating pooled temporal emergence plot...")
        fig_temporal_pooled = plot_temporal_emergence(
            results_dict=pooled_results_valid,
            colors=COLORS,
            time_bin_width=TIME_BIN_WIDTH,
            title_prefix=f"Experiment {experiment_id} - "
        )
        temporal_pooled_path = exp_output_dir / "temporal_emergence_pooled.png"
        fig_temporal_pooled.savefig(temporal_pooled_path, dpi=300, bbox_inches='tight')
        plt.close(fig_temporal_pooled)
        print(f"    ✓ Saved: {temporal_pooled_path.name}")
    
    return exp_results


# =============================================================================
# Comparison Across Experiments
# =============================================================================

def compare_experiments(all_exp_results):
    """Compare results across experiments to assess consistency."""
    print("\n" + "="*70)
    print("CROSS-EXPERIMENT COMPARISON")
    print("="*70)
    
    # Collect results by comparison type
    comparison_types = set()
    for exp_data in all_exp_results.values():
        comparison_types.update(exp_data['cluster_vs_wt'].keys())
        comparison_types.update(exp_data['pooled'].keys())
    
    comparison_summary = []
    
    for comp_type in sorted(comparison_types):
        print(f"\n{comp_type}:")
        print("-" * 50)
        
        row = {'comparison': comp_type}
        
        for exp_id, exp_data in all_exp_results.items():
            # Find results in either cluster_vs_wt or pooled
            results = None
            if comp_type in exp_data['cluster_vs_wt']:
                results = exp_data['cluster_vs_wt'][comp_type]
            elif comp_type in exp_data['pooled']:
                results = exp_data['pooled'][comp_type]
            
            if results is not None:
                summary = results['summary']
                max_auroc = summary['max_auroc'] if summary['max_auroc'] is not None else 0.0
                earliest = summary['earliest_significant_hpf']
                print(f"  Exp {exp_id}: AUROC={max_auroc:.3f}, Earliest={earliest} hpf")
                
                row[f'exp_{exp_id}_auroc'] = max_auroc
                row[f'exp_{exp_id}_earliest_hpf'] = earliest
            else:
                print(f"  Exp {exp_id}: No data / Skipped")
                row[f'exp_{exp_id}_auroc'] = None
                row[f'exp_{exp_id}_earliest_hpf'] = None
        
        comparison_summary.append(row)
    
    # Create summary DataFrame
    df_summary = pd.DataFrame(comparison_summary)
    summary_path = OUTPUT_DIR / "cross_experiment_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"\n✓ Saved cross-experiment summary: {summary_path}")
    
    return df_summary


def plot_cross_experiment_comparison(df_summary, all_exp_results):
    """Create plots comparing AUROC across experiments."""
    print("\n" + "="*70)
    print("GENERATING COMPARISON PLOTS")
    print("="*70)
    
    experiments = sorted(all_exp_results.keys())
    
    # Get comparison types
    cluster_comparisons = [c for c in df_summary['comparison'] if '_vs_WT' in c and c in CLUSTERS_TO_ANALYZE or any(cl in c for cl in CLUSTERS_TO_ANALYZE)]
    pooled_comparisons = [c for c in df_summary['comparison'] if 'Homo_vs' in c]
    
    # Plot 1: Per-Cluster Comparisons
    if cluster_comparisons:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = np.arange(len(cluster_comparisons))
        width = 0.35
        
        for i, exp_id in enumerate(experiments):
            auroc_col = f'exp_{exp_id}_auroc'
            aurocs = df_summary[df_summary['comparison'].isin(cluster_comparisons)][auroc_col].values
            
            offset = width * (i - len(experiments)/2 + 0.5)
            ax.bar(x_pos + offset, aurocs, width, label=f'Exp {exp_id}', alpha=0.8)
        
        ax.axhline(y=0.5, color='gray', linestyle=':', label='Chance')
        ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.3, label='Good separation')
        
        ax.set_xlabel('Comparison', fontsize=12)
        ax.set_ylabel('Max AUROC', fontsize=12)
        ax.set_title('Per-Cluster Classification: AUROC Across Experiments', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(cluster_comparisons, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0.4, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_path = OUTPUT_DIR / "per_cluster_cross_experiment.png"
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Saved: {plot_path}")
    
    # Plot 2: Pooled Comparisons
    if pooled_comparisons:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        x_pos = np.arange(len(pooled_comparisons))
        width = 0.35
        
        for i, exp_id in enumerate(experiments):
            auroc_col = f'exp_{exp_id}_auroc'
            aurocs = df_summary[df_summary['comparison'].isin(pooled_comparisons)][auroc_col].values
            
            offset = width * (i - len(experiments)/2 + 0.5)
            ax.bar(x_pos + offset, aurocs, width, label=f'Exp {exp_id}', alpha=0.8)
        
        ax.axhline(y=0.5, color='gray', linestyle=':', label='Chance')
        ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.3, label='Good separation')
        
        ax.set_xlabel('Comparison', fontsize=12)
        ax.set_ylabel('Max AUROC', fontsize=12)
        ax.set_title('Pooled Classification: AUROC Across Experiments', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(pooled_comparisons, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0.4, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_path = OUTPUT_DIR / "pooled_cross_experiment.png"
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Saved: {plot_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*70)
    print("CEP290 PER-EXPERIMENT ANALYSIS")
    print("="*70)
    print("\nObjective: Test if phenotype signals are consistent across experiments")
    print("(Controls for batch effects)")
    
    # Load data
    df = load_data()
    
    # Get unique experiments
    experiments = sorted(df['experiment_date'].unique())
    print(f"\nFound {len(experiments)} experiments: {experiments}")
    
    # Run analysis per experiment
    all_exp_results = {}
    
    for exp_id in experiments:
        exp_results = run_per_experiment_analysis(df, exp_id)
        all_exp_results[exp_id] = exp_results
        
        # Save per-experiment results
        exp_records = []
        
        # Cluster vs WT results
        for label, results in exp_results['cluster_vs_wt'].items():
            if results is not None:
                df_class = results['classification'].copy()
                df_class['comparison'] = label
                df_class['experiment_id'] = exp_id
                exp_records.append(df_class)
        
        # Pooled results
        for label, results in exp_results['pooled'].items():
            if results is not None:
                df_class = results['classification'].copy()
                df_class['comparison'] = label
                df_class['experiment_id'] = exp_id
                exp_records.append(df_class)
        
        if exp_records:
            df_exp = pd.concat(exp_records, ignore_index=True)
            csv_path = OUTPUT_DIR / f"experiment_{exp_id}_results.csv"
            df_exp.to_csv(csv_path, index=False)
            print(f"\n✓ Saved experiment {exp_id} results: {csv_path}")
    
    # Cross-experiment comparison
    df_summary = compare_experiments(all_exp_results)
    
    # Generate comparison plots
    plot_cross_experiment_comparison(df_summary, all_exp_results)
    
    # Final summary
    print("\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)
    print("\n✓ ROBUST SIGNAL:")
    print("  - Similar AUROC across both experiments")
    print("  - Similar earliest_significant_hpf across experiments")
    print("  → Phenotype is real, not a batch artifact")
    
    print("\n⚠️  POTENTIAL BATCH ARTIFACT:")
    print("  - High AUROC in one experiment, low in another")
    print("  - Signal only appears in pooled data")
    print("  → May be driven by technical differences between experiments")
    
    print("\n⚠️  INSUFFICIENT POWER:")
    print("  - Low sample size per experiment")
    print("  - May need pooled data for detection")
    print("  → Check if batch control AUROC is low to justify pooling")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
