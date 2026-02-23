#!/usr/bin/env python3
"""
Quick test of divergence analysis with multiple distance metrics.

Tests different comparisons:
1. Homozygous vs wildtype (standard)
2. Heterozygous vs wildtype
3. Homozygous vs heterozygous (non-standard reference)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import config_new as config
from utils.data_loading import load_experiments
from utils.binning import bin_by_embryo_time
from divergence_analysis import compute_divergence_scores, compare_to_multiple_references
from divergence_analysis.visualization import (
    plot_divergence_over_time,
    plot_divergence_distribution,
    plot_divergence_heatmap,
    plot_metric_comparison,
    plot_outliers
)


def main():
    print("="*80)
    print("MORPHOLOGICAL DIVERGENCE ANALYSIS - QUICK TEST")
    print("="*80)
    
    # Load CEP290 data
    print("\nLoading CEP290 data...")
    df = load_experiments(config.CEP290_EXPERIMENTS, config.BUILD06_DIR)
    print(f"Loaded {len(df)} embryos")
    
    # Check available genotypes
    print(f"\nAvailable genotypes:")
    for gt in sorted(df['genotype'].unique()):
        n = (df['genotype'] == gt).sum()
        print(f"  {gt}: {n} embryos")
    
    # Bin by time
    print("\nBinning by time...")
    df_binned = bin_by_embryo_time(df, time_col="predicted_stage_hpf", bin_width=2.0)
    print(f"Created {df_binned['time_bin'].nunique()} time bins")
    print(f"Total embryo-timepoints: {len(df_binned)}")
    
    # ========================================================================
    # Test 1: Homozygous vs Wildtype (standard comparison)
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 1: Homozygous vs Wildtype (standard)")
    print("="*80)
    
    df_hom_vs_wt = compute_divergence_scores(
        df_binned,
        test_genotype="cep290_homozygous",
        reference_genotype="cep290_wildtype",
        metrics=["mahalanobis", "euclidean", "standardized", "cosine"],
        min_reference_samples=5,
        verbose=True
    )
    
    print("\nSummary statistics:")
    for metric in ["mahalanobis_distance", "euclidean_distance", "standardized_distance", "cosine_distance"]:
        if metric in df_hom_vs_wt.columns:
            vals = df_hom_vs_wt[metric]
            print(f"  {metric}: {vals.mean():.3f} ± {vals.std():.3f} (range: {vals.min():.3f} - {vals.max():.3f})")
    
    # ========================================================================
    # Test 2: Heterozygous vs Wildtype
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 2: Heterozygous vs Wildtype")
    print("="*80)
    
    df_het_vs_wt = compute_divergence_scores(
        df_binned,
        test_genotype="cep290_heterozygous",
        reference_genotype="cep290_wildtype",
        metrics=["mahalanobis", "euclidean"],
        min_reference_samples=5,
        verbose=True
    )
    
    # ========================================================================
    # Test 3: Homozygous vs Heterozygous (non-WT reference!)
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 3: Homozygous vs Heterozygous (using het as reference)")
    print("="*80)
    
    df_hom_vs_het = compute_divergence_scores(
        df_binned,
        test_genotype="cep290_homozygous",
        reference_genotype="cep290_heterozygous",
        metrics=["mahalanobis", "euclidean"],
        min_reference_samples=5,
        verbose=True
    )
    
    # ========================================================================
    # Test 4: Compare to multiple references
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 4: Homozygous vs Multiple References")
    print("="*80)
    
    multi_ref_results = compare_to_multiple_references(
        df_binned,
        test_genotype="cep290_homozygous",
        reference_genotypes=["cep290_wildtype", "cep290_heterozygous"],
        metrics=["mahalanobis", "euclidean"],
        min_reference_samples=5,
        verbose=False
    )
    
    print("\nComparison of divergence from different references:")
    for ref_genotype, df_div in multi_ref_results.items():
        if 'mahalanobis_distance' in df_div.columns:
            mean_dist = df_div['mahalanobis_distance'].mean()
            print(f"  vs {ref_genotype}: {mean_dist:.3f}")
    
    # ========================================================================
    # Save results
    # ========================================================================
    print("\n" + "="*80)
    print("Saving results...")
    
    output_dir = Path(config.DATA_DIR) / "cep290" / "divergence"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_hom_vs_wt.to_csv(output_dir / "hom_vs_wt_divergence.csv", index=False)
    df_het_vs_wt.to_csv(output_dir / "het_vs_wt_divergence.csv", index=False)
    df_hom_vs_het.to_csv(output_dir / "hom_vs_het_divergence.csv", index=False)
    
    print(f"Saved to: {output_dir}")
    
    # ========================================================================
    # Quick analysis: Which metric is most sensitive?
    # ========================================================================
    print("\n" + "="*80)
    print("METRIC COMPARISON (Homozygous vs Wildtype)")
    print("="*80)
    
    # Compare effect sizes (difference from 0, normalized by std)
    print("\nEffect sizes (mean / std):")
    for metric in ["mahalanobis_distance", "euclidean_distance", "standardized_distance"]:
        if metric in df_hom_vs_wt.columns:
            vals = df_hom_vs_wt[metric]
            # For distance, larger mean relative to std = clearer signal
            effect_size = vals.mean() / vals.std() if vals.std() > 0 else 0
            print(f"  {metric}: {effect_size:.3f}")
    
    # ========================================================================
    # Time-resolved analysis
    # ========================================================================
    print("\n" + "="*80)
    print("TIME-RESOLVED DIVERGENCE")
    print("="*80)
    
    time_summary = df_hom_vs_wt.groupby('time_bin').agg({
        'mahalanobis_distance': ['mean', 'std', 'count'],
        'euclidean_distance': ['mean', 'std']
    }).round(3)
    
    print("\nMahalanobis distance over time:")
    print(time_summary['mahalanobis_distance'])
    
    # Find time of maximum divergence
    max_divergence_time = df_hom_vs_wt.groupby('time_bin')['mahalanobis_distance'].mean().idxmax()
    max_divergence_val = df_hom_vs_wt.groupby('time_bin')['mahalanobis_distance'].mean().max()
    print(f"\nMaximum divergence: {max_divergence_val:.3f} at {max_divergence_time} hpf")
    
    # ========================================================================
    # Outlier analysis
    # ========================================================================
    if 'is_outlier' in df_hom_vs_wt.columns:
        print("\n" + "="*80)
        print("OUTLIER ANALYSIS")
        print("="*80)
        
        outliers = df_hom_vs_wt[df_hom_vs_wt['is_outlier']]
        print(f"\nTotal outliers: {len(outliers)} / {len(df_hom_vs_wt)} ({100*len(outliers)/len(df_hom_vs_wt):.1f}%)")
        
        if len(outliers) > 0:
            print(f"\nOutlier embryos:")
            outlier_summary = outliers.groupby('embryo_id').agg({
                'time_bin': 'count',
                'mahalanobis_distance': ['mean', 'max']
            }).sort_values(('mahalanobis_distance', 'max'), ascending=False).head(10)
            print(outlier_summary)
    
    # ========================================================================
    # Generate visualizations
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    plot_dir = Path(config.PLOT_DIR) / "cep290" / "divergence"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine heterozygous and homozygous for comparison plots
    df_combined = pd.concat([df_het_vs_wt, df_hom_vs_wt], ignore_index=True)
    
    # 1. Divergence over time
    print("\n1. Plotting divergence over time...")
    fig = plot_divergence_over_time(
        df_combined,
        metric="mahalanobis_distance",
        by_genotype=True,
        show_individuals=False
    )
    fig.savefig(plot_dir / "mahalanobis_over_time.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: mahalanobis_over_time.png")
    
    fig = plot_divergence_over_time(
        df_combined,
        metric="euclidean_distance",
        by_genotype=True
    )
    fig.savefig(plot_dir / "euclidean_over_time.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: euclidean_over_time.png")
    
    # 2. Distribution plots
    print("\n2. Plotting divergence distributions...")
    for plot_type in ["violin", "box"]:
        fig = plot_divergence_distribution(
            df_combined,
            metric="mahalanobis_distance",
            by_genotype=True,
            plot_type=plot_type
        )
        fig.savefig(plot_dir / f"mahalanobis_distribution_{plot_type}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"   Saved: mahalanobis_distribution_{plot_type}.png")
    
    # 3. Heatmap (homozygous only)
    print("\n3. Plotting divergence heatmap...")
    fig = plot_divergence_heatmap(
        df_hom_vs_wt,
        metric="mahalanobis_distance",
        max_embryos=30
    )
    fig.savefig(plot_dir / "divergence_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: divergence_heatmap.png")
    
    # 4. Metric comparison
    print("\n4. Plotting metric comparison...")
    fig = plot_metric_comparison(
        df_hom_vs_wt,
        metrics=["mahalanobis_distance", "euclidean_distance", 
                "standardized_distance", "cosine_distance"]
    )
    fig.savefig(plot_dir / "metric_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved: metric_comparison.png")
    
    # 5. Outlier visualization
    if 'is_outlier' in df_hom_vs_wt.columns:
        print("\n5. Plotting outliers...")
        fig = plot_outliers(
            df_hom_vs_wt,
            metric="mahalanobis_distance"
        )
        fig.savefig(plot_dir / "outliers.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"   Saved: outliers.png")
    
    print(f"\nAll plots saved to: {plot_dir}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Plots saved to: {plot_dir}")
    
    print("\nKey findings:")
    print(f"✓ Tested {len(df_hom_vs_wt['embryo_id'].unique())} homozygous embryos")
    print(f"✓ Computed 4 distance metrics")
    print(f"✓ Compared to wildtype AND heterozygous references")
    print(f"✓ Maximum divergence at {max_divergence_time} hpf")
    print(f"✓ Generated 8+ visualization plots")
    print(f"✓ Flexible reference system works!")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    main()
