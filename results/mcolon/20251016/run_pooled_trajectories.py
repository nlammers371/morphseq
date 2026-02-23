#!/usr/bin/env python3
"""
Pooled Cross-Gene Divergence Analysis

Compares ALL homozygous mutants (cep290, b9d2, tmem67) vs ALL heterozygous mutants
to a SHARED wildtype reference pool (wik, wik-ab, ab).

Purpose: Test whether homozygotes show a shared phenotype vs wildtype,
and likewise for heterozygotes, regardless of which specific gene is mutated.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import config_new as config
from utils.data_loading import load_experiments
from utils.binning import bin_embryos_by_time
from divergence_analysis.workflow import (
    compute_reference_distribution,
    compute_divergence_scores
)
from plot_embryo_trajectories import plot_genotype_comparison


def main():
    print("="*80)
    print("POOLED CROSS-GENE DIVERGENCE ANALYSIS")
    print("="*80)
    print("\nHypothesis: All homozygous mutants (regardless of gene) share a")
    print("common phenotype vs wildtype. Same for heterozygotes.")
    print("\nComparisons:")
    print("  1. ALL homozygotes (cep290, b9d2, tmem67) vs SHARED WT pool")
    print("  2. ALL heterozygotes (cep290, b9d2, tmem67) vs SHARED WT pool")
    print("  Shared WT pool: wik, wik-ab, ab")

    # Setup directories
    data_dir = Path(config.DATA_DIR) / "pooled" / "divergence"
    plot_dir = Path(config.PLOT_DIR) / "pooled" / "divergence"
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # STEP 1: Load ALL experiments
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: Loading ALL experiments")
    print("="*80)

    df_raw = load_experiments(
        experiment_ids=config.ALL_EXPERIMENTS,
        build_dir=config.BUILD06_DIR,
        verbose=True
    )

    print(f"\nLoaded {len(df_raw)} timepoints from {df_raw['embryo_id'].nunique()} embryos")
    print(f"All genotypes: {df_raw['genotype'].value_counts().to_dict()}")

    # ========================================================================
    # STEP 2: Bin by developmental time
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: Binning by developmental time")
    print("="*80)

    df_binned = bin_embryos_by_time(df_raw, bin_width=2.0)
    df_binned.to_csv(data_dir / "binned_data.csv", index=False)

    print(f"Binned data saved to: {data_dir / 'binned_data.csv'}")
    print(f"  {len(df_binned)} embryo-timepoints")
    print(f"  {df_binned['embryo_id'].nunique()} unique embryos")
    print(f"  Time range: {df_binned['time_bin'].min()} - {df_binned['time_bin'].max()} hpf")
    print(f"  Genotypes: {df_binned['genotype'].value_counts().to_dict()}")

    # ========================================================================
    # STEP 3: Compute SHARED WT reference (pool wik, wik-ab, ab)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: Computing SHARED wildtype reference distribution")
    print("="*80)

    ref_stats = compute_reference_distribution(
        df_binned,
        reference_genotypes=['wik', 'wik-ab', 'ab'],  # Pooled WT
        min_samples=10
    )

    # ========================================================================
    # STEP 4: Compute divergence scores for pooled genotypes
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: Computing divergence scores")
    print("="*80)

    # All homozygotes vs shared WT
    print("\nComputing ALL homozygotes divergence...")
    df_all_hom_div = compute_divergence_scores(
        df_binned,
        ref_stats,
        test_genotypes=['cep290_homozygous', 'b9d2_homozygous', 'tmem67_homozygous']
    )
    df_all_hom_div.to_csv(data_dir / "all_hom_vs_shared_wt_divergence.csv", index=False)
    print(f"  Saved to: {data_dir / 'all_hom_vs_shared_wt_divergence.csv'}")

    # All heterozygotes vs shared WT
    print("\nComputing ALL heterozygotes divergence...")
    df_all_het_div = compute_divergence_scores(
        df_binned,
        ref_stats,
        test_genotypes=['cep290_heterozygous', 'b9d2_heterozygous', 'tmem67_heterozygote']
    )
    df_all_het_div.to_csv(data_dir / "all_het_vs_shared_wt_divergence.csv", index=False)
    print(f"  Saved to: {data_dir / 'all_het_vs_shared_wt_divergence.csv'}")

    # ========================================================================
    # STEP 5: Generate comparison plots
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: Generating comparison plots")
    print("="*80)

    # Combine homozygotes and heterozygotes for comparison
    df_combined = pd.concat([df_all_hom_div, df_all_het_div], ignore_index=True)
    print(f"\nCombined dataset: {len(df_combined)} timepoints")
    print(f"  Genotypes: {df_combined['genotype'].unique().tolist()}")

    # Create a simplified genotype label for plotting
    # Map individual genotypes to 'homozygous' or 'heterozygous'
    genotype_map = {
        'cep290_homozygous': 'All Homozygous',
        'b9d2_homozygous': 'All Homozygous',
        'tmem67_homozygous': 'All Homozygous',
        'cep290_heterozygous': 'All Heterozygous',
        'b9d2_heterozygous': 'All Heterozygous',
        'tmem67_heterozygote': 'All Heterozygous'
    }
    df_combined['genotype_group'] = df_combined['genotype'].map(genotype_map)

    # Side-by-side comparison - Mahalanobis
    print("\nGenerating Mahalanobis side-by-side plot...")
    # Note: plot_genotype_comparison expects specific genotype names
    # We'll create separate plots for the pooled analysis

    # Plot 1: Homozygotes only (all genes combined)
    print("\nPlotting all homozygotes...")
    fig = plot_genotype_comparison(
        df_combined=df_all_hom_div,
        metric="mahalanobis_distance",
        remove_outliers=True,
        outlier_percentile=99.0,
        figsize=(14, 8),
        plot_type="overlay"
    )
    fig.savefig(plot_dir / "all_homozygotes_mahalanobis.png", dpi=300, bbox_inches='tight')
    print(f"  Saved to: {plot_dir / 'all_homozygotes_mahalanobis.png'}")

    # Plot 2: Heterozygotes only (all genes combined)
    print("\nPlotting all heterozygotes...")
    fig = plot_genotype_comparison(
        df_combined=df_all_het_div,
        metric="mahalanobis_distance",
        remove_outliers=True,
        outlier_percentile=99.0,
        figsize=(14, 8),
        plot_type="overlay"
    )
    fig.savefig(plot_dir / "all_heterozygotes_mahalanobis.png", dpi=300, bbox_inches='tight')
    print(f"  Saved to: {plot_dir / 'all_heterozygotes_mahalanobis.png'}")

    # Plot 3: Direct comparison using simplified labels
    # We need to modify the genotype column temporarily for the plot
    df_plot = df_combined.copy()
    df_plot['genotype'] = df_plot['genotype_group']

    print("\nGenerating pooled comparison plots...")

    # Mahalanobis side-by-side
    fig = plot_genotype_comparison(
        df_combined=df_plot,
        metric="mahalanobis_distance",
        remove_outliers=True,
        outlier_percentile=99.0,
        figsize=(18, 8),
        plot_type="side_by_side"
    )
    fig.savefig(plot_dir / "pooled_comparison_mahalanobis_sidebyside.png", dpi=300, bbox_inches='tight')
    print(f"  Saved to: {plot_dir / 'pooled_comparison_mahalanobis_sidebyside.png'}")

    # Mahalanobis overlay
    fig = plot_genotype_comparison(
        df_combined=df_plot,
        metric="mahalanobis_distance",
        remove_outliers=True,
        outlier_percentile=99.0,
        figsize=(14, 8),
        plot_type="overlay"
    )
    fig.savefig(plot_dir / "pooled_comparison_mahalanobis_overlay.png", dpi=300, bbox_inches='tight')
    print(f"  Saved to: {plot_dir / 'pooled_comparison_mahalanobis_overlay.png'}")

    # Euclidean side-by-side
    fig = plot_genotype_comparison(
        df_combined=df_plot,
        metric="euclidean_distance",
        remove_outliers=True,
        outlier_percentile=99.0,
        figsize=(18, 8),
        plot_type="side_by_side"
    )
    fig.savefig(plot_dir / "pooled_comparison_euclidean_sidebyside.png", dpi=300, bbox_inches='tight')
    print(f"  Saved to: {plot_dir / 'pooled_comparison_euclidean_sidebyside.png'}")

    # Euclidean overlay
    fig = plot_genotype_comparison(
        df_combined=df_plot,
        metric="euclidean_distance",
        remove_outliers=True,
        outlier_percentile=99.0,
        figsize=(14, 8),
        plot_type="overlay"
    )
    fig.savefig(plot_dir / "pooled_comparison_euclidean_overlay.png", dpi=300, bbox_inches='tight')
    print(f"  Saved to: {plot_dir / 'pooled_comparison_euclidean_overlay.png'}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("POOLED CROSS-GENE ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nData saved to: {data_dir}")
    print(f"Plots saved to: {plot_dir}")
    print("\nGenerated files:")
    print("  DATA:")
    print("    - binned_data.csv")
    print("    - all_hom_vs_shared_wt_divergence.csv")
    print("    - all_het_vs_shared_wt_divergence.csv")
    print("  PLOTS:")
    print("    - all_homozygotes_mahalanobis.png")
    print("    - all_heterozygotes_mahalanobis.png")
    print("    - pooled_comparison_mahalanobis_sidebyside.png")
    print("    - pooled_comparison_mahalanobis_overlay.png")
    print("    - pooled_comparison_euclidean_sidebyside.png")
    print("    - pooled_comparison_euclidean_overlay.png")
    print("\nKey findings:")
    print("  - Tests if homozygotes (across genes) share phenotype vs WT")
    print("  - Tests if heterozygotes (across genes) share phenotype vs WT")
    print("  - Uses shared WT pool (wik, wik-ab, ab) as reference")


if __name__ == "__main__":
    main()
