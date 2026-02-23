#!/usr/bin/env python3
"""
B9D2 Divergence Analysis

Compare b9d2_homozygous and b9d2_heterozygous embryos to b9d2_wildtype reference.
Generates trajectory plots showing divergence from wildtype over developmental time.
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
    print("B9D2 DIVERGENCE ANALYSIS")
    print("="*80)

    # Setup directories
    data_dir = Path(config.DATA_DIR) / "b9d2" / "divergence"
    plot_dir = Path(config.PLOT_DIR) / "b9d2" / "divergence"
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # STEP 1: Load raw data
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: Loading B9D2 experiments")
    print("="*80)

    df_raw = load_experiments(
        experiment_ids=config.B9D2_EXPERIMENTS,
        build_dir=config.BUILD06_DIR,
        verbose=True
    )

    print(f"\nLoaded {len(df_raw)} timepoints from {df_raw['embryo_id'].nunique()} embryos")
    print(f"Genotypes: {df_raw['genotype'].value_counts().to_dict()}")

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

    # ========================================================================
    # STEP 3: Compute reference distribution (b9d2_wildtype)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: Computing b9d2_wildtype reference distribution")
    print("="*80)

    ref_stats = compute_reference_distribution(
        df_binned,
        reference_genotypes=['b9d2_wildtype'],
        min_samples=10
    )

    # ========================================================================
    # STEP 4: Compute divergence scores
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: Computing divergence scores")
    print("="*80)

    # Homozygous vs WT
    print("\nComputing homozygous divergence...")
    df_hom_div = compute_divergence_scores(
        df_binned,
        ref_stats,
        test_genotypes=['b9d2_homozygous']
    )
    df_hom_div.to_csv(data_dir / "hom_vs_wt_divergence.csv", index=False)
    print(f"  Saved to: {data_dir / 'hom_vs_wt_divergence.csv'}")

    # Heterozygous vs WT
    print("\nComputing heterozygous divergence...")
    df_het_div = compute_divergence_scores(
        df_binned,
        ref_stats,
        test_genotypes=['b9d2_heterozygous']
    )
    df_het_div.to_csv(data_dir / "het_vs_wt_divergence.csv", index=False)
    print(f"  Saved to: {data_dir / 'het_vs_wt_divergence.csv'}")

    # ========================================================================
    # STEP 5: Generate comparison plots
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: Generating comparison plots")
    print("="*80)

    # Combine both genotypes for comparison
    df_combined = pd.concat([df_hom_div, df_het_div], ignore_index=True)
    print(f"\nCombined dataset: {len(df_combined)} timepoints")
    print(f"  Genotypes: {df_combined['genotype'].unique().tolist()}")

    # Side-by-side comparison - Mahalanobis
    print("\nGenerating Mahalanobis side-by-side plot...")
    fig = plot_genotype_comparison(
        df_combined=df_combined,
        metric="mahalanobis_distance",
        remove_outliers=True,
        outlier_percentile=99.0,
        figsize=(18, 8),
        plot_type="side_by_side"
    )
    fig.savefig(plot_dir / "genotype_comparison_mahalanobis_sidebyside.png", dpi=300, bbox_inches='tight')
    print(f"  Saved to: {plot_dir / 'genotype_comparison_mahalanobis_sidebyside.png'}")

    # Overlay comparison - Mahalanobis
    print("\nGenerating Mahalanobis overlay plot...")
    fig = plot_genotype_comparison(
        df_combined=df_combined,
        metric="mahalanobis_distance",
        remove_outliers=True,
        outlier_percentile=99.0,
        figsize=(14, 8),
        plot_type="overlay"
    )
    fig.savefig(plot_dir / "genotype_comparison_mahalanobis_overlay.png", dpi=300, bbox_inches='tight')
    print(f"  Saved to: {plot_dir / 'genotype_comparison_mahalanobis_overlay.png'}")

    # Side-by-side comparison - Euclidean
    print("\nGenerating Euclidean side-by-side plot...")
    fig = plot_genotype_comparison(
        df_combined=df_combined,
        metric="euclidean_distance",
        remove_outliers=True,
        outlier_percentile=99.0,
        figsize=(18, 8),
        plot_type="side_by_side"
    )
    fig.savefig(plot_dir / "genotype_comparison_euclidean_sidebyside.png", dpi=300, bbox_inches='tight')
    print(f"  Saved to: {plot_dir / 'genotype_comparison_euclidean_sidebyside.png'}")

    # Overlay comparison - Euclidean
    print("\nGenerating Euclidean overlay plot...")
    fig = plot_genotype_comparison(
        df_combined=df_combined,
        metric="euclidean_distance",
        remove_outliers=True,
        outlier_percentile=99.0,
        figsize=(14, 8),
        plot_type="overlay"
    )
    fig.savefig(plot_dir / "genotype_comparison_euclidean_overlay.png", dpi=300, bbox_inches='tight')
    print(f"  Saved to: {plot_dir / 'genotype_comparison_euclidean_overlay.png'}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("B9D2 ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nData saved to: {data_dir}")
    print(f"Plots saved to: {plot_dir}")
    print("\nGenerated files:")
    print("  DATA:")
    print("    - binned_data.csv")
    print("    - hom_vs_wt_divergence.csv")
    print("    - het_vs_wt_divergence.csv")
    print("  PLOTS:")
    print("    - genotype_comparison_mahalanobis_sidebyside.png")
    print("    - genotype_comparison_mahalanobis_overlay.png")
    print("    - genotype_comparison_euclidean_sidebyside.png")
    print("    - genotype_comparison_euclidean_overlay.png")


if __name__ == "__main__":
    main()
