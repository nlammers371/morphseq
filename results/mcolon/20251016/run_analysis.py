#!/usr/bin/env python3
"""
Main analysis script for robust phenotype emergence classification.

This script orchestrates the complete analysis pipeline:
1. Load experimental data
2. Bin by embryo and time
3. Run pairwise classification tests
4. Compute penetrance metrics
5. Generate visualizations
6. Save results

Usage:
    python run_analysis.py

    # Or with custom parameters
    MORPHSEQ_N_PERMUTATIONS=500 python run_analysis.py
"""

import os
import sys
from itertools import combinations

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils import (
    load_experiments,
    filter_by_genotypes,
    bin_by_embryo_time,
    make_safe_comparison_name,
    get_plot_path,
    get_data_path,
    save_dataframe
)
from classification import (
    predictive_signal_test,
    compute_embryo_penetrance
)
from visualization import (
    plot_auroc_over_time,
    plot_auroc_with_significance,
    plot_signed_margin_trajectories,
    plot_signed_margin_heatmap,
    plot_penetrance_distribution,
    summarize_significant_bins
)


def main():
    """Run the complete phenotype emergence analysis pipeline."""

    # Print configuration
    config.print_config()

    # Create output directories
    config.make_dirs()

    print("\n" + "="*80)
    print("LOADING EXPERIMENTAL DATA")
    print("="*80)

    # Load all experiments
    combined_df = load_experiments(
        config.ALL_EXPERIMENTS,
        config.BUILD06_DIR,
        verbose=True
    )

    print("\n" + "="*80)
    print("PREDICTIVE CLASSIFICATION ANALYSIS")
    print("="*80)

    # Analyze each gene family
    for gene_label, genotype_values in config.GENOTYPE_GROUPS.items():
        print("\n" + "="*80)
        print(f"ANALYSIS FOR {gene_label.upper()}")
        print("="*80)

        # Filter to genotype family
        df_family = filter_by_genotypes(combined_df, genotype_values, verbose=True)

        if df_family.empty:
            print(f"No data found for genotype group '{gene_label}', skipping.")
            continue

        # Bin embeddings by embryo and time
        print("\nBinning embeddings by embryo and time...")
        df_binned = bin_by_embryo_time(
            df_family,
            time_col=config.TIME_COLUMN,
            bin_width=config.TIME_BIN_WIDTH
        )
        print(f"Binned data: {len(df_binned)} rows")

        # Drop rows with NaN in latent columns
        binned_z_cols = [c for c in df_binned.columns if "_binned" in c]
        nan_counts = df_binned[binned_z_cols].isna().sum()
        if nan_counts.sum() > 0:
            dropped = df_binned[binned_z_cols].isna().any(axis=1).sum()
            print(f"Dropping {dropped} rows with NaNs in latent columns")
            df_binned = df_binned.dropna(subset=binned_z_cols)
            print(f"Remaining rows: {len(df_binned)}")

        # Get all genotypes present in the data
        present_genotypes = [g for g in genotype_values if g in df_binned['genotype'].unique()]
        print(f"\nPresent genotypes: {present_genotypes}")

        # Run pairwise comparisons
        pairwise_comparisons = list(combinations(present_genotypes, 2))
        print(f"Running {len(pairwise_comparisons)} pairwise comparisons...")

        all_results = []

        for idx, (group1, group2) in enumerate(pairwise_comparisons, 1):
            print(f"\n[{idx}/{len(pairwise_comparisons)}] Comparing: {group1} vs {group2}")

            # Filter to just these two genotypes
            df_pair = df_binned[df_binned['genotype'].isin([group1, group2])].copy()

            if len(df_pair) < config.MIN_SAMPLES_TOTAL:
                print(f"  Skipping: insufficient data ({len(df_pair)} samples)")
                continue

            # Run predictive signal test
            print(f"  Running predictive signal test...")
            df_auc, df_embryo_probs = predictive_signal_test(
                df_pair,
                group_col="genotype",
                n_splits=config.N_CV_SPLITS,
                n_perm=config.N_PERMUTATIONS,
                random_state=config.RANDOM_SEED,
                return_embryo_probs=True,
                use_class_weights=config.USE_CLASS_WEIGHTS  # Default: True
            )

            if df_auc.empty:
                print(f"  No valid time bins for this comparison")
                continue

            # Add comparison info
            df_auc['group1'] = group1
            df_auc['group2'] = group2
            df_auc['comparison'] = f"{group1}_vs_{group2}"
            all_results.append(df_auc)

            print(f"  Results for {len(df_auc)} time bins, "
                  f"{len(df_embryo_probs['embryo_id'].unique())} embryos")

            # Summarize significant bins
            sig_summary = summarize_significant_bins(df_auc, alpha=config.ALPHA)
            if sig_summary['has_significant_signal']:
                print(f"  ✓ First significant predictive signal:")
                print(f"    Time: {sig_summary['first_onset_time']} hpf")
                print(f"    AUROC: {sig_summary['first_onset_auroc']:.3f}")
                print(f"    P-value: {sig_summary['first_onset_pval']:.4f}")
            else:
                print(f"  ⚠ No significant predictive signal detected")

            # Compute embryo-level penetrance metrics
            print(f"  Computing embryo-level penetrance...")
            df_penetrance = compute_embryo_penetrance(
                df_embryo_probs,
                confidence_threshold=config.CONFIDENCE_THRESHOLD,
                penetrance_bins=config.PENETRANCE_BINS,
                penetrance_labels=config.PENETRANCE_LABELS
            )

            if not df_penetrance.empty:
                print(f"  Penetrance summary:")
                print(f"    Mean confidence: "
                      f"{df_penetrance['mean_confidence'].mean():.3f} ± "
                      f"{df_penetrance['mean_confidence'].std():.3f}")
                if 'mean_support_true' in df_penetrance.columns:
                    print(f"    Mean support (true class): "
                          f"{df_penetrance['mean_support_true'].mean():.3f} ± "
                          f"{df_penetrance['mean_support_true'].std():.3f}")
                if 'mean_signed_margin' in df_penetrance.columns:
                    print(f"    Mean signed margin: "
                          f"{df_penetrance['mean_signed_margin'].mean():.3f} ± "
                          f"{df_penetrance['mean_signed_margin'].std():.3f}")
                print(f"    Temporal consistency: "
                      f"{df_penetrance['temporal_consistency'].mean():.3f} ± "
                      f"{df_penetrance['temporal_consistency'].std():.3f}")

                # Count penetrance categories
                if 'penetrance_category' in df_penetrance.columns:
                    cat_counts = df_penetrance['penetrance_category'].value_counts()
                    print(f"    Penetrance categories: {dict(cat_counts)}")

            # Generate plots for this comparison
            print(f"  Generating plots...")

            # Create safe filename
            safe_comp_name = make_safe_comparison_name(group1, group2)

            # AUROC plots
            plot_auroc_over_time(
                df_auc,
                group1,
                group2,
                output_path=get_plot_path(config.PLOT_DIR, gene_label,
                                         'auroc', safe_comp_name)
            )

            plot_auroc_with_significance(
                df_auc,
                group1,
                group2,
                alpha=config.ALPHA,
                output_path=get_plot_path(config.PLOT_DIR, gene_label,
                                         'auroc_with_pvalues', safe_comp_name)
            )

            # Embryo-level signed margin heatmap
            plot_signed_margin_heatmap(
                df_embryo_probs,
                df_penetrance,
                group1,
                group2,
                output_path=get_plot_path(config.PLOT_DIR, gene_label,
                                         'signed_margin_heatmap', safe_comp_name)
            )

            # Signed margin trajectory plot
            if 'signed_margin' in df_embryo_probs.columns:
                plot_signed_margin_trajectories(
                    df_embryo_probs,
                    df_penetrance,
                    group1,
                    group2,
                    max_embryos=config.MAX_TRAJECTORY_EMBRYOS,
                    output_path=get_plot_path(config.PLOT_DIR, gene_label,
                                             'signed_margin_trajectories', safe_comp_name)
                )

            # Penetrance distribution plot
            plot_penetrance_distribution(
                df_penetrance,
                group1,
                group2,
                output_path=get_plot_path(config.PLOT_DIR, gene_label,
                                         'penetrance_distribution', safe_comp_name)
            )

            # Save embryo-level data
            if not df_embryo_probs.empty:
                save_dataframe(
                    df_embryo_probs,
                    get_data_path(config.DATA_DIR, gene_label,
                                 'embryo_predictions', safe_comp_name),
                    verbose=True
                )

            if not df_penetrance.empty:
                save_dataframe(
                    df_penetrance,
                    get_data_path(config.DATA_DIR, gene_label,
                                 'embryo_penetrance', safe_comp_name),
                    verbose=True
                )

        # Combine all results and save
        if all_results:
            import pandas as pd
            combined_results = pd.concat(all_results, ignore_index=True)
            save_dataframe(
                combined_results,
                get_data_path(config.DATA_DIR, gene_label,
                             'classification_results_all_comparisons'),
                verbose=True
            )

            # Print summary
            print(f"\n{'='*80}")
            print("SUMMARY OF ALL COMPARISONS")
            print(f"{'='*80}")
            for comp in combined_results['comparison'].unique():
                comp_data = combined_results[combined_results['comparison'] == comp]
                sig_count = (comp_data['pval'] < config.ALPHA).sum()
                first_sig = comp_data[comp_data['pval'] < config.ALPHA].sort_values('time_bin')
                if len(first_sig) > 0:
                    onset = first_sig.iloc[0]['time_bin']
                    auroc = first_sig.iloc[0]['AUROC_obs']
                    print(f"\n{comp}:")
                    print(f"  Onset: {onset} hpf (AUROC={auroc:.3f})")
                    print(f"  Significant bins: {sig_count}/{len(comp_data)}")
                else:
                    print(f"\n{comp}:")
                    print(f"  No significant onset detected")
        else:
            print(f"\n⚠ No valid comparisons completed for {gene_label}")

    print("\n" + "="*80)
    print("CLASSIFICATION ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {config.RESULTS_DIR}")
    print(f"  Data: {config.DATA_DIR}")
    print(f"  Plots: {config.PLOT_DIR}")


if __name__ == "__main__":
    main()
