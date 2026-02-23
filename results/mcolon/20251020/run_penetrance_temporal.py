#!/usr/bin/env python3
"""
Temporal Analysis: Time-Dependent Penetrance

Investigates how the distance-probability relationship changes over developmental
time to identify when phenotypes emerge and whether cutoffs should be time-dependent.

This addresses the hypothesis that early timepoints (<30 hpf) may have weak phenotypes,
causing breakdown of the linear relationship in aggregate analyses.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "20251016"))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import penetrance analysis tools
from penetrance_analysis import (
    # Temporal analysis
    compute_per_bin_regression,
    compute_sliding_window_regression,
    fit_interaction_model,
    identify_penetrance_onset,
    compute_temporal_cutoffs,
    compare_early_vs_late,
    # Temporal visualizations
    plot_correlation_over_time,
    plot_r_squared_evolution,
    plot_slope_evolution,
    plot_temporal_cutoffs,
    plot_scatter_by_timebin,
    plot_distance_time_heatmap,
    plot_interaction_model,
    plot_penetrance_onset
)


def analyze_genotype_temporal(
    genotype: str,
    data_dir: Path,
    plot_dir: Path,
    time_cutoff: float = 30.0,
    min_samples_per_bin: int = 10
) -> dict:
    """
    Run temporal analysis for one genotype.

    Parameters
    ----------
    genotype : str
        Genotype name (e.g., 'cep290_homozygous')
    data_dir : Path
        Directory with penetrance data from Step 1
    plot_dir : Path
        Directory for output plots
    time_cutoff : float
        Time threshold for early vs late comparison
    min_samples_per_bin : int
        Minimum samples per time bin

    Returns
    -------
    dict
        Temporal analysis results
    """
    print("=" * 80)
    print(f"TEMPORAL ANALYSIS: {genotype.upper()}")
    print("=" * 80)

    # ========================================
    # LOAD DATA FROM STEP 1
    # ========================================
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    # Extract genotype family (e.g., 'cep290' from 'cep290_homozygous')
    genotype_family = genotype.split('_')[0]

    distances_file = data_dir / f"{genotype_family}_distances.csv"
    predictions_file = data_dir / f"{genotype_family}_predictions.csv"

    print(f"\nLoading distances: {distances_file}")
    print(f"Loading predictions: {predictions_file}")

    if not distances_file.exists() or not predictions_file.exists():
        raise FileNotFoundError(f"Step 1 outputs not found for {genotype}. Expected: {distances_file} and {predictions_file}")

    df_distances = pd.read_csv(distances_file)
    df_predictions = pd.read_csv(predictions_file)

    # Find the mutant probability column
    prob_cols = [c for c in df_predictions.columns if c.startswith('pred_proba_') and genotype in c]
    if len(prob_cols) == 0:
        raise ValueError(f"Cannot find probability column for {genotype}")

    prob_col = prob_cols[0]
    print(f"Using probability column: {prob_col}")

    # Rename for convenience
    df_predictions['pred_prob_mutant'] = df_predictions[prob_col]

    print(f"\n  Distances: {len(df_distances)} timepoints")
    print(f"  Predictions: {len(df_predictions)} timepoints")
    print(f"  Time range: {df_distances['time_bin'].min():.1f} - {df_distances['time_bin'].max():.1f} hpf")

    results = {'genotype': genotype}

    # ========================================
    # PER-BIN REGRESSION
    # ========================================
    print("\n" + "=" * 80)
    print("PER-TIME-BIN REGRESSION ANALYSIS")
    print("=" * 80)

    temporal_results = compute_per_bin_regression(
        df_distances,
        df_predictions,
        genotype,
        min_samples=min_samples_per_bin
    )

    print(f"\n  Analyzed {len(temporal_results)} time bins")
    print(f"  Time range: {temporal_results['time_bin'].min():.1f} - {temporal_results['time_bin'].max():.1f} hpf")

    # Summary statistics
    print(f"\n  R² range: {temporal_results['r_squared'].min():.3f} - {temporal_results['r_squared'].max():.3f}")
    print(f"  Mean R²: {temporal_results['r_squared'].mean():.3f}")
    print(f"  Slope range: {temporal_results['beta1'].min():.4f} - {temporal_results['beta1'].max():.4f}")
    print(f"  Mean slope: {temporal_results['beta1'].mean():.4f}")

    # Save temporal results
    temporal_file = data_dir / "temporal" / f"{genotype}_temporal_regression.csv"
    temporal_file.parent.mkdir(parents=True, exist_ok=True)
    temporal_results.to_csv(temporal_file, index=False)
    print(f"\nSaved: {temporal_file}")

    results['temporal_regression'] = temporal_results

    # ========================================
    # PENETRANCE ONSET DETECTION
    # ========================================
    print("\n" + "=" * 80)
    print("PENETRANCE ONSET DETECTION")
    print("=" * 80)

    onset_dict = identify_penetrance_onset(
        temporal_results,
        r_squared_threshold=0.3,
        slope_threshold=0.05,
        pval_threshold=0.05
    )

    print(f"\n  Onset time: {onset_dict['onset_time']:.1f} hpf" if not np.isnan(onset_dict['onset_time']) else "\n  No clear onset detected")
    if not np.isnan(onset_dict['onset_time']):
        print(f"  R² at onset: {onset_dict['onset_r_squared']:.3f}")
        print(f"  Slope at onset: {onset_dict['onset_slope']:.4f}")
        print(f"  Bins meeting criteria: {onset_dict['n_bins_detected']}/{onset_dict['n_bins_total']}")

    results['onset'] = onset_dict

    # ========================================
    # EARLY VS LATE COMPARISON
    # ========================================
    print("\n" + "=" * 80)
    print(f"EARLY VS LATE COMPARISON (cutoff = {time_cutoff} hpf)")
    print("=" * 80)

    early_late = compare_early_vs_late(
        df_distances,
        df_predictions,
        genotype,
        time_cutoff=time_cutoff
    )

    print(f"\nEARLY (<{time_cutoff} hpf):")
    print(f"  N samples: {early_late['early']['n_samples']}")
    print(f"  N embryos: {early_late['early']['n_embryos']}")
    print(f"  Time range: {early_late['early']['time_range'][0]:.1f} - {early_late['early']['time_range'][1]:.1f} hpf")
    print(f"  Pearson r: {early_late['early']['pearson_r']:.3f} (p={early_late['early']['pearson_p']:.3e})")
    print(f"  R²: {early_late['early']['r_squared']:.3f}")
    print(f"  Slope: {early_late['early']['beta1']:.4f}")

    print(f"\nLATE (≥{time_cutoff} hpf):")
    print(f"  N samples: {early_late['late']['n_samples']}")
    print(f"  N embryos: {early_late['late']['n_embryos']}")
    print(f"  Time range: {early_late['late']['time_range'][0]:.1f} - {early_late['late']['time_range'][1]:.1f} hpf")
    print(f"  Pearson r: {early_late['late']['pearson_r']:.3f} (p={early_late['late']['pearson_p']:.3e})")
    print(f"  R²: {early_late['late']['r_squared']:.3f}")
    print(f"  Slope: {early_late['late']['beta1']:.4f}")

    # Compute improvement
    r2_improvement = early_late['late']['r_squared'] - early_late['early']['r_squared']
    slope_change = early_late['late']['beta1'] - early_late['early']['beta1']

    print(f"\nCHANGE (Late - Early):")
    print(f"  ΔR² = {r2_improvement:+.3f}")
    print(f"  ΔSlope = {slope_change:+.4f}")

    if r2_improvement > 0.1:
        print(f"  → Substantial improvement in model fit at later timepoints")
    elif r2_improvement < -0.1:
        print(f"  → Model fit actually worse at later timepoints (unexpected)")
    else:
        print(f"  → Model fit relatively constant across development")

    results['early_late'] = early_late

    # ========================================
    # INTERACTION MODEL
    # ========================================
    print("\n" + "=" * 80)
    print("INTERACTION MODEL: Distance × Time")
    print("=" * 80)

    interaction_results, interaction_summary = fit_interaction_model(
        df_distances,
        df_predictions,
        genotype
    )

    print(f"\nInteraction Model Results:")
    print(f"  β₀ (Intercept) = {interaction_summary['beta0']:.4f} ± {interaction_summary['beta0_se']:.4f}")
    print(f"  β₁ (Distance) = {interaction_summary['beta1']:.4f} ± {interaction_summary['beta1_se']:.4f}")
    print(f"  β₂ (Time) = {interaction_summary['beta2']:.4f} ± {interaction_summary['beta2_se']:.4f}")
    print(f"  β₃ (Distance × Time) = {interaction_summary['beta3']:.4f} ± {interaction_summary['beta3_se']:.4f}")
    print(f"  β₃ p-value = {interaction_summary['beta3_pval']:.3e}")
    print(f"  R² = {interaction_summary['r_squared']:.3f}")

    if interaction_summary['beta3_pval'] < 0.05:
        print(f"\n  ✓ SIGNIFICANT INTERACTION (p < 0.05)")
        if interaction_summary['beta3'] > 0:
            print(f"  → Distance effect STRENGTHENS over time")
        else:
            print(f"  → Distance effect WEAKENS over time (unusual)")
    else:
        print(f"\n  ✗ No significant interaction (p ≥ 0.05)")
        print(f"  → Distance effect is CONSTANT across development")

    # Save interaction model
    interaction_file = data_dir / "temporal" / f"{genotype}_interaction_model.csv"
    pd.DataFrame([interaction_summary]).to_csv(interaction_file, index=False)
    print(f"\nSaved: {interaction_file}")

    results['interaction'] = interaction_summary

    # ========================================
    # TEMPORAL CUTOFFS
    # ========================================
    print("\n" + "=" * 80)
    print("TIME-DEPENDENT PENETRANCE CUTOFFS")
    print("=" * 80)

    temporal_cutoffs = compute_temporal_cutoffs(temporal_results)

    print(f"\n  Valid cutoffs for {len(temporal_cutoffs)} time bins")
    print(f"  Cutoff range: {temporal_cutoffs['d_star'].min():.3f} - {temporal_cutoffs['d_star'].max():.3f}")
    print(f"  Mean cutoff: {temporal_cutoffs['d_star'].mean():.3f}")

    # Save cutoffs
    cutoffs_file = data_dir / "temporal" / f"{genotype}_temporal_cutoffs.csv"
    temporal_cutoffs.to_csv(cutoffs_file, index=False)
    print(f"Saved: {cutoffs_file}")

    results['temporal_cutoffs'] = temporal_cutoffs

    # ========================================
    # GENERATE PLOTS
    # ========================================
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)

    onset_time = onset_dict['onset_time'] if not np.isnan(onset_dict['onset_time']) else None

    # 1. Correlation over time
    print("\n  Creating correlation over time plot...")
    fig = plot_correlation_over_time(
        temporal_results,
        genotype,
        onset_time=onset_time,
        save_path=plot_dir / f"{genotype}_correlation_over_time.png"
    )
    plt.close(fig)

    # 2. Slope evolution
    print("  Creating slope evolution plot...")
    fig = plot_slope_evolution(
        temporal_results,
        genotype,
        onset_time=onset_time,
        save_path=plot_dir / f"{genotype}_slope_evolution.png"
    )
    plt.close(fig)

    # 3. Penetrance onset
    print("  Creating penetrance onset plot...")
    fig = plot_penetrance_onset(
        temporal_results,
        onset_dict,
        genotype,
        save_path=plot_dir / f"{genotype}_penetrance_onset.png"
    )
    plt.close(fig)

    # 4. Scatter by time bin (select subset)
    print("  Creating scatter by time bin plot...")
    # Select evenly spaced bins (max 12)
    all_bins = sorted(temporal_results['time_bin'].unique())
    if len(all_bins) > 12:
        step = len(all_bins) // 12
        bins_to_plot = all_bins[::step]
    else:
        bins_to_plot = all_bins

    fig = plot_scatter_by_timebin(
        df_distances,
        df_predictions,
        temporal_results,
        genotype,
        bins_to_plot=bins_to_plot,
        save_path=plot_dir / f"{genotype}_scatter_by_timebin.png"
    )
    plt.close(fig)

    # 5. Distance × Time heatmap
    print("  Creating distance-time heatmap...")
    fig = plot_distance_time_heatmap(
        df_distances,
        df_predictions,
        genotype,
        save_path=plot_dir / f"{genotype}_heatmap_distance_time.png"
    )
    plt.close(fig)

    # 6. Interaction model
    print("  Creating interaction model plot...")
    fig = plot_interaction_model(
        df_distances,
        df_predictions,
        interaction_results,
        interaction_summary,
        genotype,
        save_path=plot_dir / f"{genotype}_interaction_model.png"
    )
    plt.close(fig)

    print("\n  All plots generated successfully!")

    return results


def main():
    """Run temporal analysis for both CEP290 and TMEM67."""

    output_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251020")
    data_dir = output_dir / "data" / "penetrance"
    plot_dir = output_dir / "plots" / "penetrance" / "temporal"

    # Create plot directory
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("TEMPORAL ANALYSIS: TIME-DEPENDENT PENETRANCE")
    print("=" * 80)
    print(f"\nData directory: {data_dir}")
    print(f"Plot directory: {plot_dir}")

    all_results = {}
    temporal_results_dict = {}

    # ========================================
    # ANALYZE CEP290
    # ========================================
    cep290_results = analyze_genotype_temporal(
        genotype='cep290_homozygous',
        data_dir=data_dir,
        plot_dir=plot_dir,
        time_cutoff=30.0
    )
    all_results['cep290_homozygous'] = cep290_results
    temporal_results_dict['cep290_homozygous'] = cep290_results['temporal_regression']

    # ========================================
    # ANALYZE TMEM67
    # ========================================
    tmem67_results = analyze_genotype_temporal(
        genotype='tmem67_homozygous',
        data_dir=data_dir,
        plot_dir=plot_dir,
        time_cutoff=30.0
    )
    all_results['tmem67_homozygous'] = tmem67_results
    temporal_results_dict['tmem67_homozygous'] = tmem67_results['temporal_regression']

    # ========================================
    # CROSS-GENOTYPE COMPARISONS
    # ========================================
    print("\n" + "=" * 80)
    print("CROSS-GENOTYPE COMPARISON PLOTS")
    print("=" * 80)

    # R² evolution comparison
    print("\n  Creating R² evolution comparison...")
    onset_times = {
        'cep290_homozygous': all_results['cep290_homozygous']['onset']['onset_time'],
        'tmem67_homozygous': all_results['tmem67_homozygous']['onset']['onset_time']
    }

    fig = plot_r_squared_evolution(
        temporal_results_dict,
        onset_times=onset_times,
        save_path=plot_dir / "r_squared_evolution_comparison.png"
    )
    plt.close(fig)

    # Temporal cutoffs comparison
    print("  Creating temporal cutoffs comparison...")
    cutoffs_dict = {
        'cep290_homozygous': all_results['cep290_homozygous']['temporal_cutoffs'],
        'tmem67_homozygous': all_results['tmem67_homozygous']['temporal_cutoffs']
    }

    fig = plot_temporal_cutoffs(
        cutoffs_dict,
        save_path=plot_dir / "cutoffs_evolution_comparison.png"
    )
    plt.close(fig)

    # ========================================
    # SUMMARY STATISTICS
    # ========================================
    print("\n" + "=" * 80)
    print("CREATING SUMMARY")
    print("=" * 80)

    # Create summary DataFrame
    summary_rows = []

    for genotype, results in all_results.items():
        # Aggregate statistics
        temporal_df = results['temporal_regression']

        summary_rows.append({
            'genotype': genotype,
            'n_time_bins': len(temporal_df),
            'time_range_min': temporal_df['time_bin'].min(),
            'time_range_max': temporal_df['time_bin'].max(),
            'mean_r_squared': temporal_df['r_squared'].mean(),
            'min_r_squared': temporal_df['r_squared'].min(),
            'max_r_squared': temporal_df['r_squared'].max(),
            'mean_slope': temporal_df['beta1'].mean(),
            'min_slope': temporal_df['beta1'].min(),
            'max_slope': temporal_df['beta1'].max(),
            'onset_time': results['onset']['onset_time'],
            'onset_r_squared': results['onset']['onset_r_squared'],
            'early_r_squared': results['early_late']['early']['r_squared'],
            'late_r_squared': results['early_late']['late']['r_squared'],
            'r_squared_improvement': results['early_late']['late']['r_squared'] - results['early_late']['early']['r_squared'],
            'interaction_beta3': results['interaction']['beta3'],
            'interaction_beta3_pval': results['interaction']['beta3_pval'],
            'interaction_significant': results['interaction']['beta3_pval'] < 0.05
        })

    summary_df = pd.DataFrame(summary_rows)

    # Save summary
    summary_file = data_dir / "temporal" / "temporal_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSaved: {summary_file}")

    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "=" * 80)
    print("TEMPORAL ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print("  DATA:")
    print("    - data/penetrance/temporal/cep290_homozygous_temporal_regression.csv")
    print("    - data/penetrance/temporal/cep290_homozygous_interaction_model.csv")
    print("    - data/penetrance/temporal/cep290_homozygous_temporal_cutoffs.csv")
    print("    - data/penetrance/temporal/tmem67_homozygous_temporal_regression.csv")
    print("    - data/penetrance/temporal/tmem67_homozygous_interaction_model.csv")
    print("    - data/penetrance/temporal/tmem67_homozygous_temporal_cutoffs.csv")
    print("    - data/penetrance/temporal/temporal_summary.csv")
    print("  PLOTS:")
    print("    - plots/penetrance/temporal/{genotype}_correlation_over_time.png")
    print("    - plots/penetrance/temporal/{genotype}_slope_evolution.png")
    print("    - plots/penetrance/temporal/{genotype}_penetrance_onset.png")
    print("    - plots/penetrance/temporal/{genotype}_scatter_by_timebin.png")
    print("    - plots/penetrance/temporal/{genotype}_heatmap_distance_time.png")
    print("    - plots/penetrance/temporal/{genotype}_interaction_model.png")
    print("    - plots/penetrance/temporal/r_squared_evolution_comparison.png")
    print("    - plots/penetrance/temporal/cutoffs_evolution_comparison.png")

    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    print("=" * 80)

    for _, row in summary_df.iterrows():
        print(f"\n{row['genotype'].upper()}:")
        print(f"  Time range: {row['time_range_min']:.1f} - {row['time_range_max']:.1f} hpf ({row['n_time_bins']} bins)")
        print(f"  R² range: {row['min_r_squared']:.3f} - {row['max_r_squared']:.3f} (mean: {row['mean_r_squared']:.3f})")
        print(f"  Slope range: {row['min_slope']:.4f} - {row['max_slope']:.4f} (mean: {row['mean_slope']:.4f})")

        if not np.isnan(row['onset_time']):
            print(f"  Penetrance onset: {row['onset_time']:.1f} hpf (R²={row['onset_r_squared']:.3f})")
        else:
            print(f"  Penetrance onset: Not detected")

        print(f"  Early (<30 hpf) R²: {row['early_r_squared']:.3f}")
        print(f"  Late (≥30 hpf) R²: {row['late_r_squared']:.3f}")
        print(f"  R² improvement (late - early): {row['r_squared_improvement']:+.3f}")

        if row['interaction_significant']:
            direction = "strengthens" if row['interaction_beta3'] > 0 else "weakens"
            print(f"  Interaction: SIGNIFICANT (β₃={row['interaction_beta3']:.4f}, p={row['interaction_beta3_pval']:.3e})")
            print(f"  → Distance effect {direction} over time")
        else:
            print(f"  Interaction: Not significant (p={row['interaction_beta3_pval']:.3e})")
            print(f"  → Distance effect constant across development")

    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("=" * 80)

    # Interpretation based on findings
    cep_early = summary_df[summary_df['genotype'] == 'cep290_homozygous']['early_r_squared'].iloc[0]
    cep_late = summary_df[summary_df['genotype'] == 'cep290_homozygous']['late_r_squared'].iloc[0]
    cep_improvement = cep_late - cep_early

    if cep_improvement > 0.15:
        print("\n✓ HYPOTHESIS CONFIRMED for CEP290:")
        print(f"  Early data (<30 hpf) has weak phenotype (R²={cep_early:.3f})")
        print(f"  Late data (≥30 hpf) has strong phenotype (R²={cep_late:.3f})")
        print(f"  Improvement: {cep_improvement:+.3f}")
        print(f"  → Aggregate R²=0.32 is dragged down by early timepoints")
        print(f"  → Consider restricting analysis to post-onset timepoints")
    else:
        print("\n✗ HYPOTHESIS NOT STRONGLY SUPPORTED for CEP290:")
        print(f"  R² improvement from early to late is modest ({cep_improvement:+.3f})")
        print(f"  → CEP290's lower R² may reflect true biological variability")

    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("\nBased on temporal findings, consider:")
    print("1. Re-run Steps 1-2 with filtered data (post-onset only)")
    print("2. Use time-dependent cutoffs for Step 3 penetrance classification")
    print("3. Focus downstream analyses on developmental windows with strong signal")
    print("4. Investigate biological mechanisms of penetrance onset")


if __name__ == "__main__":
    main()
