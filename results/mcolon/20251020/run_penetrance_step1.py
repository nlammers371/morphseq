#!/usr/bin/env python3
"""
Step 1: Correlation Analysis for Incomplete Penetrance

Quantifies the relationship between morphological distance (from WT) and
classifier-based mutant probability for homozygous embryos.

Strong correlation (r > 0.5) indicates that distance is a good phenotypic readout
for identifying penetrant vs non-penetrant mutants.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "20251016"))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

# Import existing utilities
from utils.data_loading import load_experiments
from utils.binning import bin_embryos_by_time
from divergence_analysis.distances import compute_euclidean_distance
import config_new as config

# Import classifier
from classification.predictive_test import predictive_signal_test

# Import our new penetrance analysis tools
from penetrance_analysis import (
    compute_per_embryo_metrics,
    compute_correlation_statistics,
    bootstrap_correlation_ci,
    plot_distance_vs_probability,
    plot_correlation_summary
)


def compute_distances_for_genotypes(
    df_binned: pd.DataFrame,
    reference_genotypes: list,
    all_genotypes: list,
    z_cols: list = None,
    time_col: str = "time_bin",
    min_ref_samples: int = 10
) -> pd.DataFrame:
    """
    Compute Euclidean distances to WT reference for all genotypes.

    Parameters
    ----------
    df_binned : pd.DataFrame
        Binned embryo data
    reference_genotypes : list
        WT genotype(s) for reference
    all_genotypes : list
        All genotypes to compute distances for
    z_cols : list, optional
        Latent feature columns
    time_col : str
        Time bin column
    min_ref_samples : int
        Minimum reference samples per time bin

    Returns
    -------
    pd.DataFrame
        Distances with columns: embryo_id, time_bin, genotype, euclidean_distance
    """
    # Auto-detect latent columns
    if z_cols is None:
        z_cols = [c for c in df_binned.columns if c.endswith("_binned")]

    # Compute reference stats per time bin
    df_ref = df_binned[df_binned['genotype'].isin(reference_genotypes)].copy()
    print(f"\nReference: {reference_genotypes}")
    print(f"  {df_ref['embryo_id'].nunique()} embryos")

    # Get all embryos
    df_all = df_binned[df_binned['genotype'].isin(all_genotypes)].copy()
    print(f"\nAll genotypes: {all_genotypes}")
    print(f"  {df_all['embryo_id'].nunique()} embryos")

    results = []

    for time_bin in sorted(df_all[time_col].unique()):
        ref_group = df_ref[df_ref[time_col] == time_bin]
        if len(ref_group) < min_ref_samples:
            continue

        # Compute reference mean
        X_ref = ref_group[z_cols].values
        mu_ref = X_ref.mean(axis=0)

        # Get all embryos for this time bin
        all_group = df_all[df_all[time_col] == time_bin]
        X_all = all_group[z_cols].values

        # Compute Euclidean distances
        eucl_dist = compute_euclidean_distance(X_all, mu_ref)

        # Store results
        for i, (idx, row) in enumerate(all_group.iterrows()):
            results.append({
                'embryo_id': row['embryo_id'],
                'time_bin': time_bin,
                'genotype': row['genotype'],
                'euclidean_distance': eucl_dist[i]
            })

    df_distances = pd.DataFrame(results)
    print(f"\nComputed distances: {len(df_distances)} timepoints across {len(df_distances['time_bin'].unique())} bins")

    return df_distances


def analyze_penetrance_correlation(
    genotype_family: str,
    experiment_ids: list,
    reference_genotypes: list,
    homozygous_genotype: str,
    output_dir: Path,
    bin_width: float = 2.0
) -> dict:
    """
    Run complete Step 1 correlation analysis for one genotype family.

    Parameters
    ----------
    genotype_family : str
        Name (e.g., 'cep290', 'tmem67')
    experiment_ids : list
        Experiment IDs to load
    reference_genotypes : list
        WT genotypes for reference
    homozygous_genotype : str
        Homozygous genotype to analyze
    output_dir : Path
        Output directory
    bin_width : float
        Time bin width

    Returns
    -------
    dict
        Summary statistics
    """
    print("=" * 80)
    print(f"PENETRANCE CORRELATION ANALYSIS: {genotype_family.upper()}")
    print("=" * 80)

    # Create output directories
    data_dir = output_dir / "data" / "penetrance"
    plot_dir = output_dir / "plots" / "penetrance"
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # STEP 1: Load and bin data
    print("\n" + "=" * 80)
    print("STEP 1: Loading and binning data")
    print("=" * 80)

    df_raw = load_experiments(
        experiment_ids=experiment_ids,
        build_dir=config.BUILD06_DIR,
        verbose=True
    )

    df_binned = bin_embryos_by_time(df_raw, bin_width=bin_width)
    print(f"Binned: {len(df_binned)} timepoints from {df_binned['embryo_id'].nunique()} embryos")

    # STEP 2: Compute distances
    print("\n" + "=" * 80)
    print("STEP 2: Computing Euclidean distances to WT")
    print("=" * 80)

    all_genotypes = reference_genotypes + [homozygous_genotype]
    df_distances = compute_distances_for_genotypes(
        df_binned,
        reference_genotypes=reference_genotypes,
        all_genotypes=all_genotypes
    )

    # Save distances
    df_distances.to_csv(data_dir / f"{genotype_family}_distances.csv", index=False)
    print(f"Saved: {data_dir / f'{genotype_family}_distances.csv'}")

    # STEP 3: Generate classifier predictions
    print("\n" + "=" * 80)
    print("STEP 3: Generating classifier predictions (WT vs homozygous)")
    print("=" * 80)

    # Filter to only WT and homozygous for binary classification
    df_binned_binary = df_binned[df_binned['genotype'].isin(reference_genotypes + [homozygous_genotype])].copy()
    print(f"  Filtering to binary classification:")
    print(f"    WT: {reference_genotypes}")
    print(f"    Homozygous: {homozygous_genotype}")
    print(f"    Total embryos: {df_binned_binary['embryo_id'].nunique()}")

    # Run predictive test (this generates predictions)
    df_results, df_embryo_probs = predictive_signal_test(
        df_binned_binary,
        group_col="genotype",
        time_col="time_bin",
        n_splits=5,
        n_perm=100,  # Reduced for speed
        random_state=42,
        return_embryo_probs=True,
        use_class_weights=True
    )

    print(f"\n  Generated predictions for {len(df_embryo_probs)} timepoints")
    print(f"  Prediction columns: {df_embryo_probs.columns.tolist()}")

    # Find the class-labeled probability columns (new format: pred_proba_{class_name})
    prob_cols = [c for c in df_embryo_probs.columns if c.startswith('pred_proba_')]
    print(f"\n  Found {len(prob_cols)} class-labeled probability columns:")
    for col in prob_cols:
        print(f"    {col}")

    # Identify which column represents the mutant (homozygous) class
    mutant_prob_col = f'pred_proba_{homozygous_genotype}'
    if mutant_prob_col not in df_embryo_probs.columns:
        # Fallback: find the column that's NOT wildtype
        wt_prob_cols = [c for c in prob_cols if 'wildtype' in c.lower()]
        mutant_prob_cols = [c for c in prob_cols if c not in wt_prob_cols]
        if len(mutant_prob_cols) == 1:
            mutant_prob_col = mutant_prob_cols[0]
            print(f"\n  ⚠ Using {mutant_prob_col} as mutant probability column")
        else:
            raise ValueError(f"Cannot identify mutant probability column. Found: {prob_cols}")

    # Create unified column for downstream analysis
    df_embryo_probs['pred_prob_mutant'] = df_embryo_probs[mutant_prob_col]

    # Verification: Check mean probabilities per genotype
    sample_labels = df_embryo_probs['true_label'].unique()
    print(f"\n  Verification - {mutant_prob_col}:")
    for label in sample_labels:
        mean_prob = df_embryo_probs[df_embryo_probs['true_label'] == label][mutant_prob_col].mean()
        print(f"    {label}: mean = {mean_prob:.3f}")

    # Sanity check: probabilities should sum to ~1.0
    if len(prob_cols) == 2:
        prob_sum = df_embryo_probs[prob_cols].sum(axis=1).mean()
        print(f"\n  Sanity check: mean sum of probabilities = {prob_sum:.3f} (should be ~1.0)")

    # Save predictions
    df_embryo_probs.to_csv(data_dir / f"{genotype_family}_predictions.csv", index=False)
    print(f"Saved: {data_dir / f'{genotype_family}_predictions.csv'}")

    # STEP 4: Compute per-embryo metrics
    print("\n" + "=" * 80)
    print("STEP 4: Computing per-embryo averages (homozygous only)")
    print("=" * 80)

    embryo_metrics = compute_per_embryo_metrics(
        df_distances,
        df_embryo_probs,
        genotype=homozygous_genotype,
        distance_col='euclidean_distance',
        prob_col='pred_prob_mutant'  # Use the corrected probability
    )

    print(f"  {len(embryo_metrics)} homozygous embryos with complete data")

    # Save per-embryo metrics
    embryo_metrics.to_csv(data_dir / f"{homozygous_genotype}_per_embryo_metrics.csv", index=False)
    print(f"Saved: {data_dir / f'{homozygous_genotype}_per_embryo_metrics.csv'}")

    # STEP 5: Compute correlations
    print("\n" + "=" * 80)
    print("STEP 5: Computing correlation statistics")
    print("=" * 80)

    corr_stats = compute_correlation_statistics(embryo_metrics)

    print(f"\n{homozygous_genotype} Correlation Results:")
    print(f"  N embryos: {corr_stats['n_embryos']}")
    print(f"  Pearson r:  {corr_stats['pearson_r']:.3f} (p={corr_stats['pearson_p']:.3e})")
    print(f"  Spearman ρ: {corr_stats['spearman_rho']:.3f} (p={corr_stats['spearman_p']:.3e})")
    print(f"  Mean distance: {corr_stats['mean_distance']:.3f}")
    print(f"  Mean prob: {corr_stats['mean_prob']:.3f}")

    # STEP 6: Bootstrap CIs
    print("\n" + "=" * 80)
    print("STEP 6: Computing bootstrap confidence intervals")
    print("=" * 80)

    boot_cis = bootstrap_correlation_ci(
        embryo_metrics,
        n_bootstrap=1000,
        random_state=42
    )

    print(f"\nBootstrap 95% CIs:")
    print(f"  Pearson:  [{boot_cis['pearson_ci'][0]:.3f}, {boot_cis['pearson_ci'][1]:.3f}]")
    print(f"  Spearman: [{boot_cis['spearman_ci'][0]:.3f}, {boot_cis['spearman_ci'][1]:.3f}]")

    # STEP 7: Visualization
    print("\n" + "=" * 80)
    print("STEP 7: Creating visualizations")
    print("=" * 80)

    # Scatter plot
    fig = plot_distance_vs_probability(
        embryo_metrics,
        corr_stats,
        genotype=homozygous_genotype,
        save_path=plot_dir / f"{homozygous_genotype}_scatter.png"
    )
    plt.close(fig)

    # Combine results for output
    results_df = pd.DataFrame([{
        'genotype_family': genotype_family,
        'genotype': homozygous_genotype,
        **corr_stats,
        'pearson_ci_lower': boot_cis['pearson_ci'][0],
        'pearson_ci_upper': boot_cis['pearson_ci'][1],
        'spearman_ci_lower': boot_cis['spearman_ci'][0],
        'spearman_ci_upper': boot_cis['spearman_ci'][1]
    }])

    results_df.to_csv(data_dir / f"{homozygous_genotype}_correlation_summary.csv", index=False)
    print(f"Saved: {data_dir / f'{homozygous_genotype}_correlation_summary.csv'}")

    return corr_stats


def main():
    """Run Step 1 correlation analysis for both CEP290 and TMEM67."""

    output_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251020")

    print("\n" + "=" * 80)
    print("STEP 1: CORRELATION ANALYSIS FOR INCOMPLETE PENETRANCE")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")

    all_results = []

    # Analyze CEP290
    cep290_stats = analyze_penetrance_correlation(
        genotype_family='cep290',
        experiment_ids=config.CEP290_EXPERIMENTS,
        reference_genotypes=['cep290_wildtype'],
        homozygous_genotype='cep290_homozygous',
        output_dir=output_dir,
        bin_width=2.0
    )
    all_results.append({
        'genotype': 'cep290_homozygous',
        **cep290_stats
    })

    # Analyze TMEM67
    tmem67_stats = analyze_penetrance_correlation(
        genotype_family='tmem67',
        experiment_ids=config.TMEM67_EXPERIMENTS,
        reference_genotypes=['tmem67_wildtype'],
        homozygous_genotype='tmem67_homozygous',
        output_dir=output_dir,
        bin_width=2.0
    )
    all_results.append({
        'genotype': 'tmem67_homozygous',
        **tmem67_stats
    })

    # Create summary comparison plot
    print("\n" + "=" * 80)
    print("Creating summary comparison plot")
    print("=" * 80)

    df_all_results = pd.DataFrame(all_results)
    plot_dir = output_dir / "plots" / "penetrance"

    fig = plot_correlation_summary(
        df_all_results,
        save_path=plot_dir / "correlation_comparison.png"
    )
    plt.close(fig)

    print("\n" + "=" * 80)
    print("STEP 1 ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print("  DATA:")
    print("    - data/penetrance/{genotype}_distances.csv")
    print("    - data/penetrance/{genotype}_predictions.csv")
    print("    - data/penetrance/{genotype}_per_embryo_metrics.csv")
    print("    - data/penetrance/{genotype}_correlation_summary.csv")
    print("  PLOTS:")
    print("    - plots/penetrance/{genotype}_scatter.png")
    print("    - plots/penetrance/correlation_comparison.png")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()
