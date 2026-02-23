#!/usr/bin/env python3
"""
Compare Distance Metrics for WT/Mutant Separation via AUROC Analysis

Treats distance-to-WT as a 1D binary classifier:
- WT embryos should have SMALL distance (class 0)
- Mutant embryos should have LARGE distance (class 1)

Compares Diagonal Mahalanobis vs. Euclidean distance using:
- AUROC (area under ROC curve)
- PR-AUC (precision-recall AUC)
- Balanced accuracy
- Bootstrap CIs for paired differences

Runs analysis for both CEP290 and TMEM67 genotypes.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "20251016"))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import existing utilities from 20251016
from utils.data_loading import load_experiments
from utils.binning import bin_embryos_by_time
import config_new as config

# Import distance computation
from divergence_analysis.robust_distances import compute_diagonal_mahalanobis
from divergence_analysis.distances import compute_euclidean_distance

# Import our new AUROC comparison tools
from distance_metrics import (
    compute_distance_auroc,
    compare_distance_metrics,
    bootstrap_paired_difference,
    compute_roc_curves,
    aggregate_across_time_bins
)


def compute_distances_for_all_embryos(
    df_binned: pd.DataFrame,
    reference_genotypes: list,
    test_genotypes: list,
    z_cols: list = None,
    time_col: str = "time_bin",
    min_ref_samples: int = 10
) -> pd.DataFrame:
    """
    Compute both diagonal Mahalanobis and Euclidean distances for all embryos.

    Parameters
    ----------
    df_binned : pd.DataFrame
        Binned embryo data
    reference_genotypes : list
        WT genotype(s) to use as reference
    test_genotypes : list
        Mutant genotypes to test
    z_cols : list, optional
        Latent feature columns
    time_col : str
        Time bin column
    min_ref_samples : int
        Minimum reference samples per time bin

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - embryo_id, time_bin, genotype
        - diagonal_mahalanobis_distance
        - euclidean_distance
    """
    # Auto-detect latent columns
    if z_cols is None:
        z_cols = [c for c in df_binned.columns if c.endswith("_binned")]
        if not z_cols:
            raise ValueError("No latent columns found")

    # Compute reference statistics per time bin
    df_ref = df_binned[df_binned['genotype'].isin(reference_genotypes)].copy()
    print(f"\nReference genotypes: {reference_genotypes}")
    print(f"  Total reference embryos: {df_ref['embryo_id'].nunique()}")

    # Get all embryos (WT + mutants) for comparison
    all_genotypes = reference_genotypes + test_genotypes
    df_all = df_binned[df_binned['genotype'].isin(all_genotypes)].copy()
    print(f"\nAll genotypes for analysis: {all_genotypes}")
    print(f"  Total embryos: {df_all['embryo_id'].nunique()}")

    results = []
    unique_time_bins = sorted(df_all[time_col].unique())
    n_total_bins = len(unique_time_bins)
    processed_bins = 0

    print(f"  Computing distances across {n_total_bins} time bins...")

    # Process each time bin
    for time_bin in unique_time_bins:
        # Get reference samples for this time bin
        ref_group = df_ref[df_ref[time_col] == time_bin]
        if len(ref_group) < min_ref_samples:
            print(f"  Skipping time bin {time_bin}: only {len(ref_group)} reference samples")
            continue

        # Compute reference statistics
        X_ref = ref_group[z_cols].values
        mu_ref = X_ref.mean(axis=0)
        std_ref = X_ref.std(axis=0)

        # Get all embryos for this time bin
        all_group = df_all[df_all[time_col] == time_bin]

        # Compute distances for all embryos (including WT)
        X_all = all_group[z_cols].values

        # Diagonal Mahalanobis (standardized Euclidean)
        diag_mahal_dist = compute_diagonal_mahalanobis(X_all, mu_ref, std_ref)

        # Euclidean
        eucl_dist = compute_euclidean_distance(X_all, mu_ref)

        # Store results
        for i, (idx, row) in enumerate(all_group.iterrows()):
            results.append({
                'embryo_id': row['embryo_id'],
                'time_bin': time_bin,
                'genotype': row['genotype'],
                'diagonal_mahalanobis_distance': diag_mahal_dist[i],
                'euclidean_distance': eucl_dist[i]
            })

        processed_bins += 1
        if processed_bins % 5 == 0 or processed_bins == n_total_bins:
            print(f"    Progress: {processed_bins}/{n_total_bins} bins processed")

    df_distances = pd.DataFrame(results)
    print(f"  ✓ Computed distances for {len(df_distances)} timepoints across {len(df_distances['time_bin'].unique())} time bins")

    return df_distances


def analyze_genotype_family(
    genotype_family: str,
    experiment_ids: list,
    reference_genotypes: list,
    test_genotypes: list,
    output_dir: Path,
    bin_width: float = 2.0
) -> dict:
    """
    Run complete distance comparison analysis for one genotype family.

    Parameters
    ----------
    genotype_family : str
        Name of genotype family (e.g., 'cep290', 'tmem67')
    experiment_ids : list
        Experiment IDs to load
    reference_genotypes : list
        WT genotypes for reference
    test_genotypes : list
        Mutant genotypes to compare
    output_dir : Path
        Output directory for results
    bin_width : float
        Time bin width in hours

    Returns
    -------
    dict
        Summary statistics
    """
    print("=" * 80)
    print(f"ANALYZING {genotype_family.upper()}")
    print("=" * 80)

    # Create output directories
    data_dir = output_dir / "data"
    plot_dir = output_dir / "plots"
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
    print(f"Binned data: {len(df_binned)} timepoints from {df_binned['embryo_id'].nunique()} embryos")

    # STEP 2: Compute distances
    print("\n" + "=" * 80)
    print("STEP 2: Computing distances for all embryos")
    print("=" * 80)

    df_distances = compute_distances_for_all_embryos(
        df_binned,
        reference_genotypes=reference_genotypes,
        test_genotypes=test_genotypes
    )

    # Save distance data
    df_distances.to_csv(data_dir / f"{genotype_family}_distances.csv", index=False)
    print(f"Saved: {data_dir / f'{genotype_family}_distances.csv'}")

    # STEP 3: AUROC analysis per time bin
    print("\n" + "=" * 80)
    print("STEP 3: AUROC analysis per time bin")
    print("=" * 80)

    per_bin_results = []
    unique_time_bins = sorted(df_distances['time_bin'].unique())
    n_bins = len(unique_time_bins)

    print(f"  Analyzing {n_bins} time bins...")

    for idx, time_bin in enumerate(unique_time_bins, 1):
        bin_data = df_distances[df_distances['time_bin'] == time_bin]

        # Create binary labels: 0 for WT, 1 for mutant
        is_wt = bin_data['genotype'].isin(reference_genotypes)
        labels = (~is_wt).astype(int)

        # Skip if we don't have both classes
        if len(np.unique(labels)) != 2:
            print(f"  Skipping time bin {time_bin}: only one class present")
            continue

        # Compute AUROC for both metrics
        distance_dict = {
            'diagonal_mahalanobis': bin_data['diagonal_mahalanobis_distance'].values,
            'euclidean': bin_data['euclidean_distance'].values
        }

        comparison = compare_distance_metrics(distance_dict, labels)
        comparison['time_bin'] = time_bin

        per_bin_results.append(comparison)

        # Show progress
        diag_auroc = comparison[comparison['metric']=='diagonal_mahalanobis']['auroc'].values[0]
        eucl_auroc = comparison[comparison['metric']=='euclidean']['auroc'].values[0]
        print(f"  [{idx}/{n_bins}] Time bin {time_bin} hpf: Diag_Mahal={diag_auroc:.3f}, Euclidean={eucl_auroc:.3f}")

    df_per_bin = pd.concat(per_bin_results, ignore_index=True)
    df_per_bin.to_csv(data_dir / f"{genotype_family}_per_bin_auroc.csv", index=False)
    print(f"\nSaved: {data_dir / f'{genotype_family}_per_bin_auroc.csv'}")

    # STEP 4: Aggregate statistics
    print("\n" + "=" * 80)
    print("STEP 4: Computing aggregate statistics")
    print("=" * 80)

    summary = {}
    for metric in ['diagonal_mahalanobis', 'euclidean']:
        metric_data = df_per_bin[df_per_bin['metric'] == metric]
        agg_stats = aggregate_across_time_bins(metric_data, metric_col='auroc')
        summary[metric] = agg_stats

        print(f"\n{metric.upper()}:")
        print(f"  Median AUROC: {agg_stats['median']:.3f}")
        print(f"  Std: {agg_stats['std']:.3f}")
        print(f"  Range: [{agg_stats['min']:.3f}, {agg_stats['max']:.3f}]")

    # STEP 5: Bootstrap paired difference
    print("\n" + "=" * 80)
    print("STEP 5: Bootstrap paired difference test")
    print("=" * 80)

    # Use all data (across time bins) for bootstrap
    is_wt = df_distances['genotype'].isin(reference_genotypes)
    all_labels = (~is_wt).astype(int)

    bootstrap_result = bootstrap_paired_difference(
        distances_a=df_distances['diagonal_mahalanobis_distance'].values,
        distances_b=df_distances['euclidean_distance'].values,
        labels=all_labels,
        metric='auroc',
        n_bootstrap=100,  # Reduced from 1000 for speed
        random_state=42,
        verbose=True
    )

    print(f"\nPaired difference (Diagonal Mahalanobis - Euclidean):")
    print(f"  Mean difference: {bootstrap_result['mean_diff']:.4f}")
    print(f"  95% CI: [{bootstrap_result['ci_lower']:.4f}, {bootstrap_result['ci_upper']:.4f}]")
    print(f"  P-value: {bootstrap_result['p_value']:.4f}")

    # Save summary
    summary_df = pd.DataFrame([
        {
            'genotype_family': genotype_family,
            'metric': metric,
            **stats
        }
        for metric, stats in summary.items()
    ])
    summary_df['bootstrap_mean_diff'] = bootstrap_result['mean_diff']
    summary_df['bootstrap_ci_lower'] = bootstrap_result['ci_lower']
    summary_df['bootstrap_ci_upper'] = bootstrap_result['ci_upper']
    summary_df['bootstrap_p_value'] = bootstrap_result['p_value']

    summary_df.to_csv(data_dir / f"{genotype_family}_summary.csv", index=False)
    print(f"\nSaved: {data_dir / f'{genotype_family}_summary.csv'}")

    # STEP 6: Visualization
    print("\n" + "=" * 80)
    print("STEP 6: Creating visualizations")
    print("=" * 80)

    # Plot 1: AUROC by time bin
    fig, ax = plt.subplots(figsize=(12, 6))

    for metric in ['diagonal_mahalanobis', 'euclidean']:
        metric_data = df_per_bin[df_per_bin['metric'] == metric]
        ax.plot(
            metric_data['time_bin'],
            metric_data['auroc'],
            marker='o',
            label=metric.replace('_', ' ').title(),
            linewidth=2,
            markersize=8
        )

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.set_xlabel('Time (hpf)', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title(f'{genotype_family.upper()}: Distance Metric Comparison Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    fig.savefig(plot_dir / f"{genotype_family}_auroc_by_time.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_dir / f'{genotype_family}_auroc_by_time.png'}")
    plt.close()

    # Plot 2: ROC curves (using all data)
    fig, ax = plt.subplots(figsize=(8, 8))

    distance_dict = {
        'Diagonal Mahalanobis': df_distances['diagonal_mahalanobis_distance'].values,
        'Euclidean': df_distances['euclidean_distance'].values
    }

    roc_curves = compute_roc_curves(distance_dict, all_labels)

    for metric_name, (fpr, tpr, _) in roc_curves.items():
        auroc = roc_auc_score(all_labels, distance_dict[metric_name])
        ax.plot(fpr, tpr, linewidth=2, label=f'{metric_name} (AUROC={auroc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'{genotype_family.upper()}: ROC Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(plot_dir / f"{genotype_family}_roc_curves.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_dir / f'{genotype_family}_roc_curves.png'}")
    plt.close()

    return summary


def main():
    """Run distance metric comparison for both CEP290 and TMEM67."""

    output_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251020")

    print("\n" + "=" * 80)
    print("DISTANCE METRIC COMPARISON: DIAGONAL MAHALANOBIS vs. EUCLIDEAN")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")

    # Analyze CEP290
    cep290_summary = analyze_genotype_family(
        genotype_family='cep290',
        experiment_ids=config.CEP290_EXPERIMENTS,
        reference_genotypes=['cep290_wildtype'],
        test_genotypes=['cep290_heterozygous', 'cep290_homozygous'],
        output_dir=output_dir,
        bin_width=2.0
    )

    # Analyze TMEM67
    tmem67_summary = analyze_genotype_family(
        genotype_family='tmem67',
        experiment_ids=config.TMEM67_EXPERIMENTS,
        reference_genotypes=['tmem67_wildtype'],
        test_genotypes=['tmem67_heterozygote', 'tmem67_homozygous'],
        output_dir=output_dir,
        bin_width=2.0
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print("  DATA:")
    print("    - cep290_distances.csv")
    print("    - cep290_per_bin_auroc.csv")
    print("    - cep290_summary.csv")
    print("    - tmem67_distances.csv")
    print("    - tmem67_per_bin_auroc.csv")
    print("    - tmem67_summary.csv")
    print("  PLOTS:")
    print("    - cep290_auroc_by_time.png")
    print("    - cep290_roc_curves.png")
    print("    - tmem67_auroc_by_time.png")
    print("    - tmem67_roc_curves.png")


if __name__ == "__main__":
    # Need to import sklearn
    from sklearn.metrics import roc_auc_score
    main()
