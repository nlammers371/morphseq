#!/usr/bin/env python3
"""
Debug script to investigate why Mahalanobis distance increases while Euclidean doesn't.

This script will:
1. Add wildtype self-comparison as a control
2. Examine covariance matrices and their properties
3. Check for systematic issues in distance computation
4. Compare distributions across genotypes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config_new as config
from utils.data_loading import load_experiments
from utils.binning import bin_embryos_by_time
from divergence_analysis.workflow import (
    compute_reference_distribution,
    compute_divergence_scores
)


def analyze_covariance_properties(reference_stats):
    """Analyze properties of covariance matrices across time."""
    print("\n" + "="*80)
    print("COVARIANCE MATRIX ANALYSIS")
    print("="*80)

    results = []
    for time_bin, stats in reference_stats.items():
        cov = stats['cov']

        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(cov)

        # Compute condition number (ratio of largest to smallest eigenvalue)
        condition_number = np.max(eigenvalues) / max(np.min(eigenvalues), 1e-10)

        # Determinant
        det = np.linalg.det(cov)

        # Trace
        trace = np.trace(cov)

        # Frobenius norm
        frob_norm = np.linalg.norm(cov, 'fro')

        results.append({
            'time_bin': time_bin,
            'n_samples': stats['n_samples'],
            'condition_number': condition_number,
            'det': det,
            'trace': trace,
            'frob_norm': frob_norm,
            'min_eigenvalue': np.min(eigenvalues),
            'max_eigenvalue': np.max(eigenvalues),
            'mean_eigenvalue': np.mean(eigenvalues)
        })

    df_cov_analysis = pd.DataFrame(results)

    print(f"\nCovariance matrix statistics across {len(df_cov_analysis)} time bins:")
    print(f"  Condition number range: {df_cov_analysis['condition_number'].min():.2f} - {df_cov_analysis['condition_number'].max():.2f}")
    print(f"  Determinant range: {df_cov_analysis['det'].min():.2e} - {df_cov_analysis['det'].max():.2e}")
    print(f"  Trace range: {df_cov_analysis['trace'].min():.2f} - {df_cov_analysis['trace'].max():.2f}")
    print(f"  Min eigenvalue range: {df_cov_analysis['min_eigenvalue'].min():.2e} - {df_cov_analysis['min_eigenvalue'].max():.2e}")

    # Check for potential issues
    if (df_cov_analysis['condition_number'] > 1e6).any():
        print("\n⚠️  WARNING: Some covariance matrices have very high condition numbers (>1e6)")
        print("    This suggests near-singularity and may cause Mahalanobis distance issues")

    if (df_cov_analysis['det'] < 1e-10).any():
        print("\n⚠️  WARNING: Some covariance matrices have very small determinants (<1e-10)")
        print("    This suggests near-singularity")

    return df_cov_analysis


def compare_distance_scaling(df_binned, reference_stats, genotypes):
    """Compare how distances scale across genotypes."""
    print("\n" + "="*80)
    print("DISTANCE SCALING ANALYSIS")
    print("="*80)

    results = []

    for genotype in genotypes:
        print(f"\nAnalyzing {genotype}...")

        df_gen = df_binned[df_binned['genotype'] == genotype]

        for idx, row in df_gen.iterrows():
            time_bin = row['time_bin']

            if time_bin not in reference_stats:
                continue

            ref = reference_stats[time_bin]

            # Get feature columns
            z_cols = [c for c in df_binned.columns if c.endswith("_binned")]
            X = np.asarray(row[z_cols].values, dtype=np.float64)
            mu_ref = np.asarray(ref['mean'], dtype=np.float64)
            cov_ref = np.asarray(ref['cov'], dtype=np.float64)

            # Raw difference vector
            diff = X - mu_ref

            # Euclidean distance
            eucl_dist = np.linalg.norm(diff)

            # Mahalanobis distance
            cov_reg = cov_ref + 1e-6 * np.eye(cov_ref.shape[0])
            try:
                cov_inv = np.linalg.inv(cov_reg)
                mahal_dist = np.sqrt(diff @ cov_inv @ diff.T)
            except np.linalg.LinAlgError:
                cov_inv = np.linalg.pinv(cov_reg)
                mahal_dist = np.sqrt(diff @ cov_inv @ diff.T)

            # Compute ratio
            ratio = mahal_dist / max(eucl_dist, 1e-10)

            # Store results
            results.append({
                'embryo_id': row['embryo_id'],
                'genotype': genotype,
                'time_bin': time_bin,
                'euclidean': eucl_dist,
                'mahalanobis': mahal_dist,
                'ratio': ratio,
                'n_ref_samples': ref['n_samples']
            })

    df_scaling = pd.DataFrame(results)

    print("\nDistance scaling summary:")
    for genotype in genotypes:
        df_gen = df_scaling[df_scaling['genotype'] == genotype]
        if len(df_gen) == 0:
            continue
        print(f"\n  {genotype}:")
        print(f"    Euclidean: {df_gen['euclidean'].mean():.3f} ± {df_gen['euclidean'].std():.3f}")
        print(f"    Mahalanobis: {df_gen['mahalanobis'].mean():.3f} ± {df_gen['mahalanobis'].std():.3f}")
        print(f"    Ratio (M/E): {df_gen['ratio'].mean():.3f} ± {df_gen['ratio'].std():.3f}")

    return df_scaling


def plot_distance_comparison(df_scaling, output_path):
    """Create diagnostic plots comparing distance metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    genotypes = df_scaling['genotype'].unique()
    colors = sns.color_palette("husl", len(genotypes))

    # Plot 1: Mahalanobis vs Euclidean scatter
    ax = axes[0, 0]
    for genotype, color in zip(genotypes, colors):
        df_gen = df_scaling[df_scaling['genotype'] == genotype]
        ax.scatter(df_gen['euclidean'], df_gen['mahalanobis'],
                  alpha=0.5, label=genotype, color=color, s=20)
    ax.plot([0, ax.get_xlim()[1]], [0, ax.get_xlim()[1]], 'k--', alpha=0.3, label='y=x')
    ax.set_xlabel('Euclidean Distance')
    ax.set_ylabel('Mahalanobis Distance')
    ax.set_title('Mahalanobis vs Euclidean Distance')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Ratio over time
    ax = axes[0, 1]
    for genotype, color in zip(genotypes, colors):
        df_gen = df_scaling[df_scaling['genotype'] == genotype]
        df_time = df_gen.groupby('time_bin')['ratio'].agg(['mean', 'std']).reset_index()
        ax.plot(df_time['time_bin'], df_time['mean'],
               marker='o', label=genotype, color=color, linewidth=2)
        ax.fill_between(df_time['time_bin'],
                       df_time['mean'] - df_time['std'],
                       df_time['mean'] + df_time['std'],
                       alpha=0.2, color=color)
    ax.set_xlabel('Time (hpf)')
    ax.set_ylabel('Mahalanobis / Euclidean Ratio')
    ax.set_title('Distance Ratio Over Time')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 3: Mahalanobis over time
    ax = axes[1, 0]
    for genotype, color in zip(genotypes, colors):
        df_gen = df_scaling[df_scaling['genotype'] == genotype]
        df_time = df_gen.groupby('time_bin')['mahalanobis'].agg(['mean', 'std']).reset_index()
        ax.plot(df_time['time_bin'], df_time['mean'],
               marker='o', label=genotype, color=color, linewidth=2)
        ax.fill_between(df_time['time_bin'],
                       df_time['mean'] - df_time['std'],
                       df_time['mean'] + df_time['std'],
                       alpha=0.2, color=color)
    ax.set_xlabel('Time (hpf)')
    ax.set_ylabel('Mahalanobis Distance')
    ax.set_title('Mahalanobis Distance Trajectories')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 4: Euclidean over time
    ax = axes[1, 1]
    for genotype, color in zip(genotypes, colors):
        df_gen = df_scaling[df_scaling['genotype'] == genotype]
        df_time = df_gen.groupby('time_bin')['euclidean'].agg(['mean', 'std']).reset_index()
        ax.plot(df_time['time_bin'], df_time['mean'],
               marker='o', label=genotype, color=color, linewidth=2)
        ax.fill_between(df_time['time_bin'],
                       df_time['mean'] - df_time['std'],
                       df_time['mean'] + df_time['std'],
                       alpha=0.2, color=color)
    ax.set_xlabel('Time (hpf)')
    ax.set_ylabel('Euclidean Distance')
    ax.set_title('Euclidean Distance Trajectories')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved diagnostic plot to: {output_path}")
    plt.close()


def main():
    print("="*80)
    print("DEBUGGING DISTANCE DISCREPANCY - TMEM67")
    print("="*80)

    # Setup directories
    data_dir = Path(config.DATA_DIR) / "tmem67" / "debug"
    plot_dir = Path(config.PLOT_DIR) / "tmem67" / "debug"
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # STEP 1: Load and bin data
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: Loading and binning data")
    print("="*80)

    df_raw = load_experiments(
        experiment_ids=config.TMEM67_EXPERIMENTS,
        build_dir=config.BUILD06_DIR,
        verbose=True
    )

    print(f"\nLoaded {len(df_raw)} timepoints from {df_raw['embryo_id'].nunique()} embryos")
    print(f"Genotypes: {df_raw['genotype'].value_counts().to_dict()}")

    df_binned = bin_embryos_by_time(df_raw, bin_width=2.0)
    print(f"\nBinned to {len(df_binned)} embryo-timepoints")

    # ========================================================================
    # STEP 2: Compute reference distribution
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: Computing reference distribution (wildtype)")
    print("="*80)

    ref_stats = compute_reference_distribution(
        df_binned,
        reference_genotypes=['tmem67_wildtype'],
        min_samples=10
    )

    # ========================================================================
    # STEP 3: Analyze covariance matrices
    # ========================================================================
    df_cov_analysis = analyze_covariance_properties(ref_stats)
    df_cov_analysis.to_csv(data_dir / "covariance_analysis.csv", index=False)
    print(f"\nCovariance analysis saved to: {data_dir / 'covariance_analysis.csv'}")

    # ========================================================================
    # STEP 4: Compare all genotypes including WT self-comparison
    # ========================================================================
    genotypes = ['tmem67_wildtype', 'tmem67_heterozygote', 'tmem67_homozygous']
    df_scaling = compare_distance_scaling(df_binned, ref_stats, genotypes)
    df_scaling.to_csv(data_dir / "distance_scaling.csv", index=False)
    print(f"\nDistance scaling analysis saved to: {data_dir / 'distance_scaling.csv'}")

    # ========================================================================
    # STEP 5: Create diagnostic plots
    # ========================================================================
    plot_distance_comparison(
        df_scaling,
        plot_dir / "distance_comparison_diagnostic.png"
    )

    # ========================================================================
    # STEP 6: Detailed examination of het behavior
    # ========================================================================
    print("\n" + "="*80)
    print("DETAILED HETEROZYGOTE ANALYSIS")
    print("="*80)

    df_het = df_scaling[df_scaling['genotype'] == 'tmem67_heterozygote']
    df_wt = df_scaling[df_scaling['genotype'] == 'tmem67_wildtype']
    df_hom = df_scaling[df_scaling['genotype'] == 'tmem67_homozygous']

    print("\nWildtype self-comparison (should be small):")
    print(f"  Euclidean: {df_wt['euclidean'].mean():.3f} ± {df_wt['euclidean'].std():.3f}")
    print(f"  Mahalanobis: {df_wt['mahalanobis'].mean():.3f} ± {df_wt['mahalanobis'].std():.3f}")
    print(f"  Ratio: {df_wt['ratio'].mean():.3f} ± {df_wt['ratio'].std():.3f}")

    print("\nHeterozygote vs wildtype:")
    print(f"  Euclidean: {df_het['euclidean'].mean():.3f} ± {df_het['euclidean'].std():.3f}")
    print(f"  Mahalanobis: {df_het['mahalanobis'].mean():.3f} ± {df_het['mahalanobis'].std():.3f}")
    print(f"  Ratio: {df_het['ratio'].mean():.3f} ± {df_het['ratio'].std():.3f}")

    print("\nHomozygous vs wildtype:")
    print(f"  Euclidean: {df_hom['euclidean'].mean():.3f} ± {df_hom['euclidean'].std():.3f}")
    print(f"  Mahalanobis: {df_hom['mahalanobis'].mean():.3f} ± {df_hom['mahalanobis'].std():.3f}")
    print(f"  Ratio: {df_hom['ratio'].mean():.3f} ± {df_hom['ratio'].std():.3f}")

    # Check if het Mahalanobis increases over time
    print("\n" + "="*80)
    print("TEMPORAL TREND ANALYSIS")
    print("="*80)

    for genotype in genotypes:
        df_gen = df_scaling[df_scaling['genotype'] == genotype]
        df_time = df_gen.groupby('time_bin')[['euclidean', 'mahalanobis']].mean().reset_index()

        if len(df_time) > 3:
            # Compute linear trend
            from scipy.stats import linregress

            eucl_trend = linregress(df_time['time_bin'], df_time['euclidean'])
            mahal_trend = linregress(df_time['time_bin'], df_time['mahalanobis'])

            print(f"\n{genotype}:")
            print(f"  Euclidean slope: {eucl_trend.slope:.4f} (p={eucl_trend.pvalue:.4f})")
            print(f"  Mahalanobis slope: {mahal_trend.slope:.4f} (p={mahal_trend.pvalue:.4f})")

            if mahal_trend.slope > 0 and mahal_trend.pvalue < 0.05 and abs(eucl_trend.slope) < 0.01:
                print(f"  ⚠️  ISSUE DETECTED: Mahalanobis increases significantly while Euclidean is flat!")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("DEBUG ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {data_dir}")
    print(f"Plots saved to: {plot_dir}")
    print("\nNext steps:")
    print("  1. Check covariance_analysis.csv for matrix conditioning issues")
    print("  2. Examine distance_comparison_diagnostic.png for visual patterns")
    print("  3. Review distance_scaling.csv for raw values")
    print("  4. If WT self-comparison shows increasing Mahalanobis, issue is in reference computation")
    print("  5. If WT is stable but het increases, issue is specific to het samples")


if __name__ == "__main__":
    main()
