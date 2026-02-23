#!/usr/bin/env python3
"""
Test robust distance methods to see which ones fix the Mahalanobis distance issue.

Compares:
1. Standard Mahalanobis (broken)
2. Diagonal Mahalanobis
3. Shrinkage methods (Ledoit-Wolf, OAS)
4. PCA-based methods (50%, 95% variance)
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
from divergence_analysis.workflow import compute_reference_distribution
from divergence_analysis.distances import (
    compute_mahalanobis_distance,
    compute_euclidean_distance
)
from divergence_analysis.robust_distances import (
    compute_diagonal_mahalanobis,
    compute_shrinkage_mahalanobis,
    compute_pca_mahalanobis,
    recommend_distance_method
)


def compute_all_distance_methods(df_binned, reference_stats, test_genotype):
    """Compute divergence using all methods for comparison."""
    print(f"\nComputing distances for {test_genotype}...")

    z_cols = [c for c in df_binned.columns if c.endswith("_binned")]
    df_test = df_binned[df_binned['genotype'] == test_genotype].copy()

    results = []
    n_skipped = 0

    for idx, row in df_test.iterrows():
        time_bin = row['time_bin']

        if time_bin not in reference_stats:
            n_skipped += 1
            continue

        ref = reference_stats[time_bin]

        # Get test point and reference
        X = np.asarray(row[z_cols].values, dtype=np.float64).reshape(1, -1)
        mu_ref = np.asarray(ref['mean'], dtype=np.float64)
        cov_ref = np.asarray(ref['cov'], dtype=np.float64)
        std_ref = np.asarray(ref['std'], dtype=np.float64)

        # Get reference samples for shrinkage/PCA methods
        df_ref_samples = df_binned[
            (df_binned['genotype'].isin(['tmem67_wildtype'])) &
            (df_binned['time_bin'] == time_bin)
        ]
        X_ref = df_ref_samples[z_cols].values.astype(np.float64)

        result = {
            'embryo_id': row['embryo_id'],
            'time_bin': time_bin,
            'genotype': test_genotype,
            'n_ref_samples': ref['n_samples'],
            'n_features': len(z_cols)
        }

        # 1. Standard methods (for comparison)
        result['euclidean'] = compute_euclidean_distance(X, mu_ref)[0]

        try:
            result['mahalanobis_standard'] = compute_mahalanobis_distance(
                X, mu_ref, cov_ref, robust=True, reg_param=1e-6
            )[0]
        except Exception as e:
            result['mahalanobis_standard'] = np.nan

        # 2. Diagonal Mahalanobis (no correlations)
        try:
            result['mahalanobis_diagonal'] = compute_diagonal_mahalanobis(
                X, mu_ref, std_ref
            )[0]
        except Exception as e:
            result['mahalanobis_diagonal'] = np.nan

        # 3. Shrinkage: Ledoit-Wolf
        try:
            result['mahalanobis_lw'] = compute_shrinkage_mahalanobis(
                X, mu_ref, X_ref, method='ledoit_wolf'
            )[0]
        except Exception as e:
            result['mahalanobis_lw'] = np.nan

        # 4. Shrinkage: OAS
        try:
            result['mahalanobis_oas'] = compute_shrinkage_mahalanobis(
                X, mu_ref, X_ref, method='oas'
            )[0]
        except Exception as e:
            result['mahalanobis_oas'] = np.nan

        # 5. PCA 95% variance
        try:
            result['mahalanobis_pca95'], pca95 = compute_pca_mahalanobis(
                X, mu_ref, X_ref, explained_variance_threshold=0.95
            )
            result['mahalanobis_pca95'] = result['mahalanobis_pca95'][0]
            result['n_components_pca95'] = pca95.n_components_
        except Exception as e:
            result['mahalanobis_pca95'] = np.nan
            result['n_components_pca95'] = np.nan

        # 6. PCA 50% variance
        try:
            result['mahalanobis_pca50'], pca50 = compute_pca_mahalanobis(
                X, mu_ref, X_ref, explained_variance_threshold=0.50
            )
            result['mahalanobis_pca50'] = result['mahalanobis_pca50'][0]
            result['n_components_pca50'] = pca50.n_components_
        except Exception as e:
            result['mahalanobis_pca50'] = np.nan
            result['n_components_pca50'] = np.nan

        # 7. PCA with fixed small dimension
        try:
            result['mahalanobis_pca10'], pca10 = compute_pca_mahalanobis(
                X, mu_ref, X_ref, n_components=10
            )
            result['mahalanobis_pca10'] = result['mahalanobis_pca10'][0]
        except Exception as e:
            result['mahalanobis_pca10'] = np.nan

        results.append(result)

    print(f"  Computed {len(results)} timepoints ({n_skipped} skipped)")
    return pd.DataFrame(results)


def plot_method_comparison(df_results, output_path):
    """Create comprehensive comparison plots."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    genotypes = df_results['genotype'].unique()
    colors = {'tmem67_wildtype': '#2ecc71', 'tmem67_heterozygote': '#3498db',
              'tmem67_homozygous': '#e74c3c'}

    distance_methods = [
        'euclidean',
        'mahalanobis_standard',
        'mahalanobis_diagonal',
        'mahalanobis_lw',
        'mahalanobis_oas',
        'mahalanobis_pca95',
        'mahalanobis_pca50',
        'mahalanobis_pca10'
    ]

    method_labels = {
        'euclidean': 'Euclidean (baseline)',
        'mahalanobis_standard': 'Mahalanobis (standard, BROKEN)',
        'mahalanobis_diagonal': 'Mahalanobis (diagonal)',
        'mahalanobis_lw': 'Mahalanobis (Ledoit-Wolf)',
        'mahalanobis_oas': 'Mahalanobis (OAS)',
        'mahalanobis_pca95': 'Mahalanobis (PCA 95%)',
        'mahalanobis_pca50': 'Mahalanobis (PCA 50%)',
        'mahalanobis_pca10': 'Mahalanobis (PCA 10 dims)'
    }

    # Plot 1-8: Time trajectories for each method
    for idx, method in enumerate(distance_methods):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])

        for genotype in genotypes:
            if genotype not in colors:
                continue

            df_gen = df_results[df_results['genotype'] == genotype]

            # Filter out NaN values
            df_plot = df_gen[['time_bin', method]].dropna()
            if len(df_plot) == 0:
                continue

            df_time = df_plot.groupby('time_bin')[method].agg(['mean', 'std']).reset_index()

            ax.plot(df_time['time_bin'], df_time['mean'],
                   marker='o', label=genotype, color=colors[genotype], linewidth=2, markersize=4)
            ax.fill_between(df_time['time_bin'],
                           df_time['mean'] - df_time['std'],
                           df_time['mean'] + df_time['std'],
                           alpha=0.2, color=colors[genotype])

        ax.set_xlabel('Time (hpf)', fontsize=10)
        ax.set_ylabel('Distance', fontsize=10)
        ax.set_title(method_labels.get(method, method), fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Set reasonable y-axis limits (exclude outliers)
        if method == 'mahalanobis_standard':
            ax.set_ylim(0, 500)  # Cap the broken one
            ax.axhline(500, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax.text(0.98, 0.98, 'Y-axis capped!', transform=ax.transAxes,
                   ha='right', va='top', color='red', fontsize=8)

    plt.suptitle('Comparison of Distance Methods: TMEM67 Analysis',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot to: {output_path}")
    plt.close()


def plot_slope_comparison(df_results, output_path):
    """Compare temporal slopes across methods."""
    from scipy.stats import linregress

    genotypes = df_results['genotype'].unique()
    distance_methods = [
        'euclidean',
        'mahalanobis_standard',
        'mahalanobis_diagonal',
        'mahalanobis_lw',
        'mahalanobis_oas',
        'mahalanobis_pca95',
        'mahalanobis_pca50',
        'mahalanobis_pca10'
    ]

    slope_data = []

    for genotype in genotypes:
        df_gen = df_results[df_results['genotype'] == genotype]

        for method in distance_methods:
            # Get clean data
            df_clean = df_gen[['time_bin', method]].dropna()
            if len(df_clean) < 5:
                continue

            df_time = df_clean.groupby('time_bin')[method].mean().reset_index()

            if len(df_time) < 3:
                continue

            # Compute linear trend
            slope, intercept, r_value, p_value, std_err = linregress(
                df_time['time_bin'], df_time[method]
            )

            slope_data.append({
                'genotype': genotype,
                'method': method,
                'slope': slope,
                'p_value': p_value,
                'r_squared': r_value**2
            })

    df_slopes = pd.DataFrame(slope_data)

    # Create bar plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, genotype in enumerate(genotypes):
        ax = axes[idx]
        df_gen = df_slopes[df_slopes['genotype'] == genotype]

        # Sort by slope
        df_gen = df_gen.sort_values('slope', ascending=False)

        # Color bars by significance
        colors = ['red' if p < 0.001 else 'orange' if p < 0.05 else 'gray'
                  for p in df_gen['p_value']]

        ax.barh(df_gen['method'], df_gen['slope'], color=colors)
        ax.set_xlabel('Temporal Slope (distance/hpf)', fontsize=11)
        ax.set_title(f'{genotype}', fontsize=12, fontweight='bold')
        ax.axvline(0, color='black', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)

        # Add significance legend
        if idx == 2:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', label='p < 0.001'),
                Patch(facecolor='orange', label='p < 0.05'),
                Patch(facecolor='gray', label='p ≥ 0.05')
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.suptitle('Temporal Slope Comparison: Which Methods Show Spurious Increasing Trends?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved slope comparison to: {output_path}")
    plt.close()

    return df_slopes


def main():
    print("="*80)
    print("TESTING ROBUST DISTANCE METHODS")
    print("="*80)

    # Setup directories
    data_dir = Path(config.DATA_DIR) / "tmem67" / "robust_test"
    plot_dir = Path(config.PLOT_DIR) / "tmem67" / "robust_test"
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # STEP 1: Load data
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: Loading data")
    print("="*80)

    df_raw = load_experiments(
        experiment_ids=config.TMEM67_EXPERIMENTS,
        build_dir=config.BUILD06_DIR,
        verbose=True
    )
    df_binned = bin_embryos_by_time(df_raw, bin_width=2.0)

    z_cols = [c for c in df_binned.columns if c.endswith("_binned")]
    print(f"\nData dimensions:")
    print(f"  Number of latent features: {len(z_cols)}")
    print(f"  Total embryo-timepoints: {len(df_binned)}")
    print(f"  Genotypes: {df_binned['genotype'].value_counts().to_dict()}")

    # ========================================================================
    # STEP 2: Compute reference
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: Computing reference distribution")
    print("="*80)

    ref_stats = compute_reference_distribution(
        df_binned,
        reference_genotypes=['tmem67_wildtype'],
        min_samples=10
    )

    # Check dimensionality
    sample_ref = list(ref_stats.values())[0]
    n_samples = sample_ref['n_samples']
    n_features = len(z_cols)
    print(f"\nDimensionality check:")
    print(f"  n_samples: {n_samples}")
    print(f"  n_features: {n_features}")
    print(f"  Ratio (n/p): {n_samples/n_features:.2f}")
    print(f"  Recommended method: {recommend_distance_method(n_samples, n_features)}")

    # ========================================================================
    # STEP 3: Compute distances with all methods
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: Computing distances with all methods")
    print("="*80)

    genotypes = ['tmem67_wildtype', 'tmem67_heterozygote', 'tmem67_homozygous']
    df_all = []

    for genotype in genotypes:
        df_gen = compute_all_distance_methods(df_binned, ref_stats, genotype)
        df_all.append(df_gen)

    df_results = pd.concat(df_all, ignore_index=True)
    df_results.to_csv(data_dir / "all_methods_comparison.csv", index=False)
    print(f"\nResults saved to: {data_dir / 'all_methods_comparison.csv'}")

    # ========================================================================
    # STEP 4: Summary statistics
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: Summary statistics")
    print("="*80)

    distance_cols = [c for c in df_results.columns if 'mahalanobis' in c or c == 'euclidean']

    for genotype in genotypes:
        print(f"\n{genotype}:")
        df_gen = df_results[df_results['genotype'] == genotype]

        for col in distance_cols:
            valid_data = df_gen[col].dropna()
            if len(valid_data) == 0:
                print(f"  {col:30s}: NO DATA")
            else:
                print(f"  {col:30s}: {valid_data.mean():8.3f} ± {valid_data.std():8.3f}")

    # ========================================================================
    # STEP 5: Generate plots
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: Generating comparison plots")
    print("="*80)

    plot_method_comparison(df_results, plot_dir / "method_comparison_trajectories.png")

    df_slopes = plot_slope_comparison(df_results, plot_dir / "method_comparison_slopes.png")
    df_slopes.to_csv(data_dir / "temporal_slopes.csv", index=False)
    print(f"Slope analysis saved to: {data_dir / 'temporal_slopes.csv'}")

    # ========================================================================
    # STEP 6: Recommendations
    # ========================================================================
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    print("\nBased on the analysis:")
    print("  1. Check method_comparison_trajectories.png to see which methods are stable")
    print("  2. Check method_comparison_slopes.png to see which avoid spurious trends")
    print("  3. Look for methods where:")
    print("     - WT self-comparison stays near baseline")
    print("     - Heterozygote/homozygous show reasonable separation")
    print("     - Trends match biological expectations")
    print("\nExpected outcomes:")
    print("  ✗ Standard Mahalanobis: Should blow up (condition number ~1e9)")
    print("  ✓ Diagonal: Should be stable but ignores correlations")
    print("  ✓ Shrinkage (LW/OAS): Should be stable and account for correlations")
    print("  ✓ PCA: Should be very stable, reduces dimensionality")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults: {data_dir}")
    print(f"Plots: {plot_dir}")


if __name__ == "__main__":
    main()
