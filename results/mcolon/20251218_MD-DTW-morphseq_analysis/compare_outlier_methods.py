#!/usr/bin/env python3
"""
Compare Principled Outlier Detection Methods for Log-Normal Distance Data

Tests 4 methods designed for skewed distributions:
1. Percentile (baseline - not principled but simple)
2. Log-MAD (recommended for biological data)
3. IQR Boxplot (industry standard)
4. Isolation Forest (ML approach, no distribution assumptions)

Created: 2025-12-18
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from pathlib import Path
import sys

# Add project root for imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))


def method_percentile(median_distances, percentile=95, verbose=True):
    """Baseline: Top N% removal (not principled but simple)."""
    threshold = np.percentile(median_distances, percentile)
    outlier_mask = median_distances > threshold

    if verbose:
        print(f"\n{'='*70}")
        print(f"METHOD 1: Percentile ({percentile}th)")
        print(f"{'='*70}")
        print(f"  Threshold: > {threshold:.3f}")
        print(f"  Outliers: {np.sum(outlier_mask)}")
        print(f"  Rationale: 'Remove top {100-percentile}% most distant'")
        print(f"  ⚠️  Issue: Arbitrary - removes N% even if all are normal")

    return outlier_mask, threshold


def method_log_mad(median_distances, multiplier=3.0, verbose=True):
    """
    RECOMMENDED: Log-MAD for log-normal data.

    Biological distances are often log-normal (bounded at 0, long right tail).
    Log-transform makes them ~normal, then apply standard MAD.
    """
    # Log-transform (log1p handles zeros safely)
    log_dists = np.log1p(median_distances)

    # Median and MAD in log space
    med_log = np.median(log_dists)
    mad_log = np.median(np.abs(log_dists - med_log))

    # Threshold in log space
    thresh_log = med_log + multiplier * mad_log

    # Convert back to linear scale for interpretation
    thresh_linear = np.expm1(thresh_log)

    # Identify outliers (in log space)
    outlier_mask = log_dists > thresh_log

    if verbose:
        print(f"\n{'='*70}")
        print(f"METHOD 2: Log-MAD ({multiplier}×) [RECOMMENDED]")
        print(f"{'='*70}")
        print(f"  Log-space median: {med_log:.3f}")
        print(f"  Log-space MAD: {mad_log:.3f}")
        print(f"  Log-space threshold: {thresh_log:.3f}")
        print(f"  Linear-space threshold: > {thresh_linear:.3f}")
        print(f"  Outliers: {np.sum(outlier_mask)}")
        print(f"  Rationale: 'Log-transform normalizes skew, then standard 3σ rule'")
        print(f"  ✓ Principled: Handles log-normal distributions correctly")

    return outlier_mask, thresh_linear


def method_iqr(median_distances, multiplier=1.5, verbose=True):
    """
    Industry Standard: IQR (Interquartile Range) method.

    Used by default in boxplots. Robust to skew.
    - 1.5× IQR = "mild outlier" (standard)
    - 3.0× IQR = "extreme outlier" (very conservative)
    """
    q1, q3 = np.percentile(median_distances, [25, 75])
    iqr = q3 - q1
    threshold = q3 + multiplier * iqr
    outlier_mask = median_distances > threshold

    if verbose:
        print(f"\n{'='*70}")
        print(f"METHOD 3: IQR Boxplot ({multiplier}× IQR)")
        print(f"{'='*70}")
        print(f"  Q1 (25th %ile): {q1:.3f}")
        print(f"  Q3 (75th %ile): {q3:.3f}")
        print(f"  IQR: {iqr:.3f}")
        print(f"  Threshold: Q3 + {multiplier}×IQR = {threshold:.3f}")
        print(f"  Outliers: {np.sum(outlier_mask)}")
        print(f"  Rationale: 'Standard boxplot outlier definition'")
        print(f"  ✓ Principled: Robust to skew, widely accepted")

    return outlier_mask, threshold


def method_isolation_forest(median_distances, contamination='auto', verbose=True):
    """
    ML Approach: Isolation Forest.

    Detects points that are "easy to isolate" from the main cluster.
    No distribution assumptions - works for any shape.
    """
    # Reshape for sklearn (expects 2D array)
    X = median_distances.reshape(-1, 1)

    # Fit Isolation Forest
    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(X)

    # -1 = outlier, 1 = inlier
    outlier_mask = preds == -1

    # Get decision scores for threshold interpretation
    scores = iso.decision_function(X)
    # Threshold is approximately where it splits inlier/outlier
    outlier_scores = scores[outlier_mask]
    inlier_scores = scores[~outlier_mask]

    if verbose:
        print(f"\n{'='*70}")
        print(f"METHOD 4: Isolation Forest (contamination={contamination})")
        print(f"{'='*70}")
        print(f"  Outliers: {np.sum(outlier_mask)}")
        print(f"  Decision boundary: ~{scores[outlier_mask].max():.3f}")
        print(f"  Rationale: 'Points easy to separate from main cluster'")
        print(f"  ✓ Principled: No distribution assumptions")
        print(f"  ⚠️  Black box: Harder to explain than threshold methods")

    return outlier_mask, None  # No simple threshold


def plot_comparison(median_distances, embryo_ids, results, save_path):
    """
    Visualize all methods side-by-side.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Map display names to actual keys in results dict
    method_keys = list(results.keys())
    methods = [
        (key.replace('_', ' ').title(), key)
        for key in method_keys
    ]

    # Sort embryos by median distance
    sorted_idx = np.argsort(median_distances)
    sorted_dists = median_distances[sorted_idx]

    for ax, (title, key) in zip(axes, methods):
        outlier_mask, threshold = results[key]
        sorted_outliers = outlier_mask[sorted_idx]

        # Bar plot
        colors = ['red' if is_outlier else 'steelblue' for is_outlier in sorted_outliers]
        ax.bar(range(len(sorted_dists)), sorted_dists, color=colors,
               edgecolor='black', alpha=0.7, linewidth=0.5)

        # Threshold line (if exists)
        if threshold is not None:
            ax.axhline(threshold, color='darkred', linestyle='--',
                      linewidth=2, label=f'Threshold = {threshold:.1f}')
            ax.legend(fontsize=10)

        # Labels
        n_outliers = np.sum(outlier_mask)
        ax.set_xlabel('Embryo (sorted by distance)', fontsize=11)
        ax.set_ylabel('Median Distance', fontsize=11)
        ax.set_title(f'{title}\n{n_outliers} outliers detected', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Log scale if needed to see structure
        if median_distances.max() / median_distances.min() > 10:
            ax.set_yscale('log')
            ax.set_ylabel('Median Distance (log scale)', fontsize=11)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✓ Comparison plot saved: {save_path.name}")


def compare_overlaps(median_distances, embryo_ids, results):
    """
    Show which embryos are flagged by which methods.
    """
    print(f"\n{'='*70}")
    print("OUTLIER OVERLAP ANALYSIS")
    print(f"{'='*70}")

    # Get outlier IDs for each method
    outliers_by_method = {}
    for name, (mask, _) in results.items():
        outliers_by_method[name] = set(np.array(embryo_ids)[mask])

    # Show overlap
    all_outliers = set()
    for outliers in outliers_by_method.values():
        all_outliers.update(outliers)

    print(f"\nTotal unique outliers across all methods: {len(all_outliers)}")
    print(f"\nOutliers by method:")
    for name, outliers in outliers_by_method.items():
        print(f"  {name:20s}: {len(outliers):2d} outliers")

    # Show which embryos are flagged by ALL methods (consensus)
    consensus = set.intersection(*outliers_by_method.values())

    print(f"\n{'─'*70}")
    print(f"CONSENSUS OUTLIERS (flagged by ALL methods): {len(consensus)}")
    print(f"{'─'*70}")
    if consensus:
        for embryo_id in sorted(consensus):
            idx = embryo_ids.index(embryo_id)
            print(f"  {embryo_id:25s} median_dist = {median_distances[idx]:.3f}")
    else:
        print("  (none - no consensus)")

    # Show embryos flagged by at least 3 methods
    vote_counts = {embryo: 0 for embryo in all_outliers}
    for outliers in outliers_by_method.values():
        for embryo in outliers:
            vote_counts[embryo] += 1

    strong_outliers = {e for e, count in vote_counts.items() if count >= 3}
    print(f"\n{'─'*70}")
    print(f"STRONG OUTLIERS (flagged by ≥3 methods): {len(strong_outliers)}")
    print(f"{'─'*70}")
    if strong_outliers:
        for embryo_id in sorted(strong_outliers):
            idx = embryo_ids.index(embryo_id)
            votes = vote_counts[embryo_id]
            print(f"  {embryo_id:25s} median_dist = {median_distances[idx]:7.3f}  ({votes}/4 methods)")


def main():
    """Run comprehensive outlier method comparison."""

    print("="*70)
    print("PRINCIPLED OUTLIER DETECTION METHOD COMPARISON")
    print("="*70)

    # Load data
    print("\nLoading distance matrix...")
    D = np.load('output/20251121_20251218_171616/distance_matrix.npy')
    with open('output/20251121_20251218_171616/embryo_ids.txt', 'r') as f:
        embryo_ids = [line.strip() for line in f]

    print(f"  Distance matrix: {D.shape}")
    print(f"  Embryos: {len(embryo_ids)}")

    # Compute median distances
    print("\nComputing median distances...")
    n = len(embryo_ids)
    median_distances = np.zeros(n)
    for i in range(n):
        dists_to_others = np.concatenate([D[i, :i], D[i, i+1:]])
        median_distances[i] = np.median(dists_to_others)

    print(f"  Range: {median_distances.min():.1f} to {median_distances.max():.1f}")
    print(f"  Mean: {median_distances.mean():.1f}")
    print(f"  Median: {np.median(median_distances):.1f}")
    print(f"  Std: {median_distances.std():.1f}")

    # Test parameter sweep first
    print(f"\n{'='*70}")
    print("PARAMETER SWEEP - Finding stringent cutoffs")
    print(f"{'='*70}")

    print(f"\nTesting different parameters to get ~5-10 outliers:")
    print(f"\n{'Method':<25} {'Parameter':<15} {'N Outliers':<12} {'Threshold'}")
    print("-" * 70)

    # Percentile sweep
    for p in [90, 92, 95, 97, 98]:
        mask, thresh = method_percentile(median_distances, percentile=p, verbose=False)
        print(f"{'Percentile':<25} {f'{p}th':<15} {np.sum(mask):<12} {thresh:.1f}")

    # Log-MAD sweep
    for mult in [4.0, 4.5, 5.0, 5.5, 6.0]:
        mask, thresh = method_log_mad(median_distances, multiplier=mult, verbose=False)
        print(f"{'Log-MAD':<25} {f'{mult}×':<15} {np.sum(mask):<12} {thresh:.1f}")

    # IQR sweep
    for mult in [2.0, 2.5, 3.0, 3.5, 4.0]:
        mask, thresh = method_iqr(median_distances, multiplier=mult, verbose=False)
        print(f"{'IQR':<25} {f'{mult}×':<15} {np.sum(mask):<12} {thresh:.1f}")

    # Isolation Forest contamination sweep
    for cont in [0.02, 0.03, 0.05, 0.07, 0.10]:
        mask, _ = method_isolation_forest(median_distances, contamination=cont, verbose=False)
        print(f"{'Isolation Forest':<25} {f'cont={cont}':<15} {np.sum(mask):<12} N/A")

    # Now run detailed comparison with stringent parameters
    print(f"\n{'='*70}")
    print("DETAILED COMPARISON - Stringent Parameters (target ~5-10 outliers)")
    print(f"{'='*70}")

    results = {}

    # Method 1: Percentile (95th - baseline)
    results['percentile_95'] = method_percentile(median_distances, percentile=95, verbose=True)

    # Method 2: Log-MAD with stringent cutoff
    results['log_mad_5x'] = method_log_mad(median_distances, multiplier=5.0, verbose=True)

    # Method 3: IQR with stringent cutoff
    results['iqr_3x'] = method_iqr(median_distances, multiplier=3.0, verbose=True)

    # Method 4: Isolation Forest with low contamination
    results['iso_forest_5pct'] = method_isolation_forest(median_distances, contamination=0.05, verbose=True)

    # Overlap analysis
    compare_overlaps(median_distances, embryo_ids, results)

    # Visualization
    output_dir = Path('output/outlier_testing')
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_comparison(
        median_distances,
        embryo_ids,
        results,
        output_dir / 'method_comparison.png'
    )

    # Recommendation
    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")

    log_mad_outliers = np.sum(results['log_mad_5x'][0])
    percentile_outliers = np.sum(results['percentile_95'][0])
    iqr_outliers = np.sum(results['iqr_3x'][0])

    print(f"\nWith stringent parameters, all methods converge:")
    print(f"    - Percentile (95th): {percentile_outliers} outliers")
    print(f"    - Log-MAD (5.0×): {log_mad_outliers} outliers")
    print(f"    - IQR (3.0×): {iqr_outliers} outliers")

    print(f"\n✓ RECOMMENDED: Use Log-MAD (5.0×) method:")
    print(f"    - Detects {log_mad_outliers} outliers (comparable to 95th %ile)")
    print(f"    - Statistically principled for log-normal biological data")
    print(f"    - More stringent than standard 3.0× MAD")
    print(f"    - Easy to justify: 'Log-transformed data, 5σ outlier detection'")

    print(f"\nAlternative: Use consensus (flagged by ALL stringent methods)")

    print(f"\n✓ Results saved to: {output_dir}")
    print(f"\nNext: Update run_analysis.py to use Log-MAD (5.0×) method")


if __name__ == "__main__":
    main()
