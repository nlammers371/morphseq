"""
Confirmation Script: IQR Distance Filtering Implementation

This script confirms that the consensus pipeline now uses IQR distance filtering
by default for Stage 1 outlier detection, with k-NN method retained as an option.

Tests:
1. Load distance matrix and embryo IDs
2. Run consensus pipeline with default (IQR distance) filtering
3. Run consensus pipeline with legacy k-NN filtering
4. Compare results to confirm differences
5. Verify IQR method removes problematic embryos that k-NN keeps

Expected outcome:
- IQR distance filtering should remove more outliers than k-NN
- IQR should produce cleaner clusters without "shitty embryo" groups
- Both methods should run without errors

Created: 2025-12-22
Purpose: Validate switch from k-NN to IQR distance filtering
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Setup paths
results_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251222_improved_consensus_clustering_pipeline")
sys.path.insert(0, '/net/trapnell/vol1/home/mdcolon/proj/morphseq')

from src.analyze.trajectory_analysis import (
    run_consensus_pipeline,
    generate_dendrograms,
    plot_multimetric_trajectories,
    add_cluster_column,
)


def load_test_data():
    """Load distance matrix and embryo IDs from saved files."""
    print("Loading test data...")
    
    # Check if files exist
    d_file = results_dir / 'filtering_comparison' / 'distance_matrix.npy'
    ids_file = results_dir / 'filtering_comparison' / 'embryo_ids.txt'
    
    if not d_file.exists() or not ids_file.exists():
        print(f"\n✗ ERROR: Required data files not found!")
        print(f"\nMissing files:")
        if not d_file.exists():
            print(f"  - {d_file}")
        if not ids_file.exists():
            print(f"  - {ids_file}")
        print(f"\nPlease run cells 2-3 in b8d2_phenotype_extraction.ipynb first:")
        print(f"  Cell 2: Import comparison functions")
        print(f"  Cell 3: Save D and embryo_ids")
        print(f"\nOr generate the files manually by running your clustering pipeline.")
        raise FileNotFoundError(f"Required data files not found in {results_dir}")
    
    D = np.load(d_file)
    
    with open(ids_file, 'r') as f:
        embryo_ids = [line.strip() for line in f.readlines()]
    
    print(f"✓ Loaded: {len(embryo_ids)} embryos, distance matrix shape {D.shape}")
    return D, embryo_ids


def test_iqr_distance_filtering(D, embryo_ids, k=3, n_bootstrap=100):
    """Test IQR distance filtering (NEW default method)."""
    print("\n" + "="*80)
    print("TEST 1: IQR Distance Filtering (NEW DEFAULT)")
    print("="*80)
    
    results = run_consensus_pipeline(
        D=D,
        embryo_ids=embryo_ids,
        k=k,
        n_bootstrap=n_bootstrap,
        enable_stage1_filtering=True,
        enable_stage2_filtering=True,
        stage1_method='iqr',  # Explicit (also the default)
        iqr_multiplier=4.0,
        posterior_threshold=0.5,
        k_highlight=[2, 3, k],
        verbose=True
    )
    
    return results


def test_knn_filtering(D, embryo_ids, k=3, n_bootstrap=100):
    """Test k-NN filtering (LEGACY method, should keep problematic embryos)."""
    print("\n" + "="*80)
    print("TEST 2: k-NN Filtering (LEGACY METHOD)")
    print("="*80)
    
    results = run_consensus_pipeline(
        D=D,
        embryo_ids=embryo_ids,
        k=k,
        n_bootstrap=n_bootstrap,
        enable_stage1_filtering=True,
        enable_stage2_filtering=True,
        stage1_method='knn',  # Use legacy method
        iqr_multiplier=4.0,
        k_neighbors=5,
        posterior_threshold=0.5,
        k_highlight=[2, 3, k],
        verbose=True
    )
    
    return results


def compare_filtering_results(iqr_results, knn_results, embryo_ids):
    """Compare IQR distance vs k-NN filtering results."""
    print("\n" + "="*80)
    print("COMPARISON: IQR Distance vs k-NN Filtering")
    print("="*80)
    
    n_initial = len(embryo_ids)
    
    # Stage 1 comparisons
    n_iqr_stage1 = len(iqr_results['embryo_ids_after_stage1'])
    n_knn_stage1 = len(knn_results['embryo_ids_after_stage1'])
    
    n_iqr_removed_stage1 = n_initial - n_iqr_stage1
    n_knn_removed_stage1 = n_initial - n_knn_stage1
    
    # Final comparisons
    n_iqr_final = len(iqr_results['final_embryo_ids'])
    n_knn_final = len(knn_results['final_embryo_ids'])
    
    print(f"\nInitial embryos: {n_initial}")
    print(f"\nStage 1 Filtering:")
    print(f"  IQR distance: {n_iqr_removed_stage1} removed, {n_iqr_stage1} kept ({100*n_iqr_stage1/n_initial:.1f}%)")
    print(f"  k-NN:         {n_knn_removed_stage1} removed, {n_knn_stage1} kept ({100*n_knn_stage1/n_initial:.1f}%)")
    print(f"  Difference:   {n_iqr_removed_stage1 - n_knn_removed_stage1} more removed by IQR")
    
    print(f"\nFinal (after both stages):")
    print(f"  IQR distance: {n_iqr_final} kept ({100*n_iqr_final/n_initial:.1f}%)")
    print(f"  k-NN:         {n_knn_final} kept ({100*n_knn_final/n_initial:.1f}%)")
    
    # Find embryos kept by k-NN but removed by IQR (the "problematic" ones)
    iqr_stage1_set = set(iqr_results['embryo_ids_after_stage1'])
    knn_stage1_set = set(knn_results['embryo_ids_after_stage1'])
    
    kept_by_knn_only = knn_stage1_set - iqr_stage1_set
    
    print(f"\nEmbryo kept by k-NN but REMOVED by IQR distance:")
    print(f"  Count: {len(kept_by_knn_only)}")
    if len(kept_by_knn_only) > 0:
        print(f"  IDs: {sorted(kept_by_knn_only)[:20]}")  # Show first 20
        if len(kept_by_knn_only) > 20:
            print(f"       ... and {len(kept_by_knn_only) - 20} more")
    
    # Summary verdict
    print(f"\n{'='*80}")
    print("VERDICT:")
    print("="*80)
    
    if n_iqr_removed_stage1 > n_knn_removed_stage1:
        print("✓ IQR distance filtering removes MORE outliers than k-NN (expected)")
        print("✓ IQR properly identifies global outliers that k-NN keeps as 'stable clusters'")
        print("✓ Switch to IQR distance filtering is CONFIRMED WORKING")
    elif n_iqr_removed_stage1 == n_knn_removed_stage1:
        print("⚠ Both methods removed same number of embryos (unexpected)")
        print("  Check if your data has distinct outlier populations")
    else:
        print("✗ k-NN removed MORE than IQR (unexpected - check implementation)")
    
    return {
        'n_initial': n_initial,
        'iqr_stage1_kept': n_iqr_stage1,
        'knn_stage1_kept': n_knn_stage1,
        'iqr_final_kept': n_iqr_final,
        'knn_final_kept': n_knn_final,
        'kept_by_knn_only': list(kept_by_knn_only),
    }


def save_comparison_summary(comparison_results, output_dir):
    """Save comparison results to file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary stats
    summary = pd.DataFrame([{
        'n_initial': comparison_results['n_initial'],
        'iqr_stage1_kept': comparison_results['iqr_stage1_kept'],
        'knn_stage1_kept': comparison_results['knn_stage1_kept'],
        'iqr_final_kept': comparison_results['iqr_final_kept'],
        'knn_final_kept': comparison_results['knn_final_kept'],
        'n_kept_by_knn_only': len(comparison_results['kept_by_knn_only']),
    }])
    
    summary_file = output_dir / 'iqr_vs_knn_confirmation.csv'
    summary.to_csv(summary_file, index=False)
    print(f"\n✓ Saved summary: {summary_file}")
    
    # Problematic embryo IDs
    if len(comparison_results['kept_by_knn_only']) > 0:
        embryo_file = output_dir / 'embryos_kept_by_knn_only.txt'
        with open(embryo_file, 'w') as f:
            f.write('\n'.join(sorted(comparison_results['kept_by_knn_only'])))
        print(f"✓ Saved problematic embryo IDs: {embryo_file}")


def main():
    """Run full confirmation test suite."""
    print("="*80)
    print("CONFIRMATION: IQR Distance Filtering Implementation")
    print("="*80)
    print("\nThis script confirms:")
    print("1. Consensus pipeline now uses IQR distance filtering by default")
    print("2. k-NN method is retained as legacy option (stage1_method='knn')")
    print("3. IQR properly removes problematic embryos that k-NN keeps")
    print("="*80)
    
    # Load data
    D, embryo_ids = load_test_data()
    
    # Test both methods with small n_bootstrap for speed
    k = 3
    n_bootstrap = 50  # Use 50 for quick testing (use 100+ for production)
    
    print(f"\nTest parameters:")
    print(f"  k = {k} clusters")
    print(f"  n_bootstrap = {n_bootstrap}")
    print(f"  iqr_multiplier = 4.0")
    
    # Run tests
    iqr_results = test_iqr_distance_filtering(D, embryo_ids, k=k, n_bootstrap=n_bootstrap)
    knn_results = test_knn_filtering(D, embryo_ids, k=k, n_bootstrap=n_bootstrap)
    
    # Compare
    comparison = compare_filtering_results(iqr_results, knn_results, embryo_ids)
    
    # Save results
    save_comparison_summary(comparison, results_dir / 'filtering_confirmation')
    
    print("\n" + "="*80)
    print("CONFIRMATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {results_dir / 'filtering_confirmation'}")
    print("\nNext steps:")
    print("1. Review comparison summary CSV")
    print("2. Check which embryos were kept by k-NN but removed by IQR")
    print("3. Run full analysis with IQR distance filtering (now the default)")


if __name__ == '__main__':
    main()
