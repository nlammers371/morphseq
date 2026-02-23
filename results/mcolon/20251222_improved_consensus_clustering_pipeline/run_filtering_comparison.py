"""
Run IQR Distance vs k-NN Filtering Comparison

This script runs a comprehensive comparison of Stage 1 filtering methods:
- k-NN filtering with k=3, 5, 10
- IQR distance filtering with multipliers: 1.5, 2.0, 3.0, 4.0

For each method, it:
1. Applies Stage 1 filtering
2. Runs consensus clustering (100 bootstrap iterations)
3. Applies Stage 2 filtering
4. Generates dendrogram
5. Plots trajectories by cluster and genotype

Hypothesis: IQR distance filtering removes "shitty embryos" that k-NN keeps
because k-NN considers them stable clusters (local stability).

Usage:
    python3 run_filtering_comparison.py

Prerequisites:
    - distance_matrix.npy must exist in the results directory
    - embryo_ids.txt must exist in the results directory
    - df_filtered.pkl (optional, for trajectory plots)

Outputs:
    - filtering_comparison/filtering_comparison_summary.csv
    - filtering_comparison/filtering_comparison.png
    - filtering_comparison/dendrogram_*.png (one per method)
    - filtering_comparison/trajectories_*.png (cluster & genotype, per method)

Created: 2025-12-22
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Setup paths
results_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251222_improved_consensus_clustering_pipeline")
sys.path.insert(0, '/net/trapnell/vol1/home/mdcolon/proj/morphseq')
sys.path.insert(0, str(results_dir))

# Import comparison functions
from test_iqr_vs_knn_filtering import (
    load_distance_matrix,
    test_stage1_filtering_methods,
)

# Output directory
comparison_dir = results_dir / "filtering_comparison"
comparison_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("IQR DISTANCE vs k-NN FILTERING COMPARISON")
print("="*80)
print("\nThis script tests the hypothesis that k-NN filtering keeps 'shitty embryos'")
print("that form stable clusters, while IQR distance filtering removes them.")
print("="*80)


def load_trajectory_data():
    """Load trajectory DataFrame if available for plotting."""
    try:
        df_file = results_dir / 'df_filtered.pkl'
        if df_file.exists():
            df = pd.read_pickle(df_file)
            print(f"\n✓ Loaded trajectory data: {len(df)} rows, {df['embryo_id'].nunique()} embryos")
            return df
        else:
            print(f"\n⚠ Trajectory data not found: {df_file}")
            print("  Trajectory plots will be skipped")
            return None
    except Exception as e:
        print(f"\n⚠ Error loading trajectory data: {e}")
        print("  Trajectory plots will be skipped")
        return None


def main():
    """Run the complete filtering comparison."""
    
    # Step 1: Load distance matrix and embryo IDs
    print("\n" + "="*80)
    print("STEP 1: Load Data")
    print("="*80)
    
    D, embryo_ids = load_distance_matrix(results_dir)
    
    # Step 2: Load trajectory data (optional, for plotting)
    df_filtered = load_trajectory_data()
    
    # Step 3: Run comparison test
    print("\n" + "="*80)
    print("STEP 2: Run Filtering Comparison")
    print("="*80)
    print("\nTesting methods:")
    print("  - k-NN filtering: k=3, 5, 10")
    print("  - IQR distance filtering: multipliers=1.5, 2.0, 3.0, 4.0")
    print("\nFor each method:")
    print("  1. Apply Stage 1 filtering")
    print("  2. Run consensus clustering (100 bootstrap iterations)")
    print("  3. Apply Stage 2 filtering")
    print("  4. Generate dendrogram")
    if df_filtered is not None:
        print("  5. Plot trajectories by cluster and genotype")
    print("\nThis may take 10-20 minutes...")
    print("="*80)
    
    summary_df = test_stage1_filtering_methods(
        D=D,
        embryo_ids=embryo_ids,
        df=df_filtered,
        output_dir=comparison_dir,
        k=3,  # Number of clusters
        n_bootstrap=100
    )
    
    # Step 4: Display results
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    
    print("\n📊 Summary Results:")
    print("-" * 80)
    print(summary_df.to_string(index=False))
    
    # Key findings
    print("\n\n🔍 Key Findings:")
    print("-" * 80)
    
    knn_data = summary_df[summary_df['method'] == 'knn']
    iqr_data = summary_df[summary_df['method'] == 'iqr']
    
    if len(knn_data) > 0:
        print(f"\nk-NN Filtering (local neighborhood):")
        print(f"  Average Stage 1 removal: {knn_data['pct_removed_stage1'].mean():.1f}%")
        print(f"  Average final retention: {knn_data['pct_kept'].mean():.1f}%")
        print(f"  Range: {knn_data['pct_removed_stage1'].min():.1f}% - {knn_data['pct_removed_stage1'].max():.1f}% removed")
    
    if len(iqr_data) > 0:
        print(f"\nIQR Distance Filtering (global outliers):")
        print(f"  Average Stage 1 removal: {iqr_data['pct_removed_stage1'].mean():.1f}%")
        print(f"  Average final retention: {iqr_data['pct_kept'].mean():.1f}%")
        
        most_aggressive = iqr_data.loc[iqr_data['pct_removed_stage1'].idxmax()]
        most_conservative = iqr_data.loc[iqr_data['pct_removed_stage1'].idxmin()]
        
        print(f"  Most aggressive (IQR {most_aggressive['parameter']}×): {most_aggressive['pct_removed_stage1']:.1f}% removed")
        print(f"  Most conservative (IQR {most_conservative['parameter']}×): {most_conservative['pct_removed_stage1']:.1f}% removed")
    
    # Comparison
    if len(knn_data) > 0 and len(iqr_data) > 0:
        diff = iqr_data['pct_removed_stage1'].mean() - knn_data['pct_removed_stage1'].mean()
        print(f"\n📈 Difference:")
        print(f"  IQR removes {diff:.1f}% MORE embryos than k-NN on average")
        
        if diff > 5:
            print(f"\n✓ CONCLUSION: IQR distance filtering is more aggressive at removing")
            print(f"  global outliers that k-NN keeps as 'stable clusters'")
        elif diff < -5:
            print(f"\n⚠ UNEXPECTED: k-NN removed MORE than IQR (check data)")
        else:
            print(f"\n≈ Similar removal rates between methods")
    
    # Output files
    print("\n\n📁 Generated Files:")
    print("-" * 80)
    print(f"Results saved to: {comparison_dir}")
    print("\nFiles created:")
    print("  - filtering_comparison_summary.csv (numerical results)")
    print("  - filtering_comparison.png (visual comparison)")
    print("  - dendrogram_*.png (one per method)")
    if df_filtered is not None:
        print("  - trajectories_cluster_*.png (colored by cluster)")
        print("  - trajectories_genotype_*.png (colored by genotype)")
    
    print("\n" + "="*80)
    print("✓ COMPARISON COMPLETE")
    print("="*80)
    
    return summary_df


if __name__ == '__main__':
    summary_df = main()
