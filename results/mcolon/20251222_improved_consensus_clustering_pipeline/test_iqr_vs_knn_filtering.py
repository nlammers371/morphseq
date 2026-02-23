"""
Compare IQR Distance Filtering vs k-NN Filtering for Stage 1 Outlier Detection

Tests the hypothesis that k-NN filtering keeps "shitty embryos" that form stable 
clusters, while IQR distance filtering would remove them more effectively.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths
results_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251222_improved_consensus_clustering_pipeline")
results_dir.mkdir(parents=True, exist_ok=True)

# Import analysis tools
import sys
sys.path.insert(0, '/net/trapnell/vol1/home/mdcolon/proj/morphseq')

from src.analyze.trajectory_analysis import (
    run_consensus_pipeline,
    identify_outliers,
    plot_multimetric_trajectories,
    generate_dendrograms,
    add_cluster_column,
)

# ============================================================================
# STEP 1: Save D and embryo_ids for reproducibility
# ============================================================================

def save_distance_matrix(D, embryo_ids, output_dir):
    """Save distance matrix and embryo IDs to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as .npy (binary, fast)
    np.save(output_dir / 'distance_matrix.npy', D)
    
    # Save embryo IDs as text
    with open(output_dir / 'embryo_ids.txt', 'w') as f:
        f.write('\n'.join(embryo_ids))
    
    print(f"✓ Saved distance matrix: {output_dir / 'distance_matrix.npy'}")
    print(f"✓ Saved embryo IDs: {output_dir / 'embryo_ids.txt'}")
    print(f"  Shape: {D.shape}")
    print(f"  Distance range: [{D[D > 0].min():.3f}, {D.max():.3f}]")


def load_distance_matrix(input_dir):
    """Load distance matrix and embryo IDs from disk."""
    input_dir = Path(input_dir)
    
    D = np.load(input_dir / 'distance_matrix.npy')
    
    with open(input_dir / 'embryo_ids.txt', 'r') as f:
        embryo_ids = [line.strip() for line in f.readlines()]
    
    print(f"✓ Loaded distance matrix: {D.shape}")
    print(f"✓ Loaded embryo IDs: {len(embryo_ids)}")
    
    return D, embryo_ids


# ============================================================================
# STEP 2: Test IQR Distance vs k-NN Filtering
# ============================================================================

def test_stage1_filtering_methods(D, embryo_ids, df, output_dir, k=3, n_bootstrap=100):
    """
    Compare IQR distance filtering vs k-NN filtering.
    
    Tests multiple parameter combinations and generates dendrograms + trajectory plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define test cases
    test_cases = [
        # k-NN filtering (original)
        {'method': 'knn', 'k_neighbors': 3, 'iqr_multiplier': None, 'label': 'kNN_k3'},
        {'method': 'knn', 'k_neighbors': 5, 'iqr_multiplier': None, 'label': 'kNN_k5'},
        {'method': 'knn', 'k_neighbors': 10, 'iqr_multiplier': None, 'label': 'kNN_k10'},
        
        # IQR distance filtering (new)
        {'method': 'iqr', 'k_neighbors': None, 'iqr_multiplier': 1.5, 'label': 'IQR_1.5x'},
        {'method': 'iqr', 'k_neighbors': None, 'iqr_multiplier': 2.0, 'label': 'IQR_2.0x'},
        {'method': 'iqr', 'k_neighbors': None, 'iqr_multiplier': 3.0, 'label': 'IQR_3.0x'},
        {'method': 'iqr', 'k_neighbors': None, 'iqr_multiplier': 4.0, 'label': 'IQR_4.0x'},
    ]
    
    results_summary = []
    
    for test_case in test_cases:
        print("\n" + "="*70)
        print(f"Testing: {test_case['label']}")
        print("="*70)
        
        # Determine filtering parameters
        if test_case['method'] == 'knn':
            enable_stage1 = True
            k_neighbors = test_case['k_neighbors']
            # For k-NN, we don't use IQR in Stage 1, but still need a value for Stage 2
            iqr_multiplier = 4.0
        else:  # iqr
            # Use IQR distance filtering directly with identify_outliers
            enable_stage1 = False  # We'll do manual filtering
            k_neighbors = 5  # Unused
            iqr_multiplier = test_case['iqr_multiplier']
        
        # Apply Stage 1 filtering manually if using IQR method
        if test_case['method'] == 'iqr':
            print(f"\n  Applying IQR distance filtering (multiplier={iqr_multiplier})...")
            outlier_ids, inlier_ids, outlier_info = identify_outliers(
                D, embryo_ids, 
                method='iqr', 
                threshold=iqr_multiplier,
                verbose=True
            )
            
            # Create filtered distance matrix
            inlier_indices = [i for i, eid in enumerate(embryo_ids) if eid in inlier_ids]
            D_filtered = D[np.ix_(inlier_indices, inlier_indices)]
            embryo_ids_filtered = inlier_ids
            
            print(f"  ✓ Filtered: {len(outlier_ids)} outliers removed, {len(inlier_ids)} kept")
        else:
            D_filtered = D
            embryo_ids_filtered = embryo_ids
        
        # Run consensus pipeline
        try:
            results = run_consensus_pipeline(
                D=D_filtered,
                embryo_ids=embryo_ids_filtered,
                k=k,
                n_bootstrap=n_bootstrap,
                enable_stage1_filtering=enable_stage1,
                enable_stage2_filtering=True,
                iqr_multiplier=4.0,  # Stage 2 always uses 4.0
                k_neighbors=k_neighbors,
                posterior_threshold=0.5,
                k_highlight=[2, 3, k],
                verbose=True
            )
            
            # Extract summary metrics
            n_initial = len(embryo_ids)
            n_after_stage1 = len(results['embryo_ids_after_stage1'])
            n_final = len(results['final_embryo_ids'])
            n_removed_stage1 = n_initial - n_after_stage1
            n_removed_stage2 = n_after_stage1 - n_final
            
            # Store results
            test_result = {
                'label': test_case['label'],
                'method': test_case['method'],
                'parameter': test_case['k_neighbors'] if test_case['method'] == 'knn' else test_case['iqr_multiplier'],
                'n_initial': n_initial,
                'n_after_stage1': n_after_stage1,
                'n_final': n_final,
                'n_removed_stage1': n_removed_stage1,
                'n_removed_stage2': n_removed_stage2,
                'pct_removed_stage1': 100 * n_removed_stage1 / n_initial,
                'pct_kept': 100 * n_final / n_initial,
            }
            results_summary.append(test_result)
            
            # Generate dendrogram
            fig_dendro, _ = generate_dendrograms(
                results['final_D'],
                results['final_embryo_ids'],
                coassociation_matrix=results['final_consensus_matrix'],
                k_highlight=[2, 3, k],
                title=f"Consensus Dendrogram - {test_case['label']}\n({n_final}/{n_initial} embryos kept)",
                figsize=(16, 8),
                verbose=False
            )
            fig_dendro.savefig(output_dir / f"dendrogram_{test_case['label']}.png", dpi=300, bbox_inches='tight')
            plt.close(fig_dendro)
            
            # Generate trajectory plots
            df_with_clusters = add_cluster_column(
                df,
                dendro_info=results['dendrogram_info_final'],
                k=k,
                column_name='consensus_cluster'
            )
            
            # Filter to only kept embryos
            df_kept = df_with_clusters[df_with_clusters['embryo_id'].isin(results['final_embryo_ids'])]
            
            # Plot by cluster
            fig_traj = plot_multimetric_trajectories(
                df_kept,
                metrics=['baseline_deviation_normalized', 'total_length_um'],
                col_by='consensus_cluster',
                color_by_grouping='consensus_cluster',
                x_col='predicted_stage_hpf',
                metric_labels={
                    'baseline_deviation_normalized': 'Curvature (normalized)',
                    'total_length_um': 'Body Length (μm)',
                },
                title=f'Trajectories by Cluster - {test_case["label"]} (k={k})',
                x_label='Time (hpf)',
                backend='matplotlib',
                bin_width=2.0,
            )
            fig_traj.savefig(output_dir / f"trajectories_cluster_{test_case['label']}.png", dpi=300, bbox_inches='tight')
            plt.close(fig_traj)
            
            # Plot by genotype
            fig_geno = plot_multimetric_trajectories(
                df_kept,
                metrics=['baseline_deviation_normalized', 'total_length_um'],
                col_by='consensus_cluster',
                color_by_grouping='genotype',
                x_col='predicted_stage_hpf',
                metric_labels={
                    'baseline_deviation_normalized': 'Curvature (normalized)',
                    'total_length_um': 'Body Length (μm)',
                },
                title=f'Trajectories by Genotype - {test_case["label"]} (k={k})',
                x_label='Time (hpf)',
                backend='matplotlib',
                bin_width=2.0,
            )
            fig_geno.savefig(output_dir / f"trajectories_genotype_{test_case['label']}.png", dpi=300, bbox_inches='tight')
            plt.close(fig_geno)
            
            print(f"\n✓ Test case complete: {test_case['label']}")
            print(f"  Stage 1 removed: {n_removed_stage1}/{n_initial} ({test_result['pct_removed_stage1']:.1f}%)")
            print(f"  Stage 2 removed: {n_removed_stage2}/{n_after_stage1}")
            print(f"  Final kept: {n_final}/{n_initial} ({test_result['pct_kept']:.1f}%)")
            
        except Exception as e:
            print(f"\n✗ Test case FAILED: {test_case['label']}")
            print(f"  Error: {str(e)}")
            test_result = {
                'label': test_case['label'],
                'method': test_case['method'],
                'parameter': test_case['k_neighbors'] if test_case['method'] == 'knn' else test_case['iqr_multiplier'],
                'error': str(e),
            }
            results_summary.append(test_result)
    
    # Create comparison summary
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(output_dir / 'filtering_comparison_summary.csv', index=False)
    print(f"\n✓ Summary saved: {output_dir / 'filtering_comparison_summary.csv'}")
    
    # Create comparison visualization
    plot_filtering_comparison(summary_df, output_dir)
    
    return summary_df


def plot_filtering_comparison(summary_df, output_dir):
    """Create comparison plots for different filtering methods."""
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Stage 1 filtering effectiveness
    ax = axes[0, 0]
    knn_data = summary_df[summary_df['method'] == 'knn']
    iqr_data = summary_df[summary_df['method'] == 'iqr']
    
    if len(knn_data) > 0:
        ax.plot(knn_data['parameter'], knn_data['pct_removed_stage1'], 
                marker='o', markersize=8, linewidth=2, label='k-NN', color='steelblue')
    if len(iqr_data) > 0:
        ax.plot(iqr_data['parameter'], iqr_data['pct_removed_stage1'], 
                marker='s', markersize=8, linewidth=2, label='IQR Distance', color='coral')
    
    ax.set_xlabel('Parameter (k or IQR multiplier)', fontsize=11, fontweight='bold')
    ax.set_ylabel('% Embryos Removed (Stage 1)', fontsize=11, fontweight='bold')
    ax.set_title('Stage 1 Filtering Effectiveness', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Final retention rate
    ax = axes[0, 1]
    if len(knn_data) > 0:
        ax.plot(knn_data['parameter'], knn_data['pct_kept'], 
                marker='o', markersize=8, linewidth=2, label='k-NN', color='steelblue')
    if len(iqr_data) > 0:
        ax.plot(iqr_data['parameter'], iqr_data['pct_kept'], 
                marker='s', markersize=8, linewidth=2, label='IQR Distance', color='coral')
    
    ax.set_xlabel('Parameter (k or IQR multiplier)', fontsize=11, fontweight='bold')
    ax.set_ylabel('% Embryos Kept (Final)', fontsize=11, fontweight='bold')
    ax.set_title('Final Data Retention', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Bar chart comparison
    ax = axes[1, 0]
    x_pos = np.arange(len(summary_df))
    ax.bar(x_pos, summary_df['pct_removed_stage1'], color='orange', alpha=0.7, 
           label='Stage 1 Removed', edgecolor='black')
    ax.bar(x_pos, summary_df['n_removed_stage2'] / summary_df['n_initial'] * 100, 
           bottom=summary_df['pct_removed_stage1'], color='red', alpha=0.7,
           label='Stage 2 Removed', edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(summary_df['label'], rotation=45, ha='right')
    ax.set_ylabel('% Embryos Removed', fontsize=11, fontweight='bold')
    ax.set_title('Removal by Stage', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Summary table
    ax = axes[1, 1]
    ax.axis('off')
    table_data = summary_df[['label', 'n_removed_stage1', 'n_removed_stage2', 'n_final']].values
    table = ax.table(cellText=table_data, 
                     colLabels=['Method', 'Stage 1', 'Stage 2', 'Final'],
                     cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax.set_title('Filtering Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'filtering_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Comparison plot saved: {output_dir / 'filtering_comparison.png'}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("IQR Distance vs k-NN Filtering Comparison")
    print("="*70)
    
    # This script is meant to be imported and run from the notebook
    # See test_iqr_vs_knn_filtering.ipynb for usage
    pass
