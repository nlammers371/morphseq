"""
Test K Selection Pipeline with Multimetric Plotting

Tests the k-selection workflow using pre-computed distance matrix:
1. Load distance matrix + embryo IDs 
2. Apply light IQR filtering (2.0×)
3. Evaluate k range [2, 3, 4, 5, 6]
4. Generate membership quality plots using plot_multimetric_trajectories
5. Output summary metrics

Uses existing data from:
- filtering_comparison/distance_matrix.npy
- filtering_comparison/embryo_ids.txt

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

# Output directory for k-selection results
output_dir = results_dir / "k_selection_results"
output_dir.mkdir(parents=True, exist_ok=True)


def add_membership_column(df, classification, column_name='membership'):
    """
    Add membership category column to DataFrame.
    
    Maps embryo_id → category (core/uncertain/outlier) to enable
    using plot_multimetric_trajectories with color_by_grouping='membership'.
    """
    embryo_to_cat = dict(zip(classification['embryo_ids'], classification['category']))
    df = df.copy()
    df[column_name] = df['embryo_id'].map(embryo_to_cat)
    return df


def load_data():
    """Load distance matrix, embryo IDs, and trajectory DataFrame."""
    print("Loading data...")
    
    # Load distance matrix
    D = np.load(results_dir / 'filtering_comparison' / 'distance_matrix.npy')
    
    # Load embryo IDs
    with open(results_dir / 'filtering_comparison' / 'embryo_ids.txt', 'r') as f:
        embryo_ids = [line.strip() for line in f.readlines()]
    
    print(f"✓ Distance matrix: {D.shape}")
    print(f"✓ Embryo IDs: {len(embryo_ids)}")
    
    # Load trajectory DataFrame
    from src.analyze.trajectory_analysis.data_loading import load_experiment_dataframe
    
    experiment_ids = ['20251119', '20251121', '20251104', '20251125']
    dfs = []
    for exp_id in experiment_ids:
        try:
            df_exp = load_experiment_dataframe(exp_id, format_version='df03')
            df_exp['experiment_id'] = exp_id
            dfs.append(df_exp)
            print(f"  ✓ Loaded {exp_id}: {df_exp['embryo_id'].nunique()} embryos")
        except Exception as e:
            print(f"  ⚠ Failed to load {exp_id}: {e}")
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Filter to only embryos in our distance matrix
    df = df[df['embryo_id'].isin(embryo_ids)]
    print(f"✓ Trajectory DataFrame: {len(df)} rows, {df['embryo_id'].nunique()} embryos")
    
    return D, embryo_ids, df


def apply_iqr_filtering(D, embryo_ids, iqr_multiplier=2.0):
    """Apply IQR distance filtering to remove global outliers."""
    from src.analyze.trajectory_analysis import identify_outliers
    
    print(f"\nApplying IQR filtering (multiplier={iqr_multiplier})...")
    
    outlier_ids, inlier_ids, info = identify_outliers(
        D, embryo_ids,
        method='iqr',
        threshold=iqr_multiplier,
        verbose=True
    )
    
    # Filter distance matrix
    inlier_indices = info['inlier_indices']
    D_filtered = D[np.ix_(inlier_indices, inlier_indices)]
    
    print(f"✓ Filtered: {len(outlier_ids)} removed, {len(inlier_ids)} kept")
    
    return D_filtered, inlier_ids, outlier_ids


def evaluate_k_with_plots(D, embryo_ids, df, k_range=[2, 3, 4, 5, 6], n_bootstrap=100):
    """
    Evaluate multiple k values and generate membership plots for each.
    
    For each k:
    1. Run bootstrap clustering
    2. Compute posteriors + classification
    3. Add cluster + membership columns to df
    4. Plot multimetric trajectories colored by membership
    """
    from src.analyze.trajectory_analysis import (
        run_bootstrap_hierarchical,
        analyze_bootstrap_results,
        generate_dendrograms,
        add_cluster_column,
        plot_multimetric_trajectories,
    )
    from src.analyze.trajectory_analysis.cluster_classification import classify_membership_2d
    from src.analyze.trajectory_analysis.bootstrap_clustering import compute_coassociation_matrix
    from sklearn.metrics import silhouette_score
    
    results_by_k = {}
    summary_data = []
    
    for k in k_range:
        print(f"\n{'='*70}")
        print(f"Evaluating k = {k}")
        print('='*70)
        
        # 1. Run bootstrap clustering
        bootstrap_results = run_bootstrap_hierarchical(
            D=D,
            k=k,
            embryo_ids=embryo_ids,
            n_bootstrap=n_bootstrap,
            verbose=True
        )
        
        # 2. Compute posteriors
        posteriors = analyze_bootstrap_results(bootstrap_results)
        
        # 3. Classify membership quality
        classification = classify_membership_2d(
            max_p=posteriors['max_p'],
            log_odds_gap=posteriors['log_odds_gap'],
            modal_cluster=posteriors['modal_cluster'],
            embryo_ids=posteriors['embryo_ids']
        )
        
        # 4. Compute silhouette score
        try:
            sil_score = silhouette_score(D, posteriors['modal_cluster'], metric='precomputed')
        except:
            sil_score = np.nan
        
        # 5. Compute summary metrics
        categories = classification['category']
        n_total = len(categories)
        n_core = np.sum(categories == 'core')
        n_uncertain = np.sum(categories == 'uncertain')
        n_outlier = np.sum(categories == 'outlier')
        
        metrics = {
            'k': k,
            'n_embryos': n_total,
            'n_core': n_core,
            'n_uncertain': n_uncertain,
            'n_outlier': n_outlier,
            'pct_core': 100.0 * n_core / n_total,
            'pct_uncertain': 100.0 * n_uncertain / n_total,
            'pct_outlier': 100.0 * n_outlier / n_total,
            'mean_max_p': posteriors['max_p'].mean(),
            'mean_entropy': posteriors['entropy'].mean(),
            'silhouette': sil_score,
        }
        summary_data.append(metrics)
        
        print(f"\nk={k} Summary:")
        print(f"  Core: {n_core} ({metrics['pct_core']:.1f}%)")
        print(f"  Uncertain: {n_uncertain} ({metrics['pct_uncertain']:.1f}%)")
        print(f"  Outlier: {n_outlier} ({metrics['pct_outlier']:.1f}%)")
        print(f"  Mean max_p: {metrics['mean_max_p']:.3f}")
        print(f"  Silhouette: {sil_score:.3f}")
        
        # 6. Generate dendrogram info for cluster assignments
        consensus_matrix = compute_coassociation_matrix(bootstrap_results, verbose=False)
        _, dendro_info = generate_dendrograms(
            D, embryo_ids,
            coassociation_matrix=consensus_matrix,
            k_highlight=[k],
            verbose=False
        )
        plt.close()  # Close dendrogram figure
        
        # 7. Add cluster + membership columns to df
        df_k = df[df['embryo_id'].isin(embryo_ids)].copy()
        df_k = add_cluster_column(df_k, dendro_info=dendro_info, k=k, column_name='cluster')
        df_k = add_membership_column(df_k, classification, column_name='membership')
        
        # 8. Plot multimetric trajectories by cluster, colored by membership
        print(f"\n  Generating membership plots for k={k}...")
        
        try:
            fig = plot_multimetric_trajectories(
                df_k,
                metrics=['baseline_deviation_normalized', 'total_length_um'],
                col_by='cluster',
                color_by_grouping='membership',
                x_col='predicted_stage_hpf',
                metric_labels={
                    'baseline_deviation_normalized': 'Curvature (normalized)',
                    'total_length_um': 'Body Length (μm)',
                },
                title=f'k={k}: Membership Quality by Cluster (Core={n_core}, Uncertain={n_uncertain}, Outlier={n_outlier})',
                x_label='Time (hpf)',
                backend='matplotlib',
                bin_width=2.0,
            )
            
            # Save figure
            fig_path = output_dir / f'k{k}_membership_trajectories.png'
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"  ✓ Saved: {fig_path}")
            plt.close(fig)
            
        except Exception as e:
            print(f"  ⚠ Plot failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Store results
        results_by_k[k] = {
            'bootstrap_results': bootstrap_results,
            'posteriors': posteriors,
            'classification': classification,
            'dendro_info': dendro_info,
            'metrics': metrics,
        }
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    return results_by_k, summary_df


def plot_k_comparison(summary_df, save_path=None):
    """Plot quality metrics comparison across k values."""
    k_values = summary_df['k'].values
    
    # Find best k (highest core %)
    best_k = summary_df.loc[summary_df['pct_core'].idxmax(), 'k']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Membership % vs k
    ax = axes[0, 0]
    ax.plot(k_values, summary_df['pct_core'], 'o-', color='green', 
            linewidth=2.5, markersize=10, label='Core')
    ax.plot(k_values, summary_df['pct_uncertain'], 's-', color='orange', 
            linewidth=2.5, markersize=10, label='Uncertain')
    ax.plot(k_values, summary_df['pct_outlier'], '^-', color='red', 
            linewidth=2.5, markersize=10, label='Outlier')
    ax.axvline(best_k, color='blue', linestyle='--', alpha=0.5, label=f'Best k={best_k}')
    ax.set_xlabel('k (number of clusters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Membership Quality vs K', fontsize=13, fontweight='bold')
    ax.set_xticks(k_values)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # 2. Mean max_p vs k
    ax = axes[0, 1]
    ax.plot(k_values, summary_df['mean_max_p'], 'o-', color='steelblue', 
            linewidth=2.5, markersize=10)
    ax.axvline(best_k, color='blue', linestyle='--', alpha=0.5)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.3, label='Threshold (0.5)')
    ax.set_xlabel('k (number of clusters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Max Posterior', fontsize=12, fontweight='bold')
    ax.set_title('Cluster Confidence vs K', fontsize=13, fontweight='bold')
    ax.set_xticks(k_values)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 3. Mean entropy vs k
    ax = axes[1, 0]
    ax.plot(k_values, summary_df['mean_entropy'], 'o-', color='coral', 
            linewidth=2.5, markersize=10)
    ax.axvline(best_k, color='blue', linestyle='--', alpha=0.5)
    ax.set_xlabel('k (number of clusters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Entropy (bits)', fontsize=12, fontweight='bold')
    ax.set_title('Assignment Ambiguity vs K (lower = better)', fontsize=13, fontweight='bold')
    ax.set_xticks(k_values)
    ax.grid(True, alpha=0.3)
    
    # 4. Silhouette score vs k
    ax = axes[1, 1]
    ax.plot(k_values, summary_df['silhouette'], 'o-', color='purple', 
            linewidth=2.5, markersize=10)
    ax.axvline(best_k, color='blue', linestyle='--', alpha=0.5)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('k (number of clusters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    ax.set_title('Cluster Separation vs K (higher = better)', fontsize=13, fontweight='bold')
    ax.set_xticks(k_values)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'K Selection Analysis — Recommended k = {best_k}', 
                 fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig, best_k


def main():
    """Run complete k-selection pipeline test."""
    print("="*70)
    print("K SELECTION PIPELINE TEST")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    
    # Step 1: Load data
    D, embryo_ids, df = load_data()
    
    # Step 2: Apply IQR filtering (2.0×)
    D_filtered, filtered_ids, outlier_ids = apply_iqr_filtering(
        D, embryo_ids, iqr_multiplier=2.0
    )
    
    # Filter df to match
    df_filtered = df[df['embryo_id'].isin(filtered_ids)]
    
    # Step 3: Evaluate k range
    print("\n" + "="*70)
    print("EVALUATING K RANGE")
    print("="*70)
    
    results_by_k, summary_df = evaluate_k_with_plots(
        D=D_filtered,
        embryo_ids=filtered_ids,
        df=df_filtered,
        k_range=[2, 3, 4, 5, 6],
        n_bootstrap=100
    )
    
    # Step 4: Plot k comparison
    print("\n" + "="*70)
    print("K SELECTION SUMMARY")
    print("="*70)
    
    fig, best_k = plot_k_comparison(
        summary_df, 
        save_path=output_dir / 'k_selection_comparison.png'
    )
    plt.show()
    
    # Save summary CSV
    summary_df.to_csv(output_dir / 'k_selection_summary.csv', index=False)
    print(f"\n✓ Saved: {output_dir / 'k_selection_summary.csv'}")
    
    # Print summary table
    print("\n" + summary_df.to_string(index=False))
    
    print("\n" + "="*70)
    print(f"RECOMMENDATION: k = {best_k}")
    print("="*70)
    print(f"\nGenerated files:")
    for k in summary_df['k']:
        print(f"  - k{k}_membership_trajectories.png")
    print(f"  - k_selection_comparison.png")
    print(f"  - k_selection_summary.csv")
    
    return results_by_k, summary_df, best_k


if __name__ == '__main__':
    results_by_k, summary_df, best_k = main()
