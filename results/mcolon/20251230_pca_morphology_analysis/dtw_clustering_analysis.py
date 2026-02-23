"""
Phase B: DTW Clustering Analysis on PCA Delta Trajectories

Tests whether DTW clustering on PCA delta trajectories recapitulates the known
phenotypes (CE, HTA, BA-rescue, non-penetrant) using:
1. DTW distance computation on PCA delta columns
2. Outlier detection and removal
3. K-selection pipeline with bootstrap consensus (k=2 to k=9)
4. Dendrogram with phenotype/genotype/pair category overlays
5. Cluster flow Sankey diagram
6. Quantitative cluster-phenotype agreement analysis

Saves intermediate files for manual k exploration:
- df_with_pca_deltas.parquet: Full DataFrame with PCA columns
- distance_matrix.npy: DTW distance matrix
- distance_matrix_filtered.npy: After outlier removal
- k_results.pkl: Full k-selection results
- cluster_assignments.csv: All k assignments

Usage:
    # Open parallel computing environment first, then:
    python dtw_clustering_analysis.py

Author: Generated via Claude Code
Date: 2025-12-30
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root))

# Import from trajectory analysis
from src.analyze.trajectory_analysis import (
    # PCA utilities
    fit_pca_on_embeddings,
    transform_embeddings_to_pca,
    compute_wt_reference_by_time,
    subtract_wt_reference,
    # DTW distance (one-liner)
    compute_trajectory_distances,
    # Outlier detection
    remove_outliers_from_distance_matrix,
    # K-selection pipeline
    run_k_selection_with_plots,
    plot_cluster_flow,
    # Dendrogram with categories
    plot_dendrogram_with_categories,
    # 3D plotting
    plot_3d_scatter,
    # Trajectory plotting
    plot_multimetric_trajectories,
)
from src.analyze.trajectory_analysis.plot_config import PHENOTYPE_COLORS, PHENOTYPE_ORDER

# Import from existing b9d2 analysis
sys.path.insert(0, str(Path(__file__).parent.parent / '20251228_b9d2_phenotype_comparisons'))
from b9d2_phenotype_comparison import (
    load_all_phenotypes,
    load_experiment_data,
    extract_wildtype_embryos,
)
from b9d2_phenotype_distribution_by_pair import prepare_phenotype_dataframe

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT_IDS = ['20251121', '20251125']
OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(__file__).parent / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

# PCA settings
N_PCA_COMPONENTS = 3
BIN_WIDTH = 2.0  # hours for WT reference

# DTW settings
SAKOE_CHIBA_RADIUS = 20  # Large radius for flexibility
K_RANGE = list(range(2, 10))  # k=2 to k=9
N_BOOTSTRAP = 100  # Bootstrap iterations

# Outlier detection settings
OUTLIER_METHOD = 'iqr'
OUTLIER_THRESHOLD = 2.0  # IQR multiplier

# Column names
TIME_COL = 'predicted_stage_hpf'
EMBRYO_ID_COL = 'embryo_id'


# =============================================================================
# Main Analysis
# =============================================================================

def main():
    print("=" * 70)
    print("PHASE B: DTW CLUSTERING ON PCA DELTA TRAJECTORIES")
    print("=" * 70)

    # =========================================================================
    # Step 1: Load Data (same as Phase A)
    # =========================================================================
    print("\n[Step 1/10] Loading data...")
    phenotypes = load_all_phenotypes()
    df = load_experiment_data()
    wildtype_ids = extract_wildtype_embryos(df, phenotypes)

    print(f"  Total rows: {len(df)}")
    print(f"  Total embryos: {df[EMBRYO_ID_COL].nunique()}")

    # =========================================================================
    # Step 2: Assign Phenotype Labels (same as Phase A)
    # =========================================================================
    print("\n[Step 2/10] Assigning phenotype labels...")
    df = prepare_phenotype_dataframe(df, phenotypes, wildtype_ids, EXPERIMENT_IDS)

    print(f"  Phenotype distribution:")
    for pheno in PHENOTYPE_ORDER:
        if pheno in df['phenotype'].unique():
            n = df[df['phenotype'] == pheno][EMBRYO_ID_COL].nunique()
            print(f"    {pheno}: {n} embryos")

    # =========================================================================
    # Step 3: Fit PCA on z_mu_b columns (same as Phase A)
    # =========================================================================
    print("\n[Step 3/10] Fitting PCA on VAE embeddings...")
    z_mu_cols = [c for c in df.columns if 'z_mu_b' in c]
    print(f"  Using {len(z_mu_cols)} z_mu_b columns")

    pca, scaler, z_mu_cols = fit_pca_on_embeddings(
        df, z_mu_cols, n_components=N_PCA_COMPONENTS
    )

    # Print variance explained
    print(f"  Variance explained: {pca.explained_variance_ratio_.sum():.1%}")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"    PC{i+1}: {var:.1%}")

    # =========================================================================
    # Step 4: Transform to PCA space (same as Phase A)
    # =========================================================================
    print("\n[Step 4/10] Transforming embeddings to PCA space...")
    df = transform_embeddings_to_pca(df, pca, scaler, z_mu_cols)

    pca_cols = [f'PCA_{i+1}' for i in range(N_PCA_COMPONENTS)]
    print(f"  Added columns: {pca_cols}")

    # =========================================================================
    # Step 5: Compute WT Reference (same as Phase A)
    # =========================================================================
    print("\n[Step 5/10] Computing wildtype reference...")
    wt_reference = compute_wt_reference_by_time(
        df,
        pca_cols=pca_cols,
        time_col=TIME_COL,
        wt_embryo_ids=wildtype_ids,
        embryo_id_col=EMBRYO_ID_COL,
        bin_width=BIN_WIDTH,
    )
    print(f"  Reference time bins: {len(wt_reference)}")
    print(f"  Time range: {wt_reference['time_bin'].min():.1f} - {wt_reference['time_bin'].max():.1f} hpf")

    # =========================================================================
    # Step 6: Subtract WT Reference (same as Phase A)
    # =========================================================================
    print("\n[Step 6/10] Subtracting WT reference...")
    df = subtract_wt_reference(
        df,
        wt_reference,
        pca_cols=pca_cols,
        time_col=TIME_COL,
        bin_width=BIN_WIDTH,
    )

    delta_cols = [f'PCA_{i+1}_delta' for i in range(N_PCA_COMPONENTS)]
    print(f"  Added columns: {delta_cols}")

    # Save DataFrame with PCA deltas
    df_path = DATA_DIR / 'df_with_pca_deltas.parquet'
    df.to_parquet(df_path, index=False)
    print(f"  Saved DataFrame: {df_path}")

    # =========================================================================
    # Step 7: Compute DTW Distance Matrix
    # =========================================================================
    print("\n[Step 7/10] Computing DTW distance matrix on PCA deltas...")
    print(f"  Metrics: {delta_cols}")
    print(f"  Sakoe-Chiba radius: {SAKOE_CHIBA_RADIUS}")
    print(f"  This may take a while...")

    D, embryo_ids, time_grid = compute_trajectory_distances(
        df,
        metrics=delta_cols,
        time_col=TIME_COL,
        embryo_id_col=EMBRYO_ID_COL,
        normalize=True,
        sakoe_chiba_radius=SAKOE_CHIBA_RADIUS,
        verbose=True,
    )

    print(f"  Distance matrix shape: {D.shape}")
    print(f"  Embryo count: {len(embryo_ids)}")
    print(f"  Time grid: {len(time_grid)} points ({time_grid.min():.1f} - {time_grid.max():.1f} hpf)")

    # Save raw distance matrix
    np.save(DATA_DIR / 'distance_matrix.npy', D)
    np.save(DATA_DIR / 'time_grid.npy', time_grid)
    with open(DATA_DIR / 'embryo_ids.pkl', 'wb') as f:
        pickle.dump(embryo_ids, f)
    print(f"  Saved distance matrix and metadata to {DATA_DIR}")

    # =========================================================================
    # Step 8: Outlier Detection and Removal
    # =========================================================================
    print("\n[Step 8/10] Detecting and removing outliers...")
    D_filtered, embryo_ids_filtered, filtering_stats = remove_outliers_from_distance_matrix(
        D,
        embryo_ids,
        outlier_detection_method=OUTLIER_METHOD,
        outlier_threshold=OUTLIER_THRESHOLD,
        verbose=True,
    )

    print(f"  Original: {len(embryo_ids)} embryos")
    print(f"  After filtering: {len(embryo_ids_filtered)} embryos")
    print(f"  Outliers removed: {len(embryo_ids) - len(embryo_ids_filtered)}")

    # Save filtered distance matrix
    np.save(DATA_DIR / 'distance_matrix_filtered.npy', D_filtered)
    with open(DATA_DIR / 'embryo_ids_filtered.pkl', 'wb') as f:
        pickle.dump(embryo_ids_filtered, f)
    with open(DATA_DIR / 'filtering_stats.pkl', 'wb') as f:
        pickle.dump(filtering_stats, f)

    # =========================================================================
    # Step 9: K-Selection Pipeline with Bootstrap
    # =========================================================================
    print("\n[Step 9/10] Running k-selection pipeline (k=2 to k=9)...")
    print(f"  Bootstrap iterations: {N_BOOTSTRAP}")
    print(f"  Method: kmedoids")

    k_selection_dir = OUTPUT_DIR / 'k_selection'
    k_selection_dir.mkdir(parents=True, exist_ok=True)

    k_results = run_k_selection_with_plots(
        df=df,
        D=D_filtered,
        embryo_ids=embryo_ids_filtered,
        output_dir=k_selection_dir,
        plotting_metrics=delta_cols,  # Show PCA delta trajectories
        k_range=K_RANGE,
        n_bootstrap=N_BOOTSTRAP,
        method='kmedoids',
        x_col=TIME_COL,
        metric_labels={
            'PCA_1_delta': 'PC1 Delta (WT-subtracted)',
            'PCA_2_delta': 'PC2 Delta (WT-subtracted)',
            'PCA_3_delta': 'PC3 Delta (WT-subtracted)',
        },
        verbose=True,
    )

    # Save k_results to data directory as well
    with open(DATA_DIR / 'k_results.pkl', 'wb') as f:
        pickle.dump(k_results, f)
    print(f"  Saved k_results.pkl to {DATA_DIR}")

    # =========================================================================
    # Step 10: Visualizations
    # =========================================================================
    print("\n[Step 10/10] Creating visualizations...")

    # 10a. Cluster flow Sankey diagram
    print("\n  Creating cluster flow Sankey diagram...")
    fig_flow = plot_cluster_flow(
        k_results,
        k_range=K_RANGE,
        title="B9D2 PCA-Delta DTW Cluster Flow: How Clusters Split as k Increases",
        output_path=OUTPUT_DIR / 'cluster_flow_sankey.html',
    )

    # 10b. Dendrogram with category overlays (phenotype, genotype, pair)
    print("\n  Creating dendrogram with category overlays...")
    
    # Build category DataFrame
    category_df = df[df[EMBRYO_ID_COL].isin(embryo_ids_filtered)].drop_duplicates(EMBRYO_ID_COL)[
        [EMBRYO_ID_COL, 'phenotype', 'genotype', 'pair']
    ].copy()
    category_df = category_df.rename(columns={EMBRYO_ID_COL: 'embryo_id'})
    
    fig_dendro, dendro_info = plot_dendrogram_with_categories(
        D_filtered,
        embryo_ids_filtered,
        category_df=category_df,
        category_cols=['phenotype', 'genotype', 'pair'],
        k_highlight=[2, 3, 4, 5, 6, 7, 8, 9],
        linkage_method='average',
        title='DTW Clustering on PCA Deltas with Phenotype/Genotype/Pair Overlays',
        figsize=(20, 14),
        spacer_height=0.7,
        save_path=OUTPUT_DIR / 'dendrogram_with_categories.png',
        verbose=True,
    )
    plt.close(fig_dendro)

    # 10c. 3D scatter: DTW clusters vs phenotypes at k=4
    print("\n  Creating 3D scatter comparisons at k=4...")
    
    # Add cluster assignments to df for k=4
    k_chosen = 4
    cluster_map = k_results['clustering_by_k'][k_chosen]['assignments']['embryo_to_cluster']
    df['dtw_cluster_k4'] = df[EMBRYO_ID_COL].map(cluster_map)
    
    # Filter to only include embryos with cluster assignments
    df_clustered = df[df['dtw_cluster_k4'].notna()].copy()
    df_clustered['dtw_cluster_k4'] = df_clustered['dtw_cluster_k4'].astype(int).astype(str)
    
    # Plot clusters
    fig_clusters = plot_3d_scatter(
        df_clustered,
        coords=delta_cols,
        color_by='dtw_cluster_k4',
        line_by=EMBRYO_ID_COL,
        min_points_per_line=10,
        title=f'DTW Clusters (k={k_chosen}) in PCA Delta Space',
        output_path=OUTPUT_DIR / f'dtw_clusters_k{k_chosen}_pca_delta',
    )
    
    # Plot phenotypes for comparison
    fig_pheno = plot_3d_scatter(
        df_clustered,
        coords=delta_cols,
        color_by='phenotype',
        color_palette=PHENOTYPE_COLORS,
        color_order=PHENOTYPE_ORDER,
        line_by=EMBRYO_ID_COL,
        min_points_per_line=10,
        title='True Phenotypes in PCA Delta Space (Ground Truth)',
        output_path=OUTPUT_DIR / 'phenotypes_pca_delta',
    )

    # =========================================================================
    # Cluster-Phenotype Agreement Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("CLUSTER-PHENOTYPE AGREEMENT ANALYSIS")
    print("=" * 70)

    # Get one row per embryo with cluster and phenotype
    df_embryos = df_clustered.drop_duplicates(EMBRYO_ID_COL)[
        [EMBRYO_ID_COL, 'dtw_cluster_k4', 'phenotype']
    ].copy()

    # Confusion matrix
    confusion = pd.crosstab(
        df_embryos['phenotype'],
        df_embryos['dtw_cluster_k4'],
        rownames=['True Phenotype'],
        colnames=['DTW Cluster'],
        margins=True
    )
    print("\n" + "=" * 50)
    print("CONFUSION MATRIX: DTW Clusters vs True Phenotypes")
    print("=" * 50)
    print(confusion)

    # Save confusion matrix
    confusion.to_csv(OUTPUT_DIR / 'confusion_matrix_k4.csv')

    # Per-cluster purity
    print("\n" + "=" * 50)
    print("CLUSTER PURITY ANALYSIS")
    print("=" * 50)

    cluster_purities = []
    purity_report = []
    for cluster_id in sorted(df_embryos['dtw_cluster_k4'].unique()):
        cluster_df = df_embryos[df_embryos['dtw_cluster_k4'] == cluster_id]
        phenotype_counts = cluster_df['phenotype'].value_counts()
        dominant_pheno = phenotype_counts.index[0]
        dominant_count = phenotype_counts.iloc[0]
        total_in_cluster = len(cluster_df)
        purity = dominant_count / total_in_cluster

        cluster_purities.append(purity)
        purity_report.append({
            'cluster': cluster_id,
            'n_embryos': total_in_cluster,
            'dominant_phenotype': dominant_pheno,
            'dominant_count': dominant_count,
            'purity': purity,
            'breakdown': dict(phenotype_counts),
        })

        print(f"\nCluster {cluster_id} (n={total_in_cluster}):")
        print(f"  Dominant phenotype: {dominant_pheno} ({dominant_count}/{total_in_cluster})")
        print(f"  Purity: {purity:.2%}")
        print(f"  Breakdown: {dict(phenotype_counts)}")

    mean_purity = np.mean(cluster_purities)
    print(f"\nMean cluster purity: {mean_purity:.2%}")

    # Phenotype-to-cluster mapping
    print("\n" + "=" * 50)
    print("PHENOTYPE-TO-CLUSTER MAPPING")
    print("=" * 50)

    pheno_report = []
    for pheno in PHENOTYPE_ORDER:
        if pheno in df_embryos['phenotype'].values:
            pheno_df = df_embryos[df_embryos['phenotype'] == pheno]
            cluster_dist = pheno_df['dtw_cluster_k4'].value_counts()
            main_cluster = cluster_dist.index[0]
            main_count = cluster_dist.iloc[0]
            purity = main_count / len(pheno_df)

            pheno_report.append({
                'phenotype': pheno,
                'n_embryos': len(pheno_df),
                'main_cluster': main_cluster,
                'main_count': main_count,
                'cluster_purity': purity,
                'distribution': dict(cluster_dist),
            })

            print(f"\n{pheno} (n={len(pheno_df)}):")
            print(f"  Main cluster: {main_cluster} ({main_count}/{len(pheno_df)})")
            print(f"  Cluster purity: {purity:.2%}")
            print(f"  Distribution: {dict(cluster_dist)}")

    # Save reports
    pd.DataFrame(purity_report).to_csv(OUTPUT_DIR / 'cluster_purity_report_k4.csv', index=False)
    pd.DataFrame(pheno_report).to_csv(OUTPUT_DIR / 'phenotype_cluster_mapping_k4.csv', index=False)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE B COMPLETE")
    print("=" * 70)
    print(f"\nData files saved to: {DATA_DIR}")
    print(f"  - df_with_pca_deltas.parquet")
    print(f"  - distance_matrix.npy")
    print(f"  - distance_matrix_filtered.npy")
    print(f"  - embryo_ids.pkl / embryo_ids_filtered.pkl")
    print(f"  - k_results.pkl")
    print(f"  - filtering_stats.pkl")

    print(f"\nVisualization files saved to: {OUTPUT_DIR}")
    print(f"  - cluster_flow_sankey.html")
    print(f"  - dendrogram_with_categories.png")
    print(f"  - dtw_clusters_k4_pca_delta.html")
    print(f"  - phenotypes_pca_delta.html")
    print(f"  - k_selection/ (trajectory plots per k)")

    print(f"\nAnalysis files:")
    print(f"  - confusion_matrix_k4.csv")
    print(f"  - cluster_purity_report_k4.csv")
    print(f"  - phenotype_cluster_mapping_k4.csv")

    print(f"\nSummary statistics:")
    print(f"  Total embryos clustered: {len(df_embryos)}")
    print(f"  Best k (from pipeline): {k_results['best_k']}")
    print(f"  Mean cluster purity at k=4: {mean_purity:.2%}")

    print("\n" + "=" * 70)
    print("TO EXPLORE DIFFERENT K VALUES:")
    print("=" * 70)
    print(f"""
# Load data for manual exploration:
import pickle
import pandas as pd
import numpy as np

data_dir = Path('{DATA_DIR}')
df = pd.read_parquet(data_dir / 'df_with_pca_deltas.parquet')
with open(data_dir / 'k_results.pkl', 'rb') as f:
    k_results = pickle.load(f)

# Get cluster assignments for any k (2-9):
k = 5  # Change as needed
cluster_map = k_results['clustering_by_k'][k]['assignments']['embryo_to_cluster']
df['dtw_cluster'] = df['embryo_id'].map(cluster_map)

# View cluster sizes:
print(k_results['clustering_by_k'][k]['assignments']['cluster_to_embryos'].keys())
""")
    print("=" * 70)


if __name__ == '__main__':
    main()
