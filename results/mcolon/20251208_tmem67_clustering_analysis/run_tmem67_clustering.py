"""
tmem67 Genotyping Analysis via Bootstrap Consensus Clustering

This script identifies tmem67 mutant embryos when genotype labels are unreliable
using bootstrap consensus clustering with posterior probability analysis.

Key features:
- Tests k=3,4,5,6 clusters
- Uses DTW distance for trajectory similarity
- Computes posterior probabilities for robust cluster assignments
- Identifies mutant clusters by average baseline deviation > 0.05
- Generates interactive visualizations focusing on cluster means

Author: Generated via Claude Code
Date: 2025-12-08
"""

import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

# Add src directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Import configuration
from config import (
    EXPERIMENT_ID, OUTPUT_DIR, CURV_DIR, META_DIR,
    K_VALUES, N_BOOTSTRAP, BOOTSTRAP_FRAC, RANDOM_SEED,
    METRIC_NAME, TIME_COL, EMBRYO_ID_COL, MIN_TIMEPOINTS, GRID_STEP,
    DTW_WINDOW, THRESHOLD_MAX_P, THRESHOLD_LOG_ODDS_GAP, THRESHOLD_OUTLIER_MAX_P,
    MUTANT_THRESHOLD, GENERATE_PNG, GENERATE_PLOTLY, VERBOSE
)

# Import analysis functions from src
from src.analyze.trajectory_analysis import (
    extract_trajectories_df,
    interpolate_to_common_grid_df,
    df_to_trajectories,
    compute_dtw_distance_matrix,
    run_bootstrap_hierarchical,
    analyze_bootstrap_results,
    classify_membership_2d,
    get_classification_summary,
)

# Import local utilities
from cluster_analysis_utils import (
    compute_cluster_mean_trajectory,
    compute_trajectory_average,
    compute_composite_k_scores
)
from plotting_utils import (
    plot_trajectories_by_cluster,
    plot_cluster_means,
    plot_interactive_cluster_means,
    plot_metrics_vs_k,
    plot_optimal_k_recommendation
)


def load_and_prepare_data(experiment_id):
    """
    Load curvature + metadata and merge.

    Parameters
    ----------
    experiment_id : str
        Experiment identifier

    Returns
    -------
    df : DataFrame
        Merged DataFrame with all necessary columns
    """
    print(f"  Loading curvature metrics...")
    curv_file = CURV_DIR / f"curvature_metrics_{experiment_id}.csv"
    df_curv = pd.read_csv(curv_file)
    print(f"    Loaded {len(df_curv)} curvature records")

    print(f"  Loading metadata...")
    meta_file = META_DIR / f"df03_final_output_with_latents_{experiment_id}.csv"
    df_meta = pd.read_csv(meta_file)
    print(f"    Loaded {len(df_meta)} metadata records")

    print(f"  Merging on snip_id...")
    df = df_curv.merge(df_meta, on='snip_id', how='inner')
    print(f"    Merged: {len(df)} total records")

    # Filter valid embryos (if use_embryo_flag column exists)
    if 'use_embryo_flag' in df.columns:
        n_before = len(df)
        df = df[df['use_embryo_flag'] == 1].copy()
        print(f"    Filtered by use_embryo_flag: {len(df)}/{n_before} records")

    # Handle column naming
    if 'baseline_deviation_normalized' not in df.columns:
        if 'normalized_baseline_deviation' in df.columns:
            df['baseline_deviation_normalized'] = df['normalized_baseline_deviation']

    return df


def extract_and_align_trajectories(df):
    """
    Extract trajectories and align to common grid.

    Parameters
    ----------
    df : DataFrame
        Merged data

    Returns
    -------
    df_interpolated : DataFrame
        DataFrame with aligned trajectories
    trajectories : list
        List of arrays for DTW
    embryo_ids : list
        List of embryo identifiers
    common_grid : ndarray
        Time grid (hpf values)
    """
    print(f"  Extracting trajectories (NO genotype filter)...")

    # Prepare dataframe for trajectory extraction
    df_prep = df.copy()

    # Rename columns to match expected names if needed
    if TIME_COL not in df_prep.columns and 'predicted_stage_hpf' in df_prep.columns:
        df_prep[TIME_COL] = df_prep['predicted_stage_hpf']

    if METRIC_NAME not in df_prep.columns:
        if 'baseline_deviation_normalized' in df_prep.columns:
            df_prep[METRIC_NAME] = df_prep['baseline_deviation_normalized']
        elif 'normalized_baseline_deviation' in df_prep.columns:
            df_prep[METRIC_NAME] = df_prep['normalized_baseline_deviation']

    if EMBRYO_ID_COL not in df_prep.columns and 'embryo_id' in df_prep.columns:
        df_prep[EMBRYO_ID_COL] = df_prep['embryo_id']

    # Filter to required columns and drop NaNs
    required_cols = [EMBRYO_ID_COL, TIME_COL, METRIC_NAME]
    df_prep = df_prep[required_cols].dropna()

    # Group by embryo and filter by minimum timepoints
    embryo_counts = df_prep.groupby(EMBRYO_ID_COL).size()
    valid_embryos = embryo_counts[embryo_counts >= MIN_TIMEPOINTS].index.tolist()
    df_filtered = df_prep[df_prep[EMBRYO_ID_COL].isin(valid_embryos)].copy()

    n_embryos = df_filtered[EMBRYO_ID_COL].nunique()
    print(f"    Extracted {n_embryos} embryos with >= {MIN_TIMEPOINTS} timepoints")

    # Interpolate to common grid
    print(f"  Interpolating to common grid (step={GRID_STEP} hpf)...")

    # Rename columns for interpolation function
    df_for_interp = df_filtered.rename(columns={
        TIME_COL: 'hpf',
        METRIC_NAME: 'metric_value'
    })

    # Create common time grid
    all_times = df_for_interp['hpf'].values
    time_min = np.floor(all_times.min() / GRID_STEP) * GRID_STEP
    time_max = np.ceil(all_times.max() / GRID_STEP) * GRID_STEP
    common_grid = np.arange(time_min, time_max + GRID_STEP, GRID_STEP)

    # Interpolate each embryo to common grid
    interpolated_records = []

    for embryo_id in valid_embryos:
        embryo_data = df_for_interp[df_for_interp[EMBRYO_ID_COL] == embryo_id].sort_values('hpf')

        if len(embryo_data) < 2:
            continue

        # Interpolate
        interp_values = np.interp(common_grid, embryo_data['hpf'].values,
                                 embryo_data['metric_value'].values)

        for t, v in zip(common_grid, interp_values):
            interpolated_records.append({
                'embryo_id': embryo_id,
                'hpf': t,
                'metric_value': v
            })

    df_interpolated = pd.DataFrame(interpolated_records)

    # Convert to arrays for DTW
    trajectories = []
    embryo_ids_out = []

    for embryo_id in df_interpolated['embryo_id'].unique():
        embryo_vals = df_interpolated[df_interpolated['embryo_id'] == embryo_id].sort_values('hpf')['metric_value'].values
        trajectories.append(embryo_vals)
        embryo_ids_out.append(embryo_id)

    print(f"    Interpolated {len(trajectories)} trajectories")
    print(f"    Time grid: {common_grid.min():.1f} to {common_grid.max():.1f} hpf, n={len(common_grid)} points")

    return df_interpolated, trajectories, embryo_ids_out, common_grid


def compute_dtw_distances(trajectories):
    """
    Compute pairwise DTW distance matrix.

    Parameters
    ----------
    trajectories : list
        List of trajectory arrays

    Returns
    -------
    D : ndarray
        Distance matrix (n × n)
    """
    print(f"  Computing DTW distance matrix (window={DTW_WINDOW})...")

    D = compute_dtw_distance_matrix(
        trajectories,
        window=DTW_WINDOW,
        verbose=VERBOSE
    )

    print(f"    Distance matrix shape: {D.shape}")
    print(f"    Distance range: [{D[D>0].min():.3f}, {D.max():.3f}]")

    return D


def run_clustering_for_k(D, k, embryo_ids):
    """
    Run bootstrap clustering for a single k value.

    Parameters
    ----------
    D : ndarray
        Distance matrix
    k : int
        Number of clusters
    embryo_ids : list
        List of embryo identifiers

    Returns
    -------
    bootstrap_results : dict
        Output from run_bootstrap_hierarchical()
    posteriors : dict
        Output from analyze_bootstrap_results()
    classification : dict
        Output from classify_membership_2d()
    """
    print(f"  Running bootstrap clustering (n={N_BOOTSTRAP}, frac={BOOTSTRAP_FRAC})...")

    # Bootstrap clustering
    bootstrap_results = run_bootstrap_hierarchical(
        D, k, embryo_ids,
        n_bootstrap=N_BOOTSTRAP,
        frac=BOOTSTRAP_FRAC,
        random_state=RANDOM_SEED,
        verbose=False  # Suppress per-iteration output
    )

    print(f"  Computing posterior probabilities...")

    # Posterior analysis
    posteriors = analyze_bootstrap_results(bootstrap_results)

    print(f"  Classifying membership quality...")

    # Membership classification
    classification = classify_membership_2d(
        posteriors['max_p'],
        posteriors['log_odds_gap'],
        posteriors['modal_cluster'],
        embryo_ids=posteriors['embryo_ids'],
        threshold_max_p=THRESHOLD_MAX_P,
        threshold_log_odds_gap=THRESHOLD_LOG_ODDS_GAP,
        threshold_outlier_max_p=THRESHOLD_OUTLIER_MAX_P
    )

    return bootstrap_results, posteriors, classification


def identify_mutant_clusters(df_interpolated, posteriors, classification):
    """
    Identify which clusters are putative mutants.

    For each cluster:
    1. Extract trajectories for cluster members
    2. Compute cluster mean trajectory
    3. Calculate average value across all timepoints
    4. Flag if average > MUTANT_THRESHOLD

    Parameters
    ----------
    df_interpolated : DataFrame
        Interpolated trajectory data
    posteriors : dict
        Output from analyze_bootstrap_results()
    classification : dict
        Output from classify_membership_2d()

    Returns
    -------
    cluster_characteristics : DataFrame
        Per-cluster statistics and mutant flags
    """
    modal_clusters = posteriors['modal_cluster']
    n_clusters = posteriors['n_clusters']
    embryo_ids = posteriors['embryo_ids']

    rows = []

    for cluster_id in range(n_clusters):
        # Get embryos in this cluster
        cluster_mask = modal_clusters == cluster_id
        cluster_embryo_ids = [eid for eid, m in zip(embryo_ids, cluster_mask) if m]

        # Filter DataFrame to cluster embryos
        df_cluster = df_interpolated[
            df_interpolated['embryo_id'].isin(cluster_embryo_ids)
        ]

        # Compute mean trajectory (binned mean across timepoints)
        mean_trajectory = compute_cluster_mean_trajectory(df_cluster)

        # Compute average value (mean of mean trajectory)
        cluster_avg = compute_trajectory_average(mean_trajectory)

        # Classify as mutant
        is_mutant = cluster_avg > MUTANT_THRESHOLD

        # Get membership quality breakdown
        cluster_category = classification['category'][cluster_mask]
        n_core = np.sum(cluster_category == 'core')
        n_uncertain = np.sum(cluster_category == 'uncertain')
        n_outlier = np.sum(cluster_category == 'outlier')

        rows.append({
            'cluster_id': cluster_id,
            'n_embryos': len(cluster_embryo_ids),
            'cluster_average': cluster_avg,
            'is_putative_mutant': is_mutant,
            'n_core': n_core,
            'n_uncertain': n_uncertain,
            'n_outlier': n_outlier,
            'core_fraction': n_core / len(cluster_embryo_ids) if len(cluster_embryo_ids) > 0 else 0,
            'embryo_ids': ';'.join(cluster_embryo_ids)
        })

    return pd.DataFrame(rows)


def compute_optimal_k_metrics(D, posteriors, classification):
    """
    Compute all metrics for optimal k selection.

    Parameters
    ----------
    D : ndarray
        Distance matrix
    posteriors : dict
        Output from analyze_bootstrap_results()
    classification : dict
        Output from classify_membership_2d()

    Returns
    -------
    metrics : dict
        Dictionary with metrics:
            - avg_max_p
            - avg_entropy
            - core_fraction
            - silhouette
            - avg_within_cluster_dist (optional)
            - avg_between_cluster_dist (optional)
    """
    max_p = posteriors['max_p']
    entropy = posteriors['entropy']
    modal_cluster = posteriors['modal_cluster']
    categories = classification['category']

    # Basic posterior metrics
    avg_max_p = np.mean(max_p)
    avg_entropy = np.mean(entropy)
    core_fraction = np.sum(categories == 'core') / len(categories)

    # Silhouette score
    silhouette = silhouette_score(D, modal_cluster, metric='precomputed')

    return {
        'avg_max_p': avg_max_p,
        'avg_entropy': avg_entropy,
        'core_fraction': core_fraction,
        'silhouette': silhouette
    }


def main():
    """Main analysis pipeline."""
    print("="*80)
    print("TMEM67 GENOTYPING ANALYSIS VIA CONSENSUS CLUSTERING")
    print("="*80)
    print(f"Experiment: {EXPERIMENT_ID}")
    print(f"K values: {K_VALUES}")
    print(f"Mutant threshold: {MUTANT_THRESHOLD}")
    print("="*80)

    # Create output directories
    exp_output_dir = OUTPUT_DIR / EXPERIMENT_ID
    (exp_output_dir / "data").mkdir(parents=True, exist_ok=True)
    (exp_output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (exp_output_dir / "figures" / "comparison").mkdir(parents=True, exist_ok=True)

    # STAGE 1: Load data
    print("\n[1/8] Loading data...")
    df = load_and_prepare_data(EXPERIMENT_ID)
    print(f"  Total records: {len(df)}")
    print(f"  Unique embryos: {df[EMBRYO_ID_COL].nunique()}")

    # STAGE 2: Extract and align trajectories
    print("\n[2/8] Extracting and aligning trajectories...")
    df_interpolated, trajectories, embryo_ids, common_grid = extract_and_align_trajectories(df)

    # Save intermediate results
    df_interpolated.to_pickle(exp_output_dir / "data" / "df_interpolated.pkl")
    print(f"  Saved interpolated data")

    # STAGE 3: Compute DTW distances
    print("\n[3/8] Computing DTW distance matrix...")
    D = compute_dtw_distances(trajectories)
    np.save(exp_output_dir / "data" / "dtw_distance_matrix.npy", D)
    print(f"  Saved DTW distance matrix")

    # STAGE 4-7: Loop over k values
    all_k_results = {}

    for k in K_VALUES:
        print(f"\n{'='*80}")
        print(f"ANALYZING k={k}")
        print(f"{'='*80}")

        # Create output directory for this k
        k_fig_dir = exp_output_dir / "figures" / f"k{k}"
        k_fig_dir.mkdir(parents=True, exist_ok=True)

        # STAGE 4: Bootstrap clustering
        print(f"\n[4/8] Running bootstrap clustering for k={k}...")
        bootstrap_results, posteriors, classification = run_clustering_for_k(D, k, embryo_ids)

        # Save results
        with open(exp_output_dir / "data" / f"bootstrap_results_k{k}.pkl", 'wb') as f:
            pickle.dump(bootstrap_results, f)
        with open(exp_output_dir / "data" / f"posteriors_k{k}.pkl", 'wb') as f:
            pickle.dump(posteriors, f)

        # STAGE 5: Posterior analysis summary
        print(f"\n[5/8] Posterior analysis summary")
        print(f"  Avg max_p: {np.mean(posteriors['max_p']):.3f}")
        print(f"  Avg entropy: {np.mean(posteriors['entropy']):.3f}")

        # STAGE 6: Membership classification summary
        summary = get_classification_summary(classification)
        print(f"\n[6/8] Membership classification")
        print(f"  Core: {summary['n_core']}/{summary['n_total']} ({summary['core_fraction']:.1%})")
        print(f"  Uncertain: {summary['n_uncertain']}/{summary['n_total']} ({summary['uncertain_fraction']:.1%})")
        print(f"  Outlier: {summary['n_outlier']}/{summary['n_total']} ({summary['outlier_fraction']:.1%})")

        # STAGE 7: Identify mutant clusters
        print(f"\n[7/8] Identifying mutant clusters...")
        cluster_chars = identify_mutant_clusters(df_interpolated, posteriors, classification)
        cluster_chars.to_csv(
            exp_output_dir / "tables" / f"cluster_characteristics_k{k}.csv",
            index=False
        )

        mutant_clusters = cluster_chars[cluster_chars['is_putative_mutant']]
        print(f"  Putative mutant clusters: {mutant_clusters['cluster_id'].tolist()}")

        # Print cluster details
        for _, row in cluster_chars.iterrows():
            status = "MUTANT" if row['is_putative_mutant'] else "WT-like"
            print(f"    Cluster {row['cluster_id']}: {status}, n={row['n_embryos']}, avg={row['cluster_average']:.4f}")

        # Extract mutant embryo IDs
        mutant_embryo_ids = []
        for _, row in mutant_clusters.iterrows():
            mutant_embryo_ids.extend(row['embryo_ids'].split(';'))

        pd.DataFrame({'embryo_id': mutant_embryo_ids}).to_csv(
            exp_output_dir / "tables" / f"mutant_embryos_k{k}.csv",
            index=False
        )
        print(f"  Total putative mutants: {len(mutant_embryo_ids)} embryos")

        # STAGE 8: Generate plots
        print(f"\n[8/8] Generating visualizations...")

        if GENERATE_PNG:
            plot_trajectories_by_cluster(
                df_interpolated, posteriors, classification, cluster_chars,
                output_path=k_fig_dir / "trajectories_by_cluster.png"
            )
            plot_cluster_means(
                df_interpolated, posteriors, cluster_chars,
                output_path=k_fig_dir / "cluster_means.png"
            )

        if GENERATE_PLOTLY:
            plot_interactive_cluster_means(
                df_interpolated, posteriors, cluster_chars,
                output_path=k_fig_dir / "interactive_cluster_means.html"
            )

        # Compute optimal k metrics
        metrics = compute_optimal_k_metrics(D, posteriors, classification)

        all_k_results[k] = {
            'posteriors': posteriors,
            'classification': classification,
            'cluster_characteristics': cluster_chars,
            'metrics': metrics,
            'n_mutant_embryos': len(mutant_embryo_ids)
        }

    # Generate cross-k comparison plots and tables
    print(f"\n{'='*80}")
    print("OPTIMAL K SELECTION ANALYSIS")
    print(f"{'='*80}")

    # Create comparison table
    comparison_rows = []
    for k in K_VALUES:
        metrics = all_k_results[k]['metrics']
        comparison_rows.append({
            'k': k,
            'avg_max_p': metrics['avg_max_p'],
            'avg_entropy': metrics['avg_entropy'],
            'core_fraction': metrics['core_fraction'],
            'silhouette': metrics['silhouette'],
            'n_mutant_embryos': all_k_results[k]['n_mutant_embryos']
        })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(
        exp_output_dir / "tables" / "optimal_k_analysis.csv",
        index=False
    )

    print("\nMetrics across k values:")
    print(comparison_df.to_string(index=False))

    # Generate comparison plots
    plot_metrics_vs_k(
        comparison_df,
        output_path=exp_output_dir / "figures" / "comparison" / "posterior_metrics_vs_k.png"
    )

    # Compute composite scores
    scores = compute_composite_k_scores(comparison_df)
    optimal_k = K_VALUES[np.argmax([scores[k] for k in K_VALUES])]

    print(f"\nComposite scores:")
    for k in K_VALUES:
        print(f"  k={k}: {scores[k]:.3f}")
    print(f"\n>>> RECOMMENDED k = {optimal_k} <<<")

    plot_optimal_k_recommendation(
        comparison_df, scores, optimal_k,
        output_path=exp_output_dir / "figures" / "comparison" / "optimal_k_recommendation.png"
    )

    # Print summary of recommended k
    print(f"\n{'='*80}")
    print(f"RECOMMENDED CLUSTERING: k={optimal_k}")
    print(f"{'='*80}")

    recommended_clusters = all_k_results[optimal_k]['cluster_characteristics']
    mutant_clusters_recommended = recommended_clusters[recommended_clusters['is_putative_mutant']]

    print(f"\nCluster summary for k={optimal_k}:")
    for _, row in recommended_clusters.iterrows():
        status = "MUTANT" if row['is_putative_mutant'] else "WT-like"
        print(f"  Cluster {row['cluster_id']}: {status}, n={row['n_embryos']}, avg={row['cluster_average']:.4f}, core_frac={row['core_fraction']:.2%}")

    print(f"\nPutative mutant clusters: {mutant_clusters_recommended['cluster_id'].tolist()}")
    print(f"Total mutant embryos: {all_k_results[optimal_k]['n_mutant_embryos']}")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {exp_output_dir}")
    print(f"\nKey outputs:")
    print(f"  - Cluster characteristics: tables/cluster_characteristics_k{optimal_k}.csv")
    print(f"  - Mutant embryo IDs: tables/mutant_embryos_k{optimal_k}.csv")
    print(f"  - Interactive plot: figures/k{optimal_k}/interactive_cluster_means.html")
    print(f"  - Optimal k analysis: tables/optimal_k_analysis.csv")


if __name__ == '__main__':
    main()
