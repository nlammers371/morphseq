"""
CEP290 Genotype-Blind Pair Clustering Analysis

Runs bootstrap consensus clustering on each cep290 pair across experiments
20251106, 20251112, 20251113 to identify WT-like (< 0.05 curvature) vs
mutant-like clusters without relying on genotype labels.

Usage:
    python run_cep290_pair_clustering.py

Author: Generated via Claude Code
Date: 2025-12-10
"""

import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

# Add src directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Import configuration
from config import (
    EXPERIMENT_IDS, K_VALUES, N_BOOTSTRAP, BOOTSTRAP_FRAC, RANDOM_SEED,
    WT_THRESHOLD, METRIC_COL, TIME_COL, EMBRYO_ID_COL, PAIR_COL,
    MIN_TIMEPOINTS, GRID_STEP, DTW_WINDOW, OUTPUT_DIR,
    GENERATE_PNG, VERBOSE
)

# Import trajectory analysis functions
from src.analyze.trajectory_analysis import (
    extract_trajectories_df,
    interpolate_to_common_grid_df,
    df_to_trajectories,
    compute_dtw_distance_matrix,
    get_cluster_assignments,
    plot_cluster_trajectories_df
)

# Import data loading
from src.analyze.trajectory_analysis.data_loading import _load_df03_format


def load_experiment_data(experiment_id):
    """Load and prepare data for experiment."""
    print(f"  Loading data for experiment {experiment_id}...")
    
    df = _load_df03_format(experiment_id)
    
    # Filter valid embryos if use_embryo_flag exists
    if 'use_embryo_flag' in df.columns:
        n_before = len(df)
        df = df[df['use_embryo_flag'] == 1].copy()
        print(f"    Filtered by use_embryo_flag: {len(df)}/{n_before} records")
    
    # Handle column naming variations
    if 'baseline_deviation_normalized' not in df.columns:
        if 'normalized_baseline_deviation' in df.columns:
            df['baseline_deviation_normalized'] = df['normalized_baseline_deviation']
    
    print(f"    Loaded {len(df)} records, {df[EMBRYO_ID_COL].nunique()} embryos")
    
    return df


def compute_cluster_characteristics(df_interpolated, df_assignments, k):
    """
    Compute per-cluster statistics.
    
    Parameters
    ----------
    df_interpolated : DataFrame
        Interpolated trajectory data with columns [embryo_id, hpf, metric_value]
    df_assignments : DataFrame
        Cluster assignments from get_cluster_assignments()
    k : int
        Which k value to analyze
    
    Returns
    -------
    df_characteristics : DataFrame
        Columns: cluster_id, n_embryos, mean_value, std_value, is_wt_like, embryo_ids
    """
    cluster_col = f'cluster_k{k}'
    
    # Merge cluster assignments with trajectory data
    df_merged = df_interpolated.merge(
        df_assignments[['embryo_id', cluster_col]],
        on='embryo_id'
    )
    
    rows = []
    
    for cluster_id in sorted(df_merged[cluster_col].unique()):
        df_cluster = df_merged[df_merged[cluster_col] == cluster_id]
        
        # Get unique embryos
        cluster_embryos = df_cluster['embryo_id'].unique()
        
        # Compute cluster mean trajectory (binned mean across timepoints)
        cluster_mean_traj = df_cluster.groupby('hpf')['metric_value'].mean()
        
        # Overall mean and std across all timepoints
        mean_val = cluster_mean_traj.mean()
        std_val = cluster_mean_traj.std()
        
        # Identify WT-like based on threshold
        is_wt_like = mean_val < WT_THRESHOLD
        
        rows.append({
            'cluster_id': cluster_id,
            'n_embryos': len(cluster_embryos),
            'mean_value': mean_val,
            'std_value': std_val,
            'is_wt_like': is_wt_like,
            'embryo_ids': ';'.join(cluster_embryos)
        })
    
    return pd.DataFrame(rows)


def analyze_pair(experiment_id, pair_name, df_pair, output_base_dir):
    """
    Analyze a single pair: extract trajectories, cluster, compute characteristics.
    
    Parameters
    ----------
    experiment_id : str
        Experiment identifier
    pair_name : str
        Pair name (e.g., 'cep290_pair_1')
    df_pair : DataFrame
        Data for this pair
    output_base_dir : Path
        Base output directory
    """
    print(f"\n{'='*80}")
    print(f"  Analyzing {pair_name}")
    print(f"{'='*80}")
    
    # Create output directories
    pair_output_dir = output_base_dir / experiment_id / pair_name
    (pair_output_dir / "data").mkdir(parents=True, exist_ok=True)
    (pair_output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (pair_output_dir / "tables").mkdir(parents=True, exist_ok=True)
    
    # 1. Extract and interpolate trajectories
    print("  Extracting trajectories...")
    df_filtered = extract_trajectories_df(
        df_pair,
        genotype_filter=None,  # Don't filter by genotype (genotype-blind)
        metric_name=METRIC_COL,
        min_timepoints=MIN_TIMEPOINTS
    )

    if len(df_filtered) == 0:
        print(f"    WARNING: No valid embryos found for {pair_name}")
        return

    n_embryos = df_filtered['embryo_id'].nunique()
    print(f"    Found {n_embryos} embryos with >= {MIN_TIMEPOINTS} timepoints")

    print("  Interpolating to common grid...")
    df_interpolated = interpolate_to_common_grid_df(
        df_filtered,
        grid_step=GRID_STEP
    )
    
    # Save interpolated data
    df_interpolated.to_pickle(pair_output_dir / "data" / "df_interpolated.pkl")
    
    # 2. Convert to arrays for DTW
    print("  Converting to arrays for DTW...")
    trajectories, embryo_ids, common_grid = df_to_trajectories(df_interpolated)
    print(f"    Grid: {common_grid.min():.1f} to {common_grid.max():.1f} hpf, {len(common_grid)} points")
    
    # 3. Compute DTW distance matrix
    print("  Computing DTW distance matrix...")
    D = compute_dtw_distance_matrix(trajectories, window=DTW_WINDOW, verbose=VERBOSE)
    np.save(pair_output_dir / "data" / "dtw_distance_matrix.npy", D)
    print(f"    Distance range: [{D[D>0].min():.3f}, {D.max():.3f}]")
    
    # 4. Run clustering for multiple k values
    print(f"\n  Running clustering for k={K_VALUES}...")
    df_assignments, all_results = get_cluster_assignments(
        D, embryo_ids,
        k_values=K_VALUES,
        n_bootstrap=N_BOOTSTRAP,
        bootstrap_frac=BOOTSTRAP_FRAC,
        random_seed=RANDOM_SEED,
        verbose=VERBOSE
    )
    
    # Save cluster assignments
    df_assignments.to_csv(pair_output_dir / "data" / "cluster_assignments.csv", index=False)
    
    # 5. Compute cluster characteristics for each k
    print("\n  Computing cluster characteristics...")
    for k in K_VALUES:
        cluster_chars = compute_cluster_characteristics(df_interpolated, df_assignments, k)
        
        # Save characteristics
        cluster_chars.to_csv(
            pair_output_dir / "tables" / f"cluster_characteristics_k{k}.csv",
            index=False
        )
        
        # Print summary
        print(f"\n    k={k} clusters:")
        for _, row in cluster_chars.iterrows():
            label = "WT-like" if row['is_wt_like'] else "mutant-like"
            print(f"      Cluster {row['cluster_id']}: {label}, n={row['n_embryos']}, "
                  f"mean={row['mean_value']:.4f}, std={row['std_value']:.4f}")
        
        # Save posteriors
        with open(pair_output_dir / "data" / f"posteriors_k{k}.pkl", 'wb') as f:
            pickle.dump(all_results[k]['posteriors'], f)
    
    # 6. Generate plots
    if GENERATE_PNG:
        print("\n  Generating plots...")
        for k in K_VALUES:
            print(f"    Plotting k={k}...")
            labels = df_assignments[f'cluster_k{k}'].values
            
            fig = plot_cluster_trajectories_df(
                df_interpolated,
                labels,
                embryo_ids=embryo_ids,
                show_mean=True,
                figsize=(12, 8)
            )
            
            fig_path = pair_output_dir / "figures" / f"cluster_trajectories_k{k}.png"
            fig.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"      Saved: {fig_path}")
    
    print(f"\n  Completed analysis for {pair_name}")


def main():
    """Main analysis pipeline."""
    print("="*80)
    print("CEP290 GENOTYPE-BLIND PAIR CLUSTERING ANALYSIS")
    print("="*80)
    print(f"Experiments: {EXPERIMENT_IDS}")
    print(f"K values: {K_VALUES}")
    print(f"WT threshold: {WT_THRESHOLD}")
    print("="*80)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each experiment
    for experiment_id in EXPERIMENT_IDS:
        print(f"\n\n{'#'*80}")
        print(f"# EXPERIMENT: {experiment_id}")
        print(f"{'#'*80}")
        
        # Load experiment data
        df = load_experiment_data(experiment_id)
        
        # Check if pair column exists
        if PAIR_COL not in df.columns:
            print(f"  WARNING: '{PAIR_COL}' column not found in experiment {experiment_id}")
            print(f"  Available columns: {df.columns.tolist()}")
            continue
        
        # Get unique pairs
        pairs = sorted(df[PAIR_COL].dropna().unique())
        print(f"\n  Found {len(pairs)} pairs: {pairs}")
        
        # Analyze each pair
        for pair_name in pairs:
            df_pair = df[df[PAIR_COL] == pair_name].copy()
            
            try:
                analyze_pair(experiment_id, pair_name, df_pair, OUTPUT_DIR)
            except Exception as e:
                print(f"\n  ERROR analyzing {pair_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("  1. Review cluster characteristics tables for each pair/experiment")
    print("  2. Compare WT-like vs mutant-like clusters")
    print("  3. For experiment 20251112, validate against trusted genotypes")


if __name__ == '__main__':
    main()
