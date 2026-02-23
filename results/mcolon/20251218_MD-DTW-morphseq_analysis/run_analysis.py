#!/usr/bin/env python3
"""
MD-DTW Analysis Pipeline for b9d2 Phenotype Distinction

Main pipeline script that:
1. Loads b9d2 experiment data
2. Prepares multivariate arrays (curvature + length trajectories)
3. Computes MD-DTW distance matrix
4. Runs bootstrap hierarchical clustering
5. Generates visualizations (dendrogram, multimetric trajectories)

Objective: Distinguish HTA (high curvature, normal length) vs CE (high curvature → shortened)
phenotypes in b9d2 mutants using joint curvature+length trajectories.

Usage:
    python run_analysis.py [--experiment EXPERIMENT_ID] [--k K_VALUES] [--output OUTPUT_DIR]

Created: 2025-12-18
Location: results/mcolon/20251218_MD-DTW-morphseq_analysis/
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

# Import MD-DTW functions
from md_dtw_prototype import (
    prepare_multivariate_array,
    compute_md_dtw_distance_matrix,
    plot_dendrogram,
    plot_dendrogram_with_categories,
    identify_outliers,
)

# Import existing trajectory analysis utilities
from src.analyze.trajectory_analysis.data_loading import load_experiment_dataframe
from src.analyze.trajectory_analysis.bootstrap_clustering import (
    run_bootstrap_hierarchical,
    get_cluster_assignments,
)
from src.analyze.trajectory_analysis.cluster_posteriors import analyze_bootstrap_results
from src.analyze.trajectory_analysis.faceted_plotting import plot_multimetric_trajectories


# =============================================================================
# Configuration (editable at top of file)
# =============================================================================

# *** USER CONFIGURATION - MODIFY HERE ***
# K values to evaluate in clustering analysis
K_VALUES = [3, 4, 5, 6]

# Primary K for visualization focus
K_FOCUS = 3

# Experiment IDs to analyze (choose one approach):
# Option 1: Single experiment
EXPERIMENT_ID = '20251121'

# Option 2: Multiple experiments combined (uncomment to use, comment out EXPERIMENT_ID above)
# EXPERIMENT_IDS = ['20251121', '20251119', '20251125', '20251104']
EXPERIMENT_IDS = None

# Optional: Filter to specific genotype (None = all b9d2)
GENOTYPE_FILTER = None  # e.g., 'b9d2_homozygous'

# *** END USER CONFIGURATION ***

# Default metrics for MD-DTW (curvature + length)
DEFAULT_METRICS = [
    'baseline_deviation_normalized',  # Curvature proxy
    'total_length_um',                # Body length
]

# Default analysis parameters
DEFAULT_K_VALUES = K_VALUES
DEFAULT_K_FOCUS = K_FOCUS
DEFAULT_SAKOE_CHIBA_RADIUS = 3
DEFAULT_N_BOOTSTRAP = 100
DEFAULT_BOOTSTRAP_FRAC = 0.8

# Time range for b9d2 analysis (focus on phenotype-relevant window)
# Set to None to disable time filtering, or change tuple to desired range
DEFAULT_TIME_RANGE = None  # (18, 48)  # hpf


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_b9d2_data(
    experiment_id: Optional[str] = None,
    experiment_ids: Optional[List[str]] = None,
    genotype_filter: Optional[str] = None,
    min_timepoints: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load b9d2 experiment data from qc_staged files.

    Supports loading single experiment (backward compatible) or combining
    multiple experiments into one DataFrame.

    Args:
        experiment_id: Single experiment date ID (e.g., '20251121')
        experiment_ids: List of experiment IDs to combine (e.g., ['20251121', '20251119','20251125','20251104'])
        genotype_filter: Optional genotype to filter (e.g., 'b9d2_homozygous')
                        If None, loads all b9d2 genotypes.
        min_timepoints: Minimum timepoints per embryo to include
        verbose: Print loading information

    Returns:
        pd.DataFrame with filtered b9d2 data. If multiple experiments, includes
        'experiment_id' column to identify source experiment.

    Raises:
        ValueError: If both experiment_id and experiment_ids provided, or neither
    """
    # Validate inputs
    if experiment_id and experiment_ids:
        raise ValueError("Provide either experiment_id OR experiment_ids, not both")
    if not experiment_id and not experiment_ids:
        # Default to 20251121 for backward compatibility
        experiment_id = '20251121'

    # Normalize to list
    if experiment_id:
        exp_list = [experiment_id]
        is_combined = False
    else:
        exp_list = experiment_ids
        is_combined = True

    if verbose:
        if is_combined:
            print(f"Loading {len(exp_list)} b9d2 experiments: {exp_list}")
        else:
            print(f"Loading b9d2 data from experiment {exp_list[0]}...")

    # Load each experiment
    dfs = []
    for exp_id in exp_list:
        if verbose and is_combined:
            print(f"\n  Loading experiment {exp_id}...")

        # Load using the data_loading module
        try:
            df_exp = load_experiment_dataframe(exp_id, format_version='qc_staged')
        except FileNotFoundError:
            # Try direct path as fallback
            data_path = project_root / 'morphseq_playground' / 'metadata' / 'build04_output' / f'qc_staged_{exp_id}.csv'
            if not data_path.exists():
                raise FileNotFoundError(f"Could not find qc_staged file for {exp_id}")
            df_exp = pd.read_csv(data_path, low_memory=False)
            if verbose:
                print(f"    Loaded from: {data_path}")

        # Add experiment_id column if combining (or if it doesn't exist)
        if is_combined or 'experiment_id' not in df_exp.columns:
            df_exp['experiment_id'] = exp_id

        if verbose:
            if is_combined:
                print(f"    Total rows: {len(df_exp)}")
                print(f"    Unique embryos: {df_exp['embryo_id'].nunique()}")
            else:
                print(f"  Total rows: {len(df_exp)}")
                print(f"  Unique embryos: {df_exp['embryo_id'].nunique()}")

        dfs.append(df_exp)

    # Concatenate all experiments
    df = pd.concat(dfs, axis=0, ignore_index=True)

    if verbose and is_combined:
        print(f"\n  Combined: {len(df)} rows from {len(exp_list)} experiments")
        print(f"  Total unique embryos: {df['embryo_id'].nunique()}")

    # Filter to b9d2 genotypes
    b9d2_genotypes = df['genotype'].str.contains('b9d2', na=False)
    df_b9d2 = df[b9d2_genotypes].copy()

    if verbose:
        print(f"  b9d2 rows: {len(df_b9d2)}")
        print(f"  b9d2 genotypes: {df_b9d2['genotype'].unique()}")

    # # Filter out wildtype and unknown genotypes
    # wildtype_mask = df_b9d2['genotype'].str.contains('wildtype', case=False, na=False)
    # unknown_mask = df_b9d2['genotype'].str.contains('unknown', case=False, na=False)
    # df_b9d2 = df_b9d2[~(wildtype_mask | unknown_mask)]

    if verbose:
        print(f"  After removing wildtype/unknown: {len(df_b9d2)} rows")
        print(f"  Remaining genotypes: {df_b9d2['genotype'].unique()}")

    # Apply specific genotype filter if provided
    if genotype_filter:
        df_b9d2 = df_b9d2[df_b9d2['genotype'] == genotype_filter]
        if verbose:
            print(f"  After genotype filter '{genotype_filter}': {len(df_b9d2)} rows")

    # Filter embryos with sufficient timepoints
    embryo_counts = df_b9d2.groupby('embryo_id').size()
    valid_embryos = embryo_counts[embryo_counts >= min_timepoints].index
    df_b9d2 = df_b9d2[df_b9d2['embryo_id'].isin(valid_embryos)]

    if verbose:
        print(f"  Embryos with ≥{min_timepoints} timepoints: {df_b9d2['embryo_id'].nunique()}")

    # Filter out dead/flagged embryos if columns exist
    if 'use_embryo_flag' in df_b9d2.columns:
        df_b9d2 = df_b9d2[df_b9d2['use_embryo_flag'] == True]
        if verbose:
            print(f"  After use_embryo_flag filter: {df_b9d2['embryo_id'].nunique()} embryos")

    # Drop NaN in required columns
    required_cols = ['embryo_id', 'predicted_stage_hpf'] + DEFAULT_METRICS
    for col in required_cols:
        if col in df_b9d2.columns:
            before_count = len(df_b9d2)
            df_b9d2 = df_b9d2.dropna(subset=[col])
            after_count = len(df_b9d2)
            if verbose and before_count != after_count:
                print(f"  Dropped {before_count - after_count} rows with NaN in '{col}'")

    if verbose:
        print(f"✓ Final dataset: {len(df_b9d2)} rows, {df_b9d2['embryo_id'].nunique()} embryos")

    return df_b9d2


def filter_time_range(
    df: pd.DataFrame,
    time_col: str = 'predicted_stage_hpf',
    time_range: tuple = DEFAULT_TIME_RANGE,
    verbose: bool = True,
) -> pd.DataFrame:
    """Filter DataFrame to specified time range. Set time_range=None to disable filtering."""
    if time_range is None:
        if verbose:
            print(f"  Time range filter: DISABLED (using all timepoints)")
        return df.copy()
    
    t_min, t_max = time_range
    df_filtered = df[(df[time_col] >= t_min) & (df[time_col] <= t_max)].copy()

    if verbose:
        print(f"  Time range filter [{t_min}, {t_max}] hpf:")
        print(f"    Before: {len(df)} rows")
        print(f"    After: {len(df_filtered)} rows")

    return df_filtered


# =============================================================================
# Analysis Functions
# =============================================================================

def run_md_dtw_analysis(
    df: pd.DataFrame,
    metrics: List[str] = DEFAULT_METRICS,
    sakoe_chiba_radius: int = DEFAULT_SAKOE_CHIBA_RADIUS,
    k_values: List[int] = DEFAULT_K_VALUES,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    bootstrap_frac: float = DEFAULT_BOOTSTRAP_FRAC,
    verbose: bool = True,
) -> dict:
    """
    Run the complete MD-DTW analysis pipeline.

    Args:
        df: DataFrame with trajectory data
        metrics: List of metric columns to include
        sakoe_chiba_radius: DTW constraint parameter
        k_values: List of cluster numbers to evaluate
        n_bootstrap: Number of bootstrap iterations
        bootstrap_frac: Fraction to sample per iteration
        verbose: Print progress information

    Returns:
        Dict with all analysis results:
        - 'X': 3D multivariate array
        - 'embryo_ids': List of embryo identifiers
        - 'time_grid': Time grid used
        - 'D': MD-DTW distance matrix
        - 'clustering_results': Dict keyed by k with bootstrap results
        - 'df_assignments': DataFrame with cluster assignments for all k
    """
    results = {}

    # Step 1: Prepare multivariate array
    print("\n" + "=" * 70)
    print("Step 1: Preparing Multivariate Array")
    print("=" * 70)

    X, embryo_ids, time_grid = prepare_multivariate_array(
        df,
        metrics=metrics,
        normalize=True,
        verbose=verbose,
    )

    results['X'] = X
    results['embryo_ids'] = embryo_ids
    results['time_grid'] = time_grid
    results['metrics'] = metrics

    # Step 2: Compute MD-DTW distance matrix
    print("\n" + "=" * 70)
    print("Step 2: Computing MD-DTW Distance Matrix")
    print("=" * 70)

    D = compute_md_dtw_distance_matrix(
        X,
        sakoe_chiba_radius=sakoe_chiba_radius,
        verbose=verbose,
    )

    results['D'] = D
    results['D_original'] = D.copy()  # Keep original for comparison
    results['embryo_ids_original'] = embryo_ids.copy()

    # Step 2.5: Remove outliers using IQR method
    print("\n" + "=" * 70)
    print("Step 2.5: Outlier Detection and Removal")
    print("=" * 70)

    # Compute median distances
    median_distances = np.zeros(len(embryo_ids))
    for i in range(len(embryo_ids)):
        dists_to_others = np.concatenate([D[i, :i], D[i, i+1:]])
        median_distances[i] = np.median(dists_to_others)

    # IQR method with 4.0× multiplier (extreme outlier detection)
    q1, q3 = np.percentile(median_distances, [25, 75])
    iqr = q3 - q1
    threshold = q3 + 4.0 * iqr
    outlier_mask = median_distances > threshold

    outlier_indices = np.where(outlier_mask)[0]
    inlier_indices = np.where(~outlier_mask)[0]

    outlier_ids = [embryo_ids[i] for i in outlier_indices]
    inlier_ids = [embryo_ids[i] for i in inlier_indices]

    if verbose:
        print(f"  Method: IQR (4.0× - extreme outlier definition)")
        print(f"  Q1 (25th percentile): {q1:.3f}")
        print(f"  Q3 (75th percentile): {q3:.3f}")
        print(f"  IQR: {iqr:.3f}")
        print(f"  Threshold: Q3 + 4.0×IQR = {threshold:.3f}")
        print(f"  Outliers detected: {len(outlier_ids)}")
        print(f"  Inliers retained: {len(inlier_ids)}")

        if len(outlier_ids) > 0:
            print(f"\n  Outlier embryos (median_dist > {threshold:.1f}):")
            for embryo_id, med_dist in zip(outlier_ids, median_distances[outlier_indices]):
                print(f"    {embryo_id}: {med_dist:.3f}")

    # Extract clean distance matrix (inliers only)
    D_clean = D[np.ix_(inlier_indices, inlier_indices)]
    embryo_ids_clean = inlier_ids

    print(f"\n✓ Outliers removed")
    print(f"  Original: {D.shape[0]} embryos")
    print(f"  Cleaned: {D_clean.shape[0]} embryos")

    # Update results to use cleaned data
    results['D'] = D_clean
    results['embryo_ids'] = embryo_ids_clean
    results['outlier_info'] = {
        'method': 'IQR 4.0×',
        'threshold': threshold,
        'outlier_ids': outlier_ids,
        'inlier_ids': inlier_ids,
        'outlier_indices': outlier_indices,
        'inlier_indices': inlier_indices,
        'median_distances': median_distances,
    }

    # Use cleaned data for clustering
    embryo_ids = embryo_ids_clean
    D = D_clean

    # Step 3: Run bootstrap hierarchical clustering for multiple k
    print("\n" + "=" * 70)
    print("Step 3: Bootstrap Hierarchical Clustering (on cleaned data)")
    print("=" * 70)

    df_assignments, all_results = get_cluster_assignments(
        distance_matrix=D,
        embryo_ids=embryo_ids,
        k_values=k_values,
        n_bootstrap=n_bootstrap,
        bootstrap_frac=bootstrap_frac,
        verbose=verbose,
    )

    results['df_assignments'] = df_assignments
    results['clustering_results'] = all_results

    return results


# =============================================================================
# Visualization Functions
# =============================================================================

def generate_visualizations(
    df: pd.DataFrame,
    results: dict,
    output_dir: Path,
    k_focus: int = 3,
    verbose: bool = True,
) -> None:
    """
    Generate all visualization outputs.

    Args:
        df: Original DataFrame with trajectory data
        results: Output from run_md_dtw_analysis()
        output_dir: Directory to save figures
        k_focus: Primary k value for cluster-colored plots
        verbose: Print progress information
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    D = results['D']
    embryo_ids = results['embryo_ids']
    metrics = results['metrics']
    k_values = list(results['clustering_results'].keys())

    # 1. Dendrogram with k cutoffs
    print("\n" + "-" * 50)
    print("Generating dendrogram...")
    print("-" * 50)

    fig_dendro, dendro_info = plot_dendrogram(
        D,
        embryo_ids,
        k_highlight=k_values,
        title='b9d2 Embryo Clustering (MD-DTW: Curvature + Length)',
        save_path=output_dir / 'dendrogram_md_dtw.png',
        verbose=verbose,
    )
    plt.close(fig_dendro)

    # 1b. Dendrogram with category annotations (pair, genotype)
    print("\n" + "-" * 50)
    print("Generating dendrogram with category bars...")
    print("-" * 50)

    # Prepare category dataframe
    category_df = results['df_assignments'][['embryo_id']].copy()

    # Add pair column (get first value per embryo from original df)
    if 'pair' in df.columns:
        pair_map = df.groupby('embryo_id')['pair'].first().to_dict()
        category_df['pair'] = category_df['embryo_id'].map(pair_map)
    else:
        category_df['pair'] = 'unknown_pair'

    # Add genotype column
    if 'genotype' in df.columns:
        genotype_map = df.groupby('embryo_id')['genotype'].first().to_dict()
        category_df['genotype'] = category_df['embryo_id'].map(genotype_map)
    else:
        category_df['genotype'] = 'unknown_genotype'

    # Generate categorized dendrogram
    try:
        fig_dendro_cat, dendro_cat_info = plot_dendrogram_with_categories(
            D,
            embryo_ids,
            category_df=category_df,
            category_cols=['pair', 'genotype'],
            k_highlight=k_values,
            title='b9d2 Clustering with Pair and Genotype Context',
            save_path=output_dir / 'dendrogram_md_dtw_categorized.png',
            verbose=verbose,
        )
        plt.close(fig_dendro_cat)
    except Exception as e:
        print(f"  WARNING: Could not generate categorized dendrogram: {e}")
        if verbose:
            import traceback
            traceback.print_exc()

    # 2. Multimetric trajectory plots for ALL k values
    print("\n" + "-" * 50)
    print(f"Generating multimetric trajectory plots for all k values...")
    print("-" * 50)

    for k in k_values:
        print(f"\n  k={k}:")

        if k not in results['clustering_results']:
            print(f"    WARNING: No results for k={k}, skipping...")
            continue

        # Create subdirectory for this k value
        k_dir = output_dir / f'k{k}'
        k_dir.mkdir(parents=True, exist_ok=True)

        # Get cluster labels for this k
        cluster_labels = results['df_assignments'][f'cluster_k{k}'].values

        # Create lookup dict for embryo_id -> cluster
        label_lookup = dict(zip(results['df_assignments']['embryo_id'], cluster_labels))

        # Add cluster column to df
        df_plot = df.copy()
        df_plot['md_dtw_cluster'] = df_plot['embryo_id'].map(label_lookup)
        df_plot = df_plot.dropna(subset=['md_dtw_cluster'])
        df_plot['md_dtw_cluster'] = df_plot['md_dtw_cluster'].astype(int)

        if verbose:
            print(f"    Embryos with cluster labels: {df_plot['embryo_id'].nunique()}")
            print(f"    Cluster distribution:")
            cluster_counts = df_plot.groupby('md_dtw_cluster')['embryo_id'].nunique()
            for c, count in cluster_counts.items():
                print(f"      Cluster {c}: {count} embryos")

        # Ensure pair column exists (for spawn experiments without pair labels)
        if 'pair' not in df_plot.columns or df_plot['pair'].isna().all():
            df_plot['pair'] = 'b9d2_spawn'
            if verbose:
                print(f"    Note: No pair labels found, assigning 'b9d2_spawn' to all embryos")

        # Define plot configurations: (suffix, col_by, overlay, title_suffix)
        plot_configs = [
            # Cluster-based views (existing)
            ('clusters_by_cluster', 'md_dtw_cluster', 'md_dtw_cluster', 'Clusters by Cluster'),
            ('clusters_by_genotype', 'md_dtw_cluster', 'genotype', 'Clusters by Genotype'),
            ('clusters_by_pair', 'md_dtw_cluster', 'pair', 'Clusters by Pair'),
            # Pair-based views (NEW)
            ('pairs_by_genotype', 'pair', 'genotype', 'Pairs by Genotype'),
            ('pairs_by_cluster', 'pair', 'md_dtw_cluster', 'Pairs by Cluster'),
        ]

        # Generate all plotting variants
        for suffix, col_by_val, overlay_val, title_suffix in plot_configs:
            # Validate columns exist
            if col_by_val not in df_plot.columns:
                if verbose:
                    print(f"    Skipping '{suffix}' - column '{col_by_val}' not found")
                continue
            if overlay_val not in df_plot.columns:
                if verbose:
                    print(f"    Skipping '{suffix}' - column '{overlay_val}' not found")
                continue

            # Validate columns have data
            if df_plot[col_by_val].isna().all():
                if verbose:
                    print(f"    Skipping '{suffix}' - column '{col_by_val}' is all NaN")
                continue
            if df_plot[overlay_val].isna().all():
                if verbose:
                    print(f"    Skipping '{suffix}' - column '{overlay_val}' is all NaN")
                continue

            # Generate multimetric plot
            try:
                fig_multi = plot_multimetric_trajectories(
                    df_plot,
                    metrics=metrics,
                    col_by=col_by_val,
                    color_by_grouping=overlay_val,
                    x_col='predicted_stage_hpf',
                    title=f'b9d2 Trajectories {title_suffix} (k={k})',
                    backend='matplotlib',
                )
                output_filename = f'multimetric_trajectories_{suffix}.png'
                fig_multi.savefig(k_dir / output_filename,
                                 dpi=150, bbox_inches='tight')
                plt.close(fig_multi)
                if verbose:
                    print(f"    Saved: k{k}/{output_filename}")
            except Exception as e:
                print(f"    WARNING: Could not generate '{suffix}' plot for k={k}: {e}")

    # 2.5. Count plots (cluster and genotype distributions)
    print("\n" + "-" * 50)
    print("Generating count plots...")
    print("-" * 50)

    for k in k_values:
        print(f"\n  k={k}:")

        if k not in results['clustering_results']:
            print(f"    WARNING: No results for k={k}, skipping...")
            continue

        k_dir = output_dir / f'k{k}'
        k_dir.mkdir(parents=True, exist_ok=True)

        # Get cluster labels for this k
        cluster_labels = results['df_assignments'][f'cluster_k{k}'].values
        label_lookup = dict(zip(results['df_assignments']['embryo_id'], cluster_labels))

        # Create count dataframe with cluster assignments
        df_counts = df.copy()
        df_counts['md_dtw_cluster'] = df_counts['embryo_id'].map(label_lookup)

        # Remove embryos without cluster assignments (outliers)
        df_counts = df_counts.dropna(subset=['md_dtw_cluster'])
        df_counts['md_dtw_cluster'] = df_counts['md_dtw_cluster'].astype(int)

        # Ensure pair column exists
        if 'pair' not in df_counts.columns or df_counts['pair'].isna().all():
            df_counts['pair'] = 'b9d2_spawn'

        # Get unique embryos only (one row per embryo)
        df_unique_embryos = df_counts.groupby('embryo_id').first().reset_index()

        if verbose:
            print(f"    Unique embryos with assignments: {len(df_unique_embryos)}")

        # Plot 1: Cluster counts by pair
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            counts = pd.crosstab(df_unique_embryos['pair'], df_unique_embryos['md_dtw_cluster'])
            counts.plot(kind='bar', stacked=False, ax=ax, edgecolor='black', linewidth=0.5)
            ax.set_title(f'Cluster Distribution by Pair (k={k})', fontsize=14)
            ax.set_xlabel('Pair', fontsize=12)
            ax.set_ylabel('Number of Embryos', fontsize=12)
            ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            fig.savefig(k_dir / 'cluster_counts_by_pair.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            if verbose:
                print(f"    Saved: k{k}/cluster_counts_by_pair.png")
        except Exception as e:
            print(f"    WARNING: Could not generate cluster counts by pair for k={k}: {e}")

        # Plot 2: Genotype counts by pair
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            counts = pd.crosstab(df_unique_embryos['pair'], df_unique_embryos['genotype'])
            counts.plot(kind='bar', stacked=False, ax=ax, edgecolor='black', linewidth=0.5)
            ax.set_title(f'Genotype Distribution by Pair (k={k})', fontsize=14)
            ax.set_xlabel('Pair', fontsize=12)
            ax.set_ylabel('Number of Embryos', fontsize=12)
            ax.legend(title='Genotype', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            fig.savefig(k_dir / 'genotype_counts_by_pair.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            if verbose:
                print(f"    Saved: k{k}/genotype_counts_by_pair.png")
        except Exception as e:
            print(f"    WARNING: Could not generate genotype counts by pair for k={k}: {e}")

        # Plot 3: Genotype counts by cluster
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            counts = pd.crosstab(df_unique_embryos['md_dtw_cluster'], df_unique_embryos['genotype'])
            counts.plot(kind='bar', stacked=False, ax=ax, edgecolor='black', linewidth=0.5)
            ax.set_title(f'Genotype Distribution by Cluster (k={k})', fontsize=14)
            ax.set_xlabel('Cluster', fontsize=12)
            ax.set_ylabel('Number of Embryos', fontsize=12)
            ax.legend(title='Genotype', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            fig.savefig(k_dir / 'genotype_counts_by_cluster.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            if verbose:
                print(f"    Saved: k{k}/genotype_counts_by_cluster.png")
        except Exception as e:
            print(f"    WARNING: Could not generate genotype counts by cluster for k={k}: {e}")

    print("\n✓ Count plots generated")

    # 3. Distance matrix heatmap
    print("\n" + "-" * 50)
    print("Generating distance matrix heatmap...")
    print("-" * 50)

    fig_dist, ax = plt.subplots(figsize=(12, 10))
    
    # Sort by cluster for visual grouping
    if k_focus in results['clustering_results']:
        sort_idx = np.argsort(cluster_labels)
        D_sorted = D[np.ix_(sort_idx, sort_idx)]
        sorted_ids = [embryo_ids[i] for i in sort_idx]
    else:
        D_sorted = D
        sorted_ids = embryo_ids

    # Clip colormap to 95th percentile to see structure (outliers drive scale)
    vmax = np.percentile(D_sorted, 95)
    im = ax.imshow(D_sorted, cmap='viridis', aspect='auto', vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label='MD-DTW Distance')
    ax.set_title(f'MD-DTW Distance Matrix (sorted by cluster k={k_focus}, clipped to 95th %ile)', fontsize=14)
    ax.set_xlabel('Embryo Index')
    ax.set_ylabel('Embryo Index')
    
    fig_dist.savefig(output_dir / 'distance_matrix_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig_dist)
    if verbose:
        print(f"  Saved: distance_matrix_heatmap.png")

    # 4. Summary statistics plot
    print("\n" + "-" * 50)
    print("Generating summary statistics...")
    print("-" * 50)

    fig_summary, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Silhouette scores by k
    k_vals = []
    silhouettes = []
    for k, res in results['clustering_results'].items():
        k_vals.append(k)
        boot_silhouettes = [b['silhouette'] for b in res['bootstrap_results']['bootstrap_results']
                          if not np.isnan(b['silhouette'])]
        silhouettes.append(np.mean(boot_silhouettes) if boot_silhouettes else np.nan)

    axes[0].bar(k_vals, silhouettes, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[0].set_ylabel('Mean Silhouette Score', fontsize=12)
    axes[0].set_title('Cluster Quality by K', fontsize=14)
    axes[0].grid(axis='y', alpha=0.3)

    # Cluster sizes for k_focus
    if k_focus in results['clustering_results']:
        unique, counts = np.unique(cluster_labels, return_counts=True)
        axes[1].bar(unique, counts, color='coral', edgecolor='black')
        axes[1].set_xlabel('Cluster', fontsize=12)
        axes[1].set_ylabel('Number of Embryos', fontsize=12)
        axes[1].set_title(f'Cluster Sizes (k={k_focus})', fontsize=14)
        axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig_summary.savefig(output_dir / 'clustering_summary.png', dpi=150, bbox_inches='tight')
    plt.close(fig_summary)
    if verbose:
        print(f"  Saved: clustering_summary.png")

    print("\n✓ All visualizations generated")


# =============================================================================
# Main Pipeline
# =============================================================================

def main(
    experiment_id: Optional[str] = None,
    experiment_ids: Optional[List[str]] = None,
    genotype: Optional[str] = None,
    k_values: List[int] = DEFAULT_K_VALUES,
    k_focus: int = 3,
    output_dir: Optional[Path] = None,
    save_results: bool = True,
    verbose: bool = True,
):
    """
    Main analysis pipeline.

    Args:
        experiment_id: Single experiment date to analyze (e.g., '20251121')
        experiment_ids: List of experiment IDs to combine (e.g., ['20251121', '20251119'])
        genotype: Specific genotype filter (e.g., 'b9d2_homozygous')
        k_values: List of k values to evaluate
        k_focus: Primary k for visualization focus
        output_dir: Output directory (auto-generated if None)
        save_results: Whether to save intermediate results
        verbose: Print progress information
    """
    # Determine experiment label for printing
    if experiment_ids:
        experiment_label = f"Combined {len(experiment_ids)} experiments: {experiment_ids}"
    elif experiment_id:
        experiment_label = experiment_id
    else:
        experiment_id = '20251121'  # Default
        experiment_label = experiment_id

    print("=" * 70)
    print("MD-DTW Analysis Pipeline for b9d2 Phenotype Distinction")
    print("=" * 70)
    print(f"Experiment: {experiment_label}")
    print(f"Genotype filter: {genotype or 'all b9d2'}")
    print(f"K values: {k_values}")
    print(f"Focus K: {k_focus}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_ids:
            exp_range = f"{min(experiment_ids)}_{max(experiment_ids)}"
            output_dir = Path(__file__).parent / 'output' / f'combined_{exp_range}_{timestamp}'
        else:
            output_dir = Path(__file__).parent / 'output' / f'{experiment_id}_{timestamp}'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Step 1: Load data
    print("\n" + "=" * 70)
    print("Loading Data")
    print("=" * 70)

    df = load_b9d2_data(
        experiment_id=experiment_id,
        experiment_ids=experiment_ids,
        genotype_filter=genotype,
        verbose=verbose,
    )

    # Filter time range
    df = filter_time_range(df, verbose=verbose)

    # Step 2: Run MD-DTW analysis
    results = run_md_dtw_analysis(
        df,
        k_values=k_values,
        verbose=verbose,
    )

    # Step 3: Generate visualizations
    print("\n" + "=" * 70)
    print("Generating Visualizations")
    print("=" * 70)

    generate_visualizations(
        df,
        results,
        output_dir,
        k_focus=k_focus,
        verbose=verbose,
    )

    # Step 4: Save results
    if save_results:
        print("\n" + "=" * 70)
        print("Saving Results")
        print("=" * 70)

        # Save cluster assignments
        results['df_assignments'].to_csv(output_dir / 'cluster_assignments.csv', index=False)
        if verbose:
            print(f"  Saved: cluster_assignments.csv")

        # Save distance matrices (both original and cleaned)
        np.save(output_dir / 'distance_matrix_cleaned.npy', results['D'])
        np.save(output_dir / 'distance_matrix_original.npy', results['D_original'])
        if verbose:
            print(f"  Saved: distance_matrix_cleaned.npy")
            print(f"  Saved: distance_matrix_original.npy")

        # Save embryo IDs (both cleaned and original)
        with open(output_dir / 'embryo_ids_cleaned.txt', 'w') as f:
            f.write('\n'.join(results['embryo_ids']))
        with open(output_dir / 'embryo_ids_original.txt', 'w') as f:
            f.write('\n'.join(results['embryo_ids_original']))
        if verbose:
            print(f"  Saved: embryo_ids_cleaned.txt")
            print(f"  Saved: embryo_ids_original.txt")

        # Save outlier information
        outlier_info = results['outlier_info']
        with open(output_dir / 'outliers_removed.txt', 'w') as f:
            f.write(f"Outlier Detection Method: {outlier_info['method']}\n")
            f.write(f"Threshold: {outlier_info['threshold']:.3f}\n")
            f.write(f"Outliers removed: {len(outlier_info['outlier_ids'])}\n")
            f.write(f"Inliers retained: {len(outlier_info['inlier_ids'])}\n\n")
            f.write("Outlier embryos:\n")
            for outlier_id in outlier_info['outlier_ids']:
                idx = results['embryo_ids_original'].index(outlier_id)
                med_dist = outlier_info['median_distances'][idx]
                f.write(f"  {outlier_id}: median_distance = {med_dist:.3f}\n")
        if verbose:
            print(f"  Saved: outliers_removed.txt")

    # Summary
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Embryos analyzed: {len(results['embryo_ids'])}")
    print(f"Metrics used: {results['metrics']}")
    print(f"Time points: {len(results['time_grid'])}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return results, output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='MD-DTW Analysis Pipeline for b9d2 Phenotype Distinction'
    )
    parser.add_argument(
        '--experiment', '-e',
        type=str,
        default='20251121',
        help='Experiment ID to analyze (default: 20251121)'
    )
    parser.add_argument(
        '--experiments', '-exps',
        type=str,
        nargs='+',
        default=None,
        help='Multiple experiment IDs to combine (e.g., --experiments 20251121 20251119)'
    )
    parser.add_argument(
        '--genotype', '-g',
        type=str,
        default=None,
        help='Specific genotype to filter (e.g., b9d2_homozygous)'
    )
    parser.add_argument(
        '--k', '-k',
        type=int,
        nargs='+',
        default=DEFAULT_K_VALUES,
        help='K values to evaluate (default: 2 3 4 5)'
    )
    parser.add_argument(
        '--k-focus',
        type=int,
        default=3,
        help='Primary K for visualization (default: 3)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory (auto-generated if not specified)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    # Validate experiment arguments
    if args.experiments and args.experiment != '20251121':
        parser.error("Cannot specify both --experiment and --experiments")

    # Route to main()
    if args.experiments:
        exp_id, exp_ids = None, args.experiments
    else:
        exp_id, exp_ids = args.experiment, None

    results, output_dir = main(
        experiment_id=exp_id,
        experiment_ids=exp_ids,
        genotype=args.genotype,
        k_values=args.k,
        k_focus=args.k_focus,
        output_dir=Path(args.output) if args.output else None,
        verbose=not args.quiet,
    )
