#!/usr/bin/env python3
"""
Hierarchical Clustering with Posterior Probability Analysis

Clean implementation focusing on clustering and posterior analysis only.
For plotting, see: consensus_clustering_plotting.py

Pipeline:
1. Load curvature data for experiment/genotype
2. Extract and interpolate trajectories
3. Compute DTW distance matrix
4. For each k in [2,3,4,5,6,7]:
   - Bootstrap hierarchical clustering (stores labels)
   - Hungarian label alignment
   - Posterior probability calculation
   - 2D gating classification (max_p + log_odds_gap)
5. Save comprehensive results

Usage:
    python run_hierarchical_posterior_analysis.py [--experiment EXPR] [--genotype GENO]
"""

import sys
import numpy as np
import pandas as pd
import pickle
import argparse
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# Import posterior analysis modules
from bootstrap_posteriors import analyze_bootstrap_results
from adaptive_classification import classify_embryos_2d, get_classification_summary

# Import DTW utilities
from src.analyze.dtw_time_trend_analysis import (
    extract_trajectories,
    interpolate_to_common_grid,
    compute_dtw_distance_matrix
)

# Configuration
K_VALUES = range(2, 8)  # k=2-7 (consistent with k-medoids)
N_BOOTSTRAP = 100
BOOTSTRAP_FRAC = 0.8
RANDOM_SEED = 42
MIN_TIMEPOINTS = 3
GRID_STEP = 0.5
DTW_WINDOW = 5


# ============================================================================
# BOOTSTRAP HIERARCHICAL CLUSTERING
# ============================================================================

def run_bootstrap_hierarchical(D, k, n_bootstrap=100, frac=0.8, random_state=42, verbose=False):
    """
    Bootstrap hierarchical clustering - stores labels for posterior analysis.

    Parameters
    ----------
    D : np.ndarray, shape (n, n)
        Precomputed distance matrix
    k : int
        Number of clusters
    n_bootstrap : int
        Number of bootstrap iterations
    frac : float
        Fraction of samples per bootstrap iteration
    random_state : int
        Random seed
    verbose : bool
        Print progress

    Returns
    -------
    dict with keys:
        'reference_labels' : np.ndarray
            Cluster labels from full dataset
        'bootstrap_results' : list of dict
            Each dict has 'labels' and 'indices' for one iteration
    """
    n = len(D)
    np.random.seed(random_state)
    bootstrap_results = []

    for iteration in range(n_bootstrap):
        if verbose and iteration % 20 == 0:
            print(f"      Bootstrap iteration {iteration}/{n_bootstrap}")

        # Sample embryos (no replacement)
        sample_size = int(n * frac)
        sampled_indices = np.random.choice(n, size=sample_size, replace=False)

        # Extract submatrix
        D_sample = D[np.ix_(sampled_indices, sampled_indices)]

        # Hierarchical clustering on subsample
        hc = AgglomerativeClustering(
            n_clusters=k,
            metric='precomputed',
            linkage='average'
        )
        labels_sample = hc.fit_predict(D_sample)

        # Map labels back to full array (-1 for non-sampled)
        labels_full = np.full(n, -1, dtype=int)
        labels_full[sampled_indices] = labels_sample

        # Store iteration results
        bootstrap_results.append({
            'labels': labels_full,
            'indices': sampled_indices
        })

    # Reference clustering on full dataset
    if verbose:
        print(f"      Running reference clustering on full dataset...")

    hc_ref = AgglomerativeClustering(
        n_clusters=k,
        metric='precomputed',
        linkage='average'
    )
    reference_labels = hc_ref.fit_predict(D)

    return {
        'reference_labels': reference_labels,
        'bootstrap_results': bootstrap_results
    }


# ============================================================================
# DATA LOADING
# ============================================================================

def load_curvature_data(experiment_id, genotype):
    """
    Load curvature metrics and merge with metadata to get genotype info.

    Returns
    -------
    pd.DataFrame with columns:
        - snip_id, embryo_id, predicted_stage_hpf, normalized_baseline_deviation, genotype
    """
    # Path to curvature metrics
    curv_dir = PROJECT_ROOT / 'morphseq_playground' / 'metadata' / 'body_axis' / 'summary'
    curv_file = curv_dir / f'curvature_metrics_{experiment_id}.csv'

    if not curv_file.exists():
        raise FileNotFoundError(f"Curvature file not found: {curv_file}")

    # Path to metadata
    meta_dir = PROJECT_ROOT / 'morphseq_playground' / 'metadata' / 'build06_output'
    meta_file = meta_dir / f'df03_final_output_with_latents_{experiment_id}.csv'

    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_file}")

    # Load curvature data
    curv_df = pd.read_csv(curv_file)
    print(f"  Loaded {len(curv_df)} curvature measurements")

    # Load metadata
    meta_df = pd.read_csv(meta_file)
    print(f"  Loaded {len(meta_df)} metadata rows")

    # Merge on snip_id to get genotype and embryo_id
    required_cols = ['snip_id', 'embryo_id', 'genotype', 'predicted_stage_hpf', 'baseline_deviation_normalized']
    meta_subset = meta_df[required_cols].copy()

    df = curv_df.merge(meta_subset, on='snip_id', how='inner')

    if len(df) == 0:
        raise ValueError(f"Merge resulted in empty dataframe")

    # Rename to expected column name
    df['normalized_baseline_deviation'] = df['baseline_deviation_normalized']

    # Filter by genotype
    df = df[df['genotype'] == genotype].copy()

    print(f"  Filtered to {len(df)} snips for {genotype}")

    if len(df) == 0:
        raise ValueError(f"No data found for genotype: {genotype}")

    return df


def extract_and_process_trajectories(df, genotype, min_timepoints=MIN_TIMEPOINTS, grid_step=GRID_STEP):
    """
    Extract trajectories and interpolate to common grid.

    Returns
    -------
    trajectories : list of np.ndarray
        Interpolated trajectories
    embryo_ids : list
        Embryo identifiers
    common_grid : np.ndarray
        Time points (HPF)
    """
    # Extract trajectories using the actual function signature
    print(f"  Extracting trajectories (min {min_timepoints} timepoints)...")
    trajectories_raw, embryo_ids, df_long = extract_trajectories(
        df,
        genotype_filter=genotype,
        metric_name='normalized_baseline_deviation',
        min_timepoints=min_timepoints,
        verbose=False
    )

    print(f"  Found {len(embryo_ids)} embryos with ≥{min_timepoints} timepoints")

    if len(embryo_ids) < 5:
        raise ValueError(f"Too few embryos ({len(embryo_ids)}) for clustering")

    # Interpolate to common grid (function signature: df_long, grid_step, verbose)
    print(f"  Interpolating to common grid (step={grid_step} HPF)...")
    trajectories_interp, embryo_ids_ordered, orig_lens, common_grid = interpolate_to_common_grid(
        df_long,
        grid_step=grid_step,
        verbose=False
    )

    return trajectories_interp, embryo_ids_ordered, common_grid


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def analyze_genotype(experiment_id, genotype, k_values=K_VALUES, output_dir='output/data/hierarchical'):
    """
    Complete analysis pipeline for one genotype.

    Steps:
    1. Load curvature data
    2. Extract & interpolate trajectories
    3. Compute DTW distance matrix
    4. For each k:
       - Bootstrap hierarchical clustering
       - Compute posteriors via Hungarian alignment
       - Classify with 2D gating
    5. Save comprehensive results

    Parameters
    ----------
    experiment_id : str
        Experiment identifier (e.g., '20251017_combined')
    genotype : str
        Genotype to analyze (e.g., 'cep290_homozygous')
    k_values : range or list
        k values to test
    output_dir : str
        Output directory for results

    Returns
    -------
    dict with all results
    """
    print(f"\n{'='*70}")
    print(f"Hierarchical Posterior Analysis: {experiment_id} / {genotype}")
    print(f"{'='*70}")

    # Step 1: Load data
    print(f"\n[1/5] Loading data...")
    df = load_curvature_data(experiment_id, genotype)

    # Step 2: Extract trajectories
    print(f"\n[2/5] Extracting trajectories...")
    trajectories, embryo_ids, common_grid = extract_and_process_trajectories(df, genotype)

    # Step 3: Compute DTW distance matrix
    print(f"\n[3/5] Computing DTW distance matrix...")
    print(f"  Using window={DTW_WINDOW}")
    D = compute_dtw_distance_matrix(
        trajectories,
        window=DTW_WINDOW,
        verbose=False
    )
    print(f"  Distance matrix shape: {D.shape}")

    # Step 4: Process each k
    print(f"\n[4/5] Running bootstrap + posterior analysis for k={list(k_values)}...")
    results = {}

    for k in k_values:
        print(f"\n  k={k}:")

        try:
            # Bootstrap hierarchical clustering
            print(f"    Running {N_BOOTSTRAP} bootstrap iterations...")
            boot_data = run_bootstrap_hierarchical(
                D, k,
                n_bootstrap=N_BOOTSTRAP,
                frac=BOOTSTRAP_FRAC,
                random_state=RANDOM_SEED,
                verbose=False
            )

            # Posterior analysis
            print(f"    Computing assignment posteriors...")
            posterior_analysis = analyze_bootstrap_results(boot_data)

            # 2D gating classification
            print(f"    Classifying with 2D gating...")
            classification = classify_embryos_2d(
                max_p=posterior_analysis['max_p'],
                log_odds_gap=posterior_analysis['log_odds_gap'],
                modal_cluster=posterior_analysis['modal_cluster'],
                threshold_max_p=0.8,
                threshold_log_odds=0.7,
                threshold_outlier_max_p=0.5
            )

            summary = get_classification_summary(classification)

            # Silhouette score
            try:
                sil = silhouette_score(D, boot_data['reference_labels'], metric='precomputed')
            except ValueError:
                sil = np.nan

            # Store results
            results[k] = {
                'labels': boot_data['reference_labels'],
                'posterior_analysis': posterior_analysis,
                'classification': classification,
                'summary': summary,
                'silhouette': sil,
                'bootstrap_data': boot_data  # Keep for diagnostics
            }

            # Print summary
            print(f"    Core: {summary['n_core']} ({summary['core_fraction']:.1%})")
            print(f"    Uncertain: {summary['n_uncertain']} ({summary['uncertain_fraction']:.1%})")
            print(f"    Outlier: {summary['n_outlier']} ({summary['outlier_fraction']:.1%})")
            print(f"    Silhouette: {sil:.3f}")

        except Exception as e:
            print(f"    ERROR: {e}")
            results[k] = None

    # Step 5: Save results
    print(f"\n[5/5] Saving results...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_data = {
        'experiment_id': experiment_id,
        'genotype': genotype,
        'results': results,
        'embryo_ids': embryo_ids,
        'common_grid': common_grid,
        'distance_matrix': D,
        'trajectories': trajectories,
        'metadata': {
            'k_values': list(k_values),
            'n_bootstrap': N_BOOTSTRAP,
            'bootstrap_frac': BOOTSTRAP_FRAC,
            'random_seed': RANDOM_SEED,
            'dtw_window': DTW_WINDOW,
            'n_embryos': len(embryo_ids)
        }
    }

    output_file = output_dir / f'{genotype}_all_k.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"  ✓ Saved to {output_file}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY: {genotype}")
    print(f"{'='*70}")
    print(f"{'k':<4} {'Core':<8} {'Uncertain':<12} {'Outlier':<10} {'Silhouette':<12}")
    print(f"{'-'*70}")
    for k in k_values:
        if results[k] is not None:
            r = results[k]
            print(f"{k:<4} {r['summary']['n_core']:<8} "
                  f"{r['summary']['n_uncertain']:<12} "
                  f"{r['summary']['n_outlier']:<10} "
                  f"{r['silhouette']:.3f}")
    print(f"{'='*70}\n")

    return output_data


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run analysis for specified experiment and genotypes."""
    parser = argparse.ArgumentParser(
        description='Run hierarchical clustering with posterior analysis'
    )
    parser.add_argument(
        '--experiment', '-e',
        type=str,
        default='20251017_combined',
        help='Experiment ID (default: 20251017_combined)'
    )
    parser.add_argument(
        '--genotype', '-g',
        type=str,
        default=None,
        help='Single genotype to process (default: process all)'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='output/data/hierarchical',
        help='Output directory (default: output/data/hierarchical)'
    )

    args = parser.parse_args()

    # Define genotypes to process
    if args.genotype:
        genotypes = [args.genotype]
    else:
        genotypes = [
            'cep290_wildtype',
            'cep290_heterozygous',
            'cep290_homozygous',
            'cep290_unknown'
        ]

    # Process each genotype
    for genotype in genotypes:
        try:
            analyze_genotype(
                experiment_id=args.experiment,
                genotype=genotype,
                k_values=K_VALUES,
                output_dir=args.output_dir
            )
        except Exception as e:
            print(f"\n{'='*70}")
            print(f"ERROR processing {genotype}:")
            print(f"{e}")
            print(f"{'='*70}\n")
            continue

    print(f"\n{'='*70}")
    print(f"✓ ANALYSIS COMPLETE")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
