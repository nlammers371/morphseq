#!/usr/bin/env python3
"""
Test trajectory_analysis package with DataFrame-centric API (v0.2.0)

Replicates the analysis from cluster_assignment_quality.md using the new
improved workflow. Tests:

1. Data loading and extraction (DataFrame-first)
2. DTW-based clustering with bootstrap
3. Posterior probability analysis with label alignment
4. 2D gating membership classification
5. All visualization functions including new plot_membership_vs_k()

Usage:
    python test_trajectory_analysis.py [--genotype GENO] [--k_min K_MIN] [--k_max K_MAX]

Examples:
    python test_trajectory_analysis.py --genotype cep290_homozygous
    python test_trajectory_analysis.py --k_min 2 --k_max 5
"""

import sys
import numpy as np
import pandas as pd
import pickle
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


# Data loading utilities (used for trajectory extraction)
from src.analyze.trajectory_analysis.data_loading import (
    load_experiment_dataframe,
    extract_trajectory_dataframe
)

# Import new v0.2.0 DataFrame-centric API
from src.analyze.trajectory_analysis import (
    # Data extraction (DataFrame-centric)
    extract_trajectories_df,
    interpolate_to_common_grid_df,
    df_to_trajectories,

    # DTW & clustering
    compute_dtw_distance_matrix,
    run_bootstrap_hierarchical,

    # Posterior analysis
    analyze_bootstrap_results,

    # Classification
    classify_membership_2d,

    # Plotting (including new plot_membership_vs_k)
    plot_cluster_trajectories_df,
    plot_membership_trajectories_df,
    plot_posterior_heatmap,
    plot_2d_scatter,
    plot_membership_vs_k
)



# Configuration
K_VALUES = range(2, 8)
N_BOOTSTRAP = 100
BOOTSTRAP_FRAC = 0.8
RANDOM_SEED = 42
MIN_TIMEPOINTS = 3
GRID_STEP = 0.5
DTW_WINDOW = 5

# Output directories
OUTPUT_DIR = Path(__file__).parent / 'output'
DATA_DIR = OUTPUT_DIR / 'data'
FIGURES_DIR = OUTPUT_DIR / 'figures'


# ============================================================================
# DATA LOADING
# ============================================================================

def load_experiment_data(experiment_id: str):
    """
    Load raw experiment data using general package utilities.

    Uses load_experiment_dataframe() which searches standard locations
    for curvature and metadata files and merges them.

    Parameters
    ----------
    experiment_id : str
        Experiment identifier (e.g., '20251017_combined')

    Returns
    -------
    pd.DataFrame
        Raw data with all columns from curvature and metadata files
    """
    print(f"Loading experiment data: {experiment_id}")
    df = load_experiment_dataframe(experiment_id)
    print(f"  Loaded {len(df)} total rows")
    return df


def prepare_data(df_raw: pd.DataFrame, genotype: str) -> pd.DataFrame:
    """
    Prepare raw data for trajectory analysis.

    Filters by genotype and extracts relevant columns for trajectory analysis.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw merged data (curvature + metadata)
    genotype : str
        Genotype to filter for

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns: embryo_id, predicted_stage_hpf,
        normalized_baseline_deviation, genotype
    """
    # Filter by genotype
    df = df_raw[df_raw['genotype'] == genotype].copy()

    # Rename baseline_deviation_normalized to normalized_baseline_deviation
    # (for consistency with trajectory API expectations)
    if 'baseline_deviation_normalized' in df.columns:
        df['normalized_baseline_deviation'] = df['baseline_deviation_normalized']
    elif 'normalized_baseline_deviation' not in df.columns:
        raise ValueError(
            f"Expected 'baseline_deviation_normalized' or 'normalized_baseline_deviation' column. "
            f"Available columns: {[c for c in df.columns if 'baseline' in c.lower() or 'deviation' in c.lower()]}"
        )

    # Select relevant columns for trajectory analysis
    df = df[['embryo_id', 'predicted_stage_hpf', 'normalized_baseline_deviation', 'genotype']].copy()

    print(f"  Filtered to {len(df)} measurements for {genotype}")
    print(f"  Embryos: {df['embryo_id'].nunique()}")
    print(f"  Time range: {df['predicted_stage_hpf'].min():.1f} - {df['predicted_stage_hpf'].max():.1f} hpf")

    return df


# ============================================================================
# ANALYSIS PIPELINE
# ============================================================================

def run_analysis_for_genotype(experiment_id: str, genotype: str) -> Dict[str, Any]:
    """
    Complete analysis pipeline for a single genotype.

    Uses the new v0.2.0 DataFrame-centric API throughout.

    Parameters
    ----------
    experiment_id : str
        Experiment identifier (e.g., '20251017_combined')
    genotype : str
        Genotype to analyze (e.g., 'cep290_homozygous')
    """
    print(f"\n{'='*70}")
    print(f"Analysis: {experiment_id} - {genotype}")
    print(f"{'='*70}")

    # 1. Load and prepare data
    print("\n1. Loading data...")
    df_raw = load_experiment_data(experiment_id)
    df = prepare_data(df_raw, genotype)

    # 2. Extract trajectories (DataFrame-centric)
    print("\n2. Extracting trajectories...")
    df_filtered = extract_trajectories_df(
        df,
        genotype_filter=None,  # Already filtered above
        metric_name='normalized_baseline_deviation',
        min_timepoints=MIN_TIMEPOINTS
    )
    print(f"  Extracted {df_filtered['embryo_id'].nunique()} embryos")

    # 3. Interpolate to common grid (DataFrame-centric)
    print("\n3. Interpolating to common time grid...")
    df_interpolated = interpolate_to_common_grid_df(
        df_filtered,
        grid_step=GRID_STEP
    )
    print(f"  Grid points: {df_interpolated['hpf'].nunique()}")

    # 4. Convert to arrays for DTW (one-line helper)
    print("\n4. Converting to arrays for DTW computation...")
    trajectories, embryo_ids, common_grid = df_to_trajectories(df_interpolated)
    print(f"  Trajectories: {len(trajectories)}")
    print(f"  Grid: {len(common_grid)} points")

    # 5. Compute DTW distance matrix
    print("\n5. Computing DTW distance matrix...")
    D = compute_dtw_distance_matrix(
        trajectories,
        window=DTW_WINDOW,
        verbose=True
    )
    print(f"  Distance matrix shape: {D.shape}")

    # 6. Run analysis for each k
    results = {
        'genotype': genotype,
        'embryo_ids': embryo_ids,
        'trajectories': trajectories,
        'common_grid': common_grid,
        'df_interpolated': df_interpolated,
        'results': {}
    }

    for k in K_VALUES:
        print(f"\n6.{k}. Bootstrap clustering for k={k}...")

        try:
            # Bootstrap hierarchical clustering
            bootstrap_results = run_bootstrap_hierarchical(
                D,
                k=k,
                embryo_ids=embryo_ids,
                n_bootstrap=N_BOOTSTRAP,
                frac=BOOTSTRAP_FRAC,
                random_state=RANDOM_SEED,
                verbose=False
            )

            # Analyze posterior probabilities
            posterior_analysis = analyze_bootstrap_results(bootstrap_results)

            # Classify membership quality
            classification = classify_membership_2d(
                max_p=posterior_analysis['max_p'],
                log_odds_gap=posterior_analysis['log_odds_gap'],
                modal_cluster=posterior_analysis['modal_cluster'],
                embryo_ids=posterior_analysis['embryo_ids'],
                threshold_max_p=0.8,
                threshold_log_odds_gap=0.7,
                threshold_outlier_max_p=0.5
            )

            # Store results
            results['results'][k] = {
                'bootstrap': bootstrap_results,
                'posterior_analysis': posterior_analysis,
                'classification': classification
            }

            # Print summary
            n_core = np.sum(classification['category'] == 'core')
            n_uncertain = np.sum(classification['category'] == 'uncertain')
            n_outlier = np.sum(classification['category'] == 'outlier')
            n_total = len(classification['category'])

            print(f"  Results: Core={n_core}/{n_total} ({100*n_core/n_total:.1f}%) "
                  f"Uncertain={n_uncertain}/{n_total} ({100*n_uncertain/n_total:.1f}%) "
                  f"Outlier={n_outlier}/{n_total} ({100*n_outlier/n_total:.1f}%)")

        except Exception as e:
            print(f"  ERROR: {e}")
            results['results'][k] = None

    return results


# ============================================================================
# PLOTTING
# ============================================================================

def generate_plots(results: Dict[str, Any]):
    """Generate all plots for the analysis."""
    genotype = results['genotype']
    print(f"\n{'='*70}")
    print(f"Generating plots for {genotype}")
    print(f"{'='*70}")

    # Create subdirectories
    genotype_dir = FIGURES_DIR / genotype
    dirs = {
        'heatmaps': genotype_dir / 'posterior_heatmaps',
        'scatters': genotype_dir / 'posterior_scatters',
        'trajectories': genotype_dir / 'trajectories',
        'membership': genotype_dir / 'membership_trends'
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Plot membership vs k (once per genotype, uses new function)
    print(f"\nGenerating membership vs k trend plot...")
    try:
        fig = plot_membership_vs_k(
            results,
            genotype=genotype,
            figsize=(10, 6),
            save_path=dirs['membership'] / 'membership_vs_k.png',
            dpi=300
        )
        print(f"  ✓ Saved to {dirs['membership'] / 'membership_vs_k.png'}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Plot for each k
    for k in sorted(results['results'].keys()):
        if results['results'][k] is None:
            print(f"k={k}: No data, skipping")
            continue

        print(f"\nk={k}:")

        # Posterior heatmap
        try:
            fig = plot_posterior_heatmap(
                results['results'][k]['posterior_analysis'],
                embryo_ids=results['embryo_ids'],
                figsize=(8, 12),
                save_path=dirs['heatmaps'] / f'heatmap_k{k}.png',
                dpi=300
            )
            print(f"  ✓ Posterior heatmap")
        except Exception as e:
            print(f"  ERROR (heatmap): {e}")

        # 2D scatter
        try:
            fig = plot_2d_scatter(
                results['results'][k]['classification'],
                embryo_ids=results['embryo_ids'],
                figsize=(10, 8),
                save_path=dirs['scatters'] / f'scatter_k{k}.png',
                dpi=300
            )
            print(f"  ✓ 2D scatter plot")
        except Exception as e:
            print(f"  ERROR (scatter): {e}")

        # Cluster trajectory plot (DataFrame version - preserves time alignment)
        try:
            fig = plot_cluster_trajectories_df(
                results['df_interpolated'],
                results['results'][k]['posterior_analysis']['modal_cluster'],
                embryo_ids=results['embryo_ids'],
                show_mean=True,
                show_individual=True,
                figsize=(12, 8),
                save_path=dirs['trajectories'] / f'clusters_k{k}.png',
                dpi=300
            )
            print(f"  ✓ Cluster trajectory plot")
        except Exception as e:
            print(f"  ERROR (trajectories): {e}")

        # Membership trajectory plot (DataFrame version)
        try:
            fig = plot_membership_trajectories_df(
                results['df_interpolated'],
                results['results'][k]['classification'],
                per_cluster=True,
                figsize=(15, 10),
                save_path=dirs['trajectories'] / f'membership_k{k}.png',
                dpi=300
            )
            print(f"  ✓ Membership trajectory plot")
        except Exception as e:
            print(f"  ERROR (membership): {e}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Test trajectory_analysis package with DataFrame-centric API'
    )
    parser.add_argument(
        '--experiment', '-e',
        type=str,
        default='20251017_combined',
        help='Experiment identifier (default: 20251017_combined)'
    )
    parser.add_argument(
        '--genotype', '-g',
        type=str,
        default='cep290_homozygous',
        help='Genotype to analyze (default: cep290_homozygous)'
    )
    parser.add_argument(
        '--k_min',
        type=int,
        default=2,
        help='Minimum k value (default: 2)'
    )
    parser.add_argument(
        '--k_max',
        type=int,
        default=7,
        help='Maximum k value (default: 7)'
    )
    parser.add_argument(
        '--skip_plots',
        action='store_true',
        help='Skip plot generation'
    )

    args = parser.parse_args()

    # Update global K_VALUES if specified
    global K_VALUES
    K_VALUES = range(args.k_min, args.k_max + 1)

    # Create output directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"TRAJECTORY ANALYSIS PACKAGE TEST (v0.2.0 DataFrame-centric API)")
    print(f"{'='*70}")
    print(f"Experiment: {args.experiment}")
    print(f"Genotype: {args.genotype}")
    print(f"k range: {args.k_min}-{args.k_max}")
    print(f"Output: {OUTPUT_DIR}")

    try:
        # Run analysis
        results = run_analysis_for_genotype(args.experiment, args.genotype)

        # Save results
        results_file = DATA_DIR / f'{args.genotype}_results.pkl'
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"\n✓ Saved results to {results_file}")

        # Generate plots
        if not args.skip_plots:
            generate_plots(results)

        print(f"\n{'='*70}")
        print(f"✓ ANALYSIS COMPLETE")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"ERROR: {e}")
        print(f"{'='*70}\n")
        raise


if __name__ == '__main__':
    main()
