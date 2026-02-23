"""
Step 0: DTW Distance Matrix Precomputation

Loads data, performs trajectory preprocessing, and computes DTW distance matrix.
This script leverages src utilities for all data loading and preprocessing.

Output
------
Saves to output/0_dtw/:
- distance_matrix.pkl : scipy.spatial.distance.pdist output (condensed)
- embryo_ids.pkl : Ordered embryo IDs matching distance matrix
- trajectories.pkl : Interpolated trajectories (list of arrays)
- df_long.pkl : Long-format trajectory data with all metadata
- common_grid.pkl : Time points used for interpolation
- original_lengths.pkl : Original trajectory lengths before interpolation
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# Import configuration
from config import (
    OUTPUT_DIR, GENOTYPE_FILTER, METRIC_NAME, MIN_TIMEPOINTS,
    DTW_WINDOW, GRID_STEP, TEST_MODE, TEST_EMBRYOS, RANDOM_SEED
)

# Import data loading (from reference analysis)
sys.path.insert(0, str(Path(__file__).parent.parent / "20251029_curvature_temporal_analysis"))
from load_data import get_analysis_dataframe

# Import src utilities for trajectory processing and DTW
from src.analyze.dtw_time_trend_analysis import (
    extract_trajectories,
    interpolate_trajectories,
    interpolate_to_common_grid,
    compute_dtw_distance_matrix,
)

# Import I/O utilities
from io_module import save_data


# ============================================================================
# MAIN PRECOMPUTATION FUNCTION
# ============================================================================

def precompute_dtw(
    genotype_filter=GENOTYPE_FILTER,
    metric_name=METRIC_NAME,
    dtw_window=DTW_WINDOW,
    grid_step=GRID_STEP,
    min_timepoints=MIN_TIMEPOINTS,
    test_mode=TEST_MODE,
    test_embryos=None,
    random_seed=RANDOM_SEED,
    verbose=True
):
    """
    Complete DTW precomputation pipeline: load → extract → interpolate → compute DTW.

    Parameters
    ----------
    genotype_filter : str
        Genotype to analyze (e.g., 'cep290_homozygous')
    metric_name : str
        Metric name to cluster on (e.g., 'normalized_baseline_deviation')
    dtw_window : int
        Sakoe-Chiba band width for DTW
    grid_step : float
        Time step for common grid interpolation
    min_timepoints : int
        Minimum number of timepoints per trajectory
    test_mode : bool
        If True, use only test_embryos
    test_embryos : list, optional
        List of embryo IDs to use in test mode
    random_seed : int
        Random seed for reproducibility
    verbose : bool
        Print progress messages

    Returns
    -------
    dict
        Dictionary containing:
        - 'distance_matrix' : DTW distance matrix (condensed form for scipy)
        - 'embryo_ids' : Ordered embryo IDs
        - 'trajectories' : Interpolated trajectory arrays
        - 'df_long' : Long-format data with metadata
        - 'common_grid' : Time points for interpolation
        - 'original_lengths' : Original trajectory lengths
        - 'genotype_filter' : Genotype used
        - 'metric_name' : Metric used
    """
    np.random.seed(random_seed)

    if verbose:
        print("\n" + "="*80)
        print("STEP 0: DTW DISTANCE MATRIX PRECOMPUTATION")
        print("="*80)

    # ========== STEP 0a: Load data ==========
    if verbose:
        print(f"\nStep 0a: Loading data...")
    df, metadata = get_analysis_dataframe()

    if test_mode and test_embryos is not None:
        if verbose:
            print(f"  TEST MODE: Filtering to {len(test_embryos)} embryos")
        df = df[df['embryo_id'].isin(test_embryos)].copy()

    # ========== STEP 0b: Extract trajectories ==========
    if verbose:
        print(f"\nStep 0b: Extracting trajectories...")
        print(f"  Genotype: {genotype_filter}")
        print(f"  Metric: {metric_name}")

    trajectories, embryo_ids, df_long = extract_trajectories(
        df,
        genotype_filter=genotype_filter,
        metric_name=metric_name,
        verbose=verbose
    )

    n_trajectories_initial = len(trajectories)
    if verbose:
        print(f"  Extracted {n_trajectories_initial} trajectories")

    # ========== STEP 0c: Filter by minimum timepoints ==========
    if verbose:
        print(f"\nStep 0c: Filtering trajectories (min {min_timepoints} timepoints)...")

    # Count timepoints per embryo
    timepoints_per_embryo = df_long.groupby('embryo_id').size()
    valid_embryos = timepoints_per_embryo[timepoints_per_embryo >= min_timepoints].index.tolist()

    if len(valid_embryos) < len(embryo_ids):
        df_long = df_long[df_long['embryo_id'].isin(valid_embryos)].copy()
        mask = np.array([eid in valid_embryos for eid in embryo_ids])
        trajectories = [t for t, m in zip(trajectories, mask) if m]
        embryo_ids = [e for e, m in zip(embryo_ids, mask) if m]
        if verbose:
            print(f"  Kept {len(trajectories)} trajectories (filtered {n_trajectories_initial - len(trajectories)})")

    # ========== STEP 0d: Interpolate missing data ==========
    if verbose:
        print(f"\nStep 0d: Interpolating missing data...")

    df_long = interpolate_trajectories(df_long, verbose=verbose)

    # ========== STEP 0e: Align to common time grid ==========
    if verbose:
        print(f"\nStep 0e: Aligning to common time grid (step={grid_step})...")

    interpolated_trajs, embryo_ids_ordered, orig_lengths, common_grid = interpolate_to_common_grid(
        df_long,
        grid_step=grid_step,
        verbose=verbose
    )

    if verbose:
        print(f"  Common grid: {len(common_grid)} points from {common_grid[0]:.1f} to {common_grid[-1]:.1f}")

    # Validate trajectory processing
    if verbose:
        print(f"\nStep 0f-validation: Validating trajectory trimming...")

        # Check that trajectories are variable-length (trimmed)
        traj_lengths = [len(t) for t in interpolated_trajs]
        has_variable_lengths = len(set(traj_lengths)) > 1

        if has_variable_lengths:
            print(f"  ✓ Trajectories are variable-length (trimmed)")
            print(f"    Lengths: min={min(traj_lengths)}, max={max(traj_lengths)}, mean={np.mean(traj_lengths):.1f}")
        else:
            print(f"  WARNING: All trajectories have same length ({traj_lengths[0]})")
            print(f"    This may indicate NaN padding instead of trimming")

        # Check for NaN values
        has_nan = any(np.isnan(t).any() for t in interpolated_trajs)
        if not has_nan:
            print(f"  ✓ No NaN values detected (clean data)")
        else:
            n_nan = sum(np.isnan(t).sum() for t in interpolated_trajs)
            print(f"  WARNING: Found {n_nan} NaN values in trajectories")

    # ========== STEP 0f: Compute DTW distance matrix ==========
    if verbose:
        print(f"\nStep 0f: Computing DTW distance matrix (window={dtw_window})...")

    distance_matrix = compute_dtw_distance_matrix(
        interpolated_trajs,
        window=dtw_window,
        verbose=verbose
    )

    if verbose:
        print(f"  Distance matrix shape: {distance_matrix.shape}")

    # ========== SUMMARY ==========
    if verbose:
        print(f"\n{'='*80}")
        print("STEP 0 SUMMARY")
        print(f"{'='*80}")
        print(f"  Genotype: {genotype_filter}")
        print(f"  Metric: {metric_name}")
        print(f"  Final sample size: {len(embryo_ids_ordered)}")
        print(f"  Common grid: {len(common_grid)} timepoints")
        print(f"  Distance matrix: {distance_matrix.shape[0]} x {distance_matrix.shape[1]}")
        print(f"  DTW window: {dtw_window}")
        print(f"\n{'='*80}\n")

    results = {
        'distance_matrix': distance_matrix,
        'embryo_ids': embryo_ids_ordered,
        'trajectories': interpolated_trajs,
        'df_long': df_long,
        'common_grid': common_grid,
        'original_lengths': orig_lengths,
        'genotype_filter': genotype_filter,
        'metric_name': metric_name,
        'n_initial': n_trajectories_initial,
        'n_final': len(embryo_ids_ordered),
        'grid_step': grid_step,
        'dtw_window': dtw_window,
    }

    return results


# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == '__main__':
    # Run precomputation with config parameters
    results = precompute_dtw(
        genotype_filter=GENOTYPE_FILTER,
        metric_name=METRIC_NAME,
        dtw_window=DTW_WINDOW,
        grid_step=GRID_STEP,
        min_timepoints=MIN_TIMEPOINTS,
        test_mode=TEST_MODE,
        test_embryos=TEST_EMBRYOS if TEST_MODE else None,
        random_seed=RANDOM_SEED,
        verbose=True
    )

    # Save all outputs
    print("\nSaving precomputation results...")
    save_data(0, 'distance_matrix', results['distance_matrix'], OUTPUT_DIR)
    save_data(0, 'embryo_ids', results['embryo_ids'], OUTPUT_DIR)
    save_data(0, 'trajectories', results['trajectories'], OUTPUT_DIR)
    save_data(0, 'df_long', results['df_long'], OUTPUT_DIR)
    save_data(0, 'common_grid', results['common_grid'], OUTPUT_DIR)
    save_data(0, 'original_lengths', results['original_lengths'], OUTPUT_DIR)

    print("\n✓ Precomputation complete!")
