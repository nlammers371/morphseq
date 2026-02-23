#!/usr/bin/env python3
"""
[LEGACY - ARCHIVED] Standalone batch curvature processing script.

DEPRECATION NOTICE:
    This script has been integrated into the main Build04 pipeline.
    Curvature metrics are now automatically computed during Build04 stage inference.

    Legacy usage (for validation/testing only):
        python process_curvature_batch_legacy.py

    New integrated approach:
        - Build03: Masks are cleaned during snip export
        - Build04: Curvature metrics computed for all rows with valid mask data
        - Result: Curvature metrics in DF03 alongside other morphology metrics

Original usage (if needed for testing):
    Input: df03_final_output_with_latents_{exp}.csv
    Outputs:
        - metadata/body_axis/summary/curvature_metrics_summary_{exp}.csv
        - metadata/body_axis/arrays/curvature_arrays_{exp}.csv

Kept for validation against integrated version.
See: src/build/build04_perform_embryo_qc.py for integrated implementation.
See: src/build/utils/curvature_utils.py for extracted utility functions.
See docs/refactors/mask_cleaning_curvature_integration.md for details.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import multiprocessing as mp
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import json
import sys
import time
from typing import Dict, Tuple, List
import traceback

# Add project root to path BEFORE any local imports
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import after path setup
from segmentation_sandbox.scripts.body_axis_analysis.centerline_extraction import extract_centerline
from segmentation_sandbox.scripts.body_axis_analysis.curvature_metrics import compute_all_simple_metrics
from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle
from segmentation_sandbox.scripts.utils.mask_cleaning import clean_embryo_mask

def process_embryo_batch(batch_data: List[Tuple]) -> List[Dict]:
    """
    Process a batch of embryos in a single worker call.
    This dramatically reduces pickling overhead by processing many embryos per task.

    Args:
        batch_data: List of tuples, each containing embryo data

    Returns:
        List of result dictionaries
    """
    results = []
    for row_data in batch_data:
        result = process_single_embryo(row_data)
        results.append(result)
    return results


def process_single_embryo(row_data: Tuple) -> Dict:
    """
    Process one embryo and return curvature metrics.

    Args:
        row_data: Tuple of (snip_id, mask_rle, mask_h, mask_w, stage_hpf, embryo_id,
                           genotype, frame_idx, height_um)

    Returns:
        Dict with curvature metrics
    """
    # Handle both old tuple format and new dict format
    if isinstance(row_data, dict):
        snip_id = row_data['snip_id']
        mask_rle = row_data['mask_rle']
        mask_h = row_data['mask_height_px']
        mask_w = row_data['mask_width_px']
        stage_hpf = row_data['predicted_stage_hpf']
        embryo_id = row_data['embryo_id']
        genotype = row_data['genotype']
        frame_idx = row_data['frame_index']
        height_um = row_data['height_um']
    else:
        (snip_id, mask_rle, mask_h, mask_w, stage_hpf, embryo_id,
         genotype, frame_idx, height_um) = row_data

    result = {
        'snip_id': snip_id,
        'embryo_id': embryo_id,
        'predicted_stage_hpf': stage_hpf,
        'genotype': genotype,
        'frame_index': frame_idx,
    }

    start_time = time.time()

    try:
        # 1. Decode mask
        mask = decode_mask_rle({
            'size': [int(mask_h), int(mask_w)],
            'counts': mask_rle
        })

        # 2. Clean mask (5-step pipeline)
        cleaned_mask, cleaning_stats = clean_embryo_mask(mask, verbose=False)

        # 3. Calculate um_per_pixel from height
        um_per_pixel = float(height_um) / float(mask_h)
        result['um_per_pixel'] = um_per_pixel

        # 4. Extract centerline using Geodesic method (no pruning)
        spline_x, spline_y, curvature, arc_length = extract_centerline(
            cleaned_mask,
            method='geodesic',
            um_per_pixel=um_per_pixel,
            bspline_smoothing=5.0
        )

        # 5. Calculate summary statistics
        if len(curvature) > 0:
            result['total_length_um'] = float(arc_length[-1]) if len(arc_length) > 0 else np.nan
            result['mean_curvature_per_um'] = float(np.mean(curvature))
            result['std_curvature_per_um'] = float(np.std(curvature))
            result['max_curvature_per_um'] = float(np.max(curvature))
            result['n_centerline_points'] = len(spline_x)

            # 5a. Compute simple curvature metrics (baseline deviation, arc-length ratio, keypoint deviations)
            simple_metrics = compute_all_simple_metrics(spline_x, spline_y, um_per_pixel=um_per_pixel)
            result.update(simple_metrics)
        else:
            result['total_length_um'] = np.nan
            result['mean_curvature_per_um'] = np.nan
            result['std_curvature_per_um'] = np.nan
            result['max_curvature_per_um'] = np.nan
            result['n_centerline_points'] = 0
            # Set NaN for simple metrics if extraction failed
            result['baseline_deviation_um'] = np.nan
            result['max_baseline_deviation_um'] = np.nan
            result['baseline_deviation_std_um'] = np.nan
            result['arc_length_ratio'] = np.nan
            result['arc_length_um'] = np.nan
            result['chord_length_um'] = np.nan
            result['keypoint_deviation_q1_um'] = np.nan
            result['keypoint_deviation_mid_um'] = np.nan
            result['keypoint_deviation_q3_um'] = np.nan

        # 6. Store arrays as JSON strings
        result['centerline_x'] = json.dumps(spline_x.tolist() if isinstance(spline_x, np.ndarray) else spline_x)
        result['centerline_y'] = json.dumps(spline_y.tolist() if isinstance(spline_y, np.ndarray) else spline_y)
        result['curvature_values'] = json.dumps(curvature.tolist() if isinstance(curvature, np.ndarray) else curvature)
        result['arc_length_values'] = json.dumps(arc_length.tolist() if isinstance(arc_length, np.ndarray) else arc_length)

        # 7. Record processing metadata
        result['processing_time_s'] = time.time() - start_time
        result['cleaning_applied'] = True
        result['success'] = True
        result['error'] = None

    except Exception as e:
        # Capture full traceback for debugging
        result['error'] = str(e)
        result['traceback'] = traceback.format_exc()
        result['success'] = False
        result['processing_time_s'] = time.time() - start_time
        result['cleaning_applied'] = False
        result['um_per_pixel'] = np.nan
        result['total_length_um'] = np.nan
        result['mean_curvature_per_um'] = np.nan
        result['std_curvature_per_um'] = np.nan
        result['max_curvature_per_um'] = np.nan
        result['n_centerline_points'] = 0
        result['centerline_x'] = None
        result['centerline_y'] = None
        result['curvature_values'] = None
        result['arc_length_values'] = None

    return result


def main(test_mode=False, n_test=5, serial=False, parallel_strategy='individual', batch_size=10, n_workers=None):
    """Main processing pipeline."""

    # Paths
    project_root = Path(__file__).resolve().parents[3]
    metadata_path = (project_root /
                    "morphseq_playground/metadata/build06_output/df03_final_output_with_latents_20251017_combined.csv")
    output_dir = project_root / "morphseq_playground/metadata/body_axis"
    summary_dir = output_dir / "summary"
    arrays_dir = output_dir / "arrays"

    # Create output directories if they don't exist
    summary_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading metadata from: {metadata_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    if test_mode:
        print(f"TEST MODE: Processing only first {n_test} embryos")
        df = df.head(n_test)

    if serial:
        print(f"SERIAL MODE: Processing all {len(df)} embryos sequentially (no parallelism)")

    print(f"Loaded {len(df)} embryos")

    # Prepare data for parallel processing
    required_cols = ['snip_id', 'mask_rle', 'mask_height_px', 'mask_width_px',
                    'predicted_stage_hpf', 'embryo_id', 'genotype', 'frame_index', 'height_um']

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    row_data_list = [
        (row['snip_id'], row['mask_rle'], row['mask_height_px'], row['mask_width_px'],
         row['predicted_stage_hpf'], row['embryo_id'], row['genotype'], row['frame_index'],
         row['height_um'])
        for _, row in df.iterrows()
    ]

    # Process in parallel or serial
    if serial:
        # Serial processing when explicitly requested
        print("Processing serially...")
        results = []
        for i, row_data in enumerate(tqdm(row_data_list, desc="Processing embryos")):
            result = process_single_embryo(row_data)
            results.append(result)
            if not result['success']:
                print(f"\n❌ Failed {result['snip_id']}: {result['error']}")
                if 'traceback' in result:
                    print(result['traceback'])
    else:
        # Parallel processing with configurable strategies
        if n_workers is None:
            n_workers = min(32, max(1, mp.cpu_count() // 2))  # Use 1/2 of cores by default

        if parallel_strategy == 'batched':
            # Strategy 1: Batched processing
            # Pros: Balance between pickling overhead and responsiveness
            # Cons: Less granular progress tracking
            print(f"Processing with {n_workers} workers in BATCHED mode...")
            print(f"Batch size: {batch_size} embryos per batch")

            # Split data into batches
            batches = []
            for i in range(0, len(row_data_list), batch_size):
                batch = row_data_list[i:i+batch_size]
                batches.append(batch)

            print(f"Created {len(batches)} batches of ~{batch_size} embryos each")

            # Process batches in parallel
            all_results = []
            with Pool(processes=n_workers) as pool:
                batch_results = list(tqdm(
                    pool.imap(process_embryo_batch, batches),
                    total=len(batches),
                    desc="Processing batches",
                    unit="batch"
                ))

                for batch_result in batch_results:
                    all_results.extend(batch_result)

            results = all_results

        elif parallel_strategy == 'individual':
            # Strategy 2: Individual embryo processing (one task per embryo)
            # Pros: Most granular progress tracking, better load balancing
            # Cons: More pickling overhead
            print(f"Processing with {n_workers} workers in INDIVIDUAL mode...")
            print(f"One task per embryo (maximum granularity)")

            all_results = []
            with Pool(processes=n_workers) as pool:
                results_iter = tqdm(
                    pool.imap(process_single_embryo, row_data_list, chunksize=5),
                    total=len(row_data_list),
                    desc="Processing embryos",
                    unit="embryo"
                )
                all_results = list(results_iter)

            results = all_results

        elif parallel_strategy == 'chunksize':
            # Strategy 3: Use chunksize parameter (imap_unordered with chunksize)
            # Pros: Efficient - processes chunks but returns results as ready
            # Cons: Results not in order (need to sort by snip_id later if needed)
            print(f"Processing with {n_workers} workers in CHUNKSIZE mode...")
            print(f"Chunk size: {batch_size} embryos per chunk (unordered results)")

            all_results = []
            with Pool(processes=n_workers) as pool:
                results_iter = tqdm(
                    pool.imap_unordered(process_single_embryo, row_data_list, chunksize=batch_size),
                    total=len(row_data_list),
                    desc="Processing embryos",
                    unit="embryo"
                )
                all_results = list(results_iter)

            results = all_results

        else:
            raise ValueError(f"Unknown parallel strategy: {parallel_strategy}")

        # Report any failures
        for result in results:
            if not result['success']:
                print(f"❌ Failed {result['snip_id']}: {result['error']}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Print summary statistics
    success_count = results_df['success'].sum()
    print(f"\n✅ Successfully processed: {success_count}/{len(results_df)}")

    if success_count < len(results_df):
        print(f"❌ Failed: {len(results_df) - success_count}")
        failed = results_df[~results_df['success']]
        print("\nFailed embryos:")
        for idx, row in failed.iterrows():
            print(f"  {row['snip_id']}: {row['error']}")

    # Split into summary and arrays
    summary_cols = ['snip_id', 'embryo_id', 'predicted_stage_hpf', 'genotype',
                   'frame_index', 'um_per_pixel', 'total_length_um', 'mean_curvature_per_um',
                   'std_curvature_per_um', 'max_curvature_per_um', 'n_centerline_points',
                   # Simple curvature metrics
                   'baseline_deviation_um', 'max_baseline_deviation_um', 'baseline_deviation_std_um',
                   'arc_length_ratio', 'arc_length_um', 'chord_length_um',
                   'keypoint_deviation_q1_um', 'keypoint_deviation_mid_um', 'keypoint_deviation_q3_um',
                   # Processing metadata
                   'processing_time_s', 'cleaning_applied', 'success', 'error']

    arrays_cols = ['snip_id', 'embryo_id', 'predicted_stage_hpf', 'genotype',
                  'frame_index', 'um_per_pixel', 'centerline_x', 'centerline_y',
                  'curvature_values', 'arc_length_values', 'success', 'error']

    summary_df = results_df[summary_cols]
    arrays_df = results_df[arrays_cols]

    # Save outputs
    summary_output = summary_dir / "curvature_metrics_summary_20251017_combined.csv"
    arrays_output = arrays_dir / "curvature_arrays_20251017_combined.csv"

    summary_df.to_csv(summary_output, index=False)
    print(f"\n💾 Saved summary: {summary_output}")

    arrays_df.to_csv(arrays_output, index=False)
    print(f"💾 Saved arrays: {arrays_output}")

    # Print timing statistics
    avg_time = results_df[results_df['success']]['processing_time_s'].mean()
    total_time = results_df['processing_time_s'].sum()
    print(f"\n⏱️  Average time per embryo: {avg_time:.2f}s")
    print(f"⏱️  Total processing time: {total_time/3600:.1f} hours")

    # Print curvature statistics
    valid_results = results_df[results_df['success']]
    if len(valid_results) > 0:
        print(f"\n📊 Curvature Statistics (across {len(valid_results)} successful embryos):")
        print(f"   total_length_um: {valid_results['total_length_um'].mean():.1f} ± {valid_results['total_length_um'].std():.1f}")
        print(f"   mean_curvature: {valid_results['mean_curvature_per_um'].mean():.4f} ± {valid_results['mean_curvature_per_um'].std():.4f}")
        print(f"   max_curvature: {valid_results['max_curvature_per_um'].mean():.4f} ± {valid_results['max_curvature_per_um'].std():.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch process curvature metrics")
    parser.add_argument("--test", action="store_true", help="Run in test mode (5 embryos, serial)")
    parser.add_argument("--n-test", type=int, default=5, help="Number of embryos in test mode")
    parser.add_argument("--serial", action="store_true", help="Run in serial mode (default is parallel individual)")
    parser.add_argument("--strategy", type=str, default='individual',
                       choices=['batched', 'individual', 'chunksize'],
                       help="Parallelization strategy (default: individual)")
    parser.add_argument("--batch-size", type=int, default=10,
                       help="Batch/chunk size for parallel processing (default: 10)")
    parser.add_argument("--n-workers", type=int, default=None,
                       help="Number of worker processes (default: cpu_count//2)")
    args = parser.parse_args()

    main(test_mode=args.test, n_test=args.n_test, serial=args.serial,
         parallel_strategy=args.strategy, batch_size=args.batch_size, n_workers=args.n_workers)
