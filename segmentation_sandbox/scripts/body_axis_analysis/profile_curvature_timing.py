#!/usr/bin/env python3
"""
Profile timing for each operation in the curvature processing pipeline.

This script processes a sample of embryos and reports detailed timing for:
- Mask loading
- Centerline extraction (geodesic spline fitting)
- Curvature calculation
- Arc length calculation
- Result packaging

Usage:
    python profile_curvature_timing.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time

# Add project root to path
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from segmentation_sandbox.scripts.body_axis_analysis.centerline_extraction import extract_centerline
from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle
from segmentation_sandbox.scripts.utils.mask_cleaning import clean_embryo_mask


def time_operation(name, func, *args, **kwargs):
    """Time a single operation and print result."""
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    print(f"  {name:40s}: {elapsed:7.4f}s")
    return result, elapsed


def profile_single_embryo(row, timings):
    """Profile all operations for a single embryo (matching batch processing pipeline)."""
    print(f"\n{'='*70}")
    print(f"Processing: {row['snip_id']} (embryo: {row['embryo_id']}, stage: {row['predicted_stage_hpf']:.1f} hpf)")
    print(f"{'='*70}")

    embryo_timings = {
        'snip_id': row['snip_id'],
        'embryo_id': row['embryo_id'],
        'predicted_stage_hpf': row['predicted_stage_hpf'],
        'genotype': row['genotype'],
        'frame_index': row['frame_index']
    }

    try:
        # Time: Decode mask from RLE
        def decode_mask():
            return decode_mask_rle({
                'size': [int(row['mask_height_px']), int(row['mask_width_px'])],
                'counts': row['mask_rle']
            })

        mask, t = time_operation("1. Decode mask (RLE)", decode_mask)
        embryo_timings['decode_mask'] = t

        # Time: Clean mask (5-step pipeline)
        start = time.time()
        cleaned_mask, cleaning_stats = clean_embryo_mask(mask, verbose=False)
        t = time.time() - start
        embryo_timings['clean_mask'] = t
        print(f"  {'2. Clean mask':40s}: {t:7.4f}s")

        # Calculate um_per_pixel
        um_per_pixel = float(row['height_um']) / float(row['mask_height_px'])
        embryo_timings['um_per_pixel'] = um_per_pixel

        # Time: Extract centerline using Geodesic method
        def extract_cl():
            return extract_centerline(
                cleaned_mask,
                method='geodesic',
                um_per_pixel=um_per_pixel,
                bspline_smoothing=5.0
            )

        result, t = time_operation("3. Extract centerline (geodesic)", extract_cl)
        embryo_timings['extract_centerline'] = t
        spline_x, spline_y, curvature, arc_length = result

        # Time: Calculate summary statistics
        def calc_stats():
            return {
                'total_length_um': float(arc_length[-1]) if len(arc_length) > 0 else np.nan,
                'mean_curvature': float(np.mean(curvature)) if len(curvature) > 0 else np.nan,
                'std_curvature': float(np.std(curvature)) if len(curvature) > 0 else np.nan,
                'max_curvature': float(np.max(curvature)) if len(curvature) > 0 else np.nan,
                'n_points': len(spline_x)
            }

        stats, t = time_operation("4. Calculate statistics", calc_stats)
        embryo_timings['calc_stats'] = t
        embryo_timings.update(stats)

        embryo_timings['success'] = True

    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        embryo_timings['success'] = False
        embryo_timings['error'] = str(e)

    timings.append(embryo_timings)


def main():
    """Main profiling function."""
    print("="*70)
    print("CURVATURE ANALYSIS TIMING PROFILER")
    print("="*70)
    
    # Load metadata (same as batch processing script)
    metadata_path = (repo_root /
                    "morphseq_playground/metadata/build06_output/df03_final_output_with_latents_20251017_combined.csv")

    print(f"\nLoading metadata: {metadata_path}")
    if not metadata_path.exists():
        print(f"ERROR: Metadata file not found at {metadata_path}")
        return

    start_load = time.time()
    df = pd.read_csv(metadata_path)
    load_time = time.time() - start_load
    print(f"Loaded {len(df)} entries in {load_time:.4f} seconds")
    
    # Process subset for timing
    n_samples = 10  # Profile first 10 embryos
    print(f"\nProfiling first {n_samples} embryos...")
    
    timings = []
    total_start = time.time()
    
    for idx, row in df.head(n_samples).iterrows():
        profile_single_embryo(row, timings)
    
    total_time = time.time() - total_start
    
    # Create timing DataFrame
    timing_df = pd.DataFrame(timings)
    
    print(f"\n{'='*70}")
    print("TIMING SUMMARY")
    print(f"{'='*70}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average per embryo: {total_time/n_samples:.2f} seconds")
    print(f"Successful: {timing_df['success'].sum()}/{len(timing_df)}")
    
    # Show average timing for each operation
    if timing_df['success'].any():
        successful = timing_df[timing_df['success']]
        print(f"\n{'='*70}")
        print("AVERAGE OPERATION TIMINGS (successful embryos)")
        print(f"{'='*70}")
        
        timing_columns = [
            'decode_mask',
            'clean_mask',
            'extract_centerline',
            'calc_stats'
        ]
        
        for col in timing_columns:
            if col in successful.columns:
                avg_time = successful[col].mean()
                std_time = successful[col].std()
                total_time_per_op = successful[timing_columns].sum(axis=1).mean()
                pct = (avg_time / total_time_per_op) * 100
                print(f"{col:30s}: {avg_time:7.4f}s ± {std_time:.4f}s  ({pct:5.1f}%)")
        
        total_avg = successful[timing_columns].sum(axis=1).mean()
        print(f"\n{'Total per embryo':30s}: {total_avg:7.4f}s")
    
    # Save detailed timing results
    output_path = repo_root / "results" / "curvature_timing_profile.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timing_df.to_csv(output_path, index=False)
    print(f"\nDetailed timing saved to: {output_path}")
    
    # Show slowest operations
    if timing_df['success'].any():
        print(f"\n{'='*70}")
        print("SLOWEST EMBRYOS")
        print(f"{'='*70}")
        successful = timing_df[timing_df['success']].copy()
        successful['total_time'] = successful[timing_columns].sum(axis=1)
        slowest = successful.nlargest(3, 'total_time')
        
        for _, row in slowest.iterrows():
            print(f"\n{row['embryo_id']} ({row['total_time']:.4f}s total)")
            for col in timing_columns:
                if col in row:
                    print(f"  {col:30s}: {row[col]:.4f}s")


if __name__ == "__main__":
    main()
