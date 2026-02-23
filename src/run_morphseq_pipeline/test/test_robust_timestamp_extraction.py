#!/usr/bin/env python3
"""
Test script for robust YX1 timestamp extraction with imputation.
Copy these functions into a Jupyter notebook to test before implementing in build01B.

Usage in notebook:
- Copy and paste the functions below
- Run the test with your target experiment (e.g., 20250520, 20250711)
- Verify the output looks correct before implementing in main pipeline
"""

import numpy as np
import pandas as pd
from pathlib import Path
import nd2
import logging

# Set up logging to see function output
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)


def _get_imputed_time_vector(nd, n_t, n_w, n_z, well_indices):
    """
    Extracts timestamps, handling missing metadata, and imputes any gaps.
    Drop-in replacement for _fix_nd2_timestamp with robust gap handling.
    
    Args:
        nd: ND2File object
        n_t: Number of timepoints
        n_w: Number of wells in ND2
        n_z: Number of Z slices
        well_indices: List of 0-based well indices to use as references
    
    Returns:
        numpy array of length n_t with complete timestamps (no NaN)
    """
    # Safe reference well selection with guards
    refs = sorted(set(w for w in (well_indices or []) if 0 <= w < n_w))
    if len(refs) >= 3:
        ref_wells = [refs[0], refs[len(refs)//2], refs[-1]]  # First, middle, last
    elif refs:
        ref_wells = refs  # Use whatever valid wells we have
    else:
        ref_wells = list(range(min(n_w, 3)))  # Fallback to first few wells
    
    print(f"Using reference wells: {ref_wells} from {len(refs)} valid indices")
    
    # 1. Robustly extract timestamps, allowing for NaNs
    times = np.full((n_t,), np.nan, dtype=float)
    extraction_stats = {"successful": 0, "failed": 0}
    
    for t in range(n_t):
        for w in ref_wells:
            seq = (t * n_w + w) * n_z
            try:
                times[t] = nd.frame_metadata(seq).channels[0].time.relativeTimeMs / 1000.0
                extraction_stats["successful"] += 1
                break  # Found valid timestamp, move to next timepoint
            except Exception:
                extraction_stats["failed"] += 1
                continue  # Try next reference well
    
    print(f"Extraction stats: {extraction_stats}")
    print(f"Valid timestamps: {(~np.isnan(times)).sum()}/{n_t}")
    
    # 2. Calculate robust cycle time from ORIGINAL valid data
    s = pd.Series(times)  # Original with NaN gaps
    original_valid = s.dropna()

    if len(original_valid) >= 2:
        # Calculate cycle time from original valid timestamps
        original_diffs = original_valid.diff().dropna()
        cycle_time = original_diffs.median()
        print(f"Cycle time from {len(original_diffs)} valid intervals: {cycle_time:.2f}s")
    else:
        cycle_time = 60.0  # Default fallback
        print(f"Using default cycle time: {cycle_time:.2f}s")

    # 3. Now do imputation using this pre-calculated cycle time
    if s.isna().any():
        print(f"Starting imputation for {s.isna().sum()} missing values...")

        # DON'T use interpolation for sparse data - use direct extrapolation
        if len(original_valid) > 0:
            # Find the pattern in valid timestamps
            first_valid_idx = s.first_valid_index()
            last_valid_idx = s.last_valid_index()

            # Fill backwards from first valid
            if first_valid_idx > 0:
                first_time = s.iloc[first_valid_idx]
                for i in range(first_valid_idx - 1, -1, -1):
                    s.iloc[i] = first_time - (first_valid_idx - i) * cycle_time

            # Fill forwards from last valid  
            if last_valid_idx < len(s) - 1:
                last_time = s.iloc[last_valid_idx]
                for i in range(last_valid_idx + 1, len(s)):
                    s.iloc[i] = last_time + (i - last_valid_idx) * cycle_time

            # Fill middle gaps with linear progression
            for i in range(len(s)):
                if pd.isna(s.iloc[i]):
                    s.iloc[i] = s.iloc[0] + i * cycle_time

    
    # Last-resort safety: if still all NaN, create uniform grid
    if s.isna().all():
        print("WARNING: No valid timestamps found - using default 30 mins intervals")
        s = pd.Series(np.arange(n_t, dtype=float) * 1800.0)  # 30 min intervals
    
    # Ensure monotonic non-decreasing (prevent small numeric regressions)
    s = s.cummax()
    
    # Log final stats
    final_nan_count = s.isna().sum()
    print(f"Final result: {n_t - final_nan_count}/{n_t} valid timestamps")
    print(f"Time range: {s.min():.1f}s to {s.max():.1f}s")
    print(f"Median interval: {s.diff().median():.2f}s")
    
    return s.to_numpy()


def test_timestamp_extraction(exp_name, base_path=None):
    """
    Test the robust timestamp extraction on a specific experiment.
    
    Args:
        exp_name: Experiment name (e.g., "20250520", "20250711")
        base_path: Base path to data (defaults to morphseq_playground)
    
    Returns:
        dict with test results
    """
    if base_path is None:
        base_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground")
    
    nd2_path = next((base_path / "raw_image_data" / "YX1" / exp_name).glob("*.nd2"))
    print(f"Testing experiment: {exp_name}")
    print(f"ND2 file: {nd2_path}")
    
    results = {}
    
    with nd2.ND2File(str(nd2_path)) as f:
        # Get ND2 dimensions
        shape = f.shape  # T, W, Z, C, Y, X
        n_t, n_w, n_z = shape[:3]
        print(f"ND2 shape: T={n_t}, W={n_w}, Z={n_z}")
        
        results['shape'] = shape
        results['n_t'] = n_t
        results['n_w'] = n_w
        results['n_z'] = n_z
        
        # Simulate well selection (normally from Excel metadata)
        # For testing, use a spread of wells across the plate
        if n_w >= 10:
            simulated_well_indices = [0, n_w//4, n_w//2, 3*n_w//4, n_w-1]  # Spread across plate
        else:
            simulated_well_indices = list(range(n_w))
        
        print(f"Simulated selected wells: {simulated_well_indices}")
        results['test_wells'] = simulated_well_indices
        
        # Test our robust function
        print(f"\n=== Testing Robust Timestamp Extraction ===")
        try:
            result_times = _get_imputed_time_vector(f, n_t, n_w, n_z, simulated_well_indices)
            
            results['success'] = True
            results['result_times'] = result_times
            results['no_nan'] = not np.isnan(result_times).any()
            results['monotonic'] = np.all(np.diff(result_times) >= 0)
            results['time_range'] = (result_times.min(), result_times.max())
            results['median_interval'] = np.median(np.diff(result_times))
            
            print(f"\n=== Results Summary ===")
            print(f"✅ Extracted {len(result_times)} timestamps")
            print(f"✅ No NaN values: {results['no_nan']}")
            print(f"✅ Monotonic: {results['monotonic']}")
            print(f"📊 Time range: {results['time_range'][0]:.1f}s to {results['time_range'][1]:.1f}s")
            print(f"📊 Median interval: {results['median_interval']:.2f}s")
            print(f"📊 First 5 times: {result_times[:5]}")
            print(f"📊 Last 5 times: {result_times[-5:]}")
            
        except Exception as e:
            results['success'] = False
            results['error'] = str(e)
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test original method for comparison (if available)
        print(f"\n=== Comparison Test (Original Method) ===")
        try:
            # Try to simulate what original _fix_nd2_timestamp does
            original_times = []
            n_frames_total = f.frame_metadata(0).contents.frameCount
            
            for i in range(0, n_frames_total, n_z):
                try:
                    time_ms = f.frame_metadata(i).channels[0].time.relativeTimeMs
                    original_times.append(time_ms / 1000.0)
                except Exception:
                    break  # This is where original method would crash
            
            original_times = np.array(original_times)
            results['original_success'] = True
            results['original_length'] = len(original_times)
            results['original_vs_target'] = f"{len(original_times)}/{n_t}"
            
            print(f"📊 Original method: {len(original_times)}/{n_t} timestamps")
            if len(original_times) < n_t:
                print(f"⚠️  Original method incomplete ({len(original_times)}/{n_t} = {100*len(original_times)/n_t:.1f}%)")
            else:
                print(f"✅ Original method complete")
            
        except Exception as e:
            results['original_success'] = False
            results['original_error'] = str(e)
            print(f"❌ Original method failed: {e}")
            print(f"✅ Our robust method would handle this case!")
    
    return results


def run_multiple_tests():
    """
    Run tests on multiple experiments to validate robustness.
    """
    test_experiments = ["20250520", "20250711", "20250515_part2"]  # Add more experiments as needed
    
    all_results = {}
    
    for exp in test_experiments:
        print(f"\n{'='*60}")
        print(f"TESTING EXPERIMENT: {exp}")
        print(f"{'='*60}")
        
        try:
            results = test_timestamp_extraction(exp)
            all_results[exp] = results
        except Exception as e:
            print(f"❌ Failed to test {exp}: {e}")
            all_results[exp] = {'success': False, 'error': str(e)}
    
    # Summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    
    for exp, results in all_results.items():
        if results.get('success', False):
            coverage = f"{results['original_vs_target']}" if 'original_vs_target' in results else "N/A"
            print(f"✅ {exp}: PASS ({coverage} coverage, {results['median_interval']:.1f}s intervals)")
        else:
            print(f"❌ {exp}: FAIL ({results.get('error', 'Unknown error')})")
    
    return all_results


# Example usage code for notebook:
"""
# Copy this into your Jupyter notebook after copying the functions above:

# Test a single experiment
results_20250520 = test_timestamp_extraction("20250520")

# Test multiple experiments
all_results = run_multiple_tests()

# Check specific problematic experiment
results_20250711 = test_timestamp_extraction("20250711")

# Example of what a successful test should show:
# - No NaN values in final result
# - Monotonic timestamps
# - Reasonable time intervals (not 0 or crazy large)
# - For problematic experiments: shows imputation working
"""

if __name__ == "__main__":
    # If run as script, test default experiments
    results = run_multiple_tests()




/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python segmentation_sandbox/scripts/pipelines/03_gdino_detection.py \
  --config segmentation_sandbox/configs/pipeline_config.yaml \
  --metadata /net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/sam2_pipeline_files/raw_data_organized/20250529_36hpf_ctrl_atf6/experiment_metadata_20250529_36hpf_ctrl_atf6.json \
  --annotations /net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/sam2_pipeline_files/detections/gdino_detections_20250529_36hpf_ctrl_atf6.json \
  --confidence-threshold 0.45 \
  --iou-threshold 0.5 \
  --prompt "individual embryo" 



  python src/run_morphseq_pipeline/steps/run_sam2.py \
  --experiment 20250529_36hpf_ctrl_atf6 \
  --data-root /net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground