"""
Comparison Test: Original vs Optimized Geodesic Centerline Extraction

This script compares the original GeodesicCenterlineAnalyzer with the optimized
GeodesicCenterlineAnalyzerOptimized to validate:
1. Correctness: Same centerline length and curvature
2. Performance: Speedup metrics
3. Edge cases: Behavior on unusual morphologies

Usage:
    python test_comparison.py --sample-dir /path/to/mask/files --num-samples 10
"""

import sys
import time
import numpy as np
import argparse
from pathlib import Path
from skimage import io
import pandas as pd

# Add paths for imports - use absolute path from working directory
import importlib.util

# Load geodesic_method.py
geodesic_spec = importlib.util.spec_from_file_location(
    "geodesic_method",
    "segmentation_sandbox/scripts/body_axis_analysis/geodesic_method.py"
)
geodesic_module = importlib.util.module_from_spec(geodesic_spec)
geodesic_spec.loader.exec_module(geodesic_module)
GeodesicCenterlineAnalyzer = geodesic_module.GeodesicCenterlineAnalyzer

# Load geodesic_method_optimized.py
geodesic_opt_spec = importlib.util.spec_from_file_location(
    "geodesic_method_optimized",
    "segmentation_sandbox/scripts/body_axis_analysis/geodesic_method_optimized.py"
)
geodesic_opt_module = importlib.util.module_from_spec(geodesic_opt_spec)
geodesic_opt_spec.loader.exec_module(geodesic_opt_module)
GeodesicCenterlineAnalyzerOptimized = geodesic_opt_module.GeodesicCenterlineAnalyzerOptimized


def compare_centerlines(original, optimized, length_tolerance=0.02, curvature_tolerance=0.05):
    """
    Compare two centerline extraction results.

    Args:
        original: Results dict from GeodesicCenterlineAnalyzer.analyze()
        optimized: Results dict from GeodesicCenterlineAnalyzerOptimized.analyze()
        length_tolerance: Acceptable relative difference in length (default: ±2%)
        curvature_tolerance: Acceptable relative difference in curvature (default: ±5%)

    Returns:
        validation: Dict with comparison results and pass/fail status
    """
    validation = {
        'length_match': False,
        'curvature_match': False,
        'both_succeeded': False,
        'details': {}
    }

    # Check that both succeeded
    if original is None or optimized is None:
        validation['details']['error'] = "One or both methods failed"
        return validation

    validation['both_succeeded'] = True

    # Compare length
    orig_length = original['stats']['total_length']
    opt_length = optimized['stats']['total_length']

    if orig_length > 0:
        length_diff = abs(opt_length - orig_length) / orig_length
        validation['length_match'] = length_diff <= length_tolerance
        validation['details']['orig_length'] = float(orig_length)
        validation['details']['opt_length'] = float(opt_length)
        validation['details']['length_diff_pct'] = float(length_diff * 100)
    else:
        validation['details']['warning'] = "Original length is zero"

    # Compare curvature
    orig_mean_curv = original['stats']['mean_curvature']
    opt_mean_curv = optimized['stats']['mean_curvature']

    if orig_mean_curv > 0:
        curv_diff = abs(opt_mean_curv - orig_mean_curv) / orig_mean_curv
        validation['curvature_match'] = curv_diff <= curvature_tolerance
        validation['details']['orig_mean_curv'] = float(orig_mean_curv)
        validation['details']['opt_mean_curv'] = float(opt_mean_curv)
        validation['details']['curv_diff_pct'] = float(curv_diff * 100)
    else:
        validation['details']['warning'] = "Original curvature is zero"

    validation['passed'] = validation['both_succeeded'] and \
                          validation['length_match'] and \
                          validation['curvature_match']

    return validation


def test_on_mask(mask, mask_name="test_mask"):
    """
    Test both methods on a single mask.

    Args:
        mask: Binary numpy array
        mask_name: Name for reporting

    Returns:
        results: Dict with timing, validation, and statistics
    """
    results = {
        'mask_name': mask_name,
        'mask_shape': mask.shape,
        'mask_area': int(np.sum(mask)),
    }

    # Test original method
    original_result = None
    original_time = None
    original_error = None

    try:
        analyzer_orig = GeodesicCenterlineAnalyzer(mask, fast=True)
        t0 = time.time()
        original_result = analyzer_orig.analyze()
        original_time = time.time() - t0
    except Exception as e:
        original_error = str(e)
        results['original_error'] = original_error

    results['original_time'] = original_time

    # Test optimized method
    optimized_result = None
    optimized_time = None
    optimized_error = None

    try:
        analyzer_opt = GeodesicCenterlineAnalyzerOptimized(
            mask, fast=True, use_convolution_filter=True
        )
        t0 = time.time()
        optimized_result = analyzer_opt.analyze()
        optimized_time = time.time() - t0
    except Exception as e:
        optimized_error = str(e)
        results['optimized_error'] = optimized_error

    results['optimized_time'] = optimized_time

    # Compute speedup
    if original_time and optimized_time:
        speedup = original_time / optimized_time
        results['speedup'] = speedup
        results['time_saved_pct'] = (1 - optimized_time / original_time) * 100
    else:
        results['speedup'] = None

    # Validate correctness
    validation = compare_centerlines(original_result, optimized_result)
    results['validation'] = validation

    # Store statistics for detailed comparison
    if original_result and optimized_result:
        results['original_stats'] = original_result['stats']
        results['optimized_stats'] = optimized_result['stats']

    return results


def print_results(all_results):
    """Print comparison results in a readable format."""
    print("\n" + "="*80)
    print("GEODESIC CENTERLINE EXTRACTION: ORIGINAL vs OPTIMIZED")
    print("="*80)

    # Summary statistics
    successful = sum(1 for r in all_results if r['validation'].get('both_succeeded'))
    passed_validation = sum(1 for r in all_results if r['validation'].get('passed'))
    speedups = [r['speedup'] for r in all_results if r.get('speedup')]

    print(f"\nTotal tests: {len(all_results)}")
    print(f"Both methods succeeded: {successful}/{len(all_results)}")
    print(f"Passed validation: {passed_validation}/{successful}")

    if speedups:
        print(f"\nSpeedup Statistics:")
        print(f"  Mean speedup: {np.mean(speedups):.2f}x")
        print(f"  Median speedup: {np.median(speedups):.2f}x")
        print(f"  Min speedup: {np.min(speedups):.2f}x")
        print(f"  Max speedup: {np.max(speedups):.2f}x")
        print(f"  Time saved: {np.mean([r.get('time_saved_pct', 0) for r in all_results if r.get('speedup')]):.1f}%")

    # Per-test details
    print("\n" + "-"*80)
    print("DETAILED RESULTS PER TEST:")
    print("-"*80)

    for i, result in enumerate(all_results, 1):
        print(f"\n[Test {i}] {result['mask_name']}")
        print(f"  Mask shape: {result['mask_shape']}, Area: {result['mask_area']} pixels")

        if result.get('original_error'):
            print(f"  ❌ Original failed: {result['original_error']}")
        else:
            print(f"  Original time: {result['original_time']:.3f}s")

        if result.get('optimized_error'):
            print(f"  ❌ Optimized failed: {result['optimized_error']}")
        else:
            print(f"  Optimized time: {result['optimized_time']:.3f}s")

        if result.get('speedup'):
            print(f"  ⚡ Speedup: {result['speedup']:.2f}x ({result['time_saved_pct']:.1f}% faster)")

        validation = result['validation']
        if validation.get('both_succeeded'):
            status = "✓" if validation.get('passed') else "✗"
            print(f"  {status} Correctness validation: {'PASSED' if validation['passed'] else 'FAILED'}")

            if validation.get('details'):
                details = validation['details']
                if 'length_diff_pct' in details:
                    print(f"    - Length difference: {details['length_diff_pct']:.2f}%")
                if 'curv_diff_pct' in details:
                    print(f"    - Curvature difference: {details['curv_diff_pct']:.2f}%")
        elif not validation.get('both_succeeded'):
            print(f"  ✗ Validation skipped (one method failed)")

    print("\n" + "="*80)


def create_test_masks():
    """
    Create synthetic test masks to validate the implementation.

    Returns:
        masks: List of (mask, name) tuples for testing
    """
    masks = []

    # 1. Simple ellipse (basic case)
    h, w = 300, 200
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    ry, rx = 80, 60
    mask = ((y - cy)**2 / ry**2 + (x - cx)**2 / rx**2) <= 1
    masks.append((mask.astype(np.uint8), "simple_ellipse"))

    # 2. Curved embryo-like shape
    h, w = 400, 300
    y, x = np.ogrid[:h, :w]
    cx = w // 2
    # Create curved shape
    curve = 50 * np.sin(np.linspace(0, np.pi, h) * 0.8)
    mask = np.zeros((h, w), dtype=np.uint8)
    for row in range(h):
        col_center = cx + curve[row]
        col_min = max(0, int(col_center - 30))
        col_max = min(w, int(col_center + 30))
        mask[row, col_min:col_max] = 1
    masks.append((mask.astype(np.uint8), "curved_embryo"))

    # 3. Small mask
    h, w = 100, 80
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    ry, rx = 30, 20
    mask = ((y - cy)**2 / ry**2 + (x - cx)**2 / rx**2) <= 1
    masks.append((mask.astype(np.uint8), "small_ellipse"))

    # 4. Large mask
    h, w = 600, 500
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    ry, rx = 150, 120
    mask = ((y - cy)**2 / ry**2 + (x - cx)**2 / rx**2) <= 1
    masks.append((mask.astype(np.uint8), "large_ellipse"))

    return masks


def load_embryo_masks_from_csv(num_samples: int = 10):
    """Load real embryo masks from CSV metadata."""
    try:
        # Add project root to path
        import sys
        project_root = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq")
        sys.path.insert(0, str(project_root))

        from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle

        metadata_dir = project_root / "morphseq_playground" / "metadata" / "build06_output"

        # Try to find a CSV file
        csv_files = list(metadata_dir.glob("*.csv"))
        if not csv_files:
            print("No CSV files found in metadata directory")
            return []

        csv_path = csv_files[0]
        print(f"Loading embryo data from {csv_path.name}...")

        df = pd.read_csv(csv_path)
        masks_data = []

        # Load up to num_samples
        for idx, row in df.iloc[:num_samples].iterrows():
            try:
                mask = decode_mask_rle({
                    'size': [int(row['mask_height_px']), int(row['mask_width_px'])],
                    'counts': row['mask_rle']
                })
                snip_id = row.get('snip_id', f"embryo_{idx}")
                masks_data.append((mask.astype(np.uint8), str(snip_id)))
                print(f"  Loaded: {snip_id} ({mask.shape})")
            except Exception as e:
                print(f"  Skipped row {idx}: {e}")

        return masks_data
    except Exception as e:
        print(f"Error loading from CSV: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Compare original and optimized geodesic centerline extraction"
    )
    parser.add_argument('--use-real', action='store_true', default=False,
                       help='Use real embryo masks from CSV (default: False)')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of samples to test (if using real masks)')
    parser.add_argument('--use-synthetic', action='store_true', default=True,
                       help='Use synthetic test masks (default: True)')

    args = parser.parse_args()

    all_results = []

    # Load test masks
    if args.use_real:
        print("\n" + "="*80)
        print("LOADING REAL EMBRYO MASKS")
        print("="*80)
        real_masks = load_embryo_masks_from_csv(args.num_samples)

        if real_masks:
            for mask, name in real_masks:
                try:
                    result = test_on_mask(mask, name)
                    all_results.append(result)
                except Exception as e:
                    print(f"Error testing {name}: {e}")
            args.use_synthetic = False  # Don't use synthetic if real masks loaded

    if args.use_synthetic:
        print("\n" + "="*80)
        print("USING SYNTHETIC TEST MASKS")
        print("="*80)
        test_masks = create_test_masks()
        for mask, name in test_masks:
            result = test_on_mask(mask, name)
            all_results.append(result)

    # Print results
    print_results(all_results)

    # Save results to CSV for further analysis
    output_csv = Path(__file__).parent / "comparison_results.csv"
    df_data = []
    for r in all_results:
        row = {
            'mask_name': r['mask_name'],
            'mask_area': r['mask_area'],
            'original_time': r.get('original_time'),
            'optimized_time': r.get('optimized_time'),
            'speedup': r.get('speedup'),
            'validation_passed': r['validation'].get('passed'),
            'original_error': r.get('original_error'),
            'optimized_error': r.get('optimized_error'),
        }
        if r.get('original_stats'):
            row['orig_length'] = r['original_stats'].get('total_length')
        if r.get('optimized_stats'):
            row['opt_length'] = r['optimized_stats'].get('total_length')
        df_data.append(row)

    df = pd.DataFrame(df_data)
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")


if __name__ == '__main__':
    main()
