"""
Test the new vectorized geodesic implementation on failing embryos
"""

import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle
from segmentation_sandbox.scripts.utils.mask_cleaning import clean_embryo_mask
from segmentation_sandbox.scripts.body_axis_analysis.mask_preprocessing import apply_preprocessing
from segmentation_sandbox.scripts.body_axis_analysis.geodesic_method import GeodesicCenterlineAnalyzer


def load_embryo_data(snip_id: str):
    """Load embryo from CSV."""
    metadata_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build06_output")

    if "20251017" in snip_id:
        csv_path = metadata_dir / "df03_final_output_with_latents_20251017_combined.csv"
    elif "20250512" in snip_id:
        csv_path = metadata_dir / "df03_final_output_with_latents_20250512.csv"
    else:
        raise ValueError(f"Cannot determine CSV file for snip_id: {snip_id}")

    df = pd.read_csv(csv_path)
    embryo_row = df[df['snip_id'] == snip_id].iloc[0]

    mask = decode_mask_rle({
        'size': [int(embryo_row['mask_height_px']), int(embryo_row['mask_width_px'])],
        'counts': embryo_row['mask_rle']
    })

    um_per_pixel = embryo_row['height_um'] / int(embryo_row['mask_height_px'])

    return mask, um_per_pixel


def test_embryo(snip_id: str):
    """Test the fixed geodesic method on a single embryo."""
    print(f"\n{'='*80}")
    print(f"Testing: {snip_id}")
    print(f"{'='*80}")

    try:
        # Load data
        mask, um_per_pixel = load_embryo_data(snip_id)
        print(f"✓ Loaded mask: {mask.shape}, area={mask.sum():,} px")

        # Clean mask
        mask_cleaned, _ = clean_embryo_mask(mask, verbose=False)
        print(f"✓ Cleaned mask: {mask_cleaned.sum():,} px")

        # Preprocess (Gaussian blur)
        mask_preprocessed = apply_preprocessing(mask_cleaned, method='gaussian_blur',
                                               sigma=10.0, threshold=0.7)
        print(f"✓ Preprocessed: {mask_preprocessed.sum():,} px")

        # Use the fixed analyzer
        analyzer = GeodesicCenterlineAnalyzer(
            mask_preprocessed,
            um_per_pixel=um_per_pixel,
            bspline_smoothing=5.0,
            random_seed=42,
            fast=True
        )

        print(f"\nRunning geodesic centerline extraction...")
        results = analyzer.analyze()

        # Print results
        print(f"✓ SUCCESS!")
        print(f"  - Centerline points (raw): {results['stats']['n_centerline_points']}")
        print(f"  - Skeleton points: {results['stats']['n_skeleton_points']}")
        print(f"  - Total length: {results['stats']['total_length']:.2f} μm")
        print(f"  - Mean curvature: {results['stats']['mean_curvature']:.6f} 1/μm")
        print(f"  - Max curvature: {results['stats']['max_curvature']:.6f} 1/μm")

        return True, results

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


# Test the two failing embryos
if __name__ == "__main__":
    failing_embryos = [
        "20251017_combined_C04_e01_t0114",
        "20251017_combined_F11_e01_t0065",
    ]

    results_summary = []

    for embryo_id in failing_embryos:
        success, results = test_embryo(embryo_id)
        results_summary.append({
            'embryo_id': embryo_id,
            'success': success,
            'total_length_um': results['stats']['total_length'] if results else None,
            'mean_curvature': results['stats']['mean_curvature'] if results else None,
        })

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for result in results_summary:
        status = "✓ PASS" if result['success'] else "✗ FAIL"
        print(f"{status}: {result['embryo_id']}")
        if result['success']:
            print(f"        Length: {result['total_length_um']:.2f} μm, "
                  f"Curvature: {result['mean_curvature']:.6f} 1/μm")
