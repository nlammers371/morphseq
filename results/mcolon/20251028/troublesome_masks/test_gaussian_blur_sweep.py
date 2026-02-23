"""
Test multiple Gaussian blur sigma values on three problematic embryos.

For each embryo and sigma value, visualize:
1. Preprocessed mask (to see how blurring affects topology)
2. Resulting centerline (to see if fins/artifacts are prevented)
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from skimage import morphology

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


def test_embryo_with_sigmas(snip_id: str, sigma_values: list, output_dir: Path):
    """Test a single embryo with multiple Gaussian blur sigma values."""
    print(f"\n{'='*100}")
    print(f"TESTING: {snip_id}")
    print(f"{'='*100}")

    # Load data
    mask, um_per_pixel = load_embryo_data(snip_id)
    mask_cleaned, _ = clean_embryo_mask(mask, verbose=False)

    print(f"Original mask: {mask.shape}, area={mask.sum():,} px")
    print(f"Cleaned mask: area={mask_cleaned.sum():,} px")

    results_by_sigma = {}

    for sigma in sigma_values:
        print(f"\n  Testing sigma={sigma}...", end=" ", flush=True)
        try:
            # Apply preprocessing with this sigma
            mask_preprocessed = apply_preprocessing(
                mask_cleaned,
                method='gaussian_blur',
                sigma=sigma,
                threshold=0.7
            )

            # Run geodesic centerline extraction
            analyzer = GeodesicCenterlineAnalyzer(
                mask_preprocessed,
                um_per_pixel=um_per_pixel,
                bspline_smoothing=5.0,
                random_seed=42,
                fast=True
            )

            results = analyzer.analyze()

            results_by_sigma[sigma] = {
                'success': True,
                'preprocessed_mask': mask_preprocessed,
                'centerline_raw': results['centerline_raw'],
                'centerline_smoothed': results['centerline_smoothed'],
                'skeleton': results['skeleton'],
                'endpoints': results['endpoints'],
                'length_um': results['stats']['total_length'],
                'mean_curvature': results['stats']['mean_curvature'],
                'n_centerline_points': results['stats']['n_centerline_points'],
            }
            print(f"✓ Length: {results['stats']['total_length']:.2f} μm")

        except Exception as e:
            results_by_sigma[sigma] = {
                'success': False,
                'error': str(e)
            }
            print(f"✗ {str(e)[:50]}")

    # Create comprehensive visualization
    n_sigmas = len(sigma_values)
    fig = plt.figure(figsize=(20, 4 * n_sigmas + 2))
    gs = GridSpec(n_sigmas, 3, figure=fig, hspace=0.35, wspace=0.3)

    for row_idx, sigma in enumerate(sigma_values):
        if sigma not in results_by_sigma or not results_by_sigma[sigma]['success']:
            # Show error panel
            ax = fig.add_subplot(gs[row_idx, :])
            ax.axis('off')
            error_msg = results_by_sigma.get(sigma, {}).get('error', 'Unknown error')
            ax.text(0.5, 0.5, f"σ={sigma}: FAILED\n{error_msg}",
                   ha='center', va='center', fontsize=14, color='red',
                   transform=ax.transAxes)
            continue

        result = results_by_sigma[sigma]

        # Panel 1: Original + Preprocessed mask
        ax = fig.add_subplot(gs[row_idx, 0])
        ax.imshow(mask_cleaned, cmap='Greys', alpha=0.5, label='Cleaned')
        ax.imshow(result['preprocessed_mask'], cmap='Blues', alpha=0.6, label='Preprocessed')
        ax.set_title(f'σ={sigma}: Masks\nCleaned: {mask_cleaned.sum():,} px → Preprocessed: {result["preprocessed_mask"].sum():,} px',
                    fontsize=10, fontweight='bold')
        ax.set_ylabel('Y (px)')
        ax.axis('tight')

        # Panel 2: Skeleton with raw centerline
        ax = fig.add_subplot(gs[row_idx, 1])
        ax.imshow(result['skeleton'], cmap='gray', alpha=0.7)
        ax.plot(result['centerline_raw'][:, 0], result['centerline_raw'][:, 1], 'b-', 
               linewidth=1.5, alpha=0.7, label='Raw')
        ax.plot(result['centerline_smoothed'][:, 0], result['centerline_smoothed'][:, 1], 'r-',
               linewidth=2, label='Smoothed')
        ax.plot(result['endpoints'][:, 0], result['endpoints'][:, 1], 'g*', markersize=12)
        ax.set_title(f'Centerline ({result["n_centerline_points"]} raw pts)',
                    fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.axis('tight')

        # Panel 3: Final centerline on original mask with stats
        ax = fig.add_subplot(gs[row_idx, 2])
        ax.imshow(mask, cmap='gray', alpha=0.6)
        ax.plot(result['centerline_smoothed'][:, 0], result['centerline_smoothed'][:, 1], 'r-',
               linewidth=2.5, label='Centerline')
        ax.plot(result['endpoints'][:, 0], result['endpoints'][:, 1], 'g*', markersize=12)
        ax.set_title(f'Final Result\nLength: {result["length_um"]:.2f} μm | κ_mean: {result["mean_curvature"]:.6f} 1/μm',
                    fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.axis('tight')

    plt.suptitle(f'Gaussian Blur Sigma Sweep: {snip_id}', fontsize=14, fontweight='bold', y=0.995)
    output_path = output_dir / f"{snip_id}_gaussian_sweep.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()

    # Print summary table
    print(f"\n{'SIGMA':<8} {'SUCCESS':<10} {'LENGTH (μm)':<15} {'MEAN κ':<15} {'POINTS':<10}")
    print("-" * 60)
    for sigma in sigma_values:
        if sigma in results_by_sigma and results_by_sigma[sigma]['success']:
            r = results_by_sigma[sigma]
            print(f"{sigma:<8} {'✓':<10} {r['length_um']:<15.2f} {r['mean_curvature']:<15.6f} {r['n_centerline_points']:<10}")
        else:
            print(f"{sigma:<8} {'✗':<10} {'FAILED':<15} {'':<15} {'':<10}")

    return results_by_sigma


if __name__ == "__main__":
    output_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251028/troublesome_masks")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test these three embryos with a range of sigma values
    embryos = [
        "20251017_combined_A02_e01_t0064",
        "20251017_combined_D12_e01_t0057",
        "20251017_combined_D10_e01_t0108",
    ]

    sigma_values = [2, 5, 10, 15, 20, 25, 30]

    print(f"\n{'='*100}")
    print("GAUSSIAN BLUR SIGMA SWEEP ON THREE PROBLEMATIC EMBRYOS")
    print(f"{'='*100}")
    print(f"Sigma values to test: {sigma_values}")
    print(f"Output directory: {output_dir}")

    all_results = {}

    for embryo_id in embryos:
        try:
            results = test_embryo_with_sigmas(embryo_id, sigma_values, output_dir)
            all_results[embryo_id] = results
        except Exception as e:
            print(f"✗ Failed to process {embryo_id}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*100}")
    print("✓ SWEEP COMPLETE")
    print(f"{'='*100}\n")
