"""
Visualize the fixed geodesic method results
"""

import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from skimage import morphology
from scipy.ndimage import distance_transform_edt

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


def visualize_embryo(snip_id: str, output_dir: Path):
    """Create comprehensive visualization of fixed geodesic method."""
    print(f"Visualizing: {snip_id}")

    # Load data
    mask, um_per_pixel = load_embryo_data(snip_id)

    # Clean mask
    mask_cleaned, _ = clean_embryo_mask(mask, verbose=False)

    # Preprocess
    mask_preprocessed = apply_preprocessing(mask_cleaned, method='gaussian_blur',
                                           sigma=10.0, threshold=0.7)

    # Run analyzer
    analyzer = GeodesicCenterlineAnalyzer(
        mask_preprocessed,
        um_per_pixel=um_per_pixel,
        bspline_smoothing=5.0,
        random_seed=42,
        fast=True
    )

    results = analyzer.analyze()

    # Extract components
    centerline_raw = results['centerline_raw']
    centerline_smoothed = results['centerline_smoothed']
    skeleton = results['skeleton']
    endpoints = results['endpoints']

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Panel 1: Original mask with skeleton
    ax = axes[0, 0]
    ax.imshow(mask, cmap='gray', alpha=0.6)
    skeleton_display = morphology.skeletonize(mask_preprocessed)
    ax.contour(skeleton_display, colors='blue', linewidths=1, levels=[0.5])
    ax.plot(endpoints[:, 0], endpoints[:, 1], 'r*', markersize=15, label='Endpoints')
    ax.set_title(f'Original Mask + Skeleton\n{snip_id}', fontsize=12, fontweight='bold')
    ax.set_xlabel('X (px)')
    ax.set_ylabel('Y (px)')
    ax.legend()

    # Panel 2: Preprocessed mask with distance transform
    ax = axes[0, 1]
    dist_transform = distance_transform_edt(mask_preprocessed)
    im = ax.imshow(dist_transform, cmap='hot')
    ax.contour(mask_preprocessed, colors='white', linewidths=1, levels=[0.5])
    ax.plot(endpoints[:, 0], endpoints[:, 1], 'b*', markersize=15, label='Endpoints')
    plt.colorbar(im, ax=ax, label='Distance (px)')
    ax.set_title('Preprocessed Mask + Distance Transform', fontsize=12, fontweight='bold')
    ax.set_xlabel('X (px)')
    ax.set_ylabel('Y (px)')
    ax.legend()

    # Panel 3: Raw centerline on skeleton
    ax = axes[1, 0]
    ax.imshow(skeleton, cmap='gray', alpha=0.7)
    ax.plot(centerline_raw[:, 0], centerline_raw[:, 1], 'b-', linewidth=2, label='Raw Centerline')
    ax.plot(endpoints[:, 0], endpoints[:, 1], 'r*', markersize=15, label='Endpoints')
    ax.set_title(f'Raw Centerline ({len(centerline_raw)} points)', fontsize=12, fontweight='bold')
    ax.set_xlabel('X (px)')
    ax.set_ylabel('Y (px)')
    ax.legend()

    # Panel 4: Smoothed centerline on original mask
    ax = axes[1, 1]
    ax.imshow(mask, cmap='gray', alpha=0.6)
    ax.plot(centerline_raw[:, 0], centerline_raw[:, 1], 'b--', linewidth=1.5, 
            alpha=0.5, label='Raw Centerline')
    ax.plot(centerline_smoothed[:, 0], centerline_smoothed[:, 1], 'r-', linewidth=2.5, 
            label='Smoothed Centerline')
    ax.plot(endpoints[:, 0], endpoints[:, 1], 'g*', markersize=15, label='Endpoints')
    ax.set_title(f'Final Centerline on Original Mask\nLength: {results["stats"]["total_length"]:.2f} μm', 
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('X (px)')
    ax.set_ylabel('Y (px)')
    ax.legend()

    plt.tight_layout()
    output_path = output_dir / f"{snip_id}_fixed_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()

    # Create curvature plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Arc length vs curvature
    ax = axes[0]
    arc_length = results['arc_length']
    curvature = results['curvature']
    ax.plot(arc_length, curvature, 'b-', linewidth=2)
    ax.fill_between(arc_length, 0, curvature, alpha=0.3)
    ax.set_xlabel('Arc Length (μm)', fontsize=11)
    ax.set_ylabel('Curvature (1/μm)', fontsize=11)
    ax.set_title('Curvature Along Centerline', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Stats text
    ax = axes[1]
    ax.axis('off')
    stats_text = f"""
    GEODESIC CENTERLINE ANALYSIS - Fixed Implementation
    
    Embryo ID: {snip_id}
    
    PREPROCESSING:
      • Original area: {mask.sum():,} px
      • Cleaned area: {mask_cleaned.sum():,} px
      • Preprocessed area: {mask_preprocessed.sum():,} px
      • Retention: {mask_preprocessed.sum()/mask_cleaned.sum()*100:.1f}%
    
    SKELETON:
      • Total skeleton points: {results['stats']['n_skeleton_points']:,}
      • Centerline points (raw): {results['stats']['n_centerline_points']}
      • Centerline points (smoothed): 200
    
    CENTERLINE PROPERTIES:
      • Total length: {results['stats']['total_length']:.2f} μm
      • Mean curvature: {results['stats']['mean_curvature']:.6f} 1/μm
      • Max curvature: {results['stats']['max_curvature']:.6f} 1/μm
      • Std curvature: {results['stats']['std_curvature']:.6f} 1/μm
    
    CONFIGURATION:
      • um_per_pixel: {um_per_pixel:.6f}
      • B-spline smoothing: {results['stats']['bspline_smoothing']}
      • Method: {results['stats']['method']}
      • Fast mode: {results['stats']['fast_mode']}
    """
    ax.text(0.1, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = output_dir / f"{snip_id}_fixed_curvature.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()

    return results


if __name__ == "__main__":
    output_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251028/troublesome_masks")
    output_dir.mkdir(parents=True, exist_ok=True)

    failing_embryos = [
        "20251017_combined_C04_e01_t0114",
        "20251017_combined_F11_e01_t0065",
    ]

    print(f"\n{'='*80}")
    print("CREATING VISUALIZATIONS FOR FIXED GEODESIC METHOD")
    print(f"{'='*80}\n")

    for embryo_id in failing_embryos:
        try:
            visualize_embryo(embryo_id, output_dir)
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print("✓ ALL VISUALIZATIONS COMPLETE")
    print(f"{'='*80}\n")
