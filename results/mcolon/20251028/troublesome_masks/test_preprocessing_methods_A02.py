"""
Comprehensive diagnostic of preprocessing methods for A02 embryo (fin problem)

Tests multiple alpha shape values and Gaussian blur to find optimal preprocessing
that prevents centerline from following fin instead of body.
"""

import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter
from scipy.ndimage import distance_transform_edt
from skimage import morphology, measure
from skimage.draw import polygon

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


def apply_alpha_shape_preprocessing(mask: np.ndarray, alpha: float = 50) -> np.ndarray:
    """Apply alpha-shape preprocessing with specified alpha value."""
    cleaned, _ = clean_embryo_mask(mask, verbose=False)

    # Get mask boundary points
    points = np.column_stack(np.where(cleaned))

    if len(points) < 3:
        return cleaned

    try:
        # Compute convex hull
        hull = ConvexHull(points)

        # Create hull mask
        hull_mask = np.zeros_like(cleaned, dtype=bool)
        hull_points = points[hull.vertices]
        rr, cc = polygon(hull_points[:, 0], hull_points[:, 1], cleaned.shape)
        hull_mask[rr, cc] = True

        # Erode hull to allow concavity
        erode_radius = max(5, int(alpha / 10))
        eroded_hull = morphology.binary_erosion(hull_mask, morphology.disk(erode_radius))

        # Intersect with original
        refined = cleaned & eroded_hull

        # If erosion removed too much, use less erosion
        if refined.sum() < 0.5 * cleaned.sum():
            erode_radius = max(3, int(alpha / 20))
            eroded_hull = morphology.binary_erosion(hull_mask, morphology.disk(erode_radius))
            refined = cleaned & eroded_hull

        # Keep largest component
        labeled = measure.label(refined)
        if labeled.max() > 0:
            largest = np.argmax([np.sum(labeled == i) for i in range(1, labeled.max() + 1)]) + 1
            refined = (labeled == largest)

        return refined

    except Exception as e:
        print(f"Warning: Alpha-shape failed ({e}), returning cleaned mask")
        return cleaned


def test_preprocessing_method(snip_id: str, mask_cleaned: np.ndarray, um_per_pixel: float,
                             method_name: str, mask_preprocessed: np.ndarray):
    """
    Test a single preprocessing method and return results.
    
    Returns:
        dict with results or None if failed
    """
    try:
        analyzer = GeodesicCenterlineAnalyzer(
            mask_preprocessed,
            um_per_pixel=um_per_pixel,
            bspline_smoothing=5.0,
            random_seed=42,
            fast=True
        )

        results = analyzer.analyze()

        return {
            'method': method_name,
            'success': True,
            'preprocessed_mask': mask_preprocessed,
            'centerline_raw': results['centerline_raw'],
            'centerline_smoothed': results['centerline_smoothed'],
            'skeleton': results['skeleton'],
            'endpoints': results['endpoints'],
            'total_length': results['stats']['total_length'],
            'mean_curvature': results['stats']['mean_curvature'],
            'max_curvature': results['stats']['max_curvature'],
            'area_retention': mask_preprocessed.sum() / mask_cleaned.sum() * 100,
        }
    except Exception as e:
        print(f"    ✗ {method_name} failed: {e}")
        return {
            'method': method_name,
            'success': False,
            'error': str(e),
            'preprocessed_mask': mask_preprocessed,
        }


def analyze_embryo(snip_id: str, output_dir: Path):
    """Comprehensive analysis with multiple preprocessing methods."""
    print(f"\n{'='*90}")
    print(f"ANALYZING: {snip_id}")
    print(f"{'='*90}")

    # Load data
    mask, um_per_pixel = load_embryo_data(snip_id)
    mask_cleaned, _ = clean_embryo_mask(mask, verbose=False)

    print(f"\nOriginal area: {mask.sum():,} px")
    print(f"Cleaned area: {mask_cleaned.sum():,} px")

    results_list = []

    # Test 1: Current Gaussian blur
    print(f"\nTesting Gaussian blur (sigma=10)...")
    mask_gaussian = apply_preprocessing(mask_cleaned, method='gaussian_blur',
                                       sigma=10.0, threshold=0.7)
    result = test_preprocessing_method(snip_id, mask_cleaned, um_per_pixel,
                                      "Gaussian (σ=10)", mask_gaussian)
    results_list.append(result)
    if result['success']:
        print(f"  ✓ Length: {result['total_length']:.2f} μm, Area retention: {result['area_retention']:.1f}%")

    # Test 2-11: Alpha shapes with different values
    alpha_values = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    print(f"\nTesting alpha shapes with varying aggressiveness...")
    for alpha in alpha_values:
        print(f"  Alpha={alpha}...", end=" ", flush=True)
        mask_alpha = apply_alpha_shape_preprocessing(mask_cleaned, alpha=alpha)
        result = test_preprocessing_method(snip_id, mask_cleaned, um_per_pixel,
                                          f"Alpha={alpha}", mask_alpha)
        results_list.append(result)
        if result['success']:
            print(f"✓ Len: {result['total_length']:.0f}μm, Ret: {result['area_retention']:.0f}%")
        else:
            print(f"✗")

    return results_list, mask_cleaned, um_per_pixel


def visualize_results(snip_id: str, results_list: list, mask_cleaned: np.ndarray,
                     mask_original: np.ndarray, um_per_pixel: float, output_dir: Path):
    """Create comprehensive visualization grid."""
    
    successful_results = [r for r in results_list if r['success']]
    
    if not successful_results:
        print("No successful results to visualize")
        return
    
    n_methods = len(successful_results)
    n_cols = min(4, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(20, 5 * n_rows))
    gs = GridSpec(n_rows, n_cols * 2, figure=fig, hspace=0.4, wspace=0.3)
    
    for idx, result in enumerate(successful_results):
        row = idx // n_cols
        col = idx % n_cols
        
        # Left panel: preprocessed mask
        ax_left = fig.add_subplot(gs[row, col * 2])
        ax_left.imshow(result['preprocessed_mask'], cmap='gray')
        ax_left.contour(result['preprocessed_mask'], colors='blue', linewidths=1.5, levels=[0.5])
        ax_left.set_title(f"{result['method']}\nMask (Retention: {result['area_retention']:.1f}%)",
                         fontsize=10, fontweight='bold')
        ax_left.set_xlabel('X (px)')
        ax_left.set_ylabel('Y (px)')
        ax_left.axis('on')
        
        # Right panel: centerline on original mask
        ax_right = fig.add_subplot(gs[row, col * 2 + 1])
        ax_right.imshow(mask_original, cmap='gray', alpha=0.6)
        ax_right.plot(result['centerline_raw'][:, 0], result['centerline_raw'][:, 1],
                     'b--', linewidth=1.5, alpha=0.5, label='Raw')
        ax_right.plot(result['centerline_smoothed'][:, 0], result['centerline_smoothed'][:, 1],
                     'r-', linewidth=2.5, label='Smoothed')
        ax_right.plot(result['endpoints'][:, 0], result['endpoints'][:, 1],
                     'g*', markersize=12, label='Endpoints')
        ax_right.set_title(f"Centerline\nLen: {result['total_length']:.0f}μm, κ̄: {result['mean_curvature']:.6f}",
                          fontsize=10, fontweight='bold')
        ax_right.set_xlabel('X (px)')
        ax_right.set_ylabel('Y (px)')
        ax_right.legend(fontsize=8)
        ax_right.axis('on')
    
    plt.suptitle(f"Preprocessing Method Comparison: {snip_id}", fontsize=14, fontweight='bold', y=0.995)
    output_path = output_dir / f"{snip_id}_preprocessing_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison: {output_path}")
    plt.close()


def create_summary_table(snip_id: str, results_list: list, output_dir: Path):
    """Create summary CSV with all method results."""
    
    summary_data = []
    for result in results_list:
        if result['success']:
            summary_data.append({
                'method': result['method'],
                'total_length_um': result['total_length'],
                'mean_curvature': result['mean_curvature'],
                'max_curvature': result['max_curvature'],
                'area_retention_percent': result['area_retention'],
            })
        else:
            summary_data.append({
                'method': result['method'],
                'total_length_um': None,
                'mean_curvature': None,
                'max_curvature': None,
                'area_retention_percent': None,
            })
    
    df = pd.DataFrame(summary_data)
    output_path = output_dir / f"{snip_id}_preprocessing_summary.csv"
    df.to_csv(output_path, index=False)
    print(f"✓ Saved summary table: {output_path}")
    
    print(f"\n{snip_id} PREPROCESSING COMPARISON:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    snip_id = "20251017_combined_A02_e01_t0064"
    output_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251028/troublesome_masks")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze embryo
    results_list, mask_cleaned, um_per_pixel = analyze_embryo(snip_id, output_dir)

    # Load original for visualization
    mask_original, _ = load_embryo_data(snip_id)

    # Visualize
    visualize_results(snip_id, results_list, mask_cleaned, mask_original, um_per_pixel, output_dir)

    # Summary table
    create_summary_table(snip_id, results_list, output_dir)

    print(f"\n{'='*90}")
    print("✓ ANALYSIS COMPLETE")
    print(f"{'='*90}\n")
