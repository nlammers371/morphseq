"""
Test Mask Refinement Methods for Geodesic Distance Robustness

Compares 5 mask refinement approaches on a known problem embryo:
1. Baseline (current cleaning)
2. Gaussian Blur + Threshold
3. Large Closing (100px radius)
4. Distance Transform Core (30th percentile)
5. Alpha-Shape Hull

For each method:
- Times the mask refinement operation
- Times the geodesic extraction
- Computes Hausdorff distance vs PCA reference
- Visualizes refined mask + splines

Outputs:
- test_mask_refinement_comparison.png: 6-panel visualization
- test_mask_refinement_timing.png: Performance bar chart
- test_mask_refinement_results.csv: Detailed results
"""

import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.spatial import ConvexHull
from scipy.spatial.distance import directed_hausdorff
from skimage import morphology, measure
from skimage.morphology import skeletonize
import warnings
warnings.filterwarnings('ignore')

from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle
from segmentation_sandbox.scripts.utils.mask_cleaning import clean_embryo_mask
from segmentation_sandbox.scripts.utils.bodyaxis_spline_utils import align_spline_orientation

# Import existing analyzer classes
sys.path.insert(0, str(Path(__file__).parent.parent / "20251022"))
from geodesic_bspline_smoothing import GeodesicBSplineAnalyzer
from test_pca_smoothing import PCACurvatureAnalyzer


# =============================================================================
# MASK REFINEMENT METHODS
# =============================================================================

def refine_baseline(mask: np.ndarray) -> np.ndarray:
    """Baseline: Current mask cleaning (no additional refinement)"""
    cleaned, _ = clean_embryo_mask(mask, verbose=False)
    return cleaned


def refine_gaussian_blur(mask: np.ndarray, sigma: float = 10, threshold: float = 0.7) -> np.ndarray:
    """
    Gaussian Blur + Re-threshold to keep only core regions.

    Strategy: Blur mask heavily, then re-threshold at high value.
    Only regions with many nearby mask pixels survive.
    """
    # First apply standard cleaning
    cleaned, _ = clean_embryo_mask(mask, verbose=False)

    # Blur the cleaned mask
    blurred = gaussian_filter(cleaned.astype(float), sigma=sigma)

    # Re-threshold to keep core
    core_mask = blurred > threshold

    # Intersect with original cleaned mask to avoid growing
    refined = cleaned & core_mask

    # Small dilation to recover some lost area
    refined = morphology.binary_dilation(refined, morphology.disk(5))

    # Keep largest component
    labeled = measure.label(refined)
    if labeled.max() > 0:
        largest = np.argmax([np.sum(labeled == i) for i in range(1, labeled.max() + 1)]) + 1
        refined = (labeled == largest)

    return refined




def refine_distance_core(mask: np.ndarray, percentile: float = 30) -> np.ndarray:
    """
    Distance Transform Core: Keep only pixels where thickness > percentile threshold.

    Strategy: Compute distance transform (local radius), keep only thick regions.
    """
    # First apply standard cleaning
    cleaned, _ = clean_embryo_mask(mask, verbose=False)

    # Compute distance transform
    dist_transform = distance_transform_edt(cleaned)

    # Find threshold (Xth percentile of non-zero distances)
    mask_distances = dist_transform[cleaned]
    thickness_threshold = np.percentile(mask_distances, percentile)

    # Keep only core regions
    core_mask = dist_transform >= thickness_threshold

    # Dilate slightly to reconnect nearby regions
    core_mask = morphology.binary_dilation(core_mask, morphology.disk(10))

    # Intersect with original
    refined = cleaned & core_mask

    # Keep largest component
    labeled = measure.label(refined)
    if labeled.max() > 0:
        largest = np.argmax([np.sum(labeled == i) for i in range(1, labeled.max() + 1)]) + 1
        refined = (labeled == largest)

    return refined


def refine_alpha_shape(mask: np.ndarray, alpha: float = 50) -> np.ndarray:
    """
    Alpha-Shape Hull: Smooth convex hull with concavity tolerance.

    Strategy: Compute convex hull, then erode to allow natural concavity.
    (Simplified version - true alpha-shapes require alphashape library)
    """
    # First apply standard cleaning
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

        # Fill hull interior using polygon fill
        from skimage.draw import polygon

        # Get hull vertices
        hull_points = points[hull.vertices]
        rr, cc = polygon(hull_points[:, 0], hull_points[:, 1], cleaned.shape)
        hull_mask[rr, cc] = True

        # Erode hull to allow concavity (erosion amount based on alpha parameter)
        erode_radius = max(5, int(alpha / 10))
        eroded_hull = morphology.binary_erosion(hull_mask, morphology.disk(erode_radius))

        # Intersect with original cleaned mask
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
        print(f"Warning: Alpha-shape failed ({e}), returning baseline")
        return cleaned


# =============================================================================
# GEODESIC EXTRACTION
# =============================================================================

def extract_geodesic_spline(mask: np.ndarray) -> tuple:
    """
    Extract geodesic centerline from mask using existing GeodesicBSplineAnalyzer.

    Returns:
        spline_x, spline_y: Geodesic spline coordinates
    """
    try:
        analyzer = GeodesicBSplineAnalyzer(mask, um_per_pixel=1.0, bspline_smoothing=5.0)

        # Extract centerline
        centerline, endpoints, skeleton = analyzer.extract_centerline()

        # Smooth with B-spline
        smoothed_points, tck = analyzer.smooth_with_bspline(centerline)

        if smoothed_points is None or len(smoothed_points) == 0:
            return np.array([]), np.array([])

        return smoothed_points[:, 0], smoothed_points[:, 1]

    except Exception as e:
        print(f"Warning: Geodesic extraction failed ({e})")
        return np.array([]), np.array([])


def extract_pca_spline(mask: np.ndarray) -> tuple:
    """
    Extract PCA centerline from mask using existing PCACurvatureAnalyzer.

    Returns:
        spline_x, spline_y: PCA spline coordinates
    """
    try:
        analyzer = PCACurvatureAnalyzer(mask, um_per_pixel=1.0)

        # Extract centerline
        centerline_points = analyzer.extract_centerline_pca(n_slices=100)

        if centerline_points is None or len(centerline_points) == 0:
            return np.array([]), np.array([])

        # Compute curvature (which also returns spline coords)
        arc_length, curvature, spline_x, spline_y = analyzer.compute_curvature(
            centerline_points, smoothing=5.0
        )

        if len(spline_x) == 0:
            return np.array([]), np.array([])

        return spline_x, spline_y

    except Exception as e:
        print(f"Warning: PCA extraction failed ({e})")
        return np.array([]), np.array([])


# =============================================================================
# COMPARISON METRICS
# =============================================================================

def compute_hausdorff_distance(spline1_x, spline1_y, spline2_x, spline2_y):
    """
    Compute Hausdorff distance between two splines after alignment.

    Returns:
        hausdorff_distance: Symmetric Hausdorff distance
        was_flipped: Whether spline2 was flipped for alignment
    """
    if len(spline1_x) == 0 or len(spline2_x) == 0:
        return np.nan, False

    # Align orientations
    spline2_x_aligned, spline2_y_aligned, was_flipped = align_spline_orientation(
        spline1_x, spline1_y, spline2_x, spline2_y
    )

    # Stack into (N, 2) arrays
    spline1 = np.column_stack([spline1_x, spline1_y])
    spline2 = np.column_stack([spline2_x_aligned, spline2_y_aligned])

    # Symmetric Hausdorff distance
    hausdorff_1to2 = directed_hausdorff(spline1, spline2)[0]
    hausdorff_2to1 = directed_hausdorff(spline2, spline1)[0]
    hausdorff = max(hausdorff_1to2, hausdorff_2to1)

    return float(hausdorff), was_flipped


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_mask_spline_comparison(results: list, output_path: Path):
    """
    Generate 5-panel visualization comparing masks and splines.

    Layout (2x3 grid with last spot for PCA reference):
        [Baseline]       [Gaussian Blur]   [Distance Core]
        [Alpha-Shape]    [PCA Reference]   [empty]
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Find PCA reference result
    pca_result = None
    for r in results:
        if r['method'] == 'PCA Reference':
            pca_result = r
            break

    pca_x = pca_result['pca_x'] if pca_result else np.array([])
    pca_y = pca_result['pca_y'] if pca_result else np.array([])

    for idx, result in enumerate(results):
        ax = axes[idx]

        # Show refined mask
        ax.imshow(result['refined_mask'], cmap='gray', alpha=0.7)

        # Plot PCA spline (reference - blue)
        if len(pca_x) > 0:
            ax.plot(pca_x, pca_y, 'b-', linewidth=2, label='PCA', alpha=0.8)

        # Plot Geodesic spline (red)
        if len(result['geo_x']) > 0:
            ax.plot(result['geo_x'], result['geo_y'], 'r-', linewidth=2,
                   label='Geodesic', alpha=0.8)

        # Title with method, Hausdorff, and timing
        title = f"{result['method']}\n"
        title += f"Hausdorff: {result['hausdorff']:.2f} px\n"
        title += f"Total: {result['total_time']:.1f} ms"
        ax.set_title(title, fontsize=11, fontweight='bold')

        ax.legend(loc='upper right', fontsize=9)
        ax.axis('off')

    # Hide last empty subplot
    if len(results) < len(axes):
        axes[-1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_timing_comparison(results: list, output_path: Path):
    """
    Generate bar chart comparing timing for each method.

    Stacked bars: refinement time + geodesic time
    """
    # Filter out PCA reference for timing comparison
    method_results = [r for r in results if r['method'] != 'PCA Reference']

    methods = [r['method'] for r in method_results]
    refinement_times = [r['refinement_time'] for r in method_results]
    geodesic_times = [r['geodesic_time'] for r in method_results]
    total_times = [r['total_time'] for r in method_results]

    # Get baseline total time for reference line
    baseline_total = method_results[0]['total_time']

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(methods))
    width = 0.6

    # Stacked bars
    ax.bar(x, refinement_times, width, label='Mask Refinement', color='steelblue')
    ax.bar(x, geodesic_times, width, bottom=refinement_times,
           label='Geodesic Extraction', color='coral')

    # Baseline reference line
    ax.axhline(baseline_total, color='gray', linestyle='--', linewidth=2,
               alpha=0.7, label='Baseline Total')

    # Labels and formatting
    ax.set_ylabel('Time (milliseconds)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Refinement Method', fontsize=13, fontweight='bold')
    ax.set_title('Performance Comparison: Mask Refinement Methods',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y', linestyle=':')

    # Add total time annotations on bars
    for i, total in enumerate(total_times):
        ax.text(i, total + 2, f'{total:.1f}', ha='center', va='bottom',
               fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


# =============================================================================
# MAIN TEST PIPELINE
# =============================================================================

def test_mask_refinement_methods(snip_id: str = "20251017_part2_H07_e01_t0008"):
    """
    Test all mask refinement methods on a single embryo.

    Args:
        snip_id: Embryo identifier (default: known fin problem case)
    """
    output_dir = Path(__file__).parent

    print("="*70)
    print("MASK REFINEMENT METHOD COMPARISON")
    print("="*70)
    print(f"Test Embryo: {snip_id}")
    print()

    # 1. Load embryo mask
    print("Loading embryo mask...")
    metadata_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build06_output")

    # Try part1 first, then part2
    df_metadata = None
    for part in ['part1', 'part2']:
        csv_path = metadata_dir / f"df03_final_output_with_latents_20251017_{part}.csv"
        if csv_path.exists():
            df_part = pd.read_csv(csv_path)
            embryo_row = df_part[df_part['snip_id'] == snip_id]
            if len(embryo_row) > 0:
                df_metadata = embryo_row
                break

    if df_metadata is None or len(df_metadata) == 0:
        print(f"ERROR: Could not find embryo {snip_id} in metadata!")
        return

    embryo_data = df_metadata.iloc[0]

    # Decode mask
    mask = decode_mask_rle({
        'size': [int(embryo_data['mask_height_px']), int(embryo_data['mask_width_px'])],
        'counts': embryo_data['mask_rle']
    })

    print(f"Loaded mask: {mask.shape} (area: {mask.sum():,} px)")
    print()

    # 2. Extract PCA reference (baseline for comparison)
    print("Extracting PCA reference...")
    t0 = time.perf_counter()
    pca_x, pca_y = extract_pca_spline(mask)
    pca_time = (time.perf_counter() - t0) * 1000
    print(f"PCA extraction: {pca_time:.1f} ms")
    print()

    # 3. Test each refinement method (removed Large Closing due to memory issues)
    methods = [
        ("Baseline", refine_baseline),
        ("Gaussian Blur", refine_gaussian_blur),
        ("Distance Core", refine_distance_core),
        ("Alpha-Shape", refine_alpha_shape),
    ]

    results = []

    for method_name, refine_func in methods:
        print(f"Testing: {method_name}")
        print("-" * 50)

        # Time mask refinement
        t0 = time.perf_counter()
        refined_mask = refine_func(mask)
        refinement_time = (time.perf_counter() - t0) * 1000

        print(f"  Refinement time: {refinement_time:.1f} ms")
        print(f"  Refined mask area: {refined_mask.sum():,} px ({refined_mask.sum()/mask.sum()*100:.1f}% of original)")

        # Time geodesic extraction
        t0 = time.perf_counter()
        geo_x, geo_y = extract_geodesic_spline(refined_mask)
        geodesic_time = (time.perf_counter() - t0) * 1000

        print(f"  Geodesic time: {geodesic_time:.1f} ms")

        total_time = refinement_time + geodesic_time

        # Compute Hausdorff distance vs PCA
        if len(geo_x) > 0 and len(pca_x) > 0:
            hausdorff, _ = compute_hausdorff_distance(pca_x, pca_y, geo_x, geo_y)
            print(f"  Hausdorff distance: {hausdorff:.2f} px")
        else:
            hausdorff = np.nan
            print(f"  Hausdorff distance: FAILED (empty spline)")

        print(f"  Total time: {total_time:.1f} ms")
        print()

        results.append({
            'method': method_name,
            'refinement_time': refinement_time,
            'geodesic_time': geodesic_time,
            'total_time': total_time,
            'hausdorff': hausdorff,
            'refined_mask': refined_mask,
            'geo_x': geo_x,
            'geo_y': geo_y,
            'pca_x': pca_x,
            'pca_y': pca_y,
        })

    # Add PCA reference to results (for visualization)
    results.append({
        'method': 'PCA Reference',
        'refinement_time': 0,
        'geodesic_time': pca_time,
        'total_time': pca_time,
        'hausdorff': 0.0,
        'refined_mask': mask,
        'geo_x': pca_x,
        'geo_y': pca_y,
        'pca_x': pca_x,
        'pca_y': pca_y,
    })

    # 4. Generate visualizations
    print("Generating visualizations...")
    plot_mask_spline_comparison(results, output_dir / "test_mask_refinement_comparison.png")
    plot_timing_comparison(results, output_dir / "test_mask_refinement_timing.png")

    # 5. Save results to CSV
    results_df = pd.DataFrame([
        {
            'method': r['method'],
            'refinement_time_ms': r['refinement_time'],
            'geodesic_time_ms': r['geodesic_time'],
            'total_time_ms': r['total_time'],
            'hausdorff_distance_px': r['hausdorff'],
        }
        for r in results
    ])

    csv_path = output_dir / "test_mask_refinement_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    print()

    # 6. Print summary table
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()

    # Format table
    print(f"{'Method':<20} {'Refine (ms)':<12} {'Geodesic (ms)':<14} {'Total (ms)':<11} {'Hausdorff (px)':<15} {'Speedup'}")
    print("-" * 90)

    baseline_total = results[0]['total_time']

    for r in results:
        if r['method'] == 'PCA Reference':
            # Special formatting for PCA
            speedup = baseline_total / r['total_time']
            print(f"{'PCA (reference)':<20} {'-':<12} {r['geodesic_time']:>11.1f}    {r['total_time']:>8.1f}    {r['hausdorff']:>12.2f}      {speedup:>5.2f}x")
        else:
            speedup = baseline_total / r['total_time'] if r['total_time'] > 0 else 0
            print(f"{r['method']:<20} {r['refinement_time']:>11.1f}  {r['geodesic_time']:>11.1f}    {r['total_time']:>8.1f}    {r['hausdorff']:>12.2f}      {speedup:>5.2f}x")

    print()
    print("="*70)

    # Find best methods
    method_results = [r for r in results if r['method'] != 'PCA Reference']

    best_accuracy = min(method_results, key=lambda x: x['hausdorff'] if not np.isnan(x['hausdorff']) else float('inf'))
    best_speed = min(method_results, key=lambda x: x['total_time'])

    # Best balance: normalized score (accuracy improvement + speed)
    baseline_hausdorff = results[0]['hausdorff']
    best_balance = min(method_results, key=lambda x:
        (x['hausdorff'] / baseline_hausdorff + x['total_time'] / baseline_total) / 2
        if not np.isnan(x['hausdorff']) else float('inf')
    )

    accuracy_improvement = (1 - best_accuracy['hausdorff'] / baseline_hausdorff) * 100
    print(f"Best Accuracy: {best_accuracy['method']} ({best_accuracy['hausdorff']:.2f} px, {accuracy_improvement:.0f}% improvement)")

    speed_improvement = (1 - best_speed['total_time'] / baseline_total) * 100
    print(f"Best Speed: {best_speed['method']} ({best_speed['total_time']:.1f} ms, {speed_improvement:.0f}% faster)")

    balance_accuracy_improvement = (1 - best_balance['hausdorff'] / baseline_hausdorff) * 100
    balance_speed = best_balance['total_time'] / baseline_total * 100
    print(f"Best Balance: {best_balance['method']} ({best_balance['hausdorff']:.2f} px, {balance_accuracy_improvement:.0f}% improvement, {balance_speed:.0f}% speed)")

    print("="*70)


if __name__ == "__main__":
    # Test on known fin problem case
    test_mask_refinement_methods("20251017_part2_H07_e01_t0008")
