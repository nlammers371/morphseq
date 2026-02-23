"""
Test Two Fixes for Disconnected Skeleton Problem

Fix A: Keep largest skeleton component after skeletonization
Fix B: Use alpha shape preprocessing instead of Gaussian blur

Tests both fixes on the 2 failed embryos and compares results.
"""

import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.interpolate import splprep, splev
from skimage import morphology, measure
from skimage.draw import polygon

from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle
from segmentation_sandbox.scripts.utils.mask_cleaning import clean_embryo_mask
from segmentation_sandbox.scripts.body_axis_analysis.centerline_extraction import extract_centerline


# =============================================================================
# LOAD EMBRYO
# =============================================================================

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


# =============================================================================
# FIX A: GEODESIC WITH LARGEST SKELETON COMPONENT
# =============================================================================

def extract_centerline_fix_a(mask: np.ndarray, um_per_pixel: float):
    """
    Extract centerline using modified geodesic method that keeps only
    the largest skeleton component.

    This is the same as the current pipeline but adds a skeleton cleaning step.
    """
    from scipy.ndimage import distance_transform_edt

    # Step 1: Clean mask
    cleaned_mask, _ = clean_embryo_mask(mask, verbose=False)

    # Step 2: Apply Gaussian preprocessing (same as current)
    blurred = gaussian_filter(cleaned_mask.astype(float), sigma=10.0)
    core_mask = blurred > 0.7
    refined = cleaned_mask & core_mask
    refined = morphology.binary_dilation(refined, morphology.disk(5))

    # Keep largest component
    labeled = measure.label(refined)
    if labeled.max() > 0:
        largest = np.argmax([np.sum(labeled == i) for i in range(1, labeled.max() + 1)]) + 1
        preprocessed_mask = (labeled == largest)
    else:
        preprocessed_mask = refined

    # Step 3: Skeletonize
    skeleton = morphology.skeletonize(preprocessed_mask)

    # **FIX A: Keep only largest skeleton component**
    labeled_skel = measure.label(skeleton)
    if labeled_skel.max() > 1:
        # Multiple components - keep largest
        largest_skel = np.argmax([np.sum(labeled_skel == i) for i in range(1, labeled_skel.max() + 1)]) + 1
        skeleton = (labeled_skel == largest_skel)

    skel_points = np.column_stack(np.where(skeleton))

    if len(skel_points) == 0:
        return None, None, None, None, "Empty skeleton after cleaning"

    # Step 4: Find endpoints using distance transform
    dist_transform = distance_transform_edt(preprocessed_mask)
    distances = dist_transform[skeleton]

    if len(distances) == 0:
        return None, None, None, None, "No skeleton points"

    dist_threshold = np.percentile(distances, 90)
    endpoint_candidates = skel_points[distances >= dist_threshold]

    if len(endpoint_candidates) < 2:
        return None, None, None, None, f"Not enough endpoints ({len(endpoint_candidates)})"

    # Find two farthest endpoints
    max_dist = 0
    endpoints = None
    for i in range(len(endpoint_candidates)):
        for j in range(i+1, len(endpoint_candidates)):
            d = np.linalg.norm(endpoint_candidates[i] - endpoint_candidates[j])
            if d > max_dist:
                max_dist = d
                endpoints = np.array([endpoint_candidates[i], endpoint_candidates[j]])

    # Step 5: Build graph and find geodesic path
    point_to_index = {tuple(pt): idx for idx, pt in enumerate(skel_points)}

    rows = []
    cols = []
    weights = []

    neighbour_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    for idx, (y, x) in enumerate(skel_points):
        for dy, dx in neighbour_offsets:
            neighbour = (y + dy, x + dx)
            jdx = point_to_index.get(neighbour)
            if jdx is None or jdx <= idx:
                continue
            rows.append(idx)
            cols.append(jdx)
            weights.append(np.sqrt(dx*dx + dy*dy))

    if not rows:
        return None, None, None, None, "No edges in graph (still disconnected)"

    # Create adjacency matrix
    data = weights + weights
    adj_matrix = csr_matrix(
        (data, (rows + cols, cols + rows)),
        shape=(len(skel_points), len(skel_points)),
    )

    # Find endpoint indices
    start_idx = None
    end_idx = None
    for idx, pt in enumerate(skel_points):
        if np.array_equal(pt, endpoints[0]):
            start_idx = idx
        if np.array_equal(pt, endpoints[1]):
            end_idx = idx

    if start_idx is None or end_idx is None:
        return None, None, None, None, "Endpoints not in skeleton"

    # Dijkstra
    distances_dijkstra, predecessors = dijkstra(adj_matrix, indices=start_idx,
                                                directed=False, return_predecessors=True)

    # Trace path
    path_indices = []
    current = end_idx
    while current != -9999 and current != start_idx:
        path_indices.append(current)
        current = predecessors[current]
        if len(path_indices) > len(skel_points):
            break
    path_indices.append(start_idx)
    path_indices = path_indices[::-1]

    if len(path_indices) < 4:
        return None, None, None, None, f"Path too short ({len(path_indices)} points)"

    centerline = skel_points[path_indices]

    # Step 6: B-spline fitting
    tck, u = splprep([centerline[:, 1], centerline[:, 0]], s=5.0 * len(centerline), k=3)
    u_fine = np.linspace(0, 1, 200)
    x_vals, y_vals = splev(u_fine, tck)

    # Compute curvature
    dx, dy = splev(u_fine, tck, der=1)
    ddx, ddy = splev(u_fine, tck, der=2)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**(3/2)

    # Arc length
    arc_length = np.cumsum(np.sqrt(np.diff(x_vals)**2 + np.diff(y_vals)**2))
    arc_length = np.concatenate([[0], arc_length])
    arc_length = arc_length * um_per_pixel
    curvature = curvature / um_per_pixel

    return x_vals, y_vals, curvature, arc_length, None


# =============================================================================
# FIX B: ALPHA SHAPE PREPROCESSING
# =============================================================================

def apply_alpha_shape_preprocessing(mask: np.ndarray, alpha: float = 50) -> np.ndarray:
    """
    Alpha-shape preprocessing (from test_mask_refinement_methods.py).

    Uses convex hull + erosion to smooth mask while preserving connectivity.
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
        print(f"Warning: Alpha-shape failed ({e}), returning baseline")
        return cleaned


def extract_centerline_fix_b(mask: np.ndarray, um_per_pixel: float, alpha: float = 50):
    """
    Extract centerline using alpha shape preprocessing + geodesic with largest skeleton component.
    """
    # Apply alpha shape preprocessing
    preprocessed_mask = apply_alpha_shape_preprocessing(mask, alpha=alpha)

    # Now use Fix A logic on the alpha-preprocessed mask
    from scipy.ndimage import distance_transform_edt

    # Skeletonize
    skeleton = morphology.skeletonize(preprocessed_mask)

    # Keep only largest skeleton component
    labeled_skel = measure.label(skeleton)
    if labeled_skel.max() > 1:
        largest_skel = np.argmax([np.sum(labeled_skel == i) for i in range(1, labeled_skel.max() + 1)]) + 1
        skeleton = (labeled_skel == largest_skel)

    skel_points = np.column_stack(np.where(skeleton))

    if len(skel_points) == 0:
        return None, None, None, None, "Empty skeleton"

    # Find endpoints
    dist_transform = distance_transform_edt(preprocessed_mask)
    distances = dist_transform[skeleton]

    if len(distances) == 0:
        return None, None, None, None, "No skeleton points"

    dist_threshold = np.percentile(distances, 90)
    endpoint_candidates = skel_points[distances >= dist_threshold]

    if len(endpoint_candidates) < 2:
        return None, None, None, None, f"Not enough endpoints ({len(endpoint_candidates)})"

    # Find two farthest endpoints
    max_dist = 0
    endpoints = None
    for i in range(len(endpoint_candidates)):
        for j in range(i+1, len(endpoint_candidates)):
            d = np.linalg.norm(endpoint_candidates[i] - endpoint_candidates[j])
            if d > max_dist:
                max_dist = d
                endpoints = np.array([endpoint_candidates[i], endpoint_candidates[j]])

    # Build graph
    point_to_index = {tuple(pt): idx for idx, pt in enumerate(skel_points)}

    rows = []
    cols = []
    weights = []

    neighbour_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    for idx, (y, x) in enumerate(skel_points):
        for dy, dx in neighbour_offsets:
            neighbour = (y + dy, x + dx)
            jdx = point_to_index.get(neighbour)
            if jdx is None or jdx <= idx:
                continue
            rows.append(idx)
            cols.append(jdx)
            weights.append(np.sqrt(dx*dx + dy*dy))

    if not rows:
        return None, None, None, None, "No edges in graph"

    data = weights + weights
    adj_matrix = csr_matrix(
        (data, (rows + cols, cols + rows)),
        shape=(len(skel_points), len(skel_points)),
    )

    # Find endpoint indices
    start_idx = None
    end_idx = None
    for idx, pt in enumerate(skel_points):
        if np.array_equal(pt, endpoints[0]):
            start_idx = idx
        if np.array_equal(pt, endpoints[1]):
            end_idx = idx

    if start_idx is None or end_idx is None:
        return None, None, None, None, "Endpoints not in skeleton"

    # Dijkstra
    distances_dijkstra, predecessors = dijkstra(adj_matrix, indices=start_idx,
                                                directed=False, return_predecessors=True)

    # Trace path
    path_indices = []
    current = end_idx
    while current != -9999 and current != start_idx:
        path_indices.append(current)
        current = predecessors[current]
        if len(path_indices) > len(skel_points):
            break
    path_indices.append(start_idx)
    path_indices = path_indices[::-1]

    if len(path_indices) < 4:
        return None, None, None, None, f"Path too short ({len(path_indices)} points)"

    centerline = skel_points[path_indices]

    # B-spline fitting
    tck, u = splprep([centerline[:, 1], centerline[:, 0]], s=5.0 * len(centerline), k=3)
    u_fine = np.linspace(0, 1, 200)
    x_vals, y_vals = splev(u_fine, tck)

    # Compute curvature
    dx, dy = splev(u_fine, tck, der=1)
    ddx, ddy = splev(u_fine, tck, der=2)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**(3/2)

    # Arc length
    arc_length = np.cumsum(np.sqrt(np.diff(x_vals)**2 + np.diff(y_vals)**2))
    arc_length = np.concatenate([[0], arc_length])
    arc_length = arc_length * um_per_pixel
    curvature = curvature / um_per_pixel

    return x_vals, y_vals, curvature, arc_length, None


# =============================================================================
# TEST & VISUALIZE
# =============================================================================

def test_embryo(snip_id: str):
    """Test both fixes on a single embryo."""
    print(f"\n{'='*80}")
    print(f"Testing: {snip_id}")
    print(f"{'='*80}\n")

    mask, um_per_pixel = load_embryo_data(snip_id)

    results = {
        'snip_id': snip_id,
        'mask': mask
    }

    # Test Fix A
    print("Testing Fix A: Largest skeleton component...")
    t0 = time.perf_counter()
    x_a, y_a, curv_a, arc_a, err_a = extract_centerline_fix_a(mask, um_per_pixel)
    time_a = (time.perf_counter() - t0) * 1000

    success_a = x_a is not None
    if success_a:
        print(f"  ✓ SUCCESS - Length: {arc_a[-1]:.1f} μm, Time: {time_a:.1f} ms")
        results['fix_a'] = {
            'success': True,
            'spline_x': x_a,
            'spline_y': y_a,
            'curvature': curv_a,
            'arc_length': arc_a,
            'length': float(arc_a[-1]),
            'mean_curvature': float(np.mean(np.abs(curv_a))),
            'time_ms': time_a
        }
    else:
        print(f"  ✗ FAILED - {err_a}")
        results['fix_a'] = {'success': False, 'error': err_a}

    # Test Fix B with multiple alpha values
    print("\nTesting Fix B: Alpha shape preprocessing...")
    alpha_values = [30, 50, 70, 90, 110]
    best_alpha_result = None

    for alpha in alpha_values:
        t0 = time.perf_counter()
        x_b, y_b, curv_b, arc_b, err_b = extract_centerline_fix_b(mask, um_per_pixel, alpha=alpha)
        time_b = (time.perf_counter() - t0) * 1000

        if x_b is not None:
            print(f"  ✓ α={alpha}: SUCCESS - Length: {arc_b[-1]:.1f} μm, Time: {time_b:.1f} ms")
            if best_alpha_result is None or arc_b[-1] > best_alpha_result['length']:
                best_alpha_result = {
                    'success': True,
                    'alpha': alpha,
                    'spline_x': x_b,
                    'spline_y': y_b,
                    'curvature': curv_b,
                    'arc_length': arc_b,
                    'length': float(arc_b[-1]),
                    'mean_curvature': float(np.mean(np.abs(curv_b))),
                    'time_ms': time_b
                }
        else:
            print(f"  ✗ α={alpha}: FAILED - {err_b}")

    if best_alpha_result:
        results['fix_b'] = best_alpha_result
    else:
        results['fix_b'] = {'success': False, 'error': 'All alpha values failed'}

    return results


def plot_comparison(results: dict, output_path: Path):
    """Visualize comparison of both fixes."""
    snip_id = results['snip_id']
    mask = results['mask']

    fix_a = results.get('fix_a', {})
    fix_b = results.get('fix_b', {})

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Panel 1: Original mask
    ax = axes[0]
    ax.imshow(mask, cmap='gray', alpha=0.7)
    ax.set_title("Original Mask", fontsize=14, fontweight='bold')
    ax.axis('off')

    # Panel 2: Fix A
    ax = axes[1]
    ax.imshow(mask, cmap='gray', alpha=0.7)
    if fix_a.get('success'):
        ax.plot(fix_a['spline_x'], fix_a['spline_y'], 'r-', linewidth=2.5, label='Centerline')
        ax.plot(fix_a['spline_x'][0], fix_a['spline_y'][0], 'go', markersize=10, label='Head')
        ax.plot(fix_a['spline_x'][-1], fix_a['spline_y'][-1], 'bo', markersize=10, label='Tail')
        title = f"✓ Fix A: Largest Skeleton Component\n"
        title += f"Length: {fix_a['length']:.1f} μm\n"
        title += f"Mean κ: {fix_a['mean_curvature']:.4f} 1/μm"
        ax.set_title(title, fontsize=14, fontweight='bold', color='green')
        ax.legend()
    else:
        ax.set_title(f"✗ Fix A: FAILED\n{fix_a.get('error', 'Unknown')}",
                    fontsize=14, fontweight='bold', color='red')
    ax.axis('off')

    # Panel 3: Fix B
    ax = axes[2]
    ax.imshow(mask, cmap='gray', alpha=0.7)
    if fix_b.get('success'):
        ax.plot(fix_b['spline_x'], fix_b['spline_y'], 'r-', linewidth=2.5, label='Centerline')
        ax.plot(fix_b['spline_x'][0], fix_b['spline_y'][0], 'go', markersize=10, label='Head')
        ax.plot(fix_b['spline_x'][-1], fix_b['spline_y'][-1], 'bo', markersize=10, label='Tail')
        title = f"✓ Fix B: Alpha Shape (α={fix_b.get('alpha', '?')})\n"
        title += f"Length: {fix_b['length']:.1f} μm\n"
        title += f"Mean κ: {fix_b['mean_curvature']:.4f} 1/μm"
        ax.set_title(title, fontsize=14, fontweight='bold', color='green')
        ax.legend()
    else:
        ax.set_title(f"✗ Fix B: FAILED\n{fix_b.get('error', 'Unknown')}",
                    fontsize=14, fontweight='bold', color='red')
    ax.axis('off')

    plt.suptitle(f"Fix Comparison: {snip_id}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Test both fixes on failed embryos."""

    failed_embryos = [
        "20251017_combined_C04_e01_t0114",
        "20251017_combined_F11_e01_t0065",
    ]

    output_dir = Path(__file__).parent

    print("\n" + "="*80)
    print("TESTING FIXES FOR DISCONNECTED SKELETON PROBLEM")
    print("="*80)
    print("Fix A: Keep largest skeleton component")
    print("Fix B: Alpha shape preprocessing (test multiple alpha values)")
    print("="*80)

    all_results = []

    for snip_id in failed_embryos:
        results = test_embryo(snip_id)
        all_results.append(results)

        # Plot comparison
        safe_filename = snip_id.replace("/", "_").replace("\\", "_")
        viz_path = output_dir / f"{safe_filename}_fix_comparison.png"
        plot_comparison(results, viz_path)

    # Save summary CSV
    summary_data = []
    for res in all_results:
        row = {'snip_id': res['snip_id']}

        # Fix A
        if res['fix_a'].get('success'):
            row['fix_a_success'] = True
            row['fix_a_length_um'] = res['fix_a']['length']
            row['fix_a_mean_curvature'] = res['fix_a']['mean_curvature']
            row['fix_a_time_ms'] = res['fix_a']['time_ms']
        else:
            row['fix_a_success'] = False
            row['fix_a_error'] = res['fix_a'].get('error', 'Unknown')

        # Fix B
        if res['fix_b'].get('success'):
            row['fix_b_success'] = True
            row['fix_b_alpha'] = res['fix_b']['alpha']
            row['fix_b_length_um'] = res['fix_b']['length']
            row['fix_b_mean_curvature'] = res['fix_b']['mean_curvature']
            row['fix_b_time_ms'] = res['fix_b']['time_ms']
        else:
            row['fix_b_success'] = False
            row['fix_b_error'] = res['fix_b'].get('error', 'Unknown')

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    csv_path = output_dir / "fix_results.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved results: {csv_path}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for res in all_results:
        print(f"\n{res['snip_id']}:")
        print(f"  Fix A (Largest skeleton): {'✓ SUCCESS' if res['fix_a'].get('success') else '✗ FAILED'}")
        if res['fix_a'].get('success'):
            print(f"    Length: {res['fix_a']['length']:.1f} μm")
        print(f"  Fix B (Alpha shape):      {'✓ SUCCESS' if res['fix_b'].get('success') else '✗ FAILED'}")
        if res['fix_b'].get('success'):
            print(f"    Best α: {res['fix_b']['alpha']}, Length: {res['fix_b']['length']:.1f} μm")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
