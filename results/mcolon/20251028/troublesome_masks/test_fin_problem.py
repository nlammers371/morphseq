"""
Investigate the fin problem on A02 embryo

The issue: centerline goes down the fin instead of following the body axis
Solution: Test different preprocessing methods, especially alpha shapes with varying aggressiveness

Alpha shapes work because they:
1. Preserve the actual body length better than Gaussian blur
2. Use convex hull + erosion to remove protrusions (like fins)
3. Are more aggressive/deterministic than blur-based methods
"""

import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter
from scipy.interpolate import splprep, splev
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
from skimage import morphology, measure
from skimage.draw import polygon

from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle
from segmentation_sandbox.scripts.utils.mask_cleaning import clean_embryo_mask


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


def apply_gaussian_preprocessing(mask: np.ndarray, sigma: float = 10.0, threshold: float = 0.7):
    """Standard Gaussian blur preprocessing."""
    blurred = gaussian_filter(mask.astype(float), sigma=sigma)
    core_mask = blurred > threshold
    refined = mask & core_mask
    refined = morphology.binary_dilation(refined, morphology.disk(5))
    
    # Keep largest component
    labeled = measure.label(refined)
    if labeled.max() > 0:
        largest = np.argmax([np.sum(labeled == i) for i in range(1, labeled.max() + 1)]) + 1
        refined = (labeled == largest)
    
    return refined


def apply_alpha_shape_preprocessing(mask: np.ndarray, alpha: float = 50) -> np.ndarray:
    """
    Alpha-shape preprocessing to remove fins/protrusions.
    
    More aggressive alpha values (larger) create smaller, more aggressive erosions.
    """
    cleaned = mask.copy()
    
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
        # Higher alpha = more aggressive erosion
        erode_radius = max(3, int(alpha / 10))
        eroded_hull = morphology.binary_erosion(hull_mask, morphology.disk(erode_radius))
        
        # Intersect with original
        refined = cleaned & eroded_hull
        
        # If erosion removed too much, use less erosion
        if refined.sum() < 0.5 * cleaned.sum():
            erode_radius = max(2, int(alpha / 20))
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


def extract_centerline_from_mask(preprocessed_mask: np.ndarray, um_per_pixel: float):
    """Extract centerline using geodesic method (same logic as fixed GeodesicCenterlineAnalyzer)."""
    from scipy.ndimage import distance_transform_edt
    
    # Skeletonize
    skeleton = morphology.skeletonize(preprocessed_mask)
    y_skel, x_skel = np.where(skeleton)
    
    if len(y_skel) < 2:
        return None, None, None, "Skeleton too small"
    
    skel_points = np.column_stack([x_skel, y_skel])
    
    # Build 8-connected graph
    n_points = len(skel_points)
    point_to_index = {tuple(pt): idx for idx, pt in enumerate(skel_points)}
    
    rows = []
    cols = []
    weights = []
    
    neighbour_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    
    for idx, (x, y) in enumerate(skel_points):
        for dx, dy in neighbour_offsets:
            neighbour = (x + dx, y + dy)
            jdx = point_to_index.get(neighbour)
            if jdx is None or jdx <= idx:
                continue
            rows.append(idx)
            cols.append(jdx)
            weights.append(np.sqrt(dx*dx + dy*dy))
    
    if not rows:
        return None, None, None, "No edges in graph"
    
    # Create adjacency matrix
    data = weights + weights
    adj_matrix = csr_matrix(
        (data, (rows + cols, cols + rows)),
        shape=(n_points, n_points),
    )
    
    # Clean disconnected components
    n_components, component_labels = connected_components(adj_matrix, directed=False)
    
    if n_components > 1:
        unique_labels, counts = np.unique(component_labels, return_counts=True)
        largest_label = unique_labels[np.argmax(counts)]
        valid_mask = component_labels == largest_label
        valid_indices = np.where(valid_mask)[0]
        
        skel_points = skel_points[valid_indices]
        index_map = np.full(n_points, -1, dtype=int)
        index_map[valid_indices] = np.arange(len(valid_indices))
        
        cx, cy = adj_matrix.nonzero()
        valid_edges_mask = valid_mask[cx] & valid_mask[cy]
        cx_new = index_map[cx[valid_edges_mask]]
        cy_new = index_map[cy[valid_edges_mask]]
        data_new = adj_matrix.data[valid_edges_mask]
        
        adj_matrix = csr_matrix(
            (data_new, (cx_new, cy_new)),
            shape=(len(valid_indices), len(valid_indices))
        )
        n_points = len(skel_points)
    
    # Find endpoints
    max_dist_overall = 0
    best_pair = (0, min(1, n_points - 1))
    
    if n_points > 100:
        sample_size = min(100, n_points)
        rng = np.random.RandomState(42)
        sample_indices = rng.choice(n_points, size=sample_size, replace=False)
    else:
        sample_indices = np.arange(n_points)
    
    for idx in sample_indices:
        distances = dijkstra(adj_matrix, indices=idx, directed=False)
        finite_mask = np.isfinite(distances)
        
        if np.any(finite_mask):
            furthest_idx = np.argmax(distances)
            max_dist = distances[furthest_idx]
            
            if max_dist > max_dist_overall:
                max_dist_overall = max_dist
                best_pair = (idx, furthest_idx)
    
    start_idx, end_idx = best_pair
    endpoints = np.array([skel_points[start_idx], skel_points[end_idx]])
    
    # Trace path
    distances, predecessors = dijkstra(adj_matrix, indices=start_idx,
                                       directed=False, return_predecessors=True)
    
    path_indices = []
    current = end_idx
    while current != -9999 and current != start_idx:
        path_indices.append(current)
        current = predecessors[current]
        if len(path_indices) > n_points:
            break
    path_indices.append(start_idx)
    path_indices = path_indices[::-1]
    
    if len(path_indices) < 4:
        return None, endpoints, skeleton, f"Path too short ({len(path_indices)} points)"
    
    centerline_raw = skel_points[path_indices]
    
    # B-spline smoothing
    try:
        tck, u = splprep([centerline_raw[:, 0], centerline_raw[:, 1]],
                         s=5.0 * len(centerline_raw), k=3)
        u_fine = np.linspace(0, 1, 200)
        x_vals, y_vals = splev(u_fine, tck)
        centerline_smoothed = np.column_stack([x_vals, y_vals])
        
        # Compute arc length
        arc_length = np.cumsum(np.sqrt(np.diff(x_vals)**2 + np.diff(y_vals)**2))
        arc_length = np.concatenate([[0], arc_length])
        arc_length = arc_length * um_per_pixel
        total_length = arc_length[-1]
        
        return centerline_smoothed, endpoints, skeleton, None, total_length
    except Exception as e:
        return None, endpoints, skeleton, f"Spline fitting failed: {e}"


def test_embryo_methods(snip_id: str, output_dir: Path):
    """Test different preprocessing methods on the embryo."""
    print(f"\n{'='*80}")
    print(f"TESTING: {snip_id}")
    print(f"{'='*80}\n")
    
    # Load data
    mask, um_per_pixel = load_embryo_data(snip_id)
    mask_cleaned, _ = clean_embryo_mask(mask, verbose=False)
    
    print(f"Original mask: {mask.shape}, area={mask.sum():,} px")
    print(f"Cleaned mask: area={mask_cleaned.sum():,} px\n")
    
    # Test different methods
    methods = [
        ("Gaussian blur (σ=10)", lambda m: apply_gaussian_preprocessing(m, sigma=10.0, threshold=0.7)),
        ("Gaussian blur (σ=15)", lambda m: apply_gaussian_preprocessing(m, sigma=15.0, threshold=0.7)),
        ("Alpha shape (α=30)", lambda m: apply_alpha_shape_preprocessing(m, alpha=30)),
        ("Alpha shape (α=50)", lambda m: apply_alpha_shape_preprocessing(m, alpha=50)),
        ("Alpha shape (α=70)", lambda m: apply_alpha_shape_preprocessing(m, alpha=70)),
        ("Alpha shape (α=90)", lambda m: apply_alpha_shape_preprocessing(m, alpha=90)),
    ]
    
    results = []
    
    for method_name, preprocess_func in methods:
        print(f"Testing: {method_name}...")
        try:
            preprocessed = preprocess_func(mask_cleaned)
            print(f"  Preprocessed area: {preprocessed.sum():,} px ({preprocessed.sum()/mask_cleaned.sum()*100:.1f}%)")
            
            centerline, endpoints, skeleton, error, total_length = extract_centerline_from_mask(preprocessed, um_per_pixel)
            
            if error:
                print(f"  ✗ Error: {error}\n")
                results.append({
                    'method': method_name,
                    'success': False,
                    'error': error,
                    'preprocessed_area': preprocessed.sum()
                })
            else:
                print(f"  ✓ Success: {total_length:.2f} μm\n")
                results.append({
                    'method': method_name,
                    'success': True,
                    'centerline': centerline,
                    'endpoints': endpoints,
                    'skeleton': skeleton,
                    'total_length': total_length,
                    'preprocessed_mask': preprocessed,
                    'original_mask': mask_cleaned,
                    'error': None
                })
        except Exception as e:
            print(f"  ✗ Exception: {e}\n")
            results.append({
                'method': method_name,
                'success': False,
                'error': str(e),
                'preprocessed_area': 0
            })
    
    # Create comparison visualization
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        n_methods = len(successful_results)
        fig, axes = plt.subplots(2, n_methods, figsize=(5*n_methods, 10))
        
        if n_methods == 1:
            axes = axes.reshape(2, 1)
        
        for col, result in enumerate(successful_results):
            # Original mask with centerline
            ax = axes[0, col]
            ax.imshow(result['original_mask'], cmap='gray', alpha=0.6)
            ax.plot(result['centerline'][:, 0], result['centerline'][:, 1], 'r-', linewidth=2)
            ax.plot(result['endpoints'][:, 0], result['endpoints'][:, 1], 'g*', markersize=15)
            ax.set_title(f"{result['method']}\nLength: {result['total_length']:.2f} μm", fontsize=11, fontweight='bold')
            ax.set_xlabel('X (px)')
            ax.set_ylabel('Y (px)')
            
            # Preprocessed mask with skeleton
            ax = axes[1, col]
            ax.imshow(result['preprocessed_mask'], cmap='gray', alpha=0.6)
            ax.contour(result['skeleton'], colors='blue', linewidths=1, levels=[0.5])
            ax.plot(result['endpoints'][:, 0], result['endpoints'][:, 1], 'r*', markersize=15)
            ax.set_title("Preprocessed + Skeleton", fontsize=11)
            ax.set_xlabel('X (px)')
            ax.set_ylabel('Y (px)')
        
        plt.tight_layout()
        output_path = output_dir / f"{snip_id}_method_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved comparison: {output_path}\n")
        plt.close()
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for result in results:
        status = "✓" if result['success'] else "✗"
        print(f"{status} {result['method']}")
        if result['success']:
            print(f"  Length: {result['total_length']:.2f} μm")
        else:
            print(f"  Error: {result['error']}")
    
    return results


if __name__ == "__main__":
    output_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251028/troublesome_masks")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    snip_id = "20251017_combined_A02_e01_t0064"
    test_embryo_methods(snip_id, output_dir)
