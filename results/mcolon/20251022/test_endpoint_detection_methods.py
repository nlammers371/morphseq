"""
Test different endpoint detection methods for curved embryos

Problem: When embryos are highly curved (head near tail), simple PCA fails to find
         the correct start/end points along the embryo's actual length.

Solution: Test methods that use geodesic distance (path along embryo) instead of
          Euclidean distance (straight-line).

Test cases:
- Good: 20251017_part2_D06_ch00_t0022 (baseline that works)
- Challenging: frame_index 86 (extreme curvature)
- Very challenging: frame_index 181 (head overlaps tail)
"""

import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage import morphology, measure
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle


class EndpointDetector:
    """Test different methods to find embryo start/end points."""

    def __init__(self, mask: np.ndarray, um_per_pixel: float = 1.0):
        self.mask = np.ascontiguousarray(mask.astype(np.uint8))
        self.um_per_pixel = um_per_pixel

    # ==================== Method 1: PCA (Current Baseline) ====================
    def method1_pca_endpoints(self, n_slices=100):
        """Current PCA method - projects onto principal axis."""
        y_coords, x_coords = np.where(self.mask)
        points = np.column_stack([x_coords, y_coords])

        pca = PCA(n_components=2)
        pca.fit(points)

        principal_axis = pca.components_[0]
        center = points.mean(axis=0)
        projections = (points - center) @ principal_axis

        # Extract centerline via slicing
        min_proj, max_proj = projections.min(), projections.max()
        slice_positions = np.linspace(min_proj, max_proj, n_slices)

        centerline_points = []
        for slice_pos in slice_positions:
            band_width = (max_proj - min_proj) / (n_slices * 2)
            in_slice = np.abs(projections - slice_pos) < band_width
            if np.sum(in_slice) > 0:
                centerline_points.append(points[in_slice].mean(axis=0))

        centerline = np.array(centerline_points)
        if len(centerline) > 0:
            endpoints = np.array([centerline[0], centerline[-1]])
        else:
            endpoints = np.array([[0, 0], [0, 0]])

        return centerline, endpoints, "PCA Principal Axis"

    # ==================== Method 2: Geodesic Distance via Skeleton ====================
    def method2_geodesic_skeleton(self, n_slices=100):
        """Find endpoints using geodesic distance along skeleton."""
        # Get skeleton
        skeleton = morphology.skeletonize(self.mask)
        y_skel, x_skel = np.where(skeleton)

        if len(y_skel) < 2:
            return np.array([]), np.array([[0, 0], [0, 0]]), "Geodesic (failed)", None

        skel_points = np.column_stack([x_skel, y_skel])

        # Build adjacency graph (8-connected)
        n_points = len(skel_points)
        edges = []
        weights = []

        for i in range(n_points):
            for j in range(i+1, n_points):
                dist = np.sqrt(np.sum((skel_points[i] - skel_points[j])**2))
                if dist <= np.sqrt(2) + 0.1:  # 8-connected (diagonal = sqrt(2))
                    edges.append((i, j))
                    weights.append(dist)

        if len(edges) == 0:
            return np.array([]), np.array([[0, 0], [0, 0]]), "Geodesic (no edges)", None

        # Create sparse adjacency matrix
        rows = [e[0] for e in edges] + [e[1] for e in edges]
        cols = [e[1] for e in edges] + [e[0] for e in edges]
        data = weights + weights
        adj_matrix = csr_matrix((data, (rows, cols)), shape=(n_points, n_points))

        # Find two points with maximum geodesic distance
        # Sample subset to avoid computing full distance matrix
        sample_size = min(50, n_points)
        sample_indices = np.random.choice(n_points, size=sample_size, replace=False)

        max_dist = 0
        best_pair = (0, n_points-1)

        for idx in sample_indices:
            distances = dijkstra(adj_matrix, indices=idx, directed=False)
            furthest = np.argmax(distances[np.isfinite(distances)])
            if distances[furthest] > max_dist:
                max_dist = distances[furthest]
                best_pair = (idx, furthest)

        start_idx, end_idx = best_pair
        endpoints = np.array([skel_points[start_idx], skel_points[end_idx]])

        # Trace path from start to end using Dijkstra
        distances, predecessors = dijkstra(adj_matrix, indices=start_idx,
                                          directed=False, return_predecessors=True)

        # Backtrack from end to start
        path_indices = []
        current = end_idx
        while current != -9999 and current != start_idx:
            path_indices.append(current)
            current = predecessors[current]
            if len(path_indices) > n_points:  # Prevent infinite loop
                break
        path_indices.append(start_idx)
        path_indices = path_indices[::-1]

        if len(path_indices) > 1:
            centerline = skel_points[path_indices]
        else:
            # Fallback to PCA slicing
            centerline, _, _ = self.method1_pca_endpoints(n_slices)

        return centerline, endpoints, "Geodesic Skeleton", skeleton

    # ==================== Method 3: Distance Transform Extrema ====================
    def method3_distance_transform_extrema(self, n_slices=100):
        """Find endpoints using distance transform extrema at opposite ends."""
        # Compute distance transform
        dist_transform = distance_transform_edt(self.mask)

        # Find ridge (local maxima)
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(dist_transform, size=3) == dist_transform
        threshold = np.percentile(dist_transform[self.mask > 0], 70)
        ridge = local_max & (dist_transform > threshold)

        y_ridge, x_ridge = np.where(ridge)
        if len(y_ridge) < 2:
            return self.method1_pca_endpoints(n_slices)

        ridge_points = np.column_stack([x_ridge, y_ridge])

        # Find two points with max Euclidean distance
        # (these should be at opposite ends)
        max_dist = 0
        best_pair = (0, 1)

        for i in range(min(50, len(ridge_points))):
            for j in range(i+1, len(ridge_points)):
                dist = np.sqrt(np.sum((ridge_points[i] - ridge_points[j])**2))
                if dist > max_dist:
                    max_dist = dist
                    best_pair = (i, j)

        endpoints = np.array([ridge_points[best_pair[0]], ridge_points[best_pair[1]]])

        # Use PCA slicing but orient based on found endpoints
        centerline, _, _ = self.method1_pca_endpoints(n_slices)

        # Flip if needed to match endpoint orientation
        if len(centerline) > 0:
            dist_start_0 = np.sum((centerline[0] - endpoints[0])**2)
            dist_start_1 = np.sum((centerline[0] - endpoints[1])**2)
            if dist_start_1 < dist_start_0:
                centerline = centerline[::-1]

        return centerline, endpoints, "Distance Transform"

    # ==================== Method 4: Multiple Random Starts ====================
    def method4_random_starts_voting(self, n_slices=100, n_trials=10):
        """Try multiple random start points, keep most consistent."""
        y_coords, x_coords = np.where(self.mask)
        points = np.column_stack([x_coords, y_coords])

        # Get skeleton for sampling
        skeleton = morphology.skeletonize(self.mask)
        y_skel, x_skel = np.where(skeleton)

        if len(y_skel) < n_slices:
            return self.method1_pca_endpoints(n_slices)

        skel_points = np.column_stack([x_skel, y_skel])

        # Try multiple random starts
        centerlines = []
        for _ in range(n_trials):
            # Random start point on skeleton
            start_idx = np.random.randint(0, len(skel_points))
            start_point = skel_points[start_idx]

            # Find furthest point from start (along Euclidean distance)
            distances = np.sqrt(np.sum((skel_points - start_point)**2, axis=1))
            end_idx = np.argmax(distances)

            # Use PCA slicing
            centerline, _, _ = self.method1_pca_endpoints(n_slices)
            centerlines.append(centerline)

        # Average all centerlines (simple voting)
        if len(centerlines) > 0 and len(centerlines[0]) > 0:
            avg_centerline = np.mean(centerlines, axis=0)
            endpoints = np.array([avg_centerline[0], avg_centerline[-1]])
        else:
            avg_centerline = np.array([])
            endpoints = np.array([[0, 0], [0, 0]])

        return avg_centerline, endpoints, "Random Starts Voting"

    # ==================== Smoothing & Curvature ====================
    def smooth_gaussian(self, centerline: np.ndarray, sigma=10):
        """Apply Gaussian smoothing to centerline."""
        if len(centerline) < 4:
            return centerline

        smoothed_x = gaussian_filter1d(centerline[:, 0], sigma=sigma, mode='nearest')
        smoothed_y = gaussian_filter1d(centerline[:, 1], sigma=sigma, mode='nearest')
        return np.column_stack([smoothed_x, smoothed_y])

    def compute_curvature(self, smoothed_points: np.ndarray):
        """Compute curvature from smoothed points."""
        if len(smoothed_points) < 3:
            return np.array([]), np.array([])

        dx = np.gradient(smoothed_points[:, 0])
        dy = np.gradient(smoothed_points[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**(3/2)

        arc_length = np.cumsum(np.sqrt(np.diff(smoothed_points[:, 0])**2 +
                                       np.diff(smoothed_points[:, 1])**2))
        arc_length = np.concatenate([[0], arc_length])

        arc_length = arc_length * self.um_per_pixel
        curvature = curvature / self.um_per_pixel

        return arc_length, curvature


def load_mask_from_json(json_path: Path, snip_id: str):
    """Load and decode a specific mask from the segmentation JSON file."""
    print(f"\nLoading mask for {snip_id}...")

    import json
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Navigate through JSON structure
    mask_found = False

    if 'experiments' in data:
        for exp_id, exp_data in data['experiments'].items():
            if 'videos' not in exp_data:
                continue

            for video_id, video_data in exp_data['videos'].items():
                if 'image_ids' not in video_data:
                    continue

                for image_id, image_data in video_data['image_ids'].items():
                    if 'embryos' not in image_data:
                        continue

                    for embryo_id, embryo_data in image_data['embryos'].items():
                        if embryo_data.get('snip_id') == snip_id:
                            print(f"  ✓ Found: {exp_id}/{video_id}/{image_id}/{embryo_id}")
                            seg_data = embryo_data['segmentation']
                            mask = decode_mask_rle(seg_data)
                            print(f"  Mask shape: {mask.shape}")
                            print(f"  Mask area: {mask.sum()} pixels")
                            mask_found = True
                            return mask, embryo_data

    if not mask_found:
        raise ValueError(f"Could not find mask for {snip_id}")


def test_endpoint_methods():
    """Test all endpoint detection methods on 3 embryos."""

    # Load from grounded SAM JSON files
    json_path_part2 = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/sam2_pipeline_files/segmentation/grounded_sam_segmentations_20251017_part2.json")
    json_path_e06 = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/sam2_pipeline_files/segmentation/grounded_sam_segmentations_20250512.json")

    # Find test embryos by snip_id
    test_cases = [
        (json_path_part2, "20251017_part2_D06_e01_t0022", "Good baseline (D06)"),
        (json_path_e06, "20250512_E06_e01_t0086", "Challenging (extreme curve)"),
        (json_path_e06, "20250512_E06_e01_t0181", "Very challenging (self-overlap)")
    ]

    test_embryos = []
    for json_path, snip_id, description in test_cases:
        try:
            if not json_path.exists():
                print(f"✗ JSON file not found: {json_path}")
                continue

            mask, embryo_data = load_mask_from_json(json_path, snip_id)
            test_embryos.append((mask, embryo_data, description))
            print(f"✓ Loaded {description}")
        except Exception as e:
            print(f"✗ Could not load {description}: {e}")

    if len(test_embryos) == 0:
        print("No test embryos found!")
        return

    # Methods to test
    methods = [
        ("PCA", lambda d: (*d.method1_pca_endpoints(), None), False),
        ("Geodesic", lambda d: d.method2_geodesic_skeleton(), True),
        ("Dist Transform", lambda d: (*d.method3_distance_transform_extrema(), None), False),
        ("Random Voting", lambda d: (*d.method4_random_starts_voting(), None), False)
    ]

    # Create figure: add extra row for skeleton visualization
    n_rows = len(test_embryos) * 2  # Double rows: one for methods, one for skeleton detail
    fig, axes = plt.subplots(n_rows, len(methods), figsize=(24, 6*len(test_embryos)))
    if len(test_embryos) == 1:
        axes = axes.reshape(1, -1)

    for i, (mask, embryo_data, description) in enumerate(test_embryos):
        print(f"\n{'='*60}")
        print(f"Embryo {i+1}: {description}")
        print(f"  ID: {embryo_data['embryo_id']}")
        print(f"  Snip ID: {embryo_data['snip_id']}")
        print(f"{'='*60}")

        # Use pixel units (no um_per_pixel available in this data)
        um_per_pixel = 1.0  # Just use pixels for now
        print(f"Units: pixels")

        detector = EndpointDetector(mask, um_per_pixel=um_per_pixel)

        for j, (method_name, method_func, has_skeleton) in enumerate(methods):
            print(f"\n{method_name}:")

            try:
                result = method_func(detector)
                if len(result) == 4:
                    centerline, endpoints, full_name, skeleton = result
                else:
                    centerline, endpoints, full_name = result
                    skeleton = None

                # Top row: method result
                ax_main = axes[i*2, j] if len(test_embryos) > 1 else axes[0, j]
                ax_detail = axes[i*2+1, j] if len(test_embryos) > 1 else axes[1, j]

                if len(centerline) == 0:
                    print(f"  ✗ No centerline found")
                    ax_main.text(0.5, 0.5, "Failed", ha='center', va='center',
                                transform=ax_main.transAxes)
                    ax_main.axis('off')
                    ax_detail.axis('off')
                    continue

                print(f"  ✓ Centerline points: {len(centerline)}")
                print(f"  ✓ Endpoints: {endpoints[0]} → {endpoints[1]}")

                # Apply Gaussian smoothing
                smoothed = detector.smooth_gaussian(centerline, sigma=10)
                arc_length, curvature = detector.compute_curvature(smoothed)

                if len(arc_length) > 0:
                    print(f"  ✓ Arc length: {arc_length[-1]:.2f} μm")
                    print(f"  ✓ Mean curvature: {np.mean(curvature):.6f} μm⁻¹")
                    print(f"  ✓ Max curvature: {np.max(curvature):.6f} μm⁻¹")

                # Plot main result
                ax_main.imshow(mask, cmap='gray', alpha=0.4)
                ax_main.scatter(centerline[:, 0], centerline[:, 1],
                                  c='lightblue', s=3, alpha=0.3)
                ax_main.plot(smoothed[:, 0], smoothed[:, 1],
                               'r-', linewidth=2, label='Smoothed')

                # Mark endpoints
                ax_main.scatter(endpoints[0, 0], endpoints[0, 1],
                                  c='green', s=150, marker='o', edgecolors='white',
                                  linewidths=2, label='Start', zorder=5)
                ax_main.scatter(endpoints[1, 0], endpoints[1, 1],
                                  c='blue', s=150, marker='s', edgecolors='white',
                                  linewidths=2, label='End', zorder=5)

                ax_main.set_title(f"{full_name}\n" +
                                    f"Length: {arc_length[-1]:.1f} px\n" +
                                    f"Mean κ: {np.mean(curvature):.4f} px⁻¹")
                ax_main.axis('equal')
                ax_main.invert_yaxis()
                ax_main.legend(loc='upper right', fontsize=8)
                ax_main.axis('off')

                # Bottom row: skeleton detail (only for geodesic method)
                if skeleton is not None and has_skeleton:
                    # Show skeleton quality
                    ax_detail.imshow(mask, cmap='gray', alpha=0.3)
                    ax_detail.imshow(skeleton, cmap='Reds', alpha=0.7)
                    ax_detail.plot(centerline[:, 0], centerline[:, 1],
                                  'b-', linewidth=3, label='Traced path', alpha=0.8)
                    ax_detail.scatter(endpoints[0, 0], endpoints[0, 1],
                                     c='green', s=200, marker='o', edgecolors='black',
                                     linewidths=3, label='Start', zorder=5)
                    ax_detail.scatter(endpoints[1, 0], endpoints[1, 1],
                                     c='blue', s=200, marker='s', edgecolors='black',
                                     linewidths=3, label='End', zorder=5)
                    ax_detail.set_title('Skeleton Quality\n(Red=skeleton, Blue=path)')
                    ax_detail.axis('equal')
                    ax_detail.invert_yaxis()
                    ax_detail.legend(loc='upper right', fontsize=8)
                    ax_detail.axis('off')
                else:
                    # No skeleton detail for other methods
                    ax_detail.axis('off')

            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback
                traceback.print_exc()
                axes[i, j].text(0.5, 0.5, f"Error:\n{str(e)[:50]}",
                               ha='center', va='center',
                               transform=axes[i, j].transAxes, fontsize=8)
                axes[i, j].axis('off')

    plt.tight_layout()
    output_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251022/endpoint_methods_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n{'='*60}")
    print(f"✓ Saved to: {output_path}")
    print(f"{'='*60}")

    plt.close()


if __name__ == "__main__":
    print("="*60)
    print("Testing Endpoint Detection Methods")
    print("="*60)

    test_endpoint_methods()

    print("\nDone!")
