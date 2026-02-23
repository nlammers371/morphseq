"""
Compare different smoothing methods for curvature analysis

Methods tested:
1. B-spline (least-squares, current method)
2. Gaussian smoothing
3. Savitzky-Golay filter
4. Moving average
5. LOWESS (locally weighted regression)

All methods calibrated to roughly equivalent smoothing strength (s=5.0 equivalent)
"""

import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle


class SmoothingComparator:
    """Compare different smoothing methods for curvature analysis using Geodesic Skeleton."""

    def __init__(self, mask: np.ndarray, um_per_pixel: float = 1.0):
        """
        Initialize with a binary mask.

        Args:
            mask: Binary mask as numpy array
            um_per_pixel: Conversion factor from pixels to microns
        """
        self.mask = np.ascontiguousarray(mask.astype(np.uint8))
        self.um_per_pixel = um_per_pixel

    def extract_centerline_geodesic(self, n_slices=100):
        """Extract centerline using Geodesic Skeleton method."""
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra
        from skimage import morphology

        # Step 1: Skeletonize
        skeleton = morphology.skeletonize(self.mask)
        y_skel, x_skel = np.where(skeleton)

        if len(y_skel) < 2:
            raise ValueError("Skeleton has too few points")

        skel_points = np.column_stack([x_skel, y_skel])

        # Step 2: Build graph (8-connected neighbors)
        n_points = len(skel_points)
        edges = []
        weights = []

        for i in range(n_points):
            for j in range(i+1, n_points):
                dist = np.sqrt(np.sum((skel_points[i] - skel_points[j])**2))
                if dist <= np.sqrt(2) + 0.1:  # 8-connected
                    edges.append((i, j))
                    weights.append(dist)

        if len(edges) == 0:
            raise ValueError("Skeleton graph has no edges")

        # Step 3: Create adjacency matrix
        rows = [e[0] for e in edges] + [e[1] for e in edges]
        cols = [e[1] for e in edges] + [e[0] for e in edges]
        data = weights + weights
        adj_matrix = csr_matrix((data, (rows, cols)), shape=(n_points, n_points))

        # Step 4: Find endpoints with maximum geodesic distance
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

        # Step 5: Trace path from start to end
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

        centerline = skel_points[path_indices]

        return centerline

    # ==================== Smoothing Method 1: B-spline ====================
    def smooth_bspline(self, centerline: np.ndarray, smoothing=5.0):
        """B-spline least-squares smoothing (current method)."""
        if len(centerline) < 4:
            return centerline

        tck, u = splprep([centerline[:, 0], centerline[:, 1]],
                         s=smoothing * len(centerline), k=3)
        u_fine = np.linspace(0, 1, 200)
        x_vals, y_vals = splev(u_fine, tck)
        return np.column_stack([x_vals, y_vals]), tck

    # ==================== Smoothing Method 2: Gaussian ====================
    def smooth_gaussian(self, centerline: np.ndarray, sigma=10):
        """Gaussian smoothing (convolution-based)."""
        if len(centerline) < 4:
            return centerline

        smoothed_x = gaussian_filter1d(centerline[:, 0], sigma=sigma, mode='nearest')
        smoothed_y = gaussian_filter1d(centerline[:, 1], sigma=sigma, mode='nearest')
        return np.column_stack([smoothed_x, smoothed_y])

    # ==================== Smoothing Method 3: Savitzky-Golay ====================
    def smooth_savgol(self, centerline: np.ndarray, window_length=21, polyorder=3):
        """Savitzky-Golay filter (local polynomial fitting)."""
        if len(centerline) < window_length:
            window_length = len(centerline) if len(centerline) % 2 == 1 else len(centerline) - 1
            if window_length < polyorder + 2:
                return centerline

        smoothed_x = savgol_filter(centerline[:, 0], window_length=window_length,
                                   polyorder=polyorder, mode='nearest')
        smoothed_y = savgol_filter(centerline[:, 1], window_length=window_length,
                                   polyorder=polyorder, mode='nearest')
        return np.column_stack([smoothed_x, smoothed_y])

    # ==================== Smoothing Method 4: Moving Average ====================
    def smooth_moving_average(self, centerline: np.ndarray, window_size=15):
        """Moving average (uniform filter)."""
        if len(centerline) < 4:
            return centerline

        smoothed_x = uniform_filter1d(centerline[:, 0], size=window_size, mode='nearest')
        smoothed_y = uniform_filter1d(centerline[:, 1], size=window_size, mode='nearest')
        return np.column_stack([smoothed_x, smoothed_y])

    # ==================== Smoothing Method 5: LOWESS ====================
    def smooth_lowess(self, centerline: np.ndarray, frac=0.2):
        """LOWESS (locally weighted scatterplot smoothing)."""
        if len(centerline) < 4:
            return centerline

        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess

            # Use index as x-axis for LOWESS
            t = np.arange(len(centerline))

            smoothed_x = lowess(centerline[:, 0], t, frac=frac, return_sorted=False)
            smoothed_y = lowess(centerline[:, 1], t, frac=frac, return_sorted=False)

            return np.column_stack([smoothed_x, smoothed_y])
        except ImportError:
            print("  Warning: statsmodels not available, skipping LOWESS")
            return centerline

    # ==================== Curvature Calculation ====================
    def compute_curvature_from_points(self, smoothed_points: np.ndarray):
        """
        Compute curvature from smoothed points using finite differences.

        For methods other than B-spline, we compute derivatives numerically.
        """
        if len(smoothed_points) < 3:
            return np.array([]), np.array([])

        # Compute first derivatives (central differences)
        dx = np.gradient(smoothed_points[:, 0])
        dy = np.gradient(smoothed_points[:, 1])

        # Compute second derivatives
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        # Compute curvature: κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**(3/2)

        # Compute arc length
        arc_length = np.cumsum(np.sqrt(np.diff(smoothed_points[:, 0])**2 +
                                       np.diff(smoothed_points[:, 1])**2))
        arc_length = np.concatenate([[0], arc_length])

        # Convert to microns
        arc_length = arc_length * self.um_per_pixel
        curvature = curvature / self.um_per_pixel

        return arc_length, curvature

    def compute_curvature_bspline(self, tck):
        """Compute curvature from B-spline using analytical derivatives."""
        u_fine = np.linspace(0, 1, 200)
        x_vals, y_vals = splev(u_fine, tck)
        dx, dy = splev(u_fine, tck, der=1)
        ddx, ddy = splev(u_fine, tck, der=2)

        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**(3/2)

        arc_length = np.cumsum(np.sqrt(np.diff(x_vals)**2 + np.diff(y_vals)**2))
        arc_length = np.concatenate([[0], arc_length])

        arc_length = arc_length * self.um_per_pixel
        curvature = curvature / self.um_per_pixel

        return arc_length, curvature


def load_mask_from_json(json_path: Path, snip_id: str):
    """Load mask from grounded SAM JSON file."""
    import json

    print(f"Loading mask for {snip_id}...")

    with open(json_path, 'r') as f:
        data = json.load(f)

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
                            print(f"  Mask shape: {mask.shape}, area: {mask.sum()} px")
                            return mask, embryo_data

    raise ValueError(f"Could not find mask for {snip_id}")


def compare_smoothing_methods():
    """Compare all smoothing methods on test embryos using Geodesic Skeleton."""

    # Load from grounded SAM JSON files
    test_cases = [
        (Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/sam2_pipeline_files/segmentation/grounded_sam_segmentations_20251017_part2.json"),
         "20251017_part2_D06_e01_t0022",
         "Good baseline (D06)"),
        (Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/sam2_pipeline_files/segmentation/grounded_sam_segmentations_20250512.json"),
         "20250512_E06_e01_t0086",
         "Challenging (extreme curve)"),
        (Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/sam2_pipeline_files/segmentation/grounded_sam_segmentations_20250512.json"),
         "20250512_E06_e01_t0181",
         "Very challenging (self-overlap)")
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

    n_embryos = len(test_embryos)
    print(f"\nAnalyzing {n_embryos} embryos")

    # Smoothing methods with different aggressiveness levels
    methods = [
        ("B-spline (s=0.0)", lambda comp, cl: comp.smooth_bspline(cl, smoothing=0.0), True),
        ("B-spline (s=0.01)", lambda comp, cl: comp.smooth_bspline(cl, smoothing=0.01), True),
        ("B-spline (s=0.1)", lambda comp, cl: comp.smooth_bspline(cl, smoothing=0.1), True),
        ("B-spline (s=0.5)", lambda comp, cl: comp.smooth_bspline(cl, smoothing=0.5), True),
        ("B-spline (s=1.0)", lambda comp, cl: comp.smooth_bspline(cl, smoothing=1.0), True),
        ("B-spline (s=5.0)", lambda comp, cl: comp.smooth_bspline(cl, smoothing=5.0), True),
        ("Gaussian (σ=0)", lambda comp, cl: cl, False),
        ("Gaussian (σ=5)", lambda comp, cl: comp.smooth_gaussian(cl, sigma=5), False),
        ("Gaussian (σ=10)", lambda comp, cl: comp.smooth_gaussian(cl, sigma=10), False),
        ("Gaussian (σ=20)", lambda comp, cl: comp.smooth_gaussian(cl, sigma=20), False),
        ("Gaussian (σ=30)", lambda comp, cl: comp.smooth_gaussian(cl, sigma=30), False),
        ("Gaussian (σ=50)", lambda comp, cl: comp.smooth_gaussian(cl, sigma=50), False),
    ]

    # Create figure: n_embryos rows x len(methods) columns
    fig, axes = plt.subplots(n_embryos, len(methods), figsize=(30, 4*n_embryos))
    if n_embryos == 1:
        axes = axes.reshape(1, -1)

    for i, (mask, embryo_data, description) in enumerate(test_embryos):
        print(f"\n{'='*60}")
        print(f"Embryo {i+1}: {description}")
        print(f"  ID: {embryo_data['embryo_id']}")
        print(f"{'='*60}")

        # Use pixel units
        um_per_pixel = 1.0
        print(f"Units: pixels")

        # Create comparator
        comp = SmoothingComparator(mask, um_per_pixel=um_per_pixel)

        # Extract centerline once using GEODESIC method
        centerline = comp.extract_centerline_geodesic(n_slices=100)
        print(f"Centerline points: {len(centerline)}")

        # Test each smoothing method
        for j, (name, method, is_bspline) in enumerate(methods):
            print(f"\n{name}")

            try:
                if is_bspline:
                    smoothed, tck = method(comp, centerline)
                    arc_length, curvature = comp.compute_curvature_bspline(tck)
                else:
                    smoothed = method(comp, centerline)
                    arc_length, curvature = comp.compute_curvature_from_points(smoothed)

                if len(arc_length) > 0:
                    print(f"  ✓ Arc length: {arc_length[-1]:.2f} μm")
                    print(f"  ✓ Mean curvature: {np.mean(curvature):.6f} μm⁻¹")
                    print(f"  ✓ Max curvature: {np.max(curvature):.6f} μm⁻¹")
                    print(f"  ✓ Std curvature: {np.std(curvature):.6f} μm⁻¹")

                # Plot mask + centerline + smoothed curve
                axes[i, j].imshow(mask, cmap='gray', alpha=0.3)

                # Original centerline points in light color
                axes[i, j].scatter(centerline[:, 0], centerline[:, 1],
                                  c='lightblue', s=5, alpha=0.3, label='Raw')

                # Smoothed curve in red
                axes[i, j].plot(smoothed[:, 0], smoothed[:, 1],
                               'r-', linewidth=2, label='Smoothed')

                axes[i, j].set_title(f"{name}\n" +
                                    f"Mean κ: {np.mean(curvature):.4f} μm⁻¹\n" +
                                    f"Length: {arc_length[-1]:.1f} μm")
                axes[i, j].axis('equal')
                axes[i, j].invert_yaxis()
                axes[i, j].axis('off')

            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback
                traceback.print_exc()
                axes[i, j].text(0.5, 0.5, f"Error:\n{str(e)[:50]}",
                               ha='center', va='center',
                               transform=axes[i, j].transAxes, fontsize=8)
                axes[i, j].axis('off')

    plt.tight_layout()

    # Save figure
    output_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251022/geodesic_smoothing_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n{'='*60}")
    print(f"✓ Saved comparison to: {output_path}")
    print(f"{'='*60}")

    # Also create curvature plots
    fig2, axes2 = plt.subplots(n_embryos, len(methods), figsize=(30, 3*n_embryos))
    if n_embryos == 1:
        axes2 = axes2.reshape(1, -1)

    for i, (mask, embryo_data, description) in enumerate(test_embryos):
        um_per_pixel = 1.0

        comp = SmoothingComparator(mask, um_per_pixel=um_per_pixel)
        centerline = comp.extract_centerline_geodesic(n_slices=100)

        for j, (name, method, is_bspline) in enumerate(methods):
            try:
                if is_bspline:
                    smoothed, tck = method(comp, centerline)
                    arc_length, curvature = comp.compute_curvature_bspline(tck)
                else:
                    smoothed = method(comp, centerline)
                    arc_length, curvature = comp.compute_curvature_from_points(smoothed)

                if len(arc_length) > 0:
                    axes2[i, j].plot(arc_length, curvature, 'b-', linewidth=2)
                    axes2[i, j].set_xlabel('Arc Length (μm)', fontsize=8)
                    axes2[i, j].set_ylabel('Curvature (μm⁻¹)', fontsize=8)
                    axes2[i, j].set_title(f"{name}", fontsize=10)
                    axes2[i, j].grid(True, alpha=0.3)

            except:
                pass

    plt.tight_layout()
    output_path2 = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251022/geodesic_curvature_comparison.png")
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"✓ Saved curvature plots to: {output_path2}")

    plt.close('all')


if __name__ == "__main__":
    print("="*60)
    print("Comparing Smoothing Methods for Curvature Analysis")
    print("="*60)

    compare_smoothing_methods()

    print("\nDone!")
