"""
Compare different curvature measurement methods on embryo masks

Tests 4 methods:
1. Skeletonization (skimage)
2. Distance Transform + Ridge Detection
3. PCA-Based Slicing Approach (best for elongated embryos)
4. Contour-Based Method
"""

import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.interpolate import splprep, splev
from scipy.ndimage import distance_transform_edt, maximum_filter
from skimage import morphology, measure
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle


class CurvatureAnalyzer:
    """Analyzes embryo curvature using different centerline extraction methods."""

    def __init__(self, mask: np.ndarray, um_per_pixel: float = 1.0):
        """
        Initialize with a binary mask.

        Args:
            mask: Binary mask as numpy array
            um_per_pixel: Conversion factor from pixels to microns
        """
        # Ensure mask is C-contiguous and binary
        self.mask = np.ascontiguousarray(mask.astype(np.uint8))
        self.um_per_pixel = um_per_pixel

    # ==================== Method 1: Skeletonization ====================
    def method1_skeletonize(self):
        """Extract centerline using skeletonization."""
        skeleton = morphology.skeletonize(self.mask)
        y_coords, x_coords = np.where(skeleton)
        points = np.column_stack([x_coords, y_coords])
        ordered_points = self._order_points(points)
        return skeleton, ordered_points

    def _order_points(self, points: np.ndarray) -> np.ndarray:
        """Order skeleton points from one end to the other using greedy nearest-neighbor."""
        if len(points) == 0:
            return points

        # Start from leftmost point
        start_idx = np.argmin(points[:, 0])
        ordered = [points[start_idx]]
        remaining = list(range(len(points)))
        remaining.remove(start_idx)

        while remaining:
            current = ordered[-1]
            dists = np.sum((points[remaining] - current)**2, axis=1)
            nearest_idx = remaining[np.argmin(dists)]
            ordered.append(points[nearest_idx])
            remaining.remove(nearest_idx)

            # Stop if distance is too large (disconnected)
            if np.min(dists) > 100:
                break

        return np.array(ordered)

    # ==================== Method 2: Distance Transform + Ridge ====================
    def method2_distance_transform_ridge(self):
        """Extract centerline using distance transform and ridge detection."""
        distance_map = distance_transform_edt(self.mask)

        # Find local maxima (ridge) in distance transform
        local_max = maximum_filter(distance_map, size=3) == distance_map

        # Threshold to get significant ridge points
        threshold = np.percentile(distance_map[self.mask > 0], 70)
        ridge = local_max & (distance_map > threshold)

        # Extract and order points
        y_coords, x_coords = np.where(ridge)
        points = np.column_stack([x_coords, y_coords])
        ordered_points = self._order_points(points)

        return distance_map, ordered_points

    # ==================== Method 3: PCA-Based Slicing ====================
    def method3_pca_slicing(self, n_slices=100):
        """
        Extract centerline using PCA-based slicing.
        Most robust for elongated embryos.
        """
        # Get all mask coordinates
        y_coords, x_coords = np.where(self.mask)
        points = np.column_stack([x_coords, y_coords])

        # Perform PCA to find principal axis
        pca = PCA(n_components=2)
        pca.fit(points)

        principal_axis = pca.components_[0]
        center = points.mean(axis=0)
        centered_points = points - center
        projections = centered_points @ principal_axis

        # Create slices along the principal axis
        min_proj, max_proj = projections.min(), projections.max()
        slice_positions = np.linspace(min_proj, max_proj, n_slices)

        centerline_points = []
        for slice_pos in slice_positions:
            band_width = (max_proj - min_proj) / (n_slices * 2)
            in_slice = np.abs(projections - slice_pos) < band_width

            if np.sum(in_slice) > 0:
                slice_points = points[in_slice]
                centroid = slice_points.mean(axis=0)
                centerline_points.append(centroid)

        return principal_axis, np.array(centerline_points)

    # ==================== Method 4: Contour-Based ====================
    def method4_contour_based(self):
        """Extract centerline from contours by averaging edges."""
        contours = measure.find_contours(self.mask, 0.5)

        if len(contours) == 0:
            return [], np.array([])

        # Use the longest contour
        main_contour = max(contours, key=len)
        contour_points = main_contour[:, [1, 0]]  # flip to (x, y)

        # Sort by x-coordinate
        sorted_by_x = contour_points[np.argsort(contour_points[:, 0])]

        # Get x range for sampling
        x_min, x_max = sorted_by_x[:, 0].min(), sorted_by_x[:, 0].max()
        x_samples = np.linspace(x_min, x_max, 100)

        centerline_points = []
        for x in x_samples:
            near_x = sorted_by_x[np.abs(sorted_by_x[:, 0] - x) < (x_max - x_min) / 50]
            if len(near_x) >= 2:
                y_min = near_x[:, 1].min()
                y_max = near_x[:, 1].max()
                y_center = (y_min + y_max) / 2
                centerline_points.append([x, y_center])

        return contours, np.array(centerline_points)

    # ==================== Curvature Calculation ====================
    def compute_curvature(self, centerline_points: np.ndarray, smoothing=0.01):
        """
        Compute curvature along the centerline.

        Returns:
            arc_length: Arc length parameter (in pixels or microns)
            curvature: Curvature at each point (1/pixels or 1/microns)
        """
        if len(centerline_points) < 4:
            return np.array([]), np.array([])

        # Fit a parametric spline
        tck, u = splprep([centerline_points[:, 0], centerline_points[:, 1]],
                         s=smoothing * len(centerline_points), k=3)

        # Evaluate spline and derivatives
        u_fine = np.linspace(0, 1, 200)
        x_vals, y_vals = splev(u_fine, tck)
        dx, dy = splev(u_fine, tck, der=1)
        ddx, ddy = splev(u_fine, tck, der=2)

        # Compute curvature: κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**(3/2)

        # Compute arc length
        arc_length = np.cumsum(np.sqrt(np.diff(x_vals)**2 + np.diff(y_vals)**2))
        arc_length = np.concatenate([[0], arc_length])

        # Convert to microns if um_per_pixel is provided
        arc_length = arc_length * self.um_per_pixel
        curvature = curvature / self.um_per_pixel  # curvature has inverse units

        return arc_length, curvature


def compare_methods_on_masks():
    """Load masks and compare all 4 curvature methods."""

    # Load the CSV
    csv_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build06_output/df03_final_output_with_latents_20251020.csv")

    print(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Use first 4 masks
    n_test = 4

    # Create figure: 4 masks x 5 columns (mask + 4 methods)
    fig, axes = plt.subplots(n_test, 5, figsize=(25, 4*n_test))

    methods = [
        ("Skeletonize", lambda a: a.method1_skeletonize()),
        ("Distance Ridge", lambda a: a.method2_distance_transform_ridge()),
        ("PCA Slicing", lambda a: a.method3_pca_slicing()),
        ("Contour-Based", lambda a: a.method4_contour_based())
    ]

    for i in range(n_test):
        row = df.iloc[i]

        print(f"\n{'='*60}")
        print(f"Mask {i+1}: {row['embryo_id']}")
        print(f"{'='*60}")

        # Decode mask
        rle_data = {
            'counts': row['mask_rle'],
            'size': [int(row['mask_height_px']), int(row['mask_width_px'])]
        }
        mask = decode_mask_rle(rle_data)

        # Calculate um_per_pixel
        um_per_pixel = row['height_um'] / row['height_px']
        print(f"Scale: {um_per_pixel:.4f} μm/pixel")

        # Create analyzer
        analyzer = CurvatureAnalyzer(mask, um_per_pixel=um_per_pixel)

        # Column 0: Original mask
        axes[i, 0].imshow(mask, cmap='gray')
        axes[i, 0].set_title(f"{row['embryo_id']}\nFrame {row['frame_index']}")
        axes[i, 0].axis('off')

        # Test each method
        for j, (name, method) in enumerate(methods):
            print(f"\nMethod {j+1}: {name}")

            try:
                # Get centerline
                _, centerline = method(analyzer)

                if len(centerline) == 0:
                    print(f"  ✗ No centerline found")
                    axes[i, j+1].text(0.5, 0.5, "No centerline",
                                     ha='center', va='center',
                                     transform=axes[i, j+1].transAxes)
                    axes[i, j+1].axis('off')
                    continue

                print(f"  ✓ Centerline points: {len(centerline)}")

                # Compute curvature
                arc_length, curvature = analyzer.compute_curvature(centerline)

                if len(arc_length) == 0 or len(curvature) == 0:
                    print(f"  ✗ Could not compute curvature (too few points)")
                    axes[i, j+1].imshow(mask, cmap='gray', alpha=0.5)
                    axes[i, j+1].plot(centerline[:, 0], centerline[:, 1],
                                     'r-', linewidth=2)
                    axes[i, j+1].set_title(f"{name}\n(curvature failed)")
                    axes[i, j+1].axis('off')
                    continue

                if len(arc_length) > 0:
                    print(f"  ✓ Arc length: {arc_length[-1]:.2f} μm")
                    print(f"  ✓ Mean curvature: {np.mean(curvature):.6f} μm⁻¹")
                    print(f"  ✓ Max curvature: {np.max(curvature):.6f} μm⁻¹")

                # Plot mask + centerline
                axes[i, j+1].imshow(mask, cmap='gray', alpha=0.5)
                axes[i, j+1].plot(centerline[:, 0], centerline[:, 1],
                                 'r-', linewidth=2)
                axes[i, j+1].scatter(centerline[0, 0], centerline[0, 1],
                                    c='green', s=50, marker='o')
                axes[i, j+1].scatter(centerline[-1, 0], centerline[-1, 1],
                                    c='blue', s=50, marker='s')
                axes[i, j+1].set_title(f"{name}\nLength: {arc_length[-1]:.1f} μm")
                axes[i, j+1].axis('equal')
                axes[i, j+1].invert_yaxis()
                axes[i, j+1].axis('off')

            except Exception as e:
                print(f"  ✗ Error: {e}")
                axes[i, j+1].text(0.5, 0.5, f"Error:\n{str(e)[:50]}",
                                 ha='center', va='center',
                                 transform=axes[i, j+1].transAxes)
                axes[i, j+1].axis('off')

    plt.tight_layout()

    # Save figure
    output_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251022/curvature_methods_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n{'='*60}")
    print(f"✓ Saved comparison to: {output_path}")
    print(f"{'='*60}")

    plt.close()


if __name__ == "__main__":
    print("="*60)
    print("Comparing Curvature Measurement Methods")
    print("="*60)

    compare_methods_on_masks()

    print("\nDone!")
