"""
Embryo Curvature Analysis
-------------------------
Methods for extracting centerlines and measuring curvature from embryo masks.

This script implements 4 different methods:
1. Skeletonization (skimage)
2. Distance Transform + Ridge Detection
3. PCA-Based Slicing Approach
4. Contour-Based Method
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage
from scipy.interpolate import UnivariateSpline, splprep, splev
from scipy.ndimage import distance_transform_edt
from skimage import io, morphology, measure
from sklearn.decomposition import PCA
import cv2
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


class EmbryoCurvatureAnalyzer:
    """
    Analyzes embryo curvature using different centerline extraction methods.
    """

    def __init__(self, mask_path: Path):
        """
        Initialize with a mask file path.

        Args:
            mask_path: Path to the embryo mask PNG file
        """
        self.mask_path = Path(mask_path)
        self.mask = self._load_mask()

    def _load_mask(self) -> np.ndarray:
        """Load and decode the mask from PNG file."""
        mask = io.imread(self.mask_path)
        # If it's a multi-channel image, take the first channel
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        # Convert to binary (assuming labeled mask where embryo > 0)
        binary_mask = (mask > 0).astype(np.uint8)
        return binary_mask

    # ==================== Method 1: Skeletonization ====================
    def method1_skeletonize(self, visualize=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract centerline using skeletonization.

        Returns:
            skeleton: Binary skeleton image
            ordered_points: Ordered (x, y) coordinates along skeleton
        """
        # Skeletonize the mask
        skeleton = morphology.skeletonize(self.mask)

        # Extract coordinates
        y_coords, x_coords = np.where(skeleton)

        # Order the points along the skeleton (simple approach)
        points = np.column_stack([x_coords, y_coords])
        ordered_points = self._order_skeleton_points(points)

        if visualize:
            self._visualize_result(
                self.mask,
                ordered_points,
                title="Method 1: Skeletonization",
                overlay=skeleton
            )

        return skeleton, ordered_points

    def _order_skeleton_points(self, points: np.ndarray) -> np.ndarray:
        """
        Order skeleton points from one end to the other.
        Uses a greedy nearest-neighbor approach.
        """
        if len(points) == 0:
            return points

        # Start from a point at one extreme (e.g., leftmost)
        start_idx = np.argmin(points[:, 0])
        ordered = [points[start_idx]]
        remaining = list(range(len(points)))
        remaining.remove(start_idx)

        while remaining:
            current = ordered[-1]
            # Find nearest remaining point
            dists = np.sum((points[remaining] - current)**2, axis=1)
            nearest_idx = remaining[np.argmin(dists)]
            ordered.append(points[nearest_idx])
            remaining.remove(nearest_idx)

            # Stop if distance is too large (disconnected component)
            if np.min(dists) > 100:  # threshold
                break

        return np.array(ordered)

    # ==================== Method 2: Distance Transform + Ridge ====================
    def method2_distance_transform_ridge(self, visualize=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract centerline using distance transform and ridge detection.

        Returns:
            distance_map: Distance transform of the mask
            centerline_points: Ordered (x, y) coordinates along centerline
        """
        # Compute distance transform
        distance_map = distance_transform_edt(self.mask)

        # Find local maxima (ridge) in distance transform
        # Use morphological operations to find the ridge
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(distance_map, size=3) == distance_map

        # Threshold to get significant ridge points
        threshold = np.percentile(distance_map[self.mask > 0], 70)
        ridge = local_max & (distance_map > threshold)

        # Extract and order points
        y_coords, x_coords = np.where(ridge)
        points = np.column_stack([x_coords, y_coords])
        ordered_points = self._order_skeleton_points(points)

        if visualize:
            self._visualize_result(
                self.mask,
                ordered_points,
                title="Method 2: Distance Transform Ridge",
                overlay=distance_map
            )

        return distance_map, ordered_points

    # ==================== Method 3: PCA-Based Slicing ====================
    def method3_pca_slicing(self, n_slices=100, visualize=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract centerline using PCA-based slicing approach.
        Most robust for elongated embryos.

        Args:
            n_slices: Number of slices perpendicular to principal axis

        Returns:
            principal_axis: Principal axis vector
            centerline_points: Ordered (x, y) coordinates of centroids
        """
        # Get all mask coordinates
        y_coords, x_coords = np.where(self.mask)
        points = np.column_stack([x_coords, y_coords])

        # Perform PCA to find principal axis
        pca = PCA(n_components=2)
        pca.fit(points)

        # Principal axis (first component)
        principal_axis = pca.components_[0]

        # Project points onto principal axis
        center = points.mean(axis=0)
        centered_points = points - center
        projections = centered_points @ principal_axis

        # Create slices along the principal axis
        min_proj, max_proj = projections.min(), projections.max()
        slice_positions = np.linspace(min_proj, max_proj, n_slices)

        centerline_points = []
        for slice_pos in slice_positions:
            # Find points in this slice (within a band)
            band_width = (max_proj - min_proj) / (n_slices * 2)
            in_slice = np.abs(projections - slice_pos) < band_width

            if np.sum(in_slice) > 0:
                # Compute centroid of points in this slice
                slice_points = points[in_slice]
                centroid = slice_points.mean(axis=0)
                centerline_points.append(centroid)

        centerline_points = np.array(centerline_points)

        if visualize:
            self._visualize_result(
                self.mask,
                centerline_points,
                title="Method 3: PCA-Based Slicing",
                show_axis=True,
                axis_center=center,
                axis_direction=principal_axis
            )

        return principal_axis, centerline_points

    # ==================== Method 4: Contour-Based ====================
    def method4_contour_based(self, visualize=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract centerline from contours by averaging top and bottom edges.

        Returns:
            contours: List of contour arrays
            centerline_points: Ordered (x, y) coordinates along centerline
        """
        # Find contours
        contours = measure.find_contours(self.mask, 0.5)

        if len(contours) == 0:
            return [], np.array([])

        # Use the longest contour
        main_contour = max(contours, key=len)

        # Convert to (x, y) format
        contour_points = main_contour[:, [1, 0]]  # flip to (x, y)

        # Sort by x-coordinate to split into top and bottom edges
        sorted_by_x = contour_points[np.argsort(contour_points[:, 0])]

        # Split into roughly equal halves (top and bottom)
        mid_idx = len(sorted_by_x) // 2

        # Get x range for sampling
        x_min, x_max = sorted_by_x[:, 0].min(), sorted_by_x[:, 0].max()
        x_samples = np.linspace(x_min, x_max, 100)

        centerline_points = []
        for x in x_samples:
            # Find points near this x coordinate
            near_x = sorted_by_x[np.abs(sorted_by_x[:, 0] - x) < (x_max - x_min) / 50]
            if len(near_x) >= 2:
                # Get min and max y at this x (top and bottom edges)
                y_min = near_x[:, 1].min()
                y_max = near_x[:, 1].max()
                y_center = (y_min + y_max) / 2
                centerline_points.append([x, y_center])

        centerline_points = np.array(centerline_points)

        if visualize:
            self._visualize_result(
                self.mask,
                centerline_points,
                title="Method 4: Contour-Based",
                show_contour=True,
                contour=contour_points
            )

        return contours, centerline_points

    # ==================== Curvature Calculation ====================
    def compute_curvature(self, centerline_points: np.ndarray,
                         smoothing=0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute curvature along the centerline.

        Args:
            centerline_points: (N, 2) array of (x, y) coordinates
            smoothing: Smoothing parameter for spline fitting

        Returns:
            arc_length: Arc length parameter
            curvature: Curvature at each point
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

        return arc_length, curvature

    # ==================== Visualization ====================
    def _visualize_result(self, mask, centerline_points, title="",
                         overlay=None, show_axis=False, axis_center=None,
                         axis_direction=None, show_contour=False, contour=None):
        """Visualize the mask with centerline overlay."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: mask with overlay
        if overlay is not None and len(overlay.shape) == 2:
            axes[0].imshow(overlay, cmap='hot', alpha=0.7)
        axes[0].imshow(mask, cmap='gray', alpha=0.5)

        if show_contour and contour is not None:
            axes[0].plot(contour[:, 0], contour[:, 1], 'b-',
                        linewidth=1, alpha=0.5, label='Contour')

        if len(centerline_points) > 0:
            axes[0].plot(centerline_points[:, 0], centerline_points[:, 1],
                        'r-', linewidth=2, label='Centerline')
            axes[0].scatter(centerline_points[0, 0], centerline_points[0, 1],
                          c='green', s=100, marker='o', label='Start')
            axes[0].scatter(centerline_points[-1, 0], centerline_points[-1, 1],
                          c='blue', s=100, marker='s', label='End')

        if show_axis and axis_center is not None and axis_direction is not None:
            # Show principal axis
            scale = 200
            axes[0].arrow(axis_center[0], axis_center[1],
                         axis_direction[0] * scale, axis_direction[1] * scale,
                         head_width=20, head_length=30, fc='cyan', ec='cyan',
                         alpha=0.7, label='Principal Axis')

        axes[0].set_title(f'{title}\nMask with Centerline')
        axes[0].legend()
        axes[0].axis('equal')
        axes[0].invert_yaxis()

        # Right: curvature plot
        if len(centerline_points) > 3:
            arc_length, curvature = self.compute_curvature(centerline_points)
            if len(arc_length) > 0:
                axes[1].plot(arc_length, curvature, 'b-', linewidth=2)
                axes[1].set_xlabel('Arc Length (pixels)')
                axes[1].set_ylabel('Curvature (1/pixels)')
                axes[1].set_title('Curvature along Centerline')
                axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def compare_all_methods(self, save_path=None):
        """
        Compare all 4 methods side by side.
        """
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        methods = [
            ("Method 1: Skeletonize", self.method1_skeletonize),
            ("Method 2: Distance Ridge", self.method2_distance_transform_ridge),
            ("Method 3: PCA Slicing", self.method3_pca_slicing),
            ("Method 4: Contour-Based", self.method4_contour_based)
        ]

        for i, (name, method) in enumerate(methods):
            # Get centerline
            _, centerline = method(visualize=False)

            # Top row: mask with centerline
            axes[0, i].imshow(self.mask, cmap='gray', alpha=0.7)
            if len(centerline) > 0:
                axes[0, i].plot(centerline[:, 0], centerline[:, 1],
                              'r-', linewidth=2)
                axes[0, i].scatter(centerline[0, 0], centerline[0, 1],
                                 c='green', s=50, marker='o')
            axes[0, i].set_title(name)
            axes[0, i].axis('equal')
            axes[0, i].invert_yaxis()

            # Bottom row: curvature
            if len(centerline) > 3:
                arc_length, curvature = self.compute_curvature(centerline)
                if len(arc_length) > 0:
                    axes[1, i].plot(arc_length, curvature, 'b-', linewidth=2)
            axes[1, i].set_xlabel('Arc Length (px)')
            axes[1, i].set_ylabel('Curvature (1/px)')
            axes[1, i].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved comparison to {save_path}")

        return fig


# ==================== Example Usage ====================
if __name__ == "__main__":
    # Example: Load a mask and analyze it

    # Path to SAM2 metadata CSV
    metadata_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/sam2_pipeline_files/sam2_expr_files/sam2_metadata_20250529_30hpf_ctrl_atf6.csv")

    # Load metadata
    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} embryo masks")

    # Get the first embryo mask as an example
    first_row = df.iloc[0]
    mask_filename = first_row['exported_mask_path']
    experiment_date = first_row['experiment_id'].split('_')[0]

    # Construct full path to mask
    mask_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/sam2_pipeline_files/exported_masks") / experiment_date / "masks" / mask_filename

    print(f"\nAnalyzing mask: {mask_filename}")
    print(f"Embryo ID: {first_row['embryo_id']}")
    print(f"Mask path: {mask_path}")

    # Create analyzer
    analyzer = EmbryoCurvatureAnalyzer(mask_path)

    # Test each method individually
    print("\n" + "="*60)
    print("Running Method 1: Skeletonization")
    print("="*60)
    skeleton, points1 = analyzer.method1_skeletonize(visualize=True)

    print("\n" + "="*60)
    print("Running Method 2: Distance Transform Ridge")
    print("="*60)
    dist_map, points2 = analyzer.method2_distance_transform_ridge(visualize=True)

    print("\n" + "="*60)
    print("Running Method 3: PCA-Based Slicing")
    print("="*60)
    axis, points3 = analyzer.method3_pca_slicing(visualize=True)

    print("\n" + "="*60)
    print("Running Method 4: Contour-Based")
    print("="*60)
    contours, points4 = analyzer.method4_contour_based(visualize=True)

    # Compare all methods
    print("\n" + "="*60)
    print("Comparing All Methods")
    print("="*60)
    output_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251022")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = analyzer.compare_all_methods(
        save_path=output_dir / f"curvature_comparison_{first_row['embryo_id']}.png"
    )

    print(f"\nResults saved to {output_dir}")
    print("\nDone!")
