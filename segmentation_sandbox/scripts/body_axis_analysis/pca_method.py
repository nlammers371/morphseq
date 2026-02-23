"""
PCA Centerline Extraction Method

Extract centerline using PCA-based slicing along principal axis, then apply B-spline smoothing.

This is a faster alternative to geodesic method (~2.8x faster) but has limitations
with highly curved embryos.

Use when:
- Speed is important and embryos are not extremely curved
- Extent > 0.35, Solidity > 0.6, Eccentricity < 0.98

Avoid when:
- Embryos are highly curved (head near tail)
- Curvature is too extreme for perpendicular slicing approach
"""

import numpy as np
from scipy.interpolate import splprep, splev
from sklearn.decomposition import PCA


class PCACenterlineAnalyzer:
    """
    PCA-based curvature analysis with adjustable smoothing.

    Advantages:
    - ~2.8x faster than Geodesic method (5.2s vs 14.4s per embryo)
    - Works well for straight/moderately curved embryos
    - Simpler implementation

    Limitations:
    - Fails on highly curved embryos (curvature too extreme for perpendicular slicing)
    - ~2.5% disagreement rate with Geodesic method on challenging cases
    - Cannot handle cases where principal axis assumption breaks down

    Based on comparison analysis of 1000 embryos, PCA disagrees with Geodesic when:
    - Extent < 0.35 (highly curved, occupies small bbox fraction)
    - Solidity < 0.6 (non-convex shape)
    - Eccentricity > 0.98 (very elongated)
    """

    def __init__(self, mask: np.ndarray, um_per_pixel: float = 1.0, bspline_smoothing: float = 5.0):
        """
        Initialize with a binary mask.

        Args:
            mask: Binary mask as numpy array
            um_per_pixel: Conversion factor from pixels to microns (default=1.0 for pixel units)
            bspline_smoothing: B-spline smoothing parameter (default=5.0)
                              This is multiplied by len(centerline) for scipy's splprep
        """
        self.mask = np.ascontiguousarray(mask.astype(np.uint8))
        self.um_per_pixel = um_per_pixel
        self.bspline_smoothing = bspline_smoothing

    def extract_centerline(self, n_slices=100):
        """
        Extract centerline using PCA-based slicing.

        Process:
        1. Find principal axis via PCA on all mask pixels
        2. Create perpendicular slices along principal axis
        3. Compute centroid of each slice
        4. Return centerline as sequence of centroids

        Args:
            n_slices: Number of slices to take along principal axis (default=100)

        Returns:
            centerline: (N, 2) array of (x, y) coordinates

        Note:
            Most robust for elongated embryos where PCA captures body axis well.
            May fail for highly curved embryos where perpendicular slicing breaks down.
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

        # Project points onto principal axis
        projections = centered_points @ principal_axis

        # Create slices along the principal axis
        min_proj, max_proj = projections.min(), projections.max()
        slice_positions = np.linspace(min_proj, max_proj, n_slices)

        centerline_points = []
        for slice_pos in slice_positions:
            # Define band width (overlap helps with continuity)
            band_width = (max_proj - min_proj) / (n_slices * 2)
            in_slice = np.abs(projections - slice_pos) < band_width

            if np.sum(in_slice) > 0:
                slice_points = points[in_slice]
                centroid = slice_points.mean(axis=0)
                centerline_points.append(centroid)

        return np.array(centerline_points)

    def smooth_with_bspline(self, centerline: np.ndarray):
        """
        Apply B-spline smoothing to centerline.

        Args:
            centerline: (N, 2) array of (x, y) coordinates

        Returns:
            smoothed_points: (N_fine, 2) array of smoothed coordinates (200 points)
            tck: B-spline representation (t, c, k) for derivative computation

        Note:
            Returns original centerline and None if too few points (<4) for spline fitting
        """
        if len(centerline) < 4:
            return centerline, None

        # Fit B-spline with smoothing parameter
        # s = bspline_smoothing * len(centerline) provides adaptive smoothing
        tck, u = splprep([centerline[:, 0], centerline[:, 1]],
                         s=self.bspline_smoothing * len(centerline), k=3)

        # Evaluate spline at fine resolution (200 points)
        u_fine = np.linspace(0, 1, 200)
        x_vals, y_vals = splev(u_fine, tck)

        smoothed_points = np.column_stack([x_vals, y_vals])

        return smoothed_points, tck

    def compute_curvature(self, tck):
        """
        Compute curvature from B-spline using analytical derivatives.

        Curvature formula: κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)

        Args:
            tck: B-spline representation (t, c, k) from splprep

        Returns:
            arc_length: Arc length parameter (in um or pixels)
            curvature: Curvature at each point (in 1/um or 1/pixels)

        Note:
            Returns empty arrays if tck is None
        """
        if tck is None:
            return np.array([]), np.array([])

        # Evaluate spline and derivatives at fine resolution
        u_fine = np.linspace(0, 1, 200)
        x_vals, y_vals = splev(u_fine, tck)
        dx, dy = splev(u_fine, tck, der=1)  # First derivatives
        ddx, ddy = splev(u_fine, tck, der=2)  # Second derivatives

        # Curvature: κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**(3/2)

        # Compute arc length by integrating speed
        arc_length = np.cumsum(np.sqrt(np.diff(x_vals)**2 + np.diff(y_vals)**2))
        arc_length = np.concatenate([[0], arc_length])

        # Convert to physical units
        arc_length = arc_length * self.um_per_pixel
        curvature = curvature / self.um_per_pixel

        return arc_length, curvature

    def analyze(self):
        """
        Full analysis pipeline: extract centerline, smooth with B-spline, compute curvature.

        Returns:
            results: Dictionary with all analysis results:
                - centerline_raw: (N, 2) array of raw PCA centerline
                - centerline_smoothed: (200, 2) array of B-spline smoothed centerline
                - arc_length: Arc length parameter (200,) array
                - curvature: Curvature values (200,) array
                - bspline_tck: B-spline representation for further analysis
                - stats: Dictionary of summary statistics
        """
        # Extract centerline
        centerline = self.extract_centerline(n_slices=100)

        # Smooth with B-spline
        smoothed_points, tck = self.smooth_with_bspline(centerline)

        # Compute curvature from B-spline
        arc_length, curvature = self.compute_curvature(tck)

        results = {
            'centerline_raw': centerline,
            'centerline_smoothed': smoothed_points,
            'arc_length': arc_length,
            'curvature': curvature,
            'bspline_tck': tck,
            'stats': {
                'total_length': arc_length[-1] if len(arc_length) > 0 else 0,
                'mean_curvature': np.mean(curvature) if len(curvature) > 0 else 0,
                'std_curvature': np.std(curvature) if len(curvature) > 0 else 0,
                'max_curvature': np.max(curvature) if len(curvature) > 0 else 0,
                'n_centerline_points': len(centerline),
                'bspline_smoothing': self.bspline_smoothing,
                'method': 'pca'
            }
        }

        return results
