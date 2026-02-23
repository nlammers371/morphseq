"""
Geodesic Centerline Extraction Method

Extract centerline using geodesic distance along skeleton, then apply B-spline smoothing
for robust curvature measurement.

This is the primary method for centerline extraction as it handles highly curved embryos
(including cases where head is near tail).
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
from scipy.interpolate import splprep, splev
from skimage import morphology


class GeodesicCenterlineAnalyzer:
    """
    Analyze embryo curvature using:
    1. Geodesic skeleton for centerline extraction (handles highly curved embryos)
    2. B-spline fitting with s=5.0 for smoothing

    Advantages over PCA method:
    - Handles highly curved embryos (curvature too extreme for PCA)
    - Robust to complex shapes where head is near tail
    - Uses actual skeleton topology rather than projections

    Trade-offs:
    - ~2.8x slower than PCA method (when using fast=False)
    - Can occasionally extend through fins (mitigated by good mask cleaning)
    """

    def __init__(self, mask: np.ndarray, um_per_pixel: float = 1.0,
                 bspline_smoothing: float = 5.0, random_seed: int = 42,
                 fast: bool = True):
        """
        Initialize analyzer with a binary mask.

        Args:
            mask: Binary mask as numpy array
            um_per_pixel: Conversion factor from pixels to microns (default=1.0 for pixel units)
            bspline_smoothing: B-spline smoothing parameter (default=5.0)
                              This is multiplied by len(centerline) for scipy's splprep
            random_seed: Seed for reproducible endpoint detection via np.random.choice
                        (default=42 for deterministic results)
            fast: If True, use optimized O(N) graph building (default=True)
                  If False, use original O(N²) distance-based method
        """
        self.mask = np.ascontiguousarray(mask.astype(np.uint8))
        self.um_per_pixel = um_per_pixel
        self.bspline_smoothing = bspline_smoothing
        self.random_seed = random_seed
        self.fast = fast

    def _build_graph_fast(self, skel_points: np.ndarray):
        """
        Build graph using efficient 8-connected neighbor lookup (O(N)).
        
        This is the optimized version that only checks immediate neighbors
        instead of computing all pairwise distances.
        
        Args:
            skel_points: (N, 2) array of skeleton points as (x, y)
            
        Returns:
            adj_matrix: Sparse adjacency matrix
        """
        n_points = len(skel_points)
        point_to_index = {tuple(pt): idx for idx, pt in enumerate(skel_points)}

        rows: list = []
        cols: list = []
        weights: list = []

        # 8-connected neighborhood offsets
        neighbour_offsets = (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        )

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
            raise ValueError("Skeleton graph has no edges (disconnected)")

        # Create symmetric adjacency matrix
        data = weights + weights
        adj_matrix = csr_matrix(
            (data, (rows + cols, cols + rows)),
            shape=(n_points, n_points),
        )
        
        return adj_matrix

    def _build_graph_slow(self, skel_points: np.ndarray):
        """
        Build graph using O(N²) pairwise distance computation.
        
        This is the original method kept for backward compatibility.
        It computes distances between all pairs of points and connects
        those within a threshold distance.
        
        Args:
            skel_points: (N, 2) array of skeleton points as (x, y)
            
        Returns:
            adj_matrix: Sparse adjacency matrix
        """
        n_points = len(skel_points)
        rows: list = []
        cols: list = []
        weights: list = []
        
        # Compute all pairwise distances (O(N²))
        threshold = 1.5  # pixels - connect points within this distance
        
        for i in range(n_points):
            for j in range(i + 1, n_points):
                dist = np.linalg.norm(skel_points[i] - skel_points[j])
                if dist <= threshold:
                    rows.append(i)
                    cols.append(j)
                    weights.append(dist)
        
        if not rows:
            raise ValueError("Skeleton graph has no edges (disconnected)")
        
        # Create symmetric adjacency matrix
        data = weights + weights
        adj_matrix = csr_matrix(
            (data, (rows + cols, cols + rows)),
            shape=(n_points, n_points),
        )
        
        return adj_matrix

    def extract_centerline(self):
        """
        Extract centerline using geodesic distance along skeleton.

        Process:
        1. Skeletonize mask
        2. Build 8-connected graph of skeleton pixels (fast) or pairwise distances (slow)
        3. Find endpoints via maximum geodesic distance (sample-based search)
        4. Trace geodesic path from start to end using Dijkstra's algorithm

        Returns:
            centerline: (N, 2) array of (x, y) coordinates along skeleton path
            endpoints: (2, 2) array of start/end points
            skeleton: Binary skeleton image

        Raises:
            ValueError: If skeleton is too small or disconnected
        """
        # Step 1: Skeletonize
        skeleton = morphology.skeletonize(self.mask)
        y_skel, x_skel = np.where(skeleton)

        if len(y_skel) < 2:
            raise ValueError("Skeleton has too few points")

        skel_points = np.column_stack([x_skel, y_skel])

        # Step 2: Build graph (choose method based on fast flag)
        if self.fast:
            adj_matrix = self._build_graph_fast(skel_points)
        else:
            adj_matrix = self._build_graph_slow(skel_points)

        n_points = len(skel_points)

        # Step 2.5: Clean disconnected components - keep only largest
        n_components, component_labels = connected_components(adj_matrix, directed=False)
        
        if n_components > 1:
            # Find largest connected component
            unique_labels, counts = np.unique(component_labels, return_counts=True)
            largest_label = unique_labels[np.argmax(counts)]
            valid_mask = component_labels == largest_label
            valid_indices = np.where(valid_mask)[0]
            
            # Filter skel_points and rebuild adjacency matrix for largest component
            skel_points = skel_points[valid_indices]
            
            # Map old indices to new indices
            index_map = np.full(n_points, -1, dtype=int)
            index_map[valid_indices] = np.arange(len(valid_indices))
            
            # Rebuild adjacency matrix
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

        # Step 3: Find endpoints with maximum geodesic distance (VECTORIZED)
        # Instead of sampling, compute all pairwise distances efficiently
        # For each point, find the one farthest from it
        max_dist_overall = 0
        best_pair = (0, min(1, n_points - 1))

        # Use sampling for large skeletons (>100 points) to avoid expensive computation
        if n_points > 100:
            sample_size = min(100, n_points)
            rng = np.random.RandomState(self.random_seed)
            sample_indices = rng.choice(n_points, size=sample_size, replace=False)
        else:
            sample_indices = np.arange(n_points)

        # Vectorized computation: for each sampled point, find distances to all others
        all_max_distances = np.zeros(len(sample_indices))
        all_furthest_indices = np.zeros(len(sample_indices), dtype=int)

        for i, idx in enumerate(sample_indices):
            distances = dijkstra(adj_matrix, indices=idx, directed=False)
            finite_mask = np.isfinite(distances)

            if np.any(finite_mask):
                furthest_idx = np.argmax(distances)
                max_dist = distances[furthest_idx]
                all_max_distances[i] = max_dist
                all_furthest_indices[i] = furthest_idx

                if max_dist > max_dist_overall:
                    max_dist_overall = max_dist
                    best_pair = (idx, furthest_idx)

        start_idx, end_idx = best_pair
        endpoints = np.array([skel_points[start_idx], skel_points[end_idx]])

        # Step 4: Trace geodesic path from start to end
        distances, predecessors = dijkstra(adj_matrix, indices=start_idx,
                                          directed=False, return_predecessors=True)

        path_indices = []
        current = end_idx
        while current != -9999 and current != start_idx:
            path_indices.append(current)
            current = predecessors[current]
            if len(path_indices) > n_points:  # Safety check for infinite loops
                break
        path_indices.append(start_idx)
        path_indices = path_indices[::-1]

        centerline = skel_points[path_indices]

        return centerline, endpoints, skeleton

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
                - centerline_raw: (N, 2) array of raw skeleton centerline
                - centerline_smoothed: (200, 2) array of B-spline smoothed centerline
                - endpoints: (2, 2) array of start/end points
                - skeleton: Binary skeleton image
                - arc_length: Arc length parameter (200,) array
                - curvature: Curvature values (200,) array
                - bspline_tck: B-spline representation for further analysis
                - stats: Dictionary of summary statistics
        """
        # Extract centerline
        centerline, endpoints, skeleton = self.extract_centerline()

        # Smooth with B-spline
        smoothed_points, tck = self.smooth_with_bspline(centerline)

        # Compute curvature from B-spline
        arc_length, curvature = self.compute_curvature(tck)

        results = {
            'centerline_raw': centerline,
            'centerline_smoothed': smoothed_points,
            'endpoints': endpoints,
            'skeleton': skeleton,
            'arc_length': arc_length,
            'curvature': curvature,
            'bspline_tck': tck,
            'stats': {
                'total_length': arc_length[-1] if len(arc_length) > 0 else 0,
                'mean_curvature': np.mean(curvature) if len(curvature) > 0 else 0,
                'std_curvature': np.std(curvature) if len(curvature) > 0 else 0,
                'max_curvature': np.max(curvature) if len(curvature) > 0 else 0,
                'n_centerline_points': len(centerline),
                'n_skeleton_points': np.sum(skeleton),
                'bspline_smoothing': self.bspline_smoothing,
                'method': 'geodesic',
                'fast_mode': self.fast
            }
        }

        return results
