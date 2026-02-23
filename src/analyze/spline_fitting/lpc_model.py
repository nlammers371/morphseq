"""Local Principal Curve (LPC) implementation.

This module provides the core LocalPrincipalCurve algorithm for fitting smooth curves
through point clouds in n-dimensional space. The implementation is fully self-contained
with no dependencies on other spline_fitting modules.

The LPC algorithm iteratively fits a curve by:
1. Computing local principal components at each point
2. Following the principal direction in both forward and backward directions
3. Fitting cubic splines through the resulting path
4. Resampling points at equal arc-length intervals

Example:
    >>> from src.analyze.spline_fitting import LocalPrincipalCurve
    >>> lpc = LocalPrincipalCurve(bandwidth=0.5)
    >>> lpc.fit(data_points, start_points=anchor_point)
    >>> fitted_curve = lpc.cubic_splines[0]
"""

import numpy as np
from scipy.interpolate import CubicSpline, interp1d


class LocalPrincipalCurve:
    """Fit smooth curves through point clouds using local principal components.

    The LocalPrincipalCurve algorithm fits a smooth curve through a point cloud by
    iteratively following the direction of maximum variance (principal component)
    in local neighborhoods. The final curve is represented as a cubic spline with
    points sampled at equal arc-length intervals.

    Parameters
    ----------
    bandwidth : float, default=0.5
        Bandwidth for Gaussian kernel weights. Controls the size of local neighborhoods.
    h : float, optional
        Step size for curve progression. Defaults to bandwidth if not specified.
    max_iter : int, default=100
        Maximum number of iterations for forward/backward curve progression.
    tol : float, default=1e-4
        Convergence tolerance. Stops iteration when center of mass movement < tol.
    angle_penalty_exp : int, default=2
        Exponent for angle penalty. Higher values enforce smoother curves by
        penalizing sharp direction changes.

    Attributes
    ----------
    initializations : list of ndarray
        Starting points used for each fitted curve.
    paths : list of ndarray
        Raw paths traced by the algorithm before spline fitting.
    cubic_splines_eq : list of dict
        Cubic spline interpolators (one per dimension) in parameter space.
    cubic_splines : list of ndarray
        Final fitted curves with equal arc-length spacing, shape (num_points, n_dims).

    Examples
    --------
    Fit a curve through 3D points:

    >>> lpc = LocalPrincipalCurve(bandwidth=0.5)
    >>> lpc.fit(points_3d, start_points=start_point, num_points=500)
    >>> fitted_curve = lpc.cubic_splines[0]

    Fit with both start and end anchors:

    >>> lpc.fit(points_3d, start_points=start_point, end_point=end_point)
    """

    def __init__(self, bandwidth=0.5, max_iter=100, tol=1e-4, angle_penalty_exp=2, h=None):
        """Initialize the Local Principal Curve solver."""
        self.bandwidth = bandwidth
        self.h = h if h is not None else self.bandwidth
        self.max_iter = max_iter
        self.tol = tol
        self.angle_penalty_exp = angle_penalty_exp

        self.initializations = []
        self.paths = []
        self.cubic_splines_eq = []
        self.cubic_splines = []

    def _kernel_weights(self, dataset, x):
        """Compute Gaussian kernel weights for points relative to x."""
        dists = np.linalg.norm(dataset - x, axis=1)
        weights = np.exp(- (dists**2) / (2 * self.bandwidth**2))
        w = weights / np.sum(weights)
        return w

    def _local_center_of_mass(self, dataset, x):
        """Compute weighted center of mass in local neighborhood of x."""
        w = self._kernel_weights(dataset, x)
        mu = np.sum(dataset.T * w, axis=1)
        return mu

    def _local_covariance(self, dataset, x, mu):
        """Compute weighted covariance matrix in local neighborhood."""
        w = self._kernel_weights(dataset, x)
        centered = dataset - mu
        weighted_centered = centered * w[:, np.newaxis]
        cov = np.dot(weighted_centered.T, centered)
        return cov

    def _principal_component(self, cov, prev_vec=None):
        """Extract first principal component with direction continuity.

        If prev_vec is provided, ensures smooth direction changes via:
        1. Sign alignment: flip if pointing backward
        2. Angle penalty: blend with previous direction based on angle
        """
        vals, vecs = np.linalg.eig(cov)
        idx = np.argsort(vals)[::-1]
        vecs = vecs[:, idx]

        gamma = vecs[:, 0]  # first principal component

        # Sign/direction handling
        if prev_vec is not None and np.linalg.norm(prev_vec) != 0:
            cos_alpha = np.dot(gamma, prev_vec) / (np.linalg.norm(gamma)*np.linalg.norm(prev_vec))
            if cos_alpha < 0:
                gamma = -gamma

            # Angle penalization
            cos_alpha = np.dot(gamma, prev_vec) / (np.linalg.norm(gamma)*np.linalg.norm(prev_vec))
            a_x = (abs(cos_alpha))**self.angle_penalty_exp
            gamma = a_x * gamma + (1 - a_x) * prev_vec
            gamma /= np.linalg.norm(gamma)

        return gamma

    def _forward_run(self, dataset, x_start):
        """Trace curve forward from starting point."""
        x = x_start
        path_x = [x]
        prev_gamma = None

        for _ in range(self.max_iter):
            mu = self._local_center_of_mass(dataset, x)
            cov = self._local_covariance(dataset, x, mu)
            gamma = self._principal_component(cov, prev_vec=prev_gamma)

            x_new = mu + self.h * gamma

            if np.linalg.norm(mu - x) < self.tol:
                path_x.append(x_new)
                break

            path_x.append(x_new)
            x = x_new
            prev_gamma = gamma

        return np.array(path_x)

    def _backward_run(self, dataset, x0, gamma0):
        """Trace curve backward from starting point."""
        x = x0
        path_x = [x]
        prev_gamma = -gamma0

        for _ in range(self.max_iter):
            mu = self._local_center_of_mass(dataset, x)
            cov = self._local_covariance(dataset, x, mu)
            gamma = self._principal_component(cov, prev_vec=prev_gamma)

            x_new = mu + self.h * gamma
            if np.linalg.norm(mu - x) < self.tol:
                path_x.append(x_new)
                break

            path_x.append(x_new)
            x = x_new
            prev_gamma = gamma

        return np.array(path_x)

    def _find_starting_point(self, dataset, start_point):
        """Find closest data point to desired start_point, or random if None."""
        if start_point is None:
            idx = np.random.choice(len(dataset))
            return dataset[idx], idx
        else:
            diffs = dataset - start_point
            dists = np.linalg.norm(diffs, axis=1)
            min_idx = np.argmin(dists)
            closest_pt = dataset[min_idx]
            return closest_pt, min_idx

    def fit(self, dataset, start_points=None, end_point=None, num_points=500):
        """Fit Local Principal Curve to dataset.

        Parameters
        ----------
        dataset : ndarray, shape (n_samples, n_features)
            Point cloud to fit curve through.
        start_points : ndarray or list of ndarray, optional
            Starting point(s) for curve fitting. If None, uses random point.
            Can be single point or list of points for multiple curves.
        end_point : ndarray, optional
            End point for curve fitting. Only allowed with single start_point.
            Currently not used for truncation but reserved for future use.
        num_points : int, default=500
            Number of equally-spaced points to sample along final curve.

        Returns
        -------
        paths : list of ndarray
            Raw paths traced by the algorithm.

        Notes
        -----
        After fitting, access results via:
        - self.paths: raw traced paths
        - self.cubic_splines: final curves with equal arc-length spacing
        - self.cubic_splines_eq: parametric spline interpolators
        """
        dataset = np.array(dataset)
        self.paths = []
        self.initializations = []

        if end_point is not None and start_points is None:
            raise ValueError("end_point provided but no start_points given. end_point only allowed if start_point is provided.")

        # Ensure start_points is a list
        if start_points is not None and not isinstance(start_points, (list, tuple)):
            start_points = [start_points]

        if end_point is not None and (start_points is None or len(start_points) != 1):
            raise ValueError("If end_point is provided, exactly one start_point must be provided.")

        for sp in (start_points if start_points is not None else [None]):
            x0, _ = self._find_starting_point(dataset, sp)

            forward_path = self._forward_run(dataset, x0)
            if len(forward_path) > 1:
                initial_gamma_direction = (forward_path[1] - forward_path[0]) / self.h
            else:
                initial_gamma_direction = np.zeros(dataset.shape[1])

            if np.linalg.norm(initial_gamma_direction) > 0:
                backward_path = self._backward_run(dataset, x0, initial_gamma_direction)
                full_path = np.vstack([backward_path[::-1], forward_path[1:]])
            else:
                full_path = forward_path

            # Check orientation: ensure x0 is closer to start than end
            dist_start_to_first = np.linalg.norm(x0 - full_path[0])
            dist_start_to_last = np.linalg.norm(x0 - full_path[-1])
            if dist_start_to_last < dist_start_to_first:
                full_path = full_path[::-1]

            self.paths.append(full_path)
            self.initializations.append(x0)

        # Fit splines and compute equal arc-length
        self._fit_cubic_splines_eq()
        self._compute_equal_arc_length_spline_points(num_points=num_points)

        return self.paths

    def _fit_cubic_splines_eq(self):
        """Fit cubic splines to raw paths in parameter space."""
        self.cubic_splines_eq = []
        for path in self.paths:
            if len(path) < 4:
                self.cubic_splines_eq.append(None)
                continue
            t = np.arange(len(path))
            splines_dict = {}
            for dim in range(path.shape[1]):
                splines_dict[dim] = CubicSpline(t, path[:, dim])
            self.cubic_splines_eq.append(splines_dict)

    def _compute_cubic_spline_points(self, num_points=500):
        """Evaluate cubic splines at evenly-spaced parameter values."""
        self.cubic_splines = []
        for i, eq in enumerate(self.cubic_splines_eq):
            if eq is None:
                self.cubic_splines.append(None)
                continue
            path = self.paths[i]
            t_values = np.linspace(0, len(path) - 1, num_points)
            spline_points = self.evaluate_cubic_spline(i, t_values)
            self.cubic_splines.append(spline_points)

    def evaluate_cubic_spline(self, path_idx, t_values):
        """Evaluate cubic spline at specified parameter values.

        Parameters
        ----------
        path_idx : int
            Index of the spline to evaluate.
        t_values : ndarray
            Parameter values at which to evaluate spline.

        Returns
        -------
        points : ndarray, shape (len(t_values), n_dims)
            Spline coordinates at requested parameter values.
        """
        if path_idx >= len(self.cubic_splines_eq) or self.cubic_splines_eq[path_idx] is None:
            raise ValueError(f"No cubic spline found for path index {path_idx}.")
        spline = self.cubic_splines_eq[path_idx]
        points = np.array([spline[dim](t_values) for dim in sorted(spline.keys())]).T
        return points

    def compute_arc_length(self, spline, t_min, t_max, num_samples=10000):
        """Compute cumulative arc length along spline.

        Parameters
        ----------
        spline : dict
            Dictionary of CubicSpline objects (one per dimension).
        t_min, t_max : float
            Parameter range to compute arc length over.
        num_samples : int, default=10000
            Number of samples for numerical integration.

        Returns
        -------
        t_values : ndarray
            Dense parameter samples.
        cumulative_length : ndarray
            Cumulative arc length at each parameter value.
        """
        t_values = np.linspace(t_min, t_max, num_samples)
        points = np.array([spline[dim](t_values) for dim in sorted(spline.keys())]).T

        distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
        cumulative_length = np.insert(np.cumsum(distances), 0, 0.0)
        return t_values, cumulative_length

    def get_uniformly_spaced_points(self, spline, num_points):
        """Resample spline at equal arc-length intervals.

        Parameters
        ----------
        spline : dict
            Dictionary of CubicSpline objects (one per dimension).
        num_points : int
            Number of points to sample.

        Returns
        -------
        uniform_points : ndarray, shape (num_points, n_dims)
            Spline coordinates at equal arc-length intervals.
        """
        path_length = len(spline[0].x)
        t_min = 0
        t_max = path_length - 1

        t_vals_dense, cum_length = self.compute_arc_length(spline, t_min, t_max, num_samples=5000)
        total_length = cum_length[-1]
        desired_distances = np.linspace(0, total_length, num_points)
        t_for_dist = interp1d(cum_length, t_vals_dense, kind='linear')(desired_distances)

        uniform_points = np.array([spline[dim](t_for_dist) for dim in sorted(spline.keys())]).T
        return uniform_points

    def _compute_equal_arc_length_spline_points(self, num_points=500):
        """Compute equal arc-length spline points for all fitted curves."""
        self.cubic_splines = []
        for i, eq in enumerate(self.cubic_splines_eq):
            if eq is None:
                self.cubic_splines.append(None)
                continue
            spline_points = self.get_uniformly_spaced_points(eq, num_points)
            self.cubic_splines.append(spline_points)
