# Embryo Performance Metrics Script
import os
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
import plotly.express as px
import re
from tqdm import tqdm
import pandas as pd

def spline_fit_wrapper(df, fit_cols=None, stage_col="predicted_stage_hpf", bandwidth=0.5, max_iter=2500, tol=1e-5,
                       angle_penalty_exp=1, n_boots=10, boot_size=2500, n_spline_points=500, time_window=2,
                       obs_weights=None):

    if fit_cols is None:
        # look for PCA cols
        pattern = r"PCA_.*_bio"
        fit_cols = [col for col in df.columns if re.search(pattern, col)]

    if obs_weights is None:
        obs_weights = np.ones((df.shape[0],))
    obs_weights = obs_weights / np.sum(obs_weights)

    boot_size = np.min([df.shape[0], boot_size])

    # Extract PCA coordinates
    pert_array = df[fit_cols].values

    # Compute average early stage point
    min_time = df[stage_col].min()
    early_mask = (df[stage_col] >= min_time) & \
                 (df[stage_col] < min_time + time_window)
    early_points = df.loc[early_mask, fit_cols].values

    early_options = np.arange(early_points.shape[0])

    # Compute average late stage point
    max_time = df[stage_col].max()
    late_mask = (df[stage_col] >= (max_time - time_window))
    late_points = df.loc[late_mask, fit_cols].values
    late_options = np.arange(late_points.shape[0])

    # generate array to store spline fits
    spline_boot_array = np.zeros((n_spline_points, len(fit_cols), n_boots))

    # Randomly select a subset of points for fitting
    rng = np.random.RandomState(42)

    for n in tqdm(range(n_boots)):
        subset_indices = rng.choice(len(pert_array), size=boot_size, replace=True, p=obs_weights)
        pert_array_subset = pert_array[subset_indices, :]

        start_ind = np.random.choice(early_options, 1)[0]
        stop_ind = np.random.choice(late_options, 1)[0]
        start_point = early_points[start_ind, :]
        stop_point = late_points[stop_ind, :]

        # Fit LocalPrincipalCurve
        lpc = LocalPrincipalCurve(
            bandwidth=bandwidth,
            max_iter=max_iter,
            tol=tol,
            angle_penalty_exp=angle_penalty_exp
        )

        # Fit with the optional start_points/end_point to anchor the spline
        lpc.fit(
            pert_array_subset,
            start_points=start_point[None, :],
            end_point=stop_point[None, :],
            num_points=n_spline_points
        )

        spline_boot_array[:, :, n] = lpc.cubic_splines[0]

    # get mean and standard error
    mean_spline = np.mean(spline_boot_array, axis=2)
    se_spline = np.std(spline_boot_array, axis=2)

    # make data frame
    se_cols = [col + "_se" for col in fit_cols]
    spline_df = pd.DataFrame(mean_spline, columns=fit_cols)
    spline_df[se_cols] = se_spline

    return spline_df

def plot_trajectories_3d(splines_final, plot_cols=None, save_dir=None):
    """
    Plots PCA trajectories for different perturbations and datasets in a 3D Plotly plot.

    Parameters:
    splines_final (pd.DataFrame): DataFrame containing the trajectory data with columns
                                  ['dataset', 'Perturbation', 'point_index']

    Returns:
    None
    """
    if plot_cols is None:
        plot_cols = ["PCA_00_bio", "PCA_02_bio", "PCA_02_bio"]
    
    # Initialize the figure
    fig = px.line_3d(splines_final, x=plot_cols[0], y=plot_cols[1], z=plot_cols[2], color="Perturbation")
    fig.update_traces(line=dict, width=4)

    if save_dir:
        fig.write_html(os.path.join(save_dir, "model_splines.html"))
 
    # Show the plot
    return fig

class LocalPrincipalCurve:
    def __init__(self, bandwidth=0.5, max_iter=100, tol=1e-4, angle_penalty_exp=2, h=None):
        """
        Initialize the Local Principal Curve solver.
        """
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
        dists = np.linalg.norm(dataset - x, axis=1)
        weights = np.exp(- (dists**2) / (2 * self.bandwidth**2))
        w = weights / np.sum(weights)
        return w

    def _local_center_of_mass(self, dataset, x):
        w = self._kernel_weights(dataset, x)
        mu = np.sum(dataset.T * w, axis=1)
        return mu

    def _local_covariance(self, dataset, x, mu):
        w = self._kernel_weights(dataset, x)
        centered = dataset - mu
        # cov = np.zeros((dataset.shape[1], dataset.shape[1]))
        weighted_centered = centered * w[:, np.newaxis]  # shape: (n, d)
        cov = np.dot(weighted_centered.T, centered)  # shape: (d, d)
        # for i in range(len(dataset)):
        #     cov += w[i] * np.outer(centered[i], centered[i])
        return cov

    def _principal_component(self, cov, prev_vec=None):
        vals, vecs = np.linalg.eig(cov)
        idx = np.argsort(vals)[::-1]
        # vals = vals[idx]
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
        if start_point is None:
            idx = np.random.choice(len(dataset))
            return dataset[idx], idx
        else:
            diffs = dataset - start_point
            dists = np.linalg.norm(diffs, axis=1)
            min_idx = np.argmin(dists)
            closest_pt = dataset[min_idx]
            # if not np.allclose(closest_pt, start_point, rtol=1e-01):
            #     print(f"Starting point not in dataset. Using closest point: {closest_pt}")
            return closest_pt, min_idx

    def fit(self, dataset, start_points=None, end_point=None, num_points=500):
        """
        Fit LPC on the dataset. Optionally provide:
         - start_points: array of shape (d,) or a single point of shape (d,)
         - end_point: single point of shape (d,), only allowed if a start_point is provided.
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
            # Debugging
            # import pdb
            # pdb.set_trace()
            # Debugging
            if np.linalg.norm(initial_gamma_direction) > 0:
                backward_path = self._backward_run(dataset, x0, initial_gamma_direction)
                full_path = np.vstack([backward_path[::-1], forward_path[1:]])
            else:
                full_path = forward_path

            # Check orientation
            dist_start_to_first = np.linalg.norm(x0 - full_path[0])
            dist_start_to_last = np.linalg.norm(x0 - full_path[-1])
            if dist_start_to_last < dist_start_to_first:
                full_path = full_path[::-1]

            self.paths.append(full_path)
            self.initializations.append(x0)

        # Fit splines and compute equal arc-length
        self._fit_cubic_splines_eq()
        self._compute_equal_arc_length_spline_points(num_points=num_points)

        # If end_point provided, correct for the looping back issue
        # if end_point is not None:
        #     try:
        #         # Assuming a single path scenario
        #         spline_points = self.cubic_splines[0]
        #
        #         # 1) Find closest point on cubic_spline to end_point
        #         dists = np.linalg.norm(spline_points - end_point, axis=1)
        #         closest_idx = np.argmin(dists)
        #
        #         # 2) Determine end_direction_vector using points around closest_idx
        #         # We'll take up to 3 points: [closest_idx-1, closest_idx, closest_idx+1]
        #         # If closest_idx is at the boundary, adjust accordingly
        #         if closest_idx == 0:
        #             # At start, use next two points if available
        #             if len(spline_points) > 2:
        #                 p0 = spline_points[closest_idx]
        #                 p1 = spline_points[closest_idx + 1]
        #                 p2 = spline_points[closest_idx + 2]
        #                 end_direction_vector = ((p1 - p0) + (p2 - p1)) / 2.0
        #             else:
        #                 # If very short, just fallback
        #                 end_direction_vector = np.array([1, 0, 0])
        #         elif closest_idx == len(spline_points) - 1:
        #             # At the end, we might not have a point after it
        #             # use the two points before it if possible
        #             if len(spline_points) > 2:
        #                 p_end = spline_points[closest_idx]
        #                 p_endm1 = spline_points[closest_idx - 1]
        #                 p_endm2 = spline_points[closest_idx - 2]
        #                 end_direction_vector = ((p_end - p_endm1) + (p_endm1 - p_endm2)) / 2.0
        #             else:
        #                 end_direction_vector = np.array([1, 0, 0])
        #         else:
        #             # Middle somewhere, use prev and next
        #             p_before = spline_points[closest_idx - 1]
        #             p_mid = spline_points[closest_idx]
        #             p_after = spline_points[closest_idx + 1]
        #             end_direction_vector = ((p_mid - p_before) + (p_after - p_mid)) / 2.0
        #
        #         # Normalize end_direction_vector
        #         norm_edv = np.linalg.norm(end_direction_vector)
        #         if norm_edv > 0:
        #             end_direction_vector = end_direction_vector / norm_edv
        #         else:
        #             warnings.warn("end_direction_vector has zero magnitude. Using default direction.")
        #             end_direction_vector = np.array([1, 0, 0])
        #
        #         # 3) Check directionality after closest_idx
        #         # We'll look at pairs of points (p_j, p_{j+1}) for j > closest_idx
        #         cutoff_index = None
        #         for j in range(closest_idx + 1, len(spline_points) - 1):
        #             seg_vec = spline_points[j + 1] - spline_points[j]
        #             csim = cosine_similarity(seg_vec.reshape(1, -1), end_direction_vector.reshape(1, -1))
        #             if csim < 0.5:
        #                 cutoff_index = j + 1
        #                 break
        #
        #         # If we found a cutoff_index, truncate the spline
        #         if cutoff_index is not None:
        #             spline_points = spline_points[:cutoff_index]
        #
        #             # Refit with truncated spline_points
        #             self.paths = [spline_points]
        #             self._fit_cubic_splines_eq()
        #             self._compute_equal_arc_length_spline_points()
        #
        #     except (ValueError, IndexError, TypeError) as e:
        #         # Log a warning and exit the if block gracefully
        #         warnings.warn(
        #             f"Error processing spline with end_point: {e}. Skipping spline adjustment."
        #         )
        #         # Optionally, you can log more details for debugging
        #         # For example:
        #         # warnings.warn(f"Error processing spline: {e}. spline_points shape: {spline_points.shape}, end_point shape: {np.shape(end_point)}")
        #         return  # Exit the if block

        return self.paths

    def _fit_cubic_splines_eq(self):
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
        if path_idx >= len(self.cubic_splines_eq) or self.cubic_splines_eq[path_idx] is None:
            raise ValueError(f"No cubic spline found for path index {path_idx}.")
        spline = self.cubic_splines_eq[path_idx]
        points = np.array([spline[dim](t_values) for dim in sorted(spline.keys())]).T  # Fixed line
        return points

    def compute_arc_length(self, spline, t_min, t_max, num_samples=10000):
        t_values = np.linspace(t_min, t_max, num_samples)
        points = np.array([spline[dim](t_values) for dim in sorted(spline.keys())]).T  # Fixed line

        distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
        cumulative_length = np.insert(np.cumsum(distances), 0, 0.0)
        return t_values, cumulative_length

    def get_uniformly_spaced_points(self, spline, num_points):
        path_length = len(spline[0].x)
        t_min = 0
        t_max = path_length - 1

        t_vals_dense, cum_length = self.compute_arc_length(spline, t_min, t_max, num_samples=5000)
        total_length = cum_length[-1]
        desired_distances = np.linspace(0, total_length, num_points)
        t_for_dist = interp1d(cum_length, t_vals_dense, kind='linear')(desired_distances)

        uniform_points = np.array([spline[dim](t_for_dist) for dim in sorted(spline.keys())]).T  # Fixed line
        return uniform_points

    def _compute_equal_arc_length_spline_points(self, num_points=500):
        self.cubic_splines = []
        for i, eq in enumerate(self.cubic_splines_eq):
            if eq is None:
                self.cubic_splines.append(None)
                continue
            spline_points = self.get_uniformly_spaced_points(eq, num_points)
            self.cubic_splines.append(spline_points)
