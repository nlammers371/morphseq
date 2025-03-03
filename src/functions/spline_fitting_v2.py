# Embryo Performance Metrics Script
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from scipy.interpolate import CubicSpline, interp1d
import plotly.graph_objects as go
import warnings
import plotly.express as px


# #example ussage of function to produce all the core perfomance metrics. 
# # Call the function
# core_performance_metrics, distance_metrics_intra_inter, metrics_inter_df = produce_perfomance_metrics(
#     df_all,
#     df_hld,
#     pert_comparisons,
#     logreg_tol=1e-3,
#     subsample_fraction=0.05,
#     subsample_fraction_jaccard=0.1,
#     num_bins=20,
#     max_hpf=40,
#     random_state=100,
#     plot=True,
#     k_neighbors=5
# )


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

    # def plot_path_3d(self, path_idx=0, dataset=None):
    #     dataset = np.array(dataset)
    #     path = self.paths[path_idx]
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     if dataset is not None:
    #         ax.scatter(dataset[:,0], dataset[:,1], dataset[:,2], alpha=0.5, label='Data')
    #     ax.plot(path[:,0], path[:,1], path[:,2], 'r-', label='Local Principal Curve')
    #     ax.legend()
    #     plt.show()
    #
    # def plot_cubic_spline_3d(self, path_idx, show_path=True):
    #     if path_idx >= len(self.paths):
    #         raise IndexError(f"Path index {path_idx} is out of range. Total paths: {len(self.paths)}.")
    #     path = self.paths[path_idx]
    #     spline_points = self.cubic_splines[path_idx]
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection="3d")
    #     if show_path:
    #         ax.scatter(path[:, 0], path[:, 1], path[:, 2], label="LPC Path", alpha=0.5)
    #     ax.plot(spline_points[:, 0], spline_points[:, 1], spline_points[:, 2], color="red", label="Cubic Spline")
    #     ax.legend()
    #     plt.show()


# def extract_spline(splines_df, dataset_label, perturbation):
#     sdf = splines_df[(splines_df["dataset"] == dataset_label) & (splines_df["Perturbation"] == perturbation)]
#     sdf = sdf.sort_values("point_index")
#     pca_cols = [col for col in sdf.columns.tolist() if "PCA" in col]
#     points = sdf[pca_cols].values
#     return points
#
# def rmse(a, b):
#     return np.sqrt(np.mean((a - b)**2))
#
# def mean_l1_error(a, b):
# # a and b are Nx3 arrays of points.
# # Compute L1 distance for each point pair: sum of absolute differences across coordinates
# # Then take the mean over all points.
#     return np.mean(np.sum(np.abs(a - b), axis=1))
#
# def centroid(X):
#     return np.mean(X, axis=0)
#
# def rmsd(X, Y):
#     return np.sqrt(np.mean(np.sum((X - Y)**2, axis=1)))

# def quaternion_alignment(P, Q):
#     """
#     Compute the optimal rotation using quaternions that aligns Q onto P.
#     Returns rotation matrix R and translation vector t.
#     """
#     # Ensure P and Q have the same shape
#     assert P.shape == Q.shape, "P and Q must have the same shape"
#
#     # 1. Compute centroids and center the points
#     P_cent = centroid(P)
#     Q_cent = centroid(Q)
#     P_prime = P - P_cent
#     Q_prime = Q - Q_cent
#
#     # 2. Construct correlation matrix M
#     M = Q_prime.T @ P_prime
#
#     # 3. Construct the Kearsley (Davenport) 4x4 matrix K
#     # Refer to the equations above
#     A = np.array([
#         [ M[0,0]+M[1,1]+M[2,2],   M[1,2]-M[2,1],         M[2,0]-M[0,2],         M[0,1]-M[1,0]       ],
#         [ M[1,2]-M[2,1],         M[0,0]-M[1,1]-M[2,2],  M[0,1]+M[1,0],         M[0,2]+M[2,0]       ],
#         [ M[2,0]-M[0,2],         M[0,1]+M[1,0],         M[1,1]-M[0,0]-M[2,2],  M[1,2]+M[2,1]       ],
#         [ M[0,1]-M[1,0],         M[0,2]+M[2,0],         M[1,2]+M[2,1],         M[2,2]-M[0,0]-M[1,1]]
#     ], dtype=np.float64)
#     A = A / 3.0
#
#     # 4. Find the eigenvector of A with the highest eigenvalue
#     eigenvalues, eigenvectors = np.linalg.eigh(A)
#     max_idx = np.argmax(eigenvalues)
#     q = eigenvectors[:, max_idx]
#     q = q / np.linalg.norm(q)
#
#     # 5. Convert quaternion q into rotation matrix R
#     # Quaternion format: q = [q0, q1, q2, q3]
#     q0, q1, q2, q3 = q
#     R = np.array([
#         [q0**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 - q0*q3),         2*(q1*q3 + q0*q2)],
#         [2*(q2*q1 + q0*q3),             q0**2 - q1**2 + q2**2 - q3**2, 2*(q2*q3 - q0*q1)],
#         [2*(q3*q1 - q0*q2),             2*(q3*q2 + q0*q1),             q0**2 - q1**2 - q2**2 + q3**2]
#     ])
#
#     # 6. Compute translation
#     t = P_cent - R @ Q_cent
#
#     return R, t

# def _segment_direction_metrics(data_a, data_b, k=10):
#     """
#     Compute SegmentColinearity and SegmentCovariance for two given sets of points `data_a` and `data_b`.
#     Both data_a and data_b are np.ndarray of shape (n, 3).
#
#     If there aren't enough points for k segments, returns (np.nan, np.nan).
#     """
#     min_len = min(len(data_a), len(data_b))
#     data_a = data_a[:min_len]
#     data_b = data_b[:min_len]
#
#     if min_len < k + 1 or min_len == 0:
#         return (np.nan, np.nan)
#
#     # Define segments using data_b
#     segment_indices = np.linspace(0, min_len - 1, k + 1, dtype=int)
#
#     aligned_segment_vecs = []
#     all_segment_vecs = []
#
#     for i in range(k):
#         start_idx = segment_indices[i]
#         end_idx = segment_indices[i + 1]
#
#         start_b = data_b[start_idx]
#         end_b = data_b[end_idx]
#
#         # Find closest points in data_a to start_b and end_b
#         start_dists = np.linalg.norm(data_a - start_b, axis=1)
#         closest_start_idx = np.argmin(start_dists)
#         closest_start_a = data_a[closest_start_idx]
#
#         end_dists = np.linalg.norm(data_a - end_b, axis=1)
#         closest_end_idx = np.argmin(end_dists)
#         closest_end_a = data_a[closest_end_idx]
#
#         # Construct vectors
#         vec_a = closest_end_a - closest_start_a
#         vec_b = end_b - start_b
#
#         # Normalize
#         norm_a = np.linalg.norm(vec_a)
#         norm_b = np.linalg.norm(vec_b)
#         if norm_a > 0:
#             vec_a = vec_a / norm_a
#         else:
#             vec_a = np.zeros(3)
#         if norm_b > 0:
#             vec_b = vec_b / norm_b
#         else:
#             vec_b = np.zeros(3)
#
#         aligned_segment_vecs.append(vec_a)
#         all_segment_vecs.append(vec_b)
#
#     aligned_segment_vecs = np.array(aligned_segment_vecs)
#     all_segment_vecs = np.array(all_segment_vecs)
#
#     # Cosine similarities
#     cos_sims = []
#     for i in range(len(aligned_segment_vecs)):
#         va = aligned_segment_vecs[i].reshape(1, -1)
#         vb = all_segment_vecs[i].reshape(1, -1)
#         sim = cosine_similarity(va, vb)[0][0]
#         cos_sims.append(sim)
#
#     avg_cosine_sim = np.mean(cos_sims) if len(cos_sims) > 0 else np.nan
#
#     # Covariances
#     covariances = []
#     for dim_idx in range(3):
#         dim_a = aligned_segment_vecs[:, dim_idx]
#         dim_b = all_segment_vecs[:, dim_idx]
#         if len(dim_a) > 1:
#             cov = np.cov(dim_a, dim_b, bias=True)[0, 1]
#         else:
#             cov = np.nan
#         covariances.append(cov)
#     avg_cov = np.nanmean(covariances) if len(covariances) > 0 else np.nan
#
#     return (avg_cosine_sim, avg_cov)


    # Split the dataset into 'all' and 'hld_aligned'
    # splines_all = splines_final_df[splines_final_df["dataset"] == "all"]
    # splines_hld_aligned = splines_final_df[splines_final_df["dataset"] == "hld_aligned"]

# def segment_direction_consistency(splines_final_df, k=10):
#     """
#     Step 1 (Across): For each perturbation present in both datasets, compute SegmentColinearity and SegmentCovariance
#     by comparing splines_hld_aligned and splines_all.
#
#     Step 2 (Within): Compute these metrics for all unique pairs of perturbations within each dataset
#     (both splines_hld_aligned and splines_all separately).
#     Then compute the mean and std of these pairwise metrics for each dataset.
#
#     Returns:
#     - across_df: DataFrame with ['Perturbation', 'SegmentColinearity', 'SegmentCovariance']
#     - within_hld_aligned_df: DataFrame with ['Metric', 'Mean', 'Std'] for pairwise metrics within splines_hld_aligned
#     - within_all_df: DataFrame with ['Metric', 'Mean', 'Std'] for pairwise metrics within splines_all
#     """
#
#     splines_all = splines_final_df[splines_final_df["dataset"] == "all"]
#     splines_hld = splines_final_df[splines_final_df["dataset"] == "hld"]
#     splines_hld_aligned = splines_final_df[splines_final_df["dataset"] == "hld_aligned"]
#
#     pca_columns = [col for col in splines_all.columns.tolist() if "PCA" in col]
#     for col in pca_columns:
#         if col not in splines_hld_aligned.columns or col not in splines_all.columns:
#             raise ValueError(f"Missing required PCA column: {col}")
#
#
#
#     # Across computations
#     perts_aligned = set(splines_hld_aligned["Perturbation"].unique())
#     perts_all = set(splines_all["Perturbation"].unique())
#     common_perts = perts_aligned.intersection(perts_all)
#
#     across_results = []
#     for pert in common_perts:
#         data_a_df = splines_hld_aligned[splines_hld_aligned["Perturbation"] == pert].sort_values("point_index")
#         data_b_df = splines_all[splines_all["Perturbation"] == pert].sort_values("point_index")
#         data_a = data_a_df[pca_columns].values
#         data_b = data_b_df[pca_columns].values
#
#         sim, cov = _segment_direction_metrics(data_a, data_b, k=k)
#         across_results.append({"Perturbation": pert, "SegmentColinearity": sim, "SegmentCovariance": cov})
#
#     across_df = pd.DataFrame(across_results)
#
#     # Calculate column means (excluding the Perturbation column)
#     mean_row = across_df.iloc[:, 1:].mean()
#     mean_row["Perturbation"] = "avg_pert"
#
#     # Append the mean row to the DataFrame
#     across_df = pd.concat([across_df, pd.DataFrame([mean_row])], ignore_index=True)
#
#
#     # Within computations for splines_hld
#     perts_in_aligned = list(perts_aligned)
#     within_values_colinearity_hld = []
#     within_values_covariance_hld  = []
#
#     for i in range(len(perts_in_aligned)):
#         for j in range(i+1, len(perts_in_aligned)):
#             pert1 = perts_in_aligned[i]
#             pert2 = perts_in_aligned[j]
#
#             data_pert1 = splines_hld[splines_hld["Perturbation"] == pert1].sort_values("point_index")[pca_columns].values
#             data_pert2 = splines_hld[splines_hld["Perturbation"] == pert2].sort_values("point_index")[pca_columns].values
#
#             sim, cov = _segment_direction_metrics(data_pert1, data_pert2, k=k)
#             if not np.isnan(sim):
#                 within_values_colinearity_hld.append(sim)
#             if not np.isnan(cov):
#                 within_values_covariance_hld.append(cov)
#
#     metrics_hld = []
#     for metric_name, vals in [("SegmentColinearity", within_values_colinearity_hld),
#                               ("SegmentCovariance",  within_values_covariance_hld)]:
#         mean_val = np.nanmean(vals) if len(vals) > 0 else np.nan
#         std_val = np.nanstd(vals) if len(vals) > 0 else np.nan
#         metrics_hld.append({"Metric": metric_name, "Mean": mean_val, "Std": std_val})
#
#     within_hld_df = pd.DataFrame(metrics_hld)
#
#     # Within computations for splines_all
#     perts_in_all = list(perts_all)
#     within_values_colinearity_all = []
#     within_values_covariance_all = []
#
#     for i in range(len(perts_in_all)):
#         for j in range(i+1, len(perts_in_all)):
#             pert1 = perts_in_all[i]
#             pert2 = perts_in_all[j]
#
#             data_pert1 = splines_all[splines_all["Perturbation"] == pert1].sort_values("point_index")[pca_columns].values
#             data_pert2 = splines_all[splines_all["Perturbation"] == pert2].sort_values("point_index")[pca_columns].values
#
#             sim, cov = _segment_direction_metrics(data_pert1, data_pert2, k=k)
#             if not np.isnan(sim):
#                 within_values_colinearity_all.append(sim)
#             if not np.isnan(cov):
#                 within_values_covariance_all.append(cov)
#
#     metrics_all_list = []
#     for metric_name, vals in [("SegmentColinearity", within_values_colinearity_all),
#                               ("SegmentCovariance", within_values_covariance_all)]:
#         mean_val = np.nanmean(vals) if len(vals) > 0 else np.nan
#         std_val = np.nanstd(vals) if len(vals) > 0 else np.nan
#         metrics_all_list.append({"Metric": metric_name, "Mean": mean_val, "Std": std_val})
#
#     within_all_df = pd.DataFrame(metrics_all_list)
#
#     return across_df, within_hld_df, within_all_df

# def calculate_dispersion_metrics(splines_final_df, n=5):
#     """
#     Calculates dispersion metrics for each dataset, including:
#     - Dispersion Coefficient (slope of dispersion vs. point_index, normalized to [0, 1])
#     - Initial Dispersion (average dispersion of the first n points)
#     - Last Dispersion (average dispersion of the last n points)
#
#     Parameters:
#     - splines_final_df (pd.DataFrame): DataFrame containing all PCA trajectories with 'dataset' column.
#     - n (int): Number of initial and last points to consider for initial and last dispersion.
#
#     Returns:
#     - pd.DataFrame: DataFrame with columns ['Dataset', 'disp_coefficient', 'dispersion_first_n', 'dispersion_last_n'].
#     """
#     # Extract subsets
#     # splines_all = splines_final_df[splines_final_df["dataset"] == "all"]
#     # splines_hld = splines_final_df[splines_final_df["dataset"] == "hld"]
#     # splines_hld_aligned = splines_final_df[splines_final_df["dataset"] == "hld_aligned"]
#
#     # Ensure PCA columns are present
#     pca_columns = [col for col in splines_final_df.columns.tolist() if "PCA" in col]
#     for col in pca_columns:
#         if col not in splines_final_df.columns:
#             raise ValueError(f"Missing required PCA column: {col}")
#
#     # Get unique datasets
#     datasets = splines_final_df["dataset"].unique()
#
#     # Initialize list to store results
#     results = []
#
#     for dataset in datasets:
#         if dataset == "hld_aligned":
#             continue
#         # Filter data for the current dataset
#         dataset_df = splines_final_df[splines_final_df["dataset"] == dataset]
#
#         # Get unique point_indices
#         point_indices = sorted(dataset_df["point_index"].unique())
#
#         # Initialize lists to store dispersion and point_index
#         dispersion_list = []
#         point_index_list = []
#
#         # Initialize lists to store initial and last dispersions
#         initial_dispersions = []
#         last_dispersions = []
#
#         for pid in point_indices:
#             # Filter data for the current point_index
#             point_df = dataset_df[dataset_df["point_index"] == pid]
#
#             # Calculate dispersion: average Euclidean distance from centroid
#             dispersion = compute_dispersion(point_df, pca_columns)
#
#             # Append to lists
#             dispersion_list.append(dispersion)
#             point_index_list.append(pid)
#
#             # If within first n points, store for initial dispersion
#             if pid < n:
#                 initial_dispersions.append(dispersion)
#
#             # If within last n points, store for last dispersion
#             if pid >= max(point_indices) - n + 1:
#                 last_dispersions.append(dispersion)
#
#         # Check if there are enough points for regression
#         if len(point_index_list) < 2:
#             print(f"Warning: Dataset '{dataset}' has less than 2 unique point_indices. Setting disp_coefficient to NaN.")
#             disp_coefficient = np.nan
#         else:
#             # Prepare data for linear regression
#             X = np.array(point_index_list).reshape(-1, 1)  # Shape: (num_points, 1)
#             y = np.array(dispersion_list)  # Shape: (num_points,)
#
#             # Fit linear regression
#             reg = LinearRegression().fit(X, y)
#             disp_coefficient = reg.coef_[0]
#             disp_coefficient *= len(point_indices)  # Normalize to [0, 1]
#
#         # Calculate average initial dispersion
#         dispersion_first_n = np.mean(initial_dispersions) if initial_dispersions else np.nan
#         if np.isnan(dispersion_first_n):
#             print(f"Warning: Dataset '{dataset}' has no points within the first {n} point_indices.")
#
#         # Calculate average last dispersion
#         dispersion_last_n = np.mean(last_dispersions) if last_dispersions else np.nan
#         if np.isnan(dispersion_last_n):
#             print(f"Warning: Dataset '{dataset}' has no points within the last {n} point_indices.")
#
#         # Append results
#         results.append({
#             "Dataset": dataset,
#             "disp_coefficient": disp_coefficient,
#             "dispersion_first_n": dispersion_first_n,
#             "dispersion_last_n": dispersion_last_n
#         })
#
#     # Convert results to DataFrame
#     results_df = pd.DataFrame(results)
#
#
#
#     return results_df

# def compute_dispersion(df, pca_columns):
#     """
#     Computes the average Euclidean distance of points from their centroid.
#
#     Parameters:
#     - df (pd.DataFrame): DataFrame containing PCA coordinates.
#     - pca_columns (list): List of PCA column names.
#
#     Returns:
#     - float: Average Euclidean distance (dispersion).
#     """
#     if df.empty:
#         return np.nan
#
#     # Calculate centroid
#     centroid = df[pca_columns].mean().values
#
#     # Calculate Euclidean distances from centroid
#     distances = np.linalg.norm(df[pca_columns].values - centroid, axis=1)
#
#     # Return average distance
#     return distances.mean()

# import pandas as pd
#
# # -------------------------------
# # Helper Functions
# # -------------------------------
#
# def rename_within_metrics(df, suffix, key):
#     """Renames columns in within metrics DataFrame with a given suffix."""
#     renamed_df = df[["Metric", "Mean"]].copy()
#     renamed_df["Metric"] += suffix  # Add suffix
#     renamed_df = renamed_df.set_index("Metric").T  # Transpose for easy appending
#     renamed_df.insert(0, "model_index", key)  # Add model_index
#     return renamed_df
#
# def process_dispersion_metrics(df, key):
#     """Processes and renames dispersion metrics DataFrame."""
#     disp_all = df[df["Dataset"] == "all"].drop("Dataset", axis=1)
#     disp_all.columns = [col + "_all" for col in disp_all.columns]
#     disp_hld = df[df["Dataset"] == "hld"].drop("Dataset", axis=1)
#     disp_hld.columns = [col + "_hld" for col in disp_hld.columns]
#
#     combined_df = pd.concat([disp_all.reset_index(drop=True), disp_hld.reset_index(drop=True)], axis=1)
#     combined_df.insert(0, "model_index", key)  # Add model_index
#     return combined_df
#
# def process_segment_direction(splines_final_df, key):
#     """Calculates and processes segment direction consistency metrics."""
#     across_seg_df, within_hld_seg_df, within_all_seg_df = segment_direction_consistency(splines_final_df, k=100)
#     across_seg_df.insert(0, "model_index", key)  # Add model_index
#
#     within_hld_renamed = rename_within_metrics(within_hld_seg_df, "_mean_within_hld", key)
#     within_all_renamed = rename_within_metrics(within_all_seg_df, "_mean_within_all", key)
#
#     within_seg_measures = pd.concat([within_hld_renamed, within_all_renamed], axis=1)
#     return across_seg_df, within_seg_measures
#
# def combine_results_dict(results_dict):
#     """
#     Combines the results dictionary into a single DataFrame.
#     Handles duplicate 'model_index' columns by ensuring uniqueness during merge.
#     """
#     final_list_of_dfs = []
#
#     for model_index, metrics in results_dict.items():
#         # Start with across_seg_df as the base since it has multiple perturbations
#         if "across_seg_df" not in metrics:
#             continue  # If for some reason this key doesn't have across_seg_df, skip
#
#         base_df = metrics["across_seg_df"].copy()
#
#         # Drop duplicate 'model_index' columns from other metrics before merging
#         if "within_seg_measures" in metrics:
#             temp_within = metrics["within_seg_measures"].copy()
#             temp_within = temp_within.loc[:, ~temp_within.columns.duplicated()]  # Remove duplicate columns
#             base_df = base_df.merge(temp_within, on="model_index", how="left")
#
#         if "dispersion_metrics" in metrics:
#             temp_disp = metrics["dispersion_metrics"].copy()
#             temp_disp = temp_disp.loc[:, ~temp_disp.columns.duplicated()]  # Remove duplicate columns
#             base_df = base_df.merge(temp_disp, on="model_index", how="left")
#
#         # Append to list
#         final_list_of_dfs.append(base_df)
#
#     # Concatenate all model results
#     if final_list_of_dfs:
#         final_results_df = pd.concat(final_list_of_dfs, ignore_index=True)
#     else:
#         final_results_df = pd.DataFrame()
#
#     return final_results_df
