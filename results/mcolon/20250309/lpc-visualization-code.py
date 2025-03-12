import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import math
from tqdm import tqdm
import imageio
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline

class LocalPrincipalCurveVisualizer:
    def __init__(self, lpc, dataset, bandwidth=0.5, h=None, max_iter=100, 
                 tol=1e-4, angle_penalty_exp=2, output_dir="lpc_frames"):
        """
        Extends the LocalPrincipalCurve class with visualization capabilities.
        
        Parameters:
        -----------
        lpc : LocalPrincipalCurve instance
            The LPC algorithm instance to visualize
        dataset : numpy.ndarray
            The dataset being analyzed
        output_dir : str
            Directory to save visualization frames
        """
        self.lpc = lpc
        self.dataset = dataset
        self.output_dir = output_dir
        self.frame_count = 0
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Colors
        self.base_point_color = [0.8, 0.8, 0.8, 0.3]  # Light gray, semi-transparent for dataset
        self.weighted_point_color = [0.0, 0.5, 1.0, 1.0]  # Blue for weighted points
        self.center_color = [1.0, 0.0, 0.0, 1.0]  # Red for center of mass
        self.eigenvector_color = [0.0, 0.8, 0.0, 1.0]  # Green for eigenvector
        self.path_color = [0.7, 0.0, 0.7, 0.8]  # Purple for deposited points
        self.spline_color = [1.0, 0.5, 0.0, 1.0]  # Orange for spline
        
        # Camera parameters for 3D rotation
        self.angles = np.linspace(0, 2*np.pi, 360)
        self.current_angle_idx = 0
        self.rotate_speed = 1  # Frames to advance angle
        
        # Store the path history
        self.path_history = []
        
        # Store current step information
        self.current_x = None
        self.current_mu = None
        self.current_gamma = None
        self.current_weights = None
        self.step_counter = 0
        
    def _create_frame(self, x=None, mu=None, gamma=None, weights=None, 
                      is_spline_phase=False, final_rotation=False):
        """
        Creates a frame showing the current state of the algorithm.
        
        Parameters:
        -----------
        x : numpy.ndarray
            Current point
        mu : numpy.ndarray
            Local center of mass
        gamma : numpy.ndarray
            Principal component direction
        weights : numpy.ndarray
            Weights of all points at current step
        is_spline_phase : bool
            Whether we're in the spline fitting phase
        final_rotation : bool
            Whether to do a full rotation for the final result
        """
        # Create a new figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set viewing angle for rotation
        if final_rotation:
            # Do a full rotation
            angle_idx = self.frame_count % len(self.angles)
        elif is_spline_phase:
            # Keep the angle fixed during spline phase
            angle_idx = self.current_angle_idx
        else:
            # Slowly rotate during calculation
            if self.frame_count % self.rotate_speed == 0:
                self.current_angle_idx = (self.current_angle_idx + 1) % len(self.angles)
            angle_idx = self.current_angle_idx
        
        angle = self.angles[angle_idx]
        ax.view_init(elev=30, azim=angle * 180 / np.pi)
        
        # Plot the dataset with base transparency
        ax.scatter(self.dataset[:, 0], self.dataset[:, 1], self.dataset[:, 2], 
                   color=self.base_point_color, s=10)
        
        # Plot path history (deposited points)
        if self.path_history:
            path_points = np.array(self.path_history)
            ax.scatter(path_points[:, 0], path_points[:, 1], path_points[:, 2], 
                       color=self.path_color, s=30)
        
        # If in active calculation phase, show current step details
        if not is_spline_phase and not final_rotation and x is not None:
            # Show current step counter
            ax.set_title(f"Step: {self.step_counter}")
            
            # Plot weighted points (with color intensity based on weight)
            if weights is not None:
                # Normalize weights for visualization
                max_weight = np.max(weights) if len(weights) > 0 else 1
                for i, point in enumerate(self.dataset):
                    if weights[i] > 0.001:  # Only show points with significant weight
                        weight_alpha = min(weights[i] / max_weight, 1.0) 
                        point_color = self.weighted_point_color.copy()
                        point_color[3] = weight_alpha
                        ax.scatter(point[0], point[1], point[2], color=point_color, s=30)
            
            # Plot current point
            ax.scatter(x[0], x[1], x[2], color='black', s=50)
            
            # Plot local center of mass
            if mu is not None:
                ax.scatter(mu[0], mu[1], mu[2], color=self.center_color, s=80, marker='*')
            
            # Plot eigenvector as an arrow
            if gamma is not None and mu is not None:
                arrow_length = self.lpc.h
                arrow_end = mu + arrow_length * gamma
                ax.quiver(mu[0], mu[1], mu[2], 
                         arrow_end[0] - mu[0], arrow_end[1] - mu[1], arrow_end[2] - mu[2],
                         color=self.eigenvector_color, arrow_length_ratio=0.2)
        
        # If in spline phase or final rotation, show the spline
        if is_spline_phase or final_rotation:
            if is_spline_phase:
                ax.set_title("Fitting Cubic Spline")
            elif final_rotation:
                ax.set_title("Final Local Principal Curve")
                
            # Plot the cubic spline if available
            if hasattr(self.lpc, 'cubic_splines') and self.lpc.cubic_splines:
                spline_points = self.lpc.cubic_splines[0]
                ax.plot(spline_points[:, 0], spline_points[:, 1], spline_points[:, 2], 
                       color=self.spline_color, linewidth=3)
        
        # Set axis labels and limits
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Save the frame
        plt.tight_layout()
        frame_path = os.path.join(self.output_dir, f"frame_{self.frame_count:04d}.png")
        plt.savefig(frame_path, dpi=150)
        plt.close(fig)
        
        self.frame_count += 1
    
    def capture_lpc_step(self, x, mu, gamma, weights):
        """
        Captures information for a single step of the LPC algorithm.
        Call this method at each step of the algorithm.
        """
        self.current_x = x.copy()
        self.current_mu = mu.copy()
        self.current_gamma = gamma.copy()
        self.current_weights = weights.copy()
        self.path_history.append(mu.copy())
        self.step_counter += 1
        
        # Create a frame for this step
        self._create_frame(x, mu, gamma, weights)
    
    def capture_spline_fitting(self):
        """Captures frames for the spline fitting phase"""
        # Create several frames with the spline being fit
        for _ in range(60):  # Create 60 frames with fixed angle
            self._create_frame(is_spline_phase=True)
    
    def capture_final_result(self):
        """Captures a full rotation of the final result"""
        # Create frames for a full rotation
        original_frame_count = self.frame_count
        for i in range(120):  # Create 120 frames for a full rotation
            self.frame_count = original_frame_count + i
            self._create_frame(final_rotation=True)
    
    def create_video(self, output_path="lpc_visualization.mp4", fps=30):
        """Creates a video from the saved frames"""
        with imageio.get_writer(output_path, fps=fps) as writer:
            for i in range(self.frame_count):
                frame_path = os.path.join(self.output_dir, f"frame_{i:04d}.png")
                if os.path.exists(frame_path):
                    writer.append_data(imageio.imread(frame_path))
        
        print(f"Video saved to {output_path}")

# Modified LocalPrincipalCurve that captures visualization data during execution
class VisualizedLocalPrincipalCurve:
    def __init__(self, bandwidth=0.5, max_iter=100, tol=1e-4, angle_penalty_exp=2, h=None,
                 output_dir="lpc_frames"):
        """
        Initialize the Local Principal Curve solver with visualization.
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
        
        # Visualization components
        self.visualizer = None

        # fixing axus stu
        self.fixed_axis_range = fixed_axis_range
        self.axis_padding = axis_padding
        self.manual_axis_range = manual_axis_range  # Format: {'x': [-5, 5], 'y': [-5, 5], 'z': [-5, 5]}


    def _kernel_weights(self, dataset, x):
        dists = np.linalg.norm(dataset - x, axis=1)
        weights = np.exp(-(dists**2) / (2 * self.bandwidth**2))
        w = weights / np.sum(weights)
        return w

    def _local_center_of_mass(self, dataset, x):
        w = self._kernel_weights(dataset, x)
        mu = np.sum(dataset.T * w, axis=1)
        return mu, w

    def _local_covariance(self, dataset, x, mu, w):
        centered = dataset - mu
        weighted_centered = centered * w[:, np.newaxis]
        cov = np.dot(weighted_centered.T, centered)
        return cov

    def _principal_component(self, cov, prev_vec=None):
        vals, vecs = np.linalg.eig(cov)
        idx = np.argsort(vals)[::-1]
        vecs = vecs[:, idx]

        gamma = vecs[:, 0]  # first principal component

        # Ensure gamma is real-valued (eigenvalues might be complex in some cases)
        gamma = np.real(gamma)

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
        x = x_start.copy()
        path_x = [x.copy()]
        prev_gamma = None

        for _ in range(self.max_iter):
            mu, weights = self._local_center_of_mass(dataset, x)
            cov = self._local_covariance(dataset, x, mu, weights)
            gamma = self._principal_component(cov, prev_vec=prev_gamma)

            # Capture this step for visualization
            if self.visualizer:
                self.visualizer.capture_lpc_step(x, mu, gamma, weights)

            x_new = mu + self.h * gamma

            if np.linalg.norm(mu - x) < self.tol:
                path_x.append(x_new.copy())
                break

            path_x.append(x_new.copy())
            x = x_new.copy()
            prev_gamma = gamma.copy()

        return np.array(path_x)

    def _backward_run(self, dataset, x0, gamma0):
        x = x0.copy()
        path_x = [x.copy()]
        prev_gamma = -gamma0.copy()

        for _ in range(self.max_iter):
            mu, weights = self._local_center_of_mass(dataset, x)
            cov = self._local_covariance(dataset, x, mu, weights)
            gamma = self._principal_component(cov, prev_vec=prev_gamma)

            # Capture this step for visualization
            if self.visualizer:
                self.visualizer.capture_lpc_step(x, mu, gamma, weights)

            x_new = mu + self.h * gamma

            if np.linalg.norm(mu - x) < self.tol:
                path_x.append(x_new.copy())
                break

            path_x.append(x_new.copy())
            x = x_new.copy()
            prev_gamma = gamma.copy()

        return np.array(path_x)

    def _find_starting_point(self, dataset, start_point):
        if start_point is None:
            idx = np.random.choice(len(dataset))
            return dataset[idx].copy(), idx
        else:
            diffs = dataset - start_point
            dists = np.linalg.norm(diffs, axis=1)
            min_idx = np.argmin(dists)
            closest_pt = dataset[min_idx].copy()
            return closest_pt, min_idx

    def fit(self, dataset, start_points=None, end_point=None, num_points=500, visualize=True, output_dir="lpc_frames"):
        """
        Fit LPC on the dataset with optional visualization.
        """
        dataset = np.array(dataset)
        self.paths = []
        self.initializations = []
        
        # Initialize the visualizer if requested
        if visualize:
            self.visualizer = LocalPrincipalCurveVisualizer(self, dataset, 
                                                            bandwidth=self.bandwidth,
                                                            h=self.h,
                                                            output_dir=output_dir)

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

            # Check orientation
            dist_start_to_first = np.linalg.norm(x0 - full_path[0])
            dist_start_to_last = np.linalg.norm(x0 - full_path[-1])
            if dist_start_to_last < dist_start_to_first:
                full_path = full_path[::-1]

            self.paths.append(full_path)
            self.initializations.append(x0.copy())

        # Fit splines and compute equal arc-length
        if self.visualizer:
            self.visualizer.capture_spline_fitting()
            
        self._fit_cubic_splines_eq()
        self._compute_equal_arc_length_spline_points(num_points=num_points)
        
        # Capture the final result with a full rotation
        if self.visualizer:
            self.visualizer.capture_final_result()
            self.visualizer.create_video()

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
        points = np.array([spline[dim](t_values) for dim in sorted(spline.keys())]).T
        return points

    def _compute_equal_arc_length_spline_points(self, num_points=500):
        self.cubic_splines = []
        for i, eq in enumerate(self.cubic_splines_eq):
            if eq is None:
                self.cubic_splines.append(None)
                continue
            
            # For simplicity in visualization, just use uniform t spacing
            # This could be replaced with more sophisticated arc length parameterization
            path = self.paths[i]
            t_values = np.linspace(0, len(path) - 1, num_points)
            spline_points = np.array([eq[dim](t_values) for dim in sorted(eq.keys())]).T
            self.cubic_splines.append(spline_points)

# Example usage
def generate_lpc_visualization(dataset, start_point=None, bandwidth=0.5, h=None, 
                              max_iter=100, output_dir="lpc_frames"):
    """
    Generates a visualization of the Local Principal Curve algorithm on the given dataset.
    
    Parameters:
    -----------
    dataset : numpy.ndarray
        Dataset to fit the principal curve to (shape [n_samples, n_dimensions])
    start_point : numpy.ndarray or None
        Optional starting point (defaults to randomly chosen point)
    bandwidth : float
        Bandwidth parameter for kernel weighting
    h : float or None
        Step size (defaults to bandwidth if None)
    max_iter : int
        Maximum number of iterations
    output_dir : str
        Directory to save visualization frames
        
    Returns:
    --------
    lpc : VisualizedLocalPrincipalCurve
        The fitted LPC instance
    """
    # Create and fit the LPC model with visualization
    lpc = VisualizedLocalPrincipalCurve(bandwidth=bandwidth, h=h, max_iter=max_iter, 
                                       output_dir=output_dir)
    
    # Fit the model to the data with visualization
    lpc.fit(dataset, start_points=start_point, visualize=True)
    
    return lpc

# Generate some example 3D data (a helical spiral)
def generate_helical_data(n_points=500, noise_level=0.1):
    t = np.linspace(0, 6*np.pi, n_points)
    x = np.cos(t)
    y = np.sin(t)
    z = t / (6*np.pi) * 5
    
    # Add some noise
    data = np.column_stack([x, y, z])
    noise = np.random.normal(0, noise_level, data.shape)
    data = data + noise
    
    return data

# Example of how to use it
if __name__ == "__main__":
    # Generate example data
    data = generate_helical_data(n_points=300, noise_level=0.1)

    morph_seq_data_path = "/net/trapnell/vol1/home/mdcolon/proj/fishcaster/data/embryo_morph_df.csv"
    
    # Run LPC with visualization
    lpc = generate_lpc_visualization(data, bandwidth=0.3, output_dir="/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20250309/data/lpc_helix_visualization")
    
    print("Visualization complete. Check the video file in the output directory.")
