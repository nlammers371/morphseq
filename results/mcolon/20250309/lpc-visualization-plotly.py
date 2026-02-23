import numpy as np
import os
import math
from tqdm import tqdm
import imageio.v2 as imageio
from scipy.interpolate import CubicSpline
import plotly.graph_objects as go
import plotly.io as pio

# ==============================================
# CONFIGURABLE PARAMETERS - EASY TO MODIFY
# ==============================================
class LPCParameters:
    """Main configuration class for LPC algorithm and visualization parameters"""
    def __init__(self):
        # ===== ALGORITHM PARAMETERS =====
        self.bandwidth = 0.5         # Bandwidth for kernel weighting
        self.step_size = None        # Step size (if None, defaults to bandwidth)
        self.max_iter = 100          # Maximum number of iterations
        self.tolerance = 1e-4        # Convergence tolerance
        self.angle_penalty_exp = 2   # Exponent for angle penalization
        
        # ===== VISUALIZATION PARAMETERS =====
        self.fps = 30                # Frames per second for video output
        self.spline_pause_frames = 60  # Number of frames to show during spline fitting
        self.final_rotation_frames = 120  # Number of frames for final rotation
        self.rotation_speed = 1      # Speed of rotation (lower = faster)
        
        # ===== OUTPUT PARAMETERS =====
        self.output_dir = "lpc_frames"  # Directory to save frames
        self.video_quality = 7      # Video quality (0-10)
        self.video_width = 1000      # Video width in pixels
        self.video_height = 800      # Video height in pixels

# Visual appearance configuration
class VisualizationConfig:
    """Configuration class for visualization appearance parameters"""
    def __init__(self, 
                 # Color settings (supports named colors from plotly)
                 base_point_color='lightgray',
                 base_point_opacity=0.5,
                 weighted_point_color='royalblue',
                 weighted_point_opacity=0.9,
                 center_color='red',
                 center_opacity=1.0,
                 eigenvector_color='green',
                 eigenvector_opacity=1.0,
                 path_color='purple',
                 path_opacity=0.8,
                 spline_color='orange',
                 spline_opacity=1.0,
                 current_point_color='black',
                 current_point_opacity=1.0,
                 
                 # Size settings
                 base_point_size=5,
                 weighted_point_size=8,
                 center_size=12,
                 current_point_size=10,
                 path_point_size=8,
                 eigenvector_scale=1.5,
                 arrow_scale=1.0,
                 arrow_head_size=0.8,
                 
                 # Camera settings
                 camera_zoom=2.5,
                 initial_angle=0,   # Angle in radians (0 = looking from +X axis)
                 camera_up=dict(x=0, y=0, z=1),
                 camera_center=dict(x=0, y=0, z=0),  # Point the camera looks at
                 
                 # Misc visualization settings
                 show_step_counter=True,
                 font_size=16,
                 font_family="Arial",
                 show_axes=True,
                 show_grid=False,
                 
                 #axis range paramaters
                 fixed_axis_range=True,   # Whether to use fixed axis ranges
                 axis_padding=0,        # Extra padding around data (as a ratio)
                 manual_axis_range=None,  # Manual override for axis ranges as dict):
                 aspectmode = "cube"
                ):
        
        # Store all parameters
        self.base_point_color = base_point_color
        self.base_point_opacity = base_point_opacity
        self.weighted_point_color = weighted_point_color
        self.weighted_point_opacity = weighted_point_opacity
        self.center_color = center_color
        self.center_opacity = center_opacity
        self.eigenvector_color = eigenvector_color
        self.eigenvector_opacity = eigenvector_opacity
        self.path_color = path_color
        self.path_opacity = path_opacity
        self.spline_color = spline_color
        self.spline_opacity = spline_opacity
        self.current_point_color = current_point_color
        self.current_point_opacity = current_point_opacity
        
        self.base_point_size = base_point_size
        self.weighted_point_size = weighted_point_size
        self.center_size = center_size
        self.current_point_size = current_point_size
        self.path_point_size = path_point_size
        self.eigenvector_scale = eigenvector_scale
        self.arrow_scale = arrow_scale
        self.arrow_head_size = arrow_head_size
        
        self.camera_zoom = camera_zoom
        self.initial_angle = initial_angle
        self.camera_up = camera_up
        self.camera_center = camera_center
        
        self.show_step_counter = show_step_counter
        self.font_size = font_size
        self.font_family = font_family
        self.show_axes = show_axes
        self.show_grid = show_grid
        # Store new axis parameters
        self.fixed_axis_range = fixed_axis_range
        self.axis_padding = axis_padding
        self.manual_axis_range = manual_axis_range  # Format: {'x': [-5, 5], 'y': [-5, 5], 'z': [-5, 5]}
        self.aspectmode = aspectmode


class LocalPrincipalCurveVisualizer:
    def __init__(self, lpc, dataset, config=None, parameters=None):
        """
        Initializes a visualizer for the Local Principal Curve algorithm.
        
        Parameters:
        -----------
        lpc : LocalPrincipalCurve instance
            The LPC algorithm instance to visualize
        dataset : numpy.ndarray
            The dataset being analyzed
        config : VisualizationConfig
            Configuration settings for visualization appearance
        parameters : LPCParameters
            Parameters for the visualization process
        """
        self.lpc = lpc
        self.dataset = dataset
        self.config = config if config is not None else VisualizationConfig()
        self.parameters = parameters if parameters is not None else LPCParameters()
        
        # Set output directory
        self.output_dir = self.parameters.output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.frame_count = 0
        
        # Camera parameters for 3D rotation
        self.angles = np.linspace(0, 2*np.pi, 360)
        # Set initial angle index based on configuration
        initial_angle_normalized = self.config.initial_angle % (2*np.pi)
        self.current_angle_idx = int((initial_angle_normalized / (2*np.pi)) * 360) % 360
        
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
        Creates a frame showing the current state of the algorithm using Plotly.
        
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
        # Initialize the Plotly figure
        fig = go.Figure()
        
        # Set viewing angle for rotation
        if final_rotation:
            # Do a full rotation
            angle_idx = self.frame_count % len(self.angles)
        elif is_spline_phase:
            # Keep the angle fixed during spline phase
            angle_idx = self.current_angle_idx
        else:
            # Slowly rotate during calculation
            if self.frame_count % self.parameters.rotation_speed == 0:
                self.current_angle_idx = (self.current_angle_idx + 1) % len(self.angles)
            angle_idx = self.current_angle_idx
        
        angle = self.angles[angle_idx]
        
        # Calculate camera position
        camera_x = self.config.camera_zoom * math.cos(angle)
        camera_y = self.config.camera_zoom * math.sin(angle)
        camera_z = self.config.camera_zoom * 0.5  # Slightly elevated view
        
        # Plot the dataset with base transparency
        fig.add_trace(go.Scatter3d(
            x=self.dataset[:, 0],
            y=self.dataset[:, 1],
            z=self.dataset[:, 2],
            mode='markers',
            marker=dict(
                size=self.config.base_point_size,
                color=self.config.base_point_color,
                opacity=self.config.base_point_opacity
            ),
            name='Dataset',
            showlegend=True
        ))
        
        # Plot path history (deposited points)
        if self.path_history:
            path_points = np.array(self.path_history)
            fig.add_trace(go.Scatter3d(
                x=path_points[:, 0],
                y=path_points[:, 1],
                z=path_points[:, 2],
                mode='markers',
                marker=dict(
                    size=self.config.path_point_size,
                    color=self.config.path_color,
                    opacity=self.config.path_opacity
                ),
                name='Path History',
                showlegend=True
            ))
        
        # If in active calculation phase, show current step details
        if not is_spline_phase and not final_rotation and x is not None:
            title_text = f"Local Principal Curve - Step: {self.step_counter}" if self.config.show_step_counter else "Local Principal Curve"
            
            # Plot weighted points (with color intensity based on weight)
            if weights is not None:
                # Only show points with significant weight for clarity
                significant_indices = np.where(weights > 0.001)[0]
                if len(significant_indices) > 0:
                    # Normalize weights for visualization
                    significant_weights = weights[significant_indices]
                    normalized_weights = significant_weights / np.max(significant_weights)
                    
                    # Get points with significant weights
                    significant_points = self.dataset[significant_indices]
                    
                    # Group points into bins by weight for different opacities
                    # Define number of opacity levels 
                    num_opacity_levels = 5
                    
                    # Create bins of weights
                    min_weight = np.min(normalized_weights)
                    max_weight = np.max(normalized_weights)
                    weight_range = max_weight - min_weight
                    
                    # Add points in groups with increasing opacity based on weight
                    for i in range(num_opacity_levels):
                        lower_bound = min_weight + (i / num_opacity_levels) * weight_range
                        upper_bound = min_weight + ((i + 1) / num_opacity_levels) * weight_range
                        
                        # Find points in this weight range
                        if i < num_opacity_levels - 1:
                            bin_indices = np.where((normalized_weights >= lower_bound) & 
                                                  (normalized_weights < upper_bound))[0]
                        else:
                            # Include the upper bound in the last bin
                            bin_indices = np.where((normalized_weights >= lower_bound) & 
                                                  (normalized_weights <= upper_bound))[0]
                        
                        if len(bin_indices) > 0:
                            bin_points = significant_points[bin_indices]
                            bin_weights = normalized_weights[bin_indices]
                            
                            # Opacity proportional to weight bin
                            bin_opacity = 0.2 + 0.8 * (i / (num_opacity_levels - 1))
                            
                            # Add a single colorbar for the first bin only
                            colorbar_dict = dict(title='Point Influence') if i == 0 else None
                            
                            fig.add_trace(go.Scatter3d(
                                x=bin_points[:, 0],
                                y=bin_points[:, 1],
                                z=bin_points[:, 2],
                                mode='markers',
                                marker=dict(
                                    size=self.config.weighted_point_size,
                                    color=bin_weights,
                                    colorscale='Viridis',
                                    opacity=bin_opacity,
                                    cmin=0,
                                    cmax=1,
                                    colorbar=colorbar_dict
                                ),
                                name='Weighted Points' if i == 0 else None,  # Only add one legend entry
                                showlegend=False  # Only show in legend once
                            ))
            
            # Plot current point
            fig.add_trace(go.Scatter3d(
                x=[x[0]],
                y=[x[1]],
                z=[x[2]],
                mode='markers',
                marker=dict(
                    size=self.config.current_point_size,
                    color=self.config.current_point_color,
                    opacity=self.config.current_point_opacity,
                    symbol='circle'
                ),
                name='Current Point',
                showlegend=True
            ))
            
            # Plot local center of mass
            if mu is not None:
                fig.add_trace(go.Scatter3d(
                    x=[mu[0]],
                    y=[mu[1]],
                    z=[mu[2]],
                    mode='markers',
                    marker=dict(
                        size=self.config.center_size,
                        color=self.config.center_color,
                        opacity=self.config.center_opacity,
                        symbol='diamond'  # Changed from 'star' to 'diamond' which is supported
                    ),
                    name='Center of Mass',
                    showlegend=True
                ))
            
            # Plot eigenvector as an arrow
            if gamma is not None and mu is not None:
                arrow_length = self.lpc.h * self.config.eigenvector_scale
                arrow_end = mu + arrow_length * gamma
                
                # Create the arrow shaft
                fig.add_trace(go.Scatter3d(
                    x=[mu[0], arrow_end[0]],
                    y=[mu[1], arrow_end[1]],
                    z=[mu[2], arrow_end[2]],
                    mode='lines',
                    line=dict(
                        color=self.config.eigenvector_color,
                        width=5 * self.config.arrow_scale
                    ),
                    opacity=self.config.eigenvector_opacity,  # Move opacity to trace level
                    name='Eigenvector',
                    showlegend=True
                ))
                
                # Create arrow head using a cone
                head_size = self.config.arrow_head_size
                u, v, w = gamma  # Direction vector
                fig.add_trace(go.Cone(
                    x=[arrow_end[0] - 0.05 * u],  # Offset slightly so cone sits at arrow tip
                    y=[arrow_end[1] - 0.05 * v],
                    z=[arrow_end[2] - 0.05 * w],
                    u=[u * head_size],
                    v=[v * head_size],
                    w=[w * head_size],
                    colorscale=[[0, self.config.eigenvector_color], [1, self.config.eigenvector_color]],
                    showscale=False,
                    opacity=self.config.eigenvector_opacity,
                    name='Arrow Head',
                    showlegend=False
                ))
        else:
            if is_spline_phase:
                title_text = "Fitting Cubic Spline"
            elif final_rotation:
                title_text = "Final Local Principal Curve"
            else:
                title_text = "Local Principal Curve"
                
            # Plot the cubic spline if available
            if hasattr(self.lpc, 'cubic_splines') and self.lpc.cubic_splines:
                spline_points = self.lpc.cubic_splines[0]
                fig.add_trace(go.Scatter3d(
                    x=spline_points[:, 0],
                    y=spline_points[:, 1],
                    z=spline_points[:, 2],
                    mode='lines',
                    line=dict(
                        color=self.config.spline_color,
                        width=6
                    ),
                    opacity=self.config.spline_opacity,  # Moved to trace level
                    name='Cubic Spline',
                    showlegend=True
                ))
        
        # Set up layout
        if self.config.fixed_axis_range:
            # Calculate axis ranges if not done yet (only needed once)
            if not hasattr(self, 'axis_ranges'):
                self._calculate_axis_ranges()
            
            # Get the fixed ranges
            x_range = self.axis_ranges['x']
            y_range = self.axis_ranges['y']
            z_range = self.axis_ranges['z']
        else:
            # Let Plotly auto-scale axes
            x_range = None
            y_range = None
            z_range = None
            
        # Set up layout with fixed ranges if needed
        fig.update_layout(
            title=dict(
                text=title_text,
                font=dict(
                    family=self.config.font_family,
                    size=self.config.font_size
                )
            ),
            scene=dict(
                xaxis=dict(
                    showticklabels=self.config.show_axes,
                    showgrid=self.config.show_grid,
                    title='X',
                    range=x_range,  # Apply fixed range or None for auto
                ),
                yaxis=dict(
                    showticklabels=self.config.show_axes,
                    showgrid=self.config.show_grid,
                    title='Y',
                    range=y_range,  # Apply fixed range or None for auto
                ),
                zaxis=dict(
                    showticklabels=self.config.show_axes,
                    showgrid=self.config.show_grid,
                    title='Z',
                    range=z_range,  # Apply fixed range or None for auto
                ),
                camera=dict(
                    eye=dict(x=camera_x, y=camera_y, z=camera_z),
                    up=self.config.camera_up,
                    center=self.config.camera_center
                ),
                aspectmode=self.config.aspectmode 
            ),
            
            width=self.parameters.video_width,
            height=self.parameters.video_height,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                font=dict(
                    family=self.config.font_family,
                    size=self.config.font_size - 2
                )
            ),
            margin=dict(l=0, r=0, b=0, t=50),
        )
        
        # Save the frame as an image
        frame_path = os.path.join(self.output_dir, f"frame_{self.frame_count:04d}.png")
        pio.write_image(fig, frame_path, scale=1, width=self.parameters.video_width, height=self.parameters.video_height)
        
        self.frame_count += 1
    # Add a new method to calculate axis ranges based on dataset
    def _calculate_axis_ranges(self):
        """Calculate fixed axis ranges with padding to prevent wobbling"""
        # Use manual ranges if provided
        if self.config.manual_axis_range is not None:
            self.axis_ranges = self.config.manual_axis_range
            return
        
        # Calculate ranges from data with padding
        padding = self.config.axis_padding
        
        # Find min/max values across all dimensions
        min_vals = np.min(self.dataset, axis=0)
        max_vals = np.max(self.dataset, axis=0)
        
        # Calculate range and add padding
        ranges = max_vals - min_vals
        padded_min = min_vals - (ranges * padding)
        padded_max = max_vals + (ranges * padding)
        
        # Store ranges for each axis
        self.axis_ranges = {
            'x': [padded_min[0], padded_max[0]],
            'y': [padded_min[1], padded_max[1]],
            'z': [padded_min[2], padded_max[2]]
        }
        
        # Ensure we have a cube by making all dimensions equal
        if self.config.aspectmode == 'cube':
            # Find the largest range
            all_ranges = [self.axis_ranges[axis][1] - self.axis_ranges[axis][0] 
                        for axis in ['x', 'y', 'z']]
            max_range = max(all_ranges)
            
            # Adjust all axes to have the same range
            for axis in ['x', 'y', 'z']:
                current_center = (self.axis_ranges[axis][0] + self.axis_ranges[axis][1]) / 2
                self.axis_ranges[axis] = [
                    current_center - max_range/2,
                    current_center + max_range/2
                ]

                
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
        """
        Captures frames for the spline fitting phase
        """
        # Create several frames with the spline being fit
        for _ in range(self.parameters.spline_pause_frames):
            self._create_frame(is_spline_phase=True)
    
    def capture_final_result(self):
        """
        Captures a full rotation of the final result
        """
        # Create frames for a full rotation
        original_frame_count = self.frame_count
        for i in range(self.parameters.final_rotation_frames):
            self.frame_count = original_frame_count + i
            self._create_frame(final_rotation=True)
    
    def create_video(self, output_path=None):
        """
        Creates a video from the saved frames
        
        Parameters:
        -----------
        output_path : str or None
            Path to save the video file. If None, it will be saved in the output directory
        """
        # If no specific output path is provided, save in the output directory
        if output_path is None:
            output_path = os.path.join(self.output_dir, "lpc_visualization.mp4")
        
        try:
            # Create writer with explicitly specified format
            writer = imageio.get_writer(output_path, fps=self.parameters.fps, format='FFMPEG', 
                                        mode='I', codec='libx264', quality=self.parameters.video_quality)
            
            print(f"Creating video from {self.frame_count} frames...")
            for i in tqdm(range(self.frame_count)):
                frame_path = os.path.join(self.output_dir, f"frame_{i:04d}.png")
                if os.path.exists(frame_path):
                    frame = imageio.imread(frame_path)
                    writer.append_data(frame)
            writer.close()
            
            print(f"Video saved to {output_path}")
            return output_path
        except Exception as e:
            print(f"Error creating video: {e}")
            print("If you're having issues with FFMPEG, you can try using create_gif() instead.")

    def create_gif(self, output_path=None):
        """
        Creates a GIF from the saved frames (alternative to MP4)
        
        Parameters:
        -----------
        output_path : str or None
            Path to save the GIF file. If None, it will be saved in the output directory
        """
        import glob
        from PIL import Image
        
        # If no specific output path is provided, save in the output directory
        if output_path is None:
            output_path = os.path.join(self.output_dir, "lpc_visualization.gif")
        
        # Get all the frame files in order
        frame_paths = sorted(glob.glob(os.path.join(self.output_dir, "frame_*.png")))
        
        if not frame_paths:
            print(f"No frames found in {self.output_dir}")
            return
        
        # Load all frames
        frames = [Image.open(frame) for frame in frame_paths]
        
        # Calculate duration between frames in milliseconds
        duration = int(1000 / self.parameters.fps)
        
        # Save as GIF
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=duration,
            loop=0  # 0 means loop forever
        )
        
        print(f"GIF saved to {output_path}")

# Modified LocalPrincipalCurve that captures visualization data during execution
class VisualizedLocalPrincipalCurve:
    def __init__(self, parameters=None):
        """
        Initialize the Local Principal Curve solver with visualization.
        
        Parameters:
        -----------
        parameters : LPCParameters
            Configuration parameters for the algorithm and visualization
        """
        # Use default parameters if none provided
        self.parameters = parameters if parameters is not None else LPCParameters()
        
        # Set algorithmic parameters
        self.bandwidth = self.parameters.bandwidth
        self.h = self.parameters.step_size if self.parameters.step_size is not None else self.bandwidth
        self.max_iter = self.parameters.max_iter
        self.tol = self.parameters.tolerance
        self.angle_penalty_exp = self.parameters.angle_penalty_exp

        self.initializations = []
        self.paths = []
        self.cubic_splines_eq = []
        self.cubic_splines = []
        
        # Visualization components
        self.visualizer = None

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

    def fit(self, dataset, start_points=None, end_point=None, num_points=500, 
            visualize=True, vis_config=None):
        """
        Fit LPC on the dataset with optional visualization.
        
        Parameters:
        -----------
        dataset : numpy.ndarray
            Dataset to fit the curve to
        start_points : list or numpy.ndarray or None
            Optional starting points
        end_point : numpy.ndarray or None
            Optional end point constraint
        num_points : int
            Number of points for the cubic spline
        visualize : bool
            Whether to generate visualization
        vis_config : VisualizationConfig or None
            Visualization appearance configuration
        """
        dataset = np.array(dataset)
        self.paths = []
        self.initializations = []
        
        # Initialize the visualizer if requested
        if visualize:
            # Ensure output directory exists
            os.makedirs(self.parameters.output_dir, exist_ok=True)
            
            self.visualizer = LocalPrincipalCurveVisualizer(
                self, 
                dataset, 
                config=vis_config,
                parameters=self.parameters
            )

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
        self._fit_cubic_splines_eq()
        self._compute_equal_arc_length_spline_points(num_points=num_points)
        
        # Capture visualization of spline fitting and final result
        if self.visualizer:
            self.visualizer.capture_spline_fitting()
            self.visualizer.capture_final_result()
            
            # Generate frames but don't create video automatically to avoid freezing
            print(f"Visualization complete. {self.visualizer.frame_count} frames saved in {self.visualizer.output_dir}")
            print("To create a video from these frames, manually run:")
            print(f"  lpc.visualizer.create_video()")

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

# Helper function to generate the visualization
def generate_lpc_visualization(dataset, start_point=None, parameters=None, vis_config=None):
    """
    Generates a visualization of the Local Principal Curve algorithm on the given dataset.
    
    Parameters:
    -----------
    dataset : numpy.ndarray
        Dataset to fit the principal curve to (shape [n_samples, n_dimensions])
    start_point : numpy.ndarray or None
        Optional starting point (defaults to randomly chosen point)
    parameters : LPCParameters or None
        Configuration parameters for the algorithm and visualization (uses defaults if None)
    vis_config : VisualizationConfig or None
        Configuration for visualization appearance (uses defaults if None)
        
    Returns:
    --------
    lpc : VisualizedLocalPrincipalCurve
        The fitted LPC instance
    """
    # Use default parameters if none provided
    if parameters is None:
        parameters = LPCParameters()
    
    # Create and fit the LPC model with visualization
    lpc = VisualizedLocalPrincipalCurve(parameters=parameters)
    
    # Make sure the output directory exists
    os.makedirs(parameters.output_dir, exist_ok=True)
    
    # Fit the model to the data with visualization
    lpc.fit(dataset, start_points=start_point, visualize=True, vis_config=vis_config)
    
    return lpc

# Generate some example 3D data (a helical spiral)
def generate_helical_data(n_points=1000, noise_level=0.2, n_loops=3):
    t = np.linspace(0, 2*np.pi*n_loops, n_points)
    x = np.cos(t)
    y = np.sin(t)
    z = t / (2*np.pi*n_loops) * 5
    
    # Add some noise
    data = np.column_stack([x, y, z])
    noise = np.random.normal(0, noise_level, data.shape)
    data = data + noise
    
    return data
# Example usage showing different configuration options
if __name__ == "__main__":
    # # Generate example data
    data = generate_helical_data(n_points=1000, noise_level=0.2, n_loops=3)
    
    # # Example 1: Default configuration
    # params1 = LPCParameters()
    # params1.fps = 20
    # params1.spline_pause_frames = 40
    # params1.final_rotation_frames = 120
    # params1.max_iter =100
    # params1.bandwidth = 0.3
    # params1.output_dir = "./data/lpc_viz_default"
    
    # lpc1 = generate_lpc_visualization(data, parameters=params1)
    
    # print("\nVisualization 1 complete. To create a video, run:")
    # print("lpc1.visualizer.create_video()")
    # video_path = lpc1.visualizer.create_video()
    # print(f"Video created at: {video_path}")
    
    # # Example 2: Custom configuration with ALL available parameters
    # # This comprehensive example shows all customizable parameters with explanations
    # params2 = LPCParameters()
    # params2.bandwidth = 0.3
    # params2.output_dir = "./data/lpc_viz_custom_all_params"
    # # Configure visualization process parameters
    # params2.fps = 30
    # params2.spline_pause_frames = 60
    # params2.final_rotation_frames = 120
    # params2.rotation_speed = 2
    # params2.video_width = 1200
    # params2.video_height = 900
    # params2.video_quality = 10
    
    # # Visual appearance configuration
    # custom_config = VisualizationConfig(
    #     # ===== COLOR SETTINGS =====
    #     # Colors can be specified as named colors (e.g., 'red', 'blue') or hex values (e.g., '#FF5733')
        
    #     # Base point colors (all data points in the background)
    #     base_point_color='lightgray',     # Color of all data points in the background
    #     base_point_opacity=0.3,           # 0-1: Lower values make points more transparent
        
    #     # Weighted point colors (points that influence the current calculation)
    #     weighted_point_color='dodgerblue', # Color of points being used in current calculation
    #     weighted_point_opacity=0.9,       # 0-1: Higher values make points more visible
        
    #     # Center of mass point
    #     center_color='crimson',           # Color of the center of mass point
    #     center_opacity=1.0,               # 0-1: Usually kept fully opaque
        
    #     # Current point being processed
    #     current_point_color='black',      # Color of the current point in the algorithm
    #     current_point_opacity=1.0,        # 0-1: Usually kept fully opaque
        
    #     # Eigenvector arrow
    #     eigenvector_color='green',        # Color of the eigenvector arrow
    #     eigenvector_opacity=1.0,          # 0-1: Usually kept fully opaque
        
    #     # Path history (deposited points)
    #     path_color='purple',              # Color of previously calculated centers of mass
    #     path_opacity=0.8,                 # 0-1: How visible the path history is
        
    #     # Spline curve
    #     spline_color='darkorange',        # Color of the final spline curve
    #     spline_opacity=1.0,               # 0-1: Usually kept fully opaque
        
    #     # ===== SIZE SETTINGS =====
    #     # All size values are in pixels or relative units
        
    #     # Point sizes
    #     base_point_size=3,                # Size of background data points (smaller = less cluttered)
    #     weighted_point_size=8,            # Size of points being used in calculation
    #     center_size=12,                   # Size of center of mass point
    #     current_point_size=10,            # Size of current point being processed
    #     path_point_size=8,                # Size of deposited path points
        
    #     # Eigenvector arrow settings
    #     eigenvector_scale=2.0,            # Length of eigenvector (higher = longer arrow)
    #     arrow_scale=1.5,                  # Width of arrow (higher = thicker arrow)
    #     arrow_head_size=1.0,              # Size of arrow head (higher = larger head)
        
    #     # ===== CAMERA SETTINGS =====
    #     camera_zoom=2.0,                  # Camera distance (higher = more zoomed out)
    #     initial_angle=np.pi/4,            # Starting camera angle in radians (0=+X axis, π/2=+Y axis)
    #     camera_up=dict(x=0, y=0, z=1),    # Camera up direction (usually keep z=1)
    #     camera_center=dict(x=0, y=0, z=2.5), # Where camera looks at (center of your data)
        
    #     # ===== MISC VISUALIZATION SETTINGS =====
    #     show_step_counter=True,           # Whether to show step counter in the title
    #     font_size=18,                     # Font size for title and labels
    #     font_family="Arial",              # Font family for text
    #     show_axes=True,                   # Whether to show axis labels and ticks
    #     show_grid=True                    # Whether to show gridlines (helpful for spatial reference)
    # )
    
    # lpc2 = generate_lpc_visualization(
    #     data, 
    #     parameters=params2,
    #     vis_config=custom_config
    # )
    # video_path = lpc2.visualizer.create_video()
    # print(f"Video created at: {video_path}")
    
    # Example 3: High-contrast visualization for presentations
    params3 = LPCParameters()
    params3.bandwidth = 0.3
    params3.output_dir = "./data/lpc_viz_presentation_5"
    # Configure video settings for presentations
    params3.fps = 5
    params3.spline_pause_frames = 20
    params3.final_rotation_frames = 30
    params3.max_iter = 50
    # params3.video_width = 1920
    # params3.video_height = 1080
    params3.video_quality = 10
    
    presentation_config = VisualizationConfig(
        # Dark background effect with high contrast
        base_point_color='navy',
        base_point_opacity=0.1,
        weighted_point_color='deepskyblue',
        weighted_point_opacity=1.0,
        center_color='deeppink',
        path_color='hotpink',
        eigenvector_color='blue',
        spline_color='dimgrey',
        
        # Larger elements for visibility in presentations
        base_point_size=3,
        weighted_point_size=5,
        center_size=8,
        path_point_size=7,
        eigenvector_scale=3,
        arrow_scale=2.0,
        
        # Show grid for better spatial reference
        show_grid=False,
        font_size=22,  # Larger text for readability
        camera_zoom=2 
    )
    
    lpc3 = generate_lpc_visualization(
        data, 
        vis_config=presentation_config
    )
    video_path = lpc3.visualizer.create_video()
    print(f"Video created at: {video_path}")
    
    print("\nAll visualizations complete!")
    print("To view the generated frames, check these directories:")
    print(f"1. Default visualization: {params1.output_dir}")
    print(f"2. Custom visualization: {params2.output_dir}")
    print(f"3. Presentation visualization: {params3.output_dir}")