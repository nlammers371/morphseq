"""
Extract and analyze mask for embryo 20251017_part2_D06_ch00_t0022
Using PCA-based slicing with Gaussian smoothing (sigma=10)

This script focuses on the PCA slicing method with smoothing for clean curvature profiles.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter
from skimage import io
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class PCASlicingAnalyzer:
    """Analyzes embryo curvature using PCA-based slicing with Gaussian smoothing."""

    def __init__(self, mask: np.ndarray, sigma: float = 10.0):
        """
        Initialize with a binary mask.

        Args:
            mask: Binary mask as numpy array
            sigma: Gaussian smoothing sigma for the mask
        """
        self.original_mask = mask.astype(np.uint8)
        
        # Apply Gaussian smoothing to the mask
        print(f"Applying Gaussian smoothing with sigma={sigma}...")
        smoothed = gaussian_filter(mask.astype(float), sigma=sigma)
        
        # Re-threshold to get binary mask
        self.mask = (smoothed > 0.5).astype(np.uint8)
        self.sigma = sigma
        
        print(f"Original mask area: {self.original_mask.sum()} pixels")
        print(f"Smoothed mask area: {self.mask.sum()} pixels")

    def extract_centerline_pca(self, n_slices: int = 100):
        """
        Extract centerline using PCA-based slicing.
        Most robust for elongated embryos.

        Args:
            n_slices: Number of slices perpendicular to principal axis

        Returns:
            principal_axis: Principal axis vector
            centerline_points: Ordered (x, y) coordinates of centroids
            center: Center of mass
        """
        # Get all mask coordinates
        y_coords, x_coords = np.where(self.mask)
        points = np.column_stack([x_coords, y_coords])

        # Perform PCA to find principal axis
        pca = PCA(n_components=2)
        pca.fit(points)

        # Principal axis (first component)
        principal_axis = pca.components_[0]
        center = points.mean(axis=0)

        # Project points onto principal axis
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

        return principal_axis, centerline_points, center

    def compute_curvature(self, centerline_points: np.ndarray, smoothing: float = 0.01):
        """
        Compute curvature along the centerline.

        Args:
            centerline_points: (N, 2) array of (x, y) coordinates
            smoothing: Smoothing parameter for spline fitting

        Returns:
            arc_length: Arc length parameter
            curvature: Curvature at each point
            spline_points: Smoothed spline points (x, y)
        """
        if len(centerline_points) < 4:
            return np.array([]), np.array([]), np.array([])

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

        # Spline points for visualization
        spline_points = np.column_stack([x_vals, y_vals])

        return arc_length, curvature, spline_points


def main():
    # Define paths
    image_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/sam2_pipeline_files/raw_data_organized/20251017_part2/images/20251017_part2_D06/20251017_part2_D06_ch00_t0022.jpg")
    mask_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/sam2_pipeline_files/exported_masks/20251017_part2/masks/20251017_part2_D06_ch00_t0022_masks_emnum_1.png")
    
    print("="*70)
    print("Embryo D06 t=0022 - PCA Slicing with Gaussian Smoothing (σ=10)")
    print("="*70)
    
    print(f"\nImage path: {image_path}")
    print(f"Image exists: {image_path.exists()}")
    print(f"\nMask path: {mask_path}")
    print(f"Mask exists: {mask_path.exists()}")
    
    if not mask_path.exists() or not image_path.exists():
        print("\nERROR: Files not found!")
        return
    
    # Load the original image
    print("\nLoading image...")
    image = io.imread(image_path)
    print(f"Image shape: {image.shape}")
    
    # Load the mask
    print("\nLoading mask...")
    mask = io.imread(mask_path)
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    binary_mask = (mask > 0).astype(np.uint8)
    print(f"Mask shape: {binary_mask.shape}")
    print(f"Mask area: {binary_mask.sum()} pixels")
    
    # Create organized output directory
    output_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251022/plots/embryo_D06_t0022_pca_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Create analyzer with Gaussian smoothing
    print("\n" + "="*70)
    print("Running PCA-based centerline extraction with Gaussian smoothing")
    print("="*70)
    
    sigma = 10.0
    analyzer = PCASlicingAnalyzer(binary_mask, sigma=sigma)
    
    # Extract centerline
    principal_axis, centerline_points, center = analyzer.extract_centerline_pca(n_slices=100)
    print(f"\nExtracted {len(centerline_points)} centerline points")
    print(f"Principal axis: {principal_axis}")
    print(f"Center of mass: {center}")
    
    # Compute curvature
    print("\nComputing curvature...")
    arc_length, curvature, spline_points = analyzer.compute_curvature(centerline_points, smoothing=0.01)
    print(f"Arc length range: {arc_length.min():.1f} to {arc_length.max():.1f} pixels")
    print(f"Curvature range: {curvature.min():.6f} to {curvature.max():.6f} (1/pixels)")
    print(f"Mean curvature: {curvature.mean():.6f}")
    print(f"Max curvature: {curvature.max():.6f}")
    
    # Create comprehensive visualization
    print("\nCreating visualizations...")
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Row 1: Original image, smoothed mask, overlay
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(analyzer.original_mask, cmap='gray', alpha=0.5)
    ax2.imshow(analyzer.mask, cmap='Reds', alpha=0.5)
    ax2.set_title(f'Original (gray) + Smoothed (red)\nσ={sigma}', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(image, cmap='gray')
    ax3.imshow(analyzer.mask, cmap='Reds', alpha=0.3)
    ax3.set_title('Image + Smoothed Mask Overlay', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Row 2: Centerline extraction details
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(analyzer.mask, cmap='gray', alpha=0.7)
    ax4.plot(centerline_points[:, 0], centerline_points[:, 1], 'b.', markersize=2, alpha=0.5, label='Raw centerline points')
    ax4.scatter(centerline_points[0, 0], centerline_points[0, 1], c='lime', s=100, marker='o', label='Start', zorder=5, edgecolors='black', linewidths=2)
    ax4.scatter(centerline_points[-1, 0], centerline_points[-1, 1], c='cyan', s=100, marker='s', label='End', zorder=5, edgecolors='black', linewidths=2)
    # Draw principal axis
    scale = 300
    ax4.arrow(center[0], center[1], 
             principal_axis[0] * scale, principal_axis[1] * scale,
             head_width=30, head_length=40, fc='yellow', ec='orange',
             alpha=0.8, linewidth=2, label='Principal Axis')
    ax4.set_title('PCA-based Centerline\n(Raw Points)', fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=8)
    ax4.axis('equal')
    ax4.invert_yaxis()
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(analyzer.mask, cmap='gray', alpha=0.7)
    ax5.plot(spline_points[:, 0], spline_points[:, 1], 'r-', linewidth=2, label='Smoothed spline')
    ax5.scatter(spline_points[0, 0], spline_points[0, 1], c='lime', s=100, marker='o', label='Start', zorder=5, edgecolors='black', linewidths=2)
    ax5.scatter(spline_points[-1, 0], spline_points[-1, 1], c='cyan', s=100, marker='s', label='End', zorder=5, edgecolors='black', linewidths=2)
    ax5.set_title('Smoothed Spline\n(Curvature Calculation)', fontsize=12, fontweight='bold')
    ax5.legend(loc='best', fontsize=8)
    ax5.axis('equal')
    ax5.invert_yaxis()
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(image, cmap='gray')
    ax6.plot(spline_points[:, 0], spline_points[:, 1], 'cyan', linewidth=3, label='Centerline', alpha=0.8)
    ax6.scatter(spline_points[0, 0], spline_points[0, 1], c='lime', s=150, marker='o', label='Start', zorder=5, edgecolors='white', linewidths=2)
    ax6.scatter(spline_points[-1, 0], spline_points[-1, 1], c='magenta', s=150, marker='s', label='End', zorder=5, edgecolors='white', linewidths=2)
    ax6.set_title('Image + Centerline', fontsize=12, fontweight='bold')
    ax6.legend(loc='best', fontsize=8)
    ax6.axis('off')
    
    # Row 3: Curvature analysis (span all columns)
    ax7 = fig.add_subplot(gs[2, :])
    ax7.plot(arc_length, curvature, 'b-', linewidth=2.5, label='Curvature profile')
    ax7.fill_between(arc_length, 0, curvature, alpha=0.3)
    ax7.axhline(y=curvature.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {curvature.mean():.6f}')
    ax7.set_xlabel('Arc Length (pixels)', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Curvature (1/pixels)', fontsize=12, fontweight='bold')
    ax7.set_title('Curvature Profile Along Embryo Centerline', fontsize=14, fontweight='bold')
    ax7.grid(True, alpha=0.3, linestyle='--')
    ax7.legend(fontsize=10, loc='upper right')
    
    # Add statistics box
    stats_text = (
        f'Statistics:\n'
        f'Mean: {curvature.mean():.6f}\n'
        f'Std: {curvature.std():.6f}\n'
        f'Max: {curvature.max():.6f}\n'
        f'Min: {curvature.min():.6f}\n'
        f'Total arc length: {arc_length[-1]:.1f} px'
    )
    ax7.text(0.02, 0.98, stats_text, transform=ax7.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(f'Embryo 20251017_part2_D06 at t=0022\nPCA Slicing with Gaussian Smoothing (σ={sigma})',
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save main figure
    main_fig_path = output_dir / "embryo_D06_t0022_pca_smoothed_analysis.png"
    plt.savefig(main_fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved main figure to: {main_fig_path}")
    plt.close()
    
    # Save data files
    print("\nSaving data files...")
    
    # Save masks
    mask_original_path = output_dir / "embryo_D06_t0022_mask_original.png"
    mask_smoothed_path = output_dir / "embryo_D06_t0022_mask_smoothed.png"
    io.imsave(mask_original_path, (analyzer.original_mask * 255).astype(np.uint8))
    io.imsave(mask_smoothed_path, (analyzer.mask * 255).astype(np.uint8))
    print(f"  - Original mask: {mask_original_path}")
    print(f"  - Smoothed mask: {mask_smoothed_path}")
    
    # Save centerline points
    centerline_path = output_dir / "embryo_D06_t0022_centerline_pca.npy"
    np.save(centerline_path, centerline_points)
    print(f"  - Centerline points: {centerline_path}")
    
    # Save spline points
    spline_path = output_dir / "embryo_D06_t0022_spline_points.npy"
    np.save(spline_path, spline_points)
    print(f"  - Spline points: {spline_path}")
    
    # Save curvature data as CSV
    curvature_df = pd.DataFrame({
        'arc_length_pixels': arc_length,
        'curvature_per_pixel': curvature
    })
    curvature_csv_path = output_dir / "embryo_D06_t0022_curvature_profile.csv"
    curvature_df.to_csv(curvature_csv_path, index=False)
    print(f"  - Curvature profile: {curvature_csv_path}")
    
    # Save metadata
    metadata = {
        'embryo_id': '20251017_part2_D06',
        'timepoint': 't0022',
        'image_path': str(image_path),
        'mask_path': str(mask_path),
        'gaussian_sigma': sigma,
        'n_slices': 100,
        'spline_smoothing': 0.01,
        'mask_area_original_pixels': int(analyzer.original_mask.sum()),
        'mask_area_smoothed_pixels': int(analyzer.mask.sum()),
        'centerline_points': len(centerline_points),
        'arc_length_total_pixels': float(arc_length[-1]),
        'curvature_mean': float(curvature.mean()),
        'curvature_std': float(curvature.std()),
        'curvature_max': float(curvature.max()),
        'curvature_min': float(curvature.min()),
        'principal_axis_x': float(principal_axis[0]),
        'principal_axis_y': float(principal_axis[1]),
        'center_x': float(center[0]),
        'center_y': float(center[1])
    }
    metadata_path = output_dir / "embryo_D06_t0022_metadata.json"
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  - Metadata: {metadata_path}")
    
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE!")
    print("="*70)
    print(f"\nAll files saved to: {output_dir}")
    print("\nSummary:")
    print(f"  - Gaussian smoothing sigma: {sigma}")
    print(f"  - Original mask area: {analyzer.original_mask.sum()} pixels")
    print(f"  - Smoothed mask area: {analyzer.mask.sum()} pixels")
    print(f"  - Centerline points extracted: {len(centerline_points)}")
    print(f"  - Total arc length: {arc_length[-1]:.1f} pixels")
    print(f"  - Mean curvature: {curvature.mean():.6f} (1/pixels)")
    print(f"  - Max curvature: {curvature.max():.6f} (1/pixels)")


if __name__ == "__main__":
    main()
