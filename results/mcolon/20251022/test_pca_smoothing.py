"""
Test different smoothing levels for PCA-based curvature analysis

Smoothing parameter controls how much the spline smooths the centerline.
Higher values = more smoothing = less sensitive to noise but may miss features
Lower values = less smoothing = follows data closely but may be noisy
"""

import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle


class PCACurvatureAnalyzer:
    """PCA-based curvature analysis with adjustable smoothing."""

    def __init__(self, mask: np.ndarray, um_per_pixel: float = 1.0):
        """
        Initialize with a binary mask.

        Args:
            mask: Binary mask as numpy array
            um_per_pixel: Conversion factor from pixels to microns
        """
        self.mask = np.ascontiguousarray(mask.astype(np.uint8))
        self.um_per_pixel = um_per_pixel

    def extract_centerline_pca(self, n_slices=100):
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

        return np.array(centerline_points)

    def compute_curvature(self, centerline_points: np.ndarray, smoothing=0.01):
        """
        Compute curvature along the centerline.

        Args:
            centerline_points: (N, 2) array of (x, y) coordinates
            smoothing: Smoothing parameter for spline fitting
                      - 0: no smoothing (interpolation)
                      - 0.001-0.01: light smoothing
                      - 0.01-0.1: moderate smoothing
                      - 0.1-1.0: heavy smoothing

        Returns:
            arc_length: Arc length parameter (in microns)
            curvature: Curvature at each point (1/microns)
            spline_x, spline_y: Smoothed spline coordinates
        """
        if len(centerline_points) < 4:
            return np.array([]), np.array([]), np.array([]), np.array([])

        # Fit a parametric spline
        # s = smoothing parameter (weighted sum of squared residuals)
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

        # Convert to microns
        arc_length = arc_length * self.um_per_pixel
        curvature = curvature / self.um_per_pixel

        return arc_length, curvature, x_vals, y_vals


def test_smoothing_levels():
    """Test different smoothing levels on PCA curvature analysis."""

    # Load the CSV
    csv_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build06_output/df03_final_output_with_latents_20251020.csv")

    print(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Test different smoothing levels - from no smoothing to very aggressive
    smoothing_levels = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    n_masks = 4

    # Create figure: 4 masks x 8 smoothing levels
    fig, axes = plt.subplots(n_masks, len(smoothing_levels), figsize=(40, 4*n_masks))

    for i in range(n_masks):
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
        analyzer = PCACurvatureAnalyzer(mask, um_per_pixel=um_per_pixel)

        # Extract centerline once
        centerline = analyzer.extract_centerline_pca(n_slices=100)
        print(f"Centerline points: {len(centerline)}")

        # Test each smoothing level
        for j, smoothing in enumerate(smoothing_levels):
            print(f"\nSmoothing = {smoothing}")

            try:
                arc_length, curvature, spline_x, spline_y = analyzer.compute_curvature(
                    centerline, smoothing=smoothing
                )

                if len(arc_length) > 0:
                    print(f"  ✓ Arc length: {arc_length[-1]:.2f} μm")
                    print(f"  ✓ Mean curvature: {np.mean(curvature):.6f} μm⁻¹")
                    print(f"  ✓ Max curvature: {np.max(curvature):.6f} μm⁻¹")
                    print(f"  ✓ Std curvature: {np.std(curvature):.6f} μm⁻¹")

                # Plot mask + centerline + smoothed spline
                axes[i, j].imshow(mask, cmap='gray', alpha=0.3)

                # Original centerline points in light color
                axes[i, j].scatter(centerline[:, 0], centerline[:, 1],
                                  c='lightblue', s=10, alpha=0.5, label='Raw points')

                # Smoothed spline in red
                axes[i, j].plot(spline_x, spline_y, 'r-', linewidth=2, label='Smoothed')

                axes[i, j].set_title(f"Smoothing = {smoothing}\n" +
                                    f"Mean κ: {np.mean(curvature):.4f} μm⁻¹\n" +
                                    f"Std κ: {np.std(curvature):.4f} μm⁻¹")
                axes[i, j].axis('equal')
                axes[i, j].invert_yaxis()
                axes[i, j].axis('off')
                if j == 0:
                    axes[i, j].legend(loc='upper right', fontsize=8)

            except Exception as e:
                print(f"  ✗ Error: {e}")
                axes[i, j].text(0.5, 0.5, f"Error:\n{str(e)[:50]}",
                               ha='center', va='center',
                               transform=axes[i, j].transAxes)
                axes[i, j].axis('off')

    plt.tight_layout()

    # Save figure
    output_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251022/pca_smoothing_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n{'='*60}")
    print(f"✓ Saved comparison to: {output_path}")
    print(f"{'='*60}")

    # Also create curvature plots
    fig2, axes2 = plt.subplots(n_masks, len(smoothing_levels), figsize=(40, 3*n_masks))

    for i in range(n_masks):
        row = df.iloc[i]

        # Decode mask
        rle_data = {
            'counts': row['mask_rle'],
            'size': [int(row['mask_height_px']), int(row['mask_width_px'])]
        }
        mask = decode_mask_rle(rle_data)
        um_per_pixel = row['height_um'] / row['height_px']

        analyzer = PCACurvatureAnalyzer(mask, um_per_pixel=um_per_pixel)
        centerline = analyzer.extract_centerline_pca(n_slices=100)

        for j, smoothing in enumerate(smoothing_levels):
            try:
                arc_length, curvature, _, _ = analyzer.compute_curvature(
                    centerline, smoothing=smoothing
                )

                if len(arc_length) > 0:
                    axes2[i, j].plot(arc_length, curvature, 'b-', linewidth=2)
                    axes2[i, j].set_xlabel('Arc Length (μm)', fontsize=8)
                    axes2[i, j].set_ylabel('Curvature (μm⁻¹)', fontsize=8)
                    axes2[i, j].set_title(f"s={smoothing}", fontsize=10)
                    axes2[i, j].grid(True, alpha=0.3)

            except:
                pass

    plt.tight_layout()
    output_path2 = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251022/pca_curvature_plots.png")
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"✓ Saved curvature plots to: {output_path2}")

    plt.close('all')


if __name__ == "__main__":
    print("="*60)
    print("Testing PCA Smoothing Levels for Curvature Analysis")
    print("="*60)

    test_smoothing_levels()

    print("\nDone!")
