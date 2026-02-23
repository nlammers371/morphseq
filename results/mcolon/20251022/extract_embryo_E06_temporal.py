"""
Extract and analyze masks for embryo 20250512_E06_e01 at two timepoints
- t0086 (early timepoint)
- t0181 (late timepoint)

Uses PCA-based slicing with Gaussian smoothing (σ=10) for curvature analysis.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter
from skimage import io
from sklearn.decomposition import PCA
import sys
import warnings
warnings.filterwarnings('ignore')

# Add path to mask utils
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))
from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle


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
        smoothed = gaussian_filter(mask.astype(float), sigma=sigma)
        
        # Re-threshold to get binary mask
        self.mask = (smoothed > 0.5).astype(np.uint8)
        self.sigma = sigma

    def extract_centerline_pca(self, n_slices: int = 100):
        """Extract centerline using PCA-based slicing."""
        y_coords, x_coords = np.where(self.mask)
        points = np.column_stack([x_coords, y_coords])

        # Perform PCA
        pca = PCA(n_components=2)
        pca.fit(points)

        principal_axis = pca.components_[0]
        center = points.mean(axis=0)
        centered_points = points - center
        projections = centered_points @ principal_axis

        # Create slices
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

        return principal_axis, np.array(centerline_points), center

    def compute_curvature(self, centerline_points: np.ndarray, smoothing: float = 0.01):
        """Compute curvature along the centerline."""
        if len(centerline_points) < 4:
            return np.array([]), np.array([]), np.array([])

        # Fit parametric spline
        tck, u = splprep([centerline_points[:, 0], centerline_points[:, 1]],
                         s=smoothing * len(centerline_points), k=3)

        # Evaluate spline and derivatives
        u_fine = np.linspace(0, 1, 200)
        x_vals, y_vals = splev(u_fine, tck)
        dx, dy = splev(u_fine, tck, der=1)
        ddx, ddy = splev(u_fine, tck, der=2)

        # Compute curvature
        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**(3/2)

        # Compute arc length
        arc_length = np.cumsum(np.sqrt(np.diff(x_vals)**2 + np.diff(y_vals)**2))
        arc_length = np.concatenate([[0], arc_length])

        spline_points = np.column_stack([x_vals, y_vals])

        return arc_length, curvature, spline_points


def load_mask_from_json(json_path: Path, snip_id: str):
    """Load and decode a specific mask from the segmentation JSON file."""
    print(f"\nLoading mask for {snip_id}...")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Navigate through the JSON structure
    # Structure: data['experiments'][exp_id]['videos'][video_id]['image_ids'][image_id]['embryos'][embryo_id]
    # Note: 'image_ids' is the field name, not 'frames'
    mask_found = False
    
    if 'experiments' in data:
        for exp_id, exp_data in data['experiments'].items():
            if 'videos' not in exp_data:
                continue
            
            for video_id, video_data in exp_data['videos'].items():
                if 'image_ids' not in video_data:
                    continue
                
                for image_id, image_data in video_data['image_ids'].items():
                    if 'embryos' not in image_data:
                        continue
                    
                    for embryo_id, embryo_data in image_data['embryos'].items():
                        if embryo_data.get('snip_id') == snip_id:
                            print(f"  Found in: {exp_id}/{video_id}/{image_id}/{embryo_id}")
                            seg_data = embryo_data['segmentation']
                            
                            # Decode RLE mask
                            mask = decode_mask_rle(seg_data)
                            print(f"  Mask shape: {mask.shape}")
                            print(f"  Mask area: {mask.sum()} pixels")
                            print(f"  Reported area: {seg_data.get('area', 'N/A')}")
                            
                            mask_found = True
                            return mask, embryo_data
    
    if not mask_found:
        raise ValueError(f"Could not find mask for {snip_id}")


def analyze_embryo_timepoint(mask: np.ndarray, embryo_data: dict, snip_id: str, 
                             output_dir: Path, sigma: float = 10.0):
    """Analyze a single embryo timepoint."""
    print(f"\n{'='*70}")
    print(f"Analyzing {snip_id}")
    print(f"{'='*70}")
    
    # Create analyzer
    analyzer = PCASlicingAnalyzer(mask, sigma=sigma)
    print(f"Original mask area: {analyzer.original_mask.sum()} pixels")
    print(f"Smoothed mask area: {analyzer.mask.sum()} pixels")
    
    # Extract centerline
    principal_axis, centerline_points, center = analyzer.extract_centerline_pca(n_slices=100)
    print(f"Extracted {len(centerline_points)} centerline points")
    
    # Compute curvature
    arc_length, curvature, spline_points = analyzer.compute_curvature(centerline_points, smoothing=0.01)
    print(f"Arc length: {arc_length[-1]:.1f} pixels")
    print(f"Mean curvature: {curvature.mean():.6f}")
    print(f"Max curvature: {curvature.max():.6f}")
    
    # Save data
    timepoint = snip_id.split('_')[-1]  # e.g., t0086
    
    # Save masks
    mask_original_path = output_dir / f"{snip_id}_mask_original.png"
    mask_smoothed_path = output_dir / f"{snip_id}_mask_smoothed.png"
    io.imsave(mask_original_path, (analyzer.original_mask * 255).astype(np.uint8))
    io.imsave(mask_smoothed_path, (analyzer.mask * 255).astype(np.uint8))
    
    # Save centerline and curvature data
    np.save(output_dir / f"{snip_id}_centerline_pca.npy", centerline_points)
    np.save(output_dir / f"{snip_id}_spline_points.npy", spline_points)
    
    curvature_df = pd.DataFrame({
        'arc_length_pixels': arc_length,
        'curvature_per_pixel': curvature
    })
    curvature_df.to_csv(output_dir / f"{snip_id}_curvature_profile.csv", index=False)
    
    # Save metadata
    metadata = {
        'embryo_id': embryo_data['embryo_id'],
        'snip_id': snip_id,
        'timepoint': timepoint,
        'gaussian_sigma': sigma,
        'mask_area_original_pixels': int(analyzer.original_mask.sum()),
        'mask_area_smoothed_pixels': int(analyzer.mask.sum()),
        'arc_length_total_pixels': float(arc_length[-1]),
        'curvature_mean': float(curvature.mean()),
        'curvature_std': float(curvature.std()),
        'curvature_max': float(curvature.max()),
        'principal_axis_x': float(principal_axis[0]),
        'principal_axis_y': float(principal_axis[1]),
        'center_x': float(center[0]),
        'center_y': float(center[1])
    }
    
    with open(output_dir / f"{snip_id}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return analyzer, centerline_points, spline_points, arc_length, curvature, metadata


def create_visualization(snip_id: str, analyzer, spline_points, arc_length, curvature, 
                        output_dir: Path, sigma: float):
    """Create comprehensive visualization for a timepoint."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Top left: Original mask
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(analyzer.original_mask, cmap='gray')
    ax1.set_title(f'Original Mask\nArea: {analyzer.original_mask.sum()} pixels', fontweight='bold')
    ax1.axis('off')
    
    # Top right: Smoothed mask with centerline
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(analyzer.mask, cmap='gray', alpha=0.7)
    ax2.plot(spline_points[:, 0], spline_points[:, 1], 'r-', linewidth=2, label='Centerline')
    ax2.scatter(spline_points[0, 0], spline_points[0, 1], c='lime', s=100, marker='o', 
               label='Start', zorder=5, edgecolors='black', linewidths=2)
    ax2.scatter(spline_points[-1, 0], spline_points[-1, 1], c='cyan', s=100, marker='s', 
               label='End', zorder=5, edgecolors='black', linewidths=2)
    ax2.set_title(f'Smoothed Mask (σ={sigma}) + Centerline', fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.axis('equal')
    ax2.invert_yaxis()
    
    # Bottom: Curvature profile (span both columns)
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(arc_length, curvature, 'b-', linewidth=2.5)
    ax3.fill_between(arc_length, 0, curvature, alpha=0.3)
    ax3.axhline(y=curvature.mean(), color='r', linestyle='--', linewidth=2, 
               label=f'Mean: {curvature.mean():.6f}')
    ax3.set_xlabel('Arc Length (pixels)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Curvature (1/pixels)', fontsize=12, fontweight='bold')
    ax3.set_title('Curvature Profile', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    
    # Stats box
    stats_text = (
        f'Statistics:\n'
        f'Mean: {curvature.mean():.6f}\n'
        f'Std: {curvature.std():.6f}\n'
        f'Max: {curvature.max():.6f}\n'
        f'Total arc: {arc_length[-1]:.1f} px'
    )
    ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(f'{snip_id} - PCA Curvature Analysis', fontsize=14, fontweight='bold')
    
    fig_path = output_dir / f"{snip_id}_analysis.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved figure: {fig_path}")
    plt.close()


def create_comparison_figure(results: dict, output_dir: Path):
    """Create comparison figure for both timepoints."""
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    colors = {'t0086': 'blue', 't0181': 'red'}
    timepoints = ['t0086', 't0181']
    
    for idx, tp in enumerate(timepoints):
        data = results[tp]
        
        # Masks with centerlines
        ax1 = fig.add_subplot(gs[0, idx])
        ax1.imshow(data['analyzer'].mask, cmap='gray', alpha=0.7)
        ax1.plot(data['spline_points'][:, 0], data['spline_points'][:, 1], 
                color=colors[tp], linewidth=2, label='Centerline')
        ax1.scatter(data['spline_points'][0, 0], data['spline_points'][0, 1], 
                   c='lime', s=100, marker='o', label='Start', zorder=5)
        ax1.set_title(f'{tp}\nArea: {data["metadata"]["mask_area_smoothed_pixels"]} px', 
                     fontweight='bold', fontsize=12)
        ax1.legend(fontsize=8)
        ax1.axis('equal')
        ax1.invert_yaxis()
        
        # Individual curvature plots
        ax2 = fig.add_subplot(gs[1, idx])
        ax2.plot(data['arc_length'], data['curvature'], color=colors[tp], linewidth=2)
        ax2.fill_between(data['arc_length'], 0, data['curvature'], alpha=0.3, color=colors[tp])
        ax2.axhline(y=data['curvature'].mean(), color='black', linestyle='--', linewidth=1.5,
                   label=f"Mean: {data['curvature'].mean():.6f}")
        ax2.set_xlabel('Arc Length (pixels)', fontweight='bold')
        ax2.set_ylabel('Curvature (1/pixels)', fontweight='bold')
        ax2.set_title(f'{tp} Curvature Profile', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)
    
    # Overlay comparison
    ax3 = fig.add_subplot(gs[1, 2])
    for tp in timepoints:
        data = results[tp]
        ax3.plot(data['arc_length'], data['curvature'], color=colors[tp], 
                linewidth=2.5, label=f'{tp} (mean: {data["curvature"].mean():.6f})', alpha=0.8)
    ax3.set_xlabel('Arc Length (pixels)', fontweight='bold', fontsize=11)
    ax3.set_ylabel('Curvature (1/pixels)', fontweight='bold', fontsize=11)
    ax3.set_title('Curvature Comparison', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    
    plt.suptitle('Embryo 20250512_E06_e01 - Temporal Curvature Analysis', 
                fontsize=16, fontweight='bold')
    
    comparison_path = output_dir / "embryo_E06_e01_temporal_comparison.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison figure: {comparison_path}")
    plt.close()


def main():
    # Paths
    json_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/sam2_pipeline_files/segmentation/grounded_sam_segmentations_20250512.json")
    output_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251022/plots/embryo_E06_e01_temporal_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Embryo 20250512_E06_e01 - Temporal Curvature Analysis")
    print("="*70)
    print(f"JSON path: {json_path}")
    print(f"Output directory: {output_dir}")
    
    # Timepoints to analyze
    snip_ids = [
        "20250512_E06_e01_t0086",
        "20250512_E06_e01_t0181"
    ]
    
    sigma = 10.0
    results = {}
    
    # Analyze each timepoint
    for snip_id in snip_ids:
        # Load mask from JSON
        mask, embryo_data = load_mask_from_json(json_path, snip_id)
        
        # Analyze
        analyzer, centerline_points, spline_points, arc_length, curvature, metadata = \
            analyze_embryo_timepoint(mask, embryo_data, snip_id, output_dir, sigma=sigma)
        
        # Create individual visualization
        create_visualization(snip_id, analyzer, spline_points, arc_length, curvature, 
                           output_dir, sigma)
        
        # Store results for comparison
        timepoint = snip_id.split('_')[-1]
        results[timepoint] = {
            'analyzer': analyzer,
            'spline_points': spline_points,
            'arc_length': arc_length,
            'curvature': curvature,
            'metadata': metadata
        }
    
    # Create comparison figure
    create_comparison_figure(results, output_dir)
    
    # Create summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nAll files saved to: {output_dir}")
    print("\nSummary:")
    for snip_id in snip_ids:
        tp = snip_id.split('_')[-1]
        data = results[tp]
        print(f"\n{snip_id}:")
        print(f"  - Mask area: {data['metadata']['mask_area_smoothed_pixels']} pixels")
        print(f"  - Arc length: {data['metadata']['arc_length_total_pixels']:.1f} pixels")
        print(f"  - Mean curvature: {data['metadata']['curvature_mean']:.6f} (1/pixels)")
        print(f"  - Max curvature: {data['metadata']['curvature_max']:.6f} (1/pixels)")


if __name__ == "__main__":
    main()
