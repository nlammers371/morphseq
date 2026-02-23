"""
Geodesic Skeleton Curvature Analysis - Production Version

Uses geodesic distance along skeleton to find true embryo endpoints,
then applies Gaussian smoothing (σ=10) for robust curvature measurement.

This method works well for highly curved embryos where head is near tail.
"""

import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage import morphology
import warnings
warnings.filterwarnings('ignore')

from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle


class GeodesicCurvatureAnalyzer:
    """
    Analyze embryo curvature using geodesic skeleton method + Gaussian smoothing.

    This is the production-ready version using only the best-performing method.
    """

    def __init__(self, mask: np.ndarray, um_per_pixel: float = 1.0, sigma: float = 10.0):
        """
        Initialize analyzer with a binary mask.

        Args:
            mask: Binary mask as numpy array
            um_per_pixel: Conversion factor from pixels to microns (default=1.0 for pixel units)
            sigma: Gaussian smoothing parameter (default=10.0)
        """
        self.mask = np.ascontiguousarray(mask.astype(np.uint8))
        self.um_per_pixel = um_per_pixel
        self.sigma = sigma

    def extract_centerline(self):
        """
        Extract centerline using geodesic distance along skeleton.

        Returns:
            centerline: (N, 2) array of (x, y) coordinates
            endpoints: (2, 2) array of start/end points
            skeleton: Binary skeleton image
        """
        # Step 1: Skeletonize
        skeleton = morphology.skeletonize(self.mask)
        y_skel, x_skel = np.where(skeleton)

        if len(y_skel) < 2:
            raise ValueError("Skeleton has too few points")

        skel_points = np.column_stack([x_skel, y_skel])

        # Step 2: Build graph (8-connected neighbors)
        n_points = len(skel_points)
        edges = []
        weights = []

        for i in range(n_points):
            for j in range(i+1, n_points):
                dist = np.sqrt(np.sum((skel_points[i] - skel_points[j])**2))
                if dist <= np.sqrt(2) + 0.1:  # 8-connected
                    edges.append((i, j))
                    weights.append(dist)

        if len(edges) == 0:
            raise ValueError("Skeleton graph has no edges (disconnected)")

        # Step 3: Create adjacency matrix
        rows = [e[0] for e in edges] + [e[1] for e in edges]
        cols = [e[1] for e in edges] + [e[0] for e in edges]
        data = weights + weights
        adj_matrix = csr_matrix((data, (rows, cols)), shape=(n_points, n_points))

        # Step 4: Find endpoints with maximum geodesic distance
        sample_size = min(50, n_points)
        sample_indices = np.random.choice(n_points, size=sample_size, replace=False)

        max_dist = 0
        best_pair = (0, n_points-1)

        for idx in sample_indices:
            distances = dijkstra(adj_matrix, indices=idx, directed=False)
            furthest = np.argmax(distances[np.isfinite(distances)])
            if distances[furthest] > max_dist:
                max_dist = distances[furthest]
                best_pair = (idx, furthest)

        start_idx, end_idx = best_pair
        endpoints = np.array([skel_points[start_idx], skel_points[end_idx]])

        # Step 5: Trace path from start to end
        distances, predecessors = dijkstra(adj_matrix, indices=start_idx,
                                          directed=False, return_predecessors=True)

        path_indices = []
        current = end_idx
        while current != -9999 and current != start_idx:
            path_indices.append(current)
            current = predecessors[current]
            if len(path_indices) > n_points:
                break
        path_indices.append(start_idx)
        path_indices = path_indices[::-1]

        centerline = skel_points[path_indices]

        return centerline, endpoints, skeleton

    def smooth_centerline(self, centerline: np.ndarray, extra_smooth: bool = False):
        """
        Apply Gaussian smoothing to centerline.

        Args:
            centerline: (N, 2) array of (x, y) coordinates
            extra_smooth: If True, apply extra smoothing for curvature calculation

        Returns:
            smoothed: (N, 2) array of smoothed coordinates
        """
        if len(centerline) < 4:
            return centerline

        # Use higher sigma for curvature computation to reduce noise
        sigma = self.sigma * 2 if extra_smooth else self.sigma

        smoothed_x = gaussian_filter1d(centerline[:, 0], sigma=sigma, mode='nearest')
        smoothed_y = gaussian_filter1d(centerline[:, 1], sigma=sigma, mode='nearest')
        return np.column_stack([smoothed_x, smoothed_y])

    def compute_curvature(self, smoothed_points: np.ndarray):
        """
        Compute curvature from smoothed centerline using finite differences.

        Args:
            smoothed_points: (N, 2) array of smoothed (x, y) coordinates

        Returns:
            arc_length: Arc length parameter (in um or pixels)
            curvature: Curvature at each point (in 1/um or 1/pixels)
        """
        if len(smoothed_points) < 3:
            return np.array([]), np.array([])

        # First derivatives (central differences)
        dx = np.gradient(smoothed_points[:, 0])
        dy = np.gradient(smoothed_points[:, 1])

        # Second derivatives
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        # Curvature: κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**(3/2)

        # Arc length
        arc_length = np.cumsum(np.sqrt(np.diff(smoothed_points[:, 0])**2 +
                                       np.diff(smoothed_points[:, 1])**2))
        arc_length = np.concatenate([[0], arc_length])

        # Convert to physical units
        arc_length = arc_length * self.um_per_pixel
        curvature = curvature / self.um_per_pixel

        return arc_length, curvature

    def analyze(self):
        """
        Full analysis pipeline: extract centerline, smooth, compute curvature.

        Returns:
            results: Dictionary with all analysis results
        """
        # Extract centerline
        centerline, endpoints, skeleton = self.extract_centerline()

        # Smooth for visualization (sigma)
        smoothed_viz = self.smooth_centerline(centerline, extra_smooth=False)

        # Smooth extra for curvature (2*sigma to reduce noise in derivatives)
        smoothed_curv = self.smooth_centerline(centerline, extra_smooth=True)

        # Compute curvature on heavily smoothed version
        arc_length, curvature = self.compute_curvature(smoothed_curv)

        results = {
            'centerline_raw': centerline,
            'centerline_smoothed': smoothed_viz,
            'centerline_smoothed_curvature': smoothed_curv,
            'endpoints': endpoints,
            'skeleton': skeleton,
            'arc_length': arc_length,
            'curvature': curvature,
            'stats': {
                'total_length': arc_length[-1] if len(arc_length) > 0 else 0,
                'mean_curvature': np.mean(curvature) if len(curvature) > 0 else 0,
                'std_curvature': np.std(curvature) if len(curvature) > 0 else 0,
                'max_curvature': np.max(curvature) if len(curvature) > 0 else 0,
                'n_centerline_points': len(centerline),
                'n_skeleton_points': np.sum(skeleton)
            }
        }

        return results

    def visualize(self, results: dict, title: str = "Geodesic Curvature Analysis"):
        """
        Create comprehensive visualization of analysis results.

        Args:
            results: Dictionary from analyze()
            title: Plot title

        Returns:
            fig: Matplotlib figure
        """
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # 1. Original mask with smoothed centerline
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(self.mask, cmap='gray', alpha=0.5)
        ax1.plot(results['centerline_smoothed'][:, 0],
                results['centerline_smoothed'][:, 1],
                'r-', linewidth=3, label='Smoothed centerline')
        ax1.scatter(results['endpoints'][0, 0], results['endpoints'][0, 1],
                   c='green', s=200, marker='o', edgecolors='white',
                   linewidths=3, label='Start', zorder=5)
        ax1.scatter(results['endpoints'][1, 0], results['endpoints'][1, 1],
                   c='blue', s=200, marker='s', edgecolors='white',
                   linewidths=3, label='End', zorder=5)
        ax1.set_title('Mask + Centerline', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.axis('equal')
        ax1.invert_yaxis()
        ax1.axis('off')

        # 2. Skeleton quality
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(self.mask, cmap='gray', alpha=0.3)
        ax2.imshow(results['skeleton'], cmap='Reds', alpha=0.7)
        ax2.plot(results['centerline_raw'][:, 0], results['centerline_raw'][:, 1],
                'b-', linewidth=2, label='Traced path', alpha=0.8)
        ax2.scatter(results['endpoints'][0, 0], results['endpoints'][0, 1],
                   c='green', s=200, marker='o', edgecolors='black',
                   linewidths=3, label='Start', zorder=5)
        ax2.scatter(results['endpoints'][1, 0], results['endpoints'][1, 1],
                   c='blue', s=200, marker='s', edgecolors='black',
                   linewidths=3, label='End', zorder=5)
        ax2.set_title('Skeleton Quality\n(Red=skeleton, Blue=path)',
                     fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.axis('equal')
        ax2.invert_yaxis()
        ax2.axis('off')

        # 3. Smoothing comparison
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(self.mask, cmap='gray', alpha=0.3)
        ax3.scatter(results['centerline_raw'][:, 0], results['centerline_raw'][:, 1],
                   c='lightblue', s=10, alpha=0.5, label='Raw skeleton')
        ax3.plot(results['centerline_smoothed'][:, 0],
                results['centerline_smoothed'][:, 1],
                'orange', linewidth=2, label=f'Viz smooth (σ={self.sigma})', alpha=0.7)
        ax3.plot(results['centerline_smoothed_curvature'][:, 0],
                results['centerline_smoothed_curvature'][:, 1],
                'r-', linewidth=3, label=f'Curv smooth (σ={self.sigma*2})')
        ax3.set_title('Smoothing Levels', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.axis('equal')
        ax3.invert_yaxis()
        ax3.axis('off')

        # 4. Curvature profile (span all bottom columns)
        ax4 = fig.add_subplot(gs[1, :])
        ax4.plot(results['arc_length'], results['curvature'],
                'b-', linewidth=2.5)
        ax4.fill_between(results['arc_length'], 0, results['curvature'],
                        alpha=0.3)
        ax4.axhline(y=results['stats']['mean_curvature'],
                   color='r', linestyle='--', linewidth=2,
                   label=f'Mean: {results["stats"]["mean_curvature"]:.6f}')

        unit = 'μm' if self.um_per_pixel != 1.0 else 'px'
        ax4.set_xlabel(f'Arc Length ({unit})', fontsize=12, fontweight='bold')
        ax4.set_ylabel(f'Curvature (1/{unit})', fontsize=12, fontweight='bold')
        ax4.set_title('Curvature Profile', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=10)

        # Stats text box
        stats_text = (
            f'Statistics:\n'
            f'Length: {results["stats"]["total_length"]:.1f} {unit}\n'
            f'Mean κ: {results["stats"]["mean_curvature"]:.6f} {unit}⁻¹\n'
            f'Std κ: {results["stats"]["std_curvature"]:.6f} {unit}⁻¹\n'
            f'Max κ: {results["stats"]["max_curvature"]:.6f} {unit}⁻¹\n'
            f'Points: {results["stats"]["n_centerline_points"]}'
        )
        ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.suptitle(title, fontsize=14, fontweight='bold')

        return fig


def load_mask_from_json(json_path: Path, snip_id: str):
    """Load mask from grounded SAM JSON file."""
    import json

    print(f"Loading mask for {snip_id}...")

    with open(json_path, 'r') as f:
        data = json.load(f)

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
                            print(f"  ✓ Found: {exp_id}/{video_id}/{image_id}/{embryo_id}")
                            seg_data = embryo_data['segmentation']
                            mask = decode_mask_rle(seg_data)
                            print(f"  Mask shape: {mask.shape}, area: {mask.sum()} px")
                            return mask, embryo_data

    raise ValueError(f"Could not find mask for {snip_id}")


def demo():
    """Demo: analyze the 3 test embryos."""

    output_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251022")

    # Test cases
    test_cases = [
        (Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/sam2_pipeline_files/segmentation/grounded_sam_segmentations_20251017_part2.json"),
         "20251017_part2_D06_e01_t0022",
         "Good baseline (D06)"),
        (Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/sam2_pipeline_files/segmentation/grounded_sam_segmentations_20250512.json"),
         "20250512_E06_e01_t0086",
         "Challenging (extreme curve)"),
        (Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/sam2_pipeline_files/segmentation/grounded_sam_segmentations_20250512.json"),
         "20250512_E06_e01_t0181",
         "Very challenging (self-overlap)")
    ]

    print("="*70)
    print("Geodesic Skeleton Curvature Analysis - Demo")
    print("="*70)

    for json_path, snip_id, description in test_cases:
        print(f"\n{'='*70}")
        print(f"Analyzing: {description}")
        print(f"{'='*70}")

        try:
            # Load mask
            mask, embryo_data = load_mask_from_json(json_path, snip_id)

            # Analyze
            analyzer = GeodesicCurvatureAnalyzer(mask, um_per_pixel=1.0, sigma=10.0)
            results = analyzer.analyze()

            # Print stats
            print(f"\nResults:")
            for key, value in results['stats'].items():
                print(f"  {key}: {value}")

            # Visualize
            fig = analyzer.visualize(results, title=f"{description}\n{snip_id}")

            # Save
            safe_filename = snip_id.replace('_', '-')
            fig_path = output_dir / f"geodesic_analysis_{safe_filename}.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Saved: {fig_path}")
            plt.close()

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print("Demo complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    demo()
