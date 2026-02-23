"""
Benchmark Pipeline Performance: Mask Cleaning + Geodesic vs PCA

Times each step of the mask cleaning and analysis pipeline to identify bottlenecks
and determine if the pipeline can scale to thousands of embryos.

Compares:
- Geodesic skeleton + B-spline method
- PCA slicing + B-spline method

Outputs:
- Detailed timing CSV
- Visualizations showing breakdown of each step
- Performance summary and scaling estimates
"""

import sys
from pathlib import Path
import time

# Add project root to path
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.interpolate import splprep, splev
from skimage import morphology, measure
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle
from segmentation_sandbox.scripts.utils.mask_cleaning import clean_embryo_mask


class TimedMaskCleaning:
    """Mask cleaning with detailed timing for each step."""

    @staticmethod
    def clean_with_timing(mask: np.ndarray):
        """
        Clean mask and time each step.

        Returns:
            cleaned_mask: Cleaned mask
            timing_stats: Dict with timing for each step
        """
        timings = {}
        original_area = mask.sum()

        # Step 1: Remove debris
        t0 = time.perf_counter()
        labeled = measure.label(mask)
        if labeled.max() > 1:
            threshold = 0.10 * original_area
            component_areas = [(i, np.sum(labeled == i)) for i in range(1, labeled.max() + 1)]
            keep_labels = [label for label, area in component_areas if area >= threshold]
            if len(keep_labels) == 0:
                keep_labels = [max(component_areas, key=lambda x: x[1])[0]]
            mask_filtered = np.zeros_like(mask, dtype=bool)
            for label in keep_labels:
                mask_filtered |= (labeled == label)
            mask = mask_filtered
        timings['debris_removal'] = time.perf_counter() - t0

        # Step 2: Iterative closing
        t0 = time.perf_counter()
        if measure.label(mask).max() > 1:
            props_temp = measure.regionprops(measure.label(mask))[0]
            closing_radius = max(5, int(props_temp.perimeter / 100))
            closed = mask.copy()
            for iteration in range(5):
                selem_close = morphology.disk(closing_radius)
                closed = morphology.binary_closing(closed, selem_close)
                if measure.label(closed).max() == 1:
                    break
                closing_radius = min(closing_radius + 5, 50)
            mask = closed
        timings['iterative_closing'] = time.perf_counter() - t0

        # Step 3: Fill holes
        t0 = time.perf_counter()
        filled = ndimage.binary_fill_holes(mask)
        timings['fill_holes'] = time.perf_counter() - t0

        # Step 4: Opening
        t0 = time.perf_counter()
        props = measure.regionprops(measure.label(filled))[0]
        adaptive_radius = max(3, int(props.perimeter / 150))
        selem_open = morphology.disk(adaptive_radius)
        cleaned = morphology.binary_opening(filled, selem_open)
        timings['opening'] = time.perf_counter() - t0

        # Step 5: Keep largest component
        t0 = time.perf_counter()
        final_labeled = measure.label(cleaned)
        if final_labeled.max() > 1:
            component_sizes = [(i, np.sum(final_labeled == i)) for i in range(1, final_labeled.max() + 1)]
            largest_label = max(component_sizes, key=lambda x: x[1])[0]
            cleaned = (final_labeled == largest_label)
        timings['keep_largest'] = time.perf_counter() - t0

        timings['total_cleaning'] = sum(timings.values())

        return cleaned, timings


class TimedGeodesicAnalyzer:
    """Geodesic centerline extraction with detailed timing."""

    @staticmethod
    def analyze_with_timing(mask: np.ndarray):
        """
        Extract centerline and compute curvature with timing.

        Returns:
            results: Dict with centerline, curvature, etc.
            timing_stats: Dict with timing for each step
        """
        timings = {}

        # Step 1: Skeletonize
        t0 = time.perf_counter()
        skeleton = morphology.skeletonize(mask)
        y_skel, x_skel = np.where(skeleton)
        skel_points = np.column_stack([x_skel, y_skel])
        timings['skeleton'] = time.perf_counter() - t0

        if len(skel_points) < 2:
            raise ValueError("Skeleton too small")

        # Step 2: Build graph
        t0 = time.perf_counter()
        n_points = len(skel_points)
        edges = []
        weights = []
        for i in range(n_points):
            for j in range(i+1, n_points):
                dist = np.sqrt(np.sum((skel_points[i] - skel_points[j])**2))
                if dist <= np.sqrt(2) + 0.1:
                    edges.append((i, j))
                    weights.append(dist)
        rows = [e[0] for e in edges] + [e[1] for e in edges]
        cols = [e[1] for e in edges] + [e[0] for e in edges]
        data = weights + weights
        adj_matrix = csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
        timings['graph_build'] = time.perf_counter() - t0

        # Step 3: Dijkstra pathfinding
        t0 = time.perf_counter()
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
        timings['dijkstra'] = time.perf_counter() - t0

        # Step 4: B-spline fitting
        t0 = time.perf_counter()
        if len(centerline) >= 4:
            tck, u = splprep([centerline[:, 0], centerline[:, 1]],
                             s=5.0 * len(centerline), k=3)
            u_fine = np.linspace(0, 1, 200)
            x_vals, y_vals = splev(u_fine, tck)
            dx, dy = splev(u_fine, tck, der=1)
            ddx, ddy = splev(u_fine, tck, der=2)
            curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**(3/2)
            arc_length = np.cumsum(np.sqrt(np.diff(x_vals)**2 + np.diff(y_vals)**2))
            arc_length = np.concatenate([[0], arc_length])
        else:
            curvature = np.array([])
            arc_length = np.array([])
        timings['bspline'] = time.perf_counter() - t0

        timings['total_geodesic'] = sum(timings.values())

        results = {
            'centerline': centerline,
            'curvature': curvature,
            'arc_length': arc_length,
            'method': 'geodesic'
        }

        return results, timings


class TimedPCAAnalyzer:
    """PCA centerline extraction with detailed timing."""

    @staticmethod
    def analyze_with_timing(mask: np.ndarray):
        """
        Extract centerline using PCA and compute curvature with timing.

        Returns:
            results: Dict with centerline, curvature, etc.
            timing_stats: Dict with timing for each step
        """
        timings = {}

        # Step 1: PCA centerline extraction
        t0 = time.perf_counter()
        y_coords, x_coords = np.where(mask)
        points = np.column_stack([x_coords, y_coords])
        pca = PCA(n_components=2)
        pca.fit(points)
        principal_axis = pca.components_[0]
        center = points.mean(axis=0)
        centered_points = points - center
        projections = centered_points @ principal_axis
        min_proj, max_proj = projections.min(), projections.max()
        n_slices = 100
        slice_positions = np.linspace(min_proj, max_proj, n_slices)
        centerline_points = []
        for slice_pos in slice_positions:
            band_width = (max_proj - min_proj) / (n_slices * 2)
            in_slice = np.abs(projections - slice_pos) < band_width
            if np.sum(in_slice) > 0:
                slice_points = points[in_slice]
                centroid = slice_points.mean(axis=0)
                centerline_points.append(centroid)
        centerline = np.array(centerline_points)
        timings['pca_centerline'] = time.perf_counter() - t0

        # Step 2: B-spline fitting
        t0 = time.perf_counter()
        if len(centerline) >= 4:
            tck, u = splprep([centerline[:, 0], centerline[:, 1]],
                             s=5.0 * len(centerline), k=3)
            u_fine = np.linspace(0, 1, 200)
            x_vals, y_vals = splev(u_fine, tck)
            dx, dy = splev(u_fine, tck, der=1)
            ddx, ddy = splev(u_fine, tck, der=2)
            curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**(3/2)
            arc_length = np.cumsum(np.sqrt(np.diff(x_vals)**2 + np.diff(y_vals)**2))
            arc_length = np.concatenate([[0], arc_length])
        else:
            curvature = np.array([])
            arc_length = np.array([])
        timings['bspline'] = time.perf_counter() - t0

        timings['total_pca'] = sum(timings.values())

        results = {
            'centerline': centerline,
            'curvature': curvature,
            'arc_length': arc_length,
            'method': 'pca'
        }

        return results, timings


def benchmark_pipeline(n_embryos=10, random_seed=42):
    """
    Benchmark mask cleaning + analysis pipeline on N random embryos.

    Args:
        n_embryos: Number of embryos to test
        random_seed: Random seed for reproducibility

    Returns:
        results_df: DataFrame with detailed timing for each embryo and step
    """
    csv_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build06_output/df03_final_output_with_latents_20251017_part2.csv")

    print("="*80)
    print("PIPELINE PERFORMANCE BENCHMARKING")
    print("="*80)

    # Load data
    print(f"\nLoading: {csv_path.name}")
    df = pd.read_csv(csv_path)
    print(f"Total embryos: {len(df)}")

    # Sample random embryos
    np.random.seed(random_seed)
    sample_df = df.sample(n=n_embryos, random_state=random_seed)
    print(f"Testing on: {n_embryos} random embryos\n")

    results_list = []

    for idx, (_, row) in enumerate(sample_df.iterrows()):
        snip_id = row['snip_id']
        print(f"\n[{idx+1}/{n_embryos}] {snip_id}")
        print("-" * 60)

        try:
            # Decode mask
            mask = decode_mask_rle({
                'size': [int(row['mask_height_px']), int(row['mask_width_px'])],
                'counts': row['mask_rle']
            })
            mask = np.ascontiguousarray(mask.astype(np.uint8))
            print(f"  Mask: {mask.shape}, area: {mask.sum():,} px")

            # Time cleaning
            t_clean_start = time.perf_counter()
            cleaned_mask, clean_timings = TimedMaskCleaning.clean_with_timing(mask)
            t_clean_total = time.perf_counter() - t_clean_start
            print(f"  Cleaning: {t_clean_total:.4f}s")

            # Time Geodesic
            t_geo_start = time.perf_counter()
            geo_results, geo_timings = TimedGeodesicAnalyzer.analyze_with_timing(cleaned_mask)
            t_geo_total = time.perf_counter() - t_geo_start
            print(f"  Geodesic: {t_geo_total:.4f}s")

            # Time PCA
            t_pca_start = time.perf_counter()
            pca_results, pca_timings = TimedPCAAnalyzer.analyze_with_timing(cleaned_mask)
            t_pca_total = time.perf_counter() - t_pca_start
            print(f"  PCA: {t_pca_total:.4f}s")

            # Aggregate results
            result_row = {
                'snip_id': snip_id,
                'mask_area': mask.sum(),
                # Cleaning timings
                'clean_debris': clean_timings['debris_removal'],
                'clean_closing': clean_timings['iterative_closing'],
                'clean_holes': clean_timings['fill_holes'],
                'clean_opening': clean_timings['opening'],
                'clean_largest': clean_timings['keep_largest'],
                'clean_total': t_clean_total,
                # Geodesic timings
                'geo_skeleton': geo_timings['skeleton'],
                'geo_graph': geo_timings['graph_build'],
                'geo_dijkstra': geo_timings['dijkstra'],
                'geo_bspline': geo_timings['bspline'],
                'geo_total': t_geo_total,
                # PCA timings
                'pca_centerline': pca_timings['pca_centerline'],
                'pca_bspline': pca_timings['bspline'],
                'pca_total': t_pca_total,
                # Total pipeline
                'pipeline_geodesic_total': t_clean_total + t_geo_total,
                'pipeline_pca_total': t_clean_total + t_pca_total,
                'speedup_factor': t_geo_total / t_pca_total if t_pca_total > 0 else 0
            }

            results_list.append(result_row)

            print(f"  Total (Geodesic): {result_row['pipeline_geodesic_total']:.4f}s")
            print(f"  Total (PCA): {result_row['pipeline_pca_total']:.4f}s")
            print(f"  Speedup: {result_row['speedup_factor']:.2f}x")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    results_df = pd.DataFrame(results_list)
    return results_df


def create_visualizations(results_df: pd.DataFrame, output_dir: Path):
    """Create comprehensive timing visualizations."""

    print(f"\n{'='*80}")
    print("Creating visualizations...")
    print('='*80)

    # Calculate summary stats
    summary_stats = {
        'Cleaning': results_df['clean_total'].mean(),
        'Geodesic': results_df['geo_total'].mean(),
        'PCA': results_df['pca_total'].mean(),
        'Total (Geodesic)': results_df['pipeline_geodesic_total'].mean(),
        'Total (PCA)': results_df['pipeline_pca_total'].mean(),
    }

    # Figure 1: Overall timing comparison
    fig1, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1a: Mean time per major step
    categories = ['Cleaning', 'Geodesic\nAnalysis', 'PCA\nAnalysis']
    means = [
        results_df['clean_total'].mean(),
        results_df['geo_total'].mean(),
        results_df['pca_total'].mean()
    ]
    axes[0, 0].bar(categories, means, color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[0, 0].set_ylabel('Time (seconds)', fontweight='bold')
    axes[0, 0].set_title('Mean Time per Pipeline Stage', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(means):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}s', ha='center', fontweight='bold')

    # 1b: Cleaning breakdown
    clean_steps = ['Debris\nRemoval', 'Iterative\nClosing', 'Fill\nHoles', 'Opening', 'Keep\nLargest']
    clean_means = [
        results_df['clean_debris'].mean(),
        results_df['clean_closing'].mean(),
        results_df['clean_holes'].mean(),
        results_df['clean_opening'].mean(),
        results_df['clean_largest'].mean()
    ]
    axes[0, 1].bar(clean_steps, clean_means, color='#3498db')
    axes[0, 1].set_ylabel('Time (seconds)', fontweight='bold')
    axes[0, 1].set_title('Mask Cleaning Breakdown', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].tick_params(axis='x', rotation=0)

    # 1c: Geodesic breakdown
    geo_steps = ['Skeleton', 'Graph\nBuild', 'Dijkstra', 'B-spline']
    geo_means = [
        results_df['geo_skeleton'].mean(),
        results_df['geo_graph'].mean(),
        results_df['geo_dijkstra'].mean(),
        results_df['geo_bspline'].mean()
    ]
    axes[1, 0].bar(geo_steps, geo_means, color='#e74c3c')
    axes[1, 0].set_ylabel('Time (seconds)', fontweight='bold')
    axes[1, 0].set_title('Geodesic Method Breakdown', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # 1d: PCA breakdown
    pca_steps = ['PCA\nCenterline', 'B-spline']
    pca_means = [
        results_df['pca_centerline'].mean(),
        results_df['pca_bspline'].mean()
    ]
    axes[1, 1].bar(pca_steps, pca_means, color='#2ecc71')
    axes[1, 1].set_ylabel('Time (seconds)', fontweight='bold')
    axes[1, 1].set_title('PCA Method Breakdown', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig1_path = output_dir / "benchmark_timing_breakdown.png"
    plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig1_path}")
    plt.close()

    # Figure 2: Distribution and scaling
    fig2, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 2a: Box plot of total times
    data_for_box = [
        results_df['clean_total'],
        results_df['geo_total'],
        results_df['pca_total']
    ]
    bp = axes[0, 0].boxplot(data_for_box, labels=['Cleaning', 'Geodesic', 'PCA'],
                            patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#3498db', '#e74c3c', '#2ecc71']):
        patch.set_facecolor(color)
    axes[0, 0].set_ylabel('Time (seconds)', fontweight='bold')
    axes[0, 0].set_title('Time Distribution Across Embryos', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # 2b: Speedup factor
    speedup_mean = results_df['speedup_factor'].mean()
    axes[0, 1].hist(results_df['speedup_factor'], bins=10, color='#9b59b6', edgecolor='black')
    axes[0, 1].axvline(speedup_mean, color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {speedup_mean:.2f}x')
    axes[0, 1].set_xlabel('Speedup Factor (Geodesic / PCA)', fontweight='bold')
    axes[0, 1].set_ylabel('Count', fontweight='bold')
    axes[0, 1].set_title('Geodesic vs PCA Speedup', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 2c: Scaling estimate
    n_embryos_scale = [100, 1000, 10000, 100000]
    time_geo = [results_df['pipeline_geodesic_total'].mean() * n / 3600 for n in n_embryos_scale]
    time_pca = [results_df['pipeline_pca_total'].mean() * n / 3600 for n in n_embryos_scale]
    axes[1, 0].plot(n_embryos_scale, time_geo, 'o-', linewidth=2, markersize=8,
                   label='Geodesic', color='#e74c3c')
    axes[1, 0].plot(n_embryos_scale, time_pca, 's-', linewidth=2, markersize=8,
                   label='PCA', color='#2ecc71')
    axes[1, 0].set_xlabel('Number of Embryos', fontweight='bold')
    axes[1, 0].set_ylabel('Total Time (hours)', fontweight='bold')
    axes[1, 0].set_title('Estimated Pipeline Scaling', fontweight='bold')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, which='both')

    # 2d: Throughput
    throughput_geo = 3600 / results_df['pipeline_geodesic_total'].mean()
    throughput_pca = 3600 / results_df['pipeline_pca_total'].mean()
    axes[1, 1].bar(['Geodesic', 'PCA'], [throughput_geo, throughput_pca],
                  color=['#e74c3c', '#2ecc71'])
    axes[1, 1].set_ylabel('Embryos per Hour', fontweight='bold')
    axes[1, 1].set_title('Pipeline Throughput', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    for i, (method, v) in enumerate([('Geodesic', throughput_geo), ('PCA', throughput_pca)]):
        axes[1, 1].text(i, v + 5, f'{v:.0f}/hr', ha='center', fontweight='bold')

    plt.tight_layout()
    fig2_path = output_dir / "benchmark_scaling_analysis.png"
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig2_path}")
    plt.close()


def print_summary_report(results_df: pd.DataFrame, output_dir: Path):
    """Print and save summary performance report."""

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("PIPELINE PERFORMANCE SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append(f"\nTested on: {len(results_df)} embryos")
    report_lines.append(f"Data source: df03_final_output_with_latents_20251017_part2.csv\n")

    # Mean timings
    report_lines.append("\n" + "-" * 80)
    report_lines.append("MEAN TIMING PER EMBRYO (seconds)")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Step':<30} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    report_lines.append("-" * 80)

    timing_cols = {
        'Mask Cleaning': 'clean_total',
        '  - Debris Removal': 'clean_debris',
        '  - Iterative Closing': 'clean_closing',
        '  - Fill Holes': 'clean_holes',
        '  - Opening': 'clean_opening',
        '  - Keep Largest': 'clean_largest',
        'Geodesic Analysis': 'geo_total',
        '  - Skeleton': 'geo_skeleton',
        '  - Graph Build': 'geo_graph',
        '  - Dijkstra': 'geo_dijkstra',
        '  - B-spline': 'geo_bspline',
        'PCA Analysis': 'pca_total',
        '  - PCA Centerline': 'pca_centerline',
        '  - B-spline': 'pca_bspline',
    }

    for label, col in timing_cols.items():
        mean_val = results_df[col].mean()
        std_val = results_df[col].std()
        min_val = results_df[col].min()
        max_val = results_df[col].max()
        report_lines.append(f"{label:<30} {mean_val:<12.4f} {std_val:<12.4f} {min_val:<12.4f} {max_val:<12.4f}")

    # Total pipeline times
    report_lines.append("\n" + "-" * 80)
    report_lines.append("TOTAL PIPELINE TIME")
    report_lines.append("-" * 80)
    geo_mean = results_df['pipeline_geodesic_total'].mean()
    pca_mean = results_df['pipeline_pca_total'].mean()
    speedup = geo_mean / pca_mean if pca_mean > 0 else 0
    report_lines.append(f"Geodesic Pipeline: {geo_mean:.4f} ± {results_df['pipeline_geodesic_total'].std():.4f} s")
    report_lines.append(f"PCA Pipeline:      {pca_mean:.4f} ± {results_df['pipeline_pca_total'].std():.4f} s")
    report_lines.append(f"Speedup Factor:    {speedup:.2f}x (Geodesic / PCA)")

    # Bottlenecks
    report_lines.append("\n" + "-" * 80)
    report_lines.append("BOTTLENECK ANALYSIS")
    report_lines.append("-" * 80)
    cleaning_pct = (results_df['clean_total'].mean() / geo_mean) * 100
    report_lines.append(f"Cleaning takes {cleaning_pct:.1f}% of Geodesic pipeline time")

    # Find slowest cleaning step
    clean_steps = ['debris', 'closing', 'holes', 'opening', 'largest']
    clean_times = [results_df[f'clean_{s}'].mean() for s in clean_steps]
    slowest_clean = clean_steps[np.argmax(clean_times)]
    report_lines.append(f"Slowest cleaning step: {slowest_clean} ({max(clean_times):.4f}s)")

    # Find slowest geodesic step
    geo_steps = ['skeleton', 'graph', 'dijkstra', 'bspline']
    geo_times = [results_df[f'geo_{s}'].mean() for s in geo_steps]
    slowest_geo = geo_steps[np.argmax(geo_times)]
    report_lines.append(f"Slowest geodesic step: {slowest_geo} ({max(geo_times):.4f}s)")

    # Throughput
    report_lines.append("\n" + "-" * 80)
    report_lines.append("THROUGHPUT ESTIMATES")
    report_lines.append("-" * 80)
    throughput_geo = 3600 / geo_mean
    throughput_pca = 3600 / pca_mean
    report_lines.append(f"Geodesic: {throughput_geo:.0f} embryos/hour")
    report_lines.append(f"PCA:      {throughput_pca:.0f} embryos/hour")

    # Scaling estimates
    report_lines.append("\n" + "-" * 80)
    report_lines.append("SCALING ESTIMATES")
    report_lines.append("-" * 80)
    report_lines.append(f"{'N Embryos':<15} {'Geodesic':<20} {'PCA':<20}")
    report_lines.append("-" * 80)
    for n in [1000, 10000, 100000]:
        time_geo = (geo_mean * n) / 3600
        time_pca = (pca_mean * n) / 3600
        report_lines.append(f"{n:<15,} {time_geo:<20.1f}hrs {time_pca:<20.1f}hrs")

    # Recommendation
    report_lines.append("\n" + "-" * 80)
    report_lines.append("RECOMMENDATION FOR FULL PIPELINE")
    report_lines.append("-" * 80)
    if pca_mean < 0.5:
        report_lines.append("✓ PCA method is FAST enough for large-scale deployment (<0.5s per embryo)")
        report_lines.append(f"  Can process ~{int(throughput_pca)} embryos/hour on single CPU core")
    elif pca_mean < 2.0:
        report_lines.append("✓ PCA method is ACCEPTABLE for batch processing (<2s per embryo)")
        report_lines.append(f"  Can process ~{int(throughput_pca)} embryos/hour on single CPU core")
    else:
        report_lines.append("⚠ Both methods may be TOO SLOW for very large datasets")
        report_lines.append("  Consider parallelization or GPU acceleration")

    if speedup > 2.0:
        report_lines.append(f"\n✓ PCA is {speedup:.1f}x FASTER than Geodesic - strongly recommend PCA")
    elif speedup > 1.2:
        report_lines.append(f"\n✓ PCA is {speedup:.1f}x faster than Geodesic - recommend PCA")
    else:
        report_lines.append(f"\n✓ Methods have similar performance - choose based on accuracy")

    report_lines.append("\n" + "=" * 80)

    # Print to console
    report_text = "\n".join(report_lines)
    print(report_text)

    # Save to file
    report_path = output_dir / "benchmark_summary.txt"
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"\n✓ Saved report: {report_path}")


def main():
    """Main benchmarking execution."""
    output_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251027")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run benchmark
    results_df = benchmark_pipeline(n_embryos=10, random_seed=42)

    # Save detailed results
    csv_path = output_dir / "benchmark_results_10embryos.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved detailed results: {csv_path}")

    # Create visualizations
    create_visualizations(results_df, output_dir)

    # Print summary report
    print_summary_report(results_df, output_dir)

    print(f"\n{'='*80}")
    print("BENCHMARKING COMPLETE!")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
