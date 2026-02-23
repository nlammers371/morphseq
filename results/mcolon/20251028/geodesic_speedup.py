"""
Geodesic Centerline Speed Benchmark
===================================

This module provides a command-line utility for comparing the baseline
``GeodesicCenterlineAnalyzer`` against prototype speed improvements on the same
cleaned and preprocessed embryo masks.

Two analyzer variants are benchmarked:

1. ``baseline`` – current implementation imported from ``geodesic_method``.
   Uses O(N²) distance checks for graph construction.
2. ``fast_graph`` – optimized implementation using structured neighbour
   lookups (O(N)) instead of the baseline O(N²) distance checks.
   Produces geometrically identical results with ~13x speedup.

For each mask the script reports median runtimes, relative speed-ups, and
Hausdorff distance against the baseline smoothed centerline to confirm that
geometry is preserved.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Type

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage import morphology

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:  # pragma: no cover
    plt = None

# Add project root to path for direct execution
if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))

try:
    from imageio.v2 import imread as imageio_imread
except ImportError:  # pragma: no cover - optional dependency
    imageio_imread = None

try:
    from segmentation_sandbox.scripts.utils.mask_cleaning import clean_embryo_mask
except ImportError:  # pragma: no cover - optional dependency
    clean_embryo_mask = None

try:
    from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle
except ImportError:  # pragma: no cover - optional dependency
    decode_mask_rle = None

# Handle both relative and absolute imports
try:
    from .geodesic_method import GeodesicCenterlineAnalyzer
    from .spline_utils import align_spline_orientation
except ImportError:
    from segmentation_sandbox.scripts.body_axis_analysis.geodesic_method import GeodesicCenterlineAnalyzer
    from segmentation_sandbox.scripts.body_axis_analysis.spline_utils import align_spline_orientation


NEIGHBOUR_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)

DEFAULT_RANDOM_SEED = 42


@dataclass
class BenchmarkStats:
    """Container for per-analyzer benchmark results."""

    name: str
    durations: List[float]
    hausdorff_distance: float

    @property
    def median_runtime(self) -> float:
        return float(np.median(self.durations)) if self.durations else math.nan

    @property
    def mean_runtime(self) -> float:
        return float(np.mean(self.durations)) if self.durations else math.nan


class FastGraphGeodesicAnalyzer(GeodesicCenterlineAnalyzer):
    """Prototype analyzer that accelerates graph construction."""

    def extract_centerline(self):
        skeleton = morphology.skeletonize(self.mask)
        y_skel, x_skel = np.where(skeleton)

        if len(y_skel) < 2:
            raise ValueError("Skeleton has too few points")

        skel_points = np.column_stack([x_skel, y_skel])
        point_to_index = {tuple(pt): idx for idx, pt in enumerate(skel_points)}

        rows: List[int] = []
        cols: List[int] = []
        weights: List[float] = []

        for idx, (x, y) in enumerate(skel_points):
            for dx, dy in NEIGHBOUR_OFFSETS:
                neighbour = (x + dx, y + dy)
                jdx = point_to_index.get(neighbour)
                if jdx is None or jdx <= idx:
                    continue
                rows.append(idx)
                cols.append(jdx)
                weights.append(math.hypot(dx, dy))

        if not rows:
            raise ValueError("Skeleton graph has no edges (disconnected)")

        data = weights + weights
        adjacency = csr_matrix(
            (data, (rows + cols, cols + rows)),
            shape=(len(skel_points), len(skel_points)),
        )

        # Fallback to baseline endpoint logic for parity
        sample_size = min(50, len(skel_points))
        rng = np.random.RandomState(self.random_seed)
        sample_indices = rng.choice(len(skel_points), size=sample_size, replace=False)

        max_dist = 0.0
        best_pair = (0, len(skel_points) - 1)

        for idx in sample_indices:
            distances = dijkstra(adjacency, indices=idx, directed=False)
            finite_mask = np.isfinite(distances)
            if not np.any(finite_mask):
                continue
            furthest = int(np.argmax(distances[finite_mask]))
            furthest_idx = np.arange(len(distances))[finite_mask][furthest]
            if distances[furthest_idx] > max_dist:
                max_dist = float(distances[furthest_idx])
                best_pair = (idx, furthest_idx)

        start_idx, end_idx = best_pair

        distances, predecessors = dijkstra(
            adjacency,
            indices=start_idx,
            directed=False,
            return_predecessors=True,
        )

        path_indices: List[int] = []
        current = end_idx
        while current != -9999 and current != start_idx:
            path_indices.append(current)
            current = predecessors[current]
            if len(path_indices) > len(skel_points):
                raise RuntimeError("Detected cycle while tracing geodesic path")

        path_indices.append(start_idx)
        path_indices.reverse()

        centerline = skel_points[path_indices]
        endpoints = np.array([skel_points[start_idx], skel_points[end_idx]])

        return centerline, endpoints, skeleton




ANALYZER_CANDIDATES: Dict[str, Type[GeodesicCenterlineAnalyzer]] = {
    "baseline": GeodesicCenterlineAnalyzer,
    "fast_graph": FastGraphGeodesicAnalyzer,
}


def load_mask(path: Path) -> np.ndarray:
    """Load a 2D mask from .npy/.npz/.tif/.png."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        mask = np.load(path)
    elif suffix == ".npz":
        archive = np.load(path)
        # Heuristic: favour 'mask' key, otherwise take first array
        if "mask" in archive:
            mask = archive["mask"]
        else:
            first_key = sorted(archive.files)[0]
            mask = archive[first_key]
    elif suffix in {".tif", ".tiff", ".png"}:
        if imageio_imread is None:
            raise RuntimeError(f"imageio is required to read {suffix} files: {path}")
        mask = imageio_imread(path)
    else:
        raise ValueError(f"Unsupported mask format: {path}")

    if mask.ndim > 2:
        mask = mask.squeeze()

    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D: {path}")

    mask_binary = (mask > 0).astype(np.uint8)
    return mask_binary


def load_masks_from_df03(
    csv_paths: List[Path],
    n_samples: int = 5,
    random_seed: int = 42,
    output_dir: Optional[Path] = None,
) -> List[Tuple[str, np.ndarray]]:
    """
    Load masks from df03 CSV files.

    Args:
        csv_paths: List of paths to df03 CSV files
        n_samples: Number of masks to sample
        random_seed: Random seed for sampling
        output_dir: Optional directory to save masks as .npy files

    Returns:
        List of (snip_id, mask) tuples
    """
    if decode_mask_rle is None:
        raise RuntimeError("decode_mask_rle function is required but not available")

    print(f"Loading {len(csv_paths)} dataset(s)...")
    all_dfs = []
    for csv_path in csv_paths:
        if not csv_path.exists():
            print(f"  ! CSV not found: {csv_path}")
            continue
        print(f"  - {csv_path.name}")
        df = pd.read_csv(csv_path)
        all_dfs.append(df)
        print(f"    {len(df)} embryos")

    if not all_dfs:
        raise RuntimeError("No CSV files could be loaded")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total embryos available: {len(combined_df)}")

    np.random.seed(random_seed)
    if len(combined_df) > n_samples:
        sample_df = combined_df.sample(n=n_samples, random_state=random_seed)
        print(f"Sampling {n_samples} embryos for benchmarking")
    else:
        sample_df = combined_df
        print(f"Using all {len(sample_df)} embryos")

    masks = []
    for idx, row in sample_df.iterrows():
        try:
            snip_id = row['snip_id']
            mask = decode_mask_rle({
                'size': [int(row['mask_height_px']), int(row['mask_width_px'])],
                'counts': row['mask_rle']
            })
            mask = np.ascontiguousarray(mask.astype(np.uint8))

            # Optionally save mask
            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                mask_path = output_dir / f"{snip_id}.npy"
                np.save(mask_path, mask)
                print(f"  Saved: {mask_path.name} ({mask.shape}, {mask.sum()} pixels)")

            masks.append((snip_id, mask))

        except Exception as e:
            print(f"  ! Error loading mask for {row.get('snip_id', 'unknown')}: {e}")

    return masks


def preprocess(mask: np.ndarray, clean: bool = False) -> np.ndarray:
    """Apply optional cleaning to mask."""
    working_mask = mask.astype(np.uint8)

    if clean and clean_embryo_mask is not None:
        working_mask = clean_embryo_mask(working_mask)
    elif clean and clean_embryo_mask is None:
        raise RuntimeError("Mask cleaning requested but dependency is unavailable")

    return working_mask


def hausdorff_distance(line1: np.ndarray, line2: np.ndarray) -> float:
    """Compute symmetric Hausdorff distance between two polylines."""
    if line1.size == 0 or line2.size == 0:
        return math.nan

    # Align orientation for fairness
    aligned_x, aligned_y, _ = align_spline_orientation(
        line1[:, 0],
        line1[:, 1],
        line2[:, 0],
        line2[:, 1],
    )
    aligned_line2 = np.column_stack([aligned_x, aligned_y])

    dist_1 = np.min(np.linalg.norm(line1[:, None, :] - aligned_line2[None, :, :], axis=2), axis=1)
    dist_2 = np.min(np.linalg.norm(aligned_line2[:, None, :] - line1[None, :, :], axis=2), axis=1)

    return float(max(dist_1.max(initial=0.0), dist_2.max(initial=0.0)))


def time_analyzer(
    analyzer_cls: Type[GeodesicCenterlineAnalyzer],
    mask: np.ndarray,
    repeats: int,
) -> Tuple[List[float], dict]:
    """Time analyzer execution and capture the last result."""
    durations: List[float] = []
    result: Optional[dict] = None

    for _ in range(repeats):
        start = time.perf_counter()
        analyzer = analyzer_cls(
            mask,
            um_per_pixel=1.0,
            bspline_smoothing=5.0,
            random_seed=DEFAULT_RANDOM_SEED,
        )
        result = analyzer.analyze()
        durations.append(time.perf_counter() - start)

    if result is None:
        raise RuntimeError("Analyzer did not produce any result")

    return durations, result


def benchmark_mask(mask: np.ndarray, repeats: int) -> List[BenchmarkStats]:
    """Benchmark all analyzer candidates on a single mask."""
    baseline_durations, baseline_result = time_analyzer(
        ANALYZER_CANDIDATES["baseline"],
        mask,
        repeats=repeats,
    )

    baseline_stats = BenchmarkStats(
        name="baseline",
        durations=baseline_durations,
        hausdorff_distance=0.0,
    )

    stats: List[BenchmarkStats] = [baseline_stats]

    for name, analyzer_cls in ANALYZER_CANDIDATES.items():
        if name == "baseline":
            continue

        durations, result = time_analyzer(analyzer_cls, mask, repeats=repeats)
        hd = hausdorff_distance(
            baseline_result["centerline_smoothed"],
            result["centerline_smoothed"],
        )
        stats.append(BenchmarkStats(name=name, durations=durations, hausdorff_distance=hd))

    return stats


def summarise(stats: List[BenchmarkStats]) -> str:
    """Format benchmark results relative to the baseline."""
    baseline = next(s for s in stats if s.name == "baseline")
    lines = [
        f"baseline: median={baseline.median_runtime:.4f}s, mean={baseline.mean_runtime:.4f}s"
    ]

    for stat in stats:
        if stat.name == "baseline":
            continue
        median = stat.median_runtime
        mean = stat.mean_runtime
        speedup = baseline.median_runtime / median if median > 0 else math.nan
        lines.append(
            f"{stat.name}: median={median:.4f}s, mean={mean:.4f}s, "
            f"speedup={speedup:.2f}x, hausdorff={stat.hausdorff_distance:.3f}px"
        )

    return "\n".join(lines)


def visualize_benchmark_results(
    snip_id: str,
    mask: np.ndarray,
    baseline_result: dict,
    all_results: Dict[str, dict],
    output_dir: Path,
) -> None:
    """Create visualization of mask with overlaid splines from all analyzers."""
    if plt is None:
        print(f"  ! Matplotlib not available, skipping visualization for {snip_id}")
        return

    try:
        # Create figure with all analyzer results
        n_analyzers = len(all_results) + 1  # +1 for baseline
        fig, axes = plt.subplots(1, n_analyzers, figsize=(5*n_analyzers, 5))

        if n_analyzers == 1:
            axes = [axes]

        # Baseline
        ax = axes[0]
        ax.imshow(mask, cmap='gray', alpha=0.7)
        baseline_line = baseline_result['centerline_smoothed']
        ax.plot(baseline_line[:, 0], baseline_line[:, 1], 'b-', linewidth=2, label='baseline')
        ax.set_title('Baseline (GeodesicCenterlineAnalyzer)', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.axis('off')

        # Other analyzers
        colors = ['r', 'g', 'orange']
        for idx, (name, result) in enumerate(all_results.items(), 1):
            ax = axes[idx]
            ax.imshow(mask, cmap='gray', alpha=0.7)

            if 'centerline_smoothed' in result:
                line = result['centerline_smoothed']
                ax.plot(line[:, 0], line[:, 1], color=colors[idx-1], linewidth=2, label=name)

                # Overlay baseline for comparison
                ax.plot(baseline_line[:, 0], baseline_line[:, 1], 'b--', linewidth=1,
                       alpha=0.5, label='baseline')

            ax.set_title(name, fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.axis('off')

        plt.suptitle(f'{snip_id}\nCenterline Comparison', fontsize=12, fontweight='bold')
        plt.tight_layout()

        output_path = output_dir / f"{snip_id}_centerlines.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved visualization: {output_path.name}")

    except Exception as e:
        print(f"  ! Error creating visualization: {e}")


def create_summary_table(all_stats: Dict[str, List[BenchmarkStats]], output_dir: Path) -> None:
    """Create a summary table visualization."""
    if plt is None:
        return

    try:
        # Prepare data for table
        table_data = []
        for mask_id, stats_list in all_stats.items():
            for stat in stats_list:
                table_data.append({
                    'Mask': mask_id[:30] + '...' if len(mask_id) > 30 else mask_id,
                    'Analyzer': stat.name,
                    'Median (s)': f"{stat.median_runtime:.4f}",
                    'Hausdorff (px)': f"{stat.hausdorff_distance:.2f}",
                })

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')

        # Create table
        columns = ['Mask', 'Analyzer', 'Median (s)', 'Hausdorff (px)']
        table_data_rows = [[row[col] for col in columns] for row in table_data]

        table = ax.table(cellText=table_data_rows, colLabels=columns, loc='center',
                        cellLoc='center', colWidths=[0.4, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Color header
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(table_data_rows) + 1):
            for j in range(len(columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')

        plt.title('Benchmark Results Summary', fontsize=14, fontweight='bold', pad=20)
        output_path = output_dir / "summary_table.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved summary table: {output_path.name}")

    except Exception as e:
        print(f"! Error creating summary table: {e}")


def create_speedup_comparison(all_stats: Dict[str, List[BenchmarkStats]], output_dir: Path) -> None:
    """Create bar chart comparing speedups."""
    if plt is None:
        return

    try:
        # Extract speedups
        speedups_by_analyzer = {}
        for mask_id, stats_list in all_stats.items():
            baseline_time = next((s.median_runtime for s in stats_list if s.name == 'baseline'), None)
            if baseline_time is None:
                continue

            for stat in stats_list:
                if stat.name == 'baseline':
                    continue

                if stat.name not in speedups_by_analyzer:
                    speedups_by_analyzer[stat.name] = []

                speedup = baseline_time / stat.median_runtime if stat.median_runtime > 0 else 0
                speedups_by_analyzer[stat.name].append(speedup)

        if not speedups_by_analyzer:
            return

        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 6))

        analyzers = list(speedups_by_analyzer.keys())
        median_speedups = [np.median(speedups_by_analyzer[a]) for a in analyzers]

        bars = ax.bar(analyzers, median_speedups, color=['#2ecc71', '#e74c3c', '#3498db'])
        ax.set_ylabel('Speedup Factor (x)', fontsize=12)
        ax.set_title('Median Speedup vs Baseline', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}x',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()
        output_path = output_dir / "speedup_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved speedup chart: {output_path.name}")

    except Exception as e:
        print(f"! Error creating speedup chart: {e}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark geodesic speed improvements.")

    # All masks are loaded from df03 by default
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5,
        help="Number of masks to sample when using --from-df03 (default: 5).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results and saved masks. "
             "Default: results/mcolon/YYYYMMDD",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Apply clean_embryo_mask to masks.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timing repeats per analyzer (default: 3).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    # Determine output directory
    if args.output_dir is None:
        from datetime import datetime
        date_str = datetime.now().strftime("%Y%m%d")
        args.output_dir = Path(__file__).parents[3] / "results" / "mcolon" / date_str
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GEODESIC CENTERLINE SPEED BENCHMARK")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")

    # Load masks from df03 CSV files
    dataset_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build06_output")
    csv_paths = [
        dataset_dir / "df03_final_output_with_latents_20251017_part1.csv",
        dataset_dir / "df03_final_output_with_latents_20251017_part2.csv",
    ]

    print(f"\n==> Loading masks from df03 CSV files")
    mask_dir = args.output_dir / "masks"
    loaded_masks = load_masks_from_df03(
        csv_paths,
        n_samples=args.n_samples,
        random_seed=42,
        output_dir=mask_dir,
    )

    if not loaded_masks:
        print("No masks loaded. Exiting.")
        return

    # Process loaded masks and store results for visualization
    all_stats: Dict[str, List[BenchmarkStats]] = {}
    all_mask_results: Dict[str, Tuple[np.ndarray, Dict[str, dict]]] = {}  # mask and analyzer results

    for snip_id, raw_mask in loaded_masks:
        print(f"\n==> Processing mask: {snip_id}")
        processed_mask = preprocess(raw_mask, clean=args.clean)

        # Benchmark all analyzers and store results
        analyzer_results = {}
        stats = benchmark_mask(processed_mask, repeats=args.repeats)
        all_stats[snip_id] = stats

        # Run each analyzer once more to get the results for visualization
        for stat in stats:
            try:
                analyzer_cls = ANALYZER_CANDIDATES[stat.name]
                analyzer = analyzer_cls(
                    processed_mask,
                    um_per_pixel=1.0,
                    bspline_smoothing=5.0,
                    random_seed=DEFAULT_RANDOM_SEED,
                )
                result = analyzer.analyze()
                analyzer_results[stat.name] = result
            except Exception as e:
                print(f"  ! Error running {stat.name}: {e}")

        all_mask_results[snip_id] = (raw_mask, analyzer_results)
        print(summarise(stats))

    # Aggregate results
    if len(all_stats) > 1:
        print("\n==> Aggregated median runtimes:")
        baselines = [stats[0].median_runtime for stats in all_stats.values()]
        baseline_median = float(np.median(baselines))
        print(f"baseline median across masks: {baseline_median:.4f}s")

        for name in ANALYZER_CANDIDATES:
            if name == "baseline":
                continue
            candidate_medians = []
            hausdorff_values = []
            for stats in all_stats.values():
                stat = next(s for s in stats if s.name == name)
                candidate_medians.append(stat.median_runtime)
                hausdorff_values.append(stat.hausdorff_distance)
            combined_median = float(np.median(candidate_medians))
            speedup = baseline_median / combined_median if combined_median > 0 else math.nan
            hd_median = float(np.median(hausdorff_values))
            print(
                f"{name}: median={combined_median:.4f}s, speedup={speedup:.2f}x, "
                f"median_hausdorff={hd_median:.3f}px"
            )

    # Save results to CSV
    results_csv = args.output_dir / "benchmark_results.csv"
    results_data = []
    for mask_id, stats_list in all_stats.items():
        for stat in stats_list:
            results_data.append({
                "mask_id": mask_id,
                "analyzer": stat.name,
                "median_runtime_s": stat.median_runtime,
                "mean_runtime_s": stat.mean_runtime,
                "hausdorff_distance_px": stat.hausdorff_distance,
            })

    results_df = pd.DataFrame(results_data)
    results_df.to_csv(results_csv, index=False)
    print(f"\n==> Saved results to: {results_csv}")

    # Create visualizations
    print(f"\n==> Creating visualizations...")
    vis_dir = args.output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for snip_id, (raw_mask, analyzer_results) in all_mask_results.items():
        if 'baseline' in analyzer_results:
            baseline_result = analyzer_results['baseline']
            other_results = {k: v for k, v in analyzer_results.items() if k != 'baseline'}
            visualize_benchmark_results(snip_id, raw_mask, baseline_result, other_results, vis_dir)

    # Create summary visualizations
    create_summary_table(all_stats, vis_dir)
    create_speedup_comparison(all_stats, vis_dir)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":  # pragma: no cover
    main()
