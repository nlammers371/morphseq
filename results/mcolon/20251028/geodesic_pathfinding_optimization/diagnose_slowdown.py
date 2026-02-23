"""
Diagnostic script to identify why optimized version is slower on real masks.

Tests each optimization component individually to isolate the culprit.
"""

import sys
import time
import numpy as np
import importlib.util
from pathlib import Path

# Add paths
project_root = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq")
sys.path.insert(0, str(project_root))

from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle
import pandas as pd
from skimage import morphology

# Load original method
geodesic_spec = importlib.util.spec_from_file_location(
    "geodesic_method",
    str(project_root / "segmentation_sandbox/scripts/body_axis_analysis/geodesic_method.py")
)
geodesic_module = importlib.util.module_from_spec(geodesic_spec)
geodesic_spec.loader.exec_module(geodesic_module)
GeodesicCenterlineAnalyzer = geodesic_module.GeodesicCenterlineAnalyzer

# Load optimized method
geodesic_opt_spec = importlib.util.spec_from_file_location(
    "geodesic_method_optimized",
    str(project_root / "segmentation_sandbox/scripts/body_axis_analysis/geodesic_method_optimized.py")
)
geodesic_opt_module = importlib.util.module_from_spec(geodesic_opt_spec)
geodesic_opt_spec.loader.exec_module(geodesic_opt_module)
GeodesicCenterlineAnalyzerOptimized = geodesic_opt_module.GeodesicCenterlineAnalyzerOptimized


def load_real_masks(num_samples=3):
    """Load real embryo masks from CSV."""
    metadata_dir = project_root / "morphseq_playground" / "metadata" / "build06_output"
    csv_files = list(metadata_dir.glob("*.csv"))

    if not csv_files:
        print("No CSV files found")
        return []

    df = pd.read_csv(csv_files[0])
    masks = []

    for idx, row in df.iloc[:num_samples].iterrows():
        try:
            mask = decode_mask_rle({
                'size': [int(row['mask_height_px']), int(row['mask_width_px'])],
                'counts': row['mask_rle']
            })
            snip_id = row.get('snip_id', f"embryo_{idx}")
            masks.append((mask.astype(np.uint8), str(snip_id)))
        except Exception as e:
            print(f"Skipped: {e}")

    return masks


def test_component(mask, name, disable_thinning=False, disable_convolution=False):
    """Test optimized version with specific components disabled."""

    # Create custom analyzer
    class CustomOptimizedAnalyzer(GeodesicCenterlineAnalyzerOptimized):
        def extract_centerline(self):
            # Step 1: Skeletonize
            skeleton = morphology.skeletonize(self.mask)

            # OPTIONAL: Skeleton thinning
            if not disable_thinning:
                skeleton = morphology.thin(skeleton)

            y_skel, x_skel = np.where(skeleton)

            if len(y_skel) < 2:
                raise ValueError("Skeleton has too few points")

            skel_points = np.column_stack([x_skel, y_skel])

            # Step 2: Build graph
            if self.fast:
                adj_matrix = self._build_graph_fast(skel_points)
            else:
                adj_matrix = self._build_graph_slow(skel_points)

            n_points = len(skel_points)

            # Step 2.5: Clean disconnected components
            from scipy.sparse.csgraph import connected_components
            n_components, component_labels = connected_components(adj_matrix, directed=False)

            if n_components > 1:
                unique_labels, counts = np.unique(component_labels, return_counts=True)
                largest_label = unique_labels[np.argmax(counts)]
                valid_mask = component_labels == largest_label
                valid_indices = np.where(valid_mask)[0]

                skel_points = skel_points[valid_indices]

                index_map = np.full(n_points, -1, dtype=int)
                index_map[valid_indices] = np.arange(len(valid_indices))

                cx, cy = adj_matrix.nonzero()
                valid_edges_mask = valid_mask[cx] & valid_mask[cy]
                cx_new = index_map[cx[valid_edges_mask]]
                cy_new = index_map[cy[valid_edges_mask]]
                data_new = adj_matrix.data[valid_edges_mask]

                from scipy.sparse import csr_matrix
                adj_matrix = csr_matrix(
                    (data_new, (cx_new, cy_new)),
                    shape=(len(valid_indices), len(valid_indices))
                )
                n_points = len(skel_points)

            # Step 3: Find endpoints
            sample_indices = None

            # OPTIONAL: Convolution filtering
            if self.use_convolution_filter and not disable_convolution:
                candidate_indices = self._find_endpoint_candidates_convolution(skeleton, skel_points)
                if candidate_indices is not None and len(candidate_indices) > 0:
                    sample_indices = candidate_indices

            # Fallback to sampling
            if sample_indices is None:
                if n_points > 100:
                    sample_size = min(100, n_points)
                    rng = np.random.RandomState(self.random_seed)
                    sample_indices = rng.choice(n_points, size=sample_size, replace=False)
                else:
                    sample_indices = np.arange(n_points)

            # Find endpoint pair
            from scipy.sparse.csgraph import dijkstra
            max_dist_overall = 0
            best_pair = (0, min(1, n_points - 1))

            for i, idx in enumerate(sample_indices):
                distances = dijkstra(adj_matrix, indices=idx, directed=False)
                finite_mask = np.isfinite(distances)

                if np.any(finite_mask):
                    furthest_idx = np.argmax(distances)
                    max_dist = distances[furthest_idx]

                    if max_dist > max_dist_overall:
                        max_dist_overall = max_dist
                        best_pair = (idx, furthest_idx)

            start_idx, end_idx = best_pair
            endpoints = np.array([skel_points[start_idx], skel_points[end_idx]])

            # Step 4: Trace path
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

    analyzer = CustomOptimizedAnalyzer(mask, use_convolution_filter=(not disable_convolution))

    t0 = time.time()
    try:
        analyzer.analyze()
        elapsed = time.time() - t0
        return elapsed
    except Exception as e:
        return None


def main():
    print("\n" + "="*80)
    print("DIAGNOSTIC: Finding the Performance Bottleneck")
    print("="*80 + "\n")

    masks = load_real_masks(num_samples=3)

    if not masks:
        print("Could not load real masks")
        return

    print(f"Testing on {len(masks)} real embryo masks\n")

    for mask, name in masks:
        print(f"\n{'='*80}")
        print(f"Testing: {name}")
        print(f"Mask shape: {mask.shape}, area: {mask.sum():,} px")
        print(f"{'='*80}")

        # Test original
        t_orig = 0
        try:
            analyzer_orig = GeodesicCenterlineAnalyzer(mask, fast=True)
            t0 = time.time()
            analyzer_orig.analyze()
            t_orig = time.time() - t0
            print(f"Original:                         {t_orig:.4f}s")
        except Exception as e:
            print(f"Original failed: {e}")

        # Test optimized (both optimizations)
        t_both = test_component(mask, name, disable_thinning=False, disable_convolution=False)
        if t_both:
            speedup_both = t_orig / t_both
            print(f"Optimized (thinning + conv):      {t_both:.4f}s  ({speedup_both:.2f}x)")

        # Test with only thinning
        t_thin = test_component(mask, name, disable_thinning=False, disable_convolution=True)
        if t_thin:
            speedup_thin = t_orig / t_thin
            print(f"Optimized (only thinning):        {t_thin:.4f}s  ({speedup_thin:.2f}x)")

        # Test with only convolution
        t_conv = test_component(mask, name, disable_thinning=True, disable_convolution=False)
        if t_conv:
            speedup_conv = t_orig / t_conv
            print(f"Optimized (only convolution):     {t_conv:.4f}s  ({speedup_conv:.2f}x)")

        # Test original without thinning (for comparison)
        t_none = test_component(mask, name, disable_thinning=True, disable_convolution=True)
        if t_none:
            speedup_none = t_orig / t_none
            print(f"Optimized (no optimizations):     {t_none:.4f}s  ({speedup_none:.2f}x)")

        print()


if __name__ == '__main__':
    main()
