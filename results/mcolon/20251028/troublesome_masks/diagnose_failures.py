"""
Diagnose Geodesic Centerline Extraction Failures

Traces through the geodesic pipeline step-by-step to identify exact failure point:
1. Preprocessing & mask properties
2. Skeleton extraction
3. Endpoint detection
4. Geodesic path finding
5. B-spline fitting

For each failed embryo, captures intermediate results and identifies where the pipeline breaks.
"""

import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import morphology, measure
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.interpolate import splprep, splev

from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle
from segmentation_sandbox.scripts.utils.mask_cleaning import clean_embryo_mask
from segmentation_sandbox.scripts.body_axis_analysis.mask_preprocessing import apply_preprocessing
from segmentation_sandbox.scripts.body_axis_analysis.geodesic_method import GeodesicCenterlineAnalyzer


# =============================================================================
# LOAD EMBRYO
# =============================================================================

def load_embryo_data(snip_id: str):
    """Load embryo from CSV."""
    metadata_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build06_output")

    if "20251017" in snip_id:
        csv_path = metadata_dir / "df03_final_output_with_latents_20251017_combined.csv"
    elif "20250512" in snip_id:
        csv_path = metadata_dir / "df03_final_output_with_latents_20250512.csv"
    else:
        raise ValueError(f"Cannot determine CSV file for snip_id: {snip_id}")

    df = pd.read_csv(csv_path)
    embryo_row = df[df['snip_id'] == snip_id].iloc[0]

    mask = decode_mask_rle({
        'size': [int(embryo_row['mask_height_px']), int(embryo_row['mask_width_px'])],
        'counts': embryo_row['mask_rle']
    })

    um_per_pixel = embryo_row['height_um'] / int(embryo_row['mask_height_px'])

    return mask, um_per_pixel


# =============================================================================
# DIAGNOSTIC PIPELINE
# =============================================================================

def diagnose_embryo(snip_id: str):
    """
    Trace through geodesic pipeline step-by-step and identify failure point.

    Returns:
        Dictionary with diagnostic results for each stage
    """
    print(f"\n{'='*80}")
    print(f"DIAGNOSING: {snip_id}")
    print(f"{'='*80}\n")

    diagnostic = {
        'snip_id': snip_id,
        'stages': {}
    }

    # Load embryo
    print("Loading embryo...")
    mask_original, um_per_pixel = load_embryo_data(snip_id)
    print(f"  ✓ Loaded: {mask_original.shape}, area={mask_original.sum():,} px\n")

    # Stage 0: Clean mask
    print("STAGE 0: Mask Cleaning")
    print("-" * 40)
    mask_cleaned, _ = clean_embryo_mask(mask_original, verbose=False)
    print(f"  ✓ Cleaned area: {mask_cleaned.sum():,} px ({mask_cleaned.sum()/mask_original.sum()*100:.1f}%)\n")

    diagnostic['stages']['mask_cleaned'] = {
        'success': True,
        'area': int(mask_cleaned.sum()),
        'mask': mask_cleaned
    }

    # Stage 1: Preprocessing
    print("STAGE 1: Preprocessing (Gaussian blur)")
    print("-" * 40)
    try:
        mask_preprocessed = apply_preprocessing(mask_cleaned, method='gaussian_blur',
                                               sigma=10.0, threshold=0.7)

        # Analyze preprocessed mask
        props = measure.regionprops(measure.label(mask_preprocessed))[0] if mask_preprocessed.sum() > 0 else None

        if props:
            solidity = props.solidity
            convexity = props.area / props.convex_area if props.convex_area > 0 else 0
            n_components = len(measure.regionprops(measure.label(mask_preprocessed)))

            print(f"  ✓ Preprocessed area: {mask_preprocessed.sum():,} px ({mask_preprocessed.sum()/mask_cleaned.sum()*100:.1f}%)")
            print(f"  ✓ Solidity: {solidity:.3f}")
            print(f"  ✓ Convexity: {convexity:.3f}")
            print(f"  ✓ Connected components: {n_components}")

            diagnostic['stages']['preprocessing'] = {
                'success': True,
                'area': int(mask_preprocessed.sum()),
                'solidity': float(solidity),
                'convexity': float(convexity),
                'n_components': n_components,
                'mask': mask_preprocessed
            }
        else:
            print(f"  ✗ FAILED: Empty mask after preprocessing")
            diagnostic['stages']['preprocessing'] = {
                'success': False,
                'error': 'Empty mask after preprocessing'
            }
            return diagnostic

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        diagnostic['stages']['preprocessing'] = {
            'success': False,
            'error': str(e)
        }
        return diagnostic

    print()

    # Stage 2: Skeletonization
    print("STAGE 2: Skeleton Extraction")
    print("-" * 40)
    try:
        skeleton = morphology.skeletonize(mask_preprocessed)
        skeleton_points = np.column_stack(np.where(skeleton))
        n_skeleton_points = len(skeleton_points)

        if n_skeleton_points == 0:
            print(f"  ✗ FAILED: Empty skeleton")
            diagnostic['stages']['skeleton'] = {
                'success': False,
                'error': 'Empty skeleton - no points after skeletonization',
                'n_points': 0
            }
            return diagnostic

        # Check connectivity
        labeled_skel = measure.label(skeleton)
        n_components = labeled_skel.max()

        print(f"  ✓ Skeleton points: {n_skeleton_points:,}")
        print(f"  ✓ Connected components: {n_components}")

        if n_components > 1:
            print(f"  ⚠ WARNING: Skeleton is fragmented ({n_components} components)")
            # Get largest component
            largest_component = np.argmax([np.sum(labeled_skel == i) for i in range(1, n_components + 1)]) + 1
            skeleton_connected = (labeled_skel == largest_component)
            skeleton_points = np.column_stack(np.where(skeleton_connected))
            print(f"  → Using largest component: {len(skeleton_points):,} points")
        else:
            skeleton_connected = skeleton

        diagnostic['stages']['skeleton'] = {
            'success': True,
            'n_points': int(n_skeleton_points),
            'n_components': int(n_components),
            'skeleton': skeleton,
            'skeleton_points': skeleton_points
        }

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        diagnostic['stages']['skeleton'] = {
            'success': False,
            'error': str(e)
        }
        return diagnostic

    print()

    # Stage 3: Endpoint Detection
    print("STAGE 3: Endpoint Detection")
    print("-" * 40)
    try:
        # Use the same method as GeodesicCenterlineAnalyzer
        analyzer = GeodesicCenterlineAnalyzer(mask_preprocessed, um_per_pixel=um_per_pixel,
                                             bspline_smoothing=5.0, random_seed=42, fast=True)

        # Manually call the internal method to detect endpoints
        from scipy.ndimage import distance_transform_edt
        dist_transform = distance_transform_edt(mask_preprocessed)
        skel_points = np.column_stack(np.where(skeleton_connected))

        # Get distance values at skeleton points
        distances = dist_transform[skeleton_connected]

        if len(distances) == 0:
            print(f"  ✗ FAILED: No skeleton points for endpoint detection")
            diagnostic['stages']['endpoints'] = {
                'success': False,
                'error': 'No skeleton points available'
            }
            return diagnostic

        # Find farthest points (endpoints)
        dist_threshold = np.percentile(distances, 90)
        endpoint_candidates = skel_points[distances >= dist_threshold]

        if len(endpoint_candidates) < 2:
            print(f"  ✗ FAILED: Not enough endpoint candidates ({len(endpoint_candidates)})")
            diagnostic['stages']['endpoints'] = {
                'success': False,
                'error': f'Only {len(endpoint_candidates)} endpoint candidates found',
                'n_candidates': len(endpoint_candidates)
            }
            return diagnostic

        # Choose two endpoints that are farthest apart
        max_dist = 0
        endpoints = None
        for i in range(len(endpoint_candidates)):
            for j in range(i+1, len(endpoint_candidates)):
                d = np.linalg.norm(endpoint_candidates[i] - endpoint_candidates[j])
                if d > max_dist:
                    max_dist = d
                    endpoints = np.array([endpoint_candidates[i], endpoint_candidates[j]])

        if endpoints is None or len(endpoints) != 2:
            print(f"  ✗ FAILED: Could not find two distinct endpoints")
            diagnostic['stages']['endpoints'] = {
                'success': False,
                'error': 'Could not find two distinct endpoints'
            }
            return diagnostic

        print(f"  ✓ Found endpoints: {len(endpoint_candidates)} candidates")
        print(f"  ✓ Selected endpoints distance: {max_dist:.1f} px")
        print(f"  ✓ Endpoint 1: {endpoints[0]}")
        print(f"  ✓ Endpoint 2: {endpoints[1]}")

        diagnostic['stages']['endpoints'] = {
            'success': True,
            'n_candidates': len(endpoint_candidates),
            'endpoints': endpoints,
            'distance_apart': float(max_dist)
        }

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        diagnostic['stages']['endpoints'] = {
            'success': False,
            'error': str(e)
        }
        return diagnostic

    print()

    # Stage 4: Build Graph & Find Geodesic Path
    print("STAGE 4: Geodesic Path Finding")
    print("-" * 40)
    try:
        # Build graph using fast method
        point_to_index = {tuple(pt): idx for idx, pt in enumerate(skel_points)}

        rows = []
        cols = []
        weights = []

        # 8-connected neighborhood
        neighbour_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

        for idx, (y, x) in enumerate(skel_points):
            for dy, dx in neighbour_offsets:
                neighbour = (y + dy, x + dx)
                jdx = point_to_index.get(neighbour)
                if jdx is None or jdx <= idx:
                    continue
                rows.append(idx)
                cols.append(jdx)
                weights.append(np.sqrt(dx*dx + dy*dy))

        if not rows:
            print(f"  ✗ FAILED: No edges in skeleton graph (disconnected skeleton)")
            diagnostic['stages']['geodesic_path'] = {
                'success': False,
                'error': 'Skeleton graph has no edges - skeleton is completely disconnected'
            }
            return diagnostic

        # Create adjacency matrix
        data = weights + weights
        adj_matrix = csr_matrix(
            (data, (rows + cols, cols + rows)),
            shape=(len(skel_points), len(skel_points)),
        )

        print(f"  ✓ Built graph: {len(skel_points)} nodes, {len(rows)} edges")

        # Find indices of endpoints in skeleton points
        start_idx = None
        end_idx = None
        for idx, pt in enumerate(skel_points):
            if np.array_equal(pt, endpoints[0]):
                start_idx = idx
            if np.array_equal(pt, endpoints[1]):
                end_idx = idx

        if start_idx is None or end_idx is None:
            print(f"  ✗ FAILED: Endpoints not in skeleton points")
            diagnostic['stages']['geodesic_path'] = {
                'success': False,
                'error': f'Endpoints not found in skeleton (start={start_idx}, end={end_idx})'
            }
            return diagnostic

        # Find shortest path
        distances_dijkstra, predecessors = dijkstra(adj_matrix, indices=start_idx,
                                                    directed=False, return_predecessors=True)

        # Trace path
        path_indices = []
        current = end_idx
        while current != -9999 and current != start_idx:
            path_indices.append(current)
            current = predecessors[current]
            if len(path_indices) > len(skel_points):
                break
        path_indices.append(start_idx)
        path_indices = path_indices[::-1]

        if len(path_indices) < 2:
            print(f"  ✗ FAILED: Path has insufficient points ({len(path_indices)})")
            diagnostic['stages']['geodesic_path'] = {
                'success': False,
                'error': f'Geodesic path too short ({len(path_indices)} points)',
                'n_path_points': len(path_indices)
            }
            return diagnostic

        centerline = skel_points[path_indices]
        path_length = distances_dijkstra[end_idx]

        print(f"  ✓ Found path: {len(path_indices)} points")
        print(f"  ✓ Path length: {path_length:.1f} px")

        diagnostic['stages']['geodesic_path'] = {
            'success': True,
            'n_path_points': len(path_indices),
            'path_length_px': float(path_length),
            'centerline': centerline
        }

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        diagnostic['stages']['geodesic_path'] = {
            'success': False,
            'error': str(e)
        }
        return diagnostic

    print()

    # Stage 5: B-spline Fitting
    print("STAGE 5: B-spline Fitting")
    print("-" * 40)
    try:
        if len(centerline) < 4:
            print(f"  ✗ FAILED: Too few points for B-spline fitting ({len(centerline)} < 4)")
            diagnostic['stages']['bspline'] = {
                'success': False,
                'error': f'Too few points for spline fitting ({len(centerline)} < 4)'
            }
            return diagnostic

        # Fit B-spline
        tck, u = splprep([centerline[:, 1], centerline[:, 0]],  # Note: x,y order
                         s=5.0 * len(centerline), k=3)

        # Evaluate at 200 points
        u_fine = np.linspace(0, 1, 200)
        x_vals, y_vals = splev(u_fine, tck)

        print(f"  ✓ B-spline fitted successfully")
        print(f"  ✓ Smoothed to 200 points")

        diagnostic['stages']['bspline'] = {
            'success': True,
            'n_input_points': len(centerline),
            'n_output_points': 200
        }

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        diagnostic['stages']['bspline'] = {
            'success': False,
            'error': str(e)
        }
        return diagnostic

    print()
    print("="*80)
    print("✓ ALL STAGES PASSED - This should not fail!")
    print("="*80)

    return diagnostic


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_diagnostic(diagnostic: dict, output_path: Path):
    """
    Visualize diagnostic results showing each stage.
    """
    snip_id = diagnostic['snip_id']
    stages = diagnostic['stages']

    # Determine how many stages succeeded
    n_stages = len(stages)

    # Create figure with variable number of panels
    fig = plt.figure(figsize=(20, 10))

    panel_idx = 1

    # Panel: Cleaned mask
    if 'mask_cleaned' in stages and stages['mask_cleaned']['success']:
        ax = plt.subplot(2, 3, panel_idx)
        ax.imshow(stages['mask_cleaned']['mask'], cmap='gray')
        ax.set_title(f"✓ Stage 0: Cleaned Mask\nArea: {stages['mask_cleaned']['area']:,} px",
                     fontsize=11, fontweight='bold', color='green')
        ax.axis('off')
        panel_idx += 1

    # Panel: Preprocessed mask
    if 'preprocessing' in stages:
        ax = plt.subplot(2, 3, panel_idx)
        if stages['preprocessing']['success']:
            ax.imshow(stages['preprocessing']['mask'], cmap='gray')
            title = f"✓ Stage 1: Preprocessed\n"
            title += f"Area: {stages['preprocessing']['area']:,} px\n"
            title += f"Solidity: {stages['preprocessing']['solidity']:.3f}, Components: {stages['preprocessing']['n_components']}"
            ax.set_title(title, fontsize=11, fontweight='bold', color='green')
        else:
            ax.text(0.5, 0.5, f"✗ FAILED\n{stages['preprocessing'].get('error', 'Unknown error')}",
                   ha='center', va='center', fontsize=12, color='red', fontweight='bold')
            ax.set_title("✗ Stage 1: Preprocessing", fontsize=11, fontweight='bold', color='red')
        ax.axis('off')
        panel_idx += 1

    # Panel: Skeleton
    if 'skeleton' in stages:
        ax = plt.subplot(2, 3, panel_idx)
        if stages['skeleton']['success']:
            # Show skeleton on top of preprocessed mask
            if 'preprocessing' in stages and stages['preprocessing']['success']:
                ax.imshow(stages['preprocessing']['mask'], cmap='gray', alpha=0.3)
            ax.imshow(stages['skeleton']['skeleton'], cmap='Reds', alpha=0.7)
            title = f"✓ Stage 2: Skeleton\n"
            title += f"Points: {stages['skeleton']['n_points']:,}\n"
            title += f"Components: {stages['skeleton']['n_components']}"
            ax.set_title(title, fontsize=11, fontweight='bold', color='green')
        else:
            ax.text(0.5, 0.5, f"✗ FAILED\n{stages['skeleton'].get('error', 'Unknown error')}",
                   ha='center', va='center', fontsize=12, color='red', fontweight='bold')
            ax.set_title("✗ Stage 2: Skeleton", fontsize=11, fontweight='bold', color='red')
        ax.axis('off')
        panel_idx += 1

    # Panel: Endpoints
    if 'endpoints' in stages:
        ax = plt.subplot(2, 3, panel_idx)
        if stages['endpoints']['success']:
            if 'skeleton' in stages and stages['skeleton']['success']:
                ax.imshow(stages['skeleton']['skeleton'], cmap='gray', alpha=0.5)
                endpoints = stages['endpoints']['endpoints']
                ax.plot(endpoints[:, 1], endpoints[:, 0], 'ro', markersize=15,
                       markeredgecolor='white', markeredgewidth=2, label='Endpoints')
            title = f"✓ Stage 3: Endpoints\n"
            title += f"Distance: {stages['endpoints']['distance_apart']:.1f} px"
            ax.set_title(title, fontsize=11, fontweight='bold', color='green')
            ax.legend()
        else:
            ax.text(0.5, 0.5, f"✗ FAILED\n{stages['endpoints'].get('error', 'Unknown error')}",
                   ha='center', va='center', fontsize=12, color='red', fontweight='bold')
            ax.set_title("✗ Stage 3: Endpoints", fontsize=11, fontweight='bold', color='red')
        ax.axis('off')
        panel_idx += 1

    # Panel: Geodesic Path
    if 'geodesic_path' in stages:
        ax = plt.subplot(2, 3, panel_idx)
        if stages['geodesic_path']['success']:
            if 'preprocessing' in stages and stages['preprocessing']['success']:
                ax.imshow(stages['preprocessing']['mask'], cmap='gray', alpha=0.5)
            centerline = stages['geodesic_path']['centerline']
            ax.plot(centerline[:, 1], centerline[:, 0], 'r-', linewidth=2.5, label='Geodesic Path')
            ax.plot(centerline[0, 1], centerline[0, 0], 'go', markersize=10, label='Start')
            ax.plot(centerline[-1, 1], centerline[-1, 0], 'bo', markersize=10, label='End')
            title = f"✓ Stage 4: Geodesic Path\n"
            title += f"Points: {stages['geodesic_path']['n_path_points']}\n"
            title += f"Length: {stages['geodesic_path']['path_length_px']:.1f} px"
            ax.set_title(title, fontsize=11, fontweight='bold', color='green')
            ax.legend()
        else:
            ax.text(0.5, 0.5, f"✗ FAILED\n{stages['geodesic_path'].get('error', 'Unknown error')}",
                   ha='center', va='center', fontsize=12, color='red', fontweight='bold')
            ax.set_title("✗ Stage 4: Geodesic Path", fontsize=11, fontweight='bold', color='red')
        ax.axis('off')
        panel_idx += 1

    # Panel: B-spline
    if 'bspline' in stages:
        ax = plt.subplot(2, 3, panel_idx)
        if stages['bspline']['success']:
            ax.text(0.5, 0.5, f"✓ B-spline Fitting\nSuccessfully fitted!",
                   ha='center', va='center', fontsize=12, color='green', fontweight='bold')
            ax.set_title("✓ Stage 5: B-spline", fontsize=11, fontweight='bold', color='green')
        else:
            ax.text(0.5, 0.5, f"✗ FAILED\n{stages['bspline'].get('error', 'Unknown error')}",
                   ha='center', va='center', fontsize=12, color='red', fontweight='bold')
            ax.set_title("✗ Stage 5: B-spline", fontsize=11, fontweight='bold', color='red')
        ax.axis('off')

    plt.suptitle(f"Diagnostic: {snip_id}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved diagnostic visualization: {output_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run diagnostic on failed embryos."""

    # Failed embryos from Phase 1
    failed_embryos = [
        "20251017_combined_C04_e01_t0114",
        "20251017_combined_F11_e01_t0065",
    ]

    output_dir = Path(__file__).parent

    print("\n" + "="*80)
    print("PHASE 2: DETAILED FAILURE POINT ANALYSIS")
    print("="*80)
    print(f"Diagnosing {len(failed_embryos)} failed embryos")
    print("="*80)

    all_diagnostics = []

    for snip_id in failed_embryos:
        diagnostic = diagnose_embryo(snip_id)
        all_diagnostics.append(diagnostic)

        # Save visualization
        safe_filename = snip_id.replace("/", "_").replace("\\", "_")
        viz_path = output_dir / f"{safe_filename}_diagnostic.png"
        plot_diagnostic(diagnostic, viz_path)

    # Save diagnostic summary
    summary_data = []
    for diag in all_diagnostics:
        row = {'snip_id': diag['snip_id']}
        for stage_name, stage_data in diag['stages'].items():
            row[f'{stage_name}_success'] = stage_data['success']
            if not stage_data['success']:
                row[f'{stage_name}_error'] = stage_data.get('error', 'Unknown')
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    csv_path = output_dir / "failure_analysis.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved failure analysis: {csv_path}")

    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
