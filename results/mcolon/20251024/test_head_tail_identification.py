"""
Test Head vs Tail Identification Methods

Tests 3 methods on 36 random embryo masks:
1. Local Area - Head has larger local mask area
2. Taper Direction - Width decreases from head to tail
3. Local Eccentricity - Head is more circular (lower eccentricity)

Creates consistency heatmap and visual validation.
"""

import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy import ndimage
from skimage import morphology, measure
from sklearn.metrics import cohen_kappa_score
import warnings
warnings.filterwarnings('ignore')

from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle
from segmentation_sandbox.scripts.utils.mask_cleaning import clean_embryo_mask


# ============================================================================
# GEODESIC CENTERLINE EXTRACTION (reused from previous scripts)
# ============================================================================

def extract_geodesic_centerline(mask: np.ndarray):
    """
    Extract centerline using geodesic skeleton method.

    Returns:
        centerline: (N, 2) array of (x, y) coordinates
        endpoints: (2, 2) array of [endpoint1, endpoint2]
        skeleton: Binary skeleton image
    """
    # Skeletonize
    skeleton = morphology.skeletonize(mask)
    y_skel, x_skel = np.where(skeleton)

    if len(y_skel) < 2:
        raise ValueError("Skeleton has too few points")

    skel_points = np.column_stack([x_skel, y_skel])

    # Build graph (8-connected neighbors)
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
        raise ValueError("Skeleton graph has no edges")

    # Create adjacency matrix
    rows = [e[0] for e in edges] + [e[1] for e in edges]
    cols = [e[1] for e in edges] + [e[0] for e in edges]
    data = weights + weights
    adj_matrix = csr_matrix((data, (rows, cols)), shape=(n_points, n_points))

    # Find endpoints with maximum geodesic distance
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

    # Trace path from start to end
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


# ============================================================================
# METHOD 1: LOCAL AREA
# ============================================================================

def identify_by_local_area(mask: np.ndarray, endpoint1: np.ndarray,
                           endpoint2: np.ndarray, radius: int = 50) -> int:
    """
    Head has larger local mask area.

    Args:
        mask: Binary mask
        endpoint1: (x, y) coordinates of first endpoint
        endpoint2: (x, y) coordinates of second endpoint
        radius: Radius for local area measurement

    Returns:
        0 if endpoint1 is head, 1 if endpoint2 is head
    """
    h, w = mask.shape

    # Create circular mask for each endpoint
    def count_area_in_circle(center, radius):
        y, x = np.ogrid[:h, :w]
        circle_mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        return np.sum(mask & circle_mask)

    area1 = count_area_in_circle(endpoint1, radius)
    area2 = count_area_in_circle(endpoint2, radius)

    # Larger area = head
    return 1 if area2 > area1 else 0


# ============================================================================
# METHOD 2: TAPER DIRECTION
# ============================================================================

def identify_by_taper_direction(mask: np.ndarray, centerline: np.ndarray,
                                n_samples: int = 20, window_size: int = 10) -> int:
    """
    Width decreases from head to tail.

    Args:
        mask: Binary mask
        centerline: (N, 2) array of centerline points
        n_samples: Number of points along centerline to sample
        window_size: Window for width measurement

    Returns:
        0 if centerline[0] is head, 1 if centerline[-1] is head
    """
    if len(centerline) < n_samples:
        n_samples = len(centerline)

    # Sample points along centerline
    indices = np.linspace(0, len(centerline)-1, n_samples, dtype=int)
    sample_points = centerline[indices]

    # Measure width at each point
    widths = []
    for i, point in enumerate(sample_points):
        if i == 0 or i == len(sample_points) - 1:
            # For endpoints, just measure local radius
            widths.append(_measure_local_width(mask, point, window_size))
        else:
            # For middle points, measure perpendicular width
            # Get tangent direction
            if i < len(sample_points) - 1:
                tangent = sample_points[i+1] - sample_points[i-1]
            else:
                tangent = sample_points[i] - sample_points[i-1]

            # Perpendicular direction
            perp = np.array([-tangent[1], tangent[0]])
            perp = perp / (np.linalg.norm(perp) + 1e-10)

            # Measure width in perpendicular direction
            width = _measure_width_along_direction(mask, point, perp, window_size)
            widths.append(width)

    widths = np.array(widths)

    # Compute width gradient (positive = increasing toward end)
    # Use linear regression to get overall trend
    x = np.arange(len(widths))
    slope = np.polyfit(x, widths, 1)[0]

    # If slope is negative, width decreases from start to end
    # So start is head (return 0)
    # If slope is positive, width increases from start to end
    # So end is head (return 1)
    return 1 if slope > 0 else 0


def _measure_local_width(mask: np.ndarray, point: np.ndarray, radius: int) -> float:
    """Measure local width as diameter of circle around point."""
    h, w = mask.shape
    y, x = np.ogrid[:h, :w]
    circle_mask = (x - point[0])**2 + (y - point[1])**2 <= radius**2
    area = np.sum(mask & circle_mask)
    # Approximate width as 2 * sqrt(area/pi)
    if area > 0:
        return 2 * np.sqrt(area / np.pi)
    return 0.0


def _measure_width_along_direction(mask: np.ndarray, point: np.ndarray,
                                    direction: np.ndarray, max_dist: int) -> float:
    """Measure width by scanning in perpendicular direction."""
    # Scan in both directions perpendicular to centerline
    dist_pos = 0
    dist_neg = 0

    # Positive direction
    for d in range(1, max_dist):
        p = point + d * direction
        if 0 <= p[1] < mask.shape[0] and 0 <= p[0] < mask.shape[1]:
            if mask[int(p[1]), int(p[0])]:
                dist_pos = d
            else:
                break
        else:
            break

    # Negative direction
    for d in range(1, max_dist):
        p = point - d * direction
        if 0 <= p[1] < mask.shape[0] and 0 <= p[0] < mask.shape[1]:
            if mask[int(p[1]), int(p[0])]:
                dist_neg = d
            else:
                break
        else:
            break

    return dist_pos + dist_neg


# ============================================================================
# METHOD 3: LOCAL ECCENTRICITY
# ============================================================================

def identify_by_local_eccentricity(mask: np.ndarray, endpoint1: np.ndarray,
                                   endpoint2: np.ndarray, radius: int = 50) -> int:
    """
    Head is more circular (lower eccentricity).

    Args:
        mask: Binary mask
        endpoint1: (x, y) coordinates of first endpoint
        endpoint2: (x, y) coordinates of second endpoint
        radius: Radius for local region

    Returns:
        0 if endpoint1 is head, 1 if endpoint2 is head
    """
    ecc1 = _compute_local_eccentricity(mask, endpoint1, radius)
    ecc2 = _compute_local_eccentricity(mask, endpoint2, radius)

    # Lower eccentricity = more circular = head
    return 1 if ecc2 < ecc1 else 0


def _compute_local_eccentricity(mask: np.ndarray, center: np.ndarray,
                               radius: int) -> float:
    """Compute eccentricity of local region."""
    h, w = mask.shape

    # Extract local region
    y, x = np.ogrid[:h, :w]
    circle_mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    local_mask = mask & circle_mask

    # Find region properties
    labeled = measure.label(local_mask)
    if labeled.max() == 0:
        return 1.0  # No region, return max eccentricity

    # Get largest connected component
    props = measure.regionprops(labeled)
    if len(props) == 0:
        return 1.0

    # Sort by area and take largest
    props = sorted(props, key=lambda x: x.area, reverse=True)

    # Return eccentricity (0 = circle, 1 = line)
    return props[0].eccentricity


# ============================================================================
# TESTING FRAMEWORK
# ============================================================================

def test_head_tail_methods(csv_path: Path, n_embryos: int = 36,
                          random_seed: int = 42):
    """
    Test head/tail identification methods on random embryos.

    Args:
        csv_path: Path to CSV with mask data
        n_embryos: Number of embryos to test (will make 6x6 grid)
        random_seed: Random seed for reproducibility
    """
    print("="*70)
    print("HEAD vs TAIL IDENTIFICATION - METHOD COMPARISON")
    print("="*70)

    # Load data
    print(f"\nLoading data from {csv_path.name}...")
    df = pd.read_csv(csv_path)
    print(f"  Total embryos in CSV: {len(df)}")

    # Sample random embryos
    np.random.seed(random_seed)
    sample_df = df.sample(n=min(n_embryos, len(df)), random_state=random_seed)
    print(f"  Sampled: {len(sample_df)} embryos")

    # Results storage
    results = []

    # Process each embryo
    for idx, (_, row) in enumerate(sample_df.iterrows()):
        snip_id = row['snip_id']
        print(f"\n[{idx+1}/{len(sample_df)}] Processing {snip_id}...")

        try:
            # Decode mask
            mask_rle = row['mask_rle']
            mask = decode_mask_rle({
                'size': [int(row['mask_height_px']), int(row['mask_width_px'])],
                'counts': mask_rle
            })

            # Ensure mask is C-contiguous
            mask = np.ascontiguousarray(mask.astype(np.uint8))

            print(f"  Original mask: {mask.shape}, area: {mask.sum()} px")

            # CLEAN MASK - Remove artifacts
            mask_cleaned, cleaning_stats = clean_embryo_mask(mask, verbose=True)

            # Extract centerline from CLEANED mask
            centerline, endpoints, skeleton = extract_geodesic_centerline(mask_cleaned)
            print(f"  Centerline: {len(centerline)} points")
            print(f"  Endpoint 1: {endpoints[0]}")
            print(f"  Endpoint 2: {endpoints[1]}")

            # Apply all 3 methods on CLEANED mask
            method1_result = identify_by_local_area(mask_cleaned, endpoints[0], endpoints[1])
            method2_result = identify_by_taper_direction(mask_cleaned, centerline)
            method3_result = identify_by_local_eccentricity(mask_cleaned, endpoints[0], endpoints[1])

            print(f"  Method 1 (Local Area):        {method1_result} ({'endpoint2' if method1_result else 'endpoint1'} is head)")
            print(f"  Method 2 (Taper Direction):   {method2_result} ({'endpoint2' if method2_result else 'endpoint1'} is head)")
            print(f"  Method 3 (Local Eccentricity): {method3_result} ({'endpoint2' if method3_result else 'endpoint1'} is head)")

            # Compute agreement
            votes = [method1_result, method2_result, method3_result]
            agreement = sum(votes) / len(votes)
            consensus = "endpoint2" if agreement > 0.5 else "endpoint1"

            print(f"  Consensus: {consensus} (agreement: {agreement:.2f})")

            # Store results
            results.append({
                'snip_id': snip_id,
                'mask_area': mask_cleaned.sum(),
                'cleaning_stats': cleaning_stats,
                'centerline_length': len(centerline),
                'endpoint1_x': endpoints[0][0],
                'endpoint1_y': endpoints[0][1],
                'endpoint2_x': endpoints[1][0],
                'endpoint2_y': endpoints[1][1],
                'method1_local_area': method1_result,
                'method2_taper_direction': method2_result,
                'method3_local_eccentricity': method3_result,
                'agreement_score': agreement,
                'consensus': consensus,
                'mask': mask_cleaned,  # Store cleaned mask for visualization
                'mask_original': mask,  # Keep original for comparison
                'centerline': centerline,
                'endpoints': endpoints,
                'skeleton': skeleton
            })

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()

    return results


def analyze_consistency(results: list):
    """Analyze agreement between methods."""
    print("\n" + "="*70)
    print("CONSISTENCY ANALYSIS")
    print("="*70)

    # Extract method results
    method1 = np.array([r['method1_local_area'] for r in results])
    method2 = np.array([r['method2_taper_direction'] for r in results])
    method3 = np.array([r['method3_local_eccentricity'] for r in results])

    n = len(results)

    # Pairwise agreement
    agreement_12 = np.sum(method1 == method2) / n
    agreement_13 = np.sum(method1 == method3) / n
    agreement_23 = np.sum(method2 == method3) / n

    print(f"\nPairwise Agreement:")
    print(f"  Method 1 vs Method 2: {agreement_12:.2%}")
    print(f"  Method 1 vs Method 3: {agreement_13:.2%}")
    print(f"  Method 2 vs Method 3: {agreement_23:.2%}")

    # Three-way agreement
    all_agree = np.sum((method1 == method2) & (method2 == method3)) / n
    print(f"\nAll three methods agree: {all_agree:.2%}")

    # Cohen's kappa
    kappa_12 = cohen_kappa_score(method1, method2)
    kappa_13 = cohen_kappa_score(method1, method3)
    kappa_23 = cohen_kappa_score(method2, method3)

    print(f"\nCohen's Kappa (inter-rater reliability):")
    print(f"  Method 1 vs Method 2: {kappa_12:.3f}")
    print(f"  Method 1 vs Method 3: {kappa_13:.3f}")
    print(f"  Method 2 vs Method 3: {kappa_23:.3f}")
    print(f"  (κ > 0.8 = strong agreement, 0.6-0.8 = substantial, <0.6 = moderate)")

    # Create agreement matrix for heatmap
    agreement_matrix = np.array([
        [1.0, agreement_12, agreement_13],
        [agreement_12, 1.0, agreement_23],
        [agreement_13, agreement_23, 1.0]
    ])

    return agreement_matrix


def visualize_results(results: list, output_dir: Path):
    """Create visualizations."""
    n_embryos = len(results)
    grid_size = int(np.ceil(np.sqrt(n_embryos)))

    print(f"\nCreating visualizations ({grid_size}x{grid_size} grid)...")

    # Figure 1: Grid of embryos with head/tail annotations
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(24, 24))
    axes = axes.flatten()

    for i, result in enumerate(results):
        ax = axes[i]
        mask = result['mask']
        centerline = result['centerline']
        endpoints = result['endpoints']

        # Determine head/tail from consensus
        if result['consensus'] == 'endpoint1':
            head = endpoints[0]
            tail = endpoints[1]
        else:
            head = endpoints[1]
            tail = endpoints[0]

        # Plot mask and centerline
        ax.imshow(mask, cmap='gray', alpha=0.5)
        ax.plot(centerline[:, 0], centerline[:, 1], 'b-', linewidth=2, alpha=0.7)

        # Mark head (green) and tail (blue)
        ax.scatter(head[0], head[1], c='green', s=300, marker='o',
                  edgecolors='white', linewidths=3, label='Head', zorder=5)
        ax.scatter(tail[0], tail[1], c='blue', s=300, marker='s',
                  edgecolors='white', linewidths=3, label='Tail', zorder=5)

        # Title with method votes
        m1 = "✓" if result['method1_local_area'] == (1 if result['consensus'] == 'endpoint2' else 0) else "✗"
        m2 = "✓" if result['method2_taper_direction'] == (1 if result['consensus'] == 'endpoint2' else 0) else "✗"
        m3 = "✓" if result['method3_local_eccentricity'] == (1 if result['consensus'] == 'endpoint2' else 0) else "✗"

        ax.set_title(f"{result['snip_id'][:20]}...\nM1:{m1} M2:{m2} M3:{m3}",
                    fontsize=8)
        ax.axis('equal')
        ax.invert_yaxis()
        ax.axis('off')
        ax.legend(fontsize=6, loc='upper right')

    # Hide unused subplots
    for i in range(len(results), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    output_path1 = output_dir / "head_tail_test_grid_36embryos.png"
    plt.savefig(output_path1, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path1}")
    plt.close()

    # Figure 2: Agreement heatmap
    agreement_matrix = analyze_consistency(results)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(agreement_matrix, annot=True, fmt='.2%', cmap='RdYlGn',
                vmin=0, vmax=1, square=True, cbar_kws={'label': 'Agreement'},
                xticklabels=['Local Area', 'Taper Direction', 'Local Eccentricity'],
                yticklabels=['Local Area', 'Taper Direction', 'Local Eccentricity'],
                ax=ax)
    ax.set_title('Method Agreement Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path2 = output_dir / "method_consistency_heatmap.png"
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path2}")
    plt.close()

    # Save results CSV
    results_df = pd.DataFrame([{
        'snip_id': r['snip_id'],
        'method1_local_area': r['method1_local_area'],
        'method2_taper_direction': r['method2_taper_direction'],
        'method3_local_eccentricity': r['method3_local_eccentricity'],
        'agreement_score': r['agreement_score'],
        'consensus': r['consensus']
    } for r in results])

    output_path3 = output_dir / "head_tail_results_36embryos.csv"
    results_df.to_csv(output_path3, index=False)
    print(f"  ✓ Saved: {output_path3}")


def main():
    """Main execution."""
    csv_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build06_output/df03_final_output_with_latents_20251017_part2.csv")
    output_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251024")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test methods on 36 embryos
    results = test_head_tail_methods(csv_path, n_embryos=36, random_seed=42)

    if len(results) == 0:
        print("\n✗ No results to analyze!")
        return

    # Analyze consistency
    analyze_consistency(results)

    # Create visualizations
    visualize_results(results, output_dir)

    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
