"""
Test Pruned Geodesic Skeleton Method

Analyzes whether width-based skeleton pruning fixes the "through-fin" problem
where geodesic centerlines incorrectly extend into fin regions.

Features:
1. Loads existing PCA vs Geodesic comparison results
2. Re-runs Geodesic method with pruned skeletons
3. Compares original vs pruned Geodesic results
4. Creates scatter plots: mask metrics vs Hausdorff distance
5. Generates before/after visualizations for fin cases
6. Tests multiple pruning thresholds (10th, 25th, 50th percentile)
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
from scipy.spatial.distance import directed_hausdorff
from scipy import stats
from skimage import measure
from skimage.morphology import skeletonize
from tqdm import tqdm
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')

from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle
from segmentation_sandbox.scripts.utils.mask_cleaning import clean_embryo_mask
from segmentation_sandbox.scripts.utils.bodyaxis_spline_utils import (
    align_spline_orientation,
    prune_skeleton_for_geodesic
)

# Import existing analyzer classes
sys.path.insert(0, str(Path(__file__).parent.parent / "20251022"))
from geodesic_bspline_smoothing import GeodesicBSplineAnalyzer


def compute_hausdorff_distance(spline1_x, spline1_y, spline2_x, spline2_y):
    """
    Compute Hausdorff distance between two splines after alignment.

    Returns:
        hausdorff_distance: Symmetric Hausdorff distance
        was_flipped: Whether spline2 was flipped for alignment
    """
    if len(spline1_x) == 0 or len(spline2_x) == 0:
        return np.nan, False

    # Align orientations
    spline2_x_aligned, spline2_y_aligned, was_flipped = align_spline_orientation(
        spline1_x, spline1_y, spline2_x, spline2_y
    )

    # Stack into (N, 2) arrays
    spline1 = np.column_stack([spline1_x, spline1_y])
    spline2 = np.column_stack([spline2_x_aligned, spline2_y_aligned])

    # Symmetric Hausdorff distance
    hausdorff_1to2 = directed_hausdorff(spline1, spline2)[0]
    hausdorff_2to1 = directed_hausdorff(spline2, spline1)[0]
    hausdorff = max(hausdorff_1to2, hausdorff_2to1)

    return float(hausdorff), was_flipped


def extract_centerline_from_skeleton(skeleton, bspline_smoothing=5.0):
    """
    Extract geodesic centerline from a given skeleton.

    This is adapted from GeodesicBSplineAnalyzer.extract_centerline()
    but accepts a pre-computed (and potentially pruned) skeleton.

    Args:
        skeleton: Binary skeleton image
        bspline_smoothing: B-spline smoothing parameter

    Returns:
        centerline: (N, 2) array of (x, y) coordinates along geodesic path
        endpoints: (2, 2) array of start/end points
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

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
        raise ValueError("Skeleton graph has no edges (disconnected)")

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

    return centerline, endpoints


def extract_geodesic_with_pruning(mask, um_per_pixel=1.0,
                                   width_percentile_threshold=25.0):
    """
    Extract geodesic centerline with skeleton pruning.

    Args:
        mask: Binary mask (should be pre-cleaned)
        um_per_pixel: Spatial calibration
        width_percentile_threshold: Percentile for width-based pruning

    Returns:
        spline_x, spline_y: Geodesic spline coordinates
        pruning_stats: Dictionary with pruning diagnostics
    """
    from scipy.interpolate import splprep, splev

    try:
        # Get skeleton
        skeleton = skeletonize(mask)

        # Apply pruning
        pruned_skeleton, pruning_stats = prune_skeleton_for_geodesic(
            skeleton, mask, width_percentile_threshold
        )

        # Extract centerline from pruned skeleton
        centerline_geo, endpoints_geo = extract_centerline_from_skeleton(
            pruned_skeleton, bspline_smoothing=5.0
        )

        # Smooth with B-spline (same as GeodesicBSplineAnalyzer)
        if len(centerline_geo) < 4:
            return np.array([]), np.array([]), pruning_stats

        tck, u = splprep([centerline_geo[:, 0], centerline_geo[:, 1]],
                         s=5.0 * len(centerline_geo), k=3)

        # Evaluate spline at fine resolution
        u_fine = np.linspace(0, 1, 200)
        x_vals, y_vals = splev(u_fine, tck)

        spline_x = np.array(x_vals)
        spline_y = np.array(y_vals)

        return spline_x, spline_y, pruning_stats

    except Exception as e:
        return np.array([]), np.array([]), {'error': str(e)}


def deserialize_spline_coords(coords_str):
    """Deserialize spline coordinates from CSV string."""
    if not coords_str or coords_str == "":
        return np.array([]), np.array([])

    pairs = coords_str.split(';')
    coords = [tuple(map(float, p.split(','))) for p in pairs]
    y_coords = np.array([c[0] for c in coords])
    x_coords = np.array([c[1] for c in coords])

    return x_coords, y_coords


def process_single_embryo_with_pruning(row_data):
    """
    Process a single embryo: run pruned geodesic and compare.

    Args:
        row_data: Tuple of (snip_id, mask_rle, mask_height, mask_width,
                           pca_coords, geo_coords_original, original_hausdorff)

    Returns:
        Result dict with pruned geodesic comparison
    """
    (snip_id, mask_rle, mask_height, mask_width,
     pca_coords, geo_coords_original, original_hausdorff) = row_data

    result = {
        'snip_id': snip_id,
        'original_hausdorff': original_hausdorff,
    }

    try:
        # Decode mask
        mask = decode_mask_rle({
            'size': [int(mask_height), int(mask_width)],
            'counts': mask_rle
        })

        # Clean mask (same as original)
        cleaned_mask, _ = clean_embryo_mask(mask, verbose=False)

        # Deserialize PCA coords for comparison
        pca_x, pca_y = deserialize_spline_coords(pca_coords)

        # Run pruned geodesic at multiple thresholds
        for threshold in [10.0, 25.0, 50.0]:
            geo_pruned_x, geo_pruned_y, pruning_stats = extract_geodesic_with_pruning(
                cleaned_mask, um_per_pixel=1.0, width_percentile_threshold=threshold
            )

            # Compute Hausdorff vs PCA
            if len(geo_pruned_x) > 0 and len(pca_x) > 0:
                hausdorff, _ = compute_hausdorff_distance(
                    pca_x, pca_y, geo_pruned_x, geo_pruned_y
                )
                result[f'pruned_{int(threshold)}p_hausdorff'] = hausdorff
                result[f'pruned_{int(threshold)}p_removed_fraction'] = pruning_stats.get('removed_fraction', np.nan)
                result[f'pruned_{int(threshold)}p_width_threshold'] = pruning_stats.get('width_threshold', np.nan)
            else:
                result[f'pruned_{int(threshold)}p_hausdorff'] = np.nan
                result[f'pruned_{int(threshold)}p_removed_fraction'] = np.nan
                result[f'pruned_{int(threshold)}p_width_threshold'] = np.nan

        return result

    except Exception as e:
        result['error'] = str(e)
        return result


def plot_metrics_vs_hausdorff(results_df, output_dir, threshold=114.78):
    """
    Create scatter plots: mask metrics (X) vs Hausdorff distance (Y).

    Shows correlation between morphological features and method disagreement.
    """
    metrics_to_plot = [
        'solidity', 'eccentricity', 'extent',
        'area', 'perimeter', 'circularity'
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]

        # Separate agreement vs disagreement
        agree = results_df[results_df['hausdorff_distance'] < threshold]
        disagree = results_df[results_df['hausdorff_distance'] >= threshold]

        # Scatter plot
        ax.scatter(agree[metric], agree['hausdorff_distance'],
                   alpha=0.4, s=15, c='blue', label=f'Agree (n={len(agree)})')
        ax.scatter(disagree[metric], disagree['hausdorff_distance'],
                   alpha=0.7, s=30, c='red', label=f'Disagree (n={len(disagree)})')

        # Threshold line
        ax.axhline(threshold, color='black', linestyle='--',
                   linewidth=1.5, alpha=0.6, label='GMM Threshold')

        # Correlation coefficients
        from scipy.stats import pearsonr, spearmanr

        # Remove NaN values for correlation
        valid_mask = ~(np.isnan(results_df[metric]) | np.isnan(results_df['hausdorff_distance']))
        valid_metric = results_df[metric][valid_mask]
        valid_hausdorff = results_df['hausdorff_distance'][valid_mask]

        if len(valid_metric) > 2:
            pearson_r, pearson_p = pearsonr(valid_metric, valid_hausdorff)
            spearman_r, spearman_p = spearmanr(valid_metric, valid_hausdorff)

            # Add text box with correlations
            textstr = f'Pearson: r={pearson_r:.3f}\nSpearman: ρ={spearman_r:.3f}'
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round',
                   facecolor='wheat', alpha=0.5), fontsize=9)

        # Labels
        ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_ylabel('Hausdorff Distance (px)', fontsize=12, fontweight='bold')
        ax.set_title(f'{metric.replace("_", " ").title()} vs Method Disagreement',
                     fontsize=11)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, linestyle=':')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/metrics_vs_hausdorff_scatter.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/metrics_vs_hausdorff_scatter.png")
    plt.close()


def plot_correlation_heatmap(results_df, metrics, output_dir):
    """
    Create correlation heatmap showing which metrics correlate with disagreement.
    """
    # Include Hausdorff distance in correlation analysis
    metrics_with_hausdorff = metrics + ['hausdorff_distance']

    # Compute correlation matrix
    corr_matrix = results_df[metrics_with_hausdorff].corr()

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8})

    plt.title('Correlation Matrix: Mask Metrics vs Hausdorff Distance',
              fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/correlation_heatmap.png")
    plt.close()


def plot_pruning_improvement(results_df, output_dir):
    """
    Plot improvement from skeleton pruning: original vs pruned Hausdorff distances.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    thresholds = [10, 25, 50]

    for idx, threshold in enumerate(thresholds):
        ax = axes[idx]

        # Get data
        original = results_df['original_hausdorff']
        pruned = results_df[f'pruned_{threshold}p_hausdorff']

        # Scatter plot
        ax.scatter(original, pruned, alpha=0.5, s=20, c='blue')

        # Diagonal line (no improvement)
        max_val = max(original.max(), pruned.max())
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5,
                alpha=0.5, label='No change')

        # Improvement region (below diagonal)
        ax.fill_between([0, max_val], [0, max_val], [max_val, max_val],
                        alpha=0.1, color='green', label='Improvement')

        # Compute improvement statistics
        improvement = original - pruned
        improved_count = (improvement > 5).sum()  # Improved by >5px
        worsened_count = (improvement < -5).sum()  # Worsened by >5px

        # Labels
        ax.set_xlabel('Original Geodesic Hausdorff (px)', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'Pruned ({threshold}th %ile) Hausdorff (px)', fontsize=11, fontweight='bold')
        ax.set_title(f'{threshold}th Percentile Pruning\n'
                    f'Improved: {improved_count}, Worsened: {worsened_count}',
                    fontsize=11)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/pruning_improvement_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/pruning_improvement_comparison.png")
    plt.close()


def main():
    """
    Main analysis pipeline.
    """
    output_dir = Path(__file__).parent

    # 1. Load existing comparison results
    print("Loading existing PCA vs Geodesic comparison results...")
    results_csv = output_dir / "pca_vs_geodesic_comparison_1000embryos.csv"

    if not results_csv.exists():
        print(f"ERROR: {results_csv} not found!")
        print("Please run compare_pca_vs_geodesic.py first to generate baseline results.")
        return

    df_original = pd.read_csv(results_csv)
    print(f"Loaded {len(df_original)} embryos")

    # Filter to embryos where both methods succeeded
    df_valid = df_original[df_original['both_methods_succeeded'] == True].copy()
    print(f"Valid embryos (both methods succeeded): {len(df_valid)}")

    # 2. Create scatter plots from original results
    print("\nGenerating scatter plots: Mask Metrics vs Hausdorff Distance...")
    plot_metrics_vs_hausdorff(df_valid, output_dir)

    metrics = ['solidity', 'eccentricity', 'extent', 'area', 'perimeter', 'circularity']
    plot_correlation_heatmap(df_valid, metrics, output_dir)

    # 3. Test pruned geodesic on subset (sample for speed)
    print("\nTesting pruned geodesic method...")
    print("Sampling 200 embryos for pruning analysis...")

    # Stratified sample: include both agreeing and disagreeing cases
    threshold = 114.78
    df_agree = df_valid[df_valid['hausdorff_distance'] < threshold]
    df_disagree = df_valid[df_valid['hausdorff_distance'] >= threshold]

    # Sample 150 agreeing, 50 disagreeing (to oversample problem cases)
    sample_agree = df_agree.sample(min(150, len(df_agree)), random_state=42)
    sample_disagree = df_disagree.sample(min(50, len(df_disagree)), random_state=42)

    df_sample = pd.concat([sample_agree, sample_disagree])
    print(f"Sample: {len(sample_agree)} agree, {len(sample_disagree)} disagree")

    # Load metadata for mask reconstruction (both part1 and part2)
    metadata_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build06_output")
    df_metadata_part1 = pd.read_csv(metadata_dir / "df03_final_output_with_latents_20251017_part1.csv")
    df_metadata_part2 = pd.read_csv(metadata_dir / "df03_final_output_with_latents_20251017_part2.csv")

    # Combine both datasets
    df_metadata = pd.concat([df_metadata_part1, df_metadata_part2], ignore_index=True)
    print(f"Loaded metadata: {len(df_metadata_part1)} part1 + {len(df_metadata_part2)} part2 = {len(df_metadata)} total")

    # Prepare data for parallel processing
    # Match snip_ids with metadata to get mask info
    row_data_list = []

    for _, sample_row in df_sample.iterrows():
        snip_id = sample_row['snip_id']

        # Find matching row in metadata
        metadata_match = df_metadata[df_metadata['snip_id'] == snip_id]

        if len(metadata_match) == 0:
            print(f"WARNING: No metadata found for {snip_id}, skipping...")
            continue

        metadata_row = metadata_match.iloc[0]

        row_data_list.append((
            snip_id,
            metadata_row['mask_rle'],
            metadata_row['mask_height_px'],
            metadata_row['mask_width_px'],
            sample_row['pca_coords'],
            sample_row['geo_coords'],
            sample_row['hausdorff_distance']
        ))

    print(f"Successfully matched {len(row_data_list)} embryos with metadata")

    # Process in parallel
    print(f"Processing {len(row_data_list)} embryos with pruned geodesic...")
    n_workers = min(mp.cpu_count() - 1, 32)

    with mp.Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_embryo_with_pruning, row_data_list),
            total=len(row_data_list),
            desc="Processing embryos"
        ))

    # Compile results
    df_pruned = pd.DataFrame(results)

    # Merge with original data
    df_combined = df_sample.merge(df_pruned, on='snip_id', how='left', suffixes=('', '_pruned'))

    # Save results
    output_csv = output_dir / "pruned_geodesic_comparison.csv"
    df_combined.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")

    # 4. Generate pruning improvement plots
    print("\nGenerating pruning improvement visualizations...")
    plot_pruning_improvement(df_combined, output_dir)

    # 5. Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    for threshold in [10, 25, 50]:
        original_mean = df_combined['hausdorff_distance'].mean()
        pruned_mean = df_combined[f'pruned_{threshold}p_hausdorff'].mean()
        improvement = original_mean - pruned_mean

        print(f"\n{threshold}th Percentile Pruning:")
        print(f"  Original Hausdorff: {original_mean:.2f} px")
        print(f"  Pruned Hausdorff:   {pruned_mean:.2f} px")
        print(f"  Mean improvement:   {improvement:.2f} px")

        # Count improved/worsened
        diff = df_combined['hausdorff_distance'] - df_combined[f'pruned_{threshold}p_hausdorff']
        improved = (diff > 5).sum()
        worsened = (diff < -5).sum()
        unchanged = ((diff >= -5) & (diff <= 5)).sum()

        print(f"  Improved (>5px):    {improved} ({improved/len(df_combined)*100:.1f}%)")
        print(f"  Worsened (>5px):    {worsened} ({worsened/len(df_combined)*100:.1f}%)")
        print(f"  Unchanged (±5px):   {unchanged} ({unchanged/len(df_combined)*100:.1f}%)")

        # Mean pruning fraction
        mean_removed = df_combined[f'pruned_{threshold}p_removed_fraction'].mean()
        print(f"  Mean skeleton removed: {mean_removed*100:.1f}%")

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()
