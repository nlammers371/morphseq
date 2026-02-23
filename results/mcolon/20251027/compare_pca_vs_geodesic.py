"""
Compare PCA vs Geodesic Spline Extraction Methods

Analyzes when PCA and Geodesic methods produce different centerlines/splines,
and what morphological characteristics predict these differences.

Key features:
1. Dummy smart cleaning function (placeholder for future implementation)
2. Computes both PCA and Geodesic splines
3. Measures spline differences (mean distance, max distance, Hausdorff)
4. Records morphology metrics for decision rule analysis
5. Outputs CSV for automatic QC detection
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
from sklearn.mixture import GaussianMixture
from skimage import measure
from tqdm import tqdm
import multiprocessing as mp
import os
import warnings
warnings.filterwarnings('ignore')

from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle
from segmentation_sandbox.scripts.utils.mask_cleaning import clean_embryo_mask
from segmentation_sandbox.scripts.utils.bodyaxis_spline_utils import align_spline_orientation

# Import existing analyzer classes
sys.path.insert(0, str(Path(__file__).parent.parent / "20251022"))
from geodesic_bspline_smoothing import GeodesicBSplineAnalyzer
from test_pca_smoothing import PCACurvatureAnalyzer


def compute_mask_metrics(mask: np.ndarray):
    """
    Compute mask shape metrics for decision rules.

    Returns dict with metrics used to decide between PCA and Geodesic methods.
    """
    props = measure.regionprops(measure.label(mask))[0]

    metrics = {
        'area': props.area,
        'perimeter': props.perimeter,
        'solidity': props.solidity,
        'extent': props.extent,
        'eccentricity': props.eccentricity,
        'circularity': (props.perimeter ** 2) / (4 * np.pi * props.area),
        'perimeter_area_ratio': props.perimeter / np.sqrt(props.area),
    }

    return metrics


def compute_spline_differences(spline1_x, spline1_y, spline2_x, spline2_y):
    """
    Compute differences between two splines after aligning orientation.

    Args:
        spline1_x, spline1_y: First spline coordinates
        spline2_x, spline2_y: Second spline coordinates

    Returns:
        dict with difference metrics:
        - mean_distance: Average Euclidean distance
        - max_distance: Maximum distance
        - hausdorff_distance: Hausdorff distance (symmetric)
        - spline2_was_flipped: Boolean indicating if spline2 was flipped for alignment
    """
    if len(spline1_x) == 0 or len(spline2_x) == 0:
        return {
            'mean_distance': np.nan,
            'max_distance': np.nan,
            'hausdorff_distance': np.nan,
            'spline2_was_flipped': False,
        }

    # Align spline orientations before comparison
    spline2_x_aligned, spline2_y_aligned, was_flipped = align_spline_orientation(
        spline1_x, spline1_y, spline2_x, spline2_y
    )

    # Stack into (N, 2) arrays
    spline1 = np.column_stack([spline1_x, spline1_y])
    spline2 = np.column_stack([spline2_x_aligned, spline2_y_aligned])

    # Resample both splines to same number of points for point-wise comparison
    n_points = min(len(spline1), len(spline2), 100)

    # Resample spline1
    u1 = np.linspace(0, 1, len(spline1))
    u1_new = np.linspace(0, 1, n_points)
    spline1_resampled_x = np.interp(u1_new, u1, spline1[:, 0])
    spline1_resampled_y = np.interp(u1_new, u1, spline1[:, 1])
    spline1_resampled = np.column_stack([spline1_resampled_x, spline1_resampled_y])

    # Resample spline2
    u2 = np.linspace(0, 1, len(spline2))
    u2_new = np.linspace(0, 1, n_points)
    spline2_resampled_x = np.interp(u2_new, u2, spline2[:, 0])
    spline2_resampled_y = np.interp(u2_new, u2, spline2[:, 1])
    spline2_resampled = np.column_stack([spline2_resampled_x, spline2_resampled_y])

    # Point-wise distances
    distances = np.linalg.norm(spline1_resampled - spline2_resampled, axis=1)

    # Hausdorff distance (symmetric)
    hausdorff_1to2 = directed_hausdorff(spline1, spline2)[0]
    hausdorff_2to1 = directed_hausdorff(spline2, spline1)[0]
    hausdorff = max(hausdorff_1to2, hausdorff_2to1)

    return {
        'mean_distance': float(np.mean(distances)),
        'max_distance': float(np.max(distances)),
        'hausdorff_distance': float(hausdorff),
        'spline2_was_flipped': was_flipped,
    }


def serialize_spline_coords(x_coords: np.ndarray, y_coords: np.ndarray) -> str:
    """Serialize spline coordinates so they can be reconstructed later for plotting."""
    if len(x_coords) == 0 or len(y_coords) == 0:
        return ""
    return ';'.join(f"{float(y):.4f},{float(x):.4f}" for x, y in zip(x_coords, y_coords))


def compare_methods_single_embryo(
    mask: np.ndarray,
    snip_id: str,
    dataset_name: str = None,
    um_per_pixel: float = 1.0,
    store_splines: bool = True,
):
    """
    Compare PCA and Geodesic methods for a single embryo.

    Returns dict with all metrics, or None if either method fails.
    """
    result = {
        'snip_id': snip_id,
        'pca_coords': "",
        'geo_coords': "",
    }

    if dataset_name is not None:
        result['dataset'] = dataset_name

    try:
        # 1. Apply mask cleaning with conditional opening (solidity < 0.6)
        cleaned_mask, cleaning_stats = clean_embryo_mask(mask, verbose=False)
        result['opening_applied'] = not cleaning_stats['opening_skipped']
        result['opening_skipped'] = cleaning_stats['opening_skipped']
        result['solidity_before_opening'] = cleaning_stats['solidity_before']

        # 2. Compute mask shape metrics on cleaned mask
        mask_metrics = compute_mask_metrics(cleaned_mask)
        result.update(mask_metrics)

        # 3. Extract PCA spline
        try:
            pca_analyzer = PCACurvatureAnalyzer(cleaned_mask, um_per_pixel=um_per_pixel)
            centerline_pca = pca_analyzer.extract_centerline_pca(n_slices=100)
            arc_pca, curv_pca, spline_pca_x, spline_pca_y = pca_analyzer.compute_curvature(
                centerline_pca, smoothing=5.0
            )
            result['pca_success'] = True
            result['pca_n_points'] = len(spline_pca_x)
            if store_splines and len(spline_pca_x) > 0:
                result['pca_coords'] = serialize_spline_coords(spline_pca_x, spline_pca_y)
        except Exception as e:
            result['pca_success'] = False
            result['pca_error'] = str(e)
            result['pca_n_points'] = 0
            spline_pca_x = np.array([])
            spline_pca_y = np.array([])

        # 4. Extract Geodesic spline
        try:
            geo_analyzer = GeodesicBSplineAnalyzer(
                cleaned_mask, um_per_pixel=um_per_pixel, bspline_smoothing=5.0
            )
            # Extract centerline using geodesic skeleton
            centerline_geo, endpoints_geo, skeleton_geo = geo_analyzer.extract_centerline()
            # Smooth with B-spline
            smoothed_geo, tck_geo = geo_analyzer.smooth_with_bspline(centerline_geo)
            # Compute curvature
            arc_geo, curv_geo = geo_analyzer.compute_curvature(tck_geo)
            # Extract spline coordinates for comparison
            spline_geo_x = smoothed_geo[:, 0]
            spline_geo_y = smoothed_geo[:, 1]
            result['geodesic_success'] = True
            result['geodesic_n_points'] = len(spline_geo_x)
            if store_splines and len(spline_geo_x) > 0:
                result['geo_coords'] = serialize_spline_coords(spline_geo_x, spline_geo_y)
        except Exception as e:
            result['geodesic_success'] = False
            result['geodesic_error'] = str(e)
            result['geodesic_n_points'] = 0
            spline_geo_x = np.array([])
            spline_geo_y = np.array([])

        # 5. Compare splines if both succeeded
        if result['pca_success'] and result['geodesic_success']:
            differences = compute_spline_differences(
                spline_pca_x, spline_pca_y,
                spline_geo_x, spline_geo_y
            )
            result.update(differences)
            result['both_methods_succeeded'] = True
        else:
            result['both_methods_succeeded'] = False
            result['mean_distance'] = np.nan
            result['max_distance'] = np.nan
            result['hausdorff_distance'] = np.nan

        return result

    except Exception as e:
        result['error'] = str(e)
        return result


def process_single_row(row_data):
    """
    Process a single row for parallel execution.

    Args:
        row_data: Tuple of (snip_id, mask_rle, mask_height_px, mask_width_px, dataset_name)

    Returns:
        Result dict from compare_methods_single_embryo or None
    """
    if len(row_data) == 5:
        snip_id, mask_rle, mask_height_px, mask_width_px, dataset_name = row_data
    elif len(row_data) == 4:
        snip_id, mask_rle, mask_height_px, mask_width_px = row_data
        dataset_name = None
    else:
        raise ValueError(f"Unexpected row_data length: {len(row_data)} (expected 4 or 5)")

    try:
        # Decode mask
        mask = decode_mask_rle({
            'size': [int(mask_height_px), int(mask_width_px)],
            'counts': mask_rle
        })
        mask = np.ascontiguousarray(mask.astype(np.uint8))

        # Compare methods
        result = compare_methods_single_embryo(
            mask,
            snip_id,
            dataset_name=dataset_name,
            um_per_pixel=1.0,
            store_splines=True,
        )
        result['mask_height_px'] = int(mask_height_px)
        result['mask_width_px'] = int(mask_width_px)
        result.setdefault('dataset', dataset_name)
        return result

    except Exception as e:
        return {
            'snip_id': snip_id,
            'dataset': dataset_name,
            'error': str(e),
        }


def analyze_dataset(csv_paths, n_samples=1000, random_seed=42, n_jobs=None):
    """
    Analyze PCA vs Geodesic methods on multiple datasets.

    Args:
        csv_paths: List of paths to CSV files
        n_samples: Total number of embryos to sample across all datasets
        random_seed: Seed used for sampling
        n_jobs: Number of worker processes (None defaults to CPU count - 1)
    """
    csv_paths = [Path(p) for p in csv_paths]

    print(f"Loading {len(csv_paths)} dataset(s)...")

    # Load all datasets
    all_dfs = []
    for csv_path in csv_paths:
        print(f"  - {csv_path.name}")
        df = pd.read_csv(csv_path)
        dataset_name = csv_path.stem
        df['dataset'] = dataset_name
        df['source_dataset'] = dataset_name
        all_dfs.append(df)
        print(f"    {len(df)} embryos")

    # Combine and sample
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total embryos available: {len(combined_df)}")

    np.random.seed(random_seed)
    if len(combined_df) > n_samples:
        sample_df = combined_df.sample(n=n_samples, random_state=random_seed)
        print(f"Sampling {n_samples} embryos for analysis")
    else:
        sample_df = combined_df
        print(f"Using all {len(sample_df)} embryos")

    if sample_df.empty:
        print("No embryos available for analysis.")
        return pd.DataFrame([])

    # Determine worker count
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)
    n_workers = max(1, min(n_jobs, len(sample_df)))
    print(f"\nProcessing {len(sample_df)} embryos using {n_workers} worker(s)...")

    args_list = [
        (
            row['snip_id'],
            row['mask_rle'],
            row['mask_height_px'],
            row['mask_width_px'],
            row['dataset'],
        )
        for _, row in sample_df.iterrows()
    ]

    if n_workers == 1:
        results = [
            process_single_row(args)
            for args in tqdm(args_list, total=len(args_list), desc="Comparing methods")
        ]
    else:
        with mp.Pool(processes=n_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_row, args_list),
                total=len(args_list),
                desc="Comparing methods"
            ))

    # Filter successful results
    results = [r for r in results if r is not None]
    print(f"\nSuccessfully analyzed {len(results)}/{len(sample_df)} embryos")

    return pd.DataFrame(results)


def analyze_bimodal_distribution(results_df: pd.DataFrame, output_dir: Path):
    """
    Fit bimodal (Gaussian Mixture Model) to Hausdorff distance distribution
    to find natural threshold between "agree" and "disagree" cases.

    Args:
        results_df: DataFrame with comparison results
        output_dir: Directory to save outputs

    Returns:
        threshold: Natural threshold from bimodal fit
        gmm: Fitted Gaussian Mixture Model
    """
    print(f"\n{'='*80}")
    print("BIMODAL DISTRIBUTION ANALYSIS (Hausdorff Distance)")
    print('='*80)

    # Filter to embryos where both methods succeeded
    both_df = results_df[results_df['both_methods_succeeded']].copy()

    if len(both_df) == 0:
        print("No embryos where both methods succeeded. Cannot perform bimodal analysis.")
        return None, None

    # Get Hausdorff distances
    hausdorff_values = both_df['hausdorff_distance'].values.reshape(-1, 1)

    # Fit Gaussian Mixture Model with 2 components
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(hausdorff_values)

    # Get parameters
    means = gmm.means_.flatten()
    covariances = gmm.covariances_.flatten()
    weights = gmm.weights_

    # Sort by mean (first component = agree, second = disagree)
    sorted_idx = np.argsort(means)
    mean_agree = means[sorted_idx[0]]
    mean_disagree = means[sorted_idx[1]]
    std_agree = np.sqrt(covariances[sorted_idx[0]])
    std_disagree = np.sqrt(covariances[sorted_idx[1]])
    weight_agree = weights[sorted_idx[0]]
    weight_disagree = weights[sorted_idx[1]]

    print(f"\nGaussian Mixture Model (2 components):")
    print(f"  Component 1 (Agree):    mean={mean_agree:.2f}px, std={std_agree:.2f}px, weight={weight_agree:.2%}")
    print(f"  Component 2 (Disagree): mean={mean_disagree:.2f}px, std={std_disagree:.2f}px, weight={weight_disagree:.2%}")

    # Find threshold as intersection point between two Gaussians
    # For simplicity, use midpoint weighted by component weights
    threshold = (mean_agree + mean_disagree) / 2.0

    print(f"\nNatural threshold (midpoint): {threshold:.2f}px")
    print(f"  Below threshold: Methods agree")
    print(f"  Above threshold: Methods disagree significantly")

    # Classify embryos
    both_df['agrees'] = both_df['hausdorff_distance'] < threshold
    n_agree = both_df['agrees'].sum()
    n_disagree = len(both_df) - n_agree

    print(f"\nClassification:")
    print(f"  Agree:    {n_agree}/{len(both_df)} ({100*n_agree/len(both_df):.1f}%)")
    print(f"  Disagree: {n_disagree}/{len(both_df)} ({100*n_disagree/len(both_df):.1f}%)")

    # Plot bimodal distribution
    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    ax.hist(hausdorff_values, bins=50, density=True, alpha=0.6, color='gray', label='Data')

    # Plot GMM components
    x_range = np.linspace(hausdorff_values.min(), hausdorff_values.max(), 1000).reshape(-1, 1)
    log_prob = gmm.score_samples(x_range)
    responsibilities = gmm.predict_proba(x_range)
    pdf = np.exp(log_prob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]

    ax.plot(x_range, pdf, 'k-', linewidth=2, label='GMM fit')
    ax.plot(x_range, pdf_individual[:, sorted_idx[0]], 'b--', linewidth=2, label=f'Agree (μ={mean_agree:.1f})')
    ax.plot(x_range, pdf_individual[:, sorted_idx[1]], 'r--', linewidth=2, label=f'Disagree (μ={mean_disagree:.1f})')

    # Threshold line
    ax.axvline(threshold, color='orange', linestyle='--', linewidth=2, label=f'Threshold={threshold:.1f}px')

    ax.set_xlabel('Hausdorff Distance (pixels)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Bimodal Distribution of Hausdorff Distance\nPCA vs Geodesic Spline Agreement', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "bimodal_fit_hausdorff.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()

    return threshold, gmm


def generate_decision_rules(results_df: pd.DataFrame, output_dir: Path, threshold: float = None):
    """
    Analyze correlations between mask metrics and spline differences
    to generate decision rules for when to use Geodesic vs PCA.

    Args:
        results_df: DataFrame with comparison results
        output_dir: Directory to save outputs
        threshold: Hausdorff threshold from bimodal analysis (if None, uses quantile)
    """
    print(f"\n{'='*80}")
    print("DECISION RULE ANALYSIS")
    print('='*80)

    # Filter to embryos where both methods succeeded
    both_df = results_df[results_df['both_methods_succeeded']].copy()

    if len(both_df) == 0:
        print("No embryos where both methods succeeded. Cannot generate decision rules.")
        return

    # Define "large difference" threshold using bimodal or quantile
    if threshold is not None:
        large_diff_threshold = threshold
        both_df['large_difference'] = both_df['hausdorff_distance'] > large_diff_threshold
        print(f"\nUsing bimodal Hausdorff threshold: {large_diff_threshold:.2f} px")
    else:
        large_diff_threshold = both_df['mean_distance'].quantile(0.75)
        both_df['large_difference'] = both_df['mean_distance'] > large_diff_threshold
        print(f"\nUsing 75th percentile mean_distance threshold: {large_diff_threshold:.2f} px")

    print(f"Embryos with large differences: {both_df['large_difference'].sum()}/{len(both_df)}")

    # Compute correlations
    mask_metrics = ['solidity', 'circularity', 'extent', 'eccentricity', 'perimeter_area_ratio']
    difference_metrics = ['mean_distance', 'max_distance', 'hausdorff_distance']

    print(f"\n{'='*80}")
    print("CORRELATIONS: Mask Metrics vs Spline Differences")
    print('='*80)

    for diff_metric in difference_metrics:
        print(f"\n{diff_metric.upper()}:")
        for mask_metric in mask_metrics:
            corr = both_df[mask_metric].corr(both_df[diff_metric])
            print(f"  {mask_metric:25s}: {corr:+.3f}")

    # Identify best predictors
    print(f"\n{'='*80}")
    print("BEST PREDICTORS OF LARGE DIFFERENCES")
    print('='*80)

    large_diff_df = both_df[both_df['large_difference']]
    small_diff_df = both_df[~both_df['large_difference']]

    print(f"\nMask Metric Comparison:")
    print(f"{'Metric':<25s} | {'Small Diff (n={})'.format(len(small_diff_df)):^20s} | {'Large Diff (n={})'.format(len(large_diff_df)):^20s}")
    print('-'*70)

    for metric in mask_metrics:
        small_mean = small_diff_df[metric].mean()
        large_mean = large_diff_df[metric].mean()
        diff_pct = 100 * (large_mean - small_mean) / small_mean if small_mean != 0 else 0

        print(f"{metric:<25s} | {small_mean:^20.3f} | {large_mean:^20.3f} ({diff_pct:+.1f}%)")

    # Generate threshold recommendations
    print(f"\n{'='*80}")
    print("THRESHOLD RECOMMENDATIONS")
    print('='*80)
    print("\nSuggested thresholds for detecting when Geodesic is preferred over PCA:")

    for metric in mask_metrics:
        # Find threshold that best separates large vs small differences
        threshold_candidates = np.percentile(both_df[metric], [10, 25, 50, 75, 90])

        best_threshold = None
        best_separation = 0

        for threshold in threshold_candidates:
            if metric in ['solidity', 'extent']:
                # Lower values indicate problems
                predicted_large = both_df[metric] < threshold
            else:
                # Higher values indicate problems
                predicted_large = both_df[metric] > threshold

            # Measure separation (difference in mean_distance)
            predicted_large_mean = both_df[predicted_large]['mean_distance'].mean() if predicted_large.sum() > 0 else 0
            predicted_small_mean = both_df[~predicted_large]['mean_distance'].mean() if (~predicted_large).sum() > 0 else 0
            separation = abs(predicted_large_mean - predicted_small_mean)

            if separation > best_separation:
                best_separation = separation
                best_threshold = threshold

        if best_threshold is not None:
            if metric in ['solidity', 'extent']:
                print(f"  {metric} < {best_threshold:.3f} → Consider Geodesic")
            else:
                print(f"  {metric} > {best_threshold:.3f} → Consider Geodesic")

    print(f"\n{'='*80}")


def create_distribution_histograms(results_df: pd.DataFrame, threshold: float, output_dir: Path):
    """
    Create histogram visualizations of distance distributions.

    Args:
        results_df: DataFrame with comparison results
        threshold: Hausdorff threshold from bimodal analysis
        output_dir: Directory to save outputs
    """
    print(f"\n{'='*80}")
    print("CREATING DISTRIBUTION HISTOGRAMS")
    print('='*80)

    both_df = results_df[results_df['both_methods_succeeded']].copy()

    if len(both_df) == 0:
        print("No embryos to visualize.")
        return

    # Create 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    distance_metrics = [
        ('mean_distance', 'Mean Distance (px)', axes[0]),
        ('max_distance', 'Max Distance (px)', axes[1]),
        ('hausdorff_distance', 'Hausdorff Distance (px)', axes[2])
    ]

    for metric_name, xlabel, ax in distance_metrics:
        values = both_df[metric_name].values

        # Histogram
        n, bins, patches = ax.hist(values, bins=50, alpha=0.7, color='steelblue', edgecolor='black')

        # Add threshold line for Hausdorff
        if metric_name == 'hausdorff_distance' and threshold is not None:
            ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold={threshold:.1f}px')
            below = np.sum(values < threshold)
            above = len(values) - below
            ax.text(threshold, max(n)*0.9, f'{below} agree\n{above} disagree',
                   ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Statistics
        mean_val = np.mean(values)
        median_val = np.median(values)
        ax.axvline(mean_val, color='orange', linestyle='-', linewidth=2, label=f'Mean={mean_val:.1f}px')
        ax.axvline(median_val, color='green', linestyle=':', linewidth=2, label=f'Median={median_val:.1f}px')

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{xlabel} Distribution\n(n={len(values)})', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "distribution_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def visualize_agreement_examples(results_df: pd.DataFrame, output_dir: Path, n_examples: int = 3):
    """
    Visualize best and worst agreement cases with overlaid splines.

    Args:
        results_df: DataFrame with comparison results
        output_dir: Directory to save outputs
        n_examples: Number of examples to show for each category
    """
    print(f"\n{'='*80}")
    print("GENERATING EXAMPLE VISUALIZATIONS")
    print('='*80)

    both_df = results_df[results_df['both_methods_succeeded']].copy()

    if len(both_df) == 0:
        print("No embryos to visualize.")
        return

    # Get best and worst cases
    both_df_sorted = both_df.sort_values('hausdorff_distance')
    best_cases = both_df_sorted.head(n_examples)
    worst_cases = both_df_sorted.tail(n_examples)

    print(f"\nVisualizing {n_examples} best and {n_examples} worst agreement cases...")

    # Note: This is a placeholder - actual implementation would require
    # storing masks and spline coordinates in results_df or recomputing them
    print("  [NOTE: Full implementation requires mask/spline data storage]")
    print(f"  Best cases (lowest Hausdorff):")
    for idx, row in best_cases.iterrows():
        print(f"    - {row['snip_id']}: {row['hausdorff_distance']:.2f}px")
    print(f"  Worst cases (highest Hausdorff):")
    for idx, row in worst_cases.iterrows():
        print(f"    - {row['snip_id']}: {row['hausdorff_distance']:.2f}px")


import matplotlib.patches as mpatches

def get_mask_for_snip(datasets, dataset_name, snip_id, clean=True):
    """
    Get mask from dataset cache using proper RLE decoder.

    Args:
        datasets: Dictionary of dataset DataFrames
        dataset_name: Name of dataset
        snip_id: Embryo ID
        clean: If True, apply mask cleaning (same as used for spline extraction)

    Returns:
        Mask array (cleaned if clean=True)
    """
    if dataset_name not in datasets:
        return None

    dataset_df = datasets[dataset_name]
    mask_rows = dataset_df[dataset_df['snip_id'] == snip_id]

    if mask_rows.empty:
        return None

    mask_row = mask_rows.iloc[0]

    # Use the proper decode_mask_rle function
    mask = decode_mask_rle({
        'size': [int(mask_row['mask_height_px']), int(mask_row['mask_width_px'])],
        'counts': mask_row['mask_rle']
    })

    mask = np.ascontiguousarray(mask.astype(np.uint8))

    # Apply cleaning if requested (same pipeline used for spline extraction)
    if clean:
        mask, _ = clean_embryo_mask(mask, verbose=False)

    return mask

def visualize_spline_comparison(snip_id, mask_rle, height, width, 
                                pca_coords, geo_coords, hausdorff_dist,
                                save_path):
    """Visualize PCA vs Geodesic splines overlaid on mask."""
    mask = decode_rle_mask(mask_rle, height, width)
    if mask is None:
        return False
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Show mask
    ax.imshow(mask, cmap='gray', alpha=0.7)
    
    # Plot splines
    ax.plot(pca_coords[:, 1], pca_coords[:, 0], 'b-', linewidth=2, label='PCA', alpha=0.8)
    ax.plot(geo_coords[:, 1], geo_coords[:, 0], 'r-', linewidth=2, label='Geodesic', alpha=0.8)
    
    ax.set_title(f'{snip_id}\nHausdorff Distance: {hausdorff_dist:.1f}px', fontsize=12)
    ax.legend(loc='upper right')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True

def create_hausdorff_interval_examples(results_df, datasets, interval_size=15, 
                                       examples_per_bin=2, output_dir='results/mcolon/20251027'):
    """Create visualization of spline examples binned by Hausdorff distance intervals."""
    if not isinstance(datasets, dict) or len(datasets) == 0:
        print("Dataset cache is empty; cannot create interval examples.")
        return

    required_cols = {'hausdorff_distance', 'dataset', 'pca_coords', 'geo_coords', 'snip_id'}
    missing_cols = required_cols - set(results_df.columns)
    if missing_cols:
        print(f"Skipping interval examples; results missing columns: {sorted(missing_cols)}")
        return

    valid_df = results_df.dropna(subset=['hausdorff_distance']).copy()
    valid_df = valid_df[
        (valid_df['pca_coords'] != "") &
        (valid_df['geo_coords'] != "") &
        valid_df['both_methods_succeeded']
    ]
    valid_df = valid_df[valid_df['dataset'].notna()]
    valid_df = valid_df[valid_df['dataset'].notna()]

    if valid_df.empty:
        print("No valid rows with spline data for interval analysis.")
        return

    max_hausdorff = valid_df['hausdorff_distance'].max()
    if pd.isna(max_hausdorff) or max_hausdorff <= 0:
        print("Hausdorff distances are not positive; skipping interval examples.")
        return

    bins = np.arange(0, max_hausdorff + interval_size, interval_size)
    valid_df['hausdorff_bin'] = pd.cut(valid_df['hausdorff_distance'], bins=bins)

    # Sample from each bin
    examples = []
    for bin_range in valid_df['hausdorff_bin'].cat.categories:
        bin_data = valid_df[valid_df['hausdorff_bin'] == bin_range]
        if len(bin_data) > 0:
            sampled = bin_data.sample(n=min(examples_per_bin, len(bin_data)), random_state=42)
            examples.append(sampled)
    
    if not examples:
        print("No examples found for interval analysis")
        return
    
    examples_df = pd.concat(examples, ignore_index=True).reset_index(drop=True)
    
    # Create grid visualization
    n_examples = len(examples_df)
    n_cols = min(4, n_examples)
    n_rows = (n_examples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, (_, row) in enumerate(examples_df.iterrows()):
        ax = axes[idx // n_cols, idx % n_cols]

        # Get mask from appropriate dataset
        dataset_name = row['dataset']
        mask = get_mask_for_snip(datasets, dataset_name, row['snip_id'])

        if mask is not None:
            # Parse coordinates
            pca_coords = np.array([list(map(float, x.split(','))) for x in row['pca_coords'].split(';')])
            geo_coords = np.array([list(map(float, x.split(','))) for x in row['geo_coords'].split(';')])
            
            ax.imshow(mask, cmap='gray', alpha=0.7)
            ax.plot(pca_coords[:, 1], pca_coords[:, 0], 'b-', linewidth=2, label='PCA', alpha=0.8)
            ax.plot(geo_coords[:, 1], geo_coords[:, 0], 'r-', linewidth=2, label='Geodesic', alpha=0.8)
            
            ax.set_title(f'{row["snip_id"]}\nHausdorff: {row["hausdorff_distance"]:.1f}px\n'
                        f'Mean: {row["mean_distance"]:.1f}px', fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
        
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_examples, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis('off')
    
    plt.suptitle(f'Spline Comparison by Hausdorff Distance Intervals ({interval_size}px bins)\n' +
                 'Cleaned Masks (as used for spline extraction)',
                 fontsize=14, y=0.995)
    plt.tight_layout()

    output_path = f'{output_dir}/hausdorff_interval_examples.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved interval examples to {output_path}")

def create_threshold_comparison(results_df, datasets, threshold, 
                               n_examples=3, output_dir='results/mcolon/20251027'):
    """Show examples above and below the computed threshold."""
    if threshold is None:
        print("Threshold is None; skipping threshold comparison.")
        return

    if not isinstance(datasets, dict) or len(datasets) == 0:
        print("Dataset cache is empty; cannot create threshold comparison.")
        return

    required_cols = {'hausdorff_distance', 'dataset', 'pca_coords', 'geo_coords', 'snip_id'}
    missing_cols = required_cols - set(results_df.columns)
    if missing_cols:
        print(f"Skipping threshold comparison; results missing columns: {sorted(missing_cols)}")
        return

    valid_df = results_df.dropna(subset=['hausdorff_distance']).copy()
    valid_df = valid_df[
        (valid_df['pca_coords'] != "") &
        (valid_df['geo_coords'] != "") &
        valid_df['both_methods_succeeded']
    ]

    if valid_df.empty:
        print("No valid rows with spline data for threshold comparison.")
        return

    if (valid_df['hausdorff_distance'] >= threshold).sum() == 0:
        print("No rows above threshold; skipping threshold comparison.")
        return

    below_threshold = valid_df[valid_df['hausdorff_distance'] < threshold].nsmallest(n_examples, 'hausdorff_distance')
    above_threshold = valid_df[valid_df['hausdorff_distance'] >= threshold].nlargest(n_examples, 'hausdorff_distance')

    if below_threshold.empty or above_threshold.empty:
        print("Insufficient examples on one side of the threshold; skipping plot.")
        return

    fig, axes = plt.subplots(2, n_examples, figsize=(5*n_examples, 10))
    if n_examples == 1:
        axes = axes.reshape(2, 1)
    
    # Below threshold (agreement)
    for idx, (_, row) in enumerate(below_threshold.iterrows()):
        ax = axes[0, idx]

        dataset_name = row['dataset']
        mask = get_mask_for_snip(datasets, dataset_name, row['snip_id'])

        if mask is not None:
            pca_coords = np.array([list(map(float, x.split(','))) for x in row['pca_coords'].split(';')])
            geo_coords = np.array([list(map(float, x.split(','))) for x in row['geo_coords'].split(';')])
            
            ax.imshow(mask, cmap='gray', alpha=0.7)
            ax.plot(pca_coords[:, 1], pca_coords[:, 0], 'b-', linewidth=2, label='PCA', alpha=0.8)
            ax.plot(geo_coords[:, 1], geo_coords[:, 0], 'r-', linewidth=2, label='Geodesic', alpha=0.8)
            
            ax.set_title(f'✓ {row["snip_id"]}\nHausdorff: {row["hausdorff_distance"]:.1f}px', 
                        fontsize=10, color='green')
            ax.legend(loc='upper right', fontsize=8)
        
        ax.axis('off')
    
    # Above threshold (disagreement)
    for idx, (_, row) in enumerate(above_threshold.iterrows()):
        ax = axes[1, idx]

        dataset_name = row['dataset']
        mask = get_mask_for_snip(datasets, dataset_name, row['snip_id'])

        if mask is not None:
            pca_coords = np.array([list(map(float, x.split(','))) for x in row['pca_coords'].split(';')])
            geo_coords = np.array([list(map(float, x.split(','))) for x in row['geo_coords'].split(';')])
            
            ax.imshow(mask, cmap='gray', alpha=0.7)
            ax.plot(pca_coords[:, 1], pca_coords[:, 0], 'b-', linewidth=2, label='PCA', alpha=0.8)
            ax.plot(geo_coords[:, 1], geo_coords[:, 0], 'r-', linewidth=2, label='Geodesic', alpha=0.8)
            
            ax.set_title(f'✗ {row["snip_id"]}\nHausdorff: {row["hausdorff_distance"]:.1f}px', 
                        fontsize=10, color='red')
            ax.legend(loc='upper right', fontsize=8)
        
        ax.axis('off')
    
    axes[0, 0].text(-0.15, 0.5, f'Below Threshold\n({threshold:.1f}px)\nAGREEMENT', 
                    transform=axes[0, 0].transAxes, fontsize=12, 
                    verticalalignment='center', rotation=90, color='green', weight='bold')
    axes[1, 0].text(-0.15, 0.5, f'Above Threshold\n({threshold:.1f}px)\nDISAGREEMENT',
                    transform=axes[1, 0].transAxes, fontsize=12,
                    verticalalignment='center', rotation=90, color='red', weight='bold')

    plt.suptitle(f'Threshold Comparison: {threshold:.1f}px\nCleaned Masks (as used for spline extraction)',
                fontsize=14)
    plt.tight_layout()

    output_path = f'{output_dir}/threshold_comparison_examples.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved threshold comparison to {output_path}")

def create_hausdorff_histogram_with_bins(results_df, threshold, interval_size=15,
                                         output_dir='results/mcolon/20251027'):
    """Create histogram showing Hausdorff distribution with interval bins and threshold."""
    if threshold is None:
        print("Threshold is None; skipping histogram with bins.")
        return

    hausdorff_values = results_df['hausdorff_distance'].dropna()
    if hausdorff_values.empty:
        print("No Hausdorff distances to plot; skipping histogram.")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Histogram
    n, bins_edges, patches = ax.hist(hausdorff_values, bins=50, 
                                     alpha=0.7, color='skyblue', edgecolor='black')
    
    # Color bars by threshold
    for i, patch in enumerate(patches):
        if bins_edges[i] < threshold:
            patch.set_facecolor('green')
            patch.set_alpha(0.5)
        else:
            patch.set_facecolor('red')
            patch.set_alpha(0.5)
    
    # Add vertical lines for interval bins
    max_hausdorff = hausdorff_values.max()
    interval_edges = np.arange(0, max_hausdorff + interval_size, interval_size)
    for edge in interval_edges:
        ax.axvline(edge, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    # Threshold line
    ax.axvline(threshold, color='black', linestyle='-', linewidth=3, 
              label=f'Threshold: {threshold:.1f}px')
    
    # Statistics
    below_count = (hausdorff_values < threshold).sum()
    above_count = (hausdorff_values >= threshold).sum()
    
    ax.set_xlabel('Hausdorff Distance (pixels)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Hausdorff Distance Distribution with {interval_size}px Intervals\n'
                f'Below Threshold: {below_count} | Above Threshold: {above_count}', 
                fontsize=14)
    
    # Legend
    green_patch = mpatches.Patch(color='green', alpha=0.5, label=f'Agreement (< {threshold:.1f}px)')
    red_patch = mpatches.Patch(color='red', alpha=0.5, label=f'Disagreement (≥ {threshold:.1f}px)')
    ax.legend(handles=[green_patch, red_patch, 
                      plt.Line2D([0], [0], color='black', linewidth=3, label=f'Threshold: {threshold:.1f}px')],
             fontsize=10)
    
    plt.tight_layout()
    
    output_path = f'{output_dir}/hausdorff_histogram_with_bins.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved histogram with bins to {output_path}")


def main():
    """Main execution."""
    # Load from both part1 and part2 for diversity
    dataset_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build06_output")
    csv_paths = [
        dataset_dir / "df03_final_output_with_latents_20251017_part1.csv",
        dataset_dir / "df03_final_output_with_latents_20251017_part2.csv"
    ]
    output_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251027")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("PCA VS GEODESIC SPLINE COMPARISON (PARALLELIZED - 1000 EMBRYOS)")
    print("="*80)

    # Load datasets and keep them for visualization
    datasets = {}
    for csv_path in csv_paths:
        dataset_name = csv_path.stem
        print(f"Loading {dataset_name} from {csv_path}")
        datasets[dataset_name] = pd.read_csv(csv_path)
        print(f"  Loaded {len(datasets[dataset_name])} embryos")
    
    # Analyze 1000 embryos from both datasets with parallelization
    results_df = analyze_dataset(csv_paths, n_samples=1000, random_seed=42, n_jobs=None)

    # Save results
    output_csv = output_dir / "pca_vs_geodesic_comparison_1000embryos.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")

    # Bimodal distribution analysis
    threshold, _ = analyze_bimodal_distribution(results_df, output_dir)

    # Create distribution histograms
    if threshold is not None:
        create_distribution_histograms(results_df, threshold, output_dir)
        create_threshold_comparison(results_df, datasets, threshold, n_examples=3, output_dir=output_dir)
        create_hausdorff_histogram_with_bins(results_df, threshold, interval_size=15, output_dir=output_dir)
    else:
        print("Skipping threshold-dependent visualizations; threshold not available.")

    # Visualize example cases
    visualize_agreement_examples(results_df, output_dir, n_examples=3)

    # Generate decision rules (updated to use bimodal threshold)
    generate_decision_rules(results_df, output_dir, threshold=threshold)

    # NEW: Interval examples
    print("\n" + "="*80)
    print("CREATING HAUSDORFF INTERVAL EXAMPLES")
    print("="*80)
    create_hausdorff_interval_examples(
        results_df,
        datasets,
        interval_size=15,
        examples_per_bin=2,
        output_dir=output_dir,
    )
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
