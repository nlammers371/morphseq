"""
Curvature Metrics Analysis for Embryo Masks

Computes 5 different curvature metrics on embryo masks (after cleaning + Gaussian blur)
to identify which metrics best capture biologically meaningful shape variation.

Metrics:
1. Finite-Difference Curvature (contour-based)
2. Spline-Fitted Curvature (B-spline derivatives)
3. Signed Distance Transform Curvature (divergence of gradients)
4. Local Circle Fit Curvature (least-squares circle fitting)
5. Global Proxy Metrics (circularity, Fourier smoothness, perimeter/area)

For each metric:
- Measures computation time
- Identifies top 5 and bottom 5 embryos (highest/lowest curvature)
- Generates visualizations

Outputs (in results/mcolon/20251027/mask_curvature_metrics/):
- curvature_metrics_comparison.png: Grid of top/bottom 5 for each metric
- curvature_timing_comparison.png: Bar chart of computation times
- curvature_correlation_heatmap.png: Correlation matrix between metrics
- curvature_metrics_results.csv: Full results table
"""

import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import cv2
from scipy.ndimage import distance_transform_edt, sobel, gaussian_filter
from scipy.interpolate import splprep, splev
from skimage import measure
from tqdm import tqdm
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')

from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle
from segmentation_sandbox.scripts.utils.mask_cleaning import clean_embryo_mask


# =============================================================================
# ALPHA-SHAPE REFINEMENT (Winner from previous test)
# =============================================================================

def refine_alpha_shape(mask: np.ndarray, alpha: float = 50) -> np.ndarray:
    """
    Apply Alpha-Shape hull refinement to mask.

    This was one of the winning methods from test_mask_refinement_methods.py
    (along with Gaussian blur)
    """
    from scipy.spatial import ConvexHull
    from skimage.morphology import binary_erosion, disk
    from skimage.draw import polygon

    # First apply standard cleaning
    cleaned, _ = clean_embryo_mask(mask, verbose=False)

    # Get mask boundary points
    points = np.column_stack(np.where(cleaned))

    if len(points) < 3:
        return cleaned

    try:
        # Compute convex hull
        hull = ConvexHull(points)

        # Create hull mask
        hull_mask = np.zeros_like(cleaned, dtype=bool)

        # Fill hull interior using polygon fill
        hull_points = points[hull.vertices]
        rr, cc = polygon(hull_points[:, 0], hull_points[:, 1], cleaned.shape)
        hull_mask[rr, cc] = True

        # Erode hull to allow concavity (erosion amount based on alpha parameter)
        erode_radius = max(5, int(alpha / 10))
        eroded_hull = binary_erosion(hull_mask, disk(erode_radius))

        # Intersect with original cleaned mask
        refined = cleaned & eroded_hull

        # If erosion removed too much, use less erosion
        if refined.sum() < 0.5 * cleaned.sum():
            erode_radius = max(3, int(alpha / 20))
            eroded_hull = binary_erosion(hull_mask, disk(erode_radius))
            refined = cleaned & eroded_hull

        # Keep largest component
        labeled = measure.label(refined)
        if labeled.max() > 0:
            largest = np.argmax([np.sum(labeled == i) for i in range(1, labeled.max() + 1)]) + 1
            refined = (labeled == largest)

        return refined

    except Exception as e:
        print(f"Warning: Alpha-shape failed ({e}), returning baseline")
        return cleaned


# =============================================================================
# CURVATURE METRIC 1: Finite-Difference Curvature
# =============================================================================

def compute_finite_difference_curvature(mask: np.ndarray) -> dict:
    """
    Compute curvature using finite differences on contour.

    Returns:
        dict with 'mean_curvature', 'max_curvature', 'curvature_map', 'contour'
    """
    # Extract contour
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return {
            'mean_curvature': np.nan,
            'max_curvature': np.nan,
            'curvature_map': None,
            'contour': None
        }

    # Use largest contour
    contour = max(contours, key=cv2.contourArea).squeeze()

    if len(contour.shape) != 2 or contour.shape[1] != 2:
        return {
            'mean_curvature': np.nan,
            'max_curvature': np.nan,
            'curvature_map': None,
            'contour': None
        }

    # Compute derivatives using finite differences
    dx = np.gradient(contour[:, 0].astype(float))
    dy = np.gradient(contour[:, 1].astype(float))
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Curvature formula: κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = (dx**2 + dy**2)**(3/2) + 1e-10
    curvature = numerator / denominator

    return {
        'mean_curvature': np.mean(curvature),
        'max_curvature': np.max(curvature),
        'std_curvature': np.std(curvature),
        'curvature_map': curvature,
        'contour': contour
    }


# =============================================================================
# CURVATURE METRIC 2: Spline-Fitted Curvature
# =============================================================================

def compute_spline_fitted_curvature(mask: np.ndarray, smoothing: float = 10.0) -> dict:
    """
    Compute curvature from B-spline fit to contour.

    Args:
        mask: Binary mask
        smoothing: B-spline smoothing parameter

    Returns:
        dict with curvature statistics
    """
    # Extract contour
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return {
            'mean_curvature': np.nan,
            'max_curvature': np.nan,
            'curvature_map': None,
            'contour': None
        }

    contour = max(contours, key=cv2.contourArea).squeeze()

    if len(contour.shape) != 2 or contour.shape[1] != 2 or len(contour) < 4:
        return {
            'mean_curvature': np.nan,
            'max_curvature': np.nan,
            'curvature_map': None,
            'contour': None
        }

    try:
        # Fit periodic B-spline
        tck, u = splprep([contour[:, 0], contour[:, 1]],
                         s=smoothing * len(contour), k=3, per=True)

        # Evaluate at fine resolution
        u_fine = np.linspace(0, 1, 500)

        # First derivatives
        dx, dy = splev(u_fine, tck, der=1)

        # Second derivatives
        ddx, ddy = splev(u_fine, tck, der=2)

        # Curvature
        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = (dx**2 + dy**2)**(3/2) + 1e-10
        curvature = numerator / denominator

        # Get spline points for visualization
        spline_x, spline_y = splev(u_fine, tck)
        spline_contour = np.column_stack([spline_x, spline_y])

        return {
            'mean_curvature': np.mean(curvature),
            'max_curvature': np.max(curvature),
            'std_curvature': np.std(curvature),
            'curvature_map': curvature,
            'contour': spline_contour
        }

    except Exception as e:
        return {
            'mean_curvature': np.nan,
            'max_curvature': np.nan,
            'curvature_map': None,
            'contour': None
        }


# =============================================================================
# CURVATURE METRIC 3: Signed Distance Transform Curvature
# =============================================================================

def compute_sdt_curvature(mask: np.ndarray) -> dict:
    """
    Compute curvature as divergence of normalized gradients of signed distance transform.

    This method works directly on the mask without needing contour extraction.
    """
    # Compute signed distance transform
    dist_inside = distance_transform_edt(mask)
    dist_outside = distance_transform_edt(~mask)
    signed_dist = dist_outside - dist_inside

    # Smooth the distance field slightly to reduce noise
    signed_dist_smooth = gaussian_filter(signed_dist, sigma=1.0)

    # Compute gradients
    grad_y = sobel(signed_dist_smooth, axis=0)
    grad_x = sobel(signed_dist_smooth, axis=1)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2) + 1e-10

    # Normalize gradients
    norm_grad_x = grad_x / grad_mag
    norm_grad_y = grad_y / grad_mag

    # Compute divergence (= mean curvature)
    div_x = sobel(norm_grad_x, axis=1)
    div_y = sobel(norm_grad_y, axis=0)
    curvature_field = np.abs(div_x + div_y)

    # Extract boundary curvature (points near zero of signed distance)
    boundary_mask = np.abs(signed_dist) < 2.0
    boundary_curvature = curvature_field[boundary_mask]

    # Also extract contour for visualization
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea).squeeze() if len(contours) > 0 else None

    return {
        'mean_curvature': np.mean(boundary_curvature) if len(boundary_curvature) > 0 else np.nan,
        'max_curvature': np.max(boundary_curvature) if len(boundary_curvature) > 0 else np.nan,
        'std_curvature': np.std(boundary_curvature) if len(boundary_curvature) > 0 else np.nan,
        'curvature_map': curvature_field,
        'contour': contour,
        'boundary_mask': boundary_mask
    }


# =============================================================================
# CURVATURE METRIC 4: Local Circle Fit Curvature
# =============================================================================

def fit_circle_to_points(points):
    """
    Fit a circle to a set of 2D points using least squares.

    Returns:
        radius: Radius of best-fit circle (or np.nan if fit fails)
    """
    if len(points) < 3:
        return np.nan

    # Center points
    center = points.mean(axis=0)
    points_centered = points - center

    # Solve least-squares circle fit
    # (x - cx)^2 + (y - cy)^2 = r^2
    # x^2 + y^2 = 2*cx*x + 2*cy*y + (r^2 - cx^2 - cy^2)

    A = np.column_stack([2 * points_centered[:, 0], 2 * points_centered[:, 1], np.ones(len(points))])
    b = points_centered[:, 0]**2 + points_centered[:, 1]**2

    try:
        params = np.linalg.lstsq(A, b, rcond=None)[0]
        cx, cy, c = params
        radius = np.sqrt(c + cx**2 + cy**2)
        return radius
    except:
        return np.nan


def compute_circle_fit_curvature(mask: np.ndarray, window_size: int = 7) -> dict:
    """
    Compute curvature by fitting circles to local windows along contour.

    Args:
        mask: Binary mask
        window_size: Number of contour points for local circle fit
    """
    # Extract contour
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return {
            'mean_curvature': np.nan,
            'max_curvature': np.nan,
            'curvature_map': None,
            'contour': None
        }

    contour = max(contours, key=cv2.contourArea).squeeze()

    if len(contour.shape) != 2 or contour.shape[1] != 2 or len(contour) < window_size:
        return {
            'mean_curvature': np.nan,
            'max_curvature': np.nan,
            'curvature_map': None,
            'contour': None
        }

    n_points = len(contour)
    curvatures = []

    # For each point, fit circle to local window
    for i in range(n_points):
        # Get window indices (circular)
        half_window = window_size // 2
        indices = [(i + j - half_window) % n_points for j in range(window_size)]
        window_points = contour[indices]

        # Fit circle
        radius = fit_circle_to_points(window_points)

        if not np.isnan(radius) and radius > 0:
            curvature = 1.0 / radius
        else:
            curvature = 0.0

        curvatures.append(curvature)

    curvatures = np.array(curvatures)

    return {
        'mean_curvature': np.mean(curvatures),
        'max_curvature': np.max(curvatures),
        'std_curvature': np.std(curvatures),
        'curvature_map': curvatures,
        'contour': contour
    }


# =============================================================================
# CURVATURE METRIC 5: Circularity Ratio
# =============================================================================

def compute_circularity_ratio(mask: np.ndarray) -> dict:
    """
    Compute circularity ratio: perimeter^2 / (4π * area)

    - Perfect circle = 1.0
    - More complex/curved shape = higher value
    """
    props = measure.regionprops(measure.label(mask.astype(int)))[0]
    perimeter = props.perimeter
    area = props.area

    circularity_ratio = (perimeter**2) / (4 * np.pi * area)

    # Extract contour for visualization
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea).squeeze() if len(contours) > 0 else None

    return {
        'mean_curvature': circularity_ratio,
        'max_curvature': circularity_ratio,
        'std_curvature': 0.0,  # Global metric has no variance
        'curvature_map': None,
        'contour': contour
    }


# =============================================================================
# CURVATURE METRIC 6: Fourier Descriptor Complexity
# =============================================================================

def compute_fourier_complexity(mask: np.ndarray) -> dict:
    """
    Compute Fourier descriptor complexity.

    Measures high-frequency content in contour shape.
    Higher value = more wiggly/complex boundary.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return {
            'mean_curvature': np.nan,
            'max_curvature': np.nan,
            'std_curvature': np.nan,
            'curvature_map': None,
            'contour': None
        }

    contour = max(contours, key=cv2.contourArea).squeeze()

    if len(contour.shape) != 2 or contour.shape[1] != 2 or len(contour) < 10:
        return {
            'mean_curvature': np.nan,
            'max_curvature': np.nan,
            'std_curvature': np.nan,
            'curvature_map': None,
            'contour': contour
        }

    # Convert contour to complex numbers
    contour_complex = contour[:, 0] + 1j * contour[:, 1]

    # Compute Fourier descriptors
    fourier_desc = np.fft.fft(contour_complex)
    power_spectrum = np.abs(fourier_desc)**2

    # Complexity = ratio of high-frequency power
    # Low-frequency = first 10 coefficients (overall shape)
    # High-frequency = rest (wiggles, detail)
    n_low_freq = min(10, len(power_spectrum) // 4)
    low_freq_power = np.sum(power_spectrum[:n_low_freq])
    total_power = np.sum(power_spectrum)

    smoothness = low_freq_power / (total_power + 1e-10)
    complexity = 1.0 - smoothness  # Invert: high complexity = high curvature

    return {
        'mean_curvature': complexity,
        'max_curvature': complexity,
        'std_curvature': 0.0,
        'curvature_map': None,
        'contour': contour
    }


# =============================================================================
# CURVATURE METRIC 7: Normalized Perimeter
# =============================================================================

def compute_normalized_perimeter(mask: np.ndarray) -> dict:
    """
    Compute normalized perimeter: perimeter / sqrt(area)

    Normalizes perimeter by embryo size.
    Higher value = more complex boundary.
    """
    props = measure.regionprops(measure.label(mask.astype(int)))[0]
    perimeter = props.perimeter
    area = props.area

    normalized_perimeter = perimeter / np.sqrt(area)

    # Extract contour for visualization
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea).squeeze() if len(contours) > 0 else None

    return {
        'mean_curvature': normalized_perimeter,
        'max_curvature': normalized_perimeter,
        'std_curvature': 0.0,
        'curvature_map': None,
        'contour': contour
    }


# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================

def analyze_single_embryo(row_data):
    """
    Analyze a single embryo: clean mask, apply Gaussian blur, compute all curvature metrics.

    Args:
        row_data: Tuple of (snip_id, mask_rle, mask_height, mask_width)

    Returns:
        dict with results for all metrics
    """
    snip_id, mask_rle, mask_height, mask_width = row_data

    result = {'snip_id': snip_id}

    try:
        # Decode mask
        mask = decode_mask_rle({
            'size': [int(mask_height), int(mask_width)],
            'counts': mask_rle
        })

        # Apply Alpha-Shape refinement (winner from previous test)
        refined_mask = refine_alpha_shape(mask)

        # Compute each metric with timing (7 metrics total)
        metrics = {
            'finite_diff': compute_finite_difference_curvature,
            'spline': compute_spline_fitted_curvature,
            'sdt': compute_sdt_curvature,
            'circle_fit': compute_circle_fit_curvature,
            'circularity': compute_circularity_ratio,
            'fourier': compute_fourier_complexity,
            'norm_perimeter': compute_normalized_perimeter
        }

        for metric_name, metric_func in metrics.items():
            t0 = time.perf_counter()
            metric_result = metric_func(refined_mask)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            result[f'{metric_name}_mean_curvature'] = metric_result['mean_curvature']
            result[f'{metric_name}_max_curvature'] = metric_result['max_curvature']
            result[f'{metric_name}_std_curvature'] = metric_result.get('std_curvature', np.nan)
            result[f'{metric_name}_time_ms'] = elapsed_ms

            # Store full results for visualization (not in CSV)
            result[f'{metric_name}_full'] = metric_result

        # Store refined mask for visualization
        result['refined_mask'] = refined_mask

        return result

    except Exception as e:
        result['error'] = str(e)
        return result


def main():
    """
    Main analysis pipeline.
    """
    output_dir = Path(__file__).parent

    print("="*70)
    print("CURVATURE METRICS ANALYSIS")
    print("="*70)
    print()

    # Load existing comparison results to get same embryos
    print("Loading existing PCA vs Geodesic comparison results...")
    comparison_csv = Path(__file__).parent.parent / "pca_vs_geodesic_comparison_1000embryos.csv"

    if not comparison_csv.exists():
        print(f"ERROR: {comparison_csv} not found!")
        print("Please run compare_pca_vs_geodesic.py first.")
        return

    df_comparison = pd.read_csv(comparison_csv)
    print(f"Loaded {len(df_comparison):,} embryos from comparison results")

    # Get snip_ids
    snip_ids = df_comparison['snip_id'].values

    # Load metadata to get mask data
    print("Loading embryo metadata...")
    metadata_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build06_output")

    df_part1 = pd.read_csv(metadata_dir / "df03_final_output_with_latents_20251017_part1.csv")
    df_part2 = pd.read_csv(metadata_dir / "df03_final_output_with_latents_20251017_part2.csv")
    df_metadata = pd.concat([df_part1, df_part2], ignore_index=True)

    # Filter to only embryos in comparison results
    df_metadata = df_metadata[df_metadata['snip_id'].isin(snip_ids)]
    print(f"Matched {len(df_metadata):,} embryos with metadata")
    print()

    # Prepare data for analysis
    row_data_list = [
        (row['snip_id'], row['mask_rle'], row['mask_height_px'], row['mask_width_px'])
        for _, row in df_metadata.iterrows()
    ]

    # Analyze embryos in parallel
    print(f"Computing curvature metrics using parallelization...")
    n_workers = mp.cpu_count() - 1
    print(f"Using {n_workers} workers (max available - 1)")

    with mp.Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(analyze_single_embryo, row_data_list),
            total=len(row_data_list),
            desc="Processing embryos"
        ))

    print()

    # Convert to DataFrame (excluding visualization data)
    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if not k.endswith('_full') and k not in ['refined_mask', 'error']}
        for r in results
    ])

    # Save results CSV
    csv_path = output_dir / "curvature_metrics_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    print()

    # Merge with comparison results to identify problem embryos
    results_with_hausdorff = results_df.merge(
        df_comparison[['snip_id', 'hausdorff_distance']],
        on='snip_id',
        how='left'
    )

    # Generate visualizations
    print("Generating visualizations...")

    # 1. Timing comparison
    plot_timing_comparison(results_df, output_dir)

    # 2. Correlation heatmap
    plot_correlation_heatmap(results_df, output_dir)

    # 3. Histograms with problem embryo markers
    plot_metric_histograms(results_with_hausdorff, output_dir, threshold=114.78)

    # 4. Top/Bottom 5 for each metric
    plot_top_bottom_comparison(results, results_df, output_dir)

    # 5. Print summary statistics
    print_summary_statistics(results_df)

    print()
    print("="*70)
    print("Analysis complete!")
    print("="*70)


def plot_timing_comparison(results_df, output_dir):
    """Generate bar chart comparing computation times for each metric."""
    metrics = ['finite_diff', 'spline', 'sdt', 'circle_fit', 'circularity', 'fourier', 'norm_perimeter']
    metric_labels = [
        'Finite-Difference',
        'Spline-Fitted',
        'Signed Distance Transform',
        'Local Circle Fit',
        'Circularity Ratio',
        'Fourier Complexity',
        'Normalized Perimeter'
    ]

    mean_times = [results_df[f'{m}_time_ms'].mean() for m in metrics]
    std_times = [results_df[f'{m}_time_ms'].std() for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(metrics))
    bars = ax.bar(x, mean_times, yerr=std_times, capsize=5, color='steelblue', alpha=0.7)

    # Add value labels on bars
    for i, (bar, mean_time) in enumerate(zip(bars, mean_times)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_times[i] + 5,
               f'{mean_time:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Computation Time (ms)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Curvature Metric', fontsize=12, fontweight='bold')
    ax.set_title('Curvature Metric Computation Time Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=20, ha='right')
    ax.grid(True, alpha=0.3, axis='y', linestyle=':')

    plt.tight_layout()
    plt.savefig(output_dir / "curvature_timing_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/curvature_timing_comparison.png")
    plt.close()


def plot_metric_histograms(results_df, output_dir, threshold=114.78):
    """
    Generate histograms for all 7 metrics, marking where problem embryos fall.

    Problem embryos = those with Hausdorff distance >= threshold (disagreement between PCA and Geodesic)
    """
    metrics = ['finite_diff', 'spline', 'sdt', 'circle_fit', 'circularity', 'fourier', 'norm_perimeter']
    metric_labels = [
        'Finite-Difference Curvature',
        'Spline-Fitted Curvature',
        'Signed Distance Transform',
        'Local Circle Fit Curvature',
        'Circularity Ratio',
        'Fourier Complexity',
        'Normalized Perimeter'
    ]

    # Identify problem embryos (high Hausdorff = PCA/Geodesic disagreement)
    problem_mask = results_df['hausdorff_distance'] >= threshold
    agree_mask = results_df['hausdorff_distance'] < threshold

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]

        metric_col = f'{metric}_mean_curvature'

        # Get data
        all_values = results_df[metric_col].dropna()
        problem_values = results_df[problem_mask][metric_col].dropna()
        agree_values = results_df[agree_mask][metric_col].dropna()

        if len(all_values) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
            ax.set_title(label, fontsize=12, fontweight='bold')
            continue

        # Create histogram
        bins = 30
        ax.hist(agree_values, bins=bins, alpha=0.6, color='blue',
                label=f'Agree (n={len(agree_values)})', edgecolor='black', linewidth=0.5)
        ax.hist(problem_values, bins=bins, alpha=0.6, color='red',
                label=f'Disagree (n={len(problem_values)})', edgecolor='black', linewidth=0.5)

        # Add vertical lines for medians
        if len(agree_values) > 0:
            ax.axvline(agree_values.median(), color='blue', linestyle='--',
                      linewidth=2, alpha=0.8, label=f'Agree median: {agree_values.median():.3f}')
        if len(problem_values) > 0:
            ax.axvline(problem_values.median(), color='red', linestyle='--',
                      linewidth=2, alpha=0.8, label=f'Disagree median: {problem_values.median():.3f}')

        # Statistical test (t-test or Mann-Whitney U)
        if len(problem_values) > 5 and len(agree_values) > 5:
            from scipy.stats import mannwhitneyu
            try:
                stat, pval = mannwhitneyu(problem_values, agree_values, alternative='two-sided')
                ax.text(0.98, 0.98, f'p={pval:.3e}', transform=ax.transAxes,
                       ha='right', va='top', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            except:
                pass

        # Labels
        ax.set_xlabel(label, fontsize=11, fontweight='bold')
        ax.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax.set_title(f'{label}\n(PCA/Geodesic Agreement Analysis)', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y', linestyle=':')

    # Hide extra subplots
    for idx in range(len(metrics), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Curvature Metric Distributions: Agreement vs Disagreement Cases',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_dir / "curvature_metrics_histograms.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/curvature_metrics_histograms.png")
    plt.close()


def plot_correlation_heatmap(results_df, output_dir):
    """Generate correlation heatmap showing how metrics correlate with each other."""
    metrics = ['finite_diff', 'spline', 'sdt', 'circle_fit', 'circularity', 'fourier', 'norm_perimeter']
    metric_labels = ['Finite-Diff', 'Spline', 'SDT', 'Circle Fit', 'Circularity', 'Fourier', 'Norm Perim']

    # Extract mean curvature values for each metric
    curvature_data = results_df[[f'{m}_mean_curvature' for m in metrics]].copy()
    curvature_data.columns = metric_labels

    # Compute correlation matrix
    corr_matrix = curvature_data.corr()

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)

    ax.set_title('Correlation Matrix: Curvature Metrics', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "curvature_correlation_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/curvature_correlation_heatmap.png")
    plt.close()


def plot_top_bottom_comparison(results, results_df, output_dir):
    """
    Generate grid visualization showing top 5 and bottom 5 embryos for each metric.

    Layout: 7 rows (one per metric) × 10 columns (top 5 + bottom 5)
    """
    metrics = ['finite_diff', 'spline', 'sdt', 'circle_fit', 'circularity', 'fourier', 'norm_perimeter']
    metric_labels = [
        'Finite-Difference Curvature',
        'Spline-Fitted Curvature',
        'Signed Distance Transform',
        'Local Circle Fit',
        'Circularity Ratio',
        'Fourier Complexity',
        'Normalized Perimeter'
    ]

    fig, axes = plt.subplots(7, 10, figsize=(30, 21))

    for row_idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
        # Sort by mean curvature
        sorted_df = results_df.sort_values(f'{metric}_mean_curvature', ascending=False)

        # Get top 5 and bottom 5 snip_ids
        top5_ids = sorted_df.head(5)['snip_id'].values
        bottom5_ids = sorted_df.tail(5)['snip_id'].values

        # Plot top 5 (columns 0-4)
        for col_idx, snip_id in enumerate(top5_ids):
            ax = axes[row_idx, col_idx]

            # Find result for this embryo
            embryo_result = next((r for r in results if r['snip_id'] == snip_id), None)

            if embryo_result and 'refined_mask' in embryo_result:
                mask = embryo_result['refined_mask']
                metric_data = embryo_result[f'{metric}_full']

                # Show mask
                ax.imshow(mask, cmap='gray', alpha=0.7)

                # Overlay contour colored by curvature (if available)
                if metric_data['contour'] is not None and metric_data['curvature_map'] is not None:
                    contour = metric_data['contour']
                    curvature_map = metric_data['curvature_map']

                    # Plot contour as scatter with color = curvature
                    if len(curvature_map) == len(contour):
                        scatter = ax.scatter(contour[:, 0], contour[:, 1],
                                           c=curvature_map, cmap='hot', s=1, alpha=0.8,
                                           vmin=0, vmax=np.percentile(curvature_map, 95))

                # Title
                mean_curv = embryo_result[f'{metric}_mean_curvature']
                ax.set_title(f'{snip_id[:15]}...\n{mean_curv:.4f}', fontsize=8)

            ax.axis('off')

        # Plot bottom 5 (columns 5-9)
        for col_idx, snip_id in enumerate(bottom5_ids):
            ax = axes[row_idx, col_idx + 5]

            # Find result for this embryo
            embryo_result = next((r for r in results if r['snip_id'] == snip_id), None)

            if embryo_result and 'refined_mask' in embryo_result:
                mask = embryo_result['refined_mask']
                metric_data = embryo_result[f'{metric}_full']

                # Show mask
                ax.imshow(mask, cmap='gray', alpha=0.7)

                # Overlay contour colored by curvature (if available)
                if metric_data['contour'] is not None and metric_data['curvature_map'] is not None:
                    contour = metric_data['contour']
                    curvature_map = metric_data['curvature_map']

                    if len(curvature_map) == len(contour):
                        scatter = ax.scatter(contour[:, 0], contour[:, 1],
                                           c=curvature_map, cmap='hot', s=1, alpha=0.8,
                                           vmin=0, vmax=np.percentile(curvature_map, 95))

                # Title
                mean_curv = embryo_result[f'{metric}_mean_curvature']
                ax.set_title(f'{snip_id[:15]}...\n{mean_curv:.4f}', fontsize=8)

            ax.axis('off')

        # Add row label
        axes[row_idx, 0].text(-0.1, 0.5, metric_label, transform=axes[row_idx, 0].transAxes,
                            fontsize=12, fontweight='bold', rotation=90,
                            verticalalignment='center', horizontalalignment='right')

    # Add column headers
    fig.text(0.28, 0.95, 'Top 5 (Highest Curvature)', ha='center', fontsize=14, fontweight='bold')
    fig.text(0.72, 0.95, 'Bottom 5 (Lowest Curvature)', ha='center', fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / "curvature_metrics_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/curvature_metrics_comparison.png")
    plt.close()


def print_summary_statistics(results_df):
    """Print summary statistics for each metric."""
    metrics = ['finite_diff', 'spline', 'sdt', 'circle_fit', 'circularity', 'fourier', 'norm_perimeter']
    metric_labels = [
        'Finite-Difference',
        'Spline-Fitted',
        'Signed Distance Transform',
        'Local Circle Fit',
        'Circularity Ratio',
        'Fourier Complexity',
        'Normalized Perimeter'
    ]

    print("="*70)
    print("TIMING COMPARISON")
    print("="*70)
    print()
    print(f"{'Metric':<30} {'Mean Time (ms)':<18} {'Std (ms)'}")
    print("-"*70)

    for metric, label in zip(metrics, metric_labels):
        mean_time = results_df[f'{metric}_time_ms'].mean()
        std_time = results_df[f'{metric}_time_ms'].std()
        print(f"{label:<30} {mean_time:>15.1f}  {std_time:>10.1f}")

    print()

    # Print top/bottom 5 for each metric
    for metric, label in zip(metrics, metric_labels):
        print("="*70)
        print(f"TOP 5 HIGHEST CURVATURE - {label}")
        print("="*70)

        sorted_df = results_df.sort_values(f'{metric}_mean_curvature', ascending=False)
        top5 = sorted_df.head(5)

        for idx, (i, row) in enumerate(top5.iterrows(), 1):
            print(f"{idx}. {row['snip_id']}: mean_curvature = {row[f'{metric}_mean_curvature']:.6f}")

        print()
        print(f"BOTTOM 5 LOWEST CURVATURE - {label}")
        print("-"*70)

        bottom5 = sorted_df.tail(5)
        for idx, (i, row) in enumerate(bottom5.iterrows(), 1):
            print(f"{idx}. {row['snip_id']}: mean_curvature = {row[f'{metric}_mean_curvature']:.6f}")

        print()


if __name__ == "__main__":
    main()
