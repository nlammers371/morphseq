"""
Test Preprocessing Parameter Sweep on Specific Embryos

Compares Gaussian blur and alpha shape preprocessing across different parameter values
to determine which method is most reliable for centerline extraction.

Test Embryos:
1. 20251017_combined_A02_e01_t0064 (Gaussian blur failed)
2. 20250512_E06_e01_t0086

For each embryo × each method × each parameter:
- Apply preprocessing
- Extract centerline using geodesic method
- Visualize mask + centerline overlay
- Track success/failure

Outputs:
- preprocessing_comparison_A02_e01_t0064.png
- preprocessing_comparison_E06_e01_t0086.png
- preprocessing_test_results.csv
"""

import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull
from skimage import morphology, measure
from skimage.draw import polygon
import warnings
warnings.filterwarnings('ignore')

from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle
from segmentation_sandbox.scripts.utils.mask_cleaning import clean_embryo_mask
from segmentation_sandbox.scripts.body_axis_analysis.centerline_extraction import extract_centerline


# =============================================================================
# PREPROCESSING METHODS
# =============================================================================

def apply_gaussian_preprocessing(mask: np.ndarray, sigma: float = 10.0, threshold: float = 0.7) -> np.ndarray:
    """
    Gaussian Blur + Re-threshold to keep only core regions.

    Args:
        mask: Input binary mask
        sigma: Gaussian blur sigma parameter
        threshold: Re-threshold value after blur

    Returns:
        Preprocessed binary mask
    """
    # First apply standard cleaning
    cleaned, _ = clean_embryo_mask(mask, verbose=False)

    # Blur the cleaned mask
    blurred = gaussian_filter(cleaned.astype(float), sigma=sigma)

    # Re-threshold to keep core
    core_mask = blurred > threshold

    # Intersect with original cleaned mask to avoid growing
    refined = cleaned & core_mask

    # Small dilation to recover some lost area
    refined = morphology.binary_dilation(refined, morphology.disk(5))

    # Keep largest component
    labeled = measure.label(refined)
    if labeled.max() > 0:
        largest = np.argmax([np.sum(labeled == i) for i in range(1, labeled.max() + 1)]) + 1
        refined = (labeled == largest)

    return refined


def apply_alpha_shape_preprocessing(mask: np.ndarray, alpha: float = 50) -> np.ndarray:
    """
    Alpha-Shape Hull: Smooth convex hull with concavity tolerance.

    Strategy: Compute convex hull, then erode to allow natural concavity.
    (Simplified version - uses convex hull + erosion)

    Args:
        mask: Input binary mask
        alpha: Controls erosion amount (higher = more concave allowed)

    Returns:
        Preprocessed binary mask
    """
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
        eroded_hull = morphology.binary_erosion(hull_mask, morphology.disk(erode_radius))

        # Intersect with original cleaned mask
        refined = cleaned & eroded_hull

        # If erosion removed too much, use less erosion
        if refined.sum() < 0.5 * cleaned.sum():
            erode_radius = max(3, int(alpha / 20))
            eroded_hull = morphology.binary_erosion(hull_mask, morphology.disk(erode_radius))
            refined = cleaned & eroded_hull

        # Keep largest component
        labeled = measure.label(refined)
        if labeled.max() > 0:
            largest = np.argmax([np.sum(labeled == i) for i in range(1, labeled.max() + 1)]) + 1
            refined = (labeled == largest)

        return refined

    except Exception as e:
        print(f"Warning: Alpha-shape failed ({e}), returning baseline")
        cleaned_baseline, _ = clean_embryo_mask(mask, verbose=False)
        return cleaned_baseline


# =============================================================================
# EMBRYO LOADING
# =============================================================================

def load_embryo_data(snip_id: str):
    """
    Load embryo mask and metadata from CSV files.

    Args:
        snip_id: Embryo identifier (e.g., "20251017_combined_A02_e01_t0064")

    Returns:
        mask, um_per_pixel, embryo_metadata
    """
    metadata_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build06_output")

    # Determine which CSV to load based on snip_id
    if "20251017" in snip_id:
        csv_path = metadata_dir / "df03_final_output_with_latents_20251017_combined.csv"
    elif "20250512" in snip_id:
        csv_path = metadata_dir / "df03_final_output_with_latents_20250512.csv"
    else:
        raise ValueError(f"Cannot determine CSV file for snip_id: {snip_id}")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Load metadata
    df = pd.read_csv(csv_path)
    embryo_row = df[df['snip_id'] == snip_id]

    if len(embryo_row) == 0:
        raise ValueError(f"Embryo {snip_id} not found in {csv_path}")

    embryo_data = embryo_row.iloc[0]

    # Decode mask
    mask = decode_mask_rle({
        'size': [int(embryo_data['mask_height_px']), int(embryo_data['mask_width_px'])],
        'counts': embryo_data['mask_rle']
    })

    # Calculate um_per_pixel
    um_per_pixel = embryo_data['height_um'] / int(embryo_data['mask_height_px'])

    return mask, um_per_pixel, embryo_data


# =============================================================================
# TEST PIPELINE
# =============================================================================

def test_preprocessing_on_embryo(snip_id: str, gaussian_sigmas: list, alpha_values: list):
    """
    Test all preprocessing parameters on a single embryo.

    Args:
        snip_id: Embryo identifier
        gaussian_sigmas: List of sigma values to test for Gaussian blur
        alpha_values: List of alpha values to test for alpha shape

    Returns:
        List of result dictionaries
    """
    print(f"\n{'='*80}")
    print(f"Testing Embryo: {snip_id}")
    print(f"{'='*80}\n")

    # Load embryo data
    print("Loading embryo data...")
    mask, um_per_pixel, metadata = load_embryo_data(snip_id)
    print(f"  Mask shape: {mask.shape}")
    print(f"  Mask area: {mask.sum():,} px")
    print(f"  um_per_pixel: {um_per_pixel:.4f}")
    print()

    results = []

    # Test Gaussian blur with different sigmas
    print(f"Testing Gaussian Blur (sigmas: {gaussian_sigmas})")
    print("-" * 80)

    for sigma in gaussian_sigmas:
        print(f"\n  Testing sigma = {sigma}...")

        try:
            # Apply preprocessing
            t0 = time.perf_counter()
            preprocessed_mask = apply_gaussian_preprocessing(mask, sigma=sigma, threshold=0.7)
            preprocess_time = (time.perf_counter() - t0) * 1000

            # Extract centerline
            t0 = time.perf_counter()
            spline_x, spline_y, curvature, arc_length = extract_centerline(
                preprocessed_mask,
                method='geodesic',
                um_per_pixel=um_per_pixel,
                bspline_smoothing=5.0
            )
            extraction_time = (time.perf_counter() - t0) * 1000

            # Check if extraction succeeded
            success = len(spline_x) > 0 and len(curvature) > 0

            # Compute metrics if successful
            if success:
                mean_curv = np.mean(np.abs(curvature))
                std_curv = np.std(curvature)
                max_curv = np.max(np.abs(curvature))
                length = arc_length[-1] if len(arc_length) > 0 else 0
            else:
                mean_curv = std_curv = max_curv = length = np.nan

            print(f"    Preprocessing: {preprocess_time:.1f} ms")
            print(f"    Extraction: {extraction_time:.1f} ms")
            print(f"    Success: {success}")
            if success:
                print(f"    Length: {length:.2f} um")
                print(f"    Mean curvature: {mean_curv:.4f} 1/um")

            results.append({
                'snip_id': snip_id,
                'method': 'gaussian_blur',
                'parameter': sigma,
                'parameter_name': 'sigma',
                'success': success,
                'preprocess_time_ms': preprocess_time,
                'extraction_time_ms': extraction_time,
                'total_time_ms': preprocess_time + extraction_time,
                'length_um': length,
                'mean_curvature_per_um': mean_curv,
                'std_curvature_per_um': std_curv,
                'max_curvature_per_um': max_curv,
                'preprocessed_mask': preprocessed_mask,
                'spline_x': spline_x,
                'spline_y': spline_y,
                'curvature': curvature,
                'arc_length': arc_length,
            })

        except Exception as e:
            print(f"    FAILED: {e}")
            results.append({
                'snip_id': snip_id,
                'method': 'gaussian_blur',
                'parameter': sigma,
                'parameter_name': 'sigma',
                'success': False,
                'preprocess_time_ms': np.nan,
                'extraction_time_ms': np.nan,
                'total_time_ms': np.nan,
                'length_um': np.nan,
                'mean_curvature_per_um': np.nan,
                'std_curvature_per_um': np.nan,
                'max_curvature_per_um': np.nan,
                'preprocessed_mask': mask,  # Return original for visualization
                'spline_x': np.array([]),
                'spline_y': np.array([]),
                'curvature': np.array([]),
                'arc_length': np.array([]),
            })

    # Test alpha shape with different alpha values
    print(f"\n\nTesting Alpha Shape (alphas: {alpha_values})")
    print("-" * 80)

    for alpha in alpha_values:
        print(f"\n  Testing alpha = {alpha}...")

        try:
            # Apply preprocessing
            t0 = time.perf_counter()
            preprocessed_mask = apply_alpha_shape_preprocessing(mask, alpha=alpha)
            preprocess_time = (time.perf_counter() - t0) * 1000

            # Extract centerline
            t0 = time.perf_counter()
            spline_x, spline_y, curvature, arc_length = extract_centerline(
                preprocessed_mask,
                method='geodesic',
                um_per_pixel=um_per_pixel,
                bspline_smoothing=5.0
            )
            extraction_time = (time.perf_counter() - t0) * 1000

            # Check if extraction succeeded
            success = len(spline_x) > 0 and len(curvature) > 0

            # Compute metrics if successful
            if success:
                mean_curv = np.mean(np.abs(curvature))
                std_curv = np.std(curvature)
                max_curv = np.max(np.abs(curvature))
                length = arc_length[-1] if len(arc_length) > 0 else 0
            else:
                mean_curv = std_curv = max_curv = length = np.nan

            print(f"    Preprocessing: {preprocess_time:.1f} ms")
            print(f"    Extraction: {extraction_time:.1f} ms")
            print(f"    Success: {success}")
            if success:
                print(f"    Length: {length:.2f} um")
                print(f"    Mean curvature: {mean_curv:.4f} 1/um")

            results.append({
                'snip_id': snip_id,
                'method': 'alpha_shape',
                'parameter': alpha,
                'parameter_name': 'alpha',
                'success': success,
                'preprocess_time_ms': preprocess_time,
                'extraction_time_ms': extraction_time,
                'total_time_ms': preprocess_time + extraction_time,
                'length_um': length,
                'mean_curvature_per_um': mean_curv,
                'std_curvature_per_um': std_curv,
                'max_curvature_per_um': max_curv,
                'preprocessed_mask': preprocessed_mask,
                'spline_x': spline_x,
                'spline_y': spline_y,
                'curvature': curvature,
                'arc_length': arc_length,
            })

        except Exception as e:
            print(f"    FAILED: {e}")
            results.append({
                'snip_id': snip_id,
                'method': 'alpha_shape',
                'parameter': alpha,
                'parameter_name': 'alpha',
                'success': False,
                'preprocess_time_ms': np.nan,
                'extraction_time_ms': np.nan,
                'total_time_ms': np.nan,
                'length_um': np.nan,
                'mean_curvature_per_um': np.nan,
                'std_curvature_per_um': np.nan,
                'max_curvature_per_um': np.nan,
                'preprocessed_mask': mask,  # Return original for visualization
                'spline_x': np.array([]),
                'spline_y': np.array([]),
                'curvature': np.array([]),
                'arc_length': np.array([]),
            })

    return results, mask


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_preprocessing_comparison(results: list, original_mask: np.ndarray, snip_id: str, output_path: Path):
    """
    Generate 2×5 grid visualization comparing preprocessing methods and parameters.

    Layout:
        Row 1: Gaussian Blur (sigma = 5, 10, 15, 20, 25)
        Row 2: Alpha Shape (alpha = 30, 50, 70, 90, 110)

    Args:
        results: List of result dictionaries
        original_mask: Original mask for reference
        snip_id: Embryo identifier
        output_path: Where to save the plot
    """
    # Separate results by method
    gaussian_results = [r for r in results if r['method'] == 'gaussian_blur']
    alpha_results = [r for r in results if r['method'] == 'alpha_shape']

    # Sort by parameter value
    gaussian_results = sorted(gaussian_results, key=lambda x: x['parameter'])
    alpha_results = sorted(alpha_results, key=lambda x: x['parameter'])

    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))

    # Plot Gaussian blur results (row 0)
    for idx, result in enumerate(gaussian_results):
        ax = axes[0, idx]

        # Show mask
        ax.imshow(result['preprocessed_mask'], cmap='gray', alpha=0.7)

        # Plot centerline if successful
        if result['success'] and len(result['spline_x']) > 0:
            ax.plot(result['spline_y'], result['spline_x'], 'r-', linewidth=2.5, label='Centerline')
            ax.plot(result['spline_y'][0], result['spline_x'][0], 'go', markersize=8, label='Head')
            ax.plot(result['spline_y'][-1], result['spline_x'][-1], 'bo', markersize=8, label='Tail')

            # Add metrics annotation
            title = f"Gaussian σ={result['parameter']}\n"
            title += f"✓ Success\n"
            title += f"Length: {result['length_um']:.1f} μm\n"
            title += f"Mean κ: {result['mean_curvature_per_um']:.3f}"
            ax.set_title(title, fontsize=11, fontweight='bold', color='green')
        else:
            # Failed
            title = f"Gaussian σ={result['parameter']}\n✗ FAILED"
            ax.set_title(title, fontsize=11, fontweight='bold', color='red')

        ax.axis('off')

    # Plot alpha shape results (row 1)
    for idx, result in enumerate(alpha_results):
        ax = axes[1, idx]

        # Show mask
        ax.imshow(result['preprocessed_mask'], cmap='gray', alpha=0.7)

        # Plot centerline if successful
        if result['success'] and len(result['spline_x']) > 0:
            ax.plot(result['spline_y'], result['spline_x'], 'r-', linewidth=2.5, label='Centerline')
            ax.plot(result['spline_y'][0], result['spline_x'][0], 'go', markersize=8, label='Head')
            ax.plot(result['spline_y'][-1], result['spline_x'][-1], 'bo', markersize=8, label='Tail')

            # Add metrics annotation
            title = f"Alpha Shape α={result['parameter']}\n"
            title += f"✓ Success\n"
            title += f"Length: {result['length_um']:.1f} μm\n"
            title += f"Mean κ: {result['mean_curvature_per_um']:.3f}"
            ax.set_title(title, fontsize=11, fontweight='bold', color='green')
        else:
            # Failed
            title = f"Alpha Shape α={result['parameter']}\n✗ FAILED"
            ax.set_title(title, fontsize=11, fontweight='bold', color='red')

        ax.axis('off')

    # Add main title
    fig.suptitle(f"Preprocessing Parameter Comparison: {snip_id}",
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved visualization: {output_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main test pipeline."""

    # Define test parameters
    embryos = [
        "20251017_combined_A02_e01_t0064",  # Gaussian blur failed
        "20250512_E06_e01_t0086",
    ]

    gaussian_sigmas = [5, 10, 15, 20, 25]
    alpha_values = [30, 50, 70, 90, 110]

    # Output directory
    output_dir = Path(__file__).parent / "preprocessing_test_results"
    output_dir.mkdir(exist_ok=True)

    print("\n" + "="*80)
    print("PREPROCESSING PARAMETER SWEEP TEST")
    print("="*80)
    print(f"Testing {len(embryos)} embryos")
    print(f"Gaussian sigmas: {gaussian_sigmas}")
    print(f"Alpha values: {alpha_values}")
    print(f"Output directory: {output_dir}")
    print("="*80)

    all_results = []

    # Test each embryo
    for snip_id in embryos:
        results, original_mask = test_preprocessing_on_embryo(snip_id, gaussian_sigmas, alpha_values)
        all_results.extend(results)

        # Generate visualization for this embryo
        # Create safe filename (replace slashes)
        safe_filename = snip_id.replace("/", "_").replace("\\", "_")
        output_path = output_dir / f"preprocessing_comparison_{safe_filename}.png"
        plot_preprocessing_comparison(results, original_mask, snip_id, output_path)

    # Save results to CSV
    results_df = pd.DataFrame([
        {
            'snip_id': r['snip_id'],
            'method': r['method'],
            'parameter': r['parameter'],
            'parameter_name': r['parameter_name'],
            'success': r['success'],
            'preprocess_time_ms': r['preprocess_time_ms'],
            'extraction_time_ms': r['extraction_time_ms'],
            'total_time_ms': r['total_time_ms'],
            'length_um': r['length_um'],
            'mean_curvature_per_um': r['mean_curvature_per_um'],
            'std_curvature_per_um': r['std_curvature_per_um'],
            'max_curvature_per_um': r['max_curvature_per_um'],
        }
        for r in all_results
    ])

    csv_path = output_dir / "preprocessing_test_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved results: {csv_path}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for snip_id in embryos:
        print(f"\n{snip_id}:")
        print("-" * 80)

        embryo_results = [r for r in all_results if r['snip_id'] == snip_id]

        # Gaussian results
        gaussian_results = [r for r in embryo_results if r['method'] == 'gaussian_blur']
        gaussian_success = sum(1 for r in gaussian_results if r['success'])
        print(f"  Gaussian Blur: {gaussian_success}/{len(gaussian_results)} successful")
        for r in gaussian_results:
            status = "✓" if r['success'] else "✗"
            print(f"    {status} σ={r['parameter']:>2}: ", end="")
            if r['success']:
                print(f"Length={r['length_um']:>6.1f} μm, Mean κ={r['mean_curvature_per_um']:>6.3f}")
            else:
                print("FAILED")

        # Alpha results
        alpha_results = [r for r in embryo_results if r['method'] == 'alpha_shape']
        alpha_success = sum(1 for r in alpha_results if r['success'])
        print(f"\n  Alpha Shape: {alpha_success}/{len(alpha_results)} successful")
        for r in alpha_results:
            status = "✓" if r['success'] else "✗"
            print(f"    {status} α={r['parameter']:>3}: ", end="")
            if r['success']:
                print(f"Length={r['length_um']:>6.1f} μm, Mean κ={r['mean_curvature_per_um']:>6.3f}")
            else:
                print("FAILED")

    print("\n" + "="*80)
    print("Test complete!")
    print("="*80)


if __name__ == "__main__":
    main()
