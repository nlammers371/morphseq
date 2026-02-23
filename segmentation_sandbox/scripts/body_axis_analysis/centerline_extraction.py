"""
High-Level API for Centerline Extraction

Unified interface for extracting body axis centerlines from zebrafish embryo masks.
Automatically selects between Geodesic (robust) and PCA (fast) methods based on mask morphology.

Quick start:
    >>> from body_axis_analysis import extract_centerline
    >>> spline_x, spline_y, curvature, arc_length = extract_centerline(mask)
"""

import numpy as np
from skimage.measure import regionprops

from .geodesic_method import GeodesicCenterlineAnalyzer
from .pca_method import PCACenterlineAnalyzer
from .mask_preprocessing import apply_preprocessing
from .spline_utils import orient_spline_head_to_tail, align_spline_orientation


def extract_centerline(mask: np.ndarray,
                       method: str = 'geodesic',
                       preprocess: str = 'gaussian_blur',
                       orient_head_to_tail: bool = True,
                       um_per_pixel: float = 1.0,
                       bspline_smoothing: float = 5.0,
                       random_seed: int = 42,
                       fast: bool = True,
                       return_intermediate: bool = False,
                       **kwargs) -> tuple:
    """
    Extract centerline from embryo mask with unified API.

    This is the main entry point for centerline extraction. Preprocesses the mask
    and extracts centerline using specified method (Geodesic or PCA).

    Args:
        mask: Binary mask (2D numpy array)
        method: Centerline extraction method:
                - 'geodesic' (default): Robust geodesic skeleton approach
                  Handles highly curved embryos, slower (~14.6s/embryo)
                - 'pca': Fast PCA-based slicing approach
                  Better for normal shapes, faster (~5.3s/embryo)
        preprocess: Mask preprocessing method:
                   - 'gaussian_blur' (default): Fast and effective
        orient_head_to_tail: If True, orient spline from head to tail (default=True)
        um_per_pixel: Conversion factor from pixels to microns (default=1.0)
        bspline_smoothing: B-spline smoothing parameter (default=5.0)
                          Passed to both geodesic and PCA methods
        random_seed: Seed for reproducible geodesic endpoint detection (default=42)
        fast: Use optimized O(N) geodesic graph building (default=True)
              Set to False to use original O(N²) method for backward compatibility
        return_intermediate: If True, return full analysis results dict (default=False)
        **kwargs: Additional preprocessing parameters (passed to apply_preprocessing)

    Returns:
        If return_intermediate=False (default):
            (spline_x, spline_y, curvature, arc_length): Tuple of smoothed centerline
            and curvature measurements

        If return_intermediate=True:
            results_dict: Full analysis dictionary with all intermediate results

    Raises:
        ValueError: If mask is empty or invalid
        RuntimeError: If centerline extraction fails

    Example:
        >>> # Simple usage with default settings
        >>> spline_x, spline_y, curvature, arc_length = extract_centerline(mask)
        >>>
        >>> # Get full analysis results
        >>> results = extract_centerline(mask, return_intermediate=True)
        >>> print(f"Total length: {results['stats']['total_length']:.1f} pixels")
        >>>
        >>> # Use PCA method for speed
        >>> x, y, curv, arc = extract_centerline(mask, method='pca')
        >>>
        >>> # Tune B-spline smoothing
        >>> x, y, curv, arc = extract_centerline(mask, bspline_smoothing=3.0)
    """
    # Validate input
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("Mask must be a 2D numpy array")

    if mask.sum() == 0:
        raise ValueError("Mask is empty")

    # Step 1: Preprocess mask
    preprocessed_mask = apply_preprocessing(mask, method=preprocess, **kwargs)

    # Step 2: Extract centerline
    if method == 'geodesic':
        results = _extract_geodesic(preprocessed_mask, um_per_pixel,
                                   bspline_smoothing=bspline_smoothing,
                                   random_seed=random_seed,
                                   fast=fast)
    elif method == 'pca':
        results = _extract_pca(preprocessed_mask, um_per_pixel,
                             bspline_smoothing=bspline_smoothing)
    else:
        raise ValueError(f"Unknown method: '{method}'. Use 'geodesic' or 'pca'")

    # Step 4: Orient spline (optional)
    if orient_head_to_tail and 'centerline_smoothed' in results:
        spline_x = results['centerline_smoothed'][:, 0]
        spline_y = results['centerline_smoothed'][:, 1]
        spline_x, spline_y, _ = orient_spline_head_to_tail(spline_x, spline_y, mask)
        results['centerline_smoothed'] = np.column_stack([spline_x, spline_y])

    # Step 5: Return results
    if return_intermediate:
        return results
    else:
        # Extract standard outputs
        spline_x = results['centerline_smoothed'][:, 0]
        spline_y = results['centerline_smoothed'][:, 1]
        curvature = results['curvature']
        arc_length = results['arc_length']
        return spline_x, spline_y, curvature, arc_length


def _extract_geodesic(mask: np.ndarray, um_per_pixel: float,
                      bspline_smoothing: float = 5.0,
                      random_seed: int = 42,
                      fast: bool = True) -> dict:
    """
    Extract centerline using Geodesic method.

    Args:
        mask: Binary mask
        um_per_pixel: Conversion factor
        bspline_smoothing: B-spline smoothing parameter
        random_seed: Seed for reproducible endpoint detection
        fast: Use optimized O(N) graph building (default=True)

    Returns:
        results: Analysis dictionary

    Raises:
        RuntimeError: If extraction fails
    """
    try:
        analyzer = GeodesicCenterlineAnalyzer(mask, um_per_pixel=um_per_pixel,
                                             bspline_smoothing=bspline_smoothing,
                                             random_seed=random_seed,
                                             fast=fast)
        results = analyzer.analyze()
        return results
    except Exception as e:
        raise RuntimeError(f"Geodesic centerline extraction failed: {e}")


def _extract_pca(mask: np.ndarray, um_per_pixel: float,
                bspline_smoothing: float = 5.0) -> dict:
    """
    Extract centerline using PCA method.

    Args:
        mask: Binary mask
        um_per_pixel: Conversion factor
        bspline_smoothing: B-spline smoothing parameter

    Returns:
        results: Analysis dictionary

    Raises:
        RuntimeError: If extraction fails
    """
    try:
        analyzer = PCACenterlineAnalyzer(mask, um_per_pixel=um_per_pixel,
                                        bspline_smoothing=bspline_smoothing)
        results = analyzer.analyze()
        return results
    except Exception as e:
        raise RuntimeError(f"PCA centerline extraction failed: {e}")


def compare_methods(mask: np.ndarray, um_per_pixel: float = 1.0,
                   preprocess: str = 'gaussian_blur',
                   bspline_smoothing: float = 5.0,
                   random_seed: int = 42,
                   fast: bool = True) -> dict:
    """
    Compare both Geodesic and PCA methods on the same mask.

    Applies preprocessing to both methods for fair comparison (matches main pipeline).

    Args:
        mask: Binary mask
        um_per_pixel: Conversion factor
        preprocess: Preprocessing method (default='gaussian_blur')
        bspline_smoothing: B-spline smoothing parameter (default=5.0)
        random_seed: Seed for reproducible geodesic endpoint detection (default=42)
        fast: Use optimized O(N) geodesic graph building (default=True)

    Returns:
        comparison: Dictionary with results from both methods:
            - 'geodesic': Geodesic method results
            - 'pca': PCA method results
            - 'hausdorff_distance': Hausdorff distance between centerlines
            - 'mean_aligned_distance': Mean distance after alignment and resampling
            - 'note': Methods should be aligned before direct comparison

    Example:
        >>> comparison = compare_methods(mask)
        >>> print(f"Hausdorff distance: {comparison['hausdorff_distance']:.1f} pixels")
        >>> print(f"Methods agree: {comparison['mean_aligned_distance'] < 20:.0f}")
    """
    try:
        # Preprocess mask (same as main pipeline)
        preprocessed_mask = apply_preprocessing(mask, method=preprocess)

        # Extract with both methods
        geodesic_results = _extract_geodesic(preprocessed_mask, um_per_pixel,
                                            bspline_smoothing=bspline_smoothing,
                                            random_seed=random_seed,
                                            fast=fast)
        pca_results = _extract_pca(preprocessed_mask, um_per_pixel,
                                   bspline_smoothing=bspline_smoothing)

        # Get centerlines
        geodesic_line = geodesic_results['centerline_smoothed']
        pca_line = pca_results['centerline_smoothed']

        # Align orientations
        geodesic_x, geodesic_y, _ = geodesic_line[:, 0], geodesic_line[:, 1], False
        pca_x, pca_y, was_flipped = align_spline_orientation(
            geodesic_x, geodesic_y, pca_line[:, 0], pca_line[:, 1]
        )
        pca_aligned = np.column_stack([pca_x, pca_y])
        geodesic_aligned = np.column_stack([geodesic_x, geodesic_y])

        # Resample to same length for fair comparison
        min_len = min(len(geodesic_aligned), len(pca_aligned))
        geodesic_resampled = geodesic_aligned[
            np.linspace(0, len(geodesic_aligned)-1, min_len, dtype=int)
        ]
        pca_resampled = pca_aligned[
            np.linspace(0, len(pca_aligned)-1, min_len, dtype=int)
        ]

        # Hausdorff distance (on original unaligned lines)
        hausdorff = _compute_hausdorff_distance(geodesic_line, pca_line)

        # Mean distance after alignment and resampling
        aligned_distance = np.mean(
            np.linalg.norm(geodesic_resampled - pca_resampled, axis=1)
        )

        return {
            'geodesic': geodesic_results,
            'pca': pca_results,
            'hausdorff_distance': hausdorff,
            'mean_aligned_distance': aligned_distance,
            'pca_was_flipped': was_flipped,
            'note': 'Both methods preprocessed and compared on same mask'
        }

    except Exception as e:
        raise RuntimeError(f"Method comparison failed: {e}")


def _compute_hausdorff_distance(line1: np.ndarray, line2: np.ndarray) -> float:
    """
    Compute Hausdorff distance between two centerlines.

    Hausdorff distance = max(max(min_dist(P1→P2)), max(min_dist(P2→P1)))
    Measures how far apart the farthest point of each line is from the other.

    Args:
        line1, line2: (N, 2) arrays of (x, y) coordinates

    Returns:
        distance: Hausdorff distance in pixels
    """
    if len(line1) == 0 or len(line2) == 0:
        return float('inf')

    # Distance from each point in line1 to nearest point in line2
    distances_1to2 = []
    for point in line1:
        min_dist = np.min(np.linalg.norm(line2 - point, axis=1))
        distances_1to2.append(min_dist)

    # Distance from each point in line2 to nearest point in line1
    distances_2to1 = []
    for point in line2:
        min_dist = np.min(np.linalg.norm(line1 - point, axis=1))
        distances_2to1.append(min_dist)

    # Hausdorff distance is max of both directions
    hausdorff = max(max(distances_1to2), max(distances_2to1))
    return float(hausdorff)
