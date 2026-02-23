"""
Mask Preprocessing for Centerline Extraction

Quick, cheap preprocessing methods that are always applied before spline fitting.
These methods refine the mask boundary to produce smoother centerlines.

Currently implemented:
- apply_gaussian_preprocessing (default): Fast, smooth, works well in most cases

Experimental (not yet implemented):
- apply_alpha_shape_preprocessing: Geometric approach using concave hull
  (To be implemented in future if different geodesic methods are tested)
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def apply_gaussian_preprocessing(mask: np.ndarray, sigma: float = 15.0, threshold: float = 0.7) -> np.ndarray:
    """
    Preprocess mask using Gaussian blur and re-thresholding.

    This is the default preprocessing method - cheap, fast, and effective.
    Smooths the mask boundary by blurring and re-thresholding at a high value,
    which removes small protrusions and creates smoother boundaries.

    Process:
    1. Apply Gaussian blur with sigma parameter
    2. Re-threshold at high value (default 0.7)
    3. Only core regions with many nearby pixels survive

    Args:
        mask: Binary mask (2D numpy array)
        sigma: Gaussian blur sigma parameter (default=15.0)
               Higher = more smoothing, removes finer details
               Optimized value determined empirically (see Phase 3 analysis)
        threshold: Re-threshold value after blur (default=0.7)
                   Higher = more aggressive smoothing

    Returns:
        preprocessed_mask: Smoothed binary mask

    Note:
        This is always applied before spline fitting as it's cheap and fast.
        Typical processing time: <0.1s per mask
        Sigma=15.0 was selected after testing on multiple embryos to prevent
        fin extension while preserving embryo body structure.
    """
    # Apply Gaussian blur
    blurred = gaussian_filter(mask.astype(float), sigma=sigma)

    # Re-threshold at high value to keep only core regions
    preprocessed = blurred > threshold

    return preprocessed.astype(np.uint8)


def apply_preprocessing(mask: np.ndarray, method: str = 'gaussian_blur', **kwargs) -> np.ndarray:
    """
    Apply preprocessing to mask using specified method.

    This is the main entry point for mask preprocessing. Always call this
    before spline fitting for best results.

    Args:
        mask: Binary mask (2D numpy array)
        method: Preprocessing method ('gaussian_blur')
                Default='gaussian_blur'
                (Note: 'alpha_shape' is experimental and not yet implemented)
        **kwargs: Method-specific parameters:
            For 'gaussian_blur': sigma (default=15.0), threshold (default=0.7)

    Returns:
        preprocessed_mask: Smoothed binary mask

    Example:
        >>> # Default Gaussian blur with sigma=15.0
        >>> clean_mask = apply_preprocessing(mask)
        >>>
        >>> # Custom Gaussian blur
        >>> clean_mask = apply_preprocessing(mask, method='gaussian_blur', sigma=20.0)
    """
    if method == 'gaussian_blur':
        return apply_gaussian_preprocessing(mask, **kwargs)
    elif method == 'alpha_shape':
        raise NotImplementedError(
            "Alpha shape preprocessing is experimental and not yet implemented. "
            "Use 'gaussian_blur' (default) for now. "
            "Alpha shape will be implemented when alternative geodesic methods are tested."
        )
    else:
        raise ValueError(f"Unknown preprocessing method: {method}. "
                        f"Use 'gaussian_blur' (only implemented method)")
