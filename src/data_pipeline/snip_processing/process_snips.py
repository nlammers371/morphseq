"""
Complete snip processing pipeline.

Combines extraction, rotation, and augmentation into a single processing flow.
Supports --save-raw-crops config flag as per AGREED_CODE_CHANGES.md.
"""

import numpy as np
import pandas as pd
import skimage.io as io
from pathlib import Path
from typing import Dict, Optional
import warnings
import scipy.ndimage

from .extraction import extract_embryo_crop, crop_to_embryo_bounds
from .rotation import apply_rotation_to_snip
from .augmentation import augment_snip


def process_single_snip(
    snip_id: str,
    image_path: Path,
    mask_path: Path,
    yolk_mask_path: Optional[Path],
    output_shape: tuple,
    pixel_size_um: float,
    target_pixel_size_um: float,
    background_mean: float,
    background_std: float,
    save_raw_crops: bool = True,
    raw_crops_dir: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
) -> Dict:
    """
    Process a single embryo snip through the complete pipeline.

    Pipeline:
    1. Load image and masks
    2. Extract and rescale to target pixel size
    3. Rotate using PCA
    4. Crop to embryo-centered bounds
    5. Apply CLAHE and background blending
    6. Save outputs

    Args:
        snip_id: Unique snip identifier
        image_path: Path to stitched FF image
        mask_path: Path to embryo mask
        yolk_mask_path: Path to yolk mask (optional)
        output_shape: Target output shape (height, width)
        pixel_size_um: Source pixel size in micrometers
        target_pixel_size_um: Target pixel size for output
        background_mean: Background pixel intensity mean
        background_std: Background pixel intensity std
        save_raw_crops: Save unprocessed TIF crops (config flag)
        raw_crops_dir: Directory for raw TIF crops
        processed_dir: Directory for processed JPG snips

    Returns:
        Dictionary with processing metadata
    """
    # Load image and masks
    image = io.imread(image_path)
    mask = io.imread(mask_path)

    # Ensure grayscale
    if len(image.shape) == 3:
        image = image[:, :, 0]

    # Ensure binary mask
    if mask.max() > 1:
        mask = (mask > 0).astype(np.uint8)

    # Load yolk mask if available
    if yolk_mask_path and yolk_mask_path.exists():
        yolk_mask = io.imread(yolk_mask_path)
        yolk_mask = (yolk_mask > 0).astype(np.uint8)
    else:
        yolk_mask = np.zeros_like(mask)

    # Step 1: Extract and rescale
    image_rescaled, mask_rescaled, yolk_rescaled = extract_embryo_crop(
        image,
        mask,
        yolk_mask,
        output_shape,
        pixel_size_um,
        target_pixel_size_um,
    )

    # Step 2: Rotate using PCA
    image_rotated, mask_rotated, yolk_rotated, rotation_angle = apply_rotation_to_snip(
        image_rescaled,
        mask_rescaled,
        yolk_rescaled,
    )

    # Step 3: Crop to embryo bounds
    image_cropped, mask_cropped, yolk_cropped = crop_to_embryo_bounds(
        image_rotated,
        mask_rotated,
        yolk_rotated,
        output_shape,
    )

    # Save raw crop if requested (config flag)
    if save_raw_crops and raw_crops_dir:
        raw_path = raw_crops_dir / f"{snip_id}.tif"
        raw_crops_dir.mkdir(parents=True, exist_ok=True)
        io.imsave(raw_path, image_cropped, check_contrast=False)

    # Step 4: Apply augmentation (CLAHE + noise blending)
    augmented, clahe_only = augment_snip(
        image_cropped,
        mask_cropped,
        background_mean,
        background_std,
        blend_radius_um=30.0,
        pixel_size_um=target_pixel_size_um,
    )

    # Step 5: Save processed output
    if processed_dir:
        processed_path = processed_dir / f"{snip_id}.jpg"
        processed_dir.mkdir(parents=True, exist_ok=True)
        io.imsave(processed_path, augmented, check_contrast=False)

    # Return metadata
    return {
        'snip_id': snip_id,
        'rotation_angle': float(rotation_angle),
        'processed_path': str(processed_path) if processed_dir else None,
        'raw_crop_path': str(raw_path) if save_raw_crops and raw_crops_dir else None,
    }


def estimate_background_statistics(
    image_paths: list,
    mask_paths: list,
    n_samples: int = 100,
    seed: int = 309,
) -> tuple:
    """
    Estimate background pixel statistics from sample images.

    Extracts background pixels (outside embryo and via masks) to compute
    mean and standard deviation for noise generation.

    Args:
        image_paths: List of paths to stitched images
        mask_paths: List of paths to embryo masks
        n_samples: Number of images to sample
        seed: Random seed for sampling

    Returns:
        Tuple of (mean, std) for background pixels
    """
    np.random.seed(seed)

    # Sample subset of images
    n_available = min(len(image_paths), len(mask_paths))
    sample_indices = np.random.choice(
        n_available,
        min(n_samples, n_available),
        replace=False
    )

    background_pixels = []

    for idx in sample_indices:
        try:
            image = io.imread(image_paths[idx])
            mask = io.imread(mask_paths[idx])

            # Create background mask (inverse of embryo)
            bg_mask = np.ones_like(mask)
            bg_mask[mask > 0] = 0

            # Extract background pixels
            bg_pixels = image[bg_mask == 1].flatten().tolist()
            background_pixels.extend(bg_pixels)

        except Exception as e:
            warnings.warn(f"Failed to load sample {idx}: {e}")
            continue

    if not background_pixels:
        warnings.warn("No background pixels found, using defaults")
        return 128.0, 30.0

    mean = np.mean(background_pixels)
    std = np.std(background_pixels)

    return float(mean), float(std)
