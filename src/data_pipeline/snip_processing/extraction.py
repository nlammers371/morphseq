"""
Embryo snip extraction from full-frame images.

Extracts cropped embryo regions from stitched images based on SAM2 masks.
Extracted from build03A_process_images.py (lines 257-414).
"""

import numpy as np
import skimage.io as io
import skimage.exposure
from pathlib import Path
from typing import Tuple, Optional
import warnings
from skimage.transform import rescale, resize


def extract_embryo_crop(
    image: np.ndarray,
    mask: np.ndarray,
    yolk_mask: Optional[np.ndarray],
    target_shape: Tuple[int, int],
    pixel_size_um: float,
    target_pixel_size_um: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract and rescale embryo crop from full-frame image.

    Args:
        image: Full-frame image (H x W)
        mask: Binary embryo mask (H x W)
        yolk_mask: Binary yolk mask (H x W), can be None
        target_shape: Output shape (height, width) in pixels
        pixel_size_um: Current pixel size in micrometers
        target_pixel_size_um: Target pixel size for output

    Returns:
        Tuple of (rescaled_image, rescaled_mask, rescaled_yolk_mask)
    """
    # Handle missing yolk mask
    if yolk_mask is None:
        yolk_mask = np.zeros_like(mask)

    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = skimage.exposure.rescale_intensity(
            image, in_range='image', out_range=(0, 255)
        ).astype(np.uint8)

    # Rescale to target pixel size
    scale_factor = pixel_size_um / target_pixel_size_um

    image_rescaled = rescale(
        image,
        (scale_factor, scale_factor),
        order=1,
        preserve_range=True
    )

    mask_rescaled = resize(
        mask.astype(float),
        image_rescaled.shape,
        order=1
    )

    yolk_rescaled = resize(
        yolk_mask.astype(float),
        image_rescaled.shape,
        order=1
    )

    return image_rescaled, mask_rescaled, yolk_rescaled


def crop_to_embryo_bounds(
    image: np.ndarray,
    mask: np.ndarray,
    yolk_mask: np.ndarray,
    output_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Crop image, mask, and yolk to embryo-centered bounding box.

    Based on build03A crop_embryo_image function.

    Args:
        image: Rescaled and rotated image
        mask: Rescaled and rotated embryo mask
        yolk_mask: Rescaled and rotated yolk mask
        output_shape: Target output shape (height, width)

    Returns:
        Tuple of (cropped_image, cropped_mask, cropped_yolk)
    """
    # Handle empty mask
    if np.sum(mask) == 0:
        return (
            np.zeros(output_shape, dtype=np.uint8),
            np.zeros(output_shape),
            np.zeros(output_shape),
        )

    # Find mask bounds
    y_indices = np.where(np.max(mask, axis=1) > 0.5)[0]
    x_indices = np.where(np.max(mask, axis=0) > 0.5)[0]

    if y_indices.size == 0 or x_indices.size == 0:
        # Degenerate mask after rotation
        return (
            np.zeros(output_shape, dtype=np.uint8),
            np.zeros(output_shape),
            np.zeros(output_shape),
        )

    # Compute centroid
    y_mean = int(np.mean(y_indices))
    x_mean = int(np.mean(x_indices))

    # Calculate crop bounds
    from_shape = mask.shape
    raw_range_y = [
        y_mean - int(output_shape[0] / 2),
        y_mean + int(output_shape[0] / 2)
    ]
    from_range_y = np.asarray([
        np.max([raw_range_y[0], 0]),
        np.min([raw_range_y[1], from_shape[0]])
    ])
    to_range_y = [
        0 + (from_range_y[0] - raw_range_y[0]),
        output_shape[0] + (from_range_y[1] - raw_range_y[1])
    ]

    raw_range_x = [
        x_mean - int(output_shape[1] / 2),
        x_mean + int(output_shape[1] / 2)
    ]
    from_range_x = np.asarray([
        np.max([raw_range_x[0], 0]),
        np.min([raw_range_x[1], from_shape[1]])
    ])
    to_range_x = [
        0 + (from_range_x[0] - raw_range_x[0]),
        output_shape[1] + (from_range_x[1] - raw_range_x[1])
    ]

    # Extract crops
    image_crop = np.zeros(output_shape, dtype=np.uint8)
    image_crop[to_range_y[0]:to_range_y[1], to_range_x[0]:to_range_x[1]] = \
        image[from_range_y[0]:from_range_y[1], from_range_x[0]:from_range_x[1]]

    mask_crop = np.zeros(output_shape)
    mask_crop[to_range_y[0]:to_range_y[1], to_range_x[0]:to_range_x[1]] = \
        mask[from_range_y[0]:from_range_y[1], from_range_x[0]:from_range_x[1]]

    yolk_crop = np.zeros(output_shape)
    yolk_crop[to_range_y[0]:to_range_y[1], to_range_x[0]:to_range_x[1]] = \
        yolk_mask[from_range_y[0]:from_range_y[1], from_range_x[0]:from_range_x[1]]

    return image_crop, mask_crop, yolk_crop
