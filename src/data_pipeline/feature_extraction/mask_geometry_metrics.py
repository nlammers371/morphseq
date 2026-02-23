"""
Mask geometry feature extraction from SAM2 masks.

Computes area, perimeter, length, width, and other contour-based metrics.
Extracted from build03A_process_images.py get_embryo_stats function (lines 750-770).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from sklearn.decomposition import PCA
import skimage.io as io
from skimage.measure import find_contours


def compute_mask_geometry(
    mask: np.ndarray,
    pixel_size_um: float,
) -> Dict:
    """
    Compute geometry metrics from binary mask.

    Args:
        mask: Binary embryo mask (H x W)
        pixel_size_um: Pixel size in micrometers

    Returns:
        Dictionary with geometry metrics in micrometers
    """
    # Ensure binary mask
    mask_binary = (mask > 0).astype(np.uint8)

    # Area in pixels
    area_px = np.sum(mask_binary)

    # Convert to micrometers squared
    area_um2 = area_px * (pixel_size_um ** 2)

    # Centroid in pixels
    if area_px > 0:
        yy, xx = np.indices(mask_binary.shape)
        centroid_x_px = np.sum(xx[mask_binary == 1]) / area_px
        centroid_y_px = np.sum(yy[mask_binary == 1]) / area_px
    else:
        centroid_x_px = 0.0
        centroid_y_px = 0.0

    # Convert centroid to micrometers
    centroid_x_um = centroid_x_px * pixel_size_um
    centroid_y_um = centroid_y_px * pixel_size_um

    # Length and width via PCA
    if area_px > 1:
        yy, xx = np.indices(mask_binary.shape)
        mask_coords = np.c_[xx[mask_binary == 1], yy[mask_binary == 1]]

        pca = PCA(n_components=2)
        coords_rotated = pca.fit_transform(mask_coords)

        # Principal axes span
        length_px, width_px = (
            np.max(coords_rotated, axis=0) - np.min(coords_rotated, axis=0)
        )

        length_um = length_px * pixel_size_um
        width_um = width_px * pixel_size_um
    else:
        length_um = 0.0
        width_um = 0.0

    # Perimeter from contour
    perimeter_um = 0.0
    try:
        contours = find_contours(mask_binary, level=0.5)
        if contours:
            # Use longest contour
            longest_contour = max(contours, key=len)
            perimeter_px = len(longest_contour)
            perimeter_um = perimeter_px * pixel_size_um
    except Exception:
        pass

    return {
        'area_um2': float(area_um2),
        'perimeter_um': float(perimeter_um),
        'length_um': float(length_um),
        'width_um': float(width_um),
        'centroid_x_um': float(centroid_x_um),
        'centroid_y_um': float(centroid_y_um),
    }


def extract_geometry_metrics_batch(
    tracking_df: pd.DataFrame,
    mask_dir: Path,
    pixel_size_col: str = 'micrometers_per_pixel',
) -> pd.DataFrame:
    """
    Extract geometry metrics for batch of snips.

    Args:
        tracking_df: Segmentation tracking DataFrame with snip_id and paths
        mask_dir: Directory containing mask images
        pixel_size_col: Column name for pixel size

    Returns:
        DataFrame with geometry metrics per snip_id
    """
    results = []

    for idx, row in tracking_df.iterrows():
        snip_id = row['snip_id']

        # Load mask
        mask_path = mask_dir / f"{snip_id}_mask.png"
        if not mask_path.exists():
            # Try alternative naming
            mask_path = mask_dir / f"{row['image_id']}_masks.png"

        if not mask_path.exists():
            results.append({
                'snip_id': snip_id,
                'area_um2': np.nan,
                'perimeter_um': np.nan,
                'length_um': np.nan,
                'width_um': np.nan,
                'centroid_x_um': np.nan,
                'centroid_y_um': np.nan,
            })
            continue

        try:
            mask = io.imread(mask_path)

            # Extract pixel size
            pixel_size = row[pixel_size_col] if pixel_size_col in row else 1.0

            # Compute metrics
            metrics = compute_mask_geometry(mask, pixel_size)
            metrics['snip_id'] = snip_id

            results.append(metrics)

        except Exception as e:
            print(f"Warning: Failed to process {snip_id}: {e}")
            results.append({
                'snip_id': snip_id,
                'area_um2': np.nan,
                'perimeter_um': np.nan,
                'length_um': np.nan,
                'width_um': np.nan,
                'centroid_x_um': np.nan,
                'centroid_y_um': np.nan,
            })

    return pd.DataFrame(results)
