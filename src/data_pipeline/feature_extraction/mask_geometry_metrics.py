"""
Mask geometry feature extraction from SAM2 masks.

Computes area, perimeter, length, width, and other contour-based metrics.
Extracted from build03A_process_images.py get_embryo_stats function (lines 750-770).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import skimage.io as io
from skimage.measure import find_contours

from data_pipeline.segmentation_and_tracking.utils.mask_processing import clean_embryo_mask
from data_pipeline.shared.path_contracts import require_existing_path


def compute_mask_geometry(
    mask: np.ndarray,
    pixel_size_um: float,
) -> Dict:
    """Compute geometry metrics from binary mask."""
    mask_binary = clean_embryo_mask(mask).astype(np.uint8)
    area_px = np.sum(mask_binary)
    area_um2 = area_px * (pixel_size_um ** 2)

    if area_px > 0:
        yy, xx = np.indices(mask_binary.shape)
        centroid_x_px = np.sum(xx[mask_binary == 1]) / area_px
        centroid_y_px = np.sum(yy[mask_binary == 1]) / area_px
    else:
        centroid_x_px = 0.0
        centroid_y_px = 0.0

    centroid_x_um = centroid_x_px * pixel_size_um
    centroid_y_um = centroid_y_px * pixel_size_um

    if area_px > 1:
        yy, xx = np.indices(mask_binary.shape)
        mask_coords = np.c_[xx[mask_binary == 1], yy[mask_binary == 1]]
        pca = PCA(n_components=2)
        coords_rotated = pca.fit_transform(mask_coords)
        length_px, width_px = np.max(coords_rotated, axis=0) - np.min(coords_rotated, axis=0)
        length_um = length_px * pixel_size_um
        width_um = width_px * pixel_size_um
    else:
        length_um = 0.0
        width_um = 0.0

    perimeter_um = 0.0
    try:
        contours = find_contours(mask_binary, level=0.5)
        if contours:
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
    mask_dir: Path | None = None,
    pixel_size_col: str = 'micrometers_per_pixel',
    mask_path_col: str = 'exported_mask_path',
) -> pd.DataFrame:
    """Extract geometry metrics for batch of snips."""
    results = []
    for _, row in tracking_df.iterrows():
        snip_id = row['snip_id']
        mask_path = require_existing_path(
            row.get(mask_path_col),
            context='mask_geometry',
            field_name=mask_path_col,
            row_id=str(snip_id),
        )
        mask = io.imread(mask_path)

        if pixel_size_col not in row.index or pd.isna(row[pixel_size_col]):
            raise ValueError(f"mask_geometry: missing required pixel size column '{pixel_size_col}' for snip_id={snip_id}")
        pixel_size = float(row[pixel_size_col])
        if not np.isfinite(pixel_size) or pixel_size <= 0:
            raise ValueError(f"mask_geometry: invalid pixel size {pixel_size!r} for snip_id={snip_id}")

        metrics = compute_mask_geometry(mask, pixel_size)
        metrics['snip_id'] = snip_id
        results.append(metrics)

    return pd.DataFrame(results)
