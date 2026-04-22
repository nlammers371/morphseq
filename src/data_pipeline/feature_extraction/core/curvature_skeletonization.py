from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from data_pipeline.segmentation_and_tracking.utils.mask_processing import clean_embryo_mask


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, n_components = ndimage.label(np.asarray(mask, dtype=bool))
    if n_components <= 1:
        return np.asarray(mask, dtype=bool)
    sizes = np.bincount(labeled.ravel())
    if sizes.size <= 1:
        return np.asarray(mask, dtype=bool)
    sizes[0] = 0
    largest_label = int(np.argmax(sizes))
    return labeled == largest_label


def skeletonize_embryo_mask(
    mask: np.ndarray,
    *,
    min_component_size: int = 32,
) -> np.ndarray:
    """Return the cleaned skeleton of an embryo mask."""
    clean = clean_embryo_mask(mask, min_component_size=min_component_size)
    skel = skeletonize(clean)
    if not np.any(skel):
        return np.zeros_like(clean, dtype=bool)
    return _largest_component(skel)


def extract_centerline_points(
    mask: np.ndarray,
    *,
    min_component_size: int = 32,
) -> np.ndarray:
    """Return ordered centerline points in x/y pixel coordinates."""
    skel = skeletonize_embryo_mask(mask, min_component_size=min_component_size)
    coords_yx = np.argwhere(skel)
    if coords_yx.shape[0] < 3:
        return np.empty((0, 2), dtype=np.float64)

    coords_xy = coords_yx[:, ::-1].astype(np.float64)
    centered = coords_xy - coords_xy.mean(axis=0, keepdims=True)
    if coords_xy.shape[0] > 3:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        axis = vt[0]
        order = np.argsort(centered @ axis)
        coords_xy = coords_xy[order]

    # Remove repeated points that can break arc-length derivatives.
    keep = np.ones(len(coords_xy), dtype=bool)
    if len(coords_xy) > 1:
        deltas = np.diff(coords_xy, axis=0)
        keep[1:] = np.any(np.abs(deltas) > 0, axis=1)
    return coords_xy[keep]

