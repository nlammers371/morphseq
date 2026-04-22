from __future__ import annotations

import numpy as np
from scipy import ndimage


def normalize_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Convert arbitrary mask values to a boolean mask."""
    return np.asarray(mask) > 0


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component in a binary mask."""
    binary = normalize_binary_mask(mask)
    labeled, n_components = ndimage.label(binary)
    if n_components <= 1:
        return binary

    sizes = np.bincount(labeled.ravel())
    if sizes.size <= 1:
        return binary
    sizes[0] = 0
    largest_label = int(np.argmax(sizes))
    return labeled == largest_label


def remove_small_components(mask: np.ndarray, min_size: int = 32) -> np.ndarray:
    """Remove connected components smaller than `min_size` pixels."""
    if min_size <= 0:
        return normalize_binary_mask(mask)

    binary = normalize_binary_mask(mask)
    labeled, n_components = ndimage.label(binary)
    if n_components <= 1:
        return binary

    sizes = np.bincount(labeled.ravel())
    if sizes.size <= 1:
        return binary
    sizes[0] = 0
    keep_labels = np.where(sizes >= int(min_size))[0]
    if keep_labels.size == 0:
        return largest_connected_component(binary)
    cleaned = np.isin(labeled, keep_labels)
    return cleaned


def fill_small_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes in a binary mask."""
    binary = normalize_binary_mask(mask)
    return ndimage.binary_fill_holes(binary)


def clean_embryo_mask(
    mask: np.ndarray,
    *,
    min_component_size: int = 32,
    fill_holes_first: bool = True,
    keep_largest_component: bool = True,
) -> np.ndarray:
    """Apply generic embryo-mask cleanup before downstream measurements."""
    binary = normalize_binary_mask(mask)
    if fill_holes_first:
        binary = fill_small_holes(binary)
    binary = remove_small_components(binary, min_size=min_component_size)
    if keep_largest_component:
        binary = largest_connected_component(binary)
    return np.asarray(binary, dtype=bool)

