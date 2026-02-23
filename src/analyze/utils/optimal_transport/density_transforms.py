"""Density transforms for binary masks."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import ndimage

from analyze.utils.optimal_transport.config import MassMode


def mask_to_density_uniform(mask: np.ndarray) -> np.ndarray:
    return mask.astype(np.float32)


def mask_to_density_boundary_band(mask: np.ndarray, band_px: int = 3) -> np.ndarray:
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return mask.astype(np.float32)
    eroded = ndimage.binary_erosion(mask_bool)
    boundary = mask_bool & ~eroded
    if band_px > 1:
        boundary = ndimage.binary_dilation(boundary, iterations=band_px - 1)
    return boundary.astype(np.float32)


def mask_to_density_distance_transform(mask: np.ndarray) -> np.ndarray:
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return mask.astype(np.float32)
    dist = ndimage.distance_transform_edt(mask_bool)
    return dist.astype(np.float32)


def mask_to_density(mask: np.ndarray, mass_mode: MassMode, band_px: int = 3) -> np.ndarray:
    if mass_mode == MassMode.UNIFORM:
        return mask_to_density_uniform(mask)
    if mass_mode == MassMode.BOUNDARY_BAND:
        return mask_to_density_boundary_band(mask, band_px=band_px)
    if mass_mode == MassMode.DISTANCE_TRANSFORM:
        return mask_to_density_distance_transform(mask)
    raise ValueError(f"Unknown mass_mode: {mass_mode}")


def enforce_min_mass(
    density: np.ndarray,
    min_mass: float = 1e-8,
    fallback: Optional[np.ndarray] = None,
) -> np.ndarray:
    total = float(density.sum())
    if total >= min_mass:
        return density
    if fallback is not None:
        fallback_total = float(fallback.sum())
        if fallback_total >= min_mass:
            return fallback
    raise ValueError("Density has near-zero total mass.")
