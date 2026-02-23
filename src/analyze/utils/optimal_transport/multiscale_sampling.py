"""Downsampling and support sampling utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import ndimage

from analyze.utils.optimal_transport.config import UOTSupport, SamplingMode


def pad_to_divisible(arr: np.ndarray, divisor: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = arr.shape
    pad_h = (divisor - (h % divisor)) % divisor
    pad_w = (divisor - (w % divisor)) % divisor
    if pad_h == 0 and pad_w == 0:
        return arr, (0, 0)
    padded = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="constant")
    return padded, (pad_h, pad_w)


def downsample_density(density: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return density
    h, w = density.shape
    new_h = h // factor
    new_w = w // factor
    if new_h == 0 or new_w == 0:
        raise ValueError("Downsample factor too large for density shape.")
    trimmed = density[: new_h * factor, : new_w * factor]
    reshaped = trimmed.reshape(new_h, factor, new_w, factor)
    return reshaped.sum(axis=(1, 3))


def _sample_indices(coords: np.ndarray, weights: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0 or coords.size == 0:
        return np.array([], dtype=int)
    if n >= len(coords):
        return np.arange(len(coords))
    probs = weights.astype(np.float64)
    total = probs.sum()
    if total > 0:
        probs = probs / total
        return rng.choice(len(coords), size=n, replace=False, p=probs)
    return rng.choice(len(coords), size=n, replace=False)


def _rescale_weights(sampled: np.ndarray, total_mass: float) -> np.ndarray:
    sampled_sum = float(sampled.sum())
    if sampled_sum <= 0:
        return sampled
    return sampled * (total_mass / sampled_sum)


def build_support(
    density: np.ndarray,
    max_points: int,
    sampling_mode: SamplingMode,
    sampling_strategy: str,
    random_seed: int,
) -> Tuple[UOTSupport, dict]:
    mask = density > 0
    coords = np.column_stack(np.where(mask))
    weights = density[mask].astype(np.float64)
    total_mass = float(weights.sum())

    if coords.size == 0 or total_mass <= 0:
        raise ValueError("Empty mask or zero mass after density transform.")

    meta = {
        "support_points": int(len(coords)),
        "support_sampled": int(len(coords)),
        "sampling_strategy": "none",
    }

    if len(coords) <= max_points:
        return UOTSupport(coords_yx=coords.astype(np.float32), weights=weights.astype(np.float32)), meta

    if sampling_mode == SamplingMode.RAISE:
        raise RuntimeError(
            f"Support points {len(coords)} exceed max_support_points={max_points}"
        )

    rng = np.random.default_rng(random_seed)

    if sampling_strategy == "stratified_boundary_interior":
        boundary = mask & ~ndimage.binary_erosion(mask)
        interior = mask & ~boundary

        boundary_coords = np.column_stack(np.where(boundary))
        interior_coords = np.column_stack(np.where(interior))

        boundary_weights = density[boundary].astype(np.float64)
        interior_weights = density[interior].astype(np.float64)

        boundary_mass = float(boundary_weights.sum())
        interior_mass = float(interior_weights.sum())
        total_mass = boundary_mass + interior_mass

        if total_mass > 0:
            boundary_frac = boundary_mass / total_mass
        else:
            boundary_frac = 0.5

        n_boundary = min(len(boundary_coords), max(1, int(round(max_points * boundary_frac))))
        n_interior = max_points - n_boundary
        n_interior = min(len(interior_coords), n_interior)

        boundary_idx = _sample_indices(boundary_coords, boundary_weights, n_boundary, rng)
        interior_idx = _sample_indices(interior_coords, interior_weights, n_interior, rng)

        sampled_coords = np.vstack([
            boundary_coords[boundary_idx],
            interior_coords[interior_idx],
        ])
        sampled_weights = np.concatenate([
            _rescale_weights(boundary_weights[boundary_idx], boundary_mass),
            _rescale_weights(interior_weights[interior_idx], interior_mass),
        ])

        meta.update(
            {
                "support_sampled": int(len(sampled_coords)),
                "sampling_strategy": sampling_strategy,
                "boundary_points": int(len(boundary_coords)),
                "interior_points": int(len(interior_coords)),
            }
        )
        return UOTSupport(
            coords_yx=sampled_coords.astype(np.float32),
            weights=sampled_weights.astype(np.float32),
        ), meta

    # Fallback uniform sampling
    idx = _sample_indices(coords, weights, max_points, rng)
    sampled_weights = _rescale_weights(weights[idx], total_mass)
    meta.update(
        {
            "support_sampled": int(len(idx)),
            "sampling_strategy": "uniform",
        }
    )
    return UOTSupport(
        coords_yx=coords[idx].astype(np.float32),
        weights=sampled_weights.astype(np.float32),
    ), meta
