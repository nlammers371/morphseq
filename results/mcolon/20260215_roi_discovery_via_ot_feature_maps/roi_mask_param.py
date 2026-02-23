"""Mask parameterization utilities for Phase 2.5a fixed-model mask learning."""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from jax.image import resize as jax_resize


def mask_from_param(mask_param: jnp.ndarray, temperature: float = 1.0) -> jnp.ndarray:
    """Convert unconstrained mask parameters to soft mask in (0,1)."""
    return jax.nn.sigmoid(mask_param / jnp.float32(temperature))


def upsample_mask(mask_low: jnp.ndarray, output_hw: Tuple[int, int]) -> jnp.ndarray:
    """Bilinear upsample low-res mask to full resolution."""
    return jax_resize(mask_low[:, :, None], (*output_hw, 1), method="bilinear")[:, :, 0]


def tv_loss(mask_low: jnp.ndarray) -> jnp.ndarray:
    """Anisotropic TV penalty on a 2D soft mask."""
    dy = jnp.abs(mask_low[1:, :] - mask_low[:-1, :]).sum()
    dx = jnp.abs(mask_low[:, 1:] - mask_low[:, :-1]).sum()
    return dx + dy


def jitter_mask(mask_full: jnp.ndarray, shift_y: int, shift_x: int) -> jnp.ndarray:
    """Circularly shift mask as a simple anti-cheating jitter mechanism."""
    return jnp.roll(jnp.roll(mask_full, shift_y, axis=0), shift_x, axis=1)
