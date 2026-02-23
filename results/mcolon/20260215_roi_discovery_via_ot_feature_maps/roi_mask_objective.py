"""Differentiable perturbation objectives for Phase 2.5a."""

from __future__ import annotations

import jax.numpy as jnp


def apply_perturbation(X: jnp.ndarray, mask_full: jnp.ndarray, baseline: jnp.ndarray) -> jnp.ndarray:
    m = mask_full[None, :, :, None]
    return m * X + (1.0 - m) * baseline[None, :, :, :]


def compute_score(logits: jnp.ndarray) -> jnp.ndarray:
    return logits.mean()


def dual_objective(
    X: jnp.ndarray,
    mask_full: jnp.ndarray,
    baseline: jnp.ndarray,
    w_full: jnp.ndarray,
    b: float,
) -> jnp.ndarray:
    preserve = apply_perturbation(X, mask_full, baseline)
    delete = apply_perturbation(X, 1.0 - mask_full, baseline)

    z_preserve = jnp.sum(preserve * w_full[None, :, :, :], axis=(1, 2, 3)) + b
    z_delete = jnp.sum(delete * w_full[None, :, :, :], axis=(1, 2, 3)) + b
    return compute_score(z_preserve) - compute_score(z_delete)
