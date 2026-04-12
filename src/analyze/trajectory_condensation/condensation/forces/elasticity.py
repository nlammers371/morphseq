"""
elasticity.py
-------------
Temporal trajectory regularizers: stretch and bending penalties.

These are not spatial interaction forces — they penalize properties of each
embryo's trajectory over time. Stretch penalizes large per-step displacements;
bend penalizes large curvature (second differences).

Both are weighted by their respective dimensionless multipliers, calibrated
against s_step and s_bend from geometry_refs.
"""
from __future__ import annotations

import numpy as np


def elasticity(
    positions: np.ndarray,
    mask: np.ndarray,
    lambda_stretch: float,
    lambda_bend: float,
    elasticity_kernel: str = "quadratic",
    s_step_ref: float = 1.0,
    s_bend_ref: float = 1.0,
) -> tuple[float, np.ndarray]:
    """Stretch + bending elasticity energy and gradient."""
    _, T, _ = positions.shape
    energy = 0.0
    grad = np.zeros_like(positions)

    for t in range(T - 1):
        valid = mask[:, t] & mask[:, t + 1]
        if not np.any(valid):
            continue
        delta = positions[valid, t + 1, :] - positions[valid, t, :]
        if elasticity_kernel == "quadratic":
            sq = (delta ** 2).sum(axis=-1)
            energy += lambda_stretch * sq.sum()
            g = 2.0 * lambda_stretch * delta
        elif elasticity_kernel == "ratio_hinge":
            norms = np.linalg.norm(delta, axis=-1)
            residual = np.maximum(norms / max(s_step_ref, 1e-12) - 1.0, 0.0)
            energy += lambda_stretch * np.square(residual).sum()
            coeff = np.zeros_like(norms)
            active = norms > 1e-12
            coeff[active] = (
                2.0 * lambda_stretch * residual[active] / max(s_step_ref, 1e-12) / norms[active]
            )
            g = coeff[:, None] * delta
        else:
            raise ValueError(f"Unsupported elasticity_kernel={elasticity_kernel!r}")
        valid_idx = np.flatnonzero(valid)
        grad[valid_idx, t + 1, :] += g
        grad[valid_idx, t, :] -= g

    for t in range(1, T - 1):
        valid = mask[:, t - 1] & mask[:, t] & mask[:, t + 1]
        if not np.any(valid):
            continue
        acc = positions[valid, t + 1, :] - 2.0 * positions[valid, t, :] + positions[valid, t - 1, :]
        if elasticity_kernel == "quadratic":
            sq = (acc ** 2).sum(axis=-1)
            energy += lambda_bend * sq.sum()
            g = 2.0 * lambda_bend * acc
        elif elasticity_kernel == "ratio_hinge":
            norms = np.linalg.norm(acc, axis=-1)
            residual = np.maximum(norms / max(s_bend_ref, 1e-12) - 1.0, 0.0)
            energy += lambda_bend * np.square(residual).sum()
            coeff = np.zeros_like(norms)
            active = norms > 1e-12
            coeff[active] = (
                2.0 * lambda_bend * residual[active] / max(s_bend_ref, 1e-12) / norms[active]
            )
            g = coeff[:, None] * acc
        else:
            raise ValueError(f"Unsupported elasticity_kernel={elasticity_kernel!r}")
        valid_idx = np.flatnonzero(valid)
        grad[valid_idx, t + 1, :] += g
        grad[valid_idx, t, :] += -2.0 * g
        grad[valid_idx, t - 1, :] += g

    return energy, grad
