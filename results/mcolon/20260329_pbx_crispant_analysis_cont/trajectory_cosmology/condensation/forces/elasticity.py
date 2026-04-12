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
        sq = (delta ** 2).sum(axis=-1)
        energy += lambda_stretch * sq.sum()
        g = 2.0 * lambda_stretch * delta
        valid_idx = np.flatnonzero(valid)
        grad[valid_idx, t + 1, :] += g
        grad[valid_idx, t, :] -= g

    for t in range(1, T - 1):
        valid = mask[:, t - 1] & mask[:, t] & mask[:, t + 1]
        if not np.any(valid):
            continue
        acc = positions[valid, t + 1, :] - 2.0 * positions[valid, t, :] + positions[valid, t - 1, :]
        sq = (acc ** 2).sum(axis=-1)
        energy += lambda_bend * sq.sum()
        g = 2.0 * lambda_bend * acc
        valid_idx = np.flatnonzero(valid)
        grad[valid_idx, t + 1, :] += g
        grad[valid_idx, t, :] += -2.0 * g
        grad[valid_idx, t - 1, :] += g

    return energy, grad
