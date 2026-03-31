"""
fidelity.py
-----------
Fidelity anchor energy and gradient.

Penalizes deviation from the initial positions x0, decaying over iterations
as mu = mu0 * gamma^n. This anchors the optimization early when coherence
information is sparse, then releases as coherence builds.
"""
from __future__ import annotations

import numpy as np


def fidelity(
    positions: np.ndarray,
    x0: np.ndarray,
    mask: np.ndarray,
    mu: float,
) -> tuple[float, np.ndarray]:
    """Fidelity anchor energy and gradient."""
    grad = np.zeros_like(positions)
    residual = positions[mask] - x0[mask]
    energy = float(mu * np.sum(residual ** 2))
    grad[mask] = 2.0 * mu * residual
    return energy, grad
