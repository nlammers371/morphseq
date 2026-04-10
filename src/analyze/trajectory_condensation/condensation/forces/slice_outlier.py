"""
slice_outlier.py
----------------
Slice-relative outlier penalty for embryos that detach far from the baseline
time-slice manifold even when their trajectories remain smooth.

References are computed once from x0:
  - center_t : baseline slice centroid
  - scale_t  : baseline slice cutoff distance

The force is smooth and activates only beyond the baseline cutoff scale:

    S = ||x_it - center_t|| / scale_t
    u = softplus(beta * (S - 1)) / beta
    E = outlier_strength * sum u^4

with fixed beta=4 for v1.
"""
from __future__ import annotations

import numpy as np
from ..geometry_refs import SliceOutlierRefs


def slice_outlier(
    positions: np.ndarray,
    mask: np.ndarray,
    refs: SliceOutlierRefs | None,
    outlier_strength: float,
    beta: float = 4.0,
) -> tuple[float, np.ndarray]:
    if refs is None or outlier_strength <= 0:
        return 0.0, np.zeros_like(positions)

    energy = 0.0
    grad = np.zeros_like(positions)
    _, t_count, _ = positions.shape

    for t in range(t_count):
        if not refs.valid[t]:
            continue
        obs = np.flatnonzero(mask[:, t])
        if len(obs) == 0:
            continue
        delta = positions[obs, t, :] - refs.centers[t]
        norms = np.linalg.norm(delta, axis=1)
        scale = refs.scale[t]
        severity = norms / scale
        logits = beta * (severity - 1.0)
        u = np.log1p(np.exp(logits)) / beta
        energy += outlier_strength * np.power(u, 4).sum()

        sig = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
        coeff = np.zeros_like(norms)
        active = norms > 1e-12
        coeff[active] = outlier_strength * 4.0 * np.power(u[active], 3) * sig[active] / scale / norms[active]
        grad[obs, t, :] += coeff[:, None] * delta

    return float(energy), grad
