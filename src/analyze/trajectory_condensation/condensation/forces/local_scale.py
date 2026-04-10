"""
local_scale.py
--------------
Local neighborhood scale preservation energy and gradient.

Penalizes deviation of each point's mean distance to its fixed initial
k-nearest neighbors from the initial value r_i^(0). Anchored to x0 to
prevent the reference from drifting during optimization.

    E = lambda_scale * sum_i (r_i^(n) - r_i^(0))^2
"""
from __future__ import annotations

import numpy as np

def local_scale_preservation(
    positions: np.ndarray,
    mask: np.ndarray,
    local_scale_refs: dict,
    lambda_scale: float,
) -> tuple[float, np.ndarray]:
    """Local neighborhood scale preservation energy and gradient.

    Penalizes deviation of each point's mean distance to its FIXED initial
    k-nearest neighbors from the initial value r_i^(0):

        E = lambda_scale * sum_i (r_i^(n) - r_i^(0))^2

    Properties:
    - Anchored to x0: the target r_i^(0) never changes during optimization
    - Scale-invariant: r_i^(0) is in the same units as positions
    - Handles both over-expansion (r > r0) and collapse (r < r0)
    - Does NOT prevent whole-bundle translation or rotation
    - Does NOT fix inter-bundle distances (only local radii)

    Disabled when lambda_scale == 0 (returns zeros immediately).
    """
    if lambda_scale == 0.0:
        return 0.0, np.zeros_like(positions)

    _, T, _ = positions.shape
    energy = 0.0
    grad = np.zeros_like(positions)

    for t in range(T):
        slice_refs = local_scale_refs.get(t)
        if slice_refs is None:
            continue
        obs_idx = slice_refs.obs_idx
        nbr_idx = slice_refs.neighbor_idx   # (n_obs, k_eff)
        r0 = slice_refs.r0                  # (n_obs,) initial mean radius
        k_eff = slice_refs.k_eff
        n_obs = len(obs_idx)

        pos = positions[obs_idx, t, :]    # (n_obs, 2)

        # Current distances to the fixed initial neighbors
        # nbr_pos[i, j] = position of neighbor j of point i
        nbr_pos = pos[nbr_idx, :]         # (n_obs, k_eff, 2)
        delta = pos[:, None, :] - nbr_pos  # (n_obs, k_eff, 2)  x_i - x_j
        dist = np.sqrt((delta ** 2).sum(axis=-1) + 1e-16)  # (n_obs, k_eff)
        r_curr = dist.mean(axis=1)        # (n_obs,) current mean radius

        # Energy: sum_i (r_curr_i - r0_i)^2
        residual = r_curr - r0            # (n_obs,)
        energy += lambda_scale * float((residual ** 2).sum())

        # Gradient w.r.t. pos[i]:
        #   dE/d(x_i) = 2 * lambda_scale * residual_i * d(r_curr_i)/d(x_i)
        #   d(r_curr_i)/d(x_i) = (1/k_eff) * sum_j (x_i - x_j) / dist_ij
        unit = delta / dist[:, :, None]   # (n_obs, k_eff, 2) unit vectors i→j
        dr_dxi = unit.mean(axis=1)        # (n_obs, 2)
        grad_i = 2.0 * lambda_scale * residual[:, None] * dr_dxi  # (n_obs, 2)

        # Gradient w.r.t. pos[j] (j is a neighbor of i):
        #   d(r_curr_i)/d(x_j) = -(1/k_eff) * (x_i - x_j) / dist_ij
        # Accumulate into each neighbor
        grad_j_contrib = -2.0 * lambda_scale * (residual[:, None, None] / k_eff) * unit
        # (n_obs, k_eff, 2) — contribution to each neighbor from each point

        grad_obs = grad_i.copy()
        for k in range(k_eff):
            np.add.at(grad_obs, nbr_idx[:, k], grad_j_contrib[:, k, :])

        grad[obs_idx, t, :] += grad_obs

    return energy, grad
