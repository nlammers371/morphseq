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

from ..geometry_refs import estimate_local_spacing_ref  # re-exported for convenience


def build_neighborhood_info(
    x0: np.ndarray,
    mask: np.ndarray,
    k_local: int = 5,
) -> dict:
    """Precompute fixed local neighborhood structure from initial positions.

    Returns a dict keyed by time slice t, each containing:
        neighbor_idx : (n_obs, k_eff) int — indices into obs_idx for each point's
                       k nearest initial neighbors (fixed for all iterations)
        r0           : (n_obs,) float — initial mean distance to those neighbors
        obs_idx      : (n_obs,) int — which global embryo indices are observed at t

    This is computed once before the optimization loop and passed unchanged
    to local_scale_preservation() every iteration. Anchoring to x0 prevents
    the reference scale from drifting with the pathology.
    """
    N_e, T, _ = x0.shape
    info = {}
    for t in range(T):
        obs_idx = np.flatnonzero(mask[:, t])
        n_obs = len(obs_idx)
        if n_obs < 2:
            info[t] = None
            continue
        pos = x0[obs_idx, t, :]            # (n_obs, 2)
        diff = pos[:, None, :] - pos[None, :, :]
        sq_dist = (diff ** 2).sum(axis=-1)
        np.fill_diagonal(sq_dist, np.inf)
        k_eff = min(k_local, n_obs - 1)
        # Indices of k nearest neighbors for each point (within obs_idx space)
        neighbor_idx = np.argpartition(sq_dist, kth=k_eff - 1, axis=1)[:, :k_eff]
        # Initial mean distance to those neighbors
        r0 = np.sqrt(sq_dist[np.arange(n_obs)[:, None], neighbor_idx]).mean(axis=1)
        info[t] = {
            "obs_idx": obs_idx,
            "neighbor_idx": neighbor_idx,   # (n_obs, k_eff)
            "r0": r0,                        # (n_obs,)
            "k_eff": k_eff,
        }
    return info


def local_scale_preservation(
    positions: np.ndarray,
    mask: np.ndarray,
    neighborhood_info: dict,
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
        ninfo = neighborhood_info.get(t)
        if ninfo is None:
            continue
        obs_idx = ninfo["obs_idx"]
        nbr_idx = ninfo["neighbor_idx"]   # (n_obs, k_eff)
        r0 = ninfo["r0"]                  # (n_obs,) initial mean radius
        k_eff = ninfo["k_eff"]
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
