"""
void.py
-------
Space-usage / occupancy force terms.

Currently contains the pairwise void proxy: a broad Gaussian density-field
repulsion that pushes bundles apart globally. This is NOT a grid-based
occupancy void — it is a pairwise Gaussian kernel acting at sigma_void >> sigma_local.

Future home for:
  - Grid-based occupancy void (true space-filling pressure)
  - Other terms that equalize how the population uses the embedding space
"""
from __future__ import annotations

import numpy as np


def void_repulsion(
    positions: np.ndarray,
    mask: np.ndarray,
    epsilon_void: float,
    sigma_void: float,
) -> tuple[float, np.ndarray]:
    """Broad Gaussian density-field repulsion (pairwise void proxy).

    Each point is pushed away from regions of high point density using a
    Gaussian kernel with bandwidth sigma_void >> sigma_attract_local.
    Energy = epsilon_void * sum_i rho_i, where rho_i is the local density.
    This separates bundles globally without affecting within-bundle compactness
    (because sigma_void is large, the gradient is smooth and low-frequency).

    Disabled when epsilon_void == 0 (returns zeros immediately).
    """
    if epsilon_void == 0.0:
        return 0.0, np.zeros_like(positions)

    _, T, _ = positions.shape
    energy = 0.0
    grad = np.zeros_like(positions)

    for t in range(T):
        obs_idx = np.flatnonzero(mask[:, t])
        n_obs = len(obs_idx)
        if n_obs < 2:
            continue

        pos_obs = positions[obs_idx, t, :]
        diff = pos_obs[:, None, :] - pos_obs[None, :, :]  # (n, n, 2)
        sq_dist = (diff ** 2).sum(axis=-1)                # (n, n)
        K_void = np.exp(-sq_dist / (2.0 * sigma_void ** 2))
        np.fill_diagonal(K_void, 0.0)

        # density at each point = sum over other points' contributions
        energy += epsilon_void * K_void.sum()

        # gradient: push toward low-density (away from density peaks)
        # d(rho_i)/d(x_i) = sum_j K_void_ij * (x_i - x_j) / sigma_void^2
        K_sym = K_void + K_void.T
        grad_obs = epsilon_void * (K_sym[:, :, None] * diff).sum(axis=1) / (sigma_void ** 2)
        grad[obs_idx, t, :] += grad_obs

    return energy, grad
