"""
forces.py
---------
Energy terms and their gradients.

Each function returns (energy_scalar, gradient) where gradient has the
same shape as positions: (N_e, T, 2).

All functions are pure — no side effects, no state.
"""
from __future__ import annotations

import numpy as np

from .coherence import gaussian_kernel


def attraction(
    positions: np.ndarray,
    mask: np.ndarray,
    coherence: np.ndarray,
    sigma: float,
) -> tuple[float, np.ndarray]:
    """Persistence-gated attraction energy and gradient.

    E_attract = -sum_{t} sum_{i!=j} K_s(x_i(t), x_j(t)) * C_ij(t)

    C_ij(t) is treated as frozen (alternating update).
    """
    N_e, T, _ = positions.shape
    energy = 0.0
    grad = np.zeros_like(positions)

    for t in range(T):
        obs_idx = np.flatnonzero(mask[:, t])
        if len(obs_idx) < 2:
            continue

        pos_obs = positions[obs_idx, t, :]
        K = gaussian_kernel(pos_obs, sigma)
        C = coherence[np.ix_(obs_idx, obs_idx, [t])][:, :, 0]
        W = K * C
        np.fill_diagonal(W, 0.0)

        energy += -W.sum()

        # Full gradient accounts for both (i,j) and (j,i) pair contributions.
        # dE/dx_i = sum_j [W[i,j] + W[j,i]] * (x_i - x_j) / sigma^2
        diff = pos_obs[:, None, :] - pos_obs[None, :, :]   # (n_obs, n_obs, 2)
        W_sym = W + W.T
        grad_obs = (W_sym[:, :, None] * diff).sum(axis=1) / (sigma ** 2)
        grad[obs_idx, t, :] += grad_obs

    return energy, grad


def repulsion(
    positions: np.ndarray,
    mask: np.ndarray,
    epsilon_r: float,
    eta: float,
) -> tuple[float, np.ndarray]:
    """Soft-core repulsion energy and gradient.

    E_repel = sum_{t} sum_{i!=j} epsilon_r / (||x_i - x_j||^2 + eta)
    """
    N_e, T, _ = positions.shape
    energy = 0.0
    grad = np.zeros_like(positions)

    for t in range(T):
        obs_idx = np.flatnonzero(mask[:, t])
        if len(obs_idx) < 2:
            continue

        pos_obs = positions[obs_idx, t, :]
        diff = pos_obs[:, None, :] - pos_obs[None, :, :]
        sq_dist = (diff ** 2).sum(axis=-1)
        denom = sq_dist + eta
        valid_pairs = np.ones_like(denom)
        np.fill_diagonal(valid_pairs, 0.0)

        energy += (epsilon_r / denom * valid_pairs).sum()

        # Full gradient: both (i,j) and (j,i) pair contributions.
        # dE/dx_i = sum_j [coeff[i,j] + coeff[j,i]] * (x_i - x_j)
        coeff = -2.0 * epsilon_r / (denom ** 2) * valid_pairs
        coeff_sym = coeff + coeff.T
        grad_obs = (coeff_sym[:, :, None] * diff).sum(axis=1)
        grad[obs_idx, t, :] += grad_obs

    return energy, grad


def elasticity(
    positions: np.ndarray,
    mask: np.ndarray,
    lambda_stretch: float,
    lambda_bend: float,
) -> tuple[float, np.ndarray]:
    """Stretch + bending elasticity energy and gradient.

    E_stretch = lambda_stretch * sum_i sum_t m_i(t)*m_i(t+1) ||x_i(t+1)-x_i(t)||^2
    E_bend    = lambda_bend   * sum_i sum_t m_i(t-1)*m_i(t)*m_i(t+1) ||x_i(t+1)-2x_i(t)+x_i(t-1)||^2
    """
    N_e, T, _ = positions.shape
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


def fidelity(
    positions: np.ndarray,
    x0: np.ndarray,
    mask: np.ndarray,
    mu: float,
) -> tuple[float, np.ndarray]:
    """Fidelity anchor energy and gradient.

    E_fidelity = mu * sum_i sum_t m_i(t) ||x_i(t) - x0_i(t)||^2
    """
    grad = np.zeros_like(positions)
    residual = positions[mask] - x0[mask]
    energy = float(mu * np.sum(residual ** 2))
    grad[mask] = 2.0 * mu * residual
    return energy, grad


def total_energy_and_grad(
    positions: np.ndarray,
    x0: np.ndarray,
    mask: np.ndarray,
    coherence: np.ndarray,
    sigma: float,
    epsilon_r: float,
    eta: float,
    lambda_stretch: float,
    lambda_bend: float,
    mu: float,
) -> tuple[dict[str, float], np.ndarray]:
    """Compute all energy terms and combined gradient."""
    e_att, g_att = attraction(positions, mask, coherence, sigma)
    e_rep, g_rep = repulsion(positions, mask, epsilon_r, eta)
    e_ela, g_ela = elasticity(positions, mask, lambda_stretch, lambda_bend)
    e_fid, g_fid = fidelity(positions, x0, mask, mu)

    energies = {
        "attract": e_att,
        "repel": e_rep,
        "elastic": e_ela,
        "fidelity": e_fid,
        "total": e_att + e_rep + e_ela + e_fid,
    }
    grad = g_att + g_rep + g_ela + g_fid
    return energies, grad
