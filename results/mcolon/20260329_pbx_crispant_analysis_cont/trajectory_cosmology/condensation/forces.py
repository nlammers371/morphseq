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

    Parameters
    ----------
    positions : (N_e, T, 2)
    mask : (N_e, T) bool
    coherence : (N_e, N_e, T) — frozen C_ij(t) from coherence.py
    sigma : float

    Returns
    -------
    (energy, grad) where grad : (N_e, T, 2)
    """
    N_e, T, _ = positions.shape
    energy = 0.0
    grad = np.zeros_like(positions)

    for t in range(T):
        obs = mask[:, t]
        if obs.sum() < 2:
            continue
        K = gaussian_kernel(positions[:, t, :], sigma)   # (N_e, N_e)
        C = coherence[:, :, t]                           # (N_e, N_e)
        W = K * C                                        # (N_e, N_e)

        # zero out pairs where either embryo is unobserved
        joint = obs[:, None] * obs[None, :]
        W *= joint
        np.fill_diagonal(W, 0.0)

        energy += -W.sum()

        # gradient: dE/dx_i(t) = sum_j W_ij * (x_i - x_j) / sigma^2
        diff = positions[:, t, None, :] - positions[None, :, t, :]  # (N_e, N_e, 2)
        grad[:, t, :] += (W[:, :, None] * diff).sum(axis=1) / (sigma ** 2)

    return energy, grad


def repulsion(
    positions: np.ndarray,
    mask: np.ndarray,
    epsilon_r: float,
    eta: float,
) -> tuple[float, np.ndarray]:
    """Soft-core repulsion energy and gradient.

    E_repel = sum_{t} sum_{i!=j} epsilon_r / (||x_i - x_j||^2 + eta)

    Parameters
    ----------
    positions : (N_e, T, 2)
    mask : (N_e, T) bool
    epsilon_r : float
    eta : float

    Returns
    -------
    (energy, grad) where grad : (N_e, T, 2)
    """
    N_e, T, _ = positions.shape
    energy = 0.0
    grad = np.zeros_like(positions)

    for t in range(T):
        obs = mask[:, t]
        if obs.sum() < 2:
            continue
        diff = positions[:, t, None, :] - positions[None, :, t, :]  # (N_e, N_e, 2)
        sq_dist = (diff ** 2).sum(axis=-1)                          # (N_e, N_e)
        denom = sq_dist + eta

        joint = obs[:, None] * obs[None, :]
        np.fill_diagonal(joint, 0.0)

        energy += (epsilon_r / denom * joint).sum()

        # gradient: dE/dx_i = sum_j -2 epsilon_r * (x_i - x_j) / denom^2
        coeff = -2.0 * epsilon_r / (denom ** 2) * joint  # (N_e, N_e)
        grad[:, t, :] += (coeff[:, :, None] * diff).sum(axis=1)

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

    Parameters
    ----------
    positions : (N_e, T, 2)
    mask : (N_e, T) bool

    Returns
    -------
    (energy, grad) where grad : (N_e, T, 2)
    """
    N_e, T, _ = positions.shape
    energy = 0.0
    grad = np.zeros_like(positions)

    # stretch
    for t in range(T - 1):
        m = (mask[:, t] * mask[:, t + 1]).astype(float)   # (N_e,)
        delta = positions[:, t + 1, :] - positions[:, t, :]
        sq = (delta ** 2).sum(axis=-1)                     # (N_e,)
        energy += lambda_stretch * (m * sq).sum()
        g = 2.0 * lambda_stretch * m[:, None] * delta
        grad[:, t + 1, :] += g
        grad[:, t, :]     -= g

    # bend
    for t in range(1, T - 1):
        m = (mask[:, t - 1] * mask[:, t] * mask[:, t + 1]).astype(float)
        acc = positions[:, t + 1, :] - 2.0 * positions[:, t, :] + positions[:, t - 1, :]
        sq = (acc ** 2).sum(axis=-1)
        energy += lambda_bend * (m * sq).sum()
        g = 2.0 * lambda_bend * m[:, None] * acc
        grad[:, t + 1, :] +=  g
        grad[:, t, :]     += -2.0 * g
        grad[:, t - 1, :] +=  g

    return energy, grad


def fidelity(
    positions: np.ndarray,
    x0: np.ndarray,
    mask: np.ndarray,
    mu: float,
) -> tuple[float, np.ndarray]:
    """Fidelity anchor energy and gradient.

    E_fidelity = mu * sum_i sum_t m_i(t) ||x_i(t) - x0_i(t)||^2

    Parameters
    ----------
    positions : (N_e, T, 2)
    x0 : (N_e, T, 2) — initialization
    mask : (N_e, T) bool
    mu : float — current (annealed) weight

    Returns
    -------
    (energy, grad) where grad : (N_e, T, 2)
    """
    m = mask[:, :, None].astype(float)
    residual = positions - x0
    energy = mu * (m * residual ** 2).sum()
    grad = 2.0 * mu * m * residual
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
    """Compute all energy terms and combined gradient.

    Returns
    -------
    energies : dict with keys attract, repel, stretch, bend, fidelity, total
    grad : (N_e, T, 2)
    """
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
