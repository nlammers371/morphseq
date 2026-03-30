"""
coherence.py
------------
Spatial kernel and temporal coherence C_ij(t).

C_ij(t) is the memory of the system: it gates attraction by how persistently
two embryos have co-traveled over a causal backward window.

This module is stateless — all functions are pure (arrays in, arrays out).
"""
from __future__ import annotations

import numpy as np


def gaussian_kernel(
    positions: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Pairwise Gaussian spatial kernel at a single time slice.

    Parameters
    ----------
    positions : (N_e, 2)
    sigma : float

    Returns
    -------
    K : (N_e, N_e) — K[i,j] = exp(-||x_i - x_j||^2 / (2 sigma^2))
    """
    diff = positions[:, None, :] - positions[None, :, :]  # (N_e, N_e, 2)
    sq_dist = (diff ** 2).sum(axis=-1)                    # (N_e, N_e)
    return np.exp(-sq_dist / (2.0 * sigma ** 2))


def compute_coherence(
    positions: np.ndarray,
    mask: np.ndarray,
    sigma: float,
    delta: int,
    eps_mask: float = 1e-8,
) -> np.ndarray:
    """Compute temporal coherence C_ij(t) for all pairs and all times.

    C_ij(t) = mean over tau in [t-delta, t] of K_s(x_i(tau), x_j(tau)),
              weighted by joint observation mask m_i(tau) * m_j(tau).

    Causal: only past and present influence C_ij(t).

    Parameters
    ----------
    positions : (N_e, T, 2)
    mask : (N_e, T) bool
    sigma : float
        Gaussian kernel bandwidth.
    delta : int
        Backward window size in time bins.
    eps_mask : float
        Denominator stabilizer.

    Returns
    -------
    C : (N_e, N_e, T)
    """
    N_e, T, _ = positions.shape
    C = np.zeros((N_e, N_e, T), dtype=float)

    for t in range(T):
        t_start = max(0, t - delta)
        K_sum = np.zeros((N_e, N_e), dtype=float)
        w_sum = np.zeros((N_e, N_e), dtype=float)

        for tau in range(t_start, t + 1):
            K_tau = gaussian_kernel(positions[:, tau, :], sigma)   # (N_e, N_e)
            joint_mask = mask[:, tau, None] * mask[None, :, tau]   # (N_e, N_e) float
            K_sum += K_tau * joint_mask
            w_sum += joint_mask

        C[:, :, t] = K_sum / (w_sum + eps_mask)

    return C
