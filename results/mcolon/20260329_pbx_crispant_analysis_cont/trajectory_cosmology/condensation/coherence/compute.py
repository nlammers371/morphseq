"""
compute.py
----------
Temporal coherence C_ij(t).

C_ij(t) is the memory of the system: it gates attraction by how persistently
two embryos have co-traveled over a causal backward window.

This module is stateless — all functions are pure (arrays in, arrays out).
"""
from __future__ import annotations

import numpy as np

from .kernels import gaussian_kernel


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
            obs_idx = np.flatnonzero(mask[:, tau])
            if len(obs_idx) == 0:
                continue
            K_tau = gaussian_kernel(positions[obs_idx, tau, :], sigma)
            K_sum[np.ix_(obs_idx, obs_idx)] += K_tau
            w_sum[np.ix_(obs_idx, obs_idx)] += 1.0

        C[:, :, t] = K_sum / (w_sum + eps_mask)

    return C
