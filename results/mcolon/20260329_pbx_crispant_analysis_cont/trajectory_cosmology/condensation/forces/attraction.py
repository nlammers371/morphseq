"""
attraction.py
-------------
Coherence-weighted pairwise attraction energy and gradient.

Attraction gates spatial pulling by how persistently two embryos have
co-traveled (coherence), optionally restricted to a k-NN graph per slice.
"""
from __future__ import annotations

import numpy as np

from ..coherence.kernels import gaussian_kernel


def _knn_mask_from_sq_dist(sq_dist: np.ndarray, k_attract: int | None) -> np.ndarray | float:
    """Return a symmetric kNN mask for a single observed slice.

    Parameters
    ----------
    sq_dist : (n_obs, n_obs)
        Pairwise squared distances among observed embryos.
    k_attract : int | None
        Number of nearest neighbors per embryo. None = all pairs.
    """
    n_obs = sq_dist.shape[0]
    if k_attract is None or n_obs <= 1:
        return 1.0
    k_eff = min(k_attract, n_obs - 1)
    if k_eff <= 0 or k_eff >= n_obs - 1:
        return 1.0

    sq_for_knn = sq_dist.copy()
    np.fill_diagonal(sq_for_knn, np.inf)
    knn_idx = np.argpartition(sq_for_knn, kth=k_eff - 1, axis=1)[:, :k_eff]
    knn_mask = np.zeros((n_obs, n_obs), dtype=float)
    knn_mask[np.arange(n_obs)[:, None], knn_idx] = 1.0
    knn_mask = np.maximum(knn_mask, knn_mask.T)
    np.fill_diagonal(knn_mask, 0.0)
    return knn_mask


def attraction(
    positions: np.ndarray,
    mask: np.ndarray,
    coherence: np.ndarray,
    sigma: float,
    sigma_attract_local: float | None = None,
    k_attract: int | None = None,
    subtract_mean: bool = False,
) -> tuple[float, np.ndarray]:
    """Persistence-gated attraction energy and gradient.

    Attraction can optionally be restricted to a symmetric kNN graph
    within each time slice.

    If sigma_attract_local is provided, it is used as the Gaussian bandwidth
    instead of sigma. This allows the attraction to operate at the within-bundle
    scale rather than the inter-bundle scale, giving compact bundles a genuine
    restoring force. sigma is still used for coherence computation externally.
    """
    sigma_att = sigma_attract_local if sigma_attract_local is not None else sigma
    _, T, _ = positions.shape
    energy = 0.0
    grad = np.zeros_like(positions)

    for t in range(T):
        obs_idx = np.flatnonzero(mask[:, t])
        n_obs = len(obs_idx)
        if n_obs < 2:
            continue

        pos_obs = positions[obs_idx, t, :]
        diff = pos_obs[:, None, :] - pos_obs[None, :, :]
        sq_dist = (diff ** 2).sum(axis=-1)
        K = np.exp(-sq_dist / (2.0 * sigma_att ** 2))
        C = coherence[np.ix_(obs_idx, obs_idx, [t])][:, :, 0]
        knn_mask = _knn_mask_from_sq_dist(sq_dist, k_attract)
        W = K * C * knn_mask
        np.fill_diagonal(W, 0.0)

        energy += -W.sum()

        W_sym = W + W.T
        grad_obs = (W_sym[:, :, None] * diff).sum(axis=1) / (sigma_att ** 2)
        if subtract_mean:
            grad_obs = grad_obs - grad_obs.mean(axis=0, keepdims=True)
        grad[obs_idx, t, :] += grad_obs

    return energy, grad
