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


def repulsion(
    positions: np.ndarray,
    mask: np.ndarray,
    epsilon_r: float,
    eta: float,
    r_cut: float = 0.0,
) -> tuple[float, np.ndarray]:
    """Repulsion energy and gradient.

    Two modes controlled by r_cut:

    Classic soft-core (r_cut == 0):
        E = ε_r / (r² + η)
        Decays slowly — keeps pushing at moderate distances and
        sets the equilibrium bundle width.

    Truncated bump (r_cut > 0):
        E = ε_r * max(0, 1 - r²/r_cut²)²
        Exactly zero for r >= r_cut, smooth (C¹) at the cutoff.
        Acts only as a short-range exclusion force — prevents pathological
        collapse without setting bundle width.

        r_cut should be set SMALLER than the typical healthy within-bundle
        spacing, so that normally-spaced points are outside the repulsive
        zone. For example: r_cut = 0.25 * median_5nn_distance.
    """
    _, T, _ = positions.shape
    energy = 0.0
    grad = np.zeros_like(positions)

    for t in range(T):
        obs_idx = np.flatnonzero(mask[:, t])
        if len(obs_idx) < 2:
            continue

        pos_obs = positions[obs_idx, t, :]
        diff = pos_obs[:, None, :] - pos_obs[None, :, :]
        sq_dist = (diff ** 2).sum(axis=-1)
        valid_pairs = np.ones_like(sq_dist)
        np.fill_diagonal(valid_pairs, 0.0)

        if r_cut > 0.0:
            # Bump repulsion — only active inside exclusion zone r < r_cut
            r_cut_sq = r_cut ** 2
            inside = (sq_dist < r_cut_sq) & (valid_pairs > 0)
            u = np.where(inside, 1.0 - sq_dist / r_cut_sq, 0.0)  # in [0, 1]
            energy += epsilon_r * (u ** 2).sum()
            # d/d(x_i) E_ij = ε_r * 2u * (-2/r_cut²) * (x_i - x_j)
            coeff = np.where(inside, -4.0 * epsilon_r * u / r_cut_sq, 0.0)
            coeff_sym = coeff + coeff.T
            grad_obs = (coeff_sym[:, :, None] * diff).sum(axis=1)
        else:
            # Classic soft-core
            denom = sq_dist + eta
            energy += (epsilon_r / denom * valid_pairs).sum()
            coeff = -2.0 * epsilon_r / (denom ** 2) * valid_pairs
            coeff_sym = coeff + coeff.T
            grad_obs = (coeff_sym[:, :, None] * diff).sum(axis=1)

        grad[obs_idx, t, :] += grad_obs

    return energy, grad


def void_repulsion(
    positions: np.ndarray,
    mask: np.ndarray,
    epsilon_void: float,
    sigma_void: float,
) -> tuple[float, np.ndarray]:
    """Broad Gaussian density-field repulsion.

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


def elasticity(
    positions: np.ndarray,
    mask: np.ndarray,
    lambda_stretch: float,
    lambda_bend: float,
) -> tuple[float, np.ndarray]:
    """Stretch + bending elasticity energy and gradient."""
    _, T, _ = positions.shape
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
    """Fidelity anchor energy and gradient."""
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
    k_attract: int | None = None,
    subtract_mean_attraction: bool = False,
    sigma_attract_local: float | None = None,
    epsilon_void: float = 0.0,
    sigma_void: float | None = None,
    r_cut: float = 0.0,
) -> tuple[dict[str, float], np.ndarray]:
    """Compute all energy terms and combined gradient."""
    e_att, g_att = attraction(
        positions,
        mask,
        coherence,
        sigma,
        sigma_attract_local=sigma_attract_local,
        k_attract=k_attract,
        subtract_mean=subtract_mean_attraction,
    )
    e_rep, g_rep = repulsion(positions, mask, epsilon_r, eta, r_cut=r_cut)
    e_void, g_void = void_repulsion(
        positions, mask, epsilon_void, sigma_void if sigma_void is not None else sigma
    )
    e_ela, g_ela = elasticity(positions, mask, lambda_stretch, lambda_bend)
    e_fid, g_fid = fidelity(positions, x0, mask, mu)

    energies = {
        "attract": e_att,
        "repel": e_rep,
        "void": e_void,
        "elastic": e_ela,
        "fidelity": e_fid,
        "total": e_att + e_rep + e_void + e_ela + e_fid,
    }
    grad = g_att + g_rep + g_void + g_ela + g_fid
    return energies, grad
