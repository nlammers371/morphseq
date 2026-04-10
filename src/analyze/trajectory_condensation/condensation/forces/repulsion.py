"""
repulsion.py
------------
Pairwise repulsion energy and gradient.

Two modes:
  - Classic soft-core (r_cut == 0): E = ε_r / (r⁴ + η)
    Decays much faster at moderate distances, so repulsion stays a
    local exclusion force instead of acting like a dense global wind.
  - Truncated bump (r_cut > 0): E = ε_r * max(0, 1 - r²/r_cut²)²
    Exactly zero for r >= r_cut, smooth (C¹) at the cutoff.
    Short-range exclusion only — prevents collapse without setting bundle width.
"""
from __future__ import annotations

import numpy as np


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
        E = ε_r / (r⁴ + η)
        Decays much faster — preserves local exclusion without letting
        dense trunks create a long-range repulsive wind.

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
            # Quartic soft-core: short-range exclusion with much faster tail decay.
            denom = sq_dist ** 2 + eta
            energy += (epsilon_r / denom * valid_pairs).sum()
            coeff = -4.0 * epsilon_r * sq_dist / (denom ** 2) * valid_pairs
            coeff_sym = coeff + coeff.T
            grad_obs = (coeff_sym[:, :, None] * diff).sum(axis=1)

        grad[obs_idx, t, :] += grad_obs

    return energy, grad
