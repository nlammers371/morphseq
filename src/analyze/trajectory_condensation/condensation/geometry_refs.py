"""
geometry_refs.py
----------------
Measurement / calibration primitives for the condensation solver.

This module answers: given the initial geometry, what reference quantities
should the force terms be calibrated against?

This is a separate provenance chain from forces.py, which answers:
given positions and coefficients, what energy and gradient do I produce?

    measurement layer  →  geometry_refs.py
    mechanics layer    →  forces.py
    orchestration      →  dynamics.py / run_temporal()

All functions are pure — arrays in, scalar or dataclass out.
All estimates are computed from the INITIAL positions x0 (before any
optimization), so they capture the native geometry of the data, not a
moving target.

GeometryRefs
------------
The central output is a GeometryRefs dataclass that bundles all reference
scales in one place. Compute it once at the start of a run:

    refs = estimate_geometry_refs(x0, mask)

Then derive absolute force coefficients from refs + dimensionless multipliers:

    epsilon_r      = repulsion_strength_mult  * refs.s_local**2
    mu0            = fidelity_strength_mult   / refs.s_local**2
    lambda_stretch = stretch_strength_mult    / refs.s_step**2
    lambda_bend    = bend_strength_mult       / refs.s_bend**2
    sigma_coh      = coherence_scale_mult     * refs.s_local

Reference scales
----------------
s_local  : median k-NN distance across all observed (embryo, time) pairs.
           Calibration target for repulsion and fidelity.
           Force acts at point-to-point spacing scale.

s_step   : median per-step displacement ||x_i(t+1) - x_i(t)||.
           Calibration target for stretch penalty.
           Force acts at trajectory-step scale.

s_bend   : median second-difference ||x_i(t+1) - 2*x_i(t) + x_i(t-1)||.
           Calibration target for bend penalty.
           Force acts at trajectory-curvature scale.

s_global : mean per-slice radial spread (std of positions within each slice,
           averaged across slices).
           Calibration target for attraction bandwidth (sigma) and void term.
           Force acts at inter-bundle / whole-cloud scale.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GeometryRefs:
    """Reference scales derived from the initial geometry of a dataset.

    All values are in the same spatial units as the input positions.
    All values are computed from x0 (initial positions) before any
    optimization, so they are fixed calibration anchors.

    Attributes
    ----------
    s_local  : median k-NN distance — local point spacing scale
    s_step   : median per-step displacement — trajectory step scale
    s_bend   : median second-difference — trajectory curvature scale
    s_global : mean per-slice radial spread — cloud / inter-bundle scale
    """
    s_local:  float
    s_step:   float
    s_bend:   float
    s_global: float

    def __repr__(self) -> str:
        return (
            f"GeometryRefs("
            f"s_local={self.s_local:.4f}, "
            f"s_step={self.s_step:.4f}, "
            f"s_bend={self.s_bend:.4f}, "
            f"s_global={self.s_global:.4f})"
        )


@dataclass
class LocalScaleSliceRefs:
    """Fixed local-neighborhood anchors for one time slice.

    These are measured once from ``x0`` and then held fixed during
    optimization. They are geometry references, not force coefficients.
    """

    obs_idx: np.ndarray
    neighbor_idx: np.ndarray
    r0: np.ndarray
    k_eff: int


@dataclass(frozen=True)
class SliceOutlierRefs:
    """Fixed per-slice outlier anchors derived from ``x0``.

    These are measurement-layer references for the slice-relative outlier
    force:

    - ``centers[t]``: baseline slice centroid
    - ``scale[t]``: baseline slice cutoff distance
    """

    centers: np.ndarray
    scale: np.ndarray
    valid: np.ndarray
    cutoff_mode: str = "quantile"
    cutoff_value: float = 0.99

    @property
    def q99(self) -> np.ndarray:
        return self.scale


# ---------------------------------------------------------------------------
# Individual estimators
# ---------------------------------------------------------------------------

def estimate_local_spacing_ref(
    x0: np.ndarray,
    mask: np.ndarray,
    k: int = 5,
) -> float:
    """Median k-NN distance across all observed (embryo, time) pairs.

    This is s_local — the canonical calibration target for forces that act
    at within-bundle point-to-point spacing:

        epsilon_r = repulsion_strength_mult * s_local**2
        mu0       = fidelity_strength_mult  / s_local**2
        sigma_coh = coherence_scale_mult    * s_local

    Parameters
    ----------
    x0   : (N_e, T, 2) — initial positions
    mask : (N_e, T) bool
    k    : number of nearest neighbors (default 5)

    Returns
    -------
    s_local : float
    """
    N_e, T, _ = x0.shape
    all_knn_dists = []
    for t in range(T):
        obs_idx = np.flatnonzero(mask[:, t])
        n_obs = len(obs_idx)
        if n_obs < 2:
            continue
        pos = x0[obs_idx, t, :]
        diff = pos[:, None, :] - pos[None, :, :]
        sq_dist = (diff ** 2).sum(axis=-1)
        np.fill_diagonal(sq_dist, np.inf)
        k_eff = min(k, n_obs - 1)
        knn_sq = np.partition(sq_dist, kth=k_eff - 1, axis=1)[:, :k_eff]
        all_knn_dists.append(np.sqrt(knn_sq).ravel())
    if not all_knn_dists:
        return 1.0
    return float(np.median(np.concatenate(all_knn_dists)))


def build_local_scale_refs(
    x0: np.ndarray,
    mask: np.ndarray,
    k_local: int = 5,
) -> dict[int, LocalScaleSliceRefs | None]:
    """Precompute fixed local-neighborhood anchors from initial positions.

    Returns per-slice local geometry references used by the local-scale
    preservation term. This belongs in the measurement layer because it is
    derived once from ``x0`` and then reused unchanged by the solver.
    """
    _, t_count, _ = x0.shape
    refs: dict[int, LocalScaleSliceRefs | None] = {}
    for t in range(t_count):
        obs_idx = np.flatnonzero(mask[:, t])
        n_obs = len(obs_idx)
        if n_obs < 2:
            refs[t] = None
            continue
        pos = x0[obs_idx, t, :]
        diff = pos[:, None, :] - pos[None, :, :]
        sq_dist = (diff ** 2).sum(axis=-1)
        np.fill_diagonal(sq_dist, np.inf)
        k_eff = min(k_local, n_obs - 1)
        neighbor_idx = np.argpartition(sq_dist, kth=k_eff - 1, axis=1)[:, :k_eff]
        r0 = np.sqrt(sq_dist[np.arange(n_obs)[:, None], neighbor_idx]).mean(axis=1)
        refs[t] = LocalScaleSliceRefs(
            obs_idx=obs_idx,
            neighbor_idx=neighbor_idx,
            r0=r0,
            k_eff=k_eff,
        )
    return refs


def estimate_step_scale_ref(
    x0: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Median per-step displacement ||x_i(t+1) - x_i(t)|| across all valid transitions.

    This is s_step — the calibration target for the stretch penalty:

        lambda_stretch = stretch_strength_mult / s_step**2

    A step is valid when both x_i(t) and x_i(t+1) are observed (mask True).
    Returns 1.0 if no valid transitions exist (degenerate fallback).

    Parameters
    ----------
    x0   : (N_e, T, 2) — initial positions
    mask : (N_e, T) bool

    Returns
    -------
    s_step : float
    """
    N_e, T, _ = x0.shape
    dists = []
    for t in range(T - 1):
        valid = mask[:, t] & mask[:, t + 1]
        if valid.sum() == 0:
            continue
        step = x0[valid, t + 1, :] - x0[valid, t, :]   # (n_valid, 2)
        dists.append(np.linalg.norm(step, axis=1))
    if not dists:
        return 1.0
    return float(np.median(np.concatenate(dists)))


def estimate_bend_scale_ref(
    x0: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Median second-difference ||x_i(t+1) - 2*x_i(t) + x_i(t-1)|| across all valid triples.

    This is s_bend — the calibration target for the bend penalty:

        lambda_bend = bend_strength_mult / s_bend**2

    A triple is valid when x_i(t-1), x_i(t), x_i(t+1) are all observed.
    Returns 1.0 if no valid triples exist (degenerate fallback).

    Parameters
    ----------
    x0   : (N_e, T, 2) — initial positions
    mask : (N_e, T) bool

    Returns
    -------
    s_bend : float
    """
    N_e, T, _ = x0.shape
    dists = []
    for t in range(1, T - 1):
        valid = mask[:, t - 1] & mask[:, t] & mask[:, t + 1]
        if valid.sum() == 0:
            continue
        second_diff = (
            x0[valid, t + 1, :]
            - 2.0 * x0[valid, t, :]
            + x0[valid, t - 1, :]
        )   # (n_valid, 2)
        dists.append(np.linalg.norm(second_diff, axis=1))
    if not dists:
        return 1.0
    return float(np.median(np.concatenate(dists)))


def estimate_global_scale_ref(
    x0: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Mean per-slice radial spread (std of positions within each observed slice).

    This is s_global — the calibration target for inter-bundle forces:

        sigma            = sigma_frac * s_global   (attraction bandwidth)
        sigma_void       = sigma_void_frac * s_global
        sigma_attract_local is anchored to s_local, not s_global

    Returns 1.0 if no observed positions exist.

    Parameters
    ----------
    x0   : (N_e, T, 2) — initial positions
    mask : (N_e, T) bool

    Returns
    -------
    s_global : float
    """
    N_e, T, _ = x0.shape
    spreads = []
    for t in range(T):
        obs_idx = np.flatnonzero(mask[:, t])
        if len(obs_idx) < 2:
            continue
        pos = x0[obs_idx, t, :]
        centered = pos - pos.mean(axis=0)
        spreads.append(float(np.std(centered)))
    if not spreads:
        return 1.0
    return float(np.mean(spreads))


def build_slice_outlier_refs(
    x0: np.ndarray,
    mask: np.ndarray,
    *,
    cutoff_mode: str = "quantile",
    quantile: float = 0.99,
    robust_k: float = 3.0,
) -> SliceOutlierRefs:
    """Precompute fixed slice-relative outlier anchors from initial positions."""
    _, t_count, _ = x0.shape
    centers = np.zeros((t_count, 2), dtype=float)
    scale = np.ones(t_count, dtype=float)
    valid = np.zeros(t_count, dtype=bool)
    for t in range(t_count):
        obs = np.flatnonzero(mask[:, t])
        if len(obs) < 2:
            continue
        pos = x0[obs, t, :]
        center = pos.mean(axis=0)
        dists = np.linalg.norm(pos - center, axis=1)
        if cutoff_mode == "quantile":
            cutoff = float(np.percentile(dists, 100.0 * quantile))
        elif cutoff_mode == "robust":
            median = float(np.median(dists))
            mad = float(np.median(np.abs(dists - median)))
            cutoff = median + robust_k * 1.4826 * mad
        else:
            raise ValueError(f"Unsupported cutoff_mode={cutoff_mode!r}")
        centers[t] = center
        scale[t] = max(cutoff, 1e-6)
        valid[t] = True
    return SliceOutlierRefs(
        centers=centers,
        scale=scale,
        valid=valid,
        cutoff_mode=str(cutoff_mode),
        cutoff_value=float(quantile if cutoff_mode == "quantile" else robust_k),
    )


# ---------------------------------------------------------------------------
# Composite entry point
# ---------------------------------------------------------------------------

def estimate_geometry_refs(
    x0: np.ndarray,
    mask: np.ndarray,
    k_local: int = 5,
) -> GeometryRefs:
    """Compute all geometry reference scales from initial positions in one call.

    Parameters
    ----------
    x0      : (N_e, T, 2) — initial positions (before any optimization)
    mask    : (N_e, T) bool
    k_local : number of nearest neighbors for s_local estimation (default 5)

    Returns
    -------
    GeometryRefs with s_local, s_step, s_bend, s_global
    """
    return GeometryRefs(
        s_local=estimate_local_spacing_ref(x0, mask, k=k_local),
        s_step=estimate_step_scale_ref(x0, mask),
        s_bend=estimate_bend_scale_ref(x0, mask),
        s_global=estimate_global_scale_ref(x0, mask),
    )
