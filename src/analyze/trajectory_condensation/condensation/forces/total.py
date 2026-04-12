"""
total.py
--------
Assembler: call all enabled force terms, collect energy dict, sum gradients.

This module is orchestration only — it delegates to individual force modules.
It does not contain force logic.
"""
from __future__ import annotations

import numpy as np

from .attraction import attraction
from .repulsion import repulsion
from .void import void_repulsion
from .elasticity import elasticity
from .fidelity import fidelity
from .local_scale import local_scale_preservation
from ..geometry_refs import SliceOutlierRefs
from .slice_outlier import slice_outlier


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
    elasticity_kernel: str = "quadratic",
    s_step_ref: float = 1.0,
    s_bend_ref: float = 1.0,
    k_attract: int | None = None,
    subtract_mean_attraction: bool = False,
    sigma_attract_local: float | None = None,
    epsilon_void: float = 0.0,
    sigma_void: float | None = None,
    outlier_strength: float = 0.0,
    outlier_refs: SliceOutlierRefs | None = None,
    r_cut: float = 0.0,
    lambda_scale: float = 0.0,
    neighborhood_info: dict | None = None,
    w_attract: float = 1.0,
    w_repel: float = 1.0,
    w_fidelity: float = 1.0,
    w_elastic: float = 1.0,
    w_void: float = 1.0,
    w_scale: float = 1.0,
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
    e_ela, g_ela = elasticity(
        positions,
        mask,
        lambda_stretch,
        lambda_bend,
        elasticity_kernel=elasticity_kernel,
        s_step_ref=s_step_ref,
        s_bend_ref=s_bend_ref,
    )
    e_fid, g_fid = fidelity(positions, x0, mask, mu)
    e_scale, g_scale = local_scale_preservation(
        positions, mask, neighborhood_info if neighborhood_info is not None else {}, lambda_scale
    )
    e_out, g_out = slice_outlier(positions, mask, outlier_refs, outlier_strength)

    energies = {
        "attract": w_attract * e_att,
        "repel": w_repel * e_rep,
        "void": w_void * e_void,
        "elastic": w_elastic * e_ela,
        "fidelity": w_fidelity * e_fid,
        "scale": w_scale * e_scale,
        "outlier": e_out,
    }
    energies["total"] = sum(energies.values())
    grad = (
        w_attract * g_att
        + w_repel * g_rep
        + w_void * g_void
        + w_elastic * g_ela
        + w_fidelity * g_fid
        + w_scale * g_scale
        + g_out
    )
    return energies, grad
