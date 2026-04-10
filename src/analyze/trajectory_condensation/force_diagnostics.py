"""Reusable force-decomposition diagnostics for trajectory condensation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .condensation.coherence.compute import compute_coherence
from .condensation.engine.run import _uniform_coherence, resolve_force_balance
from .condensation.forces.attraction import attraction
from .condensation.forces.elasticity import elasticity
from .condensation.forces.fidelity import fidelity
from .condensation.forces.local_scale import local_scale_preservation
from .condensation.forces.repulsion import repulsion
from .condensation.forces.slice_outlier import slice_outlier
from .condensation.forces.void import void_repulsion
from .condensation.geometry_refs import build_local_scale_refs, build_slice_outlier_refs
from .condensation.state import CondensationConfig


@dataclass(frozen=True)
class ForceSnapshot:
    energies: dict[str, float]
    gradients: dict[str, np.ndarray]
    coherence: np.ndarray
    mu: float
    slice_centroids: np.ndarray


def force_snapshot(
    *,
    positions: np.ndarray,
    x0: np.ndarray,
    mask: np.ndarray,
    config: CondensationConfig,
    iteration: int = 0,
) -> ForceSnapshot:
    """Compute per-force energies and gradients for one solver state."""
    resolved = resolve_force_balance(x0, mask, config)

    if config.temporal_cohere_mode == "computed":
        coherence = compute_coherence(positions, mask, sigma=resolved.sigma_coh, delta=config.temporal_cohere_window)
    elif config.temporal_cohere_mode == "uniform":
        coherence = _uniform_coherence(mask)
    else:
        raise ValueError(f"Unsupported temporal_cohere_mode={config.temporal_cohere_mode!r}")

    mu = config.fidelity_init_strength * (config.fidelity_half_life ** iteration)
    neighborhood_info = build_local_scale_refs(x0, mask, k_local=config.k_local_scale)
    if config.slice_outlier_cutoff_mode == "quantile":
        outlier_refs = build_slice_outlier_refs(
            x0,
            mask,
            cutoff_mode="quantile",
            quantile=float(config.slice_outlier_cutoff_value),
        )
    elif config.slice_outlier_cutoff_mode == "robust":
        outlier_refs = build_slice_outlier_refs(
            x0,
            mask,
            cutoff_mode="robust",
            robust_k=float(config.slice_outlier_cutoff_value),
        )
    else:
        raise ValueError(f"Unsupported slice_outlier_cutoff_mode={config.slice_outlier_cutoff_mode!r}")

    e_att, g_att = attraction(
        positions,
        mask,
        coherence,
        resolved.sigma_att,
        sigma_attract_local=config.sigma_attract_local,
        k_attract=config.k_attract,
        subtract_mean=config.subtract_mean_attraction,
    )
    e_rep, g_rep = repulsion(positions, mask, config.epsilon_r, config.eta, r_cut=config.r_cut)
    e_void, g_void = void_repulsion(
        positions,
        mask,
        resolved.void_strength,
        resolved.void_bandwidth if resolved.void_bandwidth is not None else resolved.sigma_att,
    )
    e_ela, g_ela = elasticity(
        positions,
        mask,
        resolved.lambda_stretch,
        resolved.lambda_bend,
        elasticity_kernel=config.elasticity_kernel,
        s_step_ref=resolved.geometry_s_step,
        s_bend_ref=resolved.geometry_s_bend,
    )
    e_fid, g_fid = fidelity(positions, x0, mask, mu)
    e_scale, g_scale = local_scale_preservation(positions, mask, neighborhood_info, resolved.local_scale_strength)
    e_out, g_out = slice_outlier(positions, mask, outlier_refs, resolved.outlier_strength)

    gradients = {
        "attract": config.w_attract * g_att,
        "repel": config.w_repel * g_rep,
        "void": config.w_void * g_void,
        "elastic": config.w_elastic * g_ela,
        "fidelity": config.w_fidelity * g_fid,
        "scale": config.w_scale * g_scale,
        "outlier": g_out,
    }
    gradients["total"] = sum(gradients.values())

    energies = {
        "attract": config.w_attract * e_att,
        "repel": config.w_repel * e_rep,
        "void": config.w_void * e_void,
        "elastic": config.w_elastic * e_ela,
        "fidelity": config.w_fidelity * e_fid,
        "scale": config.w_scale * e_scale,
        "outlier": e_out,
    }
    energies["total"] = float(sum(energies.values()))

    slice_centroids = np.full((positions.shape[1], positions.shape[2]), np.nan, dtype=float)
    for t in range(positions.shape[1]):
        obs = np.flatnonzero(mask[:, t])
        if len(obs) == 0:
            continue
        slice_centroids[t] = positions[obs, t, :].mean(axis=0)

    return ForceSnapshot(
        energies=energies,
        gradients=gradients,
        coherence=coherence,
        mu=float(mu),
        slice_centroids=slice_centroids,
    )


def force_target_table(
    *,
    snapshot: ForceSnapshot,
    positions: np.ndarray,
    mask: np.ndarray,
    time_values: np.ndarray,
    ids: np.ndarray,
    labels: np.ndarray,
    config: CondensationConfig,
    targets: list[tuple[str, float]],
) -> pd.DataFrame:
    """Summarize per-force magnitudes and implied step sizes for selected embryo/slices."""
    id_to_idx = {str(item_id): idx for idx, item_id in enumerate(ids)}
    rows: list[dict[str, object]] = []
    immediate_mult = float(config.lr)
    steady_mult = float(config.lr / max(1.0 - config.alpha, 1e-12))

    for item_id, time_value in targets:
        item_idx = id_to_idx.get(str(item_id))
        if item_idx is None:
            rows.append({"id": item_id, "time_bin_center": float(time_value), "present": False})
            continue
        t_idx = int(np.argmin(np.abs(time_values - time_value)))
        if not mask[item_idx, t_idx]:
            rows.append({"id": item_id, "time_bin_center": float(time_values[t_idx]), "present": False})
            continue

        pos = positions[item_idx, t_idx, :]
        center = snapshot.slice_centroids[t_idx]
        delta = center - pos
        radius = float(np.linalg.norm(delta))
        if radius > 1e-12:
            radial_hat = delta / radius
        else:
            radial_hat = np.zeros_like(delta)

        row: dict[str, object] = {
            "id": str(item_id),
            "time_bin_center": float(time_values[t_idx]),
            "present": True,
            "label": str(labels[item_idx]),
            "radius_to_slice_centroid": radius,
            "mu_fidelity": snapshot.mu,
            "immediate_lr": immediate_mult,
            "steady_lr": steady_mult,
        }
        for name, grad in snapshot.gradients.items():
            g = grad[item_idx, t_idx, :]
            norm = float(np.linalg.norm(g))
            radial_component = float(np.dot(g, radial_hat))
            row[f"{name}_grad_norm"] = norm
            row[f"{name}_grad_radial_inward"] = radial_component
            row[f"{name}_step_immediate"] = immediate_mult * norm
            row[f"{name}_step_steady"] = steady_mult * norm
            row[f"{name}_step_immediate_inward"] = immediate_mult * radial_component
            row[f"{name}_step_steady_inward"] = steady_mult * radial_component
        rows.append(row)

    return pd.DataFrame(rows)
