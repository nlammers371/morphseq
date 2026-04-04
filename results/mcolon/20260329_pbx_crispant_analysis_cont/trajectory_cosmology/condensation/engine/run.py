"""
run.py
------
Damped gradient update loop, fidelity annealing, and stopping criteria.
"""
from __future__ import annotations

import numpy as np

from ..coherence.compute import compute_coherence
from ..forces.local_scale import build_neighborhood_info
from ..forces.total import total_energy_and_grad
from ..forces.attraction import summarize_attraction_support
from ..geometry_refs import estimate_geometry_refs, estimate_local_spacing_ref
from ..state import CondensationConfig, CondensationResult


def _uniform_coherence(mask: np.ndarray) -> np.ndarray:
    n_e, t_count = mask.shape
    coherence = np.zeros((n_e, n_e, t_count), dtype=float)
    for t in range(t_count):
        obs = np.flatnonzero(mask[:, t])
        if len(obs) == 0:
            continue
        coherence[np.ix_(obs, obs, [t])] = 1.0
    return coherence

from .stopping import (
    StoppingConfig,
    StoppingMonitor,
    log_metrics,
    reference_scale_from_positions,
)


def run_dynamics(
    x0: np.ndarray,
    mask: np.ndarray,
    config: CondensationConfig,
    stopping: StoppingConfig | None = None,
    log_every: int = 10,
    save_every: int | None = None,
    verbose: bool = True,
) -> CondensationResult:
    """Run the condensation dynamical system."""
    if stopping is None:
        stopping = StoppingConfig()

    positions = x0.copy()
    velocities = np.zeros_like(positions)
    metrics_history = []
    position_snapshots = []
    snapshot_iters = []

    reference_scale = reference_scale_from_positions(x0, mask)
    geometry_refs = estimate_geometry_refs(x0, mask, k_local=5)
    local_spacing_ref = geometry_refs.s_local
    sigma_att = (
        config.attraction_scale_mult * geometry_refs.s_global
        if config.attraction_scale_mult is not None
        else config.sigma
    )
    sigma_coh = (
        config.coherence_scale_mult * geometry_refs.s_local
        if config.coherence_scale_mult is not None
        else (config.sigma_coh if config.sigma_coh is not None else sigma_att)
    )
    monitor = StoppingMonitor(stopping)

    if verbose:
        print(
            f"  run_dynamics: sigma_att={sigma_att:.4f}  sigma_coh={sigma_coh:.4f}  epsilon_r={config.epsilon_r:.6f}  "
            f"s_local={geometry_refs.s_local:.4f}  s_global={geometry_refs.s_global:.4f}  "
            f"epsilon_r/s_local²={config.epsilon_r / (local_spacing_ref**2 + 1e-16):.4f}"
        )

    # Precompute fixed local neighborhood structure from initial positions.
    # This is the anchor for local_scale_preservation — never updated during the loop.
    neighborhood_info = build_neighborhood_info(x0, mask, k_local=config.k_local_scale)

    prev_total_energy: float | None = None
    prev_coherence: np.ndarray | None = None

    # sigma_att and sigma_coh are resolved once from geometry refs + dimensionless
    # multipliers when provided, keeping the calibrated scheme explicit.

    for n in range(config.max_iter):
        if config.coherence_mode == "computed":
            coherence = compute_coherence(positions, mask, sigma=sigma_coh, delta=config.delta)
        elif config.coherence_mode == "uniform":
            coherence = _uniform_coherence(mask)
        else:
            raise ValueError(f"Unsupported coherence_mode={config.coherence_mode!r}")
        mu = config.fidelity_init_strength * (config.fidelity_half_life ** n)

        energies, grad = total_energy_and_grad(
            positions=positions,
            x0=x0,
            mask=mask,
            coherence=coherence,
            sigma=sigma_att,
            epsilon_r=config.epsilon_r,
            eta=config.eta,
            lambda_stretch=config.lambda_stretch,
            lambda_bend=config.lambda_bend,
            mu=mu,
            k_attract=config.k_attract,
            subtract_mean_attraction=config.subtract_mean_attraction,
            sigma_attract_local=config.sigma_attract_local,
            epsilon_void=config.epsilon_void,
            sigma_void=config.sigma_void,
            r_cut=config.r_cut,
            lambda_scale=config.lambda_scale,
            neighborhood_info=neighborhood_info,
            w_attract=config.w_attract,
            w_repel=config.w_repel,
            w_fidelity=config.w_fidelity,
            w_elastic=config.w_elastic,
            w_void=config.w_void,
            w_scale=config.w_scale,
        )

        grad *= mask[:, :, None].astype(float)
        velocities = config.alpha * velocities - config.lr * grad
        new_positions = positions + velocities

        support_metrics = summarize_attraction_support(
            positions=positions,
            mask=mask,
            coherence=coherence,
            sigma=sigma_att,
            sigma_attract_local=config.sigma_attract_local,
            k_attract=config.k_attract,
        )

        row = log_metrics(
            iteration=n,
            x_prev=positions,
            x_curr=new_positions,
            mask=mask,
            reference_scale=reference_scale,
            energy_terms=energies,
            prev_total_energy=prev_total_energy,
            C_prev=prev_coherence,
            C_curr=coherence,
            support_metrics=support_metrics,
        )
        row['mu'] = mu
        metrics_history.append(row)

        positions = new_positions

        if save_every is not None and n % save_every == 0:
            position_snapshots.append(positions.copy())
            snapshot_iters.append(n)

        if n % log_every == 0 and verbose:
            void_str = f" void={energies['void']:+.4f}" if config.epsilon_void > 0 else ""
            scale_str = f" scale={energies['scale']:+.4f}" if config.lambda_scale > 0 else ""
            print(
                f"iter {n:4d} | total={energies['total']:+.4f} "
                f"att={energies['attract']:+.4f} rep={energies['repel']:+.4f}"
                f"{void_str}{scale_str} "
                f"ela={energies['elastic']:+.4f} fid={energies['fidelity']:+.4f} "
                f"disp_rms={row['disp_rms_rel']:.5f} disp_max={row['disp_max_rel']:.5f}"
            )

        should_stop, reason = monitor.update(row)
        if should_stop:
            if verbose:
                print(f"Stopping at iteration {n}: {reason}")
            return CondensationResult(
                positions=positions,
                x0=x0,
                mask=mask,
                metrics_history=metrics_history,
                n_iter=n + 1,
                converged=True,
                position_history=np.stack(position_snapshots) if position_snapshots else None,
                snapshot_iters=snapshot_iters,
            )

        prev_total_energy = energies['total']
        prev_coherence = coherence

    return CondensationResult(
        positions=positions,
        x0=x0,
        mask=mask,
        metrics_history=metrics_history,
        n_iter=config.max_iter,
        converged=False,
        position_history=np.stack(position_snapshots) if position_snapshots else None,
        snapshot_iters=snapshot_iters,
    )
