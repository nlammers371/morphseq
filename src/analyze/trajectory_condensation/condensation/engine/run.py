"""
run.py
------
Damped gradient update loop, fidelity annealing, and stopping criteria.
"""
from __future__ import annotations

import numpy as np

from ..coherence.compute import compute_coherence
from ..forces.total import total_energy_and_grad
from ..forces.attraction import summarize_attraction_support
from ..geometry_refs import build_local_scale_refs, build_slice_outlier_refs, estimate_geometry_refs
from ..state import CondensationConfig, CondensationResult, ForceBalanceSummary

from .stopping import (
    StoppingConfig,
    StoppingMonitor,
    log_metrics,
    reference_scale_from_positions,
)

_K_LOCAL_SCALE_DEFAULT = 5


def _uniform_coherence(mask: np.ndarray) -> np.ndarray:
    n_e, t_count = mask.shape
    coherence = np.zeros((n_e, n_e, t_count), dtype=float)
    for t in range(t_count):
        obs = np.flatnonzero(mask[:, t])
        if len(obs) == 0:
            continue
        coherence[np.ix_(obs, obs, [t])] = 1.0
    return coherence


def resolve_force_balance(
    x0: np.ndarray,
    mask: np.ndarray,
    config: CondensationConfig,
) -> ForceBalanceSummary:
    """Resolve geometry-aware force parameters for a specific run."""
    geometry_refs = estimate_geometry_refs(x0, mask, k_local=_K_LOCAL_SCALE_DEFAULT)
    sigma_att = (
        config.attract_bandwidth_mult * geometry_refs.s_global
        if config.attract_bandwidth_mult is not None
        else config.sigma
    )
    sigma_coh = (
        config.temporal_cohere_bandwidth_mult * geometry_refs.s_local
        if config.temporal_cohere_bandwidth_mult is not None
        else sigma_att
    )
    if config.elastic_strength is not None:
        elastic_mix = 0.5 if config.elastic_mix is None else config.elastic_mix
        if config.elastic_kernel == "quadratic":
            lambda_stretch = config.elastic_strength * (1.0 - elastic_mix) / max(geometry_refs.s_step**2, 1e-12)
            lambda_bend = config.elastic_strength * elastic_mix / max(geometry_refs.s_bend**2, 1e-12)
        elif config.elastic_kernel == "ratio_hinge":
            lambda_stretch = config.elastic_strength * (1.0 - elastic_mix)
            lambda_bend = config.elastic_strength * elastic_mix
        else:
            raise ValueError(f"Unsupported elastic_kernel={config.elastic_kernel!r}")
    else:
        elastic_mix = None
        lambda_stretch = config.lambda_stretch
        lambda_bend = config.lambda_bend

    return ForceBalanceSummary(
        sigma_att=float(sigma_att),
        sigma_coh=float(sigma_coh),
        epsilon_r=float(config.epsilon_r),
        lambda_stretch=float(lambda_stretch),
        lambda_bend=float(lambda_bend),
        local_scale_strength=float(config.local_scale_strength),
        elastic_strength=None if config.elastic_strength is None else float(config.elastic_strength),
        elastic_mix=None if elastic_mix is None else float(elastic_mix),
        elastic_kernel=str(config.elastic_kernel),
        outlier_strength=float(config.outlier_strength),
        outlier_cutoff_mode=str(config.outlier_cutoff_mode),
        outlier_cutoff_value=float(config.outlier_cutoff_value),
        void_strength=float(config.void_strength),
        void_bandwidth=None if config.void_bandwidth is None else float(config.void_bandwidth),
        fidelity_strength=float(config.fidelity_init_strength),
        geometry_s_local=float(geometry_refs.s_local),
        geometry_s_step=float(geometry_refs.s_step),
        geometry_s_bend=float(geometry_refs.s_bend),
        geometry_s_global=float(geometry_refs.s_global),
    )


def describe_force_balance(
    x0: np.ndarray,
    mask: np.ndarray,
    config: CondensationConfig,
) -> dict[str, float | None]:
    """Return a serializable summary of resolved force parameters."""
    return resolve_force_balance(x0, mask, config).to_dict()


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
    resolved_forces = resolve_force_balance(x0, mask, config)
    sigma_att = resolved_forces.sigma_att
    sigma_coh = resolved_forces.sigma_coh
    monitor = StoppingMonitor(stopping)

    if verbose:
        print(
            f"  run_dynamics: sigma_att={sigma_att:.4f}  sigma_coh={sigma_coh:.4f}  epsilon_r={config.epsilon_r:.6f}  "
            f"s_local={resolved_forces.geometry_s_local:.4f}  s_global={resolved_forces.geometry_s_global:.4f}  "
            f"epsilon_r/s_local²={config.epsilon_r / (resolved_forces.geometry_s_local**2 + 1e-16):.4f}"
        )
        print(
            "  force_balance:"
            f" local_scale_strength={resolved_forces.local_scale_strength:.6f}"
            f" elastic_strength={resolved_forces.elastic_strength if resolved_forces.elastic_strength is not None else 'legacy'}"
            f" elastic_mix={resolved_forces.elastic_mix if resolved_forces.elastic_mix is not None else 'legacy'}"
            f" elastic_kernel={resolved_forces.elastic_kernel}"
            f" lambda_stretch={resolved_forces.lambda_stretch:.6f}"
            f" lambda_bend={resolved_forces.lambda_bend:.6f}"
            f" outlier_strength={resolved_forces.outlier_strength:.6f}"
            f" outlier_cutoff={resolved_forces.outlier_cutoff_mode}:{resolved_forces.outlier_cutoff_value:g}"
            f" void_strength={resolved_forces.void_strength:.6f}"
            f" void_bandwidth={resolved_forces.void_bandwidth if resolved_forces.void_bandwidth is not None else 'implicit'}"
        )

    local_scale_refs = build_local_scale_refs(x0, mask, k_local=_K_LOCAL_SCALE_DEFAULT)
    if config.outlier_cutoff_mode == "quantile":
        outlier_refs = build_slice_outlier_refs(
            x0, mask,
            cutoff_mode="quantile",
            quantile=float(config.outlier_cutoff_value),
        )
    elif config.outlier_cutoff_mode == "robust":
        outlier_refs = build_slice_outlier_refs(
            x0, mask,
            cutoff_mode="robust",
            robust_k=float(config.outlier_cutoff_value),
        )
    else:
        raise ValueError(f"Unsupported outlier_cutoff_mode={config.outlier_cutoff_mode!r}")

    prev_total_energy: float | None = None
    prev_coherence: np.ndarray | None = None

    # Uniform coherence never changes — compute once before the loop.
    # Computed coherence is cached inside the loop at coherence_cache_every cadence.
    coherence_last_computed: int = -1
    if config.temporal_cohere_mode == "uniform":
        coherence: np.ndarray = _uniform_coherence(mask)
    elif config.temporal_cohere_mode == "computed":
        coherence = None  # type: ignore[assignment]  # will be set on first iteration
    else:
        raise ValueError(f"Unsupported temporal_cohere_mode={config.temporal_cohere_mode!r}")

    for n in range(config.solver_max_iter):
        # Recompute coherence on cadence (uniform was computed once above).
        # Coherence is a frozen weight matrix passed into attraction(); no explicit gradient.
        if config.temporal_cohere_mode == "computed" and (
            coherence is None
            or (n - coherence_last_computed) >= config.coherence_cache_every
        ):
            coherence = compute_coherence(
                positions, mask,
                sigma=sigma_coh,
                delta=config.temporal_cohere_window,
            )
            coherence_last_computed = n

        mu = config.fidelity_init_strength * (config.fidelity_half_life ** n)

        energies, grad = total_energy_and_grad(
            positions=positions,
            x0=x0,
            mask=mask,
            coherence=coherence,
            sigma=sigma_att,
            epsilon_r=config.epsilon_r,
            eta=1e-4,
            lambda_stretch=resolved_forces.lambda_stretch,
            lambda_bend=resolved_forces.lambda_bend,
            elasticity_kernel=config.elastic_kernel,
            s_step_ref=resolved_forces.geometry_s_step,
            s_bend_ref=resolved_forces.geometry_s_bend,
            mu=mu,
            k_attract=config.attract_k,
            subtract_mean_attraction=False,
            sigma_attract_local=None,
            epsilon_void=resolved_forces.void_strength,
            sigma_void=resolved_forces.void_bandwidth,
            outlier_strength=resolved_forces.outlier_strength,
            outlier_refs=outlier_refs,
            r_cut=0.0,
            lambda_scale=resolved_forces.local_scale_strength,
            neighborhood_info=local_scale_refs,
            w_attract=config.attract_weight,
            w_repel=1.0,
            w_fidelity=1.0,
            w_elastic=1.0,
            w_void=1.0,
            w_scale=1.0,
        )

        grad *= mask[:, :, None].astype(float)
        velocities = config.solver_momentum * velocities - config.solver_lr * grad
        new_positions = positions + velocities

        # Only compute support diagnostics on log iterations — avoids duplicating
        # the full attraction pass (diff, sq_dist, K, C, kNN) every iteration.
        if n % log_every == 0:
            support_metrics: dict[str, float] | None = summarize_attraction_support(
                positions=positions,
                mask=mask,
                coherence=coherence,
                sigma=sigma_att,
                sigma_attract_local=None,
                k_attract=config.attract_k,
            )
        else:
            support_metrics = None

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
        row['coherence_stale'] = n - coherence_last_computed
        metrics_history.append(row)

        positions = new_positions

        if save_every is not None and n % save_every == 0:
            position_snapshots.append(positions.copy())
            snapshot_iters.append(n)

        if n % log_every == 0 and verbose:
            void_str = f" void={energies['void']:+.4f}" if resolved_forces.void_strength > 0 else ""
            scale_str = f" scale={energies['scale']:+.4f}" if resolved_forces.local_scale_strength > 0 else ""
            outlier_str = f" out={energies['outlier']:+.4f}" if resolved_forces.outlier_strength > 0 else ""
            print(
                f"iter {n:4d} | total={energies['total']:+.4f} "
                f"att={energies['attract']:+.4f} rep={energies['repel']:+.4f}"
                f"{void_str}{scale_str}{outlier_str} "
                f"ela={energies['elastic']:+.4f} fid={energies['fidelity']:+.4f} "
                f"disp_rms={row['disp_rms_rel']:.5f} disp_max={row['disp_max_rel']:.5f} "
                f"coh_stale={row['coherence_stale']}"
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
        n_iter=config.solver_max_iter,
        converged=False,
        position_history=np.stack(position_snapshots) if position_snapshots else None,
        snapshot_iters=snapshot_iters,
    )
