"""
dynamics.py
-----------
Damped gradient update loop, fidelity annealing, and stopping criteria.

Implements the alternating self-consistent dynamics:
  1. Recompute C_ij(t) from current positions (freeze for this step)
  2. Compute gradient of total energy
  3. Update velocities and positions (heavy-ball damping)
  4. Decay fidelity weight
  5. Evaluate multi-metric stopping heuristics (via stopping.py)
"""
from __future__ import annotations

import numpy as np

from .coherence import compute_coherence
from .forces import total_energy_and_grad
from .state import CondensationConfig, CondensationResult
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
    """Run the condensation dynamical system.

    Parameters
    ----------
    x0 : (N_e, T, 2)
        Initial positions from init_embedding.
    mask : (N_e, T) bool
        Observation mask.
    config : CondensationConfig
        Dynamical system hyperparameters.
    stopping : StoppingConfig or None
        Multi-metric stopping heuristics. Uses defaults if None.
        Set thresholds to None to disable individual criteria.
    log_every : int
        Log and print diagnostics every N iterations.
    save_every : int or None
        Save a position snapshot every N iterations for animation.
        None disables snapshot saving. Snapshots are stored in
        CondensationResult.position_history as (n_saved, N_e, T, 2).
    verbose : bool

    Returns
    -------
    CondensationResult
    """
    if stopping is None:
        stopping = StoppingConfig()

    positions = x0.copy()
    velocities = np.zeros_like(positions)
    metrics_history = []       # one dict per logged iteration
    position_snapshots = []    # filled if save_every is set
    snapshot_iters = []

    reference_scale = reference_scale_from_positions(x0, mask)
    monitor = StoppingMonitor(stopping)

    prev_positions = x0.copy()
    prev_total_energy: float | None = None
    prev_coherence: np.ndarray | None = None

    for n in range(config.max_iter):
        # --- alternating update: recompute coherence, then freeze ---
        coherence = compute_coherence(
            positions, mask, sigma=config.sigma, delta=config.delta
        )

        mu = config.mu0 * (config.gamma ** n)

        energies, grad = total_energy_and_grad(
            positions=positions,
            x0=x0,
            mask=mask,
            coherence=coherence,
            sigma=config.sigma,
            epsilon_r=config.epsilon_r,
            eta=config.eta,
            lambda_stretch=config.lambda_stretch,
            lambda_bend=config.lambda_bend,
            mu=mu,
        )

        # zero gradient where unobserved
        grad *= mask[:, :, None].astype(float)

        # heavy-ball damped update
        velocities = config.alpha * velocities - config.lr * grad
        new_positions = positions + velocities

        # --- stopping metrics ---
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
        )
        row["mu"] = mu
        metrics_history.append(row)

        positions = new_positions

        if save_every is not None and n % save_every == 0:
            position_snapshots.append(positions.copy())
            snapshot_iters.append(n)

        if n % log_every == 0 and verbose:
            print(
                f"iter {n:4d} | total={energies['total']:+.4f} "
                f"att={energies['attract']:+.4f} "
                f"rep={energies['repel']:+.4f} "
                f"ela={energies['elastic']:+.4f} "
                f"fid={energies['fidelity']:+.4f} "
                f"disp_rms={row['disp_rms_rel']:.5f} "
                f"disp_max={row['disp_max_rel']:.5f}"
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

        prev_positions = positions.copy()
        prev_total_energy = energies["total"]
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
