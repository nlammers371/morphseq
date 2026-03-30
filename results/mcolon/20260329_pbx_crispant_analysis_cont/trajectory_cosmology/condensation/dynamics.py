"""
dynamics.py
-----------
Damped gradient update loop, fidelity annealing, and stopping criteria.

Implements the alternating self-consistent dynamics:
  1. Recompute C_ij(t) from current positions (freeze for this step)
  2. Compute gradient of total energy
  3. Update velocities and positions (heavy-ball damping)
  4. Decay fidelity weight
  5. Check convergence
"""
from __future__ import annotations

import numpy as np

from .coherence import compute_coherence
from .forces import total_energy_and_grad
from .state import CondensationConfig, CondensationResult, CondensationState


def run_dynamics(
    x0: np.ndarray,
    mask: np.ndarray,
    config: CondensationConfig,
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
    log_every : int
        Log loss every N iterations.
    save_every : int or None
        Save a position snapshot every N iterations for animation.
        None disables snapshot saving. Snapshots are stored in
        CondensationResult.position_history as (n_saved, N_e, T, 2).
    verbose : bool

    Returns
    -------
    CondensationResult
    """
    positions = x0.copy()
    velocities = np.zeros_like(positions)
    loss_history = []
    position_snapshots = []   # filled if save_every is set
    snapshot_iters = []

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
        delta_x = velocities.copy()
        positions = positions + delta_x

        max_step = np.abs(delta_x[mask]).max() if mask.any() else 0.0

        if save_every is not None and n % save_every == 0:
            position_snapshots.append(positions.copy())
            snapshot_iters.append(n)

        if n % log_every == 0:
            entry = {"iter": n, "mu": mu, **energies}
            loss_history.append(entry)
            if verbose:
                print(
                    f"iter {n:4d} | total={energies['total']:+.4f} "
                    f"att={energies['attract']:+.4f} "
                    f"rep={energies['repel']:+.4f} "
                    f"ela={energies['elastic']:+.4f} "
                    f"fid={energies['fidelity']:+.4f} "
                    f"step={max_step:.5f}"
                )

        if max_step < config.tol:
            if verbose:
                print(f"Converged at iteration {n} (max_step={max_step:.2e})")
            return CondensationResult(
                positions=positions,
                x0=x0,
                mask=mask,
                loss_history=loss_history,
                n_iter=n + 1,
                converged=True,
                position_history=np.stack(position_snapshots) if position_snapshots else None,
                snapshot_iters=snapshot_iters,
            )

    return CondensationResult(
        positions=positions,
        x0=x0,
        mask=mask,
        loss_history=loss_history,
        n_iter=config.max_iter,
        converged=False,
        position_history=np.stack(position_snapshots) if position_snapshots else None,
        snapshot_iters=snapshot_iters,
    )
