"""
state.py
--------
Dataclasses for condensation config, running state, and result.

Keeps argument lists clean throughout condensation/*.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CondensationConfig:
    """Hyperparameters for the condensation dynamical system.

    Spatial / temporal
    ------------------
    sigma : float
        Gaussian kernel bandwidth in embedding units.
    delta : int
        Causal backward window size in time bins for coherence.

    Forces
    ------
    epsilon_r : float
        Repulsion strength.
    eta : float
        Soft-core repulsion stabilizer (prevents divide-by-zero).
    lambda_stretch : float
        Stretch (step-size) penalty weight.
    lambda_bend : float
        Bending (curvature) penalty weight.
    mu0 : float
        Initial fidelity anchor weight.
    gamma : float
        Fidelity decay factor per iteration (mu = mu0 * gamma^n).

    Optimization
    ------------
    alpha : float
        Momentum damping coefficient.
    lr : float
        Learning rate.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on max position change per iteration.
    """
    sigma: float = 0.5
    delta: int = 3
    epsilon_r: float = 0.01
    eta: float = 1e-4
    lambda_stretch: float = 0.1
    lambda_bend: float = 0.05
    mu0: float = 1.0
    gamma: float = 0.97
    alpha: float = 0.9
    lr: float = 0.01
    max_iter: int = 500
    tol: float = 1e-4


@dataclass
class CondensationState:
    """Mutable state during the optimization loop."""
    positions: np.ndarray        # (N_e, T, 2)
    velocities: np.ndarray       # (N_e, T, 2)
    coherence: np.ndarray        # (N_e, N_e, T) — frozen per iteration
    iteration: int = 0
    converged: bool = False


@dataclass
class CondensationResult:
    """Output of a completed condensation run."""
    positions: np.ndarray        # (N_e, T, 2) — final positions
    x0: np.ndarray               # (N_e, T, 2) — initial positions
    mask: np.ndarray             # (N_e, T) bool
    loss_history: list[dict] = field(default_factory=list)
    # each entry: {"iter": int, "attract": float, "repel": float,
    #              "stretch": float, "bend": float, "fidelity": float, "total": float}
    n_iter: int = 0
    converged: bool = False
    position_history: np.ndarray | None = None   # (n_saved, N_e, T, 2) or None
    snapshot_iters: list[int] = field(default_factory=list)
    # iteration numbers corresponding to position_history frames
