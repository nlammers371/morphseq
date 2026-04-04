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
    fidelity_init_strength : float
        Initial fidelity anchor weight at iteration 0. Set to 0.0 to disable fidelity.
    fidelity_half_life : float
        Internal per-iteration retention multiplier gamma (mu_n = fidelity_init_strength * gamma^n).
        Derived from fidelity_half_life_iters via gamma = 2^(-1/h). Not set directly by users.
    k_attract : int | None
        Number of nearest neighbors for local attraction. None = all pairs.
    subtract_mean_attraction : bool
        Whether to subtract the per-slice mean attraction gradient.

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
    sigma: float = 0.5       # attraction bandwidth (inter-bundle scale)
    sigma_coh: float | None = None  # coherence kernel bandwidth; None = use sigma
                                    # overridden when coherence_scale_mult is set
    attraction_scale_mult: float | None = None  # if set, sigma = c_att * s_global
    coherence_scale_mult: float | None = None   # if set, sigma_coh = c_coh * s_local
    coherence_mode: str = "computed"  # computed or uniform
    delta: int = 3
    epsilon_r: float = 0.01
    eta: float = 1e-4
    lambda_stretch: float = 0.1
    lambda_bend: float = 0.05
    fidelity_init_strength: float = 1.0   # full initial anchor pull
    fidelity_half_life: float = 0.99      # internal gamma; derived from fidelity_half_life_iters
                                          # do not set directly — use TemporalRunConfig.fidelity_half_life_iters
    k_attract: int | None = 15
    subtract_mean_attraction: bool = False
    # Two-scale force law
    sigma_attract_local: float | None = None  # None = use sigma (backwards-compatible)
    sigma_void: float | None = None           # None = disabled; set to use void repulsion
    epsilon_void: float = 0.0                 # void repulsion strength (0 = off)
    # Truncated repulsion (replaces soft-core when r_cut > 0)
    r_cut: float = 0.0                        # cutoff radius; 0 = use classic soft-core repulsion
                                              # if > 0: bump repulsion E = ε_r*(1-r²/r_cut²)² for r<r_cut
    # Local neighborhood scale preservation
    lambda_scale: float = 0.0                 # soft regularizer strength; 0 = off
                                              # E = λ_scale * Σ_i (r_i^(n) - r_i^(0))^2
                                              # start small (e.g. 0.1-1.0), not a hard leash
    w_attract: float = 1.0
    w_repel: float = 1.0
    w_fidelity: float = 1.0
    w_elastic: float = 1.0
    w_void: float = 1.0
    w_scale: float = 1.0
    k_local_scale: int = 5                    # number of initial neighbors defining r_i^(0)
    alpha: float = 0.9
    lr: float = 0.01
    max_iter: int = 500
    tol: float = 1e-4


@dataclass
class CondensationState:
    """Mutable state during the optimization loop."""
    positions: np.ndarray
    velocities: np.ndarray
    coherence: np.ndarray
    iteration: int = 0
    converged: bool = False


@dataclass
class CondensationResult:
    """Output of a completed condensation run."""
    positions: np.ndarray
    x0: np.ndarray
    mask: np.ndarray
    metrics_history: list[dict] = field(default_factory=list)
    n_iter: int = 0
    converged: bool = False
    position_history: np.ndarray | None = None
    snapshot_iters: list[int] = field(default_factory=list)
