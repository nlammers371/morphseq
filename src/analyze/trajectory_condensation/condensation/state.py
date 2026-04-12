"""
state.py
--------
Dataclasses for condensation config, running state, and result.

Keeps argument lists clean throughout condensation/*.

Naming conventions
------------------
- ``attract_*``         : attraction force parameters
- ``temporal_cohere_*`` : temporal coherence parameters
- ``elastic_*``         : elasticity force parameters
- ``outlier_*``         : slice outlier correction parameters
- ``void_*``            : void/occupancy repulsion parameters
- ``solver_*``          : optimizer (SGD-with-momentum) parameters
- ``fidelity_*``        : initial position fidelity parameters
- ``local_scale_*``     : local neighborhood scale preservation parameters

Internal geometry reference scales (set by the solver, not user-facing):
- ``sigma``             : attraction bandwidth (raw, set by geometry calibration)
- ``sigma_coh``         : coherence bandwidth
- ``epsilon_r``         : repulsion core radius
- ``lambda_stretch``    : elasticity stretch coefficient (resolved from elastic_*)
- ``lambda_bend``       : elasticity bend coefficient (resolved from elastic_*)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CondensationConfig:
    """Hyperparameters for the condensation dynamical system.

    Attraction
    ----------
    attract_k : int
        Number of nearest neighbors for the attraction force.
    attract_weight : float
        Overall weight of the attraction term.
    attract_bandwidth_mult : float or None
        Multiplier on the geometry-calibrated attraction bandwidth.

    Temporal coherence
    ------------------
    temporal_cohere_window : int
        Half-window (in time bins) for coherence smoothing.
    temporal_cohere_mode : str
        Coherence computation mode. Default ``"computed"``.
    temporal_cohere_weight : float
        Weight of the temporal coherence term.
    temporal_cohere_bandwidth_mult : float or None
        Multiplier on the geometry-calibrated coherence bandwidth.

    Elasticity
    ----------
    elastic_strength : float or None
        Overall elasticity weight (resolved against geometry refs at runtime).
    elastic_mix : float or None
        Split between stretch and bend. ``0`` = all stretch, ``1`` = all bend.
    elastic_kernel : str
        Elasticity penalty family, e.g. ``"quadratic"`` or ``"ratio_hinge"``.

    Outlier correction
    ------------------
    outlier_strength : float
        Weight of the slice-relative outlier correction term.
    outlier_cutoff_mode : str
        How outliers are defined: ``"quantile"`` or ``"robust"``.
    outlier_cutoff_value : float
        Cutoff threshold (quantile level or MAD multiplier).

    Void / occupancy
    ----------------
    void_strength : float
        Weight of the broad void repulsion term.
    void_bandwidth : float or None
        Spatial bandwidth for the void force.

    Solver
    ------
    solver_lr : float
        Learning rate for SGD-with-momentum.
    solver_momentum : float
        Momentum coefficient.
    solver_max_iter : int
        Maximum number of optimization iterations.
    solver_tol : float
        Convergence tolerance (stopping criterion).

    Fidelity
    --------
    fidelity_init_strength : float
        Initial weight of the fidelity (anchor-to-x0) term.
    fidelity_half_life : float
        Per-iteration decay factor for fidelity weight.

    Local scale
    -----------
    local_scale_strength : float
        Weight of the local neighborhood scale preservation term.

    Internal geometry scales (set by calibration, not user-facing)
    --------------------------------------------------------------
    sigma : float
        Attraction bandwidth.
    epsilon_r : float
        Repulsion core radius.
    lambda_stretch : float
        Raw elasticity stretch coefficient.
    lambda_bend : float
        Raw elasticity bend coefficient.
    """

    # Attraction
    attract_k: int = 15
    attract_weight: float = 1.0
    attract_bandwidth_mult: float | None = None

    # Temporal coherence
    temporal_cohere_window: int = 3
    temporal_cohere_mode: str = "computed"
    temporal_cohere_weight: float = 1.0
    temporal_cohere_bandwidth_mult: float | None = None

    # Elasticity
    elastic_strength: float | None = None
    elastic_mix: float | None = None
    elastic_kernel: str = "quadratic"

    # Outlier correction
    outlier_strength: float = 0.0
    outlier_cutoff_mode: str = "quantile"
    outlier_cutoff_value: float = 0.99

    # Void / occupancy
    void_strength: float = 0.0
    void_bandwidth: float | None = None

    # Solver
    solver_lr: float = 0.01
    solver_momentum: float = 0.9
    solver_max_iter: int = 500
    solver_tol: float = 1e-4

    # Fidelity
    fidelity_init_strength: float = 1.0
    fidelity_half_life: float = 0.99

    # Local scale
    local_scale_strength: float = 0.0

    # Internal geometry scales (calibration-facing, not user-facing)
    sigma: float = 0.5
    epsilon_r: float = 0.01
    lambda_stretch: float = 0.1
    lambda_bend: float = 0.05

    def __post_init__(self) -> None:
        if self.elastic_mix is not None and self.elastic_strength is None:
            raise ValueError("elastic_mix requires elastic_strength.")
        if self.elastic_mix is not None and not (0.0 <= self.elastic_mix <= 1.0):
            raise ValueError("elastic_mix must be within [0, 1].")


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


@dataclass(frozen=True)
class ForceBalanceSummary:
    """Resolved force parameters after geometry-aware calibration."""

    sigma_att: float
    sigma_coh: float
    epsilon_r: float
    lambda_stretch: float
    lambda_bend: float
    local_scale_strength: float
    elastic_strength: float | None
    elastic_mix: float | None
    elastic_kernel: str
    outlier_strength: float
    outlier_cutoff_mode: str
    outlier_cutoff_value: float
    void_strength: float
    void_bandwidth: float | None
    fidelity_strength: float
    geometry_s_local: float
    geometry_s_step: float
    geometry_s_bend: float
    geometry_s_global: float

    def to_dict(self) -> dict[str, float | None]:
        return {
            "sigma_att": self.sigma_att,
            "sigma_coh": self.sigma_coh,
            "epsilon_r": self.epsilon_r,
            "lambda_stretch": self.lambda_stretch,
            "lambda_bend": self.lambda_bend,
            "local_scale_strength": self.local_scale_strength,
            "elastic_strength": self.elastic_strength,
            "elastic_mix": self.elastic_mix,
            "elastic_kernel": self.elastic_kernel,
            "outlier_strength": self.outlier_strength,
            "outlier_cutoff_mode": self.outlier_cutoff_mode,
            "outlier_cutoff_value": self.outlier_cutoff_value,
            "void_strength": self.void_strength,
            "void_bandwidth": self.void_bandwidth,
            "fidelity_strength": self.fidelity_strength,
            "s_local": self.geometry_s_local,
            "s_step": self.geometry_s_step,
            "s_bend": self.geometry_s_bend,
            "s_global": self.geometry_s_global,
        }
