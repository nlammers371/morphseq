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

    Public config is moving toward structural names like ``attract_*``,
    ``temporal_cohere_*``, and ``solver_*``. Legacy/internal names remain as
    compatibility aliases so older scripts still run.
    """

    # Public structural names
    attract_bandwidth_mult: float | None = None
    attract_scale_mult: float | None = None
    attract_k: int | None = None
    temporal_cohere_bandwidth_mult: float | None = None
    temporal_cohere_scale_mult: float | None = None
    temporal_cohere_window: int | None = None
    temporal_cohere_mode: str | None = None
    attract_weight: float | None = None
    temporal_cohere_weight: float | None = None
    solver_lr: float | None = None
    solver_max_iter: int | None = None
    solver_tol: float | None = None
    solver_momentum: float | None = None

    # Legacy / internal names kept for compatibility.
    sigma: float = 0.5
    sigma_coh: float | None = None
    attraction_scale_mult: float | None = None
    coherence_scale_mult: float | None = None
    coherence_mode: str = "computed"
    coherence_weight: float = 1.0
    delta: int = 3
    epsilon_r: float = 0.01
    eta: float = 1e-4
    lambda_stretch: float = 0.1
    lambda_bend: float = 0.05
    fidelity_init_strength: float = 1.0
    fidelity_half_life: float = 0.99
    k_attract: int | None = 15
    subtract_mean_attraction: bool = False
    sigma_attract_local: float | None = None
    sigma_void: float | None = None
    epsilon_void: float = 0.0
    r_cut: float = 0.0
    lambda_scale: float = 0.0
    w_attract: float = 1.0
    w_repel: float = 1.0
    w_fidelity: float = 1.0
    w_elastic: float = 1.0
    w_void: float = 1.0
    w_scale: float = 1.0
    k_local_scale: int = 5
    alpha: float = 0.9
    lr: float = 0.01
    max_iter: int = 500
    tol: float = 1e-4

    def __post_init__(self) -> None:
        attract_bandwidth_public = self.attract_bandwidth_mult
        if attract_bandwidth_public is None:
            attract_bandwidth_public = self.attract_scale_mult
        self.attraction_scale_mult = self._resolve_alias(
            public_value=attract_bandwidth_public,
            legacy_value=self.attraction_scale_mult,
            public_name="attract_bandwidth_mult",
            legacy_name="attraction_scale_mult",
            legacy_default=None,
        )
        self.attract_bandwidth_mult = self.attraction_scale_mult
        self.attract_scale_mult = self.attraction_scale_mult

        self.k_attract = self._resolve_alias(
            public_value=self.attract_k,
            legacy_value=self.k_attract,
            public_name="attract_k",
            legacy_name="k_attract",
            legacy_default=15,
        )
        self.attract_k = self.k_attract

        temporal_cohere_bandwidth_public = self.temporal_cohere_bandwidth_mult
        if temporal_cohere_bandwidth_public is None:
            temporal_cohere_bandwidth_public = self.temporal_cohere_scale_mult
        self.coherence_scale_mult = self._resolve_alias(
            public_value=temporal_cohere_bandwidth_public,
            legacy_value=self.coherence_scale_mult,
            public_name="temporal_cohere_bandwidth_mult",
            legacy_name="coherence_scale_mult",
            legacy_default=None,
        )
        self.temporal_cohere_bandwidth_mult = self.coherence_scale_mult
        self.temporal_cohere_scale_mult = self.coherence_scale_mult

        self.delta = self._resolve_alias(
            public_value=self.temporal_cohere_window,
            legacy_value=self.delta,
            public_name="temporal_cohere_window",
            legacy_name="delta",
            legacy_default=3,
        )
        self.temporal_cohere_window = self.delta

        self.coherence_mode = self._resolve_alias(
            public_value=self.temporal_cohere_mode,
            legacy_value=self.coherence_mode,
            public_name="temporal_cohere_mode",
            legacy_name="coherence_mode",
            legacy_default="computed",
        )
        self.temporal_cohere_mode = self.coherence_mode

        self.w_attract = self._resolve_alias(
            public_value=self.attract_weight,
            legacy_value=self.w_attract,
            public_name="attract_weight",
            legacy_name="w_attract",
            legacy_default=1.0,
        )
        self.attract_weight = self.w_attract

        self.coherence_weight = self._resolve_alias(
            public_value=self.temporal_cohere_weight,
            legacy_value=self.coherence_weight,
            public_name="temporal_cohere_weight",
            legacy_name="coherence_weight",
            legacy_default=1.0,
        )
        self.temporal_cohere_weight = self.coherence_weight

        self.lr = self._resolve_alias(
            public_value=self.solver_lr,
            legacy_value=self.lr,
            public_name="solver_lr",
            legacy_name="lr",
            legacy_default=0.01,
        )
        self.solver_lr = self.lr

        self.max_iter = self._resolve_alias(
            public_value=self.solver_max_iter,
            legacy_value=self.max_iter,
            public_name="solver_max_iter",
            legacy_name="max_iter",
            legacy_default=500,
        )
        self.solver_max_iter = self.max_iter

        self.tol = self._resolve_alias(
            public_value=self.solver_tol,
            legacy_value=self.tol,
            public_name="solver_tol",
            legacy_name="tol",
            legacy_default=1e-4,
        )
        self.solver_tol = self.tol

        self.alpha = self._resolve_alias(
            public_value=self.solver_momentum,
            legacy_value=self.alpha,
            public_name="solver_momentum",
            legacy_name="alpha",
            legacy_default=0.9,
        )
        self.solver_momentum = self.alpha

    @staticmethod
    def _resolve_alias(*, public_value, legacy_value, public_name: str, legacy_name: str, legacy_default):
        if public_value is None:
            return legacy_value
        if legacy_value is not None and legacy_value != legacy_default and public_value != legacy_value:
            raise ValueError(
                f"Conflicting values for {public_name} and {legacy_name}: "
                f"{public_value!r} vs {legacy_value!r}"
            )
        return public_value


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
