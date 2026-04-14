"""Prediction result containers for one-step and rollout forecasting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass(frozen=True)
class RolloutStepDiagnostics:
    """Aggregated support diagnostics for one rollout horizon."""

    candidate_count: float
    effective_sample_size: float
    history_mismatch: float
    search_radius: float
    selected_class_weights: Dict[str, float]
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RolloutPredictionResult:
    """Multi-step rollout output for one query."""

    predicted_mean: np.ndarray
    predicted_cov_diag: np.ndarray
    forward_samples: np.ndarray
    step_diagnostics: List[RolloutStepDiagnostics]
    diagnostics: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "RolloutPredictionResult",
    "RolloutStepDiagnostics",
]
