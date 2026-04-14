"""Transition-bank construction plus compatibility wrappers for matching."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .resampling import ResampledTrajectory
from .transition_windows import TransitionWindow, build_transition_windows_for_dataset

try:
    from sklearn.neighbors import NearestNeighbors
except ImportError:  # pragma: no cover
    NearestNeighbors = None


DEFAULT_K_STATE = 64
DEFAULT_HISTORY_LENGTH = 3
DEFAULT_OFFSET_RADIUS = 1
DEFAULT_ALPHA = np.array([0.2, 0.3, 0.5], dtype=np.float64)
DEFAULT_SIGMA_Z = 1.0
DEFAULT_SIGMA_H = 1.0
DEFAULT_LAMBDA_H = 1.0


@dataclass(frozen=True)
class TransitionBank:
    """Canonical container for transition windows and stacked lookup arrays."""

    windows: List[TransitionWindow]
    history_length: int
    state_matrix: np.ndarray
    increment_matrix: np.ndarray
    history_tensor: np.ndarray
    class_labels: np.ndarray
    embryo_ids: List[str]
    experiment_ids: List[str]
    segment_ids: np.ndarray
    touches_interpolated_gap: np.ndarray
    mean_recent_position_matrix: np.ndarray | None = None
    mean_recent_direction_matrix: np.ndarray | None = None
    total_recent_displacement_matrix: np.ndarray | None = None
    summary_feature_matrix: np.ndarray | None = None
    nearest_neighbor_index: object | None = None
    quality_flags: Dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.windows)


@dataclass(frozen=True)
class MatchResult:
    """Candidate set and concrete weighting outputs for one query."""

    candidate_indices: np.ndarray
    candidate_windows: List[TransitionWindow]
    d_state_sq: np.ndarray
    d_hist_sq: np.ndarray
    scores: np.ndarray
    class_factors: np.ndarray
    quality_factors: np.ndarray
    unnormalized_weights: np.ndarray
    normalized_weights: np.ndarray


def required_bank_history_length(history_length: int, offset_radius: int) -> int:
    """Return the minimum bank history length needed for shifted history matching."""

    if history_length < 1:
        raise ValueError("history_length must be at least 1")
    if offset_radius < 0:
        raise ValueError("offset_radius must be non-negative")
    return history_length + 2 * offset_radius


def build_transition_bank(
    trajectories: Sequence[ResampledTrajectory],
    history_length: int,
    use_state_index: bool = True,
) -> TransitionBank:
    """Build a TransitionBank from resampled trajectories."""

    if history_length < 1:
        raise ValueError("history_length must be at least 1")

    windows = build_transition_windows_for_dataset(trajectories=trajectories, history_length=history_length)
    if not windows:
        raise ValueError("No valid transition windows available for bank construction")

    state_matrix = np.vstack([window.state for window in windows]).astype(np.float64)
    increment_matrix = np.vstack([window.increment for window in windows]).astype(np.float64)
    history_tensor = np.stack([window.history_segments for window in windows]).astype(np.float64)
    class_labels = np.asarray([window.perturbation_class for window in windows], dtype=object)
    embryo_ids = [window.embryo_id for window in windows]
    experiment_ids = [window.experiment_id for window in windows]
    segment_ids = np.asarray([window.source_segment_id for window in windows], dtype=np.int64)
    touches_interpolated_gap = np.asarray([window.touches_interpolated_gap for window in windows], dtype=bool)
    mean_recent_position_matrix = np.vstack([window.mean_recent_position for window in windows]).astype(np.float64)
    mean_recent_direction_matrix = np.vstack([window.mean_recent_direction for window in windows]).astype(np.float64)
    total_recent_displacement_matrix = np.vstack([window.total_recent_displacement for window in windows]).astype(np.float64)
    summary_feature_matrix = np.column_stack(
        [
            mean_recent_position_matrix,
            mean_recent_direction_matrix,
            total_recent_displacement_matrix,
        ]
    )

    nearest_neighbor_index = None
    if use_state_index and NearestNeighbors is not None:
        nearest_neighbor_index = NearestNeighbors(metric="euclidean")
        nearest_neighbor_index.fit(state_matrix)

    return TransitionBank(
        windows=windows,
        history_length=history_length,
        state_matrix=state_matrix,
        increment_matrix=increment_matrix,
        history_tensor=history_tensor,
        class_labels=class_labels,
        embryo_ids=embryo_ids,
        experiment_ids=experiment_ids,
        segment_ids=segment_ids,
        touches_interpolated_gap=touches_interpolated_gap,
        mean_recent_position_matrix=mean_recent_position_matrix,
        mean_recent_direction_matrix=mean_recent_direction_matrix,
        total_recent_displacement_matrix=total_recent_displacement_matrix,
        summary_feature_matrix=summary_feature_matrix,
        nearest_neighbor_index=nearest_neighbor_index,
        quality_flags={"touches_interpolated_gap": touches_interpolated_gap},
    )


def retrieve_state_candidates(
    bank: TransitionBank,
    query_state: np.ndarray,
    k_state: int = DEFAULT_K_STATE,
    method: str = "nn",
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper for the model-layer candidate retrieval."""

    from dev.particle_prediction.models.matching import retrieve_state_candidates as _retrieve_state_candidates

    return _retrieve_state_candidates(bank=bank, query_state=query_state, k_state=k_state, method=method)


def compute_history_distance_sq(
    query_history_segments: np.ndarray,
    candidate_history_segments: np.ndarray,
    offset_radius: int = DEFAULT_OFFSET_RADIUS,
    alpha: Optional[Sequence[float]] = None,
) -> float:
    """Compatibility wrapper for the model-layer history distance."""

    from dev.particle_prediction.models.matching import compute_history_distance_sq as _compute_history_distance_sq

    return _compute_history_distance_sq(
        query_history_segments=query_history_segments,
        candidate_history_segments=candidate_history_segments,
        offset_radius=offset_radius,
        alpha=alpha,
    )


def match_query_to_bank(
    bank: TransitionBank,
    query_state: np.ndarray,
    query_history_segments: np.ndarray | None,
    k_state: int = DEFAULT_K_STATE,
    offset_radius: int = DEFAULT_OFFSET_RADIUS,
    alpha: Optional[Sequence[float]] = None,
    sigma_z: float = DEFAULT_SIGMA_Z,
    sigma_h: float = DEFAULT_SIGMA_H,
    lambda_h: float = DEFAULT_LAMBDA_H,
    class_priors: Optional[Dict[str, float]] = None,
    quality_factor_fn: Optional[Callable[[TransitionWindow], float]] = None,
    retrieval_method: str = "nn",
    history_mode: str = "ordered_segments",
) -> MatchResult:
    """Compatibility wrapper for model-layer matching."""

    from dev.particle_prediction.models.matching import match_query_to_bank as _match_query_to_bank

    return _match_query_to_bank(
        bank=bank,
        query_state=query_state,
        query_history_segments=query_history_segments,
        k_state=k_state,
        offset_radius=offset_radius,
        alpha=alpha,
        sigma_z=sigma_z,
        sigma_h=sigma_h,
        lambda_h=lambda_h,
        class_priors=class_priors,
        quality_factor_fn=quality_factor_fn,
        retrieval_method=retrieval_method,
        history_mode=history_mode,
    )


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_HISTORY_LENGTH",
    "DEFAULT_K_STATE",
    "DEFAULT_LAMBDA_H",
    "DEFAULT_OFFSET_RADIUS",
    "DEFAULT_SIGMA_H",
    "DEFAULT_SIGMA_Z",
    "MatchResult",
    "TransitionBank",
    "TransitionWindow",
    "build_transition_bank",
    "compute_history_distance_sq",
    "match_query_to_bank",
    "required_bank_history_length",
    "retrieve_state_candidates",
]
