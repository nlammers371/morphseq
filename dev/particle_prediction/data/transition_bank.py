"""Transition-bank construction plus state/history matching and weighting."""

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
        nearest_neighbor_index=nearest_neighbor_index,
        quality_flags={"touches_interpolated_gap": touches_interpolated_gap},
    )


def retrieve_state_candidates(
    bank: TransitionBank,
    query_state: np.ndarray,
    k_state: int = DEFAULT_K_STATE,
    method: str = "nn",
) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve candidates by Euclidean distance on current state only."""

    query_state = np.asarray(query_state, dtype=np.float64)
    if query_state.ndim != 1:
        raise ValueError("query_state must be a 1D array")
    if k_state < 1:
        raise ValueError("k_state must be at least 1")

    n_candidates = min(k_state, len(bank))
    if n_candidates == 0:
        raise ValueError("TransitionBank is empty")

    if method == "nn" and bank.nearest_neighbor_index is not None:
        distances, indices = bank.nearest_neighbor_index.kneighbors(query_state.reshape(1, -1), n_neighbors=n_candidates)
        return indices[0].astype(np.int64), np.square(distances[0].astype(np.float64))

    if method not in {"nn", "brute"}:
        raise ValueError("method must be 'nn' or 'brute'")

    d_state_sq = np.sum((bank.state_matrix - query_state[None, :]) ** 2, axis=1)
    indices = np.argsort(d_state_sq)[:n_candidates].astype(np.int64)
    return indices, d_state_sq[indices]


def _resolve_alpha(history_length: int, alpha: Optional[Sequence[float]]) -> np.ndarray:
    if alpha is None:
        if history_length == DEFAULT_HISTORY_LENGTH:
            return DEFAULT_ALPHA.copy()
        weights = np.arange(1, history_length + 1, dtype=np.float64)
        return weights / weights.sum()

    alpha_array = np.asarray(alpha, dtype=np.float64)
    if alpha_array.shape != (history_length,):
        raise ValueError("alpha must have shape (history_length,)")
    if np.any(alpha_array < 0):
        raise ValueError("alpha must be non-negative")
    if np.sum(alpha_array) <= 0:
        raise ValueError("alpha must have positive sum")
    return alpha_array / np.sum(alpha_array)


def compute_history_distance_sq(
    query_history_segments: np.ndarray,
    candidate_history_segments: np.ndarray,
    offset_radius: int = DEFAULT_OFFSET_RADIUS,
    alpha: Optional[Sequence[float]] = None,
) -> float:
    """Compute shifted ordered-history distance with recency weighting."""

    query_history_segments = np.asarray(query_history_segments, dtype=np.float64)
    candidate_history_segments = np.asarray(candidate_history_segments, dtype=np.float64)

    if query_history_segments.ndim != 2 or candidate_history_segments.ndim != 2:
        raise ValueError("query and candidate histories must be 2D arrays")
    if query_history_segments.shape[1] != candidate_history_segments.shape[1]:
        raise ValueError("query and candidate histories must have the same feature dimension")
    if offset_radius < 0:
        raise ValueError("offset_radius must be non-negative")

    history_length = query_history_segments.shape[0]
    alpha_array = _resolve_alpha(history_length, alpha)

    if candidate_history_segments.shape[0] < required_bank_history_length(history_length, offset_radius):
        raise ValueError("candidate history is too short for the requested history_length and offset_radius")

    center_start = (candidate_history_segments.shape[0] - history_length) // 2
    distances = []
    for offset in range(-offset_radius, offset_radius + 1):
        start = center_start + offset
        stop = start + history_length
        if start < 0 or stop > candidate_history_segments.shape[0]:
            continue
        diff = query_history_segments - candidate_history_segments[start:stop]
        distances.append(float(np.sum(alpha_array[:, None] * (diff ** 2))))

    if not distances:
        raise ValueError("No valid shifted history comparisons available")
    return float(min(distances))


def match_query_to_bank(
    bank: TransitionBank,
    query_state: np.ndarray,
    query_history_segments: np.ndarray,
    k_state: int = DEFAULT_K_STATE,
    offset_radius: int = DEFAULT_OFFSET_RADIUS,
    alpha: Optional[Sequence[float]] = None,
    sigma_z: float = DEFAULT_SIGMA_Z,
    sigma_h: float = DEFAULT_SIGMA_H,
    lambda_h: float = DEFAULT_LAMBDA_H,
    class_priors: Optional[Dict[str, float]] = None,
    quality_factor_fn: Optional[Callable[[TransitionWindow], float]] = None,
    retrieval_method: str = "nn",
) -> MatchResult:
    """Retrieve, rerank, and weight bank candidates for one query."""

    if sigma_z <= 0 or sigma_h <= 0:
        raise ValueError("sigma_z and sigma_h must be positive")
    if lambda_h < 0:
        raise ValueError("lambda_h must be non-negative")

    query_history_segments = np.asarray(query_history_segments, dtype=np.float64)
    history_length = query_history_segments.shape[0]
    if bank.history_length < required_bank_history_length(history_length, offset_radius):
        raise ValueError("TransitionBank history_length is too short for the requested matching configuration")

    candidate_indices, d_state_sq = retrieve_state_candidates(
        bank=bank,
        query_state=query_state,
        k_state=k_state,
        method=retrieval_method,
    )

    d_hist_sq = np.asarray(
        [
            compute_history_distance_sq(
                query_history_segments=query_history_segments,
                candidate_history_segments=bank.history_tensor[index],
                offset_radius=offset_radius,
                alpha=alpha,
            )
            for index in candidate_indices
        ],
        dtype=np.float64,
    )
    scores = d_state_sq / (2.0 * sigma_z ** 2) + lambda_h * d_hist_sq / (2.0 * sigma_h ** 2)

    unnormalized_weights = np.exp(-scores)
    candidate_windows = [bank.windows[index] for index in candidate_indices]

    class_factors = np.ones(len(candidate_indices), dtype=np.float64)
    if class_priors is not None:
        class_factors = np.asarray(
            [float(class_priors.get(window.perturbation_class, 1.0)) for window in candidate_windows],
            dtype=np.float64,
        )

    quality_factors = np.ones(len(candidate_indices), dtype=np.float64)
    if quality_factor_fn is not None:
        quality_factors = np.asarray([float(quality_factor_fn(window)) for window in candidate_windows], dtype=np.float64)

    unnormalized_weights = unnormalized_weights * class_factors * quality_factors
    total_weight = float(np.sum(unnormalized_weights))
    if total_weight <= 0:
        raise ValueError("All candidate weights are zero after applying priors and quality factors")
    normalized_weights = unnormalized_weights / total_weight

    order = np.argsort(scores)
    candidate_indices = candidate_indices[order]
    d_state_sq = d_state_sq[order]
    d_hist_sq = d_hist_sq[order]
    scores = scores[order]
    class_factors = class_factors[order]
    quality_factors = quality_factors[order]
    unnormalized_weights = unnormalized_weights[order]
    normalized_weights = normalized_weights[order]
    candidate_windows = [candidate_windows[index] for index in order]

    return MatchResult(
        candidate_indices=candidate_indices,
        candidate_windows=candidate_windows,
        d_state_sq=d_state_sq,
        d_hist_sq=d_hist_sq,
        scores=scores,
        class_factors=class_factors,
        quality_factors=quality_factors,
        unnormalized_weights=unnormalized_weights,
        normalized_weights=normalized_weights,
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
    "build_transition_bank",
    "compute_history_distance_sq",
    "match_query_to_bank",
    "required_bank_history_length",
    "retrieve_state_candidates",
]