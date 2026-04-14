"""Matching utilities for local transition retrieval and weighting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from dev.particle_prediction.data.dataset import history_summary_features_from_segments
from dev.particle_prediction.data.transition_bank import (
    DEFAULT_ALPHA,
    DEFAULT_HISTORY_LENGTH,
    DEFAULT_K_STATE,
    DEFAULT_LAMBDA_H,
    DEFAULT_OFFSET_RADIUS,
    DEFAULT_SIGMA_H,
    DEFAULT_SIGMA_Z,
    MatchResult,
    TransitionBank,
    TransitionWindow,
    required_bank_history_length,
)


@dataclass(frozen=True)
class MatchingConfig:
    """Parameter bundle for candidate retrieval and history-aware weighting."""

    k_state: int = DEFAULT_K_STATE
    offset_radius: int = DEFAULT_OFFSET_RADIUS
    alpha: Optional[Sequence[float]] = None
    sigma_z: float = DEFAULT_SIGMA_Z
    sigma_h: float = DEFAULT_SIGMA_H
    lambda_h: float = DEFAULT_LAMBDA_H
    retrieval_method: str = "nn"
    history_mode: str = "ordered_segments"


def retrieve_state_candidates(
    bank: TransitionBank,
    query_state: np.ndarray,
    k_state: int = DEFAULT_K_STATE,
    method: str = "nn",
) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve bank candidates by Euclidean distance on the current state."""

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
        return weights / np.sum(weights)

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


def compute_fast_summary_distance_sq(
    query_state: np.ndarray,
    query_history_segments: np.ndarray,
    candidate_window: TransitionWindow,
    beta_position: float = 1.0,
    beta_direction: float = 1.0,
    beta_displacement: float = 0.5,
) -> float:
    """Compute a fast approximate history distance from summary features."""

    query_summary = history_summary_features_from_segments(
        current_state=np.asarray(query_state, dtype=np.float64),
        history_segments=np.asarray(query_history_segments, dtype=np.float64),
    )

    required = (
        candidate_window.mean_recent_position,
        candidate_window.mean_recent_direction,
        candidate_window.total_recent_displacement,
    )
    if any(value is None for value in required):
        raise ValueError("candidate window is missing fast-summary features")

    d_position = np.sum((query_summary["mean_recent_position"] - candidate_window.mean_recent_position) ** 2)
    d_direction = np.sum((query_summary["mean_recent_direction"] - candidate_window.mean_recent_direction) ** 2)
    d_displacement = np.sum((query_summary["total_recent_displacement"] - candidate_window.total_recent_displacement) ** 2)
    return float(beta_position * d_position + beta_direction * d_direction + beta_displacement * d_displacement)


def _candidate_history_distance_sq(
    bank: TransitionBank,
    index: int,
    query_state: np.ndarray,
    query_history_segments: np.ndarray,
    offset_radius: int,
    alpha: Optional[Sequence[float]],
    history_mode: str,
) -> float:
    if history_mode == "ordered_segments":
        return compute_history_distance_sq(
            query_history_segments=query_history_segments,
            candidate_history_segments=bank.history_tensor[index],
            offset_radius=offset_radius,
            alpha=alpha,
        )
    if history_mode == "fast_summary":
        return compute_fast_summary_distance_sq(
            query_state=query_state,
            query_history_segments=query_history_segments,
            candidate_window=bank.windows[index],
        )
    raise ValueError("history_mode must be 'ordered_segments' or 'fast_summary'")


def match_query_to_bank(
    bank: TransitionBank,
    query_state: np.ndarray,
    query_history_segments: np.ndarray | None = None,
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
    """Retrieve, rerank, and weight bank candidates for one query."""

    if sigma_z <= 0 or sigma_h <= 0:
        raise ValueError("sigma_z and sigma_h must be positive")
    if lambda_h < 0:
        raise ValueError("lambda_h must be non-negative")

    query_state = np.asarray(query_state, dtype=np.float64)
    candidate_indices, d_state_sq = retrieve_state_candidates(
        bank=bank,
        query_state=query_state,
        k_state=k_state,
        method=retrieval_method,
    )

    if query_history_segments is None:
        d_hist_sq = np.zeros(len(candidate_indices), dtype=np.float64)
    else:
        query_history_segments = np.asarray(query_history_segments, dtype=np.float64)
        history_length = query_history_segments.shape[0]
        if history_mode == "ordered_segments":
            max_supported_history = max(1, int(bank.history_length) - 2 * int(offset_radius))
            if history_length > max_supported_history:
                query_history_segments = query_history_segments[-max_supported_history:]
                history_length = query_history_segments.shape[0]
            if bank.history_length < required_bank_history_length(history_length, offset_radius):
                raise ValueError("TransitionBank history_length is too short for the requested matching configuration")
        d_hist_sq = np.asarray(
            [
                _candidate_history_distance_sq(
                    bank=bank,
                    index=index,
                    query_state=query_state,
                    query_history_segments=query_history_segments,
                    offset_radius=offset_radius,
                    alpha=alpha,
                    history_mode=history_mode,
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


def compare_matching_modes(
    bank: TransitionBank,
    query_state: np.ndarray,
    query_history_segments: np.ndarray,
    config: MatchingConfig | None = None,
) -> Dict[str, MatchResult]:
    """Return both default and fast-summary matches for one query."""

    config = MatchingConfig() if config is None else config
    default_match = match_query_to_bank(
        bank=bank,
        query_state=query_state,
        query_history_segments=query_history_segments,
        k_state=config.k_state,
        offset_radius=config.offset_radius,
        alpha=config.alpha,
        sigma_z=config.sigma_z,
        sigma_h=config.sigma_h,
        lambda_h=config.lambda_h,
        retrieval_method=config.retrieval_method,
        history_mode="ordered_segments",
    )
    fast_match = match_query_to_bank(
        bank=bank,
        query_state=query_state,
        query_history_segments=query_history_segments,
        k_state=config.k_state,
        offset_radius=config.offset_radius,
        alpha=config.alpha,
        sigma_z=config.sigma_z,
        sigma_h=config.sigma_h,
        lambda_h=config.lambda_h,
        retrieval_method=config.retrieval_method,
        history_mode="fast_summary",
    )
    return {"default": default_match, "fast_summary": fast_match}


__all__ = [
    "MatchingConfig",
    "compare_matching_modes",
    "compute_fast_summary_distance_sq",
    "compute_history_distance_sq",
    "match_query_to_bank",
    "retrieve_state_candidates",
]
