"""Query and evaluation-task helpers for particle prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Sequence

import numpy as np

from .resampling import ResampledTrajectory


@dataclass(frozen=True)
class PredictionQuery:
    """Canonical query object for snapshot- or history-based prediction."""

    mode: Literal["snapshot", "history"]
    current_state: np.ndarray
    history_segments: np.ndarray | None = None
    recent_points: np.ndarray | None = None
    class_prior: Dict[str, float] | None = None
    query_id: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictionTask:
    """One evaluation task with a query plus future target states."""

    query: PredictionQuery
    target_states: np.ndarray
    horizons: np.ndarray
    embryo_id: str
    experiment_id: str
    perturbation_class: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def ordered_history_segments_from_points(points: np.ndarray) -> np.ndarray:
    """Convert ordered recent points to ordered recent segments."""

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array")
    if len(points) < 2:
        raise ValueError("points must contain at least two rows")
    return np.diff(points, axis=0)


def history_summary_features_from_segments(
    current_state: np.ndarray,
    history_segments: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute optional fast-matching summary features from history segments."""

    current_state = np.asarray(current_state, dtype=np.float64)
    history_segments = np.asarray(history_segments, dtype=np.float64)
    if current_state.ndim != 1:
        raise ValueError("current_state must be a 1D array")
    if history_segments.ndim != 2:
        raise ValueError("history_segments must be a 2D array")
    if history_segments.shape[1] != current_state.shape[0]:
        raise ValueError("history_segments and current_state must share feature dimension")

    points = np.empty((history_segments.shape[0] + 1, current_state.shape[0]), dtype=np.float64)
    points[-1] = current_state
    for index in range(history_segments.shape[0] - 1, -1, -1):
        points[index] = points[index + 1] - history_segments[index]

    mean_recent_position = np.mean(points, axis=0)
    mean_recent_direction = np.mean(history_segments, axis=0)
    total_recent_displacement = points[-1] - points[0]
    return {
        "recent_points": points,
        "mean_recent_position": mean_recent_position,
        "mean_recent_direction": mean_recent_direction,
        "total_recent_displacement": total_recent_displacement,
    }


def build_prediction_query(
    current_state: np.ndarray,
    history_segments: np.ndarray | None = None,
    class_prior: Dict[str, float] | None = None,
    query_id: str | None = None,
    metadata: Dict[str, Any] | None = None,
) -> PredictionQuery:
    """Build a snapshot or history query from explicit arrays."""

    current_state = np.asarray(current_state, dtype=np.float64)
    if current_state.ndim != 1:
        raise ValueError("current_state must be a 1D array")

    metadata_dict = {} if metadata is None else dict(metadata)
    if history_segments is None:
        return PredictionQuery(
            mode="snapshot",
            current_state=current_state,
            class_prior=class_prior,
            query_id=query_id,
            metadata=metadata_dict,
        )

    history_segments = np.asarray(history_segments, dtype=np.float64)
    summary = history_summary_features_from_segments(current_state=current_state, history_segments=history_segments)
    return PredictionQuery(
        mode="history",
        current_state=current_state,
        history_segments=history_segments,
        recent_points=summary["recent_points"],
        class_prior=class_prior,
        query_id=query_id,
        metadata=metadata_dict,
    )


def build_query_from_resampled_trajectory(
    trajectory: ResampledTrajectory,
    state_index: int,
    history_length: int | None = None,
    class_prior: Dict[str, float] | None = None,
    query_id: str | None = None,
) -> PredictionQuery:
    """Build a query from a specific index in a resampled trajectory."""

    points = np.asarray(trajectory.resampled, dtype=np.float64)
    if state_index < 0 or state_index >= len(points):
        raise IndexError("state_index is out of bounds")

    metadata = {
        "embryo_id": trajectory.source.embryo_id,
        "experiment_id": trajectory.source.experiment_id,
        "perturbation_class": trajectory.source.perturbation_class,
        "resampled_index": int(state_index),
        "arc_length_value": float(trajectory.arc_length[state_index]),
    }
    if trajectory.source_time_interp is not None:
        metadata["source_time_estimate"] = float(trajectory.source_time_interp[state_index])

    if history_length is None or history_length < 1:
        return build_prediction_query(
            current_state=points[state_index].copy(),
            history_segments=None,
            class_prior=class_prior,
            query_id=query_id,
            metadata=metadata,
        )

    if state_index < history_length:
        raise ValueError("state_index must be at least history_length for a history query")

    recent_points = points[state_index - history_length : state_index + 1]
    return build_prediction_query(
        current_state=points[state_index].copy(),
        history_segments=ordered_history_segments_from_points(recent_points),
        class_prior=class_prior,
        query_id=query_id,
        metadata=metadata,
    )


def build_prediction_tasks(
    trajectories: Sequence[ResampledTrajectory],
    history_length: int,
    horizons: Sequence[int],
    mode: Literal["snapshot", "history"] = "history",
) -> List[PredictionTask]:
    """Build simple evaluation tasks from resampled trajectories."""

    if not horizons:
        raise ValueError("horizons must be non-empty")
    horizons_array = np.asarray(sorted(set(int(horizon) for horizon in horizons)), dtype=np.int64)
    if np.any(horizons_array < 1):
        raise ValueError("horizons must all be positive")

    tasks: List[PredictionTask] = []
    min_start = history_length if mode == "history" else 0
    max_horizon = int(horizons_array[-1])

    for trajectory in trajectories:
        points = np.asarray(trajectory.resampled, dtype=np.float64)
        max_state_index = len(points) - 1 - max_horizon
        if max_state_index < min_start:
            continue

        for state_index in range(min_start, max_state_index + 1):
            query = build_query_from_resampled_trajectory(
                trajectory=trajectory,
                state_index=state_index,
                history_length=history_length if mode == "history" else None,
            )
            target_indices = state_index + horizons_array
            tasks.append(
                PredictionTask(
                    query=query,
                    target_states=points[target_indices].copy(),
                    horizons=horizons_array.copy(),
                    embryo_id=trajectory.source.embryo_id,
                    experiment_id=trajectory.source.experiment_id,
                    perturbation_class=trajectory.source.perturbation_class,
                    metadata={
                        "state_index": int(state_index),
                        "arc_length_value": float(trajectory.arc_length[state_index]),
                    },
                )
            )

    return tasks


def summarize_prediction_tasks(tasks: Iterable[PredictionTask]) -> Dict[str, int]:
    """Return lightweight counts for notebook and smoke-test inspection."""

    tasks = list(tasks)
    return {
        "n_tasks": len(tasks),
        "n_snapshot_queries": sum(task.query.mode == "snapshot" for task in tasks),
        "n_history_queries": sum(task.query.mode == "history" for task in tasks),
        "n_embryos": len({task.embryo_id for task in tasks}),
    }


__all__ = [
    "PredictionQuery",
    "PredictionTask",
    "build_prediction_query",
    "build_prediction_tasks",
    "build_query_from_resampled_trajectory",
    "history_summary_features_from_segments",
    "ordered_history_segments_from_points",
    "summarize_prediction_tasks",
]
