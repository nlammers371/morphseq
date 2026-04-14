"""Transition-window extraction for resampled trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from .dataset import history_summary_features_from_segments
from .resampling import ResampledTrajectory


@dataclass(frozen=True)
class TransitionWindow:
    """Canonical transition window extracted from a resampled trajectory."""

    state: np.ndarray
    increment: np.ndarray
    history_segments: np.ndarray
    embryo_id: str
    experiment_id: str
    perturbation_class: str
    source_segment_id: int
    touches_interpolated_gap: bool
    resampled_index: int
    arc_length_value: float
    mean_recent_position: np.ndarray | None = None
    mean_recent_direction: np.ndarray | None = None
    total_recent_displacement: np.ndarray | None = None
    source_time_estimate: float | None = None
    support_metadata: Dict[str, object] = field(default_factory=dict)


def _validate_resampled_trajectory(trajectory: ResampledTrajectory) -> None:
    if trajectory.resampled.ndim != 2:
        raise ValueError("trajectory.resampled must be a 2D array")
    if trajectory.arc_length.ndim != 1:
        raise ValueError("trajectory.arc_length must be a 1D array")
    if trajectory.resampled.shape[0] != trajectory.arc_length.shape[0]:
        raise ValueError("trajectory.resampled and trajectory.arc_length must have the same length")
    if np.any(~np.isfinite(trajectory.resampled)):
        raise ValueError("trajectory.resampled contains non-finite values")
    if np.any(~np.isfinite(trajectory.arc_length)):
        raise ValueError("trajectory.arc_length contains non-finite values")
    if np.any(np.diff(trajectory.arc_length) < 0):
        raise ValueError("trajectory.arc_length must be monotone nondecreasing")


def _compute_point_gap_annotations(
    trajectory: ResampledTrajectory,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = trajectory.source
    source_times = np.asarray(source.time_seconds, dtype=np.float64)
    point_times = np.asarray(trajectory.source_time_interp, dtype=np.float64)

    if len(source_times) != len(point_times) and len(source_times) == 1:
        point_segment_ids = np.zeros(len(point_times), dtype=np.int64)
        point_interpolated_gap_mask = np.zeros(len(point_times), dtype=bool)
        point_hard_gap_mask = np.zeros(len(point_times), dtype=bool)
        return point_segment_ids, point_interpolated_gap_mask, point_hard_gap_mask

    source_segment_ids = (
        np.asarray(source.segment_ids, dtype=np.int64)
        if source.segment_ids is not None
        else np.zeros(len(source_times), dtype=np.int64)
    )
    hard_gap_mask = (
        np.asarray(source.hard_gap_mask, dtype=bool)
        if source.hard_gap_mask is not None
        else np.zeros(max(len(source_times) - 1, 0), dtype=bool)
    )
    interpolatable_gap_mask = (
        np.asarray(source.interpolatable_gap_mask, dtype=bool)
        if source.interpolatable_gap_mask is not None
        else np.zeros(max(len(source_times) - 1, 0), dtype=bool)
    )

    if len(source_times) == 1:
        point_segment_ids = np.zeros(len(point_times), dtype=np.int64)
        point_interpolated_gap_mask = np.zeros(len(point_times), dtype=bool)
        point_hard_gap_mask = np.zeros(len(point_times), dtype=bool)
        return point_segment_ids, point_interpolated_gap_mask, point_hard_gap_mask

    left_indices = np.searchsorted(source_times, point_times, side="right") - 1
    left_indices = np.clip(left_indices, 0, len(source_times) - 2)
    right_indices = left_indices + 1

    dist_left = np.abs(point_times - source_times[left_indices])
    dist_right = np.abs(point_times - source_times[right_indices])
    nearest_indices = np.where(dist_right < dist_left, right_indices, left_indices)

    point_segment_ids = source_segment_ids[nearest_indices]
    inside_interval = (point_times > source_times[left_indices]) & (point_times < source_times[right_indices])
    point_interpolated_gap_mask = inside_interval & interpolatable_gap_mask[left_indices]
    point_hard_gap_mask = inside_interval & hard_gap_mask[left_indices]
    return point_segment_ids, point_interpolated_gap_mask, point_hard_gap_mask


def build_transition_windows(
    trajectory: ResampledTrajectory,
    history_length: int,
) -> List[TransitionWindow]:
    """Build transition windows for one resampled trajectory."""

    if history_length < 1:
        raise ValueError("history_length must be at least 1")

    _validate_resampled_trajectory(trajectory)

    points = np.asarray(trajectory.resampled, dtype=np.float64)
    if len(points) < history_length + 2:
        return []

    segments = np.diff(points, axis=0)
    point_segment_ids, point_interpolated_gap_mask, point_hard_gap_mask = _compute_point_gap_annotations(trajectory)
    windows: List[TransitionWindow] = []

    for state_index in range(history_length, len(points) - 1):
        involved_indices = np.arange(state_index - history_length, state_index + 2)
        if np.any(point_hard_gap_mask[involved_indices]):
            continue
        if len(np.unique(point_segment_ids[involved_indices])) != 1:
            continue

        history_segments = segments[state_index - history_length : state_index].copy()
        summary = history_summary_features_from_segments(
            current_state=points[state_index].copy(),
            history_segments=history_segments,
        )

        source_time_estimate = None
        if trajectory.source_time_interp is not None:
            source_time_estimate = float(trajectory.source_time_interp[state_index])

        windows.append(
            TransitionWindow(
                state=points[state_index].copy(),
                increment=(points[state_index + 1] - points[state_index]).copy(),
                history_segments=history_segments,
                embryo_id=trajectory.source.embryo_id,
                experiment_id=trajectory.source.experiment_id,
                perturbation_class=trajectory.source.perturbation_class,
                source_segment_id=int(point_segment_ids[state_index]),
                touches_interpolated_gap=bool(np.any(point_interpolated_gap_mask[involved_indices])),
                resampled_index=state_index,
                arc_length_value=float(trajectory.arc_length[state_index]),
                mean_recent_position=summary["mean_recent_position"],
                mean_recent_direction=summary["mean_recent_direction"],
                total_recent_displacement=summary["total_recent_displacement"],
                source_time_estimate=source_time_estimate,
                support_metadata=dict(trajectory.source.metadata),
            )
        )

    return windows


def build_transition_windows_for_dataset(
    trajectories: Sequence[ResampledTrajectory],
    history_length: int,
) -> List[TransitionWindow]:
    """Build transition windows across a collection of resampled trajectories."""

    windows: List[TransitionWindow] = []
    for trajectory in trajectories:
        windows.extend(build_transition_windows(trajectory=trajectory, history_length=history_length))
    return windows


__all__ = [
    "TransitionWindow",
    "build_transition_windows",
    "build_transition_windows_for_dataset",
]
