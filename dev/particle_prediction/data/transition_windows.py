"""Minimal transition-window extraction for resampled trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

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
    resampled_index: int
    arc_length_value: float
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


def build_transition_windows(
    trajectory: ResampledTrajectory,
    history_length: int,
) -> List[TransitionWindow]:
    """Build transition windows for one resampled trajectory.

    For current state index ``i``, the window stores ``state = z_i``,
    ``increment = z_{i+1} - z_i``, and the previous ``history_length`` ordered
    resampled segments ending at ``z_i``.
    """

    if history_length < 1:
        raise ValueError("history_length must be at least 1")

    _validate_resampled_trajectory(trajectory)

    points = np.asarray(trajectory.resampled, dtype=np.float64)
    if len(points) < history_length + 2:
        return []

    segments = np.diff(points, axis=0)
    windows: List[TransitionWindow] = []

    for state_index in range(history_length, len(points) - 1):
        source_time_estimate = None
        if trajectory.source_time_interp is not None:
            source_time_estimate = float(trajectory.source_time_interp[state_index])

        windows.append(
            TransitionWindow(
                state=points[state_index].copy(),
                increment=(points[state_index + 1] - points[state_index]).copy(),
                history_segments=segments[state_index - history_length : state_index].copy(),
                embryo_id=trajectory.source.embryo_id,
                experiment_id=trajectory.source.experiment_id,
                perturbation_class=trajectory.source.perturbation_class,
                resampled_index=state_index,
                arc_length_value=float(trajectory.arc_length[state_index]),
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