"""Arc-length reparameterization and fixed-step resampling utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from .loading import EmbryoTrajectory
from .smoothing import SmoothedTrajectory


@dataclass(frozen=True)
class ResampledTrajectory:
    """A smoothed trajectory resampled on a fixed arc-length grid."""

    source: EmbryoTrajectory
    smoothed_source: SmoothedTrajectory
    resampled: np.ndarray
    arc_length: np.ndarray
    delta_s: float
    source_time_interp: np.ndarray | None = None
    source_frame_interp: np.ndarray | None = None
    increment_norms: np.ndarray | None = None
    diagnostics: Dict[str, object] = field(default_factory=dict)


def _validate_smoothed_trajectory(trajectory: SmoothedTrajectory) -> None:
    if trajectory.smoothed.ndim != 2:
        raise ValueError("trajectory.smoothed must be a 2D array")
    if trajectory.time_seconds.ndim != 1:
        raise ValueError("trajectory.time_seconds must be a 1D array")
    if trajectory.smoothed.shape[0] != trajectory.time_seconds.shape[0]:
        raise ValueError("smoothed trajectory and time_seconds must have the same length")
    if np.any(~np.isfinite(trajectory.smoothed)):
        raise ValueError("trajectory.smoothed contains non-finite values")
    if np.any(~np.isfinite(trajectory.time_seconds)):
        raise ValueError("trajectory.time_seconds contains non-finite values")
    if np.any(np.diff(trajectory.time_seconds) < 0):
        raise ValueError("trajectory.time_seconds must be monotone nondecreasing")


def compute_cumulative_arc_length(points: np.ndarray) -> np.ndarray:
    """Return cumulative latent-space arc length for a `(T, D)` trajectory."""

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array")
    if len(points) == 0:
        raise ValueError("points must contain at least one row")
    if np.any(~np.isfinite(points)):
        raise ValueError("points contains non-finite values")

    if len(points) == 1:
        return np.array([0.0], dtype=np.float64)

    increments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(increments)))


def _keep_last_unique_samples(x: np.ndarray) -> np.ndarray:
    mask = np.ones(len(x), dtype=bool)
    if len(x) > 1:
        mask[:-1] = np.diff(x) > 0
    return mask


def _build_resampling_grid(total_arc_length: float, delta_s: float) -> np.ndarray:
    if delta_s <= 0 or not np.isfinite(delta_s):
        raise ValueError("delta_s must be finite and positive")
    if total_arc_length < 0 or not np.isfinite(total_arc_length):
        raise ValueError("total_arc_length must be finite and non-negative")
    if total_arc_length == 0:
        return np.array([0.0], dtype=np.float64)

    grid = np.arange(0.0, total_arc_length + delta_s, delta_s, dtype=np.float64)
    if grid[-1] > total_arc_length:
        grid[-1] = total_arc_length
    if not np.isclose(grid[-1], total_arc_length):
        grid = np.append(grid, total_arc_length)
    return grid


def _interpolate_rows(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        return np.interp(x_new, x, y)

    columns = [np.interp(x_new, x, y[:, dim]) for dim in range(y.shape[1])]
    return np.column_stack(columns)


def resample_smoothed_trajectory(
    trajectory: SmoothedTrajectory,
    delta_s: float,
) -> ResampledTrajectory:
    """Resample a smoothed trajectory on a fixed latent-space arc-length grid."""

    _validate_smoothed_trajectory(trajectory)

    smoothed = np.asarray(trajectory.smoothed, dtype=np.float64)
    time_seconds = np.asarray(trajectory.time_seconds, dtype=np.float64)
    cumulative_arc_length = compute_cumulative_arc_length(smoothed)
    total_arc_length = float(cumulative_arc_length[-1])
    resampled_arc_length = _build_resampling_grid(total_arc_length, delta_s)

    keep_mask = _keep_last_unique_samples(cumulative_arc_length)
    support_arc_length = cumulative_arc_length[keep_mask]
    support_points = smoothed[keep_mask]
    support_time = time_seconds[keep_mask]

    resampled_points = _interpolate_rows(support_arc_length, support_points, resampled_arc_length)
    source_time_interp = _interpolate_rows(support_arc_length, support_time, resampled_arc_length)

    source_frame_interp = None
    if trajectory.source.frame_index is not None:
        frame_index = np.asarray(trajectory.source.frame_index, dtype=np.float64)
        if frame_index.shape[0] != time_seconds.shape[0]:
            raise ValueError("source.frame_index must have the same length as the trajectory")
        source_frame_interp = _interpolate_rows(support_arc_length, frame_index[keep_mask], resampled_arc_length)

    increment_norms = np.linalg.norm(np.diff(resampled_points, axis=0), axis=1)
    diagnostics: Dict[str, object] = {
        "n_source_points": int(smoothed.shape[0]),
        "n_resampled_points": int(resampled_points.shape[0]),
        "total_arc_length": total_arc_length,
        "delta_s": float(delta_s),
    }

    return ResampledTrajectory(
        source=trajectory.source,
        smoothed_source=trajectory,
        resampled=resampled_points,
        arc_length=resampled_arc_length,
        delta_s=float(delta_s),
        source_time_interp=np.asarray(source_time_interp, dtype=np.float64),
        source_frame_interp=None if source_frame_interp is None else np.asarray(source_frame_interp, dtype=np.float64),
        increment_norms=np.asarray(increment_norms, dtype=np.float64),
        diagnostics=diagnostics,
    )


def resample_smoothed_trajectories(
    trajectories: Sequence[SmoothedTrajectory],
    delta_s: float,
) -> List[ResampledTrajectory]:
    """Resample a collection of smoothed trajectories using a shared `delta_s`."""

    return [resample_smoothed_trajectory(trajectory=trajectory, delta_s=delta_s) for trajectory in trajectories]


__all__ = [
    "ResampledTrajectory",
    "compute_cumulative_arc_length",
    "resample_smoothed_trajectory",
    "resample_smoothed_trajectories",
]
