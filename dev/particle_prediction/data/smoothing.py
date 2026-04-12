"""Savitzky-Golay smoothing for particle-prediction trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Sequence

import numpy as np

from .loading import EmbryoTrajectory


@dataclass(frozen=True)
class SmoothedTrajectory:
    """Smoothed latent trajectory derived from a raw embryo trajectory.

    The raw `EmbryoTrajectory` is preserved intact in `source`. The smoothed
    representation keeps the same sample count and time axis.
    """

    source: EmbryoTrajectory
    smoothed: np.ndarray
    time_seconds: np.ndarray
    method: Literal["savitzky_golay"]
    window_seconds: float
    window_frames: int
    poly_order: int
    residuals: np.ndarray | None = None
    diagnostics: Dict[str, object] = field(default_factory=dict)


def _nearest_odd_integer(value: float) -> int:
    rounded = int(round(value))
    if rounded % 2 == 1:
        return max(1, rounded)

    lower = rounded - 1
    upper = rounded + 1
    if lower < 1:
        return 1
    return lower if abs(value - lower) <= abs(value - upper) else upper


def _largest_odd_at_most(value: int) -> int:
    return value if value % 2 == 1 else value - 1


def _minimum_valid_window(poly_order: int) -> int:
    minimum = poly_order + 1
    return minimum if minimum % 2 == 1 else minimum + 1


def _resolve_window_frames(
    window_seconds: float,
    delta_t: float,
    trajectory_length: int,
    poly_order: int,
) -> int:
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if delta_t <= 0 or not np.isfinite(delta_t):
        raise ValueError("delta_t must be finite and positive")
    if trajectory_length < 1:
        raise ValueError("trajectory_length must be at least 1")

    target_frames = _nearest_odd_integer(window_seconds / delta_t)
    target_frames = max(target_frames, _minimum_valid_window(poly_order))
    clipped = min(target_frames, trajectory_length)
    clipped = _largest_odd_at_most(clipped)
    return max(1, clipped)


def _validate_trajectory(source: EmbryoTrajectory) -> None:
    if source.trajectory.ndim != 2:
        raise ValueError("source.trajectory must be a 2D array")
    if source.time_seconds.ndim != 1:
        raise ValueError("source.time_seconds must be a 1D array")
    if source.trajectory.shape[0] != source.time_seconds.shape[0]:
        raise ValueError("trajectory and time_seconds must have the same length")
    if np.any(~np.isfinite(source.trajectory)):
        raise ValueError("source.trajectory contains non-finite values")
    if np.any(~np.isfinite(source.time_seconds)):
        raise ValueError("source.time_seconds contains non-finite values")
    if np.any(np.diff(source.time_seconds) < 0):
        raise ValueError("source.time_seconds must be monotone nondecreasing")


def _fit_local_polynomial(times: np.ndarray, values: np.ndarray, poly_order: int) -> np.ndarray:
    design = np.vander(times, N=poly_order + 1, increasing=True)
    coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    return coefficients[0]


def _smooth_array(
    trajectory: np.ndarray,
    time_seconds: np.ndarray,
    window_frames: int,
    poly_order: int,
) -> np.ndarray:
    n_samples, _ = trajectory.shape
    if window_frames == 1 or n_samples <= poly_order:
        return trajectory.copy()

    half_window = window_frames // 2
    smoothed = np.empty_like(trajectory, dtype=np.float64)

    for index in range(n_samples):
        start = max(0, index - half_window)
        stop = min(n_samples, index + half_window + 1)

        if stop - start < window_frames:
            if start == 0:
                stop = min(n_samples, window_frames)
            else:
                start = max(0, n_samples - window_frames)

        local_times = time_seconds[start:stop] - time_seconds[index]
        local_values = trajectory[start:stop]

        local_order = min(poly_order, len(local_times) - 1)
        if local_order < 0:
            smoothed[index] = trajectory[index]
            continue

        smoothed[index] = _fit_local_polynomial(local_times, local_values, local_order)

    return smoothed


def smooth_trajectory(
    source: EmbryoTrajectory,
    window_seconds: float,
    poly_order: int = 2,
) -> SmoothedTrajectory:
    """Return a Savitzky-Golay-smoothed representation of one trajectory.

    The smoothing window is specified in seconds and converted to frames using
    the trajectory's experiment-level `delta_t`. This keeps the smoothing scale
    comparable across experiments with different acquisition cadences.
    """

    if poly_order < 0:
        raise ValueError("poly_order must be non-negative")

    _validate_trajectory(source)

    trajectory = np.asarray(source.trajectory, dtype=np.float64)
    time_seconds = np.asarray(source.time_seconds, dtype=np.float64)
    window_frames = _resolve_window_frames(
        window_seconds=window_seconds,
        delta_t=source.delta_t,
        trajectory_length=len(time_seconds),
        poly_order=poly_order,
    )

    effective_order = min(poly_order, max(0, window_frames - 1))
    smoothed = _smooth_array(
        trajectory=trajectory,
        time_seconds=time_seconds,
        window_frames=window_frames,
        poly_order=effective_order,
    )
    residuals = trajectory - smoothed

    diagnostics: Dict[str, object] = {
        "delta_t": float(source.delta_t),
        "trajectory_length": int(len(time_seconds)),
        "window_seconds": float(window_seconds),
        "window_frames": int(window_frames),
        "effective_poly_order": int(effective_order),
    }

    return SmoothedTrajectory(
        source=source,
        smoothed=smoothed,
        time_seconds=time_seconds.copy(),
        method="savitzky_golay",
        window_seconds=float(window_seconds),
        window_frames=int(window_frames),
        poly_order=int(effective_order),
        residuals=residuals,
        diagnostics=diagnostics,
    )


def smooth_trajectories(
    trajectories: Sequence[EmbryoTrajectory],
    window_seconds: float,
    poly_order: int = 2,
) -> List[SmoothedTrajectory]:
    """Smooth a collection of embryo trajectories with a shared time-scale."""

    return [
        smooth_trajectory(
            source=trajectory,
            window_seconds=window_seconds,
            poly_order=poly_order,
        )
        for trajectory in trajectories
    ]


__all__ = ["SmoothedTrajectory", "smooth_trajectory", "smooth_trajectories"]
