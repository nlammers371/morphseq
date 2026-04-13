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
    if source.observed_dts is not None and len(source.observed_dts) != len(source.time_seconds) - 1:
        raise ValueError("source.observed_dts must have length len(time_seconds) - 1")
    if source.missing_frame_counts is not None and len(source.missing_frame_counts) != len(source.time_seconds) - 1:
        raise ValueError("source.missing_frame_counts must have length len(time_seconds) - 1")
    if source.interpolatable_gap_mask is not None and len(source.interpolatable_gap_mask) != len(source.time_seconds) - 1:
        raise ValueError("source.interpolatable_gap_mask must have length len(time_seconds) - 1")
    if source.hard_gap_mask is not None and len(source.hard_gap_mask) != len(source.time_seconds) - 1:
        raise ValueError("source.hard_gap_mask must have length len(time_seconds) - 1")
    if source.segment_ids is not None and len(source.segment_ids) != len(source.time_seconds):
        raise ValueError("source.segment_ids must have the same length as time_seconds")


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


def _get_segment_slices(source: EmbryoTrajectory) -> List[slice]:
    if source.segment_ids is None or len(source.segment_ids) == 0:
        return [slice(0, len(source.time_seconds))]

    segment_slices: List[slice] = []
    segment_ids = np.asarray(source.segment_ids, dtype=np.int64)
    start = 0
    for index in range(1, len(segment_ids)):
        if segment_ids[index] != segment_ids[index - 1]:
            segment_slices.append(slice(start, index))
            start = index
    segment_slices.append(slice(start, len(segment_ids)))
    return segment_slices


def _expand_segment(
    points: np.ndarray,
    time_seconds: np.ndarray,
    missing_frame_counts: np.ndarray,
    interpolatable_gap_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    expanded_points = [points[0]]
    expanded_times = [time_seconds[0]]
    observed_positions = [0]
    n_interpolated = 0

    for index in range(len(points) - 1):
        if interpolatable_gap_mask[index]:
            n_missing = int(missing_frame_counts[index])
            for step in range(1, n_missing + 1):
                alpha = step / (n_missing + 1)
                expanded_points.append((1.0 - alpha) * points[index] + alpha * points[index + 1])
                expanded_times.append((1.0 - alpha) * time_seconds[index] + alpha * time_seconds[index + 1])
                n_interpolated += 1

        expanded_points.append(points[index + 1])
        expanded_times.append(time_seconds[index + 1])
        observed_positions.append(len(expanded_points) - 1)

    return (
        np.asarray(expanded_points, dtype=np.float64),
        np.asarray(expanded_times, dtype=np.float64),
        np.asarray(observed_positions, dtype=np.int64),
        n_interpolated,
    )


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

    smoothed = np.empty_like(trajectory, dtype=np.float64)
    per_segment_window_frames: List[int] = []
    per_segment_poly_orders: List[int] = []
    n_interpolated_points = 0

    if source.missing_frame_counts is None:
        missing_frame_counts = np.zeros(len(time_seconds) - 1, dtype=np.int64)
    else:
        missing_frame_counts = np.asarray(source.missing_frame_counts, dtype=np.int64)

    if source.interpolatable_gap_mask is None:
        interpolatable_gap_mask = np.zeros(len(time_seconds) - 1, dtype=bool)
    else:
        interpolatable_gap_mask = np.asarray(source.interpolatable_gap_mask, dtype=bool)

    for segment_slice in _get_segment_slices(source):
        segment_points = trajectory[segment_slice]
        segment_times = time_seconds[segment_slice]
        if len(segment_points) == 1:
            smoothed[segment_slice] = segment_points
            per_segment_window_frames.append(1)
            per_segment_poly_orders.append(0)
            continue

        segment_missing = missing_frame_counts[segment_slice.start : segment_slice.stop - 1]
        segment_interpolatable = interpolatable_gap_mask[segment_slice.start : segment_slice.stop - 1]
        expanded_points, expanded_times, observed_positions, n_added = _expand_segment(
            points=segment_points,
            time_seconds=segment_times,
            missing_frame_counts=segment_missing,
            interpolatable_gap_mask=segment_interpolatable,
        )
        n_interpolated_points += n_added

        segment_window_frames = _resolve_window_frames(
            window_seconds=window_seconds,
            delta_t=source.delta_t,
            trajectory_length=len(expanded_times),
            poly_order=poly_order,
        )
        segment_poly_order = min(poly_order, max(0, segment_window_frames - 1))
        segment_smoothed = _smooth_array(
            trajectory=expanded_points,
            time_seconds=expanded_times,
            window_frames=segment_window_frames,
            poly_order=segment_poly_order,
        )
        smoothed[segment_slice] = segment_smoothed[observed_positions]
        per_segment_window_frames.append(segment_window_frames)
        per_segment_poly_orders.append(segment_poly_order)

    residuals = trajectory - smoothed

    diagnostics: Dict[str, object] = {
        "delta_t": float(source.delta_t),
        "trajectory_length": int(len(time_seconds)),
        "window_seconds": float(window_seconds),
        "window_frames": int(window_frames),
        "effective_poly_order": int(min(poly_order, max(0, window_frames - 1))),
        "n_segments": int(len(_get_segment_slices(source))),
        "n_interpolated_points": int(n_interpolated_points),
        "n_hard_gaps": int(np.sum(source.hard_gap_mask)) if source.hard_gap_mask is not None else 0,
        "per_segment_window_frames": per_segment_window_frames,
        "per_segment_poly_orders": per_segment_poly_orders,
    }

    return SmoothedTrajectory(
        source=source,
        smoothed=smoothed,
        time_seconds=time_seconds.copy(),
        method="savitzky_golay",
        window_seconds=float(window_seconds),
        window_frames=int(window_frames),
        poly_order=int(poly_order),
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
