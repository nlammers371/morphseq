"""Savitzky-Golay smoothing utilities for beta particle prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .loading import EmbryoTrajectory, TrajectoryDataset

try:
    from scipy.signal import savgol_filter
except ImportError:  # pragma: no cover - exercised only when scipy is absent.
    savgol_filter = None


@dataclass(frozen=True)
class SmoothedTrajectory:
    """One smoothed trajectory with lineage back to a raw embryo track."""

    source: EmbryoTrajectory
    smoothed: np.ndarray
    time_seconds: np.ndarray
    method: str
    window_seconds: float
    window_frames: int
    poly_order: int
    residuals: Optional[np.ndarray] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SmoothedTrajectoryDataset:
    """Collection wrapper for smoothed trajectories."""

    trajectories: List[SmoothedTrajectory]
    source_dataset: TrajectoryDataset

    def __len__(self) -> int:
        return len(self.trajectories)

    @property
    def class_names(self) -> List[str]:
        return self.source_dataset.class_names


def _odd_round(value: float) -> int:
    rounded = max(1, int(round(value)))
    return rounded if rounded % 2 == 1 else rounded + 1


def _largest_valid_odd(trajectory_length: int, minimum: int) -> Optional[int]:
    if trajectory_length < minimum:
        return None
    candidate = trajectory_length if trajectory_length % 2 == 1 else trajectory_length - 1
    if candidate < minimum:
        return None
    return candidate


def resolve_window_frames(
    trajectory_length: int,
    *,
    poly_order: int = 2,
    window_frames: int = 5,
    delta_t: Optional[float] = None,
    window_seconds: Optional[float] = None,
    tres: Optional[float] = None,
    smoothing_tres: Optional[float] = None,
) -> Optional[int]:
    """Resolve the SG window length in frames.

    Precedence:
    1. `tres` and `smoothing_tres`
    2. `window_seconds` and `delta_t`
    3. `window_frames`
    """

    minimum = poly_order + 1
    if minimum % 2 == 0:
        minimum += 1

    if tres is not None and smoothing_tres is not None and tres > 0:
        raw_frames = smoothing_tres / tres
    elif window_seconds is not None and delta_t is not None and np.isfinite(delta_t) and delta_t > 0:
        raw_frames = window_seconds / delta_t
    else:
        raw_frames = float(window_frames)

    resolved = max(_odd_round(raw_frames), minimum)
    max_valid = _largest_valid_odd(trajectory_length, minimum)
    if max_valid is None:
        return None
    return min(resolved, max_valid)


def _effective_window_seconds(
    resolved_window_frames: int,
    *,
    delta_t: float,
    window_seconds: Optional[float],
    tres: Optional[float],
    smoothing_tres: Optional[float],
) -> float:
    if smoothing_tres is not None:
        return float(smoothing_tres)
    if window_seconds is not None:
        return float(window_seconds)
    if np.isfinite(delta_t):
        return float(resolved_window_frames * delta_t)
    if tres is not None:
        return float(resolved_window_frames * tres)
    return float(resolved_window_frames)


def smooth_trajectory(
    trajectory: EmbryoTrajectory,
    *,
    poly_order: int = 2,
    window_frames: int = 5,
    window_seconds: Optional[float] = None,
    tres: Optional[float] = None,
    smoothing_tres: Optional[float] = None,
    mode: str = "interp",
) -> SmoothedTrajectory:
    """Smooth one embryo trajectory with a Savitzky-Golay filter."""

    if savgol_filter is None:
        raise ImportError("scipy is required for Savitzky-Golay smoothing")

    raw = trajectory.trajectory
    resolved_window = resolve_window_frames(
        len(raw),
        poly_order=poly_order,
        window_frames=window_frames,
        delta_t=trajectory.delta_t,
        window_seconds=window_seconds,
        tres=tres,
        smoothing_tres=smoothing_tres,
    )

    diagnostics: Dict[str, Any] = {
        "requested_window_frames": window_frames,
        "window_seconds_requested": window_seconds,
        "tres": tres,
        "smoothing_tres": smoothing_tres,
    }

    if resolved_window is None:
        smoothed = raw.copy()
        diagnostics["applied"] = False
        diagnostics["reason"] = "trajectory shorter than minimum Savitzky-Golay window"
        residuals = raw - smoothed
        return SmoothedTrajectory(
            source=trajectory,
            smoothed=smoothed,
            time_seconds=trajectory.time_seconds.copy(),
            method="savitzky_golay",
            window_seconds=_effective_window_seconds(
                len(raw),
                delta_t=trajectory.delta_t,
                window_seconds=window_seconds,
                tres=tres,
                smoothing_tres=smoothing_tres,
            ),
            window_frames=len(raw),
            poly_order=poly_order,
            residuals=residuals,
            diagnostics=diagnostics,
        )

    smoothed = savgol_filter(raw, window_length=resolved_window, polyorder=poly_order, axis=0, mode=mode)
    residuals = raw - smoothed
    diagnostics["applied"] = True
    diagnostics["mode"] = mode

    return SmoothedTrajectory(
        source=trajectory,
        smoothed=smoothed,
        time_seconds=trajectory.time_seconds.copy(),
        method="savitzky_golay",
        window_seconds=_effective_window_seconds(
            resolved_window,
            delta_t=trajectory.delta_t,
            window_seconds=window_seconds,
            tres=tres,
            smoothing_tres=smoothing_tres,
        ),
        window_frames=resolved_window,
        poly_order=poly_order,
        residuals=residuals,
        diagnostics=diagnostics,
    )


def smooth_dataset(
    dataset: TrajectoryDataset,
    *,
    poly_order: int = 2,
    window_frames: int = 5,
    window_seconds: Optional[float] = None,
    tres: Optional[float] = None,
    smoothing_tres: Optional[float] = None,
    mode: str = "interp",
) -> SmoothedTrajectoryDataset:
    """Smooth every trajectory in a loaded dataset."""

    smoothed = [
        smooth_trajectory(
            trajectory,
            poly_order=poly_order,
            window_frames=window_frames,
            window_seconds=window_seconds,
            tres=tres,
            smoothing_tres=smoothing_tres,
            mode=mode,
        )
        for trajectory in dataset.trajectories
    ]
    return SmoothedTrajectoryDataset(trajectories=smoothed, source_dataset=dataset)"""Savitzky-Golay smoothing utilities for beta particle prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .loading import EmbryoTrajectory, TrajectoryDataset

try:
    from scipy.signal import savgol_filter
except ImportError:  # pragma: no cover - exercised only when scipy is absent.
    savgol_filter = None


@dataclass(frozen=True)
class SmoothedTrajectory:
    """One smoothed trajectory with lineage back to a raw embryo track."""

    source: EmbryoTrajectory
    smoothed: np.ndarray
    time_seconds: np.ndarray
    method: str
    window_seconds: float
    window_frames: int
    poly_order: int
    residuals: Optional[np.ndarray] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SmoothedTrajectoryDataset:
    """Collection wrapper for smoothed trajectories."""

    trajectories: List[SmoothedTrajectory]
    source_dataset: TrajectoryDataset

    def __len__(self) -> int:
        return len(self.trajectories)

    @property
    def class_names(self) -> List[str]:
        return self.source_dataset.class_names


def _odd_round(value: float) -> int:
    rounded = max(1, int(round(value)))
    return rounded if rounded % 2 == 1 else rounded + 1


def _largest_valid_odd(trajectory_length: int, minimum: int) -> Optional[int]:
    if trajectory_length < minimum:
        return None
    candidate = trajectory_length if trajectory_length % 2 == 1 else trajectory_length - 1
    if candidate < minimum:
        return None
    return candidate


def resolve_window_frames(
    trajectory_length: int,
    *,
    poly_order: int = 2,
    window_frames: int = 5,
    delta_t: Optional[float] = None,
    window_seconds: Optional[float] = None,
    tres: Optional[float] = None,
    smoothing_tres: Optional[float] = None,
) -> Optional[int]:
    """Resolve the SG window length in frames.

    Precedence:
    1. `tres` and `smoothing_tres`
    2. `window_seconds` and `delta_t`
    3. `window_frames`
    """

    minimum = poly_order + 1
    if minimum % 2 == 0:
        minimum += 1

    if tres is not None and smoothing_tres is not None and tres > 0:
        raw_frames = smoothing_tres / tres
    elif window_seconds is not None and delta_t is not None and np.isfinite(delta_t) and delta_t > 0:
        raw_frames = window_seconds / delta_t
    else:
        raw_frames = float(window_frames)

    resolved = max(_odd_round(raw_frames), minimum)
    max_valid = _largest_valid_odd(trajectory_length, minimum)
    if max_valid is None:
        return None
    return min(resolved, max_valid)


def _effective_window_seconds(
    resolved_window_frames: int,
    *,
    delta_t: float,
    window_seconds: Optional[float],
    tres: Optional[float],
    smoothing_tres: Optional[float],
) -> float:
    if smoothing_tres is not None:
        return float(smoothing_tres)
    if window_seconds is not None:
        return float(window_seconds)
    if np.isfinite(delta_t):
        return float(resolved_window_frames * delta_t)
    if tres is not None:
        return float(resolved_window_frames * tres)
    return float(resolved_window_frames)


def smooth_trajectory(
    trajectory: EmbryoTrajectory,
    *,
    poly_order: int = 2,
    window_frames: int = 5,
    window_seconds: Optional[float] = None,
    tres: Optional[float] = None,
    smoothing_tres: Optional[float] = None,
    mode: str = "interp",
) -> SmoothedTrajectory:
    """Smooth one embryo trajectory with a Savitzky-Golay filter."""

    if savgol_filter is None:
        raise ImportError("scipy is required for Savitzky-Golay smoothing")

    raw = trajectory.trajectory
    resolved_window = resolve_window_frames(
        len(raw),
        poly_order=poly_order,
        window_frames=window_frames,
        delta_t=trajectory.delta_t,
        window_seconds=window_seconds,
        tres=tres,
        smoothing_tres=smoothing_tres,
    )

    diagnostics: Dict[str, Any] = {
        "requested_window_frames": window_frames,
        "window_seconds_requested": window_seconds,
        "tres": tres,
        "smoothing_tres": smoothing_tres,
    }

    if resolved_window is None:
        smoothed = raw.copy()
        diagnostics["applied"] = False
        diagnostics["reason"] = "trajectory shorter than minimum Savitzky-Golay window"
        residuals = raw - smoothed
        return SmoothedTrajectory(
            source=trajectory,
            smoothed=smoothed,
            time_seconds=trajectory.time_seconds.copy(),
            method="savitzky_golay",
            window_seconds=_effective_window_seconds(
                len(raw),
                delta_t=trajectory.delta_t,
                window_seconds=window_seconds,
                tres=tres,
                smoothing_tres=smoothing_tres,
            ),
            window_frames=len(raw),
            poly_order=poly_order,
            residuals=residuals,
            diagnostics=diagnostics,
        )

    smoothed = savgol_filter(raw, window_length=resolved_window, polyorder=poly_order, axis=0, mode=mode)
    residuals = raw - smoothed
    diagnostics["applied"] = True
    diagnostics["mode"] = mode

    return SmoothedTrajectory(
        source=trajectory,
        smoothed=smoothed,
        time_seconds=trajectory.time_seconds.copy(),
        method="savitzky_golay",
        window_seconds=_effective_window_seconds(
            resolved_window,
            delta_t=trajectory.delta_t,
            window_seconds=window_seconds,
            tres=tres,
            smoothing_tres=smoothing_tres,
        ),
        window_frames=resolved_window,
        poly_order=poly_order,
        residuals=residuals,
        diagnostics=diagnostics,
    )


def smooth_dataset(
    dataset: TrajectoryDataset,
    *,
    poly_order: int = 2,
    window_frames: int = 5,
    window_seconds: Optional[float] = None,
    tres: Optional[float] = None,
    smoothing_tres: Optional[float] = None,
    mode: str = "interp",
) -> SmoothedTrajectoryDataset:
    """Smooth every trajectory in a loaded dataset."""

    smoothed = [
        smooth_trajectory(
            trajectory,
            poly_order=poly_order,
            window_frames=window_frames,
            window_seconds=window_seconds,
            tres=tres,
            smoothing_tres=smoothing_tres,
            mode=mode,
        )
        for trajectory in dataset.trajectories
    ]
    return SmoothedTrajectoryDataset(trajectories=smoothed, source_dataset=dataset)"""Savitzky-Golay smoothing utilities for beta particle prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .loading import EmbryoTrajectory, TrajectoryDataset

try:
    from scipy.signal import savgol_filter
except ImportError:  # pragma: no cover - exercised only when scipy is absent.
    savgol_filter = None


@dataclass(frozen=True)
class SmoothedTrajectory:
    """One smoothed trajectory with lineage back to a raw embryo track."""

    source: EmbryoTrajectory
    smoothed: np.ndarray
    time_seconds: np.ndarray
    method: str
    window_seconds: float
    window_frames: int
    poly_order: int
    residuals: Optional[np.ndarray] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SmoothedTrajectoryDataset:
    """Collection wrapper for smoothed trajectories."""

    trajectories: List[SmoothedTrajectory]
    source_dataset: TrajectoryDataset

    def __len__(self) -> int:
        return len(self.trajectories)

    @property
    def class_names(self) -> List[str]:
        return self.source_dataset.class_names


def _odd_round(value: float) -> int:
    rounded = max(1, int(round(value)))
    return rounded if rounded % 2 == 1 else rounded + 1


def _largest_valid_odd(trajectory_length: int, minimum: int) -> Optional[int]:
    if trajectory_length < minimum:
        return None
    candidate = trajectory_length if trajectory_length % 2 == 1 else trajectory_length - 1
    if candidate < minimum:
        return None
    return candidate


def resolve_window_frames(
    trajectory_length: int,
    *,
    poly_order: int = 2,
    window_frames: int = 5,
    delta_t: Optional[float] = None,
    window_seconds: Optional[float] = None,
    tres: Optional[float] = None,
    smoothing_tres: Optional[float] = None,
) -> Optional[int]:
    """Resolve the SG window length in frames.

    Precedence:
    1. `tres` and `smoothing_tres`
    2. `window_seconds` and `delta_t`
    3. `window_frames`
    """

    minimum = poly_order + 1
    if minimum % 2 == 0:
        minimum += 1

    if tres is not None and smoothing_tres is not None and tres > 0:
        raw_frames = smoothing_tres / tres
    elif window_seconds is not None and delta_t is not None and np.isfinite(delta_t) and delta_t > 0:
        raw_frames = window_seconds / delta_t
    else:
        raw_frames = float(window_frames)

    resolved = max(_odd_round(raw_frames), minimum)
    max_valid = _largest_valid_odd(trajectory_length, minimum)
    if max_valid is None:
        return None
    return min(resolved, max_valid)


def _effective_window_seconds(
    resolved_window_frames: int,
    *,
    delta_t: float,
    window_seconds: Optional[float],
    tres: Optional[float],
    smoothing_tres: Optional[float],
) -> float:
    if smoothing_tres is not None:
        return float(smoothing_tres)
    if window_seconds is not None:
        return float(window_seconds)
    if np.isfinite(delta_t):
        return float(resolved_window_frames * delta_t)
    if tres is not None:
        return float(resolved_window_frames * tres)
    return float(resolved_window_frames)


def smooth_trajectory(
    trajectory: EmbryoTrajectory,
    *,
    poly_order: int = 2,
    window_frames: int = 5,
    window_seconds: Optional[float] = None,
    tres: Optional[float] = None,
    smoothing_tres: Optional[float] = None,
    mode: str = "interp",
) -> SmoothedTrajectory:
    """Smooth one embryo trajectory with a Savitzky-Golay filter."""

    if savgol_filter is None:
        raise ImportError("scipy is required for Savitzky-Golay smoothing")

    raw = trajectory.trajectory
    resolved_window = resolve_window_frames(
        len(raw),
        poly_order=poly_order,
        window_frames=window_frames,
        delta_t=trajectory.delta_t,
        window_seconds=window_seconds,
        tres=tres,
        smoothing_tres=smoothing_tres,
    )

    diagnostics: Dict[str, Any] = {
        "requested_window_frames": window_frames,
        "window_seconds_requested": window_seconds,
        "tres": tres,
        "smoothing_tres": smoothing_tres,
    }

    if resolved_window is None:
        smoothed = raw.copy()
        diagnostics["applied"] = False
        diagnostics["reason"] = "trajectory shorter than minimum Savitzky-Golay window"
        residuals = raw - smoothed
        return SmoothedTrajectory(
            source=trajectory,
            smoothed=smoothed,
            time_seconds=trajectory.time_seconds.copy(),
            method="savitzky_golay",
            window_seconds=_effective_window_seconds(
                len(raw),
                delta_t=trajectory.delta_t,
                window_seconds=window_seconds,
                tres=tres,
                smoothing_tres=smoothing_tres,
            ),
            window_frames=len(raw),
            poly_order=poly_order,
            residuals=residuals,
            diagnostics=diagnostics,
        )

    smoothed = savgol_filter(raw, window_length=resolved_window, polyorder=poly_order, axis=0, mode=mode)
    residuals = raw - smoothed
    diagnostics["applied"] = True
    diagnostics["mode"] = mode

    return SmoothedTrajectory(
        source=trajectory,
        smoothed=smoothed,
        time_seconds=trajectory.time_seconds.copy(),
        method="savitzky_golay",
        window_seconds=_effective_window_seconds(
            resolved_window,
            delta_t=trajectory.delta_t,
            window_seconds=window_seconds,
            tres=tres,
            smoothing_tres=smoothing_tres,
        ),
        window_frames=resolved_window,
        poly_order=poly_order,
        residuals=residuals,
        diagnostics=diagnostics,
    )


def smooth_dataset(
    dataset: TrajectoryDataset,
    *,
    poly_order: int = 2,
    window_frames: int = 5,
    window_seconds: Optional[float] = None,
    tres: Optional[float] = None,
    smoothing_tres: Optional[float] = None,
    mode: str = "interp",
) -> SmoothedTrajectoryDataset:
    """Smooth every trajectory in a loaded dataset."""

    smoothed = [
        smooth_trajectory(
            trajectory,
            poly_order=poly_order,
            window_frames=window_frames,
            window_seconds=window_seconds,
            tres=tres,
            smoothing_tres=smoothing_tres,
            mode=mode,
        )
        for trajectory in dataset.trajectories
    ]
    return SmoothedTrajectoryDataset(trajectories=smoothed, source_dataset=dataset)


def resolve_window_frames(
    trajectory_length: int,
    poly_order: int = DEFAULT_POLY_ORDER,
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    tres: Optional[float] = None,
    smoothing_tres: Optional[float] = None,
) -> int:
    """Resolve the effective odd SG window length for one trajectory.

    If both ``tres`` and ``smoothing_tres`` are provided, the requested window is
    derived from their ratio. Otherwise the integer ``window_frames`` parameter is used.
    """

    if (tres is None) != (smoothing_tres is None):
        raise ValueError("tres and smoothing_tres must both be provided or both be None")

    requested = (
        smoothing_tres / tres
        if tres is not None and smoothing_tres is not None
        else float(window_frames)
    )
    candidate = _round_to_odd(requested)
    minimum_valid = poly_order + 1
    if minimum_valid % 2 == 0:
        minimum_valid += 1
    candidate = max(candidate, minimum_valid)

    if trajectory_length < minimum_valid:
        return 0

    if candidate > trajectory_length:
        candidate = trajectory_length if trajectory_length % 2 == 1 else trajectory_length - 1

    if candidate <= poly_order or candidate < 3:
        return 0
    return candidate


def smooth_trajectory(
    trajectory: EmbryoTrajectory,
    poly_order: int = DEFAULT_POLY_ORDER,
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    tres: Optional[float] = None,
    smoothing_tres: Optional[float] = None,
) -> SmoothedTrajectory:
    """Apply SG smoothing independently to each latent dimension."""

    effective_window_frames = resolve_window_frames(
        trajectory_length=len(trajectory.trajectory),
        poly_order=poly_order,
        window_frames=window_frames,
        tres=tres,
        smoothing_tres=smoothing_tres,
    )

    diagnostics: Dict[str, Any] = {
        "requested_window_frames": (
            smoothing_tres / tres if tres is not None and smoothing_tres is not None else float(window_frames)
        ),
        "effective_window_frames": effective_window_frames,
        "tres": tres,
        "smoothing_tres": smoothing_tres,
        "skipped": False,
    }

    if effective_window_frames == 0:
        diagnostics["skipped"] = True
        diagnostics["reason"] = "trajectory too short for requested SG configuration"
        copied = trajectory.trajectory.copy()
        return SmoothedTrajectory(
            source=trajectory,
            smoothed=copied,
            time_seconds=trajectory.time_seconds.copy(),
            method="savitzky_golay",
            window_seconds=smoothing_tres,
            window_frames=0,
            poly_order=poly_order,
            residuals=np.zeros_like(copied),
            diagnostics=diagnostics,
        )

    smoothed = savgol_filter(
        trajectory.trajectory,
        window_length=effective_window_frames,
        polyorder=poly_order,
        axis=0,
        mode="interp",
    )
    residuals = trajectory.trajectory - smoothed
    diagnostics["residual_l2_per_frame"] = np.linalg.norm(residuals, axis=1)

    return SmoothedTrajectory(
        source=trajectory,
        smoothed=smoothed,
        time_seconds=trajectory.time_seconds.copy(),
        method="savitzky_golay",
        window_seconds=smoothing_tres,
        window_frames=effective_window_frames,
        poly_order=poly_order,
        residuals=residuals,
        diagnostics=diagnostics,
    )


def smooth_dataset(
    trajectories: Sequence[EmbryoTrajectory] | Iterable[EmbryoTrajectory],
    poly_order: int = DEFAULT_POLY_ORDER,
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    tres: Optional[float] = None,
    smoothing_tres: Optional[float] = None,
) -> List[SmoothedTrajectory]:
    """Smooth every trajectory in a collection."""

    return [
        smooth_trajectory(
            trajectory=trajectory,
            poly_order=poly_order,
            window_frames=window_frames,
            tres=tres,
            smoothing_tres=smoothing_tres,
        )
        for trajectory in trajectories
    ]