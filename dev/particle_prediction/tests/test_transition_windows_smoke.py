from __future__ import annotations

import numpy as np

from dev.particle_prediction.data.loading import EmbryoTrajectory
from dev.particle_prediction.data.resampling import ResampledTrajectory
from dev.particle_prediction.data.smoothing import SmoothedTrajectory
from dev.particle_prediction.data.transition_windows import build_transition_windows


def test_build_transition_windows_smoke() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ],
        dtype=np.float64,
    )
    time_seconds = np.array([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    frame_index = np.arange(len(points), dtype=np.int64)

    source = EmbryoTrajectory(
        embryo_id="emb_a",
        trajectory=points.copy(),
        time_seconds=time_seconds.copy(),
        delta_t=10.0,
        temperature=28.5,
        perturbation_class="wt",
        experiment_id="exp01",
        metadata={"background": "bg1"},
        frame_index=frame_index,
    )
    smoothed = SmoothedTrajectory(
        source=source,
        smoothed=points.copy(),
        time_seconds=time_seconds.copy(),
        method="savitzky_golay",
        window_seconds=30.0,
        window_frames=3,
        poly_order=1,
        residuals=np.zeros_like(points),
    )
    resampled = ResampledTrajectory(
        source=source,
        smoothed_source=smoothed,
        resampled=points.copy(),
        arc_length=np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        delta_s=1.0,
        source_time_interp=time_seconds.copy(),
        source_frame_interp=frame_index.astype(np.float64),
        increment_norms=np.ones(len(points) - 1, dtype=np.float64),
    )

    windows = build_transition_windows(resampled, history_length=2)

    assert len(windows) == 2

    first = windows[0]
    assert np.array_equal(first.state, np.array([2.0, 0.0]))
    assert np.array_equal(first.increment, np.array([1.0, 0.0]))
    assert np.array_equal(first.history_segments, np.array([[1.0, 0.0], [1.0, 0.0]]))
    assert first.embryo_id == "emb_a"
    assert first.experiment_id == "exp01"
    assert first.perturbation_class == "wt"
    assert first.resampled_index == 2
    assert first.arc_length_value == 2.0
    assert first.source_time_estimate == 20.0
    assert first.support_metadata == {"background": "bg1"}

    second = windows[1]
    assert np.array_equal(second.state, np.array([3.0, 0.0]))
    assert np.array_equal(second.increment, np.array([1.0, 0.0]))
    assert np.array_equal(second.history_segments, np.array([[1.0, 0.0], [1.0, 0.0]]))
    assert second.resampled_index == 3
    assert second.arc_length_value == 3.0