from __future__ import annotations

import numpy as np

from dev.particle_prediction.data.loading import EmbryoTrajectory
from dev.particle_prediction.data.resampling import compute_cumulative_arc_length, resample_smoothed_trajectory
from dev.particle_prediction.data.smoothing import SmoothedTrajectory


def test_resample_smoothed_trajectory_smoke() -> None:
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
    time_seconds = np.array([0.0, 60.0, 120.0, 180.0, 240.0], dtype=np.float64)

    source = EmbryoTrajectory(
        embryo_id="emb_a",
        trajectory=points.copy(),
        time_seconds=time_seconds,
        delta_t=60.0,
        temperature=28.5,
        perturbation_class="wt",
        experiment_id="exp01",
        frame_index=np.arange(len(points), dtype=np.int64),
    )
    smoothed = SmoothedTrajectory(
        source=source,
        smoothed=points.copy(),
        time_seconds=time_seconds.copy(),
        method="savitzky_golay",
        window_seconds=180.0,
        window_frames=3,
        poly_order=1,
        residuals=np.zeros_like(points),
    )

    cumulative_arc_length = compute_cumulative_arc_length(smoothed.smoothed)
    assert np.allclose(cumulative_arc_length, np.array([0.0, 1.0, 2.0, 3.0, 4.0]))

    resampled = resample_smoothed_trajectory(smoothed, delta_s=0.5)

    assert resampled.source is source
    assert resampled.smoothed_source is smoothed
    assert np.isclose(resampled.delta_s, 0.5)
    assert np.allclose(resampled.arc_length, np.arange(0.0, 4.0 + 0.5, 0.5))
    assert np.allclose(resampled.resampled[:, 0], resampled.arc_length)
    assert np.allclose(resampled.resampled[:, 1], 0.0)
    assert np.allclose(resampled.source_time_interp, np.arange(0.0, 240.0 + 30.0, 30.0))
    assert resampled.source_frame_interp is not None
    assert np.allclose(resampled.source_frame_interp, np.arange(0.0, 4.0 + 0.5, 0.5))
    assert resampled.increment_norms is not None
    assert np.allclose(resampled.increment_norms, 0.5)
