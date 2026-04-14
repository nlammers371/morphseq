from __future__ import annotations

import numpy as np

from dev.particle_prediction.data.loading import EmbryoTrajectory
from dev.particle_prediction.data.smoothing import smooth_trajectory


def test_smooth_trajectory_smoke() -> None:
    time_seconds = np.arange(9, dtype=np.float64) * 60.0
    true_signal = np.column_stack(
        [
            0.05 * (np.arange(9, dtype=np.float64) - 4.0) ** 2,
            0.3 * np.arange(9, dtype=np.float64) + 1.0,
        ]
    )
    noisy_signal = true_signal + np.array(
        [
            [0.00, 0.00],
            [0.10, -0.08],
            [-0.12, 0.07],
            [0.08, -0.05],
            [0.00, 0.06],
            [-0.07, -0.04],
            [0.09, 0.05],
            [-0.10, -0.06],
            [0.00, 0.00],
        ],
        dtype=np.float64,
    )

    trajectory = EmbryoTrajectory(
        embryo_id="emb_a",
        trajectory=noisy_signal,
        time_seconds=time_seconds,
        delta_t=60.0,
        temperature=28.5,
        perturbation_class="wt",
        experiment_id="exp01",
    )

    smoothed = smooth_trajectory(trajectory, window_seconds=5.0 * 60.0, poly_order=2)

    assert smoothed.source is trajectory
    assert smoothed.method == "savitzky_golay"
    assert smoothed.window_frames == 5
    assert smoothed.smoothed.shape == noisy_signal.shape
    assert smoothed.residuals is not None
    assert smoothed.residuals.shape == noisy_signal.shape
    assert np.array_equal(smoothed.time_seconds, time_seconds)
    assert np.array_equal(trajectory.trajectory, noisy_signal)

    noisy_mse = np.mean((noisy_signal - true_signal) ** 2)
    smoothed_mse = np.mean((smoothed.smoothed - true_signal) ** 2)
    assert smoothed_mse < noisy_mse