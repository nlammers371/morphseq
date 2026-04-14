from __future__ import annotations

import numpy as np

from dev.particle_prediction.data.dataset import (
    build_prediction_query,
    build_prediction_tasks,
    build_query_from_resampled_trajectory,
    summarize_prediction_tasks,
)
from dev.particle_prediction.data.loading import EmbryoTrajectory
from dev.particle_prediction.data.resampling import ResampledTrajectory
from dev.particle_prediction.data.smoothing import SmoothedTrajectory


def _make_resampled_trajectory() -> ResampledTrajectory:
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 1.0],
            [4.0, 1.5],
            [5.0, 2.0],
        ],
        dtype=np.float64,
    )
    time_seconds = np.arange(len(points), dtype=np.float64) * 10.0
    source = EmbryoTrajectory(
        embryo_id='emb_a',
        trajectory=points.copy(),
        time_seconds=time_seconds.copy(),
        delta_t=10.0,
        temperature=28.5,
        perturbation_class='wt',
        experiment_id='exp01',
        frame_index=np.arange(len(points), dtype=np.int64),
    )
    smoothed = SmoothedTrajectory(
        source=source,
        smoothed=points.copy(),
        time_seconds=time_seconds.copy(),
        method='savitzky_golay',
        window_seconds=30.0,
        window_frames=3,
        poly_order=1,
        residuals=np.zeros_like(points),
    )
    return ResampledTrajectory(
        source=source,
        smoothed_source=smoothed,
        resampled=points.copy(),
        arc_length=np.array([0.0, 1.0, 2.0, 3.4142, 4.5322, 5.6503]),
        delta_s=1.0,
        source_time_interp=time_seconds.copy(),
        source_frame_interp=np.arange(len(points), dtype=np.float64),
        increment_norms=np.linalg.norm(np.diff(points, axis=0), axis=1),
    )


def test_prediction_query_and_task_building_smoke() -> None:
    trajectory = _make_resampled_trajectory()

    snapshot_query = build_prediction_query(current_state=trajectory.resampled[2])
    assert snapshot_query.mode == 'snapshot'
    assert snapshot_query.history_segments is None

    history_query = build_query_from_resampled_trajectory(trajectory=trajectory, state_index=3, history_length=2)
    assert history_query.mode == 'history'
    assert history_query.history_segments is not None
    assert history_query.history_segments.shape == (2, 2)
    assert history_query.recent_points is not None
    assert history_query.metadata['embryo_id'] == 'emb_a'

    tasks = build_prediction_tasks([trajectory], history_length=2, horizons=[1, 2], mode='history')
    assert len(tasks) == 2
    assert tasks[0].target_states.shape == (2, 2)
    assert np.array_equal(tasks[0].horizons, np.array([1, 2]))

    summary = summarize_prediction_tasks(tasks)
    assert summary == {
        'n_tasks': 2,
        'n_snapshot_queries': 0,
        'n_history_queries': 2,
        'n_embryos': 1,
    }
