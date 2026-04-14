from __future__ import annotations

import numpy as np

from dev.particle_prediction.data.dataset import build_prediction_tasks
from dev.particle_prediction.data.loading import EmbryoTrajectory
from dev.particle_prediction.data.resampling import ResampledTrajectory
from dev.particle_prediction.data.smoothing import SmoothedTrajectory
from dev.particle_prediction.data.transition_bank import build_transition_bank
from dev.particle_prediction.eval.evaluate import (
    comparison_table,
    evaluate_linear_extrapolation_baseline,
    evaluate_persistence_baseline,
    run_evaluation_suite,
)
from dev.particle_prediction.models.matching import MatchingConfig


def _make_resampled_trajectory(embryo_id: str, points: np.ndarray, perturbation_class: str = "wt") -> ResampledTrajectory:
    time_seconds = np.arange(len(points), dtype=np.float64) * 10.0
    source = EmbryoTrajectory(
        embryo_id=embryo_id,
        trajectory=points.copy(),
        time_seconds=time_seconds.copy(),
        delta_t=10.0,
        temperature=28.5,
        perturbation_class=perturbation_class,
        experiment_id="exp01",
        frame_index=np.arange(len(points), dtype=np.int64),
        observed_dts=np.diff(time_seconds),
        missing_frame_counts=np.zeros(len(points) - 1, dtype=np.int64),
        interpolatable_gap_mask=np.zeros(len(points) - 1, dtype=bool),
        hard_gap_mask=np.zeros(len(points) - 1, dtype=bool),
        segment_ids=np.zeros(len(points), dtype=np.int64),
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
    return ResampledTrajectory(
        source=source,
        smoothed_source=smoothed,
        resampled=points.copy(),
        arc_length=np.arange(len(points), dtype=np.float64),
        delta_s=1.0,
        source_time_interp=time_seconds.copy(),
        source_frame_interp=np.arange(len(points), dtype=np.float64),
        increment_norms=np.linalg.norm(np.diff(points, axis=0), axis=1),
    )


def test_evaluation_suite_smoke() -> None:
    trajectory_a = _make_resampled_trajectory(
        embryo_id="emb_a",
        points=np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.5], [4., 1.0], [5., 1.5], [6., 2.0], [7., 2.5], [8., 3.0]]),
        perturbation_class="wt",
    )
    trajectory_b = _make_resampled_trajectory(
        embryo_id="emb_b",
        points=np.array([[0., 0.], [0., 1.], [0., 2.], [1., 2.], [2., 2.], [3., 2.], [4., 2.], [5., 2.], [6., 2.]]),
        perturbation_class="mut",
    )
    tasks = build_prediction_tasks([trajectory_a, trajectory_b], history_length=3, horizons=[1, 2, 3], mode="history")
    bank = build_transition_bank([trajectory_a, trajectory_b], history_length=5, use_state_index=True)

    persistence = evaluate_persistence_baseline(tasks)
    linear = evaluate_linear_extrapolation_baseline(tasks)
    suite = run_evaluation_suite(
        tasks=tasks,
        bank=bank,
        n_particles=24,
        random_seed=7,
        matching_config=MatchingConfig(k_state=8, retrieval_method="brute"),
    )

    assert persistence.model_name == "persistence"
    assert linear.model_name == "linear_extrapolation"
    assert set(suite) == {"persistence", "linear_extrapolation", "local_no_history", "local_history", "local_fast_summary"}

    local_history = suite["local_history"]
    assert local_history.horizons.tolist() == [1, 2, 3]
    assert len(local_history.task_results) == len(tasks)
    assert local_history.task_results[0].predicted_mean.shape == (3, 2)
    assert local_history.task_results[0].forward_samples.shape[0] == 3
    assert len(local_history.endpoint_error_summary) == 3
    assert local_history.summary_metrics["n_tasks"] == len(tasks)
    assert "candidate_count" in local_history.support_summaries

    rows = comparison_table(suite)
    assert len(rows) == 5
    assert {row["model_name"] for row in rows} == set(suite)
