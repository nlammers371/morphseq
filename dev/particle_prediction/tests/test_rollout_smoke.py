from __future__ import annotations

import numpy as np

from dev.particle_prediction.data.dataset import build_prediction_query
from dev.particle_prediction.data.loading import EmbryoTrajectory
from dev.particle_prediction.data.resampling import ResampledTrajectory
from dev.particle_prediction.data.smoothing import SmoothedTrajectory
from dev.particle_prediction.data.transition_bank import build_transition_bank
from dev.particle_prediction.models.local_transition_pf import LocalTransitionPredictor
from dev.particle_prediction.models.matching import MatchingConfig


def _make_resampled_trajectory(embryo_id: str, points: np.ndarray, perturbation_class: str = 'wt') -> ResampledTrajectory:
    time_seconds = np.arange(len(points), dtype=np.float64) * 10.0
    source = EmbryoTrajectory(
        embryo_id=embryo_id,
        trajectory=points.copy(),
        time_seconds=time_seconds.copy(),
        delta_t=10.0,
        temperature=28.5,
        perturbation_class=perturbation_class,
        experiment_id='exp01',
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
        arc_length=np.arange(len(points), dtype=np.float64),
        delta_s=1.0,
        source_time_interp=time_seconds.copy(),
        source_frame_interp=np.arange(len(points), dtype=np.float64),
        increment_norms=np.linalg.norm(np.diff(points, axis=0), axis=1),
    )


def test_snapshot_rollout_smoke() -> None:
    trajectory = _make_resampled_trajectory(
        embryo_id='emb_a',
        points=np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.], [4., 0.], [5., 0.], [6., 0.], [7., 0.], [8., 0.]]),
    )
    bank = build_transition_bank([trajectory], history_length=5, use_state_index=True)
    predictor = LocalTransitionPredictor(
        bank=bank,
        matching_config=MatchingConfig(k_state=1, retrieval_method='brute'),
        sigma_parallel=0.0,
        sigma_perp=0.0,
        jitter_mode='none',
    )
    query = build_prediction_query(current_state=np.array([2.0, 0.0]))
    rollout = predictor.rollout_query(query, n_steps=3, n_particles=4, rng=np.random.default_rng(11))

    assert rollout.predicted_mean.shape == (3, 2)
    assert rollout.predicted_cov_diag.shape == (3, 2)
    assert rollout.forward_samples.shape == (3, 4, 2)
    assert len(rollout.step_diagnostics) == 3
    assert np.allclose(rollout.forward_samples[:, :, 0], np.array([[3.0], [4.0], [5.0]]))
    assert np.allclose(rollout.forward_samples[:, :, 1], 0.0)
    assert np.allclose(rollout.predicted_cov_diag, 0.0)


def test_history_rollout_support_diagnostics_smoke() -> None:
    trajectory_a = _make_resampled_trajectory(
        embryo_id='emb_a',
        points=np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.5], [4., 1.0], [5., 1.5], [6., 2.0], [7., 2.5], [8., 3.0]]),
        perturbation_class='wt',
    )
    trajectory_b = _make_resampled_trajectory(
        embryo_id='emb_b',
        points=np.array([[0., 0.], [0., 1.], [0., 2.], [1., 2.], [2., 2.], [3., 2.], [4., 2.], [5., 2.], [6., 2.]]),
        perturbation_class='mut',
    )
    bank = build_transition_bank([trajectory_a, trajectory_b], history_length=7, use_state_index=True)
    window = bank.windows[0]
    query = build_prediction_query(current_state=window.state, history_segments=window.history_segments)

    predictor = LocalTransitionPredictor(bank=bank, sigma_parallel=0.03, sigma_perp=0.05, jitter_mode='tangent')
    rollout = predictor.rollout_query(query, n_steps=4, n_particles=16, rng=np.random.default_rng(5))

    assert rollout.predicted_mean.shape == (4, 2)
    assert rollout.predicted_cov_diag.shape == (4, 2)
    assert rollout.forward_samples.shape == (4, 16, 2)
    assert len(rollout.step_diagnostics) == 4
    assert np.all(np.isfinite(rollout.predicted_mean))
    assert np.all(rollout.predicted_cov_diag >= 0)
    assert all(step.candidate_count > 0 for step in rollout.step_diagnostics)
    assert all(step.effective_sample_size > 0 for step in rollout.step_diagnostics)
    assert all(np.isfinite(step.history_mismatch) for step in rollout.step_diagnostics)
