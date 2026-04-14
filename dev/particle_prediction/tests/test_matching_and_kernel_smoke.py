from __future__ import annotations

import numpy as np

from dev.particle_prediction.data.dataset import build_prediction_query
from dev.particle_prediction.data.loading import EmbryoTrajectory
from dev.particle_prediction.data.resampling import ResampledTrajectory
from dev.particle_prediction.data.smoothing import SmoothedTrajectory
from dev.particle_prediction.data.transition_bank import build_transition_bank
from dev.particle_prediction.models.kernels import construct_tangent_aligned_covariance
from dev.particle_prediction.models.local_transition_pf import LocalTransitionPredictor
from dev.particle_prediction.models.matching import compare_matching_modes


def _make_resampled_trajectory(
    embryo_id: str,
    points: np.ndarray,
    perturbation_class: str,
) -> ResampledTrajectory:
    time_seconds = np.arange(len(points), dtype=np.float64) * 10.0
    source = EmbryoTrajectory(
        embryo_id=embryo_id,
        trajectory=points.copy(),
        time_seconds=time_seconds.copy(),
        delta_t=10.0,
        temperature=28.5,
        perturbation_class=perturbation_class,
        experiment_id='exp01',
        metadata={'source': embryo_id},
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


def test_fast_matching_and_local_kernel_smoke() -> None:
    trajectory_a = _make_resampled_trajectory(
        embryo_id='emb_a',
        points=np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.], [4., 0.], [5., 0.], [6., 0.], [7., 0.], [8., 0.]]),
        perturbation_class='wt',
    )
    trajectory_b = _make_resampled_trajectory(
        embryo_id='emb_b',
        points=np.array([[0., 0.], [0., 1.], [0., 2.], [1., 2.], [2., 2.], [3., 2.], [4., 2.], [5., 2.], [6., 2.]]),
        perturbation_class='mut',
    )
    bank = build_transition_bank([trajectory_a, trajectory_b], history_length=7, use_state_index=True)

    query = build_prediction_query(
        current_state=bank.windows[0].state,
        history_segments=bank.windows[0].history_segments,
    )
    matches = compare_matching_modes(bank, query.current_state, query.history_segments)
    assert set(matches) == {'default', 'fast_summary'}
    assert np.isclose(np.sum(matches['default'].normalized_weights), 1.0)
    assert np.isclose(np.sum(matches['fast_summary'].normalized_weights), 1.0)

    predictor = LocalTransitionPredictor(bank=bank, sigma_parallel=0.05, sigma_perp=0.08, jitter_mode='tangent')
    result = predictor.predict_query(query, n_samples=64, rng=np.random.default_rng(7))
    assert result.forward_samples.shape == (64, 2)
    assert result.predicted_mean.shape == (2,)
    assert result.predicted_cov_diag.shape == (2,)
    assert result.candidate_count == len(result.match_result.candidate_indices)
    assert result.effective_sample_size > 0
    assert np.all(result.predicted_cov_diag >= 0)

    covariance = construct_tangent_aligned_covariance(np.array([1.0, 0.0]), sigma_parallel=0.05, sigma_perp=0.08)
    eigvals = np.linalg.eigvalsh(covariance)
    assert np.all(eigvals > 0)
