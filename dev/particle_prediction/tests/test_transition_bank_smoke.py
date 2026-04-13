from __future__ import annotations

import numpy as np

from dev.particle_prediction.data.loading import EmbryoTrajectory
from dev.particle_prediction.data.resampling import ResampledTrajectory
from dev.particle_prediction.data.smoothing import SmoothedTrajectory
from dev.particle_prediction.data.transition_bank import (
    DEFAULT_ALPHA,
    build_transition_bank,
    match_query_to_bank,
    retrieve_state_candidates,
)


def _make_resampled_trajectory(
    embryo_id: str,
    points: np.ndarray,
    source_times: np.ndarray,
    source_time_interp: np.ndarray,
    segment_ids: np.ndarray,
    interpolatable_gap_mask: np.ndarray,
    hard_gap_mask: np.ndarray,
) -> ResampledTrajectory:
    source = EmbryoTrajectory(
        embryo_id=embryo_id,
        trajectory=points[: len(source_times)].copy(),
        time_seconds=source_times.copy(),
        delta_t=10.0,
        temperature=28.5,
        perturbation_class="wt" if embryo_id != "emb_b" else "mut",
        experiment_id="exp01",
        metadata={"source": embryo_id},
        frame_index=np.arange(len(source_times), dtype=np.int64),
        observed_dts=np.diff(source_times),
        missing_frame_counts=np.maximum(np.rint(np.diff(source_times) / 10.0).astype(np.int64) - 1, 0),
        interpolatable_gap_mask=interpolatable_gap_mask,
        hard_gap_mask=hard_gap_mask,
        segment_ids=segment_ids,
    )
    smoothed = SmoothedTrajectory(
        source=source,
        smoothed=points[: len(source_times)].copy(),
        time_seconds=source_times.copy(),
        method="savitzky_golay",
        window_seconds=30.0,
        window_frames=3,
        poly_order=1,
        residuals=np.zeros((len(source_times), points.shape[1]), dtype=np.float64),
    )
    return ResampledTrajectory(
        source=source,
        smoothed_source=smoothed,
        resampled=points.copy(),
        arc_length=np.arange(len(points), dtype=np.float64),
        delta_s=1.0,
        source_time_interp=source_time_interp.copy(),
        source_frame_interp=np.linspace(0, len(source_times) - 1, len(points), dtype=np.float64),
        increment_norms=np.linalg.norm(np.diff(points, axis=0), axis=1),
    )


def test_transition_bank_matching_smoke() -> None:
    trajectory_a = _make_resampled_trajectory(
        embryo_id="emb_a",
        points=np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.], [4., 0.], [5., 0.], [6., 0.]]),
        source_times=np.array([0., 10., 20., 40., 50., 60., 70.]),
        source_time_interp=np.array([0., 10., 20., 30., 40., 50., 60.]),
        segment_ids=np.zeros(7, dtype=np.int64),
        interpolatable_gap_mask=np.array([False, False, True, False, False, False]),
        hard_gap_mask=np.zeros(6, dtype=bool),
    )
    trajectory_b = _make_resampled_trajectory(
        embryo_id="emb_b",
        points=np.array([[0., 0.], [0., 1.], [0., 2.], [1., 2.], [2., 2.], [5., 0.1], [6., 0.1]]),
        source_times=np.array([0., 10., 20., 30., 40., 50., 60.]),
        source_time_interp=np.array([0., 10., 20., 30., 40., 50., 60.]),
        segment_ids=np.zeros(7, dtype=np.int64),
        interpolatable_gap_mask=np.zeros(6, dtype=bool),
        hard_gap_mask=np.zeros(6, dtype=bool),
    )
    trajectory_c = _make_resampled_trajectory(
        embryo_id="emb_c",
        points=np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.], [4., 0.], [5., 0.], [6., 0.]]),
        source_times=np.array([0., 10., 20., 140., 150., 160., 170.]),
        source_time_interp=np.array([0., 10., 20., 60., 100., 140., 150.]),
        segment_ids=np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.int64),
        interpolatable_gap_mask=np.zeros(6, dtype=bool),
        hard_gap_mask=np.array([False, False, True, False, False, False]),
    )

    bank = build_transition_bank([trajectory_a, trajectory_b, trajectory_c], history_length=5, use_state_index=True)

    assert len(bank) == 2
    assert bank.history_tensor.shape == (2, 5, 2)
    assert np.array_equal(bank.segment_ids, np.array([0, 0]))
    assert np.array_equal(bank.touches_interpolated_gap, np.array([True, False]))

    query_window = bank.windows[0]
    query_history = query_window.history_segments[-3:]

    nn_indices, nn_d_state_sq = retrieve_state_candidates(bank, query_window.state, k_state=2, method="nn")
    brute_indices, brute_d_state_sq = retrieve_state_candidates(bank, query_window.state, k_state=2, method="brute")

    assert np.array_equal(nn_indices, brute_indices)
    assert np.allclose(nn_d_state_sq, brute_d_state_sq)

    match_result = match_query_to_bank(
        bank=bank,
        query_state=query_window.state,
        query_history_segments=query_history,
        k_state=2,
        offset_radius=1,
        alpha=DEFAULT_ALPHA,
        sigma_z=1.0,
        sigma_h=1.0,
        lambda_h=1.0,
        retrieval_method="brute",
    )

    assert len(match_result.candidate_indices) == 2
    assert np.isclose(np.sum(match_result.normalized_weights), 1.0)
    assert match_result.candidate_windows[0].embryo_id == "emb_a"
    assert match_result.d_hist_sq[0] <= match_result.d_hist_sq[1]
    assert match_result.scores[0] <= match_result.scores[1]
    assert match_result.normalized_weights[0] > match_result.normalized_weights[1]