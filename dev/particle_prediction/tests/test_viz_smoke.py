from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from dev.particle_prediction.data.dataset import build_prediction_tasks
from dev.particle_prediction.data.loading import EmbryoTrajectory
from dev.particle_prediction.data.resampling import ResampledTrajectory
from dev.particle_prediction.data.smoothing import SmoothedTrajectory, smooth_trajectory
from dev.particle_prediction.data.transition_bank import build_transition_bank
from dev.particle_prediction.data.transition_windows import build_transition_windows
from dev.particle_prediction.eval.evaluate import run_evaluation_suite
from dev.particle_prediction.models.local_transition_pf import LocalTransitionPredictor
from dev.particle_prediction.viz.evaluation import (
    plot_error_vs_horizon,
    plot_error_vs_support,
    plot_failure_gallery,
    plot_model_comparison_table,
)
from dev.particle_prediction.viz.matching import (
    compare_default_vs_fast_matching,
    plot_history_offset_heatmap,
    plot_history_reranking,
    plot_query_and_candidate_neighbors,
)
from dev.particle_prediction.viz.prediction import (
    plot_jitter_ellipse_or_covariance,
    plot_local_increment_cloud,
    plot_prediction_fan,
    plot_rollout_against_truth,
    plot_sampled_next_steps,
    plot_support_diagnostics_along_rollout,
)
from dev.particle_prediction.viz.smoothing import (
    plot_latent_trajectory_before_after_smoothing,
    plot_raw_vs_smoothed_timeseries,
    plot_sg_parameter_sweep,
)
from dev.particle_prediction.viz.transition_bank import (
    plot_arc_length_vs_time,
    plot_bank_state_density,
    plot_history_segments_example,
    plot_increment_norm_distribution,
    plot_resampled_points_on_trajectory,
    plot_transition_windows_for_embryo,
)
from dev.particle_prediction.models.matching import compare_matching_modes


def _make_bank_fixture():
    points = np.array(
        [[0., 0.], [1., 0.], [2., 0.], [3., 0.5], [4., 1.0], [5., 1.5], [6., 2.0], [7., 2.5], [8., 3.0]],
        dtype=np.float64,
    )
    alt_points = np.array(
        [[0., 0.], [0., 1.], [0., 2.], [1., 2.], [2., 2.], [3., 2.], [4., 2.], [5., 2.], [6., 2.]],
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
        observed_dts=np.diff(time_seconds),
        missing_frame_counts=np.zeros(len(points) - 1, dtype=np.int64),
        interpolatable_gap_mask=np.zeros(len(points) - 1, dtype=bool),
        hard_gap_mask=np.zeros(len(points) - 1, dtype=bool),
        segment_ids=np.zeros(len(points), dtype=np.int64),
    )
    source_b = EmbryoTrajectory(
        embryo_id='emb_b',
        trajectory=alt_points.copy(),
        time_seconds=time_seconds.copy(),
        delta_t=10.0,
        temperature=28.5,
        perturbation_class='mut',
        experiment_id='exp01',
        frame_index=np.arange(len(alt_points), dtype=np.int64),
        observed_dts=np.diff(time_seconds),
        missing_frame_counts=np.zeros(len(alt_points) - 1, dtype=np.int64),
        interpolatable_gap_mask=np.zeros(len(alt_points) - 1, dtype=bool),
        hard_gap_mask=np.zeros(len(alt_points) - 1, dtype=bool),
        segment_ids=np.zeros(len(alt_points), dtype=np.int64),
    )
    smoothed = smooth_trajectory(source, window_seconds=30.0, poly_order=1)
    smoothed_b = smooth_trajectory(source_b, window_seconds=30.0, poly_order=1)
    resampled = ResampledTrajectory(
        source=source,
        smoothed_source=smoothed,
        resampled=points.copy(),
        arc_length=np.arange(len(points), dtype=np.float64),
        delta_s=1.0,
        source_time_interp=time_seconds.copy(),
        source_frame_interp=np.arange(len(points), dtype=np.float64),
        increment_norms=np.linalg.norm(np.diff(points, axis=0), axis=1),
    )
    resampled_b = ResampledTrajectory(
        source=source_b,
        smoothed_source=smoothed_b,
        resampled=alt_points.copy(),
        arc_length=np.arange(len(alt_points), dtype=np.float64),
        delta_s=1.0,
        source_time_interp=time_seconds.copy(),
        source_frame_interp=np.arange(len(alt_points), dtype=np.float64),
        increment_norms=np.linalg.norm(np.diff(alt_points, axis=0), axis=1),
    )
    bank = build_transition_bank([resampled, resampled_b], history_length=7, use_state_index=True)
    windows = build_transition_windows(resampled, history_length=3)
    predictor = LocalTransitionPredictor(bank=bank, sigma_parallel=0.05, sigma_perp=0.08, jitter_mode='tangent')
    query = bank.windows[0]
    prediction = predictor.predict_query(
        query=type('Q', (), {
            'mode': 'history',
            'current_state': query.state,
            'history_segments': query.history_segments,
            'class_prior': None,
        })(),
        n_samples=32,
        rng=np.random.default_rng(3),
    )
    rollout = predictor.rollout_query(
        query=type('Q', (), {
            'mode': 'history',
            'current_state': query.state,
            'history_segments': query.history_segments,
            'class_prior': None,
        })(),
        n_steps=3,
        n_particles=24,
        rng=np.random.default_rng(4),
    )
    matches = compare_matching_modes(bank, query.state, query.history_segments)
    tasks = build_prediction_tasks([resampled, resampled_b], history_length=3, horizons=[1, 2, 3], mode='history')
    eval_results = run_evaluation_suite(tasks=tasks, bank=bank, n_particles=16, random_seed=5)
    return source, smoothed, resampled, bank, windows, prediction, rollout, matches, eval_results


def test_visualization_smoke() -> None:
    source, smoothed, resampled, bank, windows, prediction, rollout, matches, eval_results = _make_bank_fixture()
    fig_factories = [
        lambda: plot_raw_vs_smoothed_timeseries(smoothed),
        lambda: plot_latent_trajectory_before_after_smoothing(smoothed),
        lambda: plot_sg_parameter_sweep(source, window_seconds_values=[20.0, 30.0]),
        lambda: plot_arc_length_vs_time(resampled),
        lambda: plot_resampled_points_on_trajectory(resampled),
        lambda: plot_increment_norm_distribution([resampled]),
        lambda: plot_transition_windows_for_embryo(resampled, windows),
        lambda: plot_history_segments_example(windows[0]),
        lambda: plot_bank_state_density(bank),
        lambda: plot_query_and_candidate_neighbors(windows[0].state, matches['default']),
        lambda: plot_history_reranking(matches['default'], matches['fast_summary']),
        lambda: plot_history_offset_heatmap(windows[0].history_segments, windows[0], offset_radius=0),
        lambda: compare_default_vs_fast_matching(matches['default'], matches['fast_summary']),
        lambda: plot_local_increment_cloud(windows[0].state, prediction),
        lambda: plot_sampled_next_steps(windows[0].state, prediction),
        lambda: plot_jitter_ellipse_or_covariance(windows[0].increment, sigma_parallel=0.05, sigma_perp=0.08),
        lambda: plot_prediction_fan(windows[0].state, rollout, true_future=resampled.resampled[windows[0].resampled_index + 1: windows[0].resampled_index + 4]),
        lambda: plot_rollout_against_truth(
            windows[0].state,
            rollout,
            true_future=resampled.resampled[windows[0].resampled_index + 1: windows[0].resampled_index + 4],
            context_points=resampled.resampled[windows[0].resampled_index - 3: windows[0].resampled_index + 1],
        ),
        lambda: plot_support_diagnostics_along_rollout(rollout),
        lambda: plot_error_vs_horizon(eval_results),
        lambda: plot_model_comparison_table(eval_results),
        lambda: plot_error_vs_support(eval_results),
        lambda: plot_failure_gallery(eval_results['local_history'], n_examples=3),
    ]
    for factory in fig_factories:
        fig = factory()
        try:
            fig.canvas.draw()
            assert len(fig.axes) >= 1
        finally:
            plt.close(fig)
