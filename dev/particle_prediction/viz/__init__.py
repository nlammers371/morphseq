"""Visualization helpers for particle prediction."""

from .evaluation import (
    plot_error_vs_horizon,
    plot_error_vs_support,
    plot_failure_gallery,
    plot_model_comparison_table,
)
from .matching import (
    compare_default_vs_fast_matching,
    plot_history_offset_heatmap,
    plot_history_reranking,
    plot_query_and_candidate_neighbors,
)
from .prediction import (
    plot_jitter_ellipse_or_covariance,
    plot_local_increment_cloud,
    plot_prediction_fan,
    plot_rollout_against_truth,
    plot_sampled_next_steps,
    plot_support_diagnostics_along_rollout,
)
from .smoothing import (
    plot_latent_trajectory_before_after_smoothing,
    plot_raw_vs_smoothed_timeseries,
    plot_sg_parameter_sweep,
)
from .transition_bank import (
    plot_arc_length_vs_time,
    plot_bank_state_density,
    plot_history_segments_example,
    plot_increment_norm_distribution,
    plot_resampled_points_on_trajectory,
    plot_transition_windows_for_embryo,
)

__all__ = [
    "compare_default_vs_fast_matching",
    "plot_arc_length_vs_time",
    "plot_bank_state_density",
    "plot_error_vs_horizon",
    "plot_error_vs_support",
    "plot_failure_gallery",
    "plot_history_offset_heatmap",
    "plot_history_reranking",
    "plot_history_segments_example",
    "plot_increment_norm_distribution",
    "plot_jitter_ellipse_or_covariance",
    "plot_latent_trajectory_before_after_smoothing",
    "plot_local_increment_cloud",
    "plot_model_comparison_table",
    "plot_prediction_fan",
    "plot_query_and_candidate_neighbors",
    "plot_raw_vs_smoothed_timeseries",
    "plot_resampled_points_on_trajectory",
    "plot_rollout_against_truth",
    "plot_sampled_next_steps",
    "plot_sg_parameter_sweep",
    "plot_support_diagnostics_along_rollout",
    "plot_transition_windows_for_embryo",
]
