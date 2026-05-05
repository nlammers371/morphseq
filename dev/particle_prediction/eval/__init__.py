"""Evaluation-layer exports for particle prediction."""

from .evaluate import (
    ModelEvaluationResult,
    TaskEvaluationResult,
    build_local_predictor_variants,
    comparison_table,
    evaluate_linear_extrapolation_baseline,
    evaluate_local_model,
    evaluate_persistence_baseline,
    run_evaluation_suite,
)
from .metrics import (
    HorizonMetricSummary,
    average_displacement_error,
    endpoint_errors_by_horizon,
    summarize_metric_by_horizon,
    summarize_model_metrics,
    support_correlation_summary,
    truth_in_cloud_distance,
)
from .predictions import RolloutPredictionResult, RolloutStepDiagnostics
from .latent_order import (
    effective_rank,
    infer_latent_columns,
    latent_corr_order_summary,
    mean_abs_offdiag_corr,
    rms_offdiag_corr,
    summarize_latent_order_by_group,
    summarize_latent_order_dataframe,
    top_k_eigen_fraction,
)

__all__ = [
    "HorizonMetricSummary",
    "ModelEvaluationResult",
    "RolloutPredictionResult",
    "RolloutStepDiagnostics",
    "TaskEvaluationResult",
    "average_displacement_error",
    "build_local_predictor_variants",
    "comparison_table",
    "effective_rank",
    "endpoint_errors_by_horizon",
    "evaluate_linear_extrapolation_baseline",
    "evaluate_local_model",
    "evaluate_persistence_baseline",
    "infer_latent_columns",
    "latent_corr_order_summary",
    "mean_abs_offdiag_corr",
    "rms_offdiag_corr",
    "run_evaluation_suite",
    "summarize_latent_order_by_group",
    "summarize_latent_order_dataframe",
    "summarize_metric_by_horizon",
    "summarize_model_metrics",
    "support_correlation_summary",
    "top_k_eigen_fraction",
    "truth_in_cloud_distance",
]
