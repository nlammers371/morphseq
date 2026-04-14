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

__all__ = [
    "HorizonMetricSummary",
    "ModelEvaluationResult",
    "RolloutPredictionResult",
    "RolloutStepDiagnostics",
    "TaskEvaluationResult",
    "average_displacement_error",
    "build_local_predictor_variants",
    "comparison_table",
    "endpoint_errors_by_horizon",
    "evaluate_linear_extrapolation_baseline",
    "evaluate_local_model",
    "evaluate_persistence_baseline",
    "run_evaluation_suite",
    "summarize_metric_by_horizon",
    "summarize_model_metrics",
    "support_correlation_summary",
    "truth_in_cloud_distance",
]
