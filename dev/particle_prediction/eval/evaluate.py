"""Evaluation runners and baselines for particle prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from dev.particle_prediction.data.dataset import PredictionQuery, PredictionTask
from dev.particle_prediction.eval.metrics import (
    HorizonMetricSummary,
    average_displacement_error,
    endpoint_errors_by_horizon,
    summarize_metric_by_horizon,
    summarize_model_metrics,
    support_correlation_summary,
    truth_in_cloud_distance,
)
from dev.particle_prediction.eval.predictions import RolloutPredictionResult, RolloutStepDiagnostics
from dev.particle_prediction.models.local_transition_pf import LocalTransitionPredictor
from dev.particle_prediction.models.matching import MatchingConfig


@dataclass(frozen=True)
class TaskEvaluationResult:
    """Task-level prediction outputs and derived metrics."""

    model_name: str
    query_mode: str
    horizons: np.ndarray
    target_states: np.ndarray
    predicted_mean: np.ndarray
    predicted_cov_diag: np.ndarray
    forward_samples: np.ndarray
    endpoint_errors: np.ndarray
    average_displacement_error: float
    truth_in_cloud_distance: np.ndarray | None
    step_diagnostics: List[RolloutStepDiagnostics]
    embryo_id: str
    experiment_id: str
    perturbation_class: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelEvaluationResult:
    """Collection of task-level results and model summaries."""

    model_name: str
    task_results: List[TaskEvaluationResult]
    horizons: np.ndarray
    summary_metrics: Dict[str, float]
    endpoint_error_summary: List[HorizonMetricSummary]
    truth_in_cloud_summary: List[HorizonMetricSummary] | None
    support_summaries: Dict[str, Dict[str, float]]


def _recent_velocity(query: PredictionQuery) -> np.ndarray:
    if query.history_segments is None or query.history_segments.size == 0:
        return np.zeros_like(query.current_state)
    history_segments = np.asarray(query.history_segments, dtype=np.float64)
    return np.mean(history_segments, axis=0)


def _baseline_rollout(
    query: PredictionQuery,
    n_steps: int,
    mode: str,
) -> RolloutPredictionResult:
    current = np.asarray(query.current_state, dtype=np.float64)
    n_dims = current.shape[0]

    if mode == "persistence":
        increments = np.zeros((n_steps, n_dims), dtype=np.float64)
    elif mode == "linear_extrapolation":
        velocity = _recent_velocity(query)
        increments = np.repeat(velocity[None, :], n_steps, axis=0)
    else:
        raise ValueError("Unsupported baseline mode")

    predicted_mean = current[None, :] + np.cumsum(increments, axis=0)
    forward_samples = predicted_mean[:, None, :].copy()
    predicted_cov_diag = np.zeros_like(predicted_mean)
    step_diagnostics = [
        RolloutStepDiagnostics(
            candidate_count=1.0,
            effective_sample_size=1.0,
            history_mismatch=0.0,
            search_radius=0.0,
            selected_class_weights={},
            diagnostics={"baseline_mode": mode, "step_index": int(step_index)},
        )
        for step_index in range(n_steps)
    ]
    return RolloutPredictionResult(
        predicted_mean=predicted_mean,
        predicted_cov_diag=predicted_cov_diag,
        forward_samples=forward_samples,
        step_diagnostics=step_diagnostics,
        diagnostics={"baseline_mode": mode},
    )


def _slice_rollout_to_task(
    rollout: RolloutPredictionResult,
    horizons: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[RolloutStepDiagnostics]]:
    horizon_indices = np.asarray(horizons, dtype=np.int64) - 1
    return (
        rollout.predicted_mean[horizon_indices].copy(),
        rollout.predicted_cov_diag[horizon_indices].copy(),
        rollout.forward_samples[horizon_indices].copy(),
        [rollout.step_diagnostics[int(index)] for index in horizon_indices],
    )


def _assemble_model_result(
    *,
    model_name: str,
    task_results: Sequence[TaskEvaluationResult],
    horizons: np.ndarray,
) -> ModelEvaluationResult:
    endpoint_matrix = np.vstack([task.endpoint_errors for task in task_results]) if task_results else np.empty((0, len(horizons)))
    truth_in_cloud_matrix = (
        np.vstack([task.truth_in_cloud_distance for task in task_results]) if task_results else np.empty((0, len(horizons)))
    )
    candidate_count_matrix = (
        np.vstack(
            [np.asarray([step.candidate_count for step in task.step_diagnostics], dtype=np.float64) for task in task_results]
        )
        if task_results
        else np.empty((0, len(horizons)))
    )
    ess_matrix = (
        np.vstack(
            [np.asarray([step.effective_sample_size for step in task.step_diagnostics], dtype=np.float64) for task in task_results]
        )
        if task_results
        else np.empty((0, len(horizons)))
    )
    mismatch_matrix = (
        np.vstack(
            [np.asarray([step.history_mismatch for step in task.step_diagnostics], dtype=np.float64) for task in task_results]
        )
        if task_results
        else np.empty((0, len(horizons)))
    )
    radius_matrix = (
        np.vstack(
            [np.asarray([step.search_radius for step in task.step_diagnostics], dtype=np.float64) for task in task_results]
        )
        if task_results
        else np.empty((0, len(horizons)))
    )

    support_summaries = {
        "candidate_count": support_correlation_summary(endpoint_matrix, candidate_count_matrix),
        "effective_sample_size": support_correlation_summary(endpoint_matrix, ess_matrix),
        "history_mismatch": support_correlation_summary(endpoint_matrix, mismatch_matrix),
        "search_radius": support_correlation_summary(endpoint_matrix, radius_matrix),
    }
    return ModelEvaluationResult(
        model_name=model_name,
        task_results=list(task_results),
        horizons=horizons,
        summary_metrics=summarize_model_metrics(task_results),
        endpoint_error_summary=summarize_metric_by_horizon(horizons, endpoint_matrix),
        truth_in_cloud_summary=summarize_metric_by_horizon(horizons, truth_in_cloud_matrix),
        support_summaries=support_summaries,
    )


def evaluate_rollout_predictions(
    *,
    model_name: str,
    tasks: Sequence[PredictionTask],
    rollout_fn,
) -> ModelEvaluationResult:
    """Evaluate a rollout-producing callable on a set of tasks."""

    task_results: List[TaskEvaluationResult] = []
    horizons = np.asarray(tasks[0].horizons, dtype=np.int64) if tasks else np.asarray([], dtype=np.int64)
    for task_index, task in enumerate(tasks):
        rollout = rollout_fn(task.query, int(task.horizons[-1]))
        predicted_mean, predicted_cov_diag, forward_samples, step_diagnostics = _slice_rollout_to_task(
            rollout=rollout,
            horizons=task.horizons,
        )
        endpoint_errors = endpoint_errors_by_horizon(predicted=predicted_mean, truth=task.target_states)
        cloud_distance = truth_in_cloud_distance(forward_samples=forward_samples, truth=task.target_states)
        task_results.append(
            TaskEvaluationResult(
                model_name=model_name,
                query_mode=task.query.mode,
                horizons=task.horizons.copy(),
                target_states=task.target_states.copy(),
                predicted_mean=predicted_mean,
                predicted_cov_diag=predicted_cov_diag,
                forward_samples=forward_samples,
                endpoint_errors=endpoint_errors,
                average_displacement_error=average_displacement_error(predicted=predicted_mean, truth=task.target_states),
                truth_in_cloud_distance=cloud_distance,
                step_diagnostics=step_diagnostics,
                embryo_id=task.embryo_id,
                experiment_id=task.experiment_id,
                perturbation_class=task.perturbation_class,
                metadata={"task_index": int(task_index), **task.metadata},
            )
        )
    return _assemble_model_result(model_name=model_name, task_results=task_results, horizons=horizons)


def evaluate_persistence_baseline(tasks: Sequence[PredictionTask]) -> ModelEvaluationResult:
    """Evaluate the persistence baseline."""

    return evaluate_rollout_predictions(
        model_name="persistence",
        tasks=tasks,
        rollout_fn=lambda query, n_steps: _baseline_rollout(query=query, n_steps=n_steps, mode="persistence"),
    )


def evaluate_linear_extrapolation_baseline(tasks: Sequence[PredictionTask]) -> ModelEvaluationResult:
    """Evaluate a constant-velocity baseline from recent history."""

    return evaluate_rollout_predictions(
        model_name="linear_extrapolation",
        tasks=tasks,
        rollout_fn=lambda query, n_steps: _baseline_rollout(query=query, n_steps=n_steps, mode="linear_extrapolation"),
    )


def evaluate_local_model(
    *,
    tasks: Sequence[PredictionTask],
    predictor: LocalTransitionPredictor,
    model_name: str,
    n_particles: int = 128,
    random_seed: int = 0,
) -> ModelEvaluationResult:
    """Evaluate the local empirical predictor on a set of tasks."""

    seed_sequence = np.random.SeedSequence(random_seed)
    child_states = seed_sequence.spawn(max(1, len(tasks)))
    task_results: List[TaskEvaluationResult] = []
    horizons = np.asarray(tasks[0].horizons, dtype=np.int64) if tasks else np.asarray([], dtype=np.int64)

    for task_index, task in enumerate(tasks):
        rng = np.random.default_rng(child_states[task_index])
        rollout = predictor.rollout_query(
            query=task.query,
            n_steps=int(task.horizons[-1]),
            n_particles=n_particles,
            rng=rng,
        )
        predicted_mean, predicted_cov_diag, forward_samples, step_diagnostics = _slice_rollout_to_task(
            rollout=rollout,
            horizons=task.horizons,
        )
        endpoint_errors = endpoint_errors_by_horizon(predicted=predicted_mean, truth=task.target_states)
        cloud_distance = truth_in_cloud_distance(forward_samples=forward_samples, truth=task.target_states)
        task_results.append(
            TaskEvaluationResult(
                model_name=model_name,
                query_mode=task.query.mode,
                horizons=task.horizons.copy(),
                target_states=task.target_states.copy(),
                predicted_mean=predicted_mean,
                predicted_cov_diag=predicted_cov_diag,
                forward_samples=forward_samples,
                endpoint_errors=endpoint_errors,
                average_displacement_error=average_displacement_error(predicted=predicted_mean, truth=task.target_states),
                truth_in_cloud_distance=cloud_distance,
                step_diagnostics=step_diagnostics,
                embryo_id=task.embryo_id,
                experiment_id=task.experiment_id,
                perturbation_class=task.perturbation_class,
                metadata={"task_index": int(task_index), **task.metadata},
            )
        )
    return _assemble_model_result(model_name=model_name, task_results=task_results, horizons=horizons)


def build_local_predictor_variants(
    *,
    bank,
    sigma_parallel: float = 0.05,
    sigma_perp: float = 0.1,
    jitter_mode: str = "tangent",
    base_matching_config: MatchingConfig | None = None,
) -> Dict[str, LocalTransitionPredictor]:
    """Build the documented local-model ablation suite."""

    base = MatchingConfig() if base_matching_config is None else base_matching_config
    configs = {
        "local_no_history": MatchingConfig(
            k_state=base.k_state,
            offset_radius=base.offset_radius,
            alpha=base.alpha,
            sigma_z=base.sigma_z,
            sigma_h=base.sigma_h,
            lambda_h=0.0,
            retrieval_method=base.retrieval_method,
            history_mode=base.history_mode,
        ),
        "local_history": base,
        "local_fast_summary": MatchingConfig(
            k_state=base.k_state,
            offset_radius=base.offset_radius,
            alpha=base.alpha,
            sigma_z=base.sigma_z,
            sigma_h=base.sigma_h,
            lambda_h=base.lambda_h,
            retrieval_method=base.retrieval_method,
            history_mode="fast_summary",
        ),
    }
    return {
        name: LocalTransitionPredictor(
            bank=bank,
            matching_config=config,
            sigma_parallel=sigma_parallel,
            sigma_perp=sigma_perp,
            jitter_mode=jitter_mode,
        )
        for name, config in configs.items()
    }


def run_evaluation_suite(
    *,
    tasks: Sequence[PredictionTask],
    bank,
    n_particles: int = 128,
    random_seed: int = 0,
    sigma_parallel: float = 0.05,
    sigma_perp: float = 0.1,
    jitter_mode: str = "tangent",
    matching_config: MatchingConfig | None = None,
) -> Dict[str, ModelEvaluationResult]:
    """Run the baseline and local-model comparison suite."""

    results: Dict[str, ModelEvaluationResult] = {
        "persistence": evaluate_persistence_baseline(tasks),
        "linear_extrapolation": evaluate_linear_extrapolation_baseline(tasks),
    }
    predictors = build_local_predictor_variants(
        bank=bank,
        sigma_parallel=sigma_parallel,
        sigma_perp=sigma_perp,
        jitter_mode=jitter_mode,
        base_matching_config=matching_config,
    )
    for model_name, predictor in predictors.items():
        results[model_name] = evaluate_local_model(
            tasks=tasks,
            predictor=predictor,
            model_name=model_name,
            n_particles=n_particles,
            random_seed=random_seed,
        )
    return results


def comparison_table(results: Mapping[str, ModelEvaluationResult]) -> List[Dict[str, float]]:
    """Return a lightweight summary table for notebook display."""

    table: List[Dict[str, float]] = []
    for model_name, result in results.items():
        row = {"model_name": model_name, **result.summary_metrics}
        row["last_horizon_mean_error"] = (
            float(result.endpoint_error_summary[-1].mean) if result.endpoint_error_summary else np.nan
        )
        table.append(row)
    return table


__all__ = [
    "ModelEvaluationResult",
    "TaskEvaluationResult",
    "build_local_predictor_variants",
    "comparison_table",
    "evaluate_linear_extrapolation_baseline",
    "evaluate_local_model",
    "evaluate_persistence_baseline",
    "evaluate_rollout_predictions",
    "run_evaluation_suite",
]
