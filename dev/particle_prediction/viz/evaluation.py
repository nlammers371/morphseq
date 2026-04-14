"""Evaluation visualization helpers for model comparison and failure analysis."""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from dev.particle_prediction.eval.evaluate import ModelEvaluationResult


def _result_items(results: Mapping[str, ModelEvaluationResult] | Sequence[ModelEvaluationResult]):
    if isinstance(results, Mapping):
        return list(results.items())
    return [(result.model_name, result) for result in results]


def plot_error_vs_horizon(
    results: Mapping[str, ModelEvaluationResult] | Sequence[ModelEvaluationResult],
) -> plt.Figure:
    """Plot mean endpoint error versus horizon."""

    fig, axis = plt.subplots(figsize=(7.0, 4.8))
    for model_name, result in _result_items(results):
        if not result.endpoint_error_summary:
            continue
        horizons = [summary.horizon for summary in result.endpoint_error_summary]
        means = [summary.mean for summary in result.endpoint_error_summary]
        q25 = [summary.q25 for summary in result.endpoint_error_summary]
        q75 = [summary.q75 for summary in result.endpoint_error_summary]
        axis.plot(horizons, means, marker="o", linewidth=2.0, label=model_name)
        axis.fill_between(horizons, q25, q75, alpha=0.15)
    axis.set_xlabel("rollout horizon")
    axis.set_ylabel("endpoint error")
    axis.set_title("Endpoint error versus horizon")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_model_comparison_table(
    results: Mapping[str, ModelEvaluationResult] | Sequence[ModelEvaluationResult],
) -> plt.Figure:
    """Render a compact model-comparison summary table."""

    rows = []
    for model_name, result in _result_items(results):
        last_horizon = result.endpoint_error_summary[-1].mean if result.endpoint_error_summary else np.nan
        rows.append(
            [
                model_name,
                f"{result.summary_metrics['one_step_mean_error']:.3f}",
                f"{result.summary_metrics['rollout_ade_mean']:.3f}",
                f"{last_horizon:.3f}",
                f"{result.summary_metrics['truth_in_cloud_mean']:.3f}",
            ]
        )

    fig, axis = plt.subplots(figsize=(8.0, max(2.2, 0.6 * len(rows) + 1.4)))
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=["model", "one-step", "ADE", "last horizon", "truth-in-cloud"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.3)
    axis.set_title("Model comparison summary", pad=10.0)
    fig.tight_layout()
    return fig


def plot_error_vs_support(
    results: Mapping[str, ModelEvaluationResult] | Sequence[ModelEvaluationResult],
    support_key: str = "candidate_count",
) -> plt.Figure:
    """Plot flattened endpoint error against one support covariate."""

    fig, axis = plt.subplots(figsize=(6.8, 5.2))
    for model_name, result in _result_items(results):
        if not result.task_results:
            continue
        errors = np.concatenate([task.endpoint_errors for task in result.task_results])
        support_values = np.concatenate(
            [
                np.asarray([getattr(step, support_key) for step in task.step_diagnostics], dtype=np.float64)
                for task in result.task_results
            ]
        )
        axis.scatter(support_values, errors, s=22, alpha=0.35, label=model_name)
    axis.set_xlabel(support_key.replace("_", " "))
    axis.set_ylabel("endpoint error")
    axis.set_title(f"Endpoint error versus {support_key.replace('_', ' ')}")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_failure_gallery(
    result: ModelEvaluationResult,
    n_examples: int = 6,
    dims: tuple[int, int] = (0, 1),
) -> plt.Figure:
    """Show the worst-rollout examples for one model."""

    if not result.task_results:
        raise ValueError("result must contain at least one task result")

    sorted_tasks = sorted(result.task_results, key=lambda task: task.average_displacement_error, reverse=True)
    selected = sorted_tasks[: max(1, min(n_examples, len(sorted_tasks)))]
    n_cols = min(3, len(selected))
    n_rows = int(np.ceil(len(selected) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.7 * n_rows), squeeze=False)

    for axis in axes.flat:
        axis.axis("off")

    for axis, task in zip(axes.flat, selected):
        axis.axis("on")
        axis.scatter(
            task.forward_samples[:, :, dims[0]].reshape(-1),
            task.forward_samples[:, :, dims[1]].reshape(-1),
            s=8,
            alpha=0.12,
            color="#6aaed6",
            label="samples",
        )
        axis.plot(task.predicted_mean[:, dims[0]], task.predicted_mean[:, dims[1]], color="black", linewidth=2.0)
        axis.plot(task.target_states[:, dims[0]], task.target_states[:, dims[1]], color="#d94801", linewidth=2.0)
        axis.set_title(f"{task.embryo_id} | ADE={task.average_displacement_error:.2f}", fontsize=10)
        axis.grid(alpha=0.2)

    fig.suptitle(f"Failure gallery: {result.model_name}", y=0.98)
    fig.tight_layout()
    return fig


__all__ = [
    "plot_error_vs_horizon",
    "plot_error_vs_support",
    "plot_failure_gallery",
    "plot_model_comparison_table",
]
