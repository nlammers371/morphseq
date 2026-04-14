"""Geometry-first metrics for one-step and rollout evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np


@dataclass(frozen=True)
class HorizonMetricSummary:
    """Aggregated metric values for one rollout horizon."""

    horizon: int
    mean: float
    median: float
    q25: float
    q75: float
    count: int


def euclidean_error(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Return per-row Euclidean distances."""

    predicted = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if predicted.shape != truth.shape:
        raise ValueError("predicted and truth must have the same shape")
    if predicted.ndim != 2:
        raise ValueError("predicted and truth must be 2D arrays")
    return np.linalg.norm(predicted - truth, axis=1)


def endpoint_errors_by_horizon(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Return endpoint errors for each requested horizon."""

    return euclidean_error(predicted=predicted, truth=truth)


def average_displacement_error(predicted: np.ndarray, truth: np.ndarray) -> float:
    """Return mean rollout error across horizons for one task."""

    errors = endpoint_errors_by_horizon(predicted=predicted, truth=truth)
    return float(np.mean(errors))


def truth_in_cloud_distance(forward_samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Return the closest sample-to-truth distance for each horizon."""

    forward_samples = np.asarray(forward_samples, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if forward_samples.ndim != 3:
        raise ValueError("forward_samples must be a 3D array of shape (H, P, D)")
    if truth.ndim != 2:
        raise ValueError("truth must be a 2D array of shape (H, D)")
    if forward_samples.shape[0] != truth.shape[0] or forward_samples.shape[2] != truth.shape[1]:
        raise ValueError("forward_samples and truth must agree on horizons and feature dimension")

    distances = np.linalg.norm(forward_samples - truth[:, None, :], axis=2)
    return np.min(distances, axis=1)


def summarize_metric_by_horizon(
    horizons: Sequence[int],
    values: np.ndarray,
) -> List[HorizonMetricSummary]:
    """Summarize a metric matrix with shape (n_tasks, n_horizons)."""

    values = np.asarray(values, dtype=np.float64)
    horizons_array = np.asarray(horizons, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError("values must be a 2D array")
    if values.shape[1] != horizons_array.shape[0]:
        raise ValueError("values second dimension must match number of horizons")

    summaries: List[HorizonMetricSummary] = []
    for column_index, horizon in enumerate(horizons_array):
        column = np.asarray([], dtype=np.float64) if values.shape[0] == 0 else values[:, column_index]
        finite = column[np.isfinite(column)] if column.size else column
        if finite.size == 0:
            summaries.append(HorizonMetricSummary(int(horizon), np.nan, np.nan, np.nan, np.nan, 0))
            continue
        summaries.append(
            HorizonMetricSummary(
                horizon=int(horizon),
                mean=float(np.mean(finite)),
                median=float(np.median(finite)),
                q25=float(np.quantile(finite, 0.25)),
                q75=float(np.quantile(finite, 0.75)),
                count=int(finite.size),
            )
        )
    return summaries


def support_correlation_summary(
    errors: np.ndarray,
    support_values: np.ndarray,
) -> Dict[str, float]:
    """Return lightweight correlation stats for support diagnostics."""

    errors = np.asarray(errors, dtype=np.float64).reshape(-1)
    support_values = np.asarray(support_values, dtype=np.float64).reshape(-1)
    if errors.shape != support_values.shape:
        raise ValueError("errors and support_values must have matching flattened shapes")

    mask = np.isfinite(errors) & np.isfinite(support_values)
    if np.sum(mask) < 2:
        return {"pearson_r": np.nan, "mean_error": np.nan, "mean_support": np.nan, "count": int(np.sum(mask))}

    centered_errors = errors[mask] - np.mean(errors[mask])
    centered_support = support_values[mask] - np.mean(support_values[mask])
    denom = np.sqrt(np.sum(centered_errors ** 2) * np.sum(centered_support ** 2))
    pearson_r = float(np.sum(centered_errors * centered_support) / denom) if denom > 0 else np.nan
    return {
        "pearson_r": pearson_r,
        "mean_error": float(np.mean(errors[mask])),
        "mean_support": float(np.mean(support_values[mask])),
        "count": int(np.sum(mask)),
    }


def summarize_model_metrics(results: Iterable[object]) -> Dict[str, float]:
    """Aggregate top-line metrics across evaluated tasks."""

    results = list(results)
    if not results:
        return {
            "n_tasks": 0,
            "one_step_mean_error": np.nan,
            "rollout_ade_mean": np.nan,
            "truth_in_cloud_mean": np.nan,
        }

    one_step_errors = np.asarray([result.endpoint_errors[0] for result in results], dtype=np.float64)
    ades = np.asarray([result.average_displacement_error for result in results], dtype=np.float64)
    cloud_values = [
        result.truth_in_cloud_distance for result in results if result.truth_in_cloud_distance is not None
    ]
    truth_in_cloud = np.concatenate(cloud_values) if cloud_values else np.asarray([], dtype=np.float64)
    return {
        "n_tasks": int(len(results)),
        "one_step_mean_error": float(np.mean(one_step_errors)),
        "rollout_ade_mean": float(np.mean(ades)),
        "truth_in_cloud_mean": float(np.mean(truth_in_cloud)) if truth_in_cloud.size else np.nan,
    }


__all__ = [
    "HorizonMetricSummary",
    "average_displacement_error",
    "endpoint_errors_by_horizon",
    "euclidean_error",
    "summarize_metric_by_horizon",
    "summarize_model_metrics",
    "support_correlation_summary",
    "truth_in_cloud_distance",
]
