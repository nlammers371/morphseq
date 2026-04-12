"""
space_density_metrics.py
------------------------
Pure geometry and local-density metrics for saved condensation iterations.
"""
from __future__ import annotations

import numpy as np


def finite_rows(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0, points.shape[-1]), dtype=float)
    keep = np.isfinite(points).all(axis=1)
    return points[keep]


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros((0,), dtype=float)
    diffs = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=2))
    tri = np.triu_indices(len(points), k=1)
    return dists[tri]


def knn_density_values(points: np.ndarray, k: int = 5) -> np.ndarray:
    if len(points) <= 1:
        return np.zeros((0,), dtype=float)
    k_eff = max(1, min(k, len(points) - 1))
    diffs = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=2))
    np.fill_diagonal(dists, np.inf)
    nearest = np.partition(dists, kth=k_eff - 1, axis=1)[:, :k_eff]
    mean_knn = np.nanmean(nearest, axis=1)
    return 1.0 / (mean_knn + 1e-12)


def safe_nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return float("nan")
    return float(np.nanmean(arr))


def safe_nanmedian(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return float("nan")
    return float(np.nanmedian(arr))


def summarize_iteration_geometry(
    positions: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    time_values: np.ndarray,
) -> dict[str, float]:
    compactness_by_time: list[float] = []
    separation_by_time: list[float] = []
    crowding_by_time: list[float] = []
    density_means: list[float] = []
    density_cvs: list[float] = []
    active_counts: list[int] = []

    unique_labels = np.unique(labels)
    centroids_prev: dict[str, np.ndarray] | None = None
    centroid_shift_norms: list[float] = []

    for t_idx, _ in enumerate(time_values):
        obs = np.where(mask[:, t_idx])[0]
        if len(obs) < 2:
            continue
        pts_t = positions[obs, t_idx, :]
        finite_obs = np.isfinite(pts_t).all(axis=1)
        obs = obs[finite_obs]
        pts_t = pts_t[finite_obs]
        if len(obs) < 2:
            continue
        active_counts.append(len(obs))

        densities = knn_density_values(pts_t, k=5)
        if densities.size:
            density_means.append(float(np.nanmean(densities)))
            density_cvs.append(float(np.nanstd(densities) / (np.nanmean(densities) + 1e-12)))

        centroids_now: dict[str, np.ndarray] = {}
        within_dists: list[float] = []
        for label in unique_labels:
            idx = obs[labels[obs] == label]
            if len(idx) == 0:
                continue
            pts_label = finite_rows(positions[idx, t_idx, :])
            if len(pts_label) == 0:
                continue
            centroid = np.nanmean(pts_label, axis=0)
            centroids_now[str(label)] = centroid
            radial = np.sqrt(np.sum((pts_label - centroid[None, :]) ** 2, axis=1))
            within_dists.append(float(np.nanmean(radial)))

        if within_dists:
            compactness_by_time.append(float(np.nanmean(within_dists)))

        centroid_list = list(centroids_now.values())
        if len(centroid_list) >= 2:
            between = pairwise_distances(np.vstack(centroid_list))
            if between.size:
                separation_by_time.append(float(np.nanmedian(between)))
                crowding_by_time.append(float(np.nanpercentile(between, 10)))

        if centroids_prev is not None:
            shared = sorted(set(centroids_prev).intersection(centroids_now))
            if shared:
                shifts = [
                    float(np.linalg.norm(centroids_now[label] - centroids_prev[label]))
                    for label in shared
                ]
                centroid_shift_norms.append(float(np.nanmean(shifts)))
        centroids_prev = centroids_now

    compactness = safe_nanmean(compactness_by_time)
    separation = safe_nanmedian(separation_by_time)
    crowding = safe_nanmedian(crowding_by_time)
    shift = safe_nanmean(centroid_shift_norms)
    support = float(np.nanmean(active_counts)) if active_counts else float("nan")
    density_mean = safe_nanmean(density_means)
    density_cv = safe_nanmean(density_cvs)

    within_over_between = compactness / (separation + 1e-12) if np.isfinite(compactness) and np.isfinite(separation) else float("nan")
    crowding_score = crowding / (separation + 1e-12) if np.isfinite(crowding) and np.isfinite(separation) else float("nan")

    return {
        "compactness_mean": compactness,
        "centroid_separation_median": separation,
        "crowding_p10": crowding,
        "centroid_shift_mean": shift,
        "support_mean": support,
        "density_knn_mean": density_mean,
        "density_knn_cv": density_cv,
        "within_over_between": within_over_between,
        "crowding_score": crowding_score,
    }
