from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def _positive_robust_z(values: np.ndarray, median: float, mad: float, eps: float = 1e-6) -> np.ndarray:
    scale = 1.4826 * mad + eps
    z = (values - median) / scale
    return np.clip(z, 0.0, None)


def _fit_axis_geometry(X_scaled: np.ndarray, y_binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    group0 = X_scaled[y_binary == 0]
    group1 = X_scaled[y_binary == 1]
    centroid0 = group0.mean(axis=0)
    centroid1 = group1.mean(axis=0)
    midpoint = 0.5 * (centroid0 + centroid1)
    diff = centroid1 - centroid0
    norm = float(np.linalg.norm(diff))
    if norm <= 1e-12:
        axis_unit = np.zeros(X_scaled.shape[1], dtype=float)
    else:
        axis_unit = diff / norm
    return midpoint, axis_unit


def _axis_metrics(X_scaled: np.ndarray, midpoint: np.ndarray, axis_unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = X_scaled - midpoint
    projection = centered @ axis_unit
    residual = np.linalg.norm(centered - np.outer(projection, axis_unit), axis=1)
    return projection, residual


def _training_knn_distances(X_scaled: np.ndarray, n_neighbors: int) -> np.ndarray:
    if len(X_scaled) <= 1:
        return np.zeros(len(X_scaled), dtype=float)
    k = min(max(1, int(n_neighbors)), len(X_scaled) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)
    return distances[:, 1:].mean(axis=1)


def _query_knn_distances(X_train_scaled: np.ndarray, X_query_scaled: np.ndarray, n_neighbors: int) -> np.ndarray:
    distances, _ = _query_knn_details(X_train_scaled, X_query_scaled, n_neighbors)
    return distances.mean(axis=1)


def _query_knn_details(
    X_train_scaled: np.ndarray,
    X_query_scaled: np.ndarray,
    n_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(X_train_scaled) == 0:
        n = len(X_query_scaled)
        return (
            np.full((n, 1), np.nan, dtype=float),
            np.full((n, 1), -1, dtype=int),
        )
    k = min(max(1, int(n_neighbors)), len(X_train_scaled))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X_train_scaled)
    distances, indices = nn.kneighbors(X_query_scaled)
    return distances, indices


def build_support_reference(
    X_train_raw: np.ndarray,
    y_binary: np.ndarray,
    *,
    feature_cols: list[str],
    group1: str,
    group2: str,
    k_neighbors: int,
) -> dict[str, Any]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_scaled = scaler.fit_transform(X_train_raw)
    midpoint, axis_unit = _fit_axis_geometry(X_train_scaled, y_binary)

    train_projection, train_residual = _axis_metrics(X_train_scaled, midpoint, axis_unit)
    train_novelty = _training_knn_distances(X_train_scaled, n_neighbors=k_neighbors)

    return {
        "feature_cols": list(feature_cols),
        "group1": group1,
        "group2": group2,
        "scaler": scaler,
        "X_train_scaled": X_train_scaled,
        "axis_midpoint": midpoint,
        "axis_unit": axis_unit,
        "train_projection": train_projection,
        "train_residual": train_residual,
        "train_novelty": train_novelty,
        "train_residual_median": float(np.median(train_residual)),
        "train_residual_mad": float(np.median(np.abs(train_residual - np.median(train_residual)))),
        "train_novelty_median": float(np.median(train_novelty)),
        "train_novelty_mad": float(np.median(np.abs(train_novelty - np.median(train_novelty)))),
        "k_neighbors": int(min(max(1, int(k_neighbors)), len(X_train_scaled))),
    }


def score_support_metrics(
    reference: dict[str, Any],
    df_bin: pd.DataFrame,
    *,
    feature_cols: list[str],
) -> pd.DataFrame:
    X_query_raw = df_bin[feature_cols].to_numpy(dtype=float)
    X_query_scaled = reference["scaler"].transform(X_query_raw)
    axis_projection, axis_residual = _axis_metrics(
        X_query_scaled,
        reference["axis_midpoint"],
        reference["axis_unit"],
    )
    knn_novelty = _query_knn_distances(
        reference["X_train_scaled"],
        X_query_scaled,
        n_neighbors=reference["k_neighbors"],
    )
    axis_residual_z = _positive_robust_z(
        axis_residual,
        reference["train_residual_median"],
        reference["train_residual_mad"],
    )
    knn_novelty_z = _positive_robust_z(
        knn_novelty,
        reference["train_novelty_median"],
        reference["train_novelty_mad"],
    )

    out = df_bin[["embryo_id", "genotype", "experiment_id", "_time_bin", "time_bin_center"]].copy()
    out["axis_projection"] = axis_projection
    out["axis_residual"] = axis_residual
    out["axis_residual_z"] = axis_residual_z
    out["knn_novelty"] = knn_novelty
    out["knn_novelty_z"] = knn_novelty_z
    return out


def add_support_weights(axis_df: pd.DataFrame) -> pd.DataFrame:
    out = axis_df.copy()
    out["variance_weight"] = 1.0 / (1.0 + out["position_logit_sd"].fillna(np.inf).replace(np.inf, np.nan))
    out["variance_weight"] = out["variance_weight"].fillna(0.0)
    out["residual_weight"] = np.exp(-out["axis_residual_z"].fillna(np.inf).replace(np.inf, np.nan))
    out["residual_weight"] = out["residual_weight"].fillna(0.0)
    out["novelty_weight"] = np.exp(-out["knn_novelty_z"].fillna(np.inf).replace(np.inf, np.nan))
    out["novelty_weight"] = out["novelty_weight"].fillna(0.0)
    out["support_weight"] = out["variance_weight"] * out["residual_weight"] * out["novelty_weight"]
    out.loc[~out["model_available"], "support_weight"] = 0.0
    out["supported_position"] = out["position_logit_mean"].fillna(0.0) * out["support_weight"]
    return out


def summarize_feature_support(
    reference: dict[str, Any],
    df_bin: pd.DataFrame,
    *,
    feature_cols: list[str],
) -> pd.DataFrame:
    X_query_raw = df_bin[feature_cols].to_numpy(dtype=float)
    X_query_scaled = reference["scaler"].transform(X_query_raw)

    centered = X_query_scaled - reference["axis_midpoint"]
    projection = centered @ reference["axis_unit"]
    residual_vectors = centered - np.outer(projection, reference["axis_unit"])
    residual_sq = residual_vectors**2
    residual_sq_total = residual_sq.sum(axis=1, keepdims=True)
    residual_fraction = np.divide(
        residual_sq,
        np.maximum(residual_sq_total, 1e-12),
        out=np.zeros_like(residual_sq),
        where=np.isfinite(residual_sq),
    )

    _, knn_indices = _query_knn_details(
        reference["X_train_scaled"],
        X_query_scaled,
        n_neighbors=reference["k_neighbors"],
    )
    novelty_sq = np.full_like(residual_sq, np.nan, dtype=float)
    valid = knn_indices[:, 0] >= 0
    if np.any(valid):
        neighbor_points = reference["X_train_scaled"][knn_indices[valid]]
        deltas = X_query_scaled[valid, None, :] - neighbor_points
        novelty_sq[valid] = np.mean(deltas**2, axis=1)
    novelty_sq_total = np.nansum(novelty_sq, axis=1, keepdims=True)
    novelty_fraction = np.divide(
        novelty_sq,
        np.maximum(novelty_sq_total, 1e-12),
        out=np.zeros_like(novelty_sq),
        where=np.isfinite(novelty_sq),
    )

    summary_cols: dict[str, Any] = {"genotype": df_bin["genotype"].to_numpy()}
    for col_idx, feature in enumerate(feature_cols):
        summary_cols[f"{feature}__residual_sq"] = residual_sq[:, col_idx]
        summary_cols[f"{feature}__residual_fraction"] = residual_fraction[:, col_idx]
        summary_cols[f"{feature}__novelty_sq"] = novelty_sq[:, col_idx]
        summary_cols[f"{feature}__novelty_fraction"] = novelty_fraction[:, col_idx]
    summary_source = pd.DataFrame(summary_cols)

    rows: list[dict[str, Any]] = []
    grouped_frames = [("__all__", summary_source)]
    grouped_frames.extend((str(genotype), grp) for genotype, grp in summary_source.groupby("genotype"))
    for genotype, grp in grouped_frames:
        for feature in feature_cols:
            rows.append(
                {
                    "genotype": genotype,
                    "feature": feature,
                    "n_rows": int(len(grp)),
                    "mean_axis_residual_sq_contrib": float(grp[f"{feature}__residual_sq"].mean()),
                    "mean_axis_residual_fraction": float(grp[f"{feature}__residual_fraction"].mean()),
                    "mean_knn_novelty_sq_contrib": float(grp[f"{feature}__novelty_sq"].mean()),
                    "mean_knn_novelty_fraction": float(grp[f"{feature}__novelty_fraction"].mean()),
                }
            )
    return pd.DataFrame(rows)


def summarize_support_components(axis_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        axis_df.groupby(
            ["pair_id", "group1", "group2", "_time_bin", "time_bin_center", "genotype"],
            as_index=False,
        )
        .agg(
            n_rows=("embryo_id", "count"),
            mean_variance_weight=("variance_weight", "mean"),
            median_variance_weight=("variance_weight", "median"),
            mean_residual_weight=("residual_weight", "mean"),
            median_residual_weight=("residual_weight", "median"),
            mean_novelty_weight=("novelty_weight", "mean"),
            median_novelty_weight=("novelty_weight", "median"),
            mean_support_weight=("support_weight", "mean"),
            median_support_weight=("support_weight", "median"),
        )
    )
    return summary.sort_values(["pair_id", "_time_bin", "genotype"]).reset_index(drop=True)
