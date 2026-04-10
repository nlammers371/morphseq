"""Vector math for PBX phenotype direction analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analyze.classification.engine.analysis import ClassifierDirections


def _as_vector(value: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).ravel()
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def cosine_alignment(
    u: np.ndarray,
    v: np.ndarray,
    *,
    allow_sign_flip: bool = False,
) -> float:
    """Return cosine similarity; optionally treat opposite signs as same axis."""
    left = _as_vector(u, name="u")
    right = _as_vector(v, name="v")
    if left.shape != right.shape:
        raise ValueError(f"Vector shapes differ: {left.shape} != {right.shape}.")
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        raise ValueError("Cosine alignment is undefined for zero vectors.")
    value = float(np.dot(left, right) / denom)
    value = float(np.clip(value, -1.0, 1.0))
    return abs(value) if allow_sign_flip else value


def axis_alignment(u: np.ndarray, v: np.ndarray) -> float:
    """Return sign-free axis alignment, abs(cosine_alignment(u, v))."""
    return cosine_alignment(u, v, allow_sign_flip=True)


def _filter_metadata(
    metadata: pd.DataFrame,
    *,
    feature_set: str | None,
    comparison_id: str | None,
) -> pd.DataFrame:
    out = metadata.copy()
    if feature_set is not None:
        out = out[out["feature_set"] == feature_set]
    if comparison_id is not None:
        out = out[out["comparison_id"] == comparison_id]
    return out.sort_values(["feature_set", "comparison_id", "time_bin_center"]).reset_index(drop=True)


def direction_matrix(
    directions: ClassifierDirections,
    *,
    feature_set: str | None = None,
    comparison_id: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Return metadata, stacked vectors, and the ordered feature names."""
    metadata = _filter_metadata(
        directions.metadata,
        feature_set=feature_set,
        comparison_id=comparison_id,
    )
    if metadata.empty:
        raise ValueError("No classifier direction rows matched the requested filters.")
    feature_sets = metadata["feature_set"].unique().tolist()
    if len(feature_sets) != 1:
        raise ValueError("direction_matrix requires exactly one feature_set.")
    names = directions.feature_names[str(feature_sets[0])]
    vectors = np.vstack([directions.vectors[str(vector_id)] for vector_id in metadata["vector_id"]])
    if vectors.shape[1] != len(names):
        raise ValueError("Stacked direction matrix width does not match feature names.")
    return metadata, vectors, names


def weighted_axis(
    directions: ClassifierDirections,
    *,
    feature_set: str,
    comparison_id: str,
    weight_mode: str = "auroc_minus_half",
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build an AUROC-weighted average unit axis for one comparison."""
    metadata, vectors, _ = direction_matrix(
        directions,
        feature_set=feature_set,
        comparison_id=comparison_id,
    )
    if weight_mode == "auroc_minus_half":
        weights = np.maximum(metadata["auroc_obs"].to_numpy(dtype=float) - 0.5, 0.0)
    elif weight_mode == "auroc":
        weights = metadata["auroc_obs"].to_numpy(dtype=float)
    elif weight_mode == "uniform":
        weights = np.ones(len(metadata), dtype=float)
    else:
        raise ValueError(
            "weight_mode must be one of {'auroc_minus_half', 'auroc', 'uniform'}."
        )
    if not np.all(np.isfinite(weights)) or float(weights.sum()) <= 0.0:
        weights = np.ones(len(metadata), dtype=float)
    axis = np.sum(vectors * weights[:, None], axis=0)
    norm = float(np.linalg.norm(axis))
    if norm == 0.0:
        raise ValueError("Weighted classifier axis collapsed to a zero vector.")
    return axis / norm, metadata.assign(axis_weight=weights)
