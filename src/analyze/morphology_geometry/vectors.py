"""Vector math for morphology geometry analysis.

All functions accept ValidatedDirections, not raw ClassifierDirections.
No classification imports here.

Public API
----------
cosine_alignment(u, v, *, allow_sign_flip) -> float
axis_alignment(u, v) -> float
direction_matrix(vd, *, comparison_id) -> (metadata, vectors, feature_names)
weighted_axis(vd, *, comparison_id, weight_mode) -> (axis, metadata_with_weights)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analyze.morphology_geometry.validation import ValidatedDirections


def _as_vector(value: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).ravel()
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def cosine_alignment(
    u: np.ndarray,
    v: np.ndarray,
    *,
    allow_sign_flip: bool = False,
) -> float:
    """Return cosine similarity between two vectors.

    Parameters
    ----------
    u, v : array-like
        1-D vectors. Need not be unit-norm.
    allow_sign_flip : bool
        If True, return abs(cosine) so opposite-sign axes are treated as aligned.

    Returns
    -------
    float in [-1, 1] (or [0, 1] when allow_sign_flip=True).

    Raises
    ------
    ValueError if either vector is zero or non-finite.
    """
    left = _as_vector(u, name="u")
    right = _as_vector(v, name="v")
    if left.shape != right.shape:
        raise ValueError(f"Vector shapes differ: {left.shape} != {right.shape}.")
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        raise ValueError("Cosine alignment is undefined for zero vectors.")
    value = float(np.clip(np.dot(left, right) / denom, -1.0, 1.0))
    return abs(value) if allow_sign_flip else value


def axis_alignment(u: np.ndarray, v: np.ndarray) -> float:
    """Sign-free axis alignment: abs(cosine_alignment(u, v))."""
    return cosine_alignment(u, v, allow_sign_flip=True)


def direction_matrix(
    vd: ValidatedDirections,
    *,
    comparison_id: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Return the metadata slice, stacked vector matrix, and feature names.

    Parameters
    ----------
    vd : ValidatedDirections
        Already-validated artifact for a single feature_set.
    comparison_id : str, optional
        If given, restrict to rows with this comparison_id.

    Returns
    -------
    metadata : pd.DataFrame
        Filtered rows, sorted by (comparison_id, time_bin_center).
    vectors : np.ndarray
        Shape (n_rows, n_features). Rows aligned with metadata.
    feature_names : list[str]
        Authoritative column order (from vd.feature_names).

    Raises
    ------
    ValueError if comparison_id is given but not present in vd.
    """
    meta = vd.metadata
    if comparison_id is not None:
        if comparison_id not in vd.comparison_ids:
            raise ValueError(
                f"comparison_id {comparison_id!r} not found in ValidatedDirections. "
                f"Available: {list(vd.comparison_ids)}"
            )
        mask = meta["comparison_id"] == comparison_id
        meta = meta[mask].reset_index(drop=True)
        idx = meta.index.tolist()
        vectors = vd.vectors[idx]
    else:
        vectors = vd.vectors

    return meta, vectors, vd.feature_names


def weighted_axis(
    vd: ValidatedDirections,
    *,
    comparison_id: str,
    weight_mode: str = "auroc_minus_half",
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build a weighted average unit axis for one comparison.

    Parameters
    ----------
    vd : ValidatedDirections
    comparison_id : str
        Which comparison's per-bin directions to aggregate.
    weight_mode : str
        One of ``"auroc_minus_half"`` (default), ``"auroc"``, or ``"uniform"``.
        ``"auroc_minus_half"`` and ``"auroc"`` require ``vd.has_auroc=True``;
        if ``has_auroc=False``, they fall back to ``"uniform"`` automatically.

    Returns
    -------
    axis : np.ndarray
        Shape (n_features,), unit-norm.
    metadata_with_weights : pd.DataFrame
        The filtered metadata with an added ``axis_weight`` column.

    Raises
    ------
    ValueError if the weighted sum collapses to a zero vector.
    ValueError if weight_mode is not recognised.
    """
    meta, vectors, _ = direction_matrix(vd, comparison_id=comparison_id)

    valid_modes = {"auroc_minus_half", "auroc", "uniform"}
    if weight_mode not in valid_modes:
        raise ValueError(f"weight_mode must be one of {valid_modes}.")

    # Fall back to uniform if auroc is unavailable
    effective_mode = weight_mode
    if weight_mode in {"auroc_minus_half", "auroc"} and not vd.has_auroc:
        effective_mode = "uniform"

    if effective_mode == "auroc_minus_half":
        weights = np.maximum(meta["auroc_obs"].to_numpy(dtype=float) - 0.5, 0.0)
    elif effective_mode == "auroc":
        weights = meta["auroc_obs"].to_numpy(dtype=float)
    else:
        weights = np.ones(len(meta), dtype=float)

    # Fall back to uniform if weights are degenerate
    if not np.all(np.isfinite(weights)) or float(weights.sum()) <= 0.0:
        weights = np.ones(len(meta), dtype=float)

    axis = np.sum(vectors * weights[:, None], axis=0)
    norm = float(np.linalg.norm(axis))
    if norm == 0.0:
        raise ValueError(
            f"Weighted classifier axis for comparison_id={comparison_id!r} "
            "collapsed to a zero vector."
        )
    return axis / norm, meta.assign(axis_weight=weights)
