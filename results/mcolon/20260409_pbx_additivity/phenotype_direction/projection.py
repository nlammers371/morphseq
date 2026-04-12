"""Projection helpers for PBX phenotype direction analysis."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from analyze.classification.engine.analysis import ClassifierDirections


def project_binned_features(
    df: pd.DataFrame,
    *,
    directions: ClassifierDirections,
    axis: np.ndarray,
    feature_set: str,
    id_col: str,
    time_col: str,
    bin_width: float,
    class_col: str | None = None,
    extra_group_cols: Sequence[str] = (),
    output_col: str = "phenotype_direction_score",
) -> pd.DataFrame:
    """Aggregate embryo features into saved feature order and project onto axis."""
    if feature_set not in directions.feature_names:
        raise ValueError(f"Unknown classifier direction feature_set {feature_set!r}.")
    feature_names = list(directions.feature_names[feature_set])
    missing = [col for col in feature_names if col not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing projection feature columns: {missing}")
    axis_arr = np.asarray(axis, dtype=float).ravel()
    if len(axis_arr) != len(feature_names):
        raise ValueError(
            f"Axis length {len(axis_arr)} does not match {len(feature_names)} features."
        )

    work = df.copy()
    work["_time_bin"] = (np.floor(work[time_col] / bin_width) * bin_width).astype(int)
    work["time_bin_center"] = work["_time_bin"].astype(float) + float(bin_width) / 2.0

    group_cols = [id_col, "_time_bin", "time_bin_center"]
    if class_col is not None:
        group_cols.append(class_col)
    group_cols.extend(extra_group_cols)

    binned = work.groupby(group_cols, as_index=False)[feature_names].mean()
    X = binned[feature_names].to_numpy(dtype=float)
    binned[output_col] = X @ axis_arr
    return binned
