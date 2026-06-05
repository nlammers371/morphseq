"""Projection helpers for morphology geometry analysis.

Projects per-embryo binned feature vectors onto a reference axis.
Accepts ValidatedDirections for the feature column ordering contract.
No classification imports here.

Public API
----------
project_binned_features(df, *, vd, axis, id_col, time_col, bin_width, ...) -> pd.DataFrame
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from analyze.morphology_geometry.validation import ValidatedDirections
from analyze.utils.binning import add_time_bins


def project_binned_features(
    df: pd.DataFrame,
    *,
    vd: ValidatedDirections,
    axis: np.ndarray,
    id_col: str,
    time_col: str,
    bin_width: float,
    class_col: str | None = None,
    extra_group_cols: Sequence[str] = (),
    output_col: str = "phenotype_direction_score",
) -> pd.DataFrame:
    """Aggregate embryo features into bins and project onto a reference axis.

    Feature columns are taken from ``vd.feature_names`` in order, guaranteeing
    that the projection is invariant to the column order of *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Per-cell or per-embryo data. Must contain ``id_col``, ``time_col``,
        and all columns in ``vd.feature_names``.
    vd : ValidatedDirections
        Validated artifact that supplies the authoritative feature column order.
    axis : np.ndarray
        Shape (n_features,). The reference direction to project onto.
        Need not be unit-norm (raw dot product is returned).
    id_col : str
        Column identifying individual embryos.
    time_col : str
        Column with continuous time values (e.g., hpf).
    bin_width : float
        Width of time bins in the same units as *time_col*.
    class_col : str, optional
        If given, included in the groupby and preserved in the output.
    extra_group_cols : sequence of str
        Additional columns to preserve through the bin-aggregation groupby.
    output_col : str
        Name of the projection score column in the output. Default
        ``"phenotype_direction_score"``.

    Returns
    -------
    pd.DataFrame
        One row per (id_col, time_bin_center[, class_col, extra_group_cols]).
        Columns: all group columns, ``time_bin_center``, ``output_col``.

    Raises
    ------
    ValueError if any feature column from vd.feature_names is absent from df.
    ValueError if axis length does not match the number of features.
    """
    feature_names = vd.feature_names

    missing = [col for col in feature_names if col not in df.columns]
    if missing:
        raise ValueError(
            f"Input DataFrame is missing feature columns required for projection: "
            f"{missing}"
        )

    axis_arr = np.asarray(axis, dtype=float).ravel()
    if len(axis_arr) != len(feature_names):
        raise ValueError(
            f"Axis length {len(axis_arr)} does not match "
            f"{len(feature_names)} features in ValidatedDirections."
        )

    work = add_time_bins(df, time_col=time_col, bin_width=bin_width, bin_col="_time_bin")
    work["time_bin_center"] = work["_time_bin"].astype(float) + float(bin_width) / 2.0

    group_cols = [id_col, "_time_bin", "time_bin_center"]
    if class_col is not None:
        group_cols.append(class_col)
    group_cols.extend(extra_group_cols)

    binned = work.groupby(group_cols, as_index=False)[feature_names].mean()
    X = binned[feature_names].to_numpy(dtype=float)
    binned[output_col] = X @ axis_arr
    binned = binned.drop(columns=["_time_bin"])
    return binned
