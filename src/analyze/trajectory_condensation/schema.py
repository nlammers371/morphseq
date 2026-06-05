"""
schema.py
---------
Ingests upstream data in different formats and returns canonical tensors.

All downstream code works with CondensationData only.
No raw DataFrames leak past this module.

Invariants enforced on every CondensationData instance:
  - time_values is sorted ascending and shared across all embryos
  - features[i, t, :] is valid iff mask[i, t] is True
  - labels[i] is constant across time (genotype does not change per embryo)
  - shapes are mutually consistent (validated on construction)

Preferred entry point
---------------------
from_classifier_directions(df, vd, ...) — builds CondensationData from
ValidatedDirections (morphology_geometry). Each feature dimension is the raw
dot product of an embryo's binned feature vector with one classifier direction.

Legacy entry points (deprecated)
---------------------------------
from_pairwise_margin_csv() and from_multiclass_csv() accepted CSVs produced by
the old contrast-coordinate path. Those values were clipped by the SVM margin and
are not a clean geometric projection. Prefer from_classifier_directions() for all
new work. The legacy functions are preserved for historical script compatibility
but are no longer re-exported from the package __init__.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from analyze.morphology_geometry.validation import ValidatedDirections


@dataclass
class CondensationData:
    """Canonical input contract for trajectory cosmology.

    Attributes
    ----------
    features : (N_e, T, K) float array
        Per-embryo per-timebin feature vectors.
        K = number of classifier directions (one per vector_id).
        features[i, t, :] is meaningful only where mask[i, t] is True.
    mask : (N_e, T) bool array
        True where embryo i is observed at time bin t.
        Missingness is explicit — never padded with NaN or 0.
    embryo_ids : (N_e,) str array
        Unique embryo identifiers, sorted lexicographically.
    time_values : (T,) float array
        Time bin centers in hpf, sorted ascending.
        This is the global time grid — shared across all embryos.
    labels : (N_e,) str array
        Per-embryo genotype label, constant across time.
        Labels only enter the system at the diagnostics stage;
        condensation never sees them.
    feature_names : list of str, length K
        Column names for the K feature dimensions.
    embryo_index : dict[str, int]
        Maps embryo_id → row index in N_e dimension.
    time_index : dict[float, int]
        Maps time_bin_center (hpf) → column index in T dimension.
    """
    features: np.ndarray
    mask: np.ndarray
    embryo_ids: np.ndarray
    time_values: np.ndarray
    labels: np.ndarray
    feature_names: list[str]
    embryo_index: dict[str, int]
    time_index: dict[float, int]


def validate(data: CondensationData, *, allow_feature_nans: bool = False) -> None:
    """Assert all shape and semantic invariants on a CondensationData instance.

    Raises AssertionError with a descriptive message on violation.
    Call once after construction; cheap enough to leave in production paths.
    """
    N_e, T, K = data.features.shape

    assert data.mask.shape == (N_e, T), (
        f"mask shape {data.mask.shape} != features (N_e, T) = ({N_e}, {T})"
    )
    assert len(data.embryo_ids) == N_e, (
        f"embryo_ids length {len(data.embryo_ids)} != N_e={N_e}"
    )
    assert len(data.time_values) == T, (
        f"time_values length {len(data.time_values)} != T={T}"
    )
    assert len(data.labels) == N_e, (
        f"labels length {len(data.labels)} != N_e={N_e}"
    )
    assert len(data.feature_names) == K, (
        f"feature_names length {len(data.feature_names)} != K={K}"
    )
    assert np.all(np.diff(data.time_values) >= 0), "time_values must be sorted ascending"
    assert len(set(data.embryo_ids.tolist())) == N_e, "embryo_ids must be unique"
    assert len(set(map(float, data.time_values.tolist()))) == T, "time_values must be unique"
    assert len(data.embryo_index) == N_e, (
        "embryo_index must have one entry per embryo"
    )
    assert len(data.time_index) == T, (
        "time_index must have one entry per time bin"
    )

    for i, embryo_id in enumerate(data.embryo_ids):
        assert data.embryo_index[embryo_id] == i, (
            f"embryo_index mismatch for {embryo_id}: expected {i}"
        )
    for j, time_value in enumerate(data.time_values):
        assert data.time_index[float(time_value)] == j, (
            f"time_index mismatch for {time_value}: expected {j}"
        )

    observed = data.features[data.mask]
    if observed.size and not allow_feature_nans:
        assert np.isfinite(observed).all(), "observed feature values must be finite"

    unobserved = data.features[~data.mask]
    if unobserved.size:
        assert np.isnan(unobserved).all(), (
            "unobserved feature values must be NaN to prevent masked-data leakage"
        )


def from_classifier_directions(
    df: pd.DataFrame,
    vd: "ValidatedDirections",
    *,
    id_col: str = "embryo_id",
    time_col: str,
    label_col: str = "genotype",
    centering: str = "raw",
) -> CondensationData:
    """Build CondensationData from classifier directions (preferred path).

    Each feature dimension in the resulting CondensationData is the raw dot
    product of an embryo's binned feature vector with one classifier direction
    (one column per vector_id = comparison_id × time_bin).

    This is the canonical input path for trajectory condensation. Use it instead
    of the legacy CSV loaders — contrast-coordinate CSVs from the old
    classification path had values clipped by the SVM margin, making them a
    distorted geometric representation.

    Parameters
    ----------
    df : pd.DataFrame
        Per-embryo per-cell data. Must contain id_col, time_col, label_col, and
        all columns in vd.feature_names.
    vd : ValidatedDirections
        Loaded and validated classifier directions artifact (from
        analyze.morphology_geometry.io.load_classifier_directions).
    id_col : str
        Column identifying individual embryos.
    time_col : str
        Column with continuous time values (hpf).
    label_col : str
        Column with genotype or other category label.
    centering : str
        How to center the raw projection scores. Currently only "raw" is
        supported (no centering — pure dot product). Intercept-centering and
        other variants should be applied as a pre-processing step in the
        results-side script before calling this function.

    Returns
    -------
    CondensationData
        One feature dimension per vector_id, ordered by (comparison_id,
        time_bin_center). feature_names lists the vector_ids.
    """
    if centering != "raw":
        raise ValueError(
            f"centering={centering!r} is not supported. Use 'raw' (the default). "
            "Apply other centering variants in your results-side script before "
            "calling from_classifier_directions."
        )

    from analyze.morphology_geometry.projection import project_binned_features

    meta = vd.metadata.sort_values(["comparison_id", "time_bin_center"]).reset_index(drop=True)
    bin_width = float(vd.inferred_bin_width)

    long_rows: list[dict] = []
    for idx, row in meta.iterrows():
        axis = vd.vectors[idx]
        proj = project_binned_features(
            df,
            vd=vd,
            axis=axis,
            id_col=id_col,
            time_col=time_col,
            bin_width=bin_width,
            class_col=label_col,
            output_col="_score",
        )
        for _, prow in proj.iterrows():
            long_rows.append({
                id_col: str(prow[id_col]),
                "time_bin_center": float(prow["time_bin_center"]),
                label_col: str(prow[label_col]) if label_col in proj.columns else "",
                "vector_id": str(row["vector_id"]),
                "_score": float(prow["_score"]),
            })

    long_df = pd.DataFrame(long_rows)
    wide = (
        long_df.pivot_table(
            index=[id_col, "time_bin_center", label_col],
            columns="vector_id",
            values="_score",
            aggfunc="first",
        )
        .reset_index()
    )
    wide.columns.name = None
    vector_ids = list(meta["vector_id"].unique())
    wide = wide[[id_col, "time_bin_center", label_col, *vector_ids]]

    return _build_canonical(
        wide,
        vector_ids,
        embryo_col=id_col,
        time_col="time_bin_center",
        label_col=label_col,
        allow_feature_nans=True,
    )


def from_multiclass_csv(
    path: str | Path,
    prob_cols: Sequence[str] | None = None,
    embryo_col: str = "embryo_id",
    time_col: str = "time_bin_center",
    label_col: str = "genotype",
) -> CondensationData:
    """Load a multiclass probability table and return canonical tensors.

    .. deprecated::
        Use from_classifier_directions() instead. Multiclass probability values
        are clipped by softmax saturation and do not give a clean geometric
        projection. This function is retained for historical script compatibility
        (results/mcolon/20260329_pbx_crispant_analysis_cont/).

    Parameters
    ----------
    path
        CSV with columns: embryo_id, time_bin_center, genotype, p_<condition>, ...
    prob_cols
        Probability column names. Auto-detected as columns starting with "p_"
        or "pred_proba_" if None.
    """
    warnings.warn(
        "from_multiclass_csv() is deprecated. Use from_classifier_directions() "
        "with a ValidatedDirections artifact instead. Multiclass probability "
        "values are clipped and give a distorted geometric projection.",
        DeprecationWarning,
        stacklevel=2,
    )
    df = pd.read_csv(path)

    if prob_cols is None:
        prob_cols = [
            c for c in df.columns
            if c.startswith("p_") or c.startswith("pred_proba_")
        ]
    if not prob_cols:
        raise ValueError(f"No probability columns found in {path}")

    return _build_canonical(df, prob_cols, embryo_col, time_col, label_col, allow_feature_nans=False)


def from_pairwise_margin_csv(
    path: str | Path,
    margin_cols: Sequence[str] | None = None,
    embryo_col: str = "embryo_id",
    time_col: str = "time_bin_center",
    label_col: str = "genotype",
) -> CondensationData:
    """Load a pairwise signed-margin table and return canonical tensors.

    .. deprecated::
        Use from_classifier_directions() instead. Pairwise margin values from
        the SVM contrast-coordinate path are clipped at the decision boundary and
        do not reflect a clean geometric projection onto the classifier axis.
        This function is retained for historical script compatibility
        (results/mcolon/20260329_pbx_crispant_analysis_cont/).

    Parameters
    ----------
    path
        CSV from phenotypic_positioning_phase2 (raw_position_vectors.csv
        or support_position_vectors.csv).
    margin_cols
        Pairwise margin column names. Auto-detected as columns containing
        "_vs_" if None.
    """
    warnings.warn(
        "from_pairwise_margin_csv() is deprecated. Use from_classifier_directions() "
        "with a ValidatedDirections artifact instead. Pairwise SVM margin values "
        "are clipped and give a distorted geometric projection.",
        DeprecationWarning,
        stacklevel=2,
    )
    df = pd.read_csv(path)

    if margin_cols is None:
        margin_cols = [c for c in df.columns if "_vs_" in c]
    if not margin_cols:
        raise ValueError(f"No pairwise margin columns found in {path}")

    return _build_canonical(df, margin_cols, embryo_col, time_col, label_col, allow_feature_nans=True)


def subset_pairwise(
    data: CondensationData,
    keep_genotypes: Sequence[str],
    *,
    drop_irrelevant_comparisons: bool = True,
    nan_policy: str = "warn_drop",
) -> CondensationData:
    """Subset a CondensationData to a specific set of genotypes.

    .. deprecated::
        This function is tied to the legacy contrast-coordinate path. For new
        work, filter the embryo df before calling from_classifier_directions().

    Parameters
    ----------
    data : CondensationData from from_pairwise_margin_csv()
    keep_genotypes : genotype labels to retain
    drop_irrelevant_comparisons : drop A__vs__B cols irrelevant to keep_genotypes
    nan_policy : "warn_drop" | "error" | "keep"
    """
    warnings.warn(
        "subset_pairwise() is deprecated. Filter your embryo dataframe before "
        "calling from_classifier_directions() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from ..classification.pairwise_outputs import subset_pairwise_cube

    r = subset_pairwise_cube(
        features=data.features,
        mask=data.mask,
        embryo_ids=data.embryo_ids,
        time_values=data.time_values,
        labels=data.labels,
        feature_names=data.feature_names,
        keep_genotypes=keep_genotypes,
        drop_irrelevant_comparisons=drop_irrelevant_comparisons,
        nan_policy=nan_policy,
    )
    embryo_index = {e: i for i, e in enumerate(r.embryo_ids)}
    time_index = {float(t): j for j, t in enumerate(r.time_values)}
    result = CondensationData(
        features=r.features,
        mask=r.mask,
        embryo_ids=r.embryo_ids,
        time_values=r.time_values,
        labels=r.labels,
        feature_names=r.feature_names,
        embryo_index=embryo_index,
        time_index=time_index,
    )
    validate(result, allow_feature_nans=True)
    return result


def _build_canonical(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    embryo_col: str,
    time_col: str,
    label_col: str,
    allow_feature_nans: bool = False,
) -> CondensationData:
    embryo_ids = np.array(sorted(df[embryo_col].unique()))
    time_values = np.array(sorted(df[time_col].unique()), dtype=float)
    N_e, T, K = len(embryo_ids), len(time_values), len(feature_cols)

    embryo_index = {e: i for i, e in enumerate(embryo_ids)}
    time_index = {float(t): j for j, t in enumerate(time_values)}

    features = np.full((N_e, T, K), np.nan, dtype=float)
    mask = np.zeros((N_e, T), dtype=bool)
    labels = np.full(N_e, "", dtype=object)
    label_sets: dict[int, set[str]] = {i: set() for i in range(N_e)}

    for _, row in df.iterrows():
        i = embryo_index[row[embryo_col]]
        j = time_index[float(row[time_col])]
        features[i, j, :] = row[list(feature_cols)].values.astype(float)
        mask[i, j] = True
        label_value = str(row[label_col])
        label_sets[i].add(label_value)
        if labels[i] == "":
            labels[i] = label_value

    for i, observed_labels in label_sets.items():
        if not observed_labels:
            continue
        if len(observed_labels) != 1:
            embryo_id = embryo_ids[i]
            raise AssertionError(
                f"embryo {embryo_id} has inconsistent labels across rows: {sorted(observed_labels)}"
            )

    data = CondensationData(
        features=features,
        mask=mask,
        embryo_ids=embryo_ids,
        time_values=time_values,
        labels=labels,
        feature_names=list(feature_cols),
        embryo_index=embryo_index,
        time_index=time_index,
    )
    validate(data, allow_feature_nans=allow_feature_nans)
    return data
