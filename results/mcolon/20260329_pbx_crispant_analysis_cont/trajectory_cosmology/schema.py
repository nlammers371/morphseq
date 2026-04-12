"""
schema.py
---------
Ingests upstream data in different formats and returns canonical tensors.

All downstream code works with CosmologyData only.
No raw DataFrames leak past this module.

Invariants enforced on every CosmologyData instance:
  - time_values is sorted ascending and shared across all embryos
  - features[i, t, :] is valid iff mask[i, t] is True
  - labels[i] is constant across time (genotype does not change per embryo)
  - shapes are mutually consistent (validated on construction)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class CosmologyData:
    """Canonical input contract for trajectory cosmology.

    Attributes
    ----------
    features : (N_e, T, K) float array
        Per-embryo per-timebin feature vectors.
        K = G for multiclass softmax, K = n_pairs for pairwise margins.
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


def validate(data: CosmologyData, *, allow_feature_nans: bool = False) -> None:
    """Assert all shape and semantic invariants on a CosmologyData instance.

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


def from_multiclass_csv(
    path: str | Path,
    prob_cols: Sequence[str] | None = None,
    embryo_col: str = "embryo_id",
    time_col: str = "time_bin_center",
    label_col: str = "genotype",
) -> CosmologyData:
    """Load a multiclass probability table and return canonical tensors.

    Parameters
    ----------
    path
        CSV with columns: embryo_id, time_bin_center, genotype, p_<condition>, ...
    prob_cols
        Probability column names. Auto-detected as columns starting with "p_"
        or "pred_proba_" if None.
    """
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
) -> CosmologyData:
    """Load a pairwise signed-margin table and return canonical tensors.

    Parameters
    ----------
    path
        CSV from phenotypic_positioning_phase2 (raw_position_vectors.csv
        or support_position_vectors.csv).
    margin_cols
        Pairwise margin column names. Auto-detected as columns containing
        "_vs_" if None.
    """
    df = pd.read_csv(path)

    if margin_cols is None:
        margin_cols = [c for c in df.columns if "_vs_" in c]
    if not margin_cols:
        raise ValueError(f"No pairwise margin columns found in {path}")

    return _build_canonical(df, margin_cols, embryo_col, time_col, label_col, allow_feature_nans=True)


def _build_canonical(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    embryo_col: str,
    time_col: str,
    label_col: str,
    allow_feature_nans: bool = False,
) -> CosmologyData:
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

    data = CosmologyData(
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
