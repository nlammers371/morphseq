"""
schema.py
---------
Ingests upstream data in different formats and returns canonical tensors.

All downstream code works with the CosmologyData namedtuple only.
No file I/O happens outside this module.
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

    features : (N_e, T, K) float array
        Per-embryo per-timebin feature vectors.
        K = G for multiclass softmax, K = n_pairs for pairwise margins.
    mask : (N_e, T) bool array
        True where embryo i is observed at time bin t.
    embryo_ids : (N_e,) array of str
    time_values : (T,) float array  — time bin centers in hpf
    labels : (N_e,) array of str   — genotype labels
    feature_names : list of str    — column names for the K feature dimensions
    """
    features: np.ndarray
    mask: np.ndarray
    embryo_ids: np.ndarray
    time_values: np.ndarray
    labels: np.ndarray
    feature_names: list[str]


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
        Path to CSV with columns: embryo_id, time_bin_center, genotype,
        p_<condition>, ...
    prob_cols
        Names of the probability columns. If None, auto-detected as columns
        starting with "p_".
    """
    df = pd.read_csv(path)

    if prob_cols is None:
        prob_cols = [c for c in df.columns if c.startswith("p_")]
    if not prob_cols:
        raise ValueError(f"No probability columns found in {path}")

    return _build_canonical(df, prob_cols, embryo_col, time_col, label_col)


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
        Path to CSV produced by phenotypic_positioning_phase2
        (raw_position_vectors.csv or support_position_vectors.csv).
    margin_cols
        Names of the pairwise margin columns. If None, auto-detected as
        columns containing "_vs_".
    """
    df = pd.read_csv(path)

    if margin_cols is None:
        margin_cols = [c for c in df.columns if "_vs_" in c]
    if not margin_cols:
        raise ValueError(f"No pairwise margin columns found in {path}")

    return _build_canonical(df, margin_cols, embryo_col, time_col, label_col)


def _build_canonical(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    embryo_col: str,
    time_col: str,
    label_col: str,
) -> CosmologyData:
    embryo_ids = np.array(sorted(df[embryo_col].unique()))
    time_values = np.array(sorted(df[time_col].unique()), dtype=float)
    N_e, T, K = len(embryo_ids), len(time_values), len(feature_cols)

    embryo_index = {e: i for i, e in enumerate(embryo_ids)}
    time_index = {t: j for j, t in enumerate(time_values)}

    features = np.full((N_e, T, K), np.nan, dtype=float)
    mask = np.zeros((N_e, T), dtype=bool)
    labels = np.full(N_e, "", dtype=object)

    for _, row in df.iterrows():
        i = embryo_index[row[embryo_col]]
        j = time_index[row[time_col]]
        features[i, j, :] = row[list(feature_cols)].values.astype(float)
        mask[i, j] = True
        if labels[i] == "":
            labels[i] = row[label_col]

    return CosmologyData(
        features=features,
        mask=mask,
        embryo_ids=embryo_ids,
        time_values=time_values,
        labels=labels,
        feature_names=list(feature_cols),
    )
