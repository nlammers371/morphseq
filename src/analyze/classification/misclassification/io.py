from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Stage1Metadata:
    class_labels: list[str]
    time_bin_definition: list[int] | list[float]
    time_bin_center_formula: str
    time_bin_edges_sha256: str
    seed: int
    n_permutations: int
    git_commit: str
    timestamp: str
    schema_version: str


def load_stage1_metadata(input_dir: Path) -> Stage1Metadata:
    p = input_dir / "null" / "null_metadata.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing Stage 1 metadata: {p}")

    meta = json.loads(p.read_text())
    required = {
        "class_labels",
        "time_bin_definition",
        "time_bin_center_formula",
        "time_bin_edges_sha256",
        "seed",
        "n_permutations",
        "git_commit",
        "timestamp",
        "schema_version",
    }
    missing = required - set(meta.keys())
    if missing:
        raise ValueError(f"Stage 1 metadata missing keys: {sorted(missing)}")

    return Stage1Metadata(
        class_labels=list(meta["class_labels"]),
        time_bin_definition=list(meta["time_bin_definition"]),
        time_bin_center_formula=str(meta["time_bin_center_formula"]),
        time_bin_edges_sha256=str(meta["time_bin_edges_sha256"]),
        seed=int(meta["seed"]),
        n_permutations=int(meta["n_permutations"]),
        git_commit=str(meta["git_commit"]),
        timestamp=str(meta["timestamp"]),
        schema_version=str(meta["schema_version"]),
    )


def infer_class_labels_from_predictions(df: pd.DataFrame) -> list[str]:
    labels = sorted(set(df["true_class"].astype(str).unique()) | set(df["pred_class"].astype(str).unique()))
    if not labels:
        raise ValueError("Could not infer class labels from embryo predictions")
    return labels


def load_embryo_predictions(input_dir: Path) -> pd.DataFrame:
    p = input_dir / "embryo_predictions_augmented.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing Stage 1 embryo predictions: {p}")
    return pd.read_parquet(p)


def validate_stage2_inputs(
    df: pd.DataFrame,
    *,
    class_labels: list[str],
    require_strict_time_order_within_embryo: bool = True,
) -> None:
    required_cols = {
        "embryo_id",
        "time_bin",
        "time_bin_center",
        "true_class",
        "pred_class",
        "p_true",
        "p_pred",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for c in class_labels:
        col = f"pred_proba_{c}"
        if col not in df.columns:
            raise ValueError(f"Missing required probability column: {col}")

    critical_no_na = ["embryo_id", "time_bin", "true_class", "pred_class"]
    for col in critical_no_na:
        if df[col].isna().any():
            raise ValueError(f"Column '{col}' contains NA values")

    if not (pd.api.types.is_object_dtype(df["embryo_id"]) or pd.api.types.is_string_dtype(df["embryo_id"])):
        raise ValueError(f"embryo_id must be object/string dtype; got {df['embryo_id'].dtype}")
    if not pd.api.types.is_integer_dtype(df["time_bin"]):
        raise ValueError(f"time_bin must be integer dtype; got {df['time_bin'].dtype}")
    if not pd.api.types.is_float_dtype(df["time_bin_center"]):
        raise ValueError(f"time_bin_center must be float dtype; got {df['time_bin_center'].dtype}")
    if not pd.api.types.is_float_dtype(df["p_true"]):
        raise ValueError(f"p_true must be float dtype; got {df['p_true'].dtype}")
    if not pd.api.types.is_float_dtype(df["p_pred"]):
        raise ValueError(f"p_pred must be float dtype; got {df['p_pred'].dtype}")

    bad_true = set(df["true_class"].astype(str).unique()) - set(class_labels)
    bad_pred = set(df["pred_class"].astype(str).unique()) - set(class_labels)
    if bad_true:
        raise ValueError(f"true_class has labels outside class_labels: {sorted(bad_true)}")
    if bad_pred:
        raise ValueError(f"pred_class has labels outside class_labels: {sorted(bad_pred)}")

    prob_cols = [f"pred_proba_{c}" for c in class_labels] + ["p_true", "p_pred"]
    for col in prob_cols:
        x = df[col].to_numpy(dtype=float)
        if np.isnan(x).any():
            raise ValueError(f"Probability column '{col}' contains NaN")
        if np.any((x < -1e-6) | (x > 1 + 1e-6)):
            mn, mx = float(np.min(x)), float(np.max(x))
            raise ValueError(f"Probability column '{col}' out of [0,1] (tol): min={mn:.4f}, max={mx:.4f}")

    if require_strict_time_order_within_embryo:
        for embryo_id, grp in df.sort_values(["embryo_id", "time_bin"]).groupby("embryo_id"):
            bins = grp["time_bin"].to_numpy(dtype=int)
            if len(bins) > 1 and not np.all(np.diff(bins) > 0):
                raise ValueError(
                    f"embryo {embryo_id}: time_bin not strictly increasing (duplicates/unsorted): {bins.tolist()}"
                )
