from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


META_COLS = [
    "embryo_id",
    "genotype",
    "true_class",
    "pred_class",
    "experiment_id",
    "time_bin",
    "_time_bin",
    "time_bin_center",
    "feature_set",
    "is_correct",
    "is_wrong",
    "p_true",
    "p_pred",
]


def probability_columns_for_labels(class_labels: list[str]) -> list[str]:
    return [f"pred_proba_{label}" for label in class_labels]


def prepare_multiclass_predictions(
    pred_df: pd.DataFrame,
    *,
    embryo_meta: pd.DataFrame,
    class_labels: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    if pred_df.empty:
        raise ValueError("Multiclass prediction table is empty.")

    prob_cols = probability_columns_for_labels(class_labels)
    missing = sorted(set(prob_cols).difference(pred_df.columns))
    if missing:
        raise ValueError(f"Missing probability columns: {missing}")

    meta = embryo_meta[["embryo_id", "experiment_id"]].drop_duplicates()
    prepared = pred_df.copy()
    prepared["time_bin"] = prepared["time_bin"].astype(int)
    prepared["_time_bin"] = prepared["time_bin"].astype(int)
    prepared["genotype"] = prepared["true_class"].astype(str)
    prepared = prepared.merge(meta, on="embryo_id", how="left", validate="many_to_one")
    prepared = prepared[
        META_COLS + prob_cols
    ].sort_values(["_time_bin", "embryo_id"]).reset_index(drop=True)

    prob_sum = prepared[prob_cols].sum(axis=1).to_numpy(dtype=float)
    if not np.allclose(prob_sum, 1.0, atol=1e-6):
        raise ValueError("Multiclass probability rows do not sum to 1.")

    return prepared, prob_cols


def build_multiclass_probability_vectors(
    pred_df: pd.DataFrame,
    *,
    class_labels: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    prob_cols = probability_columns_for_labels(class_labels)
    missing = sorted(set(prob_cols).difference(pred_df.columns))
    if missing:
        raise ValueError(f"Missing probability columns: {missing}")
    vectors = pred_df[META_COLS + prob_cols].copy()
    return vectors.sort_values(["_time_bin", "embryo_id"]).reset_index(drop=True), prob_cols


def build_multiclass_logit_vectors(
    vector_df: pd.DataFrame,
    *,
    class_labels: list[str],
    eps: float = 1e-6,
) -> tuple[pd.DataFrame, list[str]]:
    prob_cols = probability_columns_for_labels(class_labels)
    logit_cols = [f"logit_{label}" for label in class_labels]
    logits = vector_df[META_COLS].copy()
    for label, prob_col, logit_col in zip(class_labels, prob_cols, logit_cols):
        del label
        p = np.clip(vector_df[prob_col].to_numpy(dtype=float), eps, 1.0 - eps)
        logits[logit_col] = np.log(p / (1.0 - p))
    return logits, logit_cols


def prepare_multiclass_confusion_summary(
    confusion_df: pd.DataFrame | None,
    *,
    class_labels: list[str],
    bin_width: float,
) -> pd.DataFrame:
    if confusion_df is None or confusion_df.empty:
        return pd.DataFrame(
            columns=[
                "feature_set",
                "time_bin",
                "time_bin_center",
                "true_class",
                "predicted_class",
                "proportion",
                "count",
                "is_correct",
            ]
        )

    prepared = confusion_df.copy()
    if "time_bin" not in prepared.columns:
        prepared["time_bin"] = np.round(
            prepared["time_bin_center"].astype(float) - float(bin_width) / 2.0
        ).astype(int)
    else:
        prepared["time_bin"] = prepared["time_bin"].astype(int)
    prepared = prepared[prepared["true_class"].isin(class_labels) & prepared["predicted_class"].isin(class_labels)].copy()
    keep_cols = [
        "feature_set",
        "time_bin",
        "time_bin_center",
        "true_class",
        "predicted_class",
        "proportion",
        "count",
        "is_correct",
    ]
    return prepared[keep_cols].sort_values(["time_bin", "true_class", "predicted_class"]).reset_index(drop=True)


def summarize_multiclass_centroids(
    vector_df: pd.DataFrame,
    *,
    class_labels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prob_cols = probability_columns_for_labels(class_labels)
    centroids = (
        vector_df.groupby(["time_bin", "time_bin_center", "genotype"], as_index=False)[prob_cols]
        .median()
        .sort_values(["time_bin", "genotype"])
        .reset_index(drop=True)
    )

    rows: list[dict[str, object]] = []
    for (time_bin, time_bin_center), group in centroids.groupby(["time_bin", "time_bin_center"]):
        for genotype_1, genotype_2 in itertools.combinations(group["genotype"].tolist(), 2):
            row_1 = group[group["genotype"] == genotype_1].iloc[0]
            row_2 = group[group["genotype"] == genotype_2].iloc[0]
            v1 = row_1[prob_cols].to_numpy(dtype=float)
            v2 = row_2[prob_cols].to_numpy(dtype=float)
            rows.append(
                {
                    "time_bin": int(time_bin),
                    "time_bin_center": float(time_bin_center),
                    "genotype_1": genotype_1,
                    "genotype_2": genotype_2,
                    "distance_l2": float(np.linalg.norm(v1 - v2)),
                    "distance_l1": float(np.abs(v1 - v2).sum()),
                }
            )
    distances = pd.DataFrame(rows).sort_values(["time_bin", "genotype_1", "genotype_2"]).reset_index(drop=True)
    return centroids, distances


def summarize_probability_trajectories(
    vector_df: pd.DataFrame,
    *,
    class_labels: list[str],
) -> pd.DataFrame:
    prob_cols = probability_columns_for_labels(class_labels)
    summary = (
        vector_df.groupby(["time_bin", "time_bin_center", "genotype"], as_index=False)[prob_cols]
        .median()
        .sort_values(["time_bin", "genotype"])
        .reset_index(drop=True)
    )
    long_df = summary.melt(
        id_vars=["time_bin", "time_bin_center", "genotype"],
        value_vars=prob_cols,
        var_name="probability_column",
        value_name="median_probability",
    )
    long_df["predicted_class"] = long_df["probability_column"].str.replace("pred_proba_", "", regex=False)
    return long_df.drop(columns=["probability_column"]).sort_values(
        ["genotype", "predicted_class", "time_bin"]
    ).reset_index(drop=True)
