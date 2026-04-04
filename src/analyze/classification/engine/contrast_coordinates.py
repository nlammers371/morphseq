"""Assembly helpers for pairwise contrast-coordinate artifacts."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .contrast_support import assemble_contrast_support


def _validate_id_metadata(
    id_metadata: pd.DataFrame,
    *,
    id_col: str,
    class_col: str,
) -> pd.DataFrame:
    required = {id_col, class_col}
    missing = required.difference(id_metadata.columns)
    if missing:
        raise ValueError(f"id_metadata missing required columns: {sorted(missing)}")

    cleaned = id_metadata[[id_col, class_col]].drop_duplicates().copy()
    dupes = cleaned.duplicated(subset=[id_col], keep=False)
    if dupes.any():
        bad_ids = cleaned.loc[dupes, id_col].astype(str).unique().tolist()
        preview = ", ".join(bad_ids[:5])
        raise ValueError(
            f"{id_col!r} must map to exactly one {class_col!r}; duplicates found for {preview}"
        )
    return cleaned.rename(columns={class_col: "genotype"}).reset_index(drop=True)


def _ordered_probe_ids(scores: pd.DataFrame) -> list[str]:
    pairs = (
        scores[["feature_set", "comparison_id"]]
        .drop_duplicates()
        .sort_values(["feature_set", "comparison_id"])
    )
    return pairs["comparison_id"].drop_duplicates().tolist()


def _pivot_coordinates(
    rows: pd.DataFrame,
    *,
    id_col: str,
    value_col: str,
    probe_columns: list[str],
) -> pd.DataFrame:
    index_cols = ["feature_set", id_col, "genotype", "time_bin", "time_bin_center"]
    pivot = rows.pivot_table(
        index=index_cols,
        columns="comparison_id",
        values=value_col,
        aggfunc="mean",
    )
    pivot = pivot.reindex(columns=probe_columns)
    pivot.columns.name = None
    return pivot.reset_index().sort_values(["feature_set", "time_bin", id_col]).reset_index(drop=True)


def _subtract_coordinates(
    raw_coordinates: pd.DataFrame,
    shrunk_coordinates: pd.DataFrame,
    *,
    probe_columns: list[str],
) -> pd.DataFrame:
    residual = raw_coordinates.copy()
    residual.loc[:, probe_columns] = (
        raw_coordinates[probe_columns].astype(float) - shrunk_coordinates[probe_columns].astype(float)
    )
    return residual


def _build_probe_index(
    scores: pd.DataFrame,
    *,
    probe_columns: list[str],
) -> pd.DataFrame:
    base = (
        scores[["feature_set", "comparison_id", "positive_label", "negative_label"]]
        .drop_duplicates()
        .sort_values(["feature_set", "comparison_id"])
        .reset_index(drop=True)
    )
    rows: list[dict[str, Any]] = []
    for feature_set, group in base.groupby("feature_set", sort=True):
        group = group.reset_index(drop=True)
        order_map = {comparison_id: i for i, comparison_id in enumerate(group["comparison_id"].tolist())}
        for _, row in group.iterrows():
            rows.append({
                "feature_set": feature_set,
                "column_name": row["comparison_id"],
                "column_order": int(order_map[row["comparison_id"]]),
                "comparison_id": row["comparison_id"],
                "positive_label": row["positive_label"],
                "negative_label": row["negative_label"],
            })
    probe_index = pd.DataFrame(rows)
    if probe_index.empty:
        return probe_index
    probe_index = probe_index[probe_index["column_name"].isin(probe_columns)].copy()
    return probe_index.sort_values(["feature_set", "column_order", "comparison_id"]).reset_index(drop=True)


def assemble_contrast_coordinates(
    margin_rows: list[dict[str, Any]],
    support_rows: list[dict[str, Any]],
    scores: pd.DataFrame,
    id_metadata: pd.DataFrame,
    id_col: str,
    class_col: str,
) -> dict[str, pd.DataFrame]:
    """Assemble pairwise contrast-coordinate artifacts from raw margins and scores."""
    if not margin_rows:
        raise ValueError("No margin rows were collected for contrast-coordinate assembly.")

    margins = pd.DataFrame(margin_rows)
    metadata = _validate_id_metadata(id_metadata, id_col=id_col, class_col=class_col)
    margins = margins.merge(metadata, on=id_col, how="left", validate="many_to_one")
    if margins["genotype"].isna().any():
        raise ValueError("Some margin rows could not be mapped back to genotype metadata.")

    raw_long = margins[
        ["feature_set", id_col, "genotype", "time_bin", "time_bin_center", "comparison_id", "m_raw"]
    ].copy().sort_values(["feature_set", "comparison_id", "time_bin", id_col]).reset_index(drop=True)

    contrast_support_long, specificity = assemble_contrast_support(
        support_rows,
        scores,
        id_col=id_col,
    )
    probe_columns = _ordered_probe_ids(scores)

    shrunk_long = raw_long.merge(
        specificity[["feature_set", "comparison_id", "time_bin", "time_bin_center", "w"]],
        on=["feature_set", "comparison_id", "time_bin", "time_bin_center"],
        how="left",
        validate="many_to_one",
    )
    if shrunk_long["w"].isna().any():
        raise ValueError("Missing shrinkage weights for some margin rows.")
    shrunk_long["m_shrunk"] = shrunk_long["w"].astype(float) * shrunk_long["m_raw"].astype(float)

    raw_coordinates = _pivot_coordinates(
        raw_long,
        id_col=id_col,
        value_col="m_raw",
        probe_columns=probe_columns,
    )
    shrunk_coordinates = _pivot_coordinates(
        shrunk_long,
        id_col=id_col,
        value_col="m_shrunk",
        probe_columns=probe_columns,
    )
    residual_coordinates = _subtract_coordinates(
        raw_coordinates,
        shrunk_coordinates,
        probe_columns=probe_columns,
    )
    probe_index = _build_probe_index(scores, probe_columns=probe_columns)

    return {
        "raw_contrast_scores_long": raw_long,
        "contrast_support_long": contrast_support_long,
        "contrast_specificity_by_timebin": specificity,
        "raw_coordinates": raw_coordinates,
        "shrunk_coordinates": shrunk_coordinates,
        "residual_coordinates": residual_coordinates,
        "probe_index": probe_index,
    }
