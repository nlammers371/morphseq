"""Support metadata helpers for pairwise contrast coordinates."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


SUPPORT_STATUS_SUPPORTED = "supported"
SUPPORT_STATUS_UNSUPPORTED_ID = "unsupported_id"
SUPPORT_STATUS_UNSUPPORTED_GROUP = "unsupported_group"


def assemble_contrast_support(
    support_rows: list[dict[str, Any]],
    scores: pd.DataFrame,
    *,
    id_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return embryo-level support rows and comparison/bin specificity rows."""
    support_long = pd.DataFrame(support_rows)
    if support_long.empty:
        raise ValueError("No support rows were collected for contrast support assembly.")

    required = {
        "feature_set",
        "comparison_id",
        id_col,
        "group_label",
        "positive_label",
        "negative_label",
        "time_bin",
        "time_bin_center",
        "group_supported",
        "id_supported",
        "support_status",
    }
    missing = required.difference(support_long.columns)
    if missing:
        raise ValueError(f"contrast support rows missing required columns: {sorted(missing)}")

    support_long = support_long.copy()
    support_long["group_supported"] = support_long["group_supported"].astype(bool)
    support_long["id_supported"] = support_long["id_supported"].astype(bool)
    support_long = support_long.sort_values(
        ["feature_set", "comparison_id", "time_bin", "group_label", id_col]
    ).reset_index(drop=True)

    group_support = (
        support_long.groupby(
            [
                "feature_set",
                "comparison_id",
                "positive_label",
                "negative_label",
                "time_bin",
                "time_bin_center",
                "group_label",
            ],
            as_index=False,
        )
        .agg(
            group_supported=("group_supported", "max"),
            n_group_ids=(id_col, "nunique"),
            n_supported_ids=("id_supported", "sum"),
        )
    )

    pos = group_support[group_support["group_label"] == group_support["positive_label"]].copy()
    neg = group_support[group_support["group_label"] == group_support["negative_label"]].copy()
    join_cols = [
        "feature_set",
        "comparison_id",
        "positive_label",
        "negative_label",
        "time_bin",
        "time_bin_center",
    ]
    specificity = pos[join_cols + ["group_supported", "n_group_ids", "n_supported_ids"]].rename(
        columns={
            "group_supported": "positive_group_supported",
            "n_group_ids": "positive_group_n_ids",
            "n_supported_ids": "positive_n",
        }
    ).merge(
        neg[join_cols + ["group_supported", "n_group_ids", "n_supported_ids"]].rename(
            columns={
                "group_supported": "negative_group_supported",
                "n_group_ids": "negative_group_n_ids",
                "n_supported_ids": "negative_n",
            }
        ),
        on=join_cols,
        how="outer",
        validate="one_to_one",
    )

    specificity["min_group_support_passed"] = (
        specificity["positive_group_supported"].fillna(False).astype(bool)
        & specificity["negative_group_supported"].fillna(False).astype(bool)
    )

    score_cols = [
        "feature_set",
        "comparison_id",
        "positive_label",
        "negative_label",
        "time_bin",
        "time_bin_center",
        "auroc_obs",
        "auroc_null_mean",
        "auroc_null_std",
        "pval",
    ]
    merged = specificity.merge(
        scores[score_cols],
        on=join_cols,
        how="left",
        validate="one_to_one",
    )
    merged["w"] = np.where(
        merged["min_group_support_passed"].fillna(False),
        np.clip(
            (merged["auroc_obs"].astype(float) - merged["auroc_null_mean"].astype(float)) / 0.5,
            0.0,
            1.0,
        ),
        np.nan,
    )

    merged = merged.sort_values(["feature_set", "comparison_id", "time_bin", "time_bin_center"]).reset_index(drop=True)
    return support_long, merged
