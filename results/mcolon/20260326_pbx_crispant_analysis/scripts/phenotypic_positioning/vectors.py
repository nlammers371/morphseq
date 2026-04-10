from __future__ import annotations

import numpy as np
import pandas as pd


INDEX_COLS = ["embryo_id", "genotype", "experiment_id", "_time_bin", "time_bin_center"]


def build_vector_table(
    axis_df: pd.DataFrame,
    *,
    value_col: str,
    fill_value: float,
) -> tuple[pd.DataFrame, list[str]]:
    pivot = axis_df.pivot_table(
        index=INDEX_COLS,
        columns="pair_id",
        values=value_col,
        aggfunc="mean",
    )
    pair_cols = sorted(pivot.columns.tolist())
    pivot = pivot.reindex(columns=pair_cols)
    pivot = pivot.fillna(float(fill_value)).reset_index()
    return pivot, pair_cols


def summarize_pairwise_support(axis_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        axis_df.groupby(["pair_id", "group1", "group2", "_time_bin", "time_bin_center"], as_index=False)
        .agg(
            n_rows=("embryo_id", "count"),
            n_supported=("model_available", "sum"),
            median_support_weight=("support_weight", "median"),
            mean_support_weight=("support_weight", "mean"),
            median_axis_residual_z=("axis_residual_z", "median"),
            median_knn_novelty_z=("knn_novelty_z", "median"),
            mean_position_sd=("position_logit_sd", "mean"),
        )
    )
    summary["support_fraction"] = np.where(
        summary["n_rows"] > 0,
        summary["n_supported"] / summary["n_rows"],
        np.nan,
    )
    return summary.sort_values(["pair_id", "_time_bin"]).reset_index(drop=True)

