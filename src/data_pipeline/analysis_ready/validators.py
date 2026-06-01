from __future__ import annotations

import pandas as pd

from data_pipeline.io.validators import validate_dataframe_schema
from data_pipeline.schemas.analysis_ready import REQUIRED_COLUMNS_ANALYSIS_READY


def assert_unique_snip_id(df: pd.DataFrame, *, key: str = "snip_id") -> None:
    if key not in df.columns:
        raise ValueError(f"Missing required key column: {key}")
    if df[key].duplicated().any():
        dupes = df.loc[df[key].duplicated(), key].astype(str).tolist()
        raise ValueError(f"Duplicate {key} values detected: {dupes[:10]}")


def assert_no_column_collisions(left_df: pd.DataFrame, right_df: pd.DataFrame, *, key: str = "snip_id") -> None:
    collisions = sorted((set(left_df.columns) & set(right_df.columns)) - {key})
    if collisions:
        raise ValueError(f"Column name collisions detected before join: {collisions}")


def assert_matching_snip_id_sets(left_df: pd.DataFrame, right_df: pd.DataFrame, *, key: str = "snip_id") -> None:
    left_ids = set(left_df[key].astype(str).tolist())
    right_ids = set(right_df[key].astype(str).tolist())
    missing_left = sorted(right_ids - left_ids)
    missing_right = sorted(left_ids - right_ids)
    if missing_left or missing_right:
        raise ValueError(
            "Join key mismatch detected: "
            f"missing_in_left={missing_left[:10]}, missing_in_right={missing_right[:10]}"
        )


def assert_1to1_join(left_df: pd.DataFrame, right_df: pd.DataFrame, *, key: str = "snip_id") -> None:
    assert_unique_snip_id(left_df, key=key)
    assert_unique_snip_id(right_df, key=key)
    assert_matching_snip_id_sets(left_df, right_df, key=key)
    if len(left_df) != len(right_df):
        raise ValueError(
            f"Join row-count mismatch: left={len(left_df)} right={len(right_df)}"
        )


def validate_analysis_ready(df: pd.DataFrame) -> None:
    validate_dataframe_schema(
        df,
        REQUIRED_COLUMNS_ANALYSIS_READY,
        "analysis_ready.csv",
        nullable_columns=["death_inflection_time_int", "death_predicted_stage_hpf"],
    )
    assert_unique_snip_id(df)

    if "embedding_calculated" not in df.columns:
        raise ValueError("analysis_ready.csv missing embedding_calculated")
    if df["embedding_calculated"].isna().any():
        raise ValueError("embedding_calculated contains null values")
    if not pd.api.types.is_bool_dtype(df["embedding_calculated"]):
        raise ValueError("embedding_calculated must be boolean dtype")
