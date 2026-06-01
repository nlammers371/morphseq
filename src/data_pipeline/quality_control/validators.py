from __future__ import annotations

import pandas as pd

from data_pipeline.io.validators import validate_dataframe_schema
from data_pipeline.schemas.quality_control import (
    REQUIRED_COLUMNS_QC,
    SNIP_EXCLUSION_FLAGS,
    SNIP_INFORMATIONAL_FLAGS,
)


def assert_unique_snip_id(df: pd.DataFrame, *, key: str = "snip_id") -> None:
    if key not in df.columns:
        raise ValueError(f"Missing required key column: {key}")
    if df[key].isna().any():
        raise ValueError(f"Column '{key}' contains null values")
    if df[key].duplicated().any():
        raise ValueError(f"Duplicate {key} values detected")


def assert_no_column_collisions(left_df: pd.DataFrame, right_df: pd.DataFrame, *, key: str = "snip_id") -> None:
    collisions = sorted((set(left_df.columns) & set(right_df.columns)) - {key})
    if collisions:
        raise ValueError(f"Column name collisions detected before join: {collisions}")


def assert_matching_snip_id_sets(left_df: pd.DataFrame, right_df: pd.DataFrame, *, key: str = "snip_id") -> None:
    left_ids = set(left_df[key].astype(str))
    right_ids = set(right_df[key].astype(str))
    missing_left = sorted(right_ids - left_ids)
    missing_right = sorted(left_ids - right_ids)
    if missing_left or missing_right:
        raise ValueError(
            f"Join key mismatch detected: missing_in_left={missing_left[:10]} missing_in_right={missing_right[:10]}"
        )


def assert_1to1_join(left_df: pd.DataFrame, right_df: pd.DataFrame, *, key: str = "snip_id") -> None:
    assert_unique_snip_id(left_df, key=key)
    assert_unique_snip_id(right_df, key=key)
    assert_matching_snip_id_sets(left_df, right_df, key=key)
    if len(left_df) != len(right_df):
        raise ValueError(f"Join row-count mismatch: left={len(left_df)} right={len(right_df)}")


def _validate_boolean_columns(df: pd.DataFrame, columns: list[str], stage_name: str) -> None:
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"Missing boolean column '{column}' in {stage_name}")
        if df[column].isna().any():
            raise ValueError(f"Column '{column}' contains null values in {stage_name}")
        if not pd.api.types.is_bool_dtype(df[column]):
            raise ValueError(f"Column '{column}' must be boolean dtype in {stage_name}")


def _assert_exact_columns(df: pd.DataFrame, expected_columns: list[str], stage_name: str) -> None:
    actual_columns = list(df.columns)
    missing = [column for column in expected_columns if column not in actual_columns]
    extra = [column for column in actual_columns if column not in expected_columns]
    if missing or extra:
        raise ValueError(f"{stage_name}: column mismatch (missing={missing}, extra={extra})")


def validate_segmentation_qc_flags(df: pd.DataFrame) -> None:
    expected = ["snip_id", "edge_flag", "discontinuous_mask_flag", "overlapping_mask_flag"]
    validate_dataframe_schema(df, expected, "segmentation_qc_flags.csv")
    _assert_exact_columns(df, expected, "segmentation_qc_flags.csv")
    assert_unique_snip_id(df)
    _validate_boolean_columns(df, ["edge_flag", "discontinuous_mask_flag", "overlapping_mask_flag"], "segmentation_qc_flags.csv")


def validate_viability_qc_flags(df: pd.DataFrame) -> None:
    expected = ["snip_id", "viability_flag"]
    validate_dataframe_schema(df, expected, "viability_qc_flags.csv")
    _assert_exact_columns(df, expected, "viability_qc_flags.csv")
    assert_unique_snip_id(df)
    _validate_boolean_columns(df, ["viability_flag"], "viability_qc_flags.csv")


def validate_death_detection_flags(df: pd.DataFrame) -> None:
    expected = ["snip_id", "dead_flag", "death_inflection_time_int", "death_predicted_stage_hpf"]
    validate_dataframe_schema(
        df,
        expected,
        "death_detection_flags.csv",
        nullable_columns=["death_inflection_time_int", "death_predicted_stage_hpf"],
    )
    _assert_exact_columns(df, expected, "death_detection_flags.csv")
    assert_unique_snip_id(df)
    _validate_boolean_columns(df, ["dead_flag"], "death_detection_flags.csv")
    if df.loc[df["dead_flag"].astype(bool), "death_inflection_time_int"].isna().any():
        raise ValueError("death_inflection_time_int must be populated when dead_flag is True")
    if df.loc[df["dead_flag"].astype(bool), "death_predicted_stage_hpf"].isna().any():
        raise ValueError("death_predicted_stage_hpf must be populated when dead_flag is True")


def validate_surface_area_qc_flags(df: pd.DataFrame) -> None:
    expected = ["snip_id", "sa_outlier_flag"]
    validate_dataframe_schema(df, expected, "surface_area_qc_flags.csv")
    _assert_exact_columns(df, expected, "surface_area_qc_flags.csv")
    assert_unique_snip_id(df)
    _validate_boolean_columns(df, ["sa_outlier_flag"], "surface_area_qc_flags.csv")


def validate_auxiliary_mask_qc_flags(df: pd.DataFrame) -> None:
    expected = ["snip_id", "yolk_flag", "bubble_flag"]
    validate_dataframe_schema(df, expected, "auxiliary_mask_qc_flags.csv")
    _assert_exact_columns(df, expected, "auxiliary_mask_qc_flags.csv")
    assert_unique_snip_id(df)
    _validate_boolean_columns(df, ["yolk_flag", "bubble_flag"], "auxiliary_mask_qc_flags.csv")


def validate_focus_qc_flags(df: pd.DataFrame) -> None:
    expected = ["snip_id", "focus_flag"]
    validate_dataframe_schema(df, expected, "focus_qc_flags.csv")
    _assert_exact_columns(df, expected, "focus_qc_flags.csv")
    assert_unique_snip_id(df)
    _validate_boolean_columns(df, ["focus_flag"], "focus_qc_flags.csv")


def validate_motion_qc_flags(df: pd.DataFrame) -> None:
    expected = ["snip_id", "motion_flag"]
    validate_dataframe_schema(df, expected, "motion_qc_flags.csv")
    _assert_exact_columns(df, expected, "motion_qc_flags.csv")
    assert_unique_snip_id(df)
    _validate_boolean_columns(df, ["motion_flag"], "motion_qc_flags.csv")


def validate_qc_flags(df: pd.DataFrame) -> None:
    validate_dataframe_schema(
        df,
        REQUIRED_COLUMNS_QC,
        "qc_flags.csv",
        nullable_columns=["death_inflection_time_int", "death_predicted_stage_hpf"],
    )
    _assert_exact_columns(df, REQUIRED_COLUMNS_QC, "qc_flags.csv")
    assert_unique_snip_id(df)
    _validate_boolean_columns(df, ["use_snip", *SNIP_EXCLUSION_FLAGS, *SNIP_INFORMATIONAL_FLAGS], "qc_flags.csv")
    if "dead_flag" in df.columns and "death_inflection_time_int" in df.columns:
        if df.loc[~df["dead_flag"].astype(bool), "death_inflection_time_int"].notna().any():
            raise ValueError("death_inflection_time_int must be null when dead_flag is False")
    if "use_snip" in df.columns:
        expected = ~df[SNIP_EXCLUSION_FLAGS].any(axis=1)
        if not expected.equals(df["use_snip"]):
            raise ValueError("use_snip does not match SNIP_EXCLUSION_FLAGS")
