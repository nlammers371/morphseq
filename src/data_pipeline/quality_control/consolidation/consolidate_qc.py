#!/usr/bin/env python3
"""QC consolidation module.

This module aligns the validated per-stage QC tables on ``snip_id``, derives
the final ``use_snip`` gate, and writes the consolidated QC contract.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pipeline.schemas.quality_control import (
    REQUIRED_COLUMNS_QC,
    SNIP_EXCLUSION_FLAGS,
    SNIP_INFORMATIONAL_FLAGS,
)


def assert_unique_snip_id(df: pd.DataFrame, stage_name: str, *, key: str = "snip_id") -> None:
    if key not in df.columns:
        raise ValueError(f"{stage_name}: missing required key column '{key}'")
    if df[key].isna().any():
        raise ValueError(f"{stage_name}: column '{key}' contains null values")
    if df[key].duplicated().any():
        raise ValueError(f"{stage_name}: duplicate '{key}' values detected")


def _exact_columns(df: pd.DataFrame, expected: list[str], stage_name: str) -> pd.DataFrame:
    missing = [column for column in expected if column not in df.columns]
    extra = [column for column in df.columns if column not in expected]
    if missing or extra:
        raise ValueError(f"{stage_name}: column mismatch (missing={missing}, extra={extra})")
    return df[expected].copy()


def _normalize_death_detection(death_qc_df: pd.DataFrame) -> pd.DataFrame:
    if "dead_flag" not in death_qc_df.columns:
        raise ValueError("death_detection: expected dead_flag column")
    expected = ["snip_id", "dead_flag", "death_inflection_time_int"]
    return _exact_columns(death_qc_df, expected, "death_detection")


def _assert_matching_snip_ids(reference: pd.DataFrame, other: pd.DataFrame, stage_name: str) -> None:
    reference_ids = set(reference["snip_id"].astype(str))
    other_ids = set(other["snip_id"].astype(str))
    missing_left = sorted(other_ids - reference_ids)
    missing_right = sorted(reference_ids - other_ids)
    if missing_left or missing_right:
        raise ValueError(
            f"{stage_name}: snip_id mismatch (missing_in_reference={missing_left[:10]}, missing_in_{stage_name}={missing_right[:10]})"
        )


def consolidate_qc_flags(
    segmentation_qc_df: pd.DataFrame,
    viability_qc_df: pd.DataFrame,
    imaging_qc_df: pd.DataFrame,
    focus_qc_df: pd.DataFrame,
    motion_qc_df: pd.DataFrame,
    death_qc_df: pd.DataFrame,
    size_qc_df: pd.DataFrame,
    merge_on: str = "snip_id",
) -> pd.DataFrame:
    """Combine validated QC tables into the final dense QC contract."""
    if merge_on != "snip_id":
        raise ValueError("consolidate_qc_flags only supports merge_on='snip_id'")

    segmentation = _exact_columns(
        segmentation_qc_df,
        ["snip_id", "edge_flag", "discontinuous_mask_flag", "overlapping_mask_flag"],
        "segmentation_qc",
    )
    viability = _exact_columns(
        viability_qc_df,
        ["snip_id", "viability_flag"],
        "viability_qc",
    )
    imaging = _exact_columns(
        imaging_qc_df,
        ["snip_id", "yolk_flag", "bubble_flag"],
        "imaging_qc",
    )
    focus = _exact_columns(focus_qc_df, ["snip_id", "focus_flag"], "focus_qc")
    motion = _exact_columns(motion_qc_df, ["snip_id", "motion_flag"], "motion_qc")
    death_detection = _normalize_death_detection(death_qc_df)
    size = _exact_columns(size_qc_df, ["snip_id", "sa_outlier_flag"], "size_qc")

    for stage_name, frame in [
        ("segmentation_qc", segmentation),
        ("viability_qc", viability),
        ("imaging_qc", imaging),
        ("focus_qc", focus),
        ("motion_qc", motion),
        ("death_detection", death_detection),
        ("size_qc", size),
    ]:
        assert_unique_snip_id(frame, stage_name)

    _assert_matching_snip_ids(segmentation, viability, "viability_qc")
    _assert_matching_snip_ids(segmentation, imaging, "imaging_qc")
    _assert_matching_snip_ids(segmentation, focus, "focus_qc")
    _assert_matching_snip_ids(segmentation, motion, "motion_qc")
    _assert_matching_snip_ids(segmentation, death_detection, "death_detection")
    _assert_matching_snip_ids(segmentation, size, "size_qc")

    consolidated = pd.concat(
        [
            segmentation.set_index("snip_id"),
            viability.set_index("snip_id"),
            imaging.set_index("snip_id"),
            focus.set_index("snip_id"),
            motion.set_index("snip_id"),
            death_detection.set_index("snip_id"),
            size.set_index("snip_id"),
        ],
        axis=1,
        join="inner",
    ).reset_index()

    consolidated["use_snip"] = ~consolidated[SNIP_EXCLUSION_FLAGS].any(axis=1)

    ordered = consolidated[
        ["snip_id", "use_snip", *SNIP_EXCLUSION_FLAGS, *SNIP_INFORMATIONAL_FLAGS, "death_inflection_time_int"]
    ].copy()

    for column in ["use_snip", *SNIP_EXCLUSION_FLAGS, *SNIP_INFORMATIONAL_FLAGS]:
        ordered[column] = ordered[column].astype(bool)
    ordered["death_inflection_time_int"] = ordered["death_inflection_time_int"].astype("Int64")

    return ordered


def validate_qc_schema(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS_QC if column not in df.columns]
    extra = [column for column in df.columns if column not in REQUIRED_COLUMNS_QC]
    if missing or extra:
        raise ValueError(f"qc_flags.csv column mismatch (missing={missing}, extra={extra})")

    if df["snip_id"].isna().any():
        raise ValueError("qc_flags.csv: snip_id contains null values")
    if df["snip_id"].duplicated().any():
        raise ValueError("qc_flags.csv: duplicate snip_id values detected")

    for column in ["use_snip", *SNIP_EXCLUSION_FLAGS, *SNIP_INFORMATIONAL_FLAGS]:
        if df[column].isna().any():
            raise ValueError(f"qc_flags.csv: column '{column}' contains null values")
        if not pd.api.types.is_bool_dtype(df[column]):
            raise ValueError(f"qc_flags.csv: column '{column}' must be boolean dtype")

    if str(df["death_inflection_time_int"].dtype) != "Int64":
        raise ValueError("qc_flags.csv: death_inflection_time_int must be nullable integer dtype")

    if not df["use_snip"].equals(~df[SNIP_EXCLUSION_FLAGS].any(axis=1)):
        raise ValueError("qc_flags.csv: use_snip does not match SNIP_EXCLUSION_FLAGS")


def print_qc_summary(df: pd.DataFrame) -> None:
    total = len(df)
    usable = int(df["use_snip"].sum()) if len(df) else 0
    print()
    print("📊 QC Consolidation Summary:")
    print(f"   Total snips: {total}")
    print(f"   Usable (use_snip=True): {usable} ({(100 * usable / total) if total else 0:.1f}%)")
    print(f"   Flagged (use_snip=False): {total - usable} ({(100 * (total - usable) / total) if total else 0:.1f}%)")
    print()
    print("   Flag breakdown:")
    for column in SNIP_EXCLUSION_FLAGS:
        count = int(df[column].sum())
        if count > 0:
            print(f"      {column}: {count} ({100 * count / total:.1f}%)")


def save_consolidated_qc(
    consolidated_df: pd.DataFrame,
    output_path: Path,
    validate_schema: bool = True,
) -> None:
    if validate_schema:
        validate_qc_schema(consolidated_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    consolidated_df.to_csv(output_path, index=False)
    print(f"💾 Saved consolidated QC: {output_path}")
    print(f"   Rows: {len(consolidated_df)}")
    print(f"   Columns: {len(consolidated_df.columns)}")
