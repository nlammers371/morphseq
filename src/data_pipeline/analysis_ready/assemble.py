#!/usr/bin/env python3
"""Analysis-ready table assembly.

The analysis-ready contract is a validated 1:1 join of consolidated features
and consolidated QC flags, with a final ``embedding_calculated`` marker.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_pipeline.schemas.analysis_ready import REQUIRED_COLUMNS_ANALYSIS_READY
from data_pipeline.schemas.quality_control import SNIP_EXCLUSION_FLAGS, SNIP_INFORMATIONAL_FLAGS


def _assert_unique_snip_id(df: pd.DataFrame, stage_name: str, *, key: str = "snip_id") -> None:
    if key not in df.columns:
        raise ValueError(f"{stage_name}: missing required key column '{key}'")
    if df[key].isna().any():
        raise ValueError(f"{stage_name}: column '{key}' contains null values")
    if df[key].duplicated().any():
        raise ValueError(f"{stage_name}: duplicate '{key}' values detected")


def _assert_no_column_collisions(left_df: pd.DataFrame, right_df: pd.DataFrame, *, key: str = "snip_id") -> None:
    collisions = sorted((set(left_df.columns) & set(right_df.columns)) - {key})
    if collisions:
        raise ValueError(f"Column name collisions detected before join: {collisions}")


def _assert_matching_snip_ids(left_df: pd.DataFrame, right_df: pd.DataFrame, *, key: str = "snip_id") -> None:
    left_ids = set(left_df[key].astype(str))
    right_ids = set(right_df[key].astype(str))
    missing_left = sorted(right_ids - left_ids)
    missing_right = sorted(left_ids - right_ids)
    if missing_left or missing_right:
        raise ValueError(
            f"Join key mismatch detected: missing_in_left={missing_left[:10]}, missing_in_right={missing_right[:10]}"
        )


def assemble_analysis_ready(
    features_df: pd.DataFrame,
    qc_df: pd.DataFrame,
    *,
    key: str = "snip_id",
) -> pd.DataFrame:
    """Join features and QC tables into the final analysis-ready contract."""
    _assert_unique_snip_id(features_df, "features")
    _assert_unique_snip_id(qc_df, "qc")
    _assert_no_column_collisions(features_df, qc_df, key=key)
    _assert_matching_snip_ids(features_df, qc_df, key=key)

    assembled = pd.concat(
        [
            features_df.set_index(key),
            qc_df.set_index(key),
        ],
        axis=1,
        join="inner",
    ).reset_index()

    assembled["embedding_calculated"] = False
    return assembled


def validate_analysis_ready_schema(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS_ANALYSIS_READY if column not in df.columns]
    extra = [column for column in df.columns if column not in REQUIRED_COLUMNS_ANALYSIS_READY]
    if missing or extra:
        raise ValueError(f"analysis_ready.csv column mismatch (missing={missing}, extra={extra})")

    if df["snip_id"].isna().any():
        raise ValueError("analysis_ready.csv: snip_id contains null values")
    if df["snip_id"].duplicated().any():
        raise ValueError("analysis_ready.csv: duplicate snip_id values detected")

    for column in ["embedding_calculated", "use_snip", *SNIP_EXCLUSION_FLAGS, *SNIP_INFORMATIONAL_FLAGS]:
        if df[column].isna().any():
            raise ValueError(f"analysis_ready.csv: column '{column}' contains null values")
        if not pd.api.types.is_bool_dtype(df[column]):
            raise ValueError(f"analysis_ready.csv: column '{column}' must be boolean dtype")


def print_analysis_ready_summary(df: pd.DataFrame) -> None:
    total = len(df)
    usable = int(df["use_snip"].sum()) if len(df) else 0
    embedded = int(df["embedding_calculated"].sum()) if len(df) else 0
    print()
    print("📊 Analysis-Ready Table Summary:")
    print(f"   Total snips: {total}")
    print(f"   Usable (use_snip=True): {usable} ({(100 * usable / total) if total else 0:.1f}%)")
    print(f"   Embeddings calculated: {embedded} ({(100 * embedded / total) if total else 0:.1f}%)")
    print(f"   Columns: {len(df.columns)}")
    if "experiment_id" in df.columns:
        print(f"   Experiments: {df['experiment_id'].nunique()}")


def save_analysis_ready(
    analysis_df: pd.DataFrame,
    output_path: Path,
    validate_schema: bool = True,
) -> None:
    if validate_schema:
        validate_analysis_ready_schema(analysis_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_df.to_csv(output_path, index=False)
    print(f"💾 Saved analysis-ready table: {output_path}")
    print(f"   Rows: {len(analysis_df)}")
    print(f"   Columns: {len(analysis_df.columns)}")


def filter_for_analysis(
    analysis_df: pd.DataFrame,
    use_snip_only: bool = True,
) -> pd.DataFrame:
    """Filter the analysis-ready table for downstream use."""
    if use_snip_only:
        return analysis_df[analysis_df["use_snip"]].copy()
    return analysis_df.copy()
