from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def assert_unique_snip_id(df: pd.DataFrame, stage_name: str) -> None:
    if "snip_id" not in df.columns:
        raise ValueError(f"{stage_name}: missing required column 'snip_id'")
    if df["snip_id"].isna().any():
        raise ValueError(f"{stage_name}: snip_id contains null values")
    duplicated = df["snip_id"].duplicated(keep=False)
    if duplicated.any():
        dupes = sorted(df.loc[duplicated, "snip_id"].astype(str).unique().tolist())
        raise ValueError(f"{stage_name}: snip_id must be unique; duplicates={dupes}")


def assert_exact_snip_id_set(
    df: pd.DataFrame,
    expected_snip_ids: Iterable[str],
    stage_name: str,
) -> None:
    actual = df["snip_id"].astype(str).tolist()
    expected = [str(value) for value in expected_snip_ids]
    actual_set = set(actual)
    expected_set = set(expected)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    if missing or extra:
        message_parts = []
        if missing:
            message_parts.append(f"missing={missing}")
        if extra:
            message_parts.append(f"extra={extra}")
        raise ValueError(f"{stage_name}: snip_id set mismatch ({'; '.join(message_parts)})")
    if len(actual) != len(expected):
        raise ValueError(
            f"{stage_name}: row count mismatch against snip universe "
            f"({len(actual)} != {len(expected)})"
        )


def assert_no_duplicate_columns(df: pd.DataFrame, stage_name: str) -> None:
    duplicated = df.columns[df.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(f"{stage_name}: duplicate columns present: {duplicated}")


def ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def align_to_universe(
    universe_df: pd.DataFrame,
    stage_df: pd.DataFrame,
    stage_name: str,
) -> pd.DataFrame:
    assert_unique_snip_id(universe_df, "snip universe")
    assert_unique_snip_id(stage_df, stage_name)
    assert_exact_snip_id_set(stage_df, universe_df["snip_id"].astype(str), stage_name)
    merged = universe_df[["snip_id"]].merge(stage_df, on="snip_id", how="left", validate="one_to_one")
    if merged.isna().any(axis=None):
        raise ValueError(f"{stage_name}: merge against snip universe introduced null rows")
    return merged
