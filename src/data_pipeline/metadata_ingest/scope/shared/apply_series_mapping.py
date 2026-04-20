"""Apply series-to-well mapping to scope metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from data_pipeline.metadata_ingest.time_helpers import add_elapsed_time_columns
from data_pipeline.metadata_ingest.time_helpers import add_frame_interval_unit_columns
from data_pipeline.metadata_ingest.time_helpers import ensure_frame_time_alias


def _build_series_lookup(mapping_df: pd.DataFrame) -> dict[int, str]:
    lookup: dict[int, str] = {}
    for _, row in mapping_df.iterrows():
        series_number = row.get("series_number")
        well_index = row.get("well_index")
        if pd.isna(series_number) or pd.isna(well_index):
            continue
        lookup[int(series_number)] = str(well_index)
    return lookup


def _resolve_mapped_well(
    scope_well_index: str,
    series_lookup: dict[int, str],
    known_wells: set[str],
    *,
    prefer_zero_based: bool,
) -> str:
    # If scope is already mapped to plate-like well names (e.g. A04), keep it.
    if scope_well_index in known_wells:
        return scope_well_index

    # Try YX1 convention where scope well index is 0-based series index.
    try:
        scope_idx_int = int(scope_well_index)
    except (TypeError, ValueError):
        return scope_well_index

    if prefer_zero_based:
        if (scope_idx_int + 1) in series_lookup:
            return series_lookup[scope_idx_int + 1]
        return scope_well_index

    # Prefer 1-based mapping, but allow 0-based scopes as a fallback.
    if scope_idx_int in series_lookup:
        return series_lookup[scope_idx_int]
    if (scope_idx_int + 1) in series_lookup:
        return series_lookup[scope_idx_int + 1]
    return scope_well_index


def apply_series_mapping(
    scope_metadata_csv: Path,
    mapping_csv: Path,
    output_csv: Path,
    experiment_id: str,
    selected_wells: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Map scope rows to plate wells and canonical IDs for downstream contracts."""
    scope_df = ensure_frame_time_alias(
        pd.read_csv(scope_metadata_csv),
        stage_name="scope_series_metadata_mapped_input",
    )
    mapping_df = pd.read_csv(mapping_csv)

    series_lookup = _build_series_lookup(mapping_df)
    known_wells = set(str(w) for w in mapping_df.get("well_index", pd.Series(dtype=str)).dropna().unique())

    mapped_df = scope_df.copy()
    mapped_df["well_index_raw"] = mapped_df["well_index"].astype(str)
    raw_ints = pd.to_numeric(mapped_df["well_index_raw"], errors="coerce").dropna().astype(int).tolist()
    prefer_zero_based = (0 in set(raw_ints))
    mapped_df["well_index"] = mapped_df["well_index_raw"].map(
        lambda raw: _resolve_mapped_well(str(raw), series_lookup, known_wells, prefer_zero_based=prefer_zero_based)
    )

    mapped_df["well_id"] = f"{experiment_id}_" + mapped_df["well_index"].astype(str)
    mapped_df["channel_id"] = mapped_df.get("channel", "BF").astype(str)

    if "raw_channel_name" in mapped_df.columns:
        mapped_df["channel_name_raw"] = mapped_df["raw_channel_name"].astype(str)
    else:
        mapped_df["channel_name_raw"] = mapped_df["channel_id"].astype(str)

    mapped_df["image_id"] = (
        mapped_df["well_id"].astype(str)
        + "_"
        + mapped_df["channel_id"].astype(str)
        + "_f"
        + mapped_df["frame_index"].astype(int).map(lambda val: f"{val:04d}")
    )

    mapped_df = add_elapsed_time_columns(
        mapped_df,
        group_cols=["experiment_id", "well_id", "channel_id"],
    )
    mapped_df = add_frame_interval_unit_columns(mapped_df)

    selected_wells_set = {str(well) for well in (selected_wells or []) if str(well)}
    if selected_wells_set:
        mapped_df = mapped_df[mapped_df["well_index"].astype(str).isin(selected_wells_set)].copy()

    front_cols = [
        "experiment_id",
        "well_id",
        "well_index",
        "frame_index",
        "channel_id",
        "image_id",
        "time_int",
        "experiment_time_s",
        "frame_interval_s",
        "frame_interval_min",
        "frame_interval_hr",
        "elapsed_time_s",
        "elapsed_time_min",
        "elapsed_time_hr",
    ]
    ordered = [col for col in front_cols if col in mapped_df.columns]
    remainder = [col for col in mapped_df.columns if col not in ordered]
    mapped_df = mapped_df.loc[:, ordered + remainder]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    mapped_df.to_csv(output_csv, index=False)
    return mapped_df
