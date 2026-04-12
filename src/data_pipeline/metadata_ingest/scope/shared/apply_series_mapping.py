from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_pipeline.io.validators import validate_dataframe_schema
from data_pipeline.schemas.scope_metadata import REQUIRED_COLUMNS_SCOPE_METADATA


def _require_columns(df: pd.DataFrame, cols: list[str], *, name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def apply_series_mapping(
    *,
    scope_metadata_csv: Path,
    series_well_mapping_csv: Path,
    output_scope_metadata_mapped_csv: Path,
) -> pd.DataFrame:
    """
    Apply a series->well mapping to scope metadata to produce canonical well_index/well_id/image_id.

    This is a physical-only operation and does not depend on plate metadata.
    """
    scope_df = pd.read_csv(scope_metadata_csv)
    validate_dataframe_schema(scope_df, REQUIRED_COLUMNS_SCOPE_METADATA, "scope_metadata.csv")

    mapping_df = pd.read_csv(series_well_mapping_csv)
    _require_columns(mapping_df, ["experiment_id", "series_number", "well_index"], name="series_well_mapping.csv")

    if scope_df.empty:
        raise ValueError("scope_metadata.csv is empty")
    exp_id = str(scope_df["experiment_id"].iloc[0])

    # We treat the current scope well_index as a 0-based series index (YX1 extractor emits "00", "01", ...).
    # series_number is 1-based, matching ND2 position index + 1.
    scope_df = scope_df.copy()
    scope_df["well_index_raw"] = scope_df["well_index"].astype(str)
    scope_df["well_id_raw"] = scope_df["well_id"].astype(str)
    scope_df["image_id_raw"] = scope_df["image_id"].astype(str)

    # Parse numeric series index from well_index_raw.
    series_zero = pd.to_numeric(scope_df["well_index_raw"], errors="raise").astype(int)
    scope_df["series_number"] = series_zero + 1

    # Restrict mapping to this experiment_id if present.
    if "experiment_id" in mapping_df.columns:
        mapping_df = mapping_df[mapping_df["experiment_id"].astype(str) == exp_id].copy()

    mapping_df = mapping_df.loc[:, ["series_number", "well_index"]].copy()
    mapping_df["series_number"] = pd.to_numeric(mapping_df["series_number"], errors="raise").astype(int)
    mapping_df["well_index"] = mapping_df["well_index"].astype(str)

    merged = scope_df.merge(mapping_df, on="series_number", how="left", suffixes=("", "_mapped"), validate="many_to_one")
    if merged["well_index_mapped"].isna().any():
        missing_series = (
            merged.loc[merged["well_index_mapped"].isna(), "series_number"].astype(int).drop_duplicates().tolist()
        )
        raise ValueError(f"Unmapped series_number values in scope metadata: {missing_series[:20]}")

    merged["well_index"] = merged["well_index_mapped"].astype(str)
    merged.drop(columns=["well_index_mapped"], inplace=True)

    merged["well_id"] = merged["experiment_id"].astype(str) + "_" + merged["well_index"].astype(str)

    # Canonical image_id uses frame_index suffix (_f), not time suffix.
    fi = pd.to_numeric(merged["frame_index"], errors="raise").astype(int)
    merged["image_id"] = (
        merged["well_id"].astype(str)
        + "_"
        + merged["channel"].astype(str)
        + "_f"
        + fi.map(lambda x: f"{x:04d}")
    )

    # Drop helper column; keep series_number only if you want it downstream (we keep for debugging).
    output_scope_metadata_mapped_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_scope_metadata_mapped_csv, index=False)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope-metadata-csv", type=Path, required=True)
    ap.add_argument("--series-well-mapping-csv", type=Path, required=True)
    ap.add_argument("--output-scope-metadata-mapped-csv", type=Path, required=True)
    args = ap.parse_args()

    apply_series_mapping(
        scope_metadata_csv=args.scope_metadata_csv,
        series_well_mapping_csv=args.series_well_mapping_csv,
        output_scope_metadata_mapped_csv=args.output_scope_metadata_mapped_csv,
    )


if __name__ == "__main__":
    main()
