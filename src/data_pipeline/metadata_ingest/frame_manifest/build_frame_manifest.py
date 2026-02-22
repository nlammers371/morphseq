"""Build frame_manifest.csv from stitched index and scope+plate metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_pipeline.io.validators import validate_dataframe_schema
from data_pipeline.schemas.frame_manifest import (
    REQUIRED_COLUMNS_FRAME_MANIFEST,
    UNIQUE_KEY_FRAME_MANIFEST,
)


def _with_frame_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    has_frame_index = "frame_index" in df.columns
    has_time_int = "time_int" in df.columns

    if not has_frame_index:
        if "time_int" not in df.columns:
            raise ValueError("Expected either frame_index or time_int in input table")
        df["frame_index"] = pd.to_numeric(df["time_int"], errors="coerce")
    if not has_time_int:
        df["time_int"] = df["frame_index"]

    frame_index = pd.to_numeric(df["frame_index"], errors="coerce")
    time_int = pd.to_numeric(df["time_int"], errors="coerce")
    if frame_index.isna().any():
        raise ValueError("frame_index contains non-numeric values")
    if time_int.isna().any():
        raise ValueError("time_int contains non-numeric values")
    if (frame_index % 1 != 0).any():
        raise ValueError("frame_index must be integer-valued")
    if (time_int % 1 != 0).any():
        raise ValueError("time_int must be integer-valued")
    if has_frame_index and has_time_int and (frame_index != time_int).any():
        raise ValueError("Detected rows where frame_index != time_int in input table")
    df["frame_index"] = frame_index.astype(int)
    df["time_int"] = time_int.astype(int)
    return df


def _canonicalize_scope_and_plate(df: pd.DataFrame) -> pd.DataFrame:
    df = _with_frame_columns(df)

    if "channel_id" not in df.columns:
        if "channel" in df.columns:
            df["channel_id"] = df["channel"].astype(str)
        else:
            df["channel_id"] = "BF"

    if "channel_name_raw" not in df.columns:
        if "raw_channel_name" in df.columns:
            df["channel_name_raw"] = df["raw_channel_name"].astype(str)
        else:
            df["channel_name_raw"] = df["channel_id"].astype(str)

    if "temperature" not in df.columns and "temperature_c" in df.columns:
        df["temperature"] = df["temperature_c"]

    if "embryos_per_well" not in df.columns:
        df["embryos_per_well"] = 1

    dedup_cols = ["experiment_id", "well_id", "well_index", "channel_id", "frame_index"]
    existing_dedup_cols = [col for col in dedup_cols if col in df.columns]
    if len(existing_dedup_cols) == len(dedup_cols):
        df = (
            df.sort_values(existing_dedup_cols)
            .drop_duplicates(subset=dedup_cols, keep="first")
            .copy()
        )

    return df


def build_frame_manifest(
    stitched_index_csv: Path,
    scope_and_plate_csv: Path,
    output_csv: Path,
) -> pd.DataFrame:
    """Build and validate frame manifest table."""
    stitched_df = _with_frame_columns(pd.read_csv(stitched_index_csv))
    scope_df = _canonicalize_scope_and_plate(pd.read_csv(scope_and_plate_csv))

    join_cols = ["experiment_id", "well_id", "well_index", "channel_id", "frame_index"]

    merged = stitched_df.merge(
        scope_df,
        on=join_cols,
        how="left",
        suffixes=("", "_scope"),
    )

    if "time_int_scope" in merged.columns:
        mismatch = merged["time_int"] != merged["time_int_scope"]
        if mismatch.any():
            preview = merged.loc[mismatch, join_cols + ["time_int", "time_int_scope"]]
            raise ValueError(
                "time_int mismatch between stitched index and scope+plate rows: "
                f"{preview.head(10).to_dict(orient='records')}"
            )

    missing_meta = merged["micrometers_per_pixel"].isna()
    if missing_meta.any():
        preview = merged.loc[missing_meta, join_cols].head(10).to_dict(orient="records")
        raise ValueError(f"Missing scope/plate metadata for stitched rows: {preview}")

    manifest = pd.DataFrame(
        {
            "experiment_id": merged["experiment_id"],
            "microscope_id": merged["microscope_id"],
            "well_id": merged["well_id"],
            "well_index": merged["well_index"],
            "channel_id": merged["channel_id"],
            "channel_name_raw": merged["channel_name_raw"],
            "time_int": merged["time_int"],
            "frame_index": merged["frame_index"],
            "image_id": merged["image_id"],
            "stitched_image_path": merged["stitched_image_path"],
            "micrometers_per_pixel": merged["micrometers_per_pixel"],
            "frame_interval_s": merged["frame_interval_s"],
            "absolute_start_time": merged["absolute_start_time"],
            "experiment_time_s": merged["experiment_time_s"],
            "image_width_px": merged["image_width_px"],
            "image_height_px": merged["image_height_px"],
            "objective_magnification": merged["objective_magnification"],
            "genotype": merged["genotype"],
            "treatment": merged["treatment"],
            "medium": merged["medium"],
            "temperature": merged["temperature"],
            "start_age_hpf": merged["start_age_hpf"],
            "embryos_per_well": merged["embryos_per_well"],
        }
    )

    validate_dataframe_schema(manifest, REQUIRED_COLUMNS_FRAME_MANIFEST, "frame_manifest")

    duplicate_mask = manifest.duplicated(subset=UNIQUE_KEY_FRAME_MANIFEST, keep=False)
    if duplicate_mask.any():
        duplicates = manifest.loc[duplicate_mask, UNIQUE_KEY_FRAME_MANIFEST]
        raise ValueError(
            "Duplicate frame_manifest keys detected: "
            f"{duplicates.head(10).to_dict(orient='records')}"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_csv, index=False)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stitched-index-csv", type=Path, required=True)
    parser.add_argument("--scope-and-plate-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_frame_manifest(args.stitched_index_csv, args.scope_and_plate_csv, args.output_csv)


if __name__ == "__main__":
    main()
