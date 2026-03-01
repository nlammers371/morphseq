"""Build physical frame_manifest.csv from stitched index and scope metadata.

This contract is plate-independent: it contains only physical frame inventory,
calibration, geometry, and timing needed for segmentation/snips.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.io.validators import validate_dataframe_schema
from data_pipeline.metadata_ingest.time_helpers import add_elapsed_time_columns
from data_pipeline.metadata_ingest.time_helpers import add_frame_interval_unit_columns
from data_pipeline.metadata_ingest.time_helpers import ensure_frame_time_alias
from data_pipeline.schemas.frame_manifest import REQUIRED_COLUMNS_FRAME_MANIFEST, UNIQUE_KEY_FRAME_MANIFEST


def _with_frame_columns(df: pd.DataFrame) -> pd.DataFrame:
    return ensure_frame_time_alias(df, stage_name="frame_manifest_inputs")


def _canonicalize_scope_metadata(df: pd.DataFrame) -> pd.DataFrame:
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

    dedup_cols = ["experiment_id", "well_id", "well_index", "channel_id", "frame_index"]
    existing_dedup_cols = [col for col in dedup_cols if col in df.columns]
    if len(existing_dedup_cols) == len(dedup_cols):
        df = (
            df.sort_values(existing_dedup_cols)
            .drop_duplicates(subset=dedup_cols, keep="first")
            .copy()
        )

    return df


def _load_scope_metadata(
    *,
    scope_and_plate_csv: Path | None,
    scope_metadata_csv: Path | None,
) -> pd.DataFrame:
    # Prefer the explicit scope metadata contract. `scope_and_plate_metadata.csv` is accepted
    # for convenience/transition, but plate fields are not used for the physical frame manifest.
    if scope_and_plate_csv is not None:
        return _canonicalize_scope_metadata(pd.read_csv(scope_and_plate_csv))
    if scope_metadata_csv is None:
        raise ValueError("Either --scope-metadata-csv or --scope-and-plate-csv must be provided.")
    return _canonicalize_scope_metadata(pd.read_csv(scope_metadata_csv))


def build_frame_manifest(
    stitched_index_csv: Path,
    scope_and_plate_csv: Path | None,
    output_csv: Path,
    *,
    scope_metadata_csv: Path | None = None,
    plate_metadata_csv: Path | None = None,  # accepted for CLI compatibility; ignored
) -> pd.DataFrame:
    """Build and validate frame manifest table."""
    stitched_df = add_elapsed_time_columns(
        _with_frame_columns(pd.read_csv(stitched_index_csv)),
        group_cols=["experiment_id", "well_id", "channel_id"],
    )
    stitched_df = add_frame_interval_unit_columns(stitched_df)
    scope_df = _load_scope_metadata(
        scope_and_plate_csv=scope_and_plate_csv,
        scope_metadata_csv=scope_metadata_csv,
    )

    join_cols = ["experiment_id", "well_id", "well_index", "channel_id", "frame_index"]

    merged = stitched_df.merge(
        scope_df,
        on=join_cols,
        how="left",
        suffixes=("", "_scope"),
    )

    if "time_int_scope" in merged.columns:
        mismatch = pd.to_numeric(merged["time_int"], errors="coerce") != pd.to_numeric(merged["time_int_scope"], errors="coerce")
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

    def _col(name: str) -> pd.Series:
        if name in merged.columns:
            return merged[name]
        scoped = f"{name}_scope"
        if scoped in merged.columns:
            return merged[scoped]
        return pd.Series(np.nan, index=merged.index)

    manifest = pd.DataFrame(
        {
            "experiment_id": merged["experiment_id"],
            "well_id": merged["well_id"],
            "well_index": merged["well_index"],
            "frame_index": merged["frame_index"],
            "channel_id": merged["channel_id"],
            "image_id": merged["image_id"],
            "time_int": merged["time_int"],
            "microscope_id": merged["microscope_id"],
            "channel_name_raw": merged["channel_name_raw"],
            "stitched_image_path": merged["stitched_image_path"],
            "micrometers_per_pixel": merged["micrometers_per_pixel"],
            "frame_interval_s": merged["frame_interval_s"],
            "frame_interval_min": _col("frame_interval_min"),
            "frame_interval_hr": _col("frame_interval_hr"),
            "absolute_start_time": merged["absolute_start_time"],
            "experiment_time_s": _col("experiment_time_s"),
            "elapsed_time_s": _col("elapsed_time_s"),
            "elapsed_time_min": _col("elapsed_time_min"),
            "elapsed_time_hr": _col("elapsed_time_hr"),
            "image_width_px": merged["image_width_px"],
            "image_height_px": merged["image_height_px"],
            "objective_magnification": merged["objective_magnification"],
        }
    )
    manifest = add_elapsed_time_columns(
        manifest,
        group_cols=["experiment_id", "well_id", "channel_id"],
    )
    manifest = add_frame_interval_unit_columns(manifest)

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
    parser.add_argument("--scope-and-plate-csv", type=Path, required=False)
    parser.add_argument("--scope-metadata-csv", type=Path, required=False)
    # Accepted for backwards compatibility; ignored for the physical manifest.
    parser.add_argument("--plate-metadata-csv", type=Path, required=False)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    if args.scope_and_plate_csv is not None and args.scope_metadata_csv is not None:
        parser.error("Provide either --scope-and-plate-csv OR --scope-metadata-csv, not both.")
    if args.scope_and_plate_csv is None and args.scope_metadata_csv is None:
        parser.error("Either --scope-metadata-csv or --scope-and-plate-csv is required.")

    return args


def main() -> None:
    args = _parse_args()
    build_frame_manifest(
        stitched_index_csv=args.stitched_index_csv,
        scope_and_plate_csv=args.scope_and_plate_csv,
        scope_metadata_csv=args.scope_metadata_csv,
        plate_metadata_csv=args.plate_metadata_csv,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
