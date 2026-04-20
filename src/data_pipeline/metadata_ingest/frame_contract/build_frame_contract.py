"""Build physical frame_contract.csv from stitched index and scope metadata.

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
from data_pipeline.schemas.frame_contract import REQUIRED_COLUMNS_FRAME_CONTRACT, UNIQUE_KEY_FRAME_CONTRACT


def _with_frame_columns(df: pd.DataFrame) -> pd.DataFrame:
    return ensure_frame_time_alias(df, stage_name="frame_contract_inputs")


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


def build_frame_contract(
    stitched_index_csv: Path,
    scope_metadata_csv: Path,
    output_csv: Path,
) -> pd.DataFrame:
    """Build and validate the frame contract table."""
    stitched_df = add_elapsed_time_columns(
        _with_frame_columns(pd.read_csv(stitched_index_csv)),
        group_cols=["experiment_id", "well_id", "channel_id"],
    )
    stitched_df = add_frame_interval_unit_columns(stitched_df)
    scope_df = _canonicalize_scope_metadata(pd.read_csv(scope_metadata_csv))

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
                "time_int mismatch between stitched index and scope rows: "
                f"{preview.head(10).to_dict(orient='records')}"
            )

    missing_meta = merged["micrometers_per_pixel"].isna()
    if missing_meta.any():
        preview = merged.loc[missing_meta, join_cols].head(10).to_dict(orient="records")
        raise ValueError(f"Missing scope metadata for stitched rows: {preview}")

    def _col(name: str) -> pd.Series:
        if name in merged.columns:
            return merged[name]
        scoped = f"{name}_scope"
        if scoped in merged.columns:
            return merged[scoped]
        return pd.Series(np.nan, index=merged.index)

    contract = pd.DataFrame(
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
    contract = add_elapsed_time_columns(
        contract,
        group_cols=["experiment_id", "well_id", "channel_id"],
    )
    contract = add_frame_interval_unit_columns(contract)

    validate_dataframe_schema(contract, REQUIRED_COLUMNS_FRAME_CONTRACT, "frame_contract")

    duplicate_mask = contract.duplicated(subset=UNIQUE_KEY_FRAME_CONTRACT, keep=False)
    if duplicate_mask.any():
        duplicates = contract.loc[duplicate_mask, UNIQUE_KEY_FRAME_CONTRACT]
        raise ValueError(
            "Duplicate frame_contract keys detected: "
            f"{duplicates.head(10).to_dict(orient='records')}"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    contract.to_csv(output_csv, index=False)
    return contract


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stitched-index-csv", type=Path, required=True)
    parser.add_argument("--scope-metadata-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_frame_contract(
        stitched_index_csv=args.stitched_index_csv,
        scope_metadata_csv=args.scope_metadata_csv,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
