"""Validate stitched_image_index.csv contract and emit sentinel file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_pipeline.io.validators import validate_dataframe_schema
from data_pipeline.schemas.stitched_image_index import (
    REQUIRED_COLUMNS_STITCHED_IMAGE_INDEX,
    UNIQUE_KEY_STITCHED_IMAGE_INDEX,
)


def _validate_integer_column(df: pd.DataFrame, column: str, stage_name: str) -> pd.Series:
    raw = pd.to_numeric(df[column], errors="coerce")
    if raw.isna().any():
        raise ValueError(f"Column '{column}' must contain integer values in {stage_name}")
    if (raw % 1 != 0).any():
        raise ValueError(f"Column '{column}' must contain integer values in {stage_name}")
    return raw.astype(int)


def _validate_time_alias(df: pd.DataFrame, stage_name: str) -> None:
    frame_index = _validate_integer_column(df, "frame_index", stage_name)
    if "time_int" not in df.columns:
        return
    time_int = _validate_integer_column(df, "time_int", stage_name)
    mismatch = frame_index != time_int
    if mismatch.any():
        preview = df.loc[mismatch, ["experiment_id", "well_id", "channel_id", "frame_index", "time_int"]]
        raise ValueError(
            "Detected rows where frame_index != time_int in "
            f"{stage_name}: {preview.head(10).to_dict(orient='records')}"
        )


def validate_stitched_image_index(input_csv: Path, output_flag: Path) -> pd.DataFrame:
    """Validate stitched-image index schema, uniqueness, and file existence."""
    df = pd.read_csv(input_csv)
    validate_dataframe_schema(df, REQUIRED_COLUMNS_STITCHED_IMAGE_INDEX, "stitched_image_index")
    _validate_time_alias(df, "stitched_image_index")

    duplicate_mask = df.duplicated(subset=UNIQUE_KEY_STITCHED_IMAGE_INDEX, keep=False)
    if duplicate_mask.any():
        duplicates = df.loc[duplicate_mask, UNIQUE_KEY_STITCHED_IMAGE_INDEX]
        raise ValueError(
            "Duplicate stitched-image keys detected: "
            f"{duplicates.head(10).to_dict(orient='records')}"
        )

    missing_files = []
    for rel_path in df["stitched_image_path"].astype(str):
        path = Path(rel_path)
        if not path.is_absolute():
            path = input_csv.parent.parent.parent / path
        if not path.exists():
            missing_files.append(str(path))
        if len(missing_files) >= 20:
            break

    if missing_files:
        raise FileNotFoundError(
            "Missing stitched image files referenced by stitched_image_index: "
            f"{missing_files}"
        )

    output_flag.parent.mkdir(parents=True, exist_ok=True)
    output_flag.write_text("validated\n")
    return df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-flag", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    validate_stitched_image_index(args.input_csv, args.output_flag)


if __name__ == "__main__":
    main()
