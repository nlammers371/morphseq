"""Validate frame_contract.csv contract and emit sentinel file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_pipeline.io.validators import validate_dataframe_schema
from data_pipeline.metadata_ingest.time_helpers import ensure_time_int_column
from data_pipeline.schemas.frame_contract import (
    REQUIRED_COLUMNS_FRAME_CONTRACT,
    UNIQUE_KEY_FRAME_CONTRACT,
)


def validate_frame_contract(input_csv: Path, output_flag: Path) -> pd.DataFrame:
    """Validate frame contract schema, uniqueness, and stitched path existence."""
    df = ensure_time_int_column(
        pd.read_csv(input_csv),
        stage_name="frame_contract",
    )
    validate_dataframe_schema(df, REQUIRED_COLUMNS_FRAME_CONTRACT, "frame_contract")

    duplicate_mask = df.duplicated(subset=UNIQUE_KEY_FRAME_CONTRACT, keep=False)
    if duplicate_mask.any():
        duplicates = df.loc[duplicate_mask, UNIQUE_KEY_FRAME_CONTRACT]
        raise ValueError(
            "Duplicate frame-contract keys detected: "
            f"{duplicates.head(10).to_dict(orient='records')}"
        )

    if df["micrometers_per_pixel"].isna().any():
        raise ValueError("micrometers_per_pixel contains null values")

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
            "Missing stitched image files referenced by frame_contract: "
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
    validate_frame_contract(args.input_csv, args.output_flag)


if __name__ == "__main__":
    main()
