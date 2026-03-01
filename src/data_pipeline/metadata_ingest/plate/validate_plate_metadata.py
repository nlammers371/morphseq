"""Validate plate_metadata.csv contract and emit sentinel file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_pipeline.io.validators import validate_dataframe_schema
from data_pipeline.schemas.plate_metadata import REQUIRED_COLUMNS_PLATE_METADATA


def validate_plate_metadata(input_csv: Path, output_flag: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    validate_dataframe_schema(df, REQUIRED_COLUMNS_PLATE_METADATA, "plate_metadata")

    # Basic uniqueness: one row per well.
    key = ["experiment_id", "well_id"]
    dup = df.duplicated(subset=key, keep=False)
    if dup.any():
        preview = df.loc[dup, key].head(10).to_dict(orient="records")
        raise ValueError(f"Duplicate plate_metadata keys detected: {preview}")

    output_flag.parent.mkdir(parents=True, exist_ok=True)
    output_flag.write_text("validated\n")
    return df


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", type=Path, required=True)
    p.add_argument("--output-flag", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    validate_plate_metadata(args.input_csv, args.output_flag)


if __name__ == "__main__":
    main()

