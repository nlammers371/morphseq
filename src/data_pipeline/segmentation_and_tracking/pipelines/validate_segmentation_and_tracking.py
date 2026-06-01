"""Validate merged segmentation_and_tracking contracts and emit sentinel file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_pipeline.io.validators import validate_dataframe_schema
from data_pipeline.schemas.segmentation import REQUIRED_COLUMNS_SEGMENTATION_TRACKING


UNIQUE_KEY_SEGMENTATION_TRACKING = ["experiment_id", "well_id", "snip_id"]


def validate_segmentation_tracking(input_csv: Path, output_flag: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    if "time_int" not in df.columns and "frame_index" in df.columns:
        df["time_int"] = pd.to_numeric(df["frame_index"], errors="raise").astype(int)
    validate_dataframe_schema(df, REQUIRED_COLUMNS_SEGMENTATION_TRACKING, "segmentation_tracking")

    dup = df.duplicated(subset=UNIQUE_KEY_SEGMENTATION_TRACKING, keep=False)
    if dup.any():
        preview = df.loc[dup, UNIQUE_KEY_SEGMENTATION_TRACKING].head(10).to_dict(orient="records")
        raise ValueError(f"Duplicate segmentation_tracking keys detected: {preview}")

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
    validate_segmentation_tracking(args.input_csv, args.output_flag)


if __name__ == "__main__":
    main()

