"""Attach plate (well-level) annotations to the physical frame_manifest.

This is an optional convenience artifact for debugging/notebooks. Downstream compute
steps should generally depend on the physical frame manifest + plate sentinel instead
of depending on this joined file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_pipeline.io.validators import validate_dataframe_schema
from data_pipeline.schemas.frame_manifest import REQUIRED_COLUMNS_FRAME_MANIFEST
from data_pipeline.schemas.plate_metadata import REQUIRED_COLUMNS_PLATE_METADATA


def attach_plate_annotations(
    *,
    frame_manifest_csv: Path,
    plate_metadata_csv: Path,
    output_csv: Path,
    output_flag: Path,
) -> pd.DataFrame:
    fm = pd.read_csv(frame_manifest_csv)
    plate = pd.read_csv(plate_metadata_csv)

    # Validate inputs against their canonical contracts.
    validate_dataframe_schema(fm, REQUIRED_COLUMNS_FRAME_MANIFEST, "frame_manifest")
    validate_dataframe_schema(plate, REQUIRED_COLUMNS_PLATE_METADATA, "plate_metadata")

    join_cols = ["experiment_id", "well_id"]
    merged = fm.merge(
        plate,
        on=join_cols,
        how="left",
        validate="many_to_one",
        suffixes=("", "_plate"),
    )

    # Fail fast if plate join is incomplete; this file is specifically for having plate fields.
    missing = merged["genotype"].isna()
    if missing.any():
        preview = merged.loc[missing, join_cols].drop_duplicates().head(10).to_dict(orient="records")
        raise ValueError(f"Missing plate annotations for frame rows (preview): {preview}")

    merged["age_hpf"] = merged["start_age_hpf"].astype(float) + (merged["elapsed_time_s"].astype(float) / 3600.0)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    output_flag.parent.mkdir(parents=True, exist_ok=True)
    output_flag.write_text("validated\n")
    return merged


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frame-manifest-csv", type=Path, required=True)
    p.add_argument("--plate-metadata-csv", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--output-flag", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    attach_plate_annotations(
        frame_manifest_csv=args.frame_manifest_csv,
        plate_metadata_csv=args.plate_metadata_csv,
        output_csv=args.output_csv,
        output_flag=args.output_flag,
    )


if __name__ == "__main__":
    main()

