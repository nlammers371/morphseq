from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_pipeline.feature_extraction.io.loaders import load_auxiliary_masks_manifest
from data_pipeline.schemas.auxiliary_masks import REQUIRED_COLUMNS_AUXILIARY_MASKS
from data_pipeline.io.validators import validate_dataframe_schema


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, nargs="+", required=True)
    ap.add_argument("--output-manifest-csv", type=Path, required=True)
    args = ap.parse_args()

    frames = [load_auxiliary_masks_manifest(path) for path in args.inputs]
    if not frames:
        raise ValueError("No auxiliary mask manifests were provided")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.loc[:, REQUIRED_COLUMNS_AUXILIARY_MASKS].copy()
    validate_dataframe_schema(merged, REQUIRED_COLUMNS_AUXILIARY_MASKS, "auxiliary_masks.csv")
    args.output_manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_manifest_csv, index=False)
    (args.output_manifest_csv.parent / ".auxiliary_masks.validated").write_text("ok\n")


if __name__ == "__main__":
    main()
