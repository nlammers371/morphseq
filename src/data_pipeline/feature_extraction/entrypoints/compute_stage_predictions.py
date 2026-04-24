from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_pipeline.feature_extraction.core.stage_inference import compute_stage_predictions_batch
from data_pipeline.feature_extraction.config import DEFAULT_FEATURE_EXTRACTION_CONFIG
from data_pipeline.feature_extraction.io.loaders import (
    load_frame_contract,
    load_plate_metadata,
    load_segmentation_tracking,
    merge_tracking_with_frame_contract,
)
from data_pipeline.feature_extraction.io.writers import write_feature_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segmentation-tracking", type=Path, required=True)
    ap.add_argument("--frame-contract", type=Path, required=True)
    ap.add_argument("--plate-metadata", type=Path, required=True)
    ap.add_argument("--stage-cfg", type=str, default="{}")
    ap.add_argument("--output-csv", type=Path, required=True)
    args = ap.parse_args()

    stage_defaults = DEFAULT_FEATURE_EXTRACTION_CONFIG["stage_predictions"]
    stage_overrides = json.loads(args.stage_cfg) if args.stage_cfg else {}
    effective_cfg = {**stage_defaults, **stage_overrides}

    tracking_df = load_segmentation_tracking(args.segmentation_tracking)
    frame_df = load_frame_contract(args.frame_contract)
    plate_df = load_plate_metadata(args.plate_metadata)
    merged = merge_tracking_with_frame_contract(tracking_df, frame_df)
    if "well_id" in plate_df.columns:
        merged = merged.merge(plate_df, on="well_id", how="left", suffixes=("", "_plate"))
    elif "experiment_id" in plate_df.columns:
        merged = merged.merge(plate_df, on="experiment_id", how="left", suffixes=("", "_plate"))
    feature_df = compute_stage_predictions_batch(
        merged,
        start_age_col=effective_cfg["start_age_col"],
        time_col=effective_cfg["time_col"],
        temp_col=effective_cfg["temp_col"],
    )
    write_feature_table(feature_df, args.output_csv)


if __name__ == "__main__":
    main()
