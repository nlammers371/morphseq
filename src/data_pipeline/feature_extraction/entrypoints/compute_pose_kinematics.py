from __future__ import annotations

import argparse
from pathlib import Path

from data_pipeline.feature_extraction.core.pose_kinematics import extract_pose_kinematics_batch
from data_pipeline.feature_extraction.io.loaders import (
    load_frame_contract,
    load_segmentation_tracking,
    merge_tracking_with_frame_contract,
)
from data_pipeline.feature_extraction.io.writers import write_feature_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segmentation-tracking", type=Path, required=True)
    ap.add_argument("--frame-contract", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    args = ap.parse_args()

    tracking_df = load_segmentation_tracking(args.segmentation_tracking)
    frame_df = load_frame_contract(args.frame_contract)
    merged = merge_tracking_with_frame_contract(tracking_df, frame_df)
    feature_df = extract_pose_kinematics_batch(merged, pixel_size_col="micrometers_per_pixel")
    write_feature_table(feature_df, args.output_csv)


if __name__ == "__main__":
    main()
