from __future__ import annotations

import argparse
from pathlib import Path

from data_pipeline.feature_extraction.core.consolidate_features import load_and_consolidate_features
from data_pipeline.feature_extraction.io.loaders import load_frame_contract, load_segmentation_tracking
from data_pipeline.feature_extraction.io.writers import write_consolidated_features_contract


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segmentation-tracking", type=Path, required=True)
    ap.add_argument("--frame-contract", type=Path, required=True)
    ap.add_argument("--mask-geometry", type=Path, required=True)
    ap.add_argument("--curvature-metrics", type=Path, required=True)
    ap.add_argument("--pose-kinematics", type=Path, required=True)
    ap.add_argument("--fraction-alive", type=Path, required=True)
    ap.add_argument("--stage-predictions", type=Path, required=True)
    ap.add_argument("--plate-metadata", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    args = ap.parse_args()

    # Explicitly require the upstream contracts before consolidation.
    # The frame contract is also merged into the final table so the canonical
    # calibration and acquisition fields survive the merge boundary.
    _ = load_segmentation_tracking(args.segmentation_tracking)
    _ = load_frame_contract(args.frame_contract)

    consolidated_df = load_and_consolidate_features(
        tracking_path=args.segmentation_tracking,
        frame_contract_path=args.frame_contract,
        geometry_path=args.mask_geometry,
        curvature_path=args.curvature_metrics,
        kinematics_path=args.pose_kinematics,
        fraction_alive_path=args.fraction_alive,
        stage_path=args.stage_predictions,
        metadata_path=args.plate_metadata,
        output_path=None,
    )
    write_consolidated_features_contract(consolidated_df, args.output_csv)


if __name__ == "__main__":
    main()
