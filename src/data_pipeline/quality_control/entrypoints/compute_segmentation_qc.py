from __future__ import annotations

import argparse
from pathlib import Path

from data_pipeline.quality_control.config import merge_qc_defaults
from data_pipeline.quality_control.core.segmentation_quality_qc import compute_segmentation_qc_flags
from data_pipeline.quality_control.io.loaders import load_features_table, load_table
from data_pipeline.quality_control.io.writers import write_qc_stage_contract
from data_pipeline.quality_control.validators import validate_segmentation_qc_flags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentation-tracking-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--edge-margin-pixels", type=int, default=None)
    parser.add_argument("--max-mask-overlap-fraction", type=float, default=None)
    parser.add_argument("--min-component-fraction", type=float, default=None)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    segmentation_df = load_table(args.segmentation_tracking_csv)
    snip_universe_df = load_features_table(args.features_csv)
    effective = merge_qc_defaults(
        "segmentation_qc",
        {
            key: value
            for key, value in {
                "edge_margin_pixels": args.edge_margin_pixels,
                "max_mask_overlap_fraction": args.max_mask_overlap_fraction,
                "min_component_fraction": args.min_component_fraction,
            }.items()
            if value is not None
        },
    )
    qc_df = compute_segmentation_qc_flags(
        segmentation_df,
        snip_universe_df,
        margin_pixels=int(effective["edge_margin_pixels"]),
        iou_threshold=float(effective["max_mask_overlap_fraction"]),
        min_component_fraction=float(effective.get("min_component_fraction", 0.05)),
    )
    write_qc_stage_contract(qc_df, args.output_csv, validator=validate_segmentation_qc_flags)


if __name__ == "__main__":
    main()
