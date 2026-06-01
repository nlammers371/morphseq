from __future__ import annotations

import argparse
from pathlib import Path

from data_pipeline.quality_control.core.consolidate_qc import consolidate_qc_flags
from data_pipeline.quality_control.io.loaders import (
    load_auxiliary_masks,
    load_death_detection_flags,
    load_features_table,
    load_focus_qc_flags,
    load_motion_qc_flags,
    load_segmentation_qc_flags,
    load_surface_area_qc_flags,
    load_viability_qc_flags,
)
from data_pipeline.quality_control.io.writers import write_qc_contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--segmentation-qc-csv", type=Path, required=True)
    parser.add_argument("--viability-qc-csv", type=Path, required=True)
    parser.add_argument("--death-detection-csv", type=Path, required=True)
    parser.add_argument("--surface-area-qc-csv", type=Path, required=True)
    parser.add_argument("--auxiliary-mask-qc-csv", type=Path, required=True)
    parser.add_argument("--focus-qc-csv", type=Path, required=True)
    parser.add_argument("--motion-qc-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    features_df = load_features_table(args.features_csv)
    segmentation_qc_df = load_segmentation_qc_flags(args.segmentation_qc_csv)
    viability_qc_df = load_viability_qc_flags(args.viability_qc_csv)
    death_detection_df = load_death_detection_flags(args.death_detection_csv)
    surface_area_qc_df = load_surface_area_qc_flags(args.surface_area_qc_csv)
    auxiliary_mask_qc_df = load_auxiliary_masks(args.auxiliary_mask_qc_csv)
    focus_qc_df = load_focus_qc_flags(args.focus_qc_csv)
    motion_qc_df = load_motion_qc_flags(args.motion_qc_csv)

    qc_df = consolidate_qc_flags(
        features_df,
        segmentation_qc_df,
        viability_qc_df,
        death_detection_df,
        surface_area_qc_df,
        auxiliary_mask_qc_df,
        focus_qc_df,
        motion_qc_df,
    )
    write_qc_contract(qc_df, args.output_csv)


if __name__ == "__main__":
    main()
