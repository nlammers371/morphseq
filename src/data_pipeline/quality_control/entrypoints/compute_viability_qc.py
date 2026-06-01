from __future__ import annotations

import argparse
from pathlib import Path

from data_pipeline.quality_control.config import merge_qc_defaults
from data_pipeline.quality_control.core.viability_qc import compute_viability_qc_flags
from data_pipeline.quality_control.io.loaders import load_features_table, load_table
from data_pipeline.quality_control.io.writers import write_qc_stage_contract
from data_pipeline.quality_control.validators import validate_viability_qc_flags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-geometry-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--min-mask-size-px", type=int, default=None)
    parser.add_argument("--aspect-ratio-threshold", type=float, default=None)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    mask_geometry_df = load_table(args.mask_geometry_csv)
    snip_universe_df = load_features_table(args.features_csv)
    effective = merge_qc_defaults(
        "viability_qc",
        {
            key: value
            for key, value in {
                "min_mask_size_px": args.min_mask_size_px,
                "aspect_ratio_threshold": args.aspect_ratio_threshold,
            }.items()
            if value is not None
        },
    )
    qc_df = compute_viability_qc_flags(
        mask_geometry_df,
        snip_universe_df,
        min_mask_size_px=int(effective["min_mask_size_px"]),
        aspect_ratio_threshold=float(effective["aspect_ratio_threshold"]),
    )
    write_qc_stage_contract(qc_df, args.output_csv, validator=validate_viability_qc_flags)


if __name__ == "__main__":
    main()
