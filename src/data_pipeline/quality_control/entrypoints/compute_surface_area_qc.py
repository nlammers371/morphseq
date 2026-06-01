from __future__ import annotations

import argparse
from pathlib import Path

from data_pipeline.quality_control.config import merge_qc_defaults
from data_pipeline.quality_control.core.surface_area_outlier_detection import compute_surface_area_qc_flags
from data_pipeline.quality_control.io.loaders import load_features_table, load_table
from data_pipeline.quality_control.io.writers import write_qc_stage_contract
from data_pipeline.quality_control.validators import validate_surface_area_qc_flags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-geometry-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--sa-reference-csv", type=Path, required=True)
    parser.add_argument("--k-upper", type=float, default=None)
    parser.add_argument("--k-lower", type=float, default=None)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    mask_geometry_df = load_table(args.mask_geometry_csv)
    snip_universe_df = load_features_table(args.features_csv)
    effective = merge_qc_defaults(
        "surface_area_qc",
        {
            key: value
            for key, value in {
                "k_upper": args.k_upper,
                "k_lower": args.k_lower,
            }.items()
            if value is not None
        },
    )
    qc_df = compute_surface_area_qc_flags(
        mask_geometry_df,
        snip_universe_df,
        sa_reference_path=args.sa_reference_csv,
        k_upper=float(effective["k_upper"]),
        k_lower=float(effective["k_lower"]),
    )
    write_qc_stage_contract(qc_df, args.output_csv, validator=validate_surface_area_qc_flags)


if __name__ == "__main__":
    main()
