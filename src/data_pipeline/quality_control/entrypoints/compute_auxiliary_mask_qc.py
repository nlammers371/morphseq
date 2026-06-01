from __future__ import annotations

import argparse
from pathlib import Path

from data_pipeline.quality_control.core.auxiliary_mask_qc import compute_auxiliary_mask_qc_flags
from data_pipeline.quality_control.io.loaders import load_features_table, load_table
from data_pipeline.quality_control.io.writers import write_qc_stage_contract
from data_pipeline.quality_control.validators import validate_auxiliary_mask_qc_flags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auxiliary-masks-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    auxiliary_masks_df = load_table(args.auxiliary_masks_csv)
    snip_universe_df = load_features_table(args.features_csv)
    qc_df = compute_auxiliary_mask_qc_flags(auxiliary_masks_df, snip_universe_df)
    write_qc_stage_contract(qc_df, args.output_csv, validator=validate_auxiliary_mask_qc_flags)


if __name__ == "__main__":
    main()
