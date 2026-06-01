from __future__ import annotations

import argparse
from pathlib import Path

from data_pipeline.quality_control.config import merge_qc_defaults
from data_pipeline.quality_control.core.death_detection import compute_death_detection_flags
from data_pipeline.quality_control.io.loaders import load_features_table, load_table
from data_pipeline.quality_control.io.writers import write_qc_stage_contract
from data_pipeline.quality_control.validators import validate_death_detection_flags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fraction-alive-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--persistence-threshold", type=float, default=None)
    parser.add_argument("--lead-time-hr", type=float, default=None)
    parser.add_argument("--decline-rate-threshold", type=float, default=None)
    parser.add_argument("--dead-fraction-threshold", type=float, default=None)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    fraction_alive_df = load_table(args.fraction_alive_csv)
    snip_universe_df = load_features_table(args.features_csv)
    effective = merge_qc_defaults(
        "death_detection",
        {
            key: value
            for key, value in {
                "persistence_threshold": args.persistence_threshold,
                "lead_time_hr": args.lead_time_hr,
                "decline_rate_threshold": args.decline_rate_threshold,
                "dead_fraction_threshold": args.dead_fraction_threshold,
            }.items()
            if value is not None
        },
    )
    qc_df = compute_death_detection_flags(
        fraction_alive_df,
        snip_universe_df,
        persistence_threshold=float(effective["persistence_threshold"]),
        lead_time_hr=float(effective["lead_time_hr"]),
        decline_rate_threshold=float(effective["decline_rate_threshold"]),
        dead_fraction_threshold=float(effective["dead_fraction_threshold"]),
    )
    write_qc_stage_contract(qc_df, args.output_csv, validator=validate_death_detection_flags)


if __name__ == "__main__":
    main()
