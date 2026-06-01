from __future__ import annotations

import argparse
from pathlib import Path

from data_pipeline.analysis_ready.core.assemble import assemble_analysis_ready
from data_pipeline.analysis_ready.io.loaders import (
    load_analysis_ready_features,
    load_analysis_ready_qc_flags,
)
from data_pipeline.analysis_ready.io.writers import write_analysis_ready_contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--qc-flags-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-schema-json", type=Path, required=False)
    args = parser.parse_args()

    features_df = load_analysis_ready_features(args.features_csv)
    qc_df = load_analysis_ready_qc_flags(args.qc_flags_csv)
    assembled = assemble_analysis_ready(features_df, qc_df)
    write_analysis_ready_contract(
        assembled,
        args.output_csv,
        schema_json_path=args.output_schema_json,
    )


if __name__ == "__main__":
    main()
