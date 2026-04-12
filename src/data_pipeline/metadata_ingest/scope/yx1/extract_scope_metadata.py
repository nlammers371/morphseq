from __future__ import annotations

import argparse
from pathlib import Path

from data_pipeline.metadata_ingest.scope.yx1_scope_metadata import extract_yx1_scope_metadata


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-yx1-experiment-dir", type=Path, required=True)
    ap.add_argument("--experiment-id", type=str, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    args = ap.parse_args()

    extract_yx1_scope_metadata(
        raw_data_dir=args.raw_yx1_experiment_dir,
        experiment_id=args.experiment_id,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()

