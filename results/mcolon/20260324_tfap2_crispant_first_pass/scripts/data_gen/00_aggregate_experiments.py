"""Aggregate TFAP2 experiments into one normalized parquet artifact."""

import sys
from pathlib import Path


def main() -> None:
    run_dir = Path(__file__).resolve().parent.parent.parent
    project_root = Path(__file__).resolve().parents[5]
    sys.path.insert(0, str(run_dir))

    from scripts.common import EXPERIMENT_IDS, EXPERIMENT_LABEL, write_aggregate_artifacts

    df, parquet_path, metadata_path = write_aggregate_artifacts(project_root, run_dir)
    embryo_count = df["embryo_id"].nunique()

    print(f"Aggregated experiments: {EXPERIMENT_IDS}")
    print(f"Experiment label: {EXPERIMENT_LABEL}")
    print(f"Rows: {len(df)}")
    print(f"Embryos: {embryo_count}")
    print(f"Parquet: {parquet_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
