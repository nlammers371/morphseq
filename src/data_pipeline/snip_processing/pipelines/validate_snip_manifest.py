"""Validate merged snip_manifest contracts and emit sentinel file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_pipeline.snip_processing.io import validate_snip_manifest_df


def validate_snip_manifest(input_path: Path, output_flag: Path) -> pd.DataFrame:
    input_path = Path(input_path)
    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
    validate_snip_manifest_df(df)
    output_flag.parent.mkdir(parents=True, exist_ok=True)
    output_flag.write_text("validated\n")
    return df


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-flag", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    validate_snip_manifest(args.input, args.output_flag)


if __name__ == "__main__":
    main()

