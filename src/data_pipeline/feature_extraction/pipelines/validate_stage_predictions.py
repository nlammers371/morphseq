"""Validate stage_predictions contract and emit sentinel file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.io.validators import validate_dataframe_schema
from data_pipeline.schemas.stage_predictions import REQUIRED_COLUMNS_STAGE_PREDICTIONS, UNIQUE_KEY_STAGE_PREDICTIONS


def validate_stage_predictions(df: pd.DataFrame) -> None:
    validate_dataframe_schema(df, REQUIRED_COLUMNS_STAGE_PREDICTIONS, "stage_predictions")

    dup = df.duplicated(subset=UNIQUE_KEY_STAGE_PREDICTIONS, keep=False)
    if dup.any():
        preview = df.loc[dup, UNIQUE_KEY_STAGE_PREDICTIONS].head(10).to_dict(orient="records")
        raise ValueError(f"Duplicate stage_predictions keys detected: {preview}")

    # Strict requirements (QC expects this to be complete).
    if df["predicted_stage_hpf"].isna().any():
        n = int(df["predicted_stage_hpf"].isna().sum())
        raise ValueError(f"predicted_stage_hpf contains {n} null values.")

    # Numeric sanity checks.
    for col in ["elapsed_time_s", "start_age_hpf", "temperature", "developmental_rate_hpf_per_h", "predicted_stage_hpf"]:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().any():
            n = int(vals.isna().sum())
            raise ValueError(f"{col} contains {n} non-numeric/null values.")
        if not np.isfinite(vals.to_numpy()).all():
            raise ValueError(f"{col} contains non-finite values.")

    if (pd.to_numeric(df["elapsed_time_s"], errors="coerce") < 0).any():
        raise ValueError("elapsed_time_s contains negative values.")

    # Broad bounds to catch obvious corruption.
    hpf = pd.to_numeric(df["predicted_stage_hpf"], errors="coerce")
    if ((hpf < 0) | (hpf > 200)).any():
        preview = df.loc[(hpf < 0) | (hpf > 200), ["snip_id", "predicted_stage_hpf"]].head(10).to_dict(orient="records")
        raise ValueError(f"predicted_stage_hpf outside [0,200] (preview): {preview}")


def validate_stage_predictions_file(*, input_path: Path, output_flag: Path) -> pd.DataFrame:
    input_path = Path(input_path)
    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
    validate_stage_predictions(df)
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
    validate_stage_predictions_file(input_path=args.input, output_flag=args.output_flag)


if __name__ == "__main__":
    main()
