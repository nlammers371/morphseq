from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_pipeline.io.validators import validate_dataframe_schema
from data_pipeline.schemas.segmentation import REQUIRED_COLUMNS_SEGMENTATION_TRACKING


def merge_segmentation_tracking(*, inputs: list[Path], output_csv: Path) -> None:
    dfs: list[pd.DataFrame] = []
    for p in inputs:
        if not p.exists():
            raise FileNotFoundError(p)
        dfs.append(pd.read_csv(p))
    if not dfs:
        raise ValueError("No inputs provided")

    merged = pd.concat(dfs, ignore_index=True)
    validate_dataframe_schema(merged, REQUIRED_COLUMNS_SEGMENTATION_TRACKING, "segmentation_tracking_merged")
    if merged["snip_id"].duplicated().any():
        dups = merged.loc[merged["snip_id"].duplicated(), "snip_id"].astype(str).head(10).tolist()
        raise ValueError(f"snip_id is not unique in merged segmentation_tracking; examples: {dups}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    (output_csv.parent / ".segmentation_tracking_merged.validated").write_text("ok\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, nargs="+", required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    args = ap.parse_args()

    merge_segmentation_tracking(inputs=[Path(p) for p in args.inputs], output_csv=args.output_csv)


if __name__ == "__main__":
    main()

