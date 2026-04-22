"""Merge per-well stage_predictions shards to an experiment-level contract."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd


def merge_stage_predictions(*, output_root: Path, experiment: str) -> dict[str, Path]:
    output_root = Path(output_root)
    exp = str(experiment)

    exp_root = output_root / "computed_features" / exp
    per_well_root = exp_root / "per_well"
    contracts_dir = exp_root / "contracts"
    contracts_dir.mkdir(parents=True, exist_ok=True)

    def _is_valid_well(well_root: Path) -> bool:
        return (well_root / "contracts" / ".stage_predictions.computed").exists()

    wells = sorted([p.name for p in per_well_root.iterdir() if p.is_dir()]) if per_well_root.exists() else []
    valid = [w for w in wells if _is_valid_well(per_well_root / w)]
    if not valid:
        raise ValueError(f"No validated stage_predictions shards found under: {per_well_root}")

    parts = []
    for w in valid:
        p = per_well_root / w / "contracts" / "stage_predictions.parquet"
        if p.exists():
            parts.append(p)
    if not parts:
        raise ValueError(f"No per-well stage_predictions.parquet files found under: {per_well_root}")

    dfs = [pd.read_parquet(p) for p in parts]
    merged = pd.concat(dfs, axis=0, ignore_index=True)
    merged = merged.sort_values([c for c in ["well_id", "time_int", "image_id", "embryo_id", "snip_id"] if c in merged.columns])

    out_pq = contracts_dir / "stage_predictions.parquet"
    out_csv = contracts_dir / "stage_predictions.csv"
    merged.to_parquet(out_pq, index=False)
    merged.to_csv(out_csv, index=False)
    return {"parquet": out_pq, "csv": out_csv}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--experiment", required=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    merge_stage_predictions(output_root=args.output_root, experiment=str(args.experiment))


if __name__ == "__main__":
    main()
