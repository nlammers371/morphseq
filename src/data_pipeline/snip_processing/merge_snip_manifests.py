from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_pipeline.snip_processing.validate_snip_manifest import validate_snip_manifest


def merge_snip_manifests(*, input_manifests: list[Path], output_parquet: Path, write_csv: bool = True) -> None:
    dfs: list[pd.DataFrame] = []
    for p in input_manifests:
        if not p.exists():
            raise FileNotFoundError(p)
        if str(p).lower().endswith(".parquet"):
            dfs.append(pd.read_parquet(p))
        else:
            dfs.append(pd.read_csv(p))

    if not dfs:
        raise ValueError("No manifests provided")

    merged = pd.concat(dfs, ignore_index=True)
    validate_snip_manifest(merged)

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_parquet, index=False)
    if write_csv:
        merged.to_csv(output_parquet.with_suffix(".csv"), index=False)
    (output_parquet.parent / ".snip_manifest_merged.validated").write_text("ok\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, nargs="+", required=True)
    ap.add_argument("--output-parquet", type=Path, required=True)
    ap.add_argument("--write-csv", type=int, default=1)
    args = ap.parse_args()

    merge_snip_manifests(
        input_manifests=[Path(p) for p in args.inputs],
        output_parquet=args.output_parquet,
        write_csv=bool(int(args.write_csv)),
    )


if __name__ == "__main__":
    main()

