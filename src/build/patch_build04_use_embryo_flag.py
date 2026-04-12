from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.build.qc import determine_use_embryo_flag
from src.run_morphseq_pipeline.paths import get_build04_output


def _coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    values = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "t": True,
        "1": True,
        "yes": True,
        "y": True,
        "false": False,
        "f": False,
        "0": False,
        "no": False,
        "n": False,
        "": False,
        "nan": False,
        "none": False,
    }
    return values.map(mapping).fillna(False).astype(bool)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch build04 CSVs by recomputing use_embryo_flag from stored QC columns.")
    parser.add_argument("--root", type=Path, default=Path("morphseq_playground"))
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    return parser.parse_args()


def patch_one(root: Path, exp: str, *, dry_run: bool, make_backup: bool) -> dict[str, object]:
    path = get_build04_output(root, exp)
    if not path.exists():
        raise FileNotFoundError(f"Missing build04 CSV: {path}")

    df = pd.read_csv(path, low_memory=False)
    if "use_embryo_flag" in df.columns:
        original = _coerce_bool(df["use_embryo_flag"])
    else:
        original = pd.Series([False] * len(df), index=df.index, dtype=bool)

    patched = determine_use_embryo_flag(df.copy())
    changed = original != patched

    summary = {
        "experiment": exp,
        "path": str(path),
        "rows": int(len(df)),
        "original_true": int(original.sum()),
        "patched_true": int(patched.sum()),
        "changed_rows": int(changed.sum()),
        "false_to_true": int((~original & patched).sum()),
        "true_to_false": int((original & ~patched).sum()),
    }

    if dry_run:
        return summary

    if make_backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = path.with_suffix(path.suffix + f".backup_pre_no_yolk_patch_{stamp}")
        path.replace(backup)
        df["use_embryo_flag"] = patched
        df.to_csv(path, index=False)
        summary["backup_path"] = str(backup)
    else:
        df["use_embryo_flag"] = patched
        df.to_csv(path, index=False)

    return summary


def main() -> None:
    args = parse_args()
    rows = []
    for exp in args.experiments:
        rows.append(
            patch_one(
                args.root,
                exp,
                dry_run=bool(args.dry_run),
                make_backup=not bool(args.no_backup),
            )
        )
    summary = pd.DataFrame(rows)
    print(summary.to_csv(index=False))


if __name__ == "__main__":
    main()
