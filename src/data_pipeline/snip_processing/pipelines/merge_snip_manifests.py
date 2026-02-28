from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd


def merge_snip_manifests(*, experiment: str, output_root: Path, write_views: bool = True) -> None:
    output_root = Path(output_root)
    exp_root = output_root / "processed_snips" / str(experiment)
    per_well_root = exp_root / "per_well"
    contracts_dir = exp_root / "contracts"
    views_root = exp_root / "views"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    if write_views:
        views_root.mkdir(parents=True, exist_ok=True)

    all_wells = sorted([p.name for p in per_well_root.iterdir() if p.is_dir()])
    qualified = [w for w in all_wells if str(w).startswith(f"{experiment}_")]
    wells = sorted(qualified or all_wells)
    if not wells:
        raise ValueError(f"No per-well snip outputs found under: {per_well_root}")

    def _slug(well_dir_name: str) -> str:
        return str(well_dir_name).split("_")[-1]

    def _is_valid_well(well_root: Path) -> bool:
        return (well_root / "contracts" / ".snip_processing.validated").exists()

    def _symlink_rel_force(src: Path, dst: Path, *, allow_missing: bool = True) -> bool:
        src = Path(src)
        dst = Path(dst)
        if not src.exists():
            if allow_missing:
                return False
            raise FileNotFoundError(f"Symlink source missing: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            elif dst.exists() and dst.is_dir():
                if any(dst.iterdir()):
                    return True
                shutil.rmtree(dst)
        except FileNotFoundError:
            pass
        rel = os.path.relpath(str(src), start=str(dst.parent))
        dst.symlink_to(rel)
        return True

    valid_wells = [w for w in wells if _is_valid_well(per_well_root / w)]
    if not valid_wells:
        raise ValueError(f"No validated per-well snip manifests found under: {per_well_root}")

    parts_pq = []
    parts_csv = []
    for w in valid_wells:
        pqp = per_well_root / w / "contracts" / "snip_manifest.parquet"
        csvp = per_well_root / w / "contracts" / "snip_manifest.csv"
        if pqp.exists():
            parts_pq.append(pqp)
        if csvp.exists():
            parts_csv.append(csvp)

    if parts_pq:
        dfs = [pd.read_parquet(p) for p in parts_pq]
        merged = pd.concat(dfs, axis=0, ignore_index=True)
        merged = merged.sort_values([c for c in ["well_id", "frame_index", "image_id", "embryo_id", "snip_id"] if c in merged.columns])
        merged.to_parquet(contracts_dir / "snip_manifest.parquet", index=False)
    if parts_csv:
        dfs = [pd.read_csv(p) for p in parts_csv]
        merged = pd.concat(dfs, axis=0, ignore_index=True)
        merged = merged.sort_values([c for c in ["well_id", "frame_index", "image_id", "embryo_id", "snip_id"] if c in merged.columns])
        merged.to_csv(contracts_dir / "snip_manifest.csv", index=False)

    if not write_views:
        return

    # Scaffold views (symlink-only).
    (views_root / "wells").mkdir(parents=True, exist_ok=True)
    (views_root / "processed").mkdir(parents=True, exist_ok=True)
    (views_root / "raw_crops").mkdir(parents=True, exist_ok=True)

    for w in valid_wells:
        slug = _slug(w)
        well_root = per_well_root / w
        _symlink_rel_force(well_root, views_root / "wells" / slug, allow_missing=True)
        _symlink_rel_force(well_root / "processed", views_root / "processed" / slug, allow_missing=True)
        _symlink_rel_force(well_root / "raw_crops", views_root / "raw_crops" / slug, allow_missing=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--write-views", default="true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    write_views = str(args.write_views).strip().lower() in {"1", "true", "yes", "y", "on"}
    merge_snip_manifests(experiment=str(args.experiment), output_root=args.output_root, write_views=write_views)


if __name__ == "__main__":
    main()

