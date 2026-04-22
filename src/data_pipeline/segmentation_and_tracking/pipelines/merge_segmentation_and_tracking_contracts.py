from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd


def merge_contracts(*, experiment: str, output_root: Path) -> None:
    output_root = Path(output_root)
    exp_root = output_root / "segmentation_and_tracking" / str(experiment)
    per_well_root = exp_root / "per_well"
    contracts_dir = exp_root / "contracts"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    views_root = exp_root / "views"
    views_root.mkdir(parents=True, exist_ok=True)

    all_wells = sorted([p.name for p in per_well_root.iterdir() if p.is_dir()])
    # Prefer experiment-qualified per_well directories (e.g. "20240418_A01") when present.
    qualified = [w for w in all_wells if str(w).startswith(f"{experiment}_")]
    wells = sorted(qualified or all_wells)
    if not wells:
        raise ValueError(f"No per-well outputs found under: {per_well_root}")

    def _slug(well_dir_name: str) -> str:
        # per_well directories are often experiment-qualified (e.g. 20240418_A01).
        return str(well_dir_name).split("_")[-1]

    def _symlink_rel_force(src: Path, dst: Path, *, allow_missing: bool = True) -> bool:
        """
        Create a portable relative symlink (dst -> src).

        Returns True if created, False if skipped because src was missing.
        """
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
                # If it's a directory, only replace if it's empty; otherwise keep.
                if any(dst.iterdir()):
                    return True
                shutil.rmtree(dst)
        except FileNotFoundError:
            pass

        rel = os.path.relpath(str(src), start=str(dst.parent))
        dst.symlink_to(rel)
        return True

    def _is_valid_well(well_root: Path) -> bool:
        return (well_root / "contracts" / ".segment_and_track.validated").exists()

    valid_wells = [w for w in wells if _is_valid_well(per_well_root / w)]
    if not valid_wells:
        raise ValueError(f"No validated per-well outputs found under: {per_well_root}")

    # Parquet shards
    for parquet_name in [
        "frame_detections.parquet",
        "seed_selection.parquet",
        "embryo_track_instances.parquet",
        "embryo_mask_rle.parquet",
    ]:
        parts = []
        for well in valid_wells:
            p = per_well_root / well / "contracts" / parquet_name
            if p.exists():
                parts.append(p)
        if not parts:
            continue
        dfs = [pd.read_parquet(p) for p in parts]
        merged = pd.concat(dfs, axis=0, ignore_index=True)
        merged = merged.sort_values([c for c in ["well_id", "time_int", "image_id", "embryo_id"] if c in merged.columns])
        merged.to_parquet(contracts_dir / parquet_name, index=False)

    # Final CSV contract
    csv_parts = []
    for well in valid_wells:
        p = per_well_root / well / "contracts" / "segmentation_tracking.csv"
        if p.exists():
            csv_parts.append(p)
    if csv_parts:
        dfs = [pd.read_csv(p) for p in csv_parts]
        merged = pd.concat(dfs, axis=0, ignore_index=True)
        merged = merged.sort_values([c for c in ["well_id", "time_int", "image_id", "embryo_id"] if c in merged.columns])
        merged.to_csv(contracts_dir / "segmentation_tracking.csv", index=False)

    def _mask_heads_for_well(well_root: Path) -> list[str]:
        """
        Discover mask heads present in a per-well shard via its contracts CSV.

        Maps `mask_type` (e.g. "embryo") -> mask_head folder name (e.g. "embryo_mask").
        """
        p = well_root / "contracts" / "segmentation_tracking.csv"
        if not p.exists():
            return []
        try:
            df = pd.read_csv(p, usecols=["mask_type"])
        except Exception:
            df = pd.read_csv(p)
        if "mask_type" not in df.columns:
            return []
        vals = sorted({str(v) for v in df["mask_type"].dropna().astype(str).tolist() if str(v)})
        return [f"{v}_mask" for v in vals if v]

    # Pre-create a predictable views scaffold (even if some sources are missing).
    (views_root / "wells").mkdir(parents=True, exist_ok=True)
    (views_root / "masks").mkdir(parents=True, exist_ok=True)
    (views_root / "videos" / "raw").mkdir(parents=True, exist_ok=True)
    (views_root / "videos" / "overlays").mkdir(parents=True, exist_ok=True)
    (views_root / "frames" / "raw").mkdir(parents=True, exist_ok=True)
    (views_root / "frames" / "overlays").mkdir(parents=True, exist_ok=True)

    # Discover all heads across the experiment for stable scaffold dirs.
    all_heads: set[str] = set()
    per_well_heads: dict[str, list[str]] = {}
    for well in valid_wells:
        well_root = per_well_root / well
        heads = _mask_heads_for_well(well_root) or ["embryo_mask"]
        per_well_heads[well] = heads
        all_heads.update(heads)

    for head in sorted(all_heads):
        (views_root / "masks" / head).mkdir(parents=True, exist_ok=True)
        (views_root / "videos" / "overlays" / head).mkdir(parents=True, exist_ok=True)
        (views_root / "frames" / "overlays" / head).mkdir(parents=True, exist_ok=True)

    # Build symlink-only browse views into the per-well shards.
    for well in valid_wells:
        slug = _slug(well)
        well_root = per_well_root / well

        _symlink_rel_force(well_root, views_root / "wells" / slug, allow_missing=True)

        # Raw frames / raw video (head-agnostic)
        _symlink_rel_force(
            well_root / "artifacts" / "raw_frames",
            views_root / "frames" / "raw" / slug,
            allow_missing=True,
        )
        _symlink_rel_force(
            well_root / "artifacts" / "raw_video" / f"{slug}_raw.mp4",
            views_root / "videos" / "raw" / f"{slug}_raw.mp4",
            allow_missing=True,
        )

        for head in per_well_heads.get(well, ["embryo_mask"]):
            _symlink_rel_force(
                well_root / "masks" / head,
                views_root / "masks" / head / slug,
                allow_missing=True,
            )
            _symlink_rel_force(
                well_root / "artifacts" / "overlays" / head / f"{slug}_{head}_overlay.mp4",
                views_root / "videos" / "overlays" / head / f"{slug}_{head}_overlay.mp4",
                allow_missing=True,
            )
            _symlink_rel_force(
                well_root / "artifacts" / "overlays" / head / "frames",
                views_root / "frames" / "overlays" / head / slug,
                allow_missing=True,
            )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True)
    p.add_argument("--output-root", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    merge_contracts(experiment=args.experiment, output_root=args.output_root)


if __name__ == "__main__":
    main()
