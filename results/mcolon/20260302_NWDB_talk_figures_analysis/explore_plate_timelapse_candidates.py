"""
Explore candidate experiments for rendering a 96-well plate timelapse mosaic.

This script scans embryo-level metadata CSVs under:
  morphseq_playground/metadata/embryo_metadata_files/*_embryo_metadata.csv
and keeps only experiments that also have snip JPEGs under:
  morphseq_playground/training_data/bf_embryo_snips/{experiment}/

It then ranks experiments to help pick a "badass" full-plate video for a talk:
- start_age_hpf close to 10
- many wells present
- long predicted_stage_hpf coverage
- good survival proxy via fraction_alive when available

Usage:
  PYTHON=/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python
  "$PYTHON" results/mcolon/20260302_NWDB_talk_figures_analysis/explore_plate_timelapse_candidates.py --top-n 25
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class CandidateSummary:
    experiment_date: str
    start_age_min: Optional[float]
    start_age_max: Optional[float]
    n_wells: int
    n_embryos: int
    min_hpf: Optional[float]
    max_hpf: Optional[float]
    max_frame_index: Optional[int]
    survival_proxy: Optional[float]
    n_rows: int
    snip_jpgs: int

    def start_age_center(self) -> Optional[float]:
        if self.start_age_min is None and self.start_age_max is None:
            return None
        if self.start_age_min is None:
            return self.start_age_max
        if self.start_age_max is None:
            return self.start_age_min
        return 0.5 * (float(self.start_age_min) + float(self.start_age_max))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rank experiments for 96-well plate timelapse mosaic rendering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--embryo-metadata-dir",
        default="morphseq_playground/metadata/embryo_metadata_files",
        help="Directory containing *_embryo_metadata.csv files.",
    )
    p.add_argument(
        "--snip-root",
        default="morphseq_playground/training_data/bf_embryo_snips",
        help="Root directory containing per-experiment snip JPEG folders.",
    )
    p.add_argument(
        "--experiments",
        default=None,
        help="Optional comma-separated experiment ids/dates to scan (limits work). Example: 20260206,20260202",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="How many ranked experiments to print.",
    )
    p.add_argument(
        "--out-csv",
        default=None,
        help="Optional output CSV path to save full ranking table.",
    )
    p.add_argument(
        "--require-min-wells",
        type=int,
        default=60,
        help="Require at least this many wells present to include an experiment.",
    )
    p.add_argument(
        "--require-snip-jpgs",
        type=int,
        default=500,
        help="Require at least this many snip JPEGs in the snip directory (fast filter).",
    )
    p.add_argument(
        "--chunksize",
        type=int,
        default=300_000,
        help="CSV read chunksize. Lower if memory constrained.",
    )
    return p.parse_args()


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, float) and math.isnan(x):
            return None
        return float(x)
    except Exception:
        return None


def _safe_int(x) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, float) and math.isnan(x):
            return None
        return int(x)
    except Exception:
        return None


def _count_snips(exp_dir: Path, max_scan: int = 200_000) -> int:
    """
    Count snip JPGs quickly. Stops early at max_scan to avoid slow full walks.
    """
    if not exp_dir.exists() or not exp_dir.is_dir():
        return 0
    n = 0
    for p in exp_dir.iterdir():
        if p.name.endswith(".jpg"):
            n += 1
            if n >= max_scan:
                break
    return n


def _summarize_one(
    embryo_metadata_csv: Path,
    snip_root: Path,
    chunksize: int,
) -> Optional[CandidateSummary]:
    exp = embryo_metadata_csv.name.replace("_embryo_metadata.csv", "")
    exp_dir = snip_root / exp
    snip_jpgs = _count_snips(exp_dir)
    if snip_jpgs == 0:
        return None

    header = pd.read_csv(embryo_metadata_csv, nrows=0)
    want = [
        "well_id",
        "embryo_id",
        "predicted_stage_hpf",
        "start_age_hpf",
        "fraction_alive",
        "frame_index",
    ]
    cols = [c for c in want if c in header.columns]
    if "well_id" not in cols or "embryo_id" not in cols:
        return None

    wells: set[str] = set()
    embryos: set[str] = set()
    start_min: Optional[float] = None
    start_max: Optional[float] = None
    min_hpf: Optional[float] = None
    max_hpf: Optional[float] = None
    max_fi: Optional[int] = None
    n_rows = 0

    # Survival proxy: fraction of embryos whose mean fraction_alive > 0.5.
    # Track running sum/count per embryo.
    alive_sum: dict[str, float] = {}
    alive_n: dict[str, int] = {}
    has_alive = "fraction_alive" in cols

    for chunk in pd.read_csv(embryo_metadata_csv, usecols=cols, chunksize=int(chunksize), low_memory=False):
        n_rows += int(len(chunk))
        wells.update(chunk["well_id"].astype(str).dropna().unique().tolist())
        embryos.update(chunk["embryo_id"].astype(str).dropna().unique().tolist())

        if "start_age_hpf" in chunk.columns:
            s = pd.to_numeric(chunk["start_age_hpf"], errors="coerce")
            if s.notna().any():
                mn = float(s.min())
                mx = float(s.max())
                start_min = mn if start_min is None else min(start_min, mn)
                start_max = mx if start_max is None else max(start_max, mx)

        if "predicted_stage_hpf" in chunk.columns:
            h = pd.to_numeric(chunk["predicted_stage_hpf"], errors="coerce")
            h = h[h.notna()]
            if not h.empty:
                mn = float(h.min())
                mx = float(h.max())
                min_hpf = mn if min_hpf is None else min(min_hpf, mn)
                max_hpf = mx if max_hpf is None else max(max_hpf, mx)

        if "frame_index" in chunk.columns:
            fi = pd.to_numeric(chunk["frame_index"], errors="coerce")
            fi = fi[fi.notna()]
            if not fi.empty:
                mx = int(fi.max())
                max_fi = mx if max_fi is None else max(max_fi, mx)

        if has_alive:
            fa = pd.to_numeric(chunk["fraction_alive"], errors="coerce")
            emb = chunk["embryo_id"].astype(str)
            ok = fa.notna()
            if ok.any():
                for embryo_id, v in zip(emb[ok].tolist(), fa[ok].astype(float).tolist(), strict=False):
                    alive_sum[embryo_id] = alive_sum.get(embryo_id, 0.0) + float(v)
                    alive_n[embryo_id] = alive_n.get(embryo_id, 0) + 1

    survival_proxy = None
    if has_alive and alive_sum:
        means = [(alive_sum[e] / alive_n[e]) for e in alive_sum.keys() if alive_n.get(e, 0) > 0]
        if means:
            survival_proxy = float(sum(1.0 for m in means if m > 0.5) / len(means))

    return CandidateSummary(
        experiment_date=exp,
        start_age_min=_safe_float(start_min),
        start_age_max=_safe_float(start_max),
        n_wells=int(len(wells)),
        n_embryos=int(len(embryos)),
        min_hpf=_safe_float(min_hpf),
        max_hpf=_safe_float(max_hpf),
        max_frame_index=_safe_int(max_fi),
        survival_proxy=_safe_float(survival_proxy),
        n_rows=int(n_rows),
        snip_jpgs=int(snip_jpgs),
    )


def _rank_key(c: CandidateSummary) -> tuple:
    start = c.start_age_center()
    start_dist = abs(start - 10.0) if start is not None else 1e9
    max_hpf = c.max_hpf if c.max_hpf is not None else -1e9
    surv = c.survival_proxy if c.survival_proxy is not None else -1.0
    return (start_dist, -c.n_wells, -max_hpf, -surv, c.experiment_date)


def main() -> None:
    args = _parse_args()
    embryo_dir = Path(args.embryo_metadata_dir)
    snip_root = Path(args.snip_root)
    if not embryo_dir.exists():
        raise FileNotFoundError(f"Missing embryo metadata dir: {embryo_dir}")
    if not snip_root.exists():
        raise FileNotFoundError(f"Missing snip root: {snip_root}")

    restrict: Optional[set[str]] = None
    if args.experiments:
        restrict = {x.strip() for x in str(args.experiments).split(",") if x.strip()}

    candidates: list[CandidateSummary] = []
    for p in sorted(embryo_dir.glob("*_embryo_metadata.csv")):
        exp = p.name.replace("_embryo_metadata.csv", "")
        if restrict is not None and exp not in restrict:
            continue
        s = _summarize_one(p, snip_root=snip_root, chunksize=int(args.chunksize))
        if s is None:
            continue
        if s.n_wells < int(args.require_min_wells):
            continue
        if s.snip_jpgs < int(args.require_snip_jpgs):
            continue
        candidates.append(s)

    if not candidates:
        print("No candidates found (check --require-min-wells/--require-snip-jpgs filters).")
        return

    candidates.sort(key=_rank_key)
    top_n = max(1, int(args.top_n))
    print("Top candidates (ranked):")
    for c in candidates[:top_n]:
        start = c.start_age_center()
        start_s = "NA" if start is None else f"{start:.2f}"
        hpf_s = "NA" if c.max_hpf is None else f"{c.max_hpf:.1f}"
        surv_s = "NA" if c.survival_proxy is None else f"{c.survival_proxy:.2f}"
        print(
            f"{c.experiment_date:>16}  start~{start_s:>5}  wells={c.n_wells:3d}  "
            f"hpf_max={hpf_s:>6}  max_fi={c.max_frame_index if c.max_frame_index is not None else 'NA':>4}  "
            f"survival~{surv_s:>4}  snips={c.snip_jpgs:,}"
        )

    if args.out_csv:
        out_csv = Path(args.out_csv)
        df = pd.DataFrame([c.__dict__ for c in candidates])
        # add derived start_age_center for convenience
        df["start_age_center"] = [c.start_age_center() for c in candidates]
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
