#!/usr/bin/env python3
"""
Audit yolk-mask consistency and staleness against embryo masks.

Scope:
- Lives in canonical aligner debug workspace (no pipeline coupling).
- By default, infers which experiments to audit from a Phase0 run directory's
  `feature_dataset/metadata.parquet`, then samples rows from the master CSV.

Checks per sampled row:
1. `yolk_inside_embryo_ratio` = |yolk ∩ embryo| / |yolk|
2. `yolk_embryo_iou` = |yolk ∩ embryo| / |yolk ∪ embryo|
3. Staleness: yolk prediction file mtime older than source sample artifacts.

Usage:
  PYTHON=/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python
  "$PYTHON" results/mcolon/20260216_canonical_aligner_debug/audit_yolk_mask_consistency.py
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import skimage.io as io
from skimage.transform import resize

import sys


MORPHSEQ_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MORPHSEQ_ROOT))
sys.path.insert(0, str(MORPHSEQ_ROOT / "src"))

from analyze.optimal_transport_morphometrics.uot_masks import frame_mask_io as fmio


DEFAULT_RUN_DIR = (
    MORPHSEQ_ROOT
    / "results/mcolon/20260215_roi_discovery_via_ot_feature_maps/scripts/output/phase0_qc_fix_rerun_alignedmasks"
)
DEFAULT_DATA_CSV = (
    MORPHSEQ_ROOT
    / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
)
DEFAULT_DATA_ROOT = MORPHSEQ_ROOT / "morphseq_playground"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "debug_results" / "yolk_consistency_audit"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit yolk-mask overlap/staleness across experiments")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--data-csv", type=Path, default=DEFAULT_DATA_CSV)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--experiments",
        type=str,
        default="",
        help="Comma-separated experiments (experiment_date). If omitted, inferred from --run-dir metadata.",
    )
    parser.add_argument("--samples-per-experiment", type=int, default=50)
    parser.add_argument(
        "--min-yolk-inside-ratio",
        type=float,
        default=0.6,
        help="Rows below this are flagged as low-overlap.",
    )
    parser.add_argument(
        "--stale-tolerance-hours",
        type=float,
        default=1.0,
        help="Yolk file older than source artifacts by more than this is flagged stale.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _exp_str(v) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _extract_time_stub(row: pd.Series) -> Optional[str]:
    for key in ("time_int", "frame_index"):
        if key in row and pd.notnull(row[key]):
            try:
                return f"{int(row[key]):04d}"
            except (TypeError, ValueError):
                continue
    return None


def _load_binary_mask(path: Path, target_shape: Optional[tuple[int, int]] = None) -> np.ndarray:
    arr_raw = io.imread(path)
    if arr_raw.max() >= 255:
        arr = (arr_raw > 127).astype(np.uint8)
    else:
        arr = (arr_raw > 0).astype(np.uint8)
    if target_shape is not None and tuple(arr.shape) != tuple(target_shape):
        arr = resize(
            arr.astype(float),
            target_shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.uint8)
    return arr


def _overlap_metrics(yolk: np.ndarray, embryo: np.ndarray) -> tuple[float, float]:
    y = yolk.astype(bool)
    e = embryo.astype(bool)
    yolk_area = float(y.sum())
    if yolk_area <= 0:
        return np.nan, np.nan
    inter = float(np.logical_and(y, e).sum())
    union = float(np.logical_or(y, e).sum())
    inside = inter / yolk_area
    iou = inter / union if union > 0 else np.nan
    return inside, iou


def _sample_even_by_time(exp_df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or exp_df.empty:
        return exp_df.iloc[0:0].copy()
    if len(exp_df) <= n:
        return exp_df.copy()
    exp_df = exp_df.sort_values(["time_int", "frame_index", "embryo_id"], na_position="last").reset_index(drop=True)
    idx = np.unique(np.round(np.linspace(0, len(exp_df) - 1, num=n)).astype(int))
    sampled = exp_df.iloc[idx].copy()
    if len(sampled) < n:
        remaining = exp_df.drop(index=idx)
        add_n = min(n - len(sampled), len(remaining))
        if add_n > 0:
            sampled = pd.concat(
                [sampled, remaining.sample(n=add_n, random_state=seed)],
                ignore_index=True,
            )
    return sampled.reset_index(drop=True)


def _selected_yolk_path(seg_root: Path, row: pd.Series, keyword: str = "yolk") -> tuple[Optional[Path], int]:
    date = _exp_str(row.get("experiment_date", ""))
    well = row.get("well", None)
    stub_time = _extract_time_stub(row)
    if not date or well is None or stub_time is None:
        return None, 0
    stub = f"{well}_t{stub_time}"

    total_candidates = 0
    selected = None
    if not seg_root.exists():
        return None, 0
    for p in seg_root.iterdir():
        if not p.is_dir() or keyword not in p.name:
            continue
        date_dir = p / date
        if not date_dir.exists():
            continue
        candidates = sorted(date_dir.glob(f"*{stub}*"))
        total_candidates += len(candidates)
        if selected is None and candidates:
            selected = candidates[0]
    return selected, total_candidates


def _source_artifact_paths(data_root: Path, row: pd.Series) -> list[Path]:
    date = _exp_str(row.get("experiment_date", ""))
    snip_id = str(row.get("snip_id", "")).strip()
    if not snip_id:
        return []
    return [
        data_root / "training_data" / "bf_embryo_snips" / date / f"{snip_id}.jpg",
        data_root / "training_data" / "bf_embryo_snips_uncropped" / date / f"{snip_id}.jpg",
        data_root / "training_data" / "bf_embryo_masks" / f"emb_{snip_id}.jpg",
    ]


def _mtime(path: Path) -> Optional[float]:
    return path.stat().st_mtime if path.exists() else None


def _mtime_iso(ts: Optional[float]) -> str:
    if ts is None:
        return ""
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def _infer_experiments_from_run_dir(run_dir: Path, full_df: pd.DataFrame) -> list[str]:
    meta_path = run_dir / "feature_dataset" / "metadata.parquet"
    if not meta_path.exists():
        return []
    run_meta = pd.read_parquet(meta_path)
    if not {"embryo_id", "frame_index"}.issubset(run_meta.columns):
        return []
    run_keys = run_meta.loc[:, ["embryo_id", "frame_index"]].drop_duplicates()
    merged = full_df.merge(run_keys, on=["embryo_id", "frame_index"], how="inner")
    exps = sorted({_exp_str(v) for v in merged["experiment_date"].tolist() if _exp_str(v)})
    return exps


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    usecols = [
        "experiment_date",
        "well",
        "time_int",
        "embryo_id",
        "frame_index",
        "snip_id",
        "mask_rle",
        "mask_height_px",
        "mask_width_px",
    ]
    df = pd.read_csv(args.data_csv, usecols=usecols, low_memory=False)
    df["experiment_date"] = df["experiment_date"].map(_exp_str)
    df = df.drop_duplicates(subset=["embryo_id", "frame_index"]).reset_index(drop=True)

    if args.experiments.strip():
        experiments = [x.strip() for x in args.experiments.split(",") if x.strip()]
    else:
        experiments = _infer_experiments_from_run_dir(args.run_dir, df)
        if not experiments:
            experiments = sorted([x for x in df["experiment_date"].unique().tolist() if x])

    seg_root = args.data_root / "segmentation"
    stale_tol_sec = float(args.stale_tolerance_hours) * 3600.0

    rows = []
    for exp in experiments:
        exp_df = df[df["experiment_date"] == exp].copy()
        sample_df = _sample_even_by_time(exp_df, args.samples_per_experiment, seed=args.seed)
        for _, row in sample_df.iterrows():
            embryo_mask = fmio.load_mask_from_rle_counts(
                rle_counts=row["mask_rle"],
                height_px=int(row["mask_height_px"]),
                width_px=int(row["mask_width_px"]),
            ).astype(np.uint8)

            yolk_path, n_cands = _selected_yolk_path(seg_root, row, keyword="yolk")
            yolk_mask = None
            if yolk_path is not None and yolk_path.exists():
                yolk_mask = _load_binary_mask(yolk_path, target_shape=embryo_mask.shape)

            inside_ratio = np.nan
            yolk_iou = np.nan
            yolk_area = 0
            if yolk_mask is not None:
                inside_ratio, yolk_iou = _overlap_metrics(yolk_mask, embryo_mask)
                yolk_area = int(yolk_mask.sum())

            src_paths = _source_artifact_paths(args.data_root, row)
            src_existing = [p for p in src_paths if p.exists()]
            src_mtimes = [_mtime(p) for p in src_existing]
            src_mtimes = [t for t in src_mtimes if t is not None]
            src_ref_mtime = max(src_mtimes) if src_mtimes else None
            yolk_mtime = _mtime(yolk_path) if yolk_path is not None else None

            stale_delta_sec = np.nan
            stale_flag = False
            if src_ref_mtime is not None and yolk_mtime is not None:
                stale_delta_sec = float(src_ref_mtime - yolk_mtime)
                stale_flag = stale_delta_sec > stale_tol_sec

            low_inside_flag = bool(np.isfinite(inside_ratio) and inside_ratio < args.min_yolk_inside_ratio)
            missing_yolk_flag = yolk_path is None
            missing_source_flag = len(src_existing) == 0
            yolk_empty_flag = bool(yolk_mask is not None and yolk_area == 0)

            rows.append(
                {
                    "experiment_date": exp,
                    "embryo_id": row["embryo_id"],
                    "frame_index": int(row["frame_index"]),
                    "time_int": int(row["time_int"]) if pd.notnull(row["time_int"]) else np.nan,
                    "snip_id": row["snip_id"],
                    "yolk_inside_embryo_ratio": inside_ratio,
                    "yolk_embryo_iou": yolk_iou,
                    "embryo_area_px": int(embryo_mask.sum()),
                    "yolk_area_px": yolk_area,
                    "n_yolk_candidates": int(n_cands),
                    "selected_yolk_path": str(yolk_path) if yolk_path is not None else "",
                    "selected_yolk_mtime": _mtime_iso(yolk_mtime),
                    "source_ref_mtime": _mtime_iso(src_ref_mtime),
                    "stale_delta_hours": (stale_delta_sec / 3600.0) if np.isfinite(stale_delta_sec) else np.nan,
                    "stale_flag": bool(stale_flag),
                    "low_inside_flag": bool(low_inside_flag),
                    "missing_yolk_flag": bool(missing_yolk_flag),
                    "missing_source_artifact_flag": bool(missing_source_flag),
                    "yolk_empty_flag": bool(yolk_empty_flag),
                    "source_artifacts": ";".join(str(p) for p in src_existing),
                }
            )

    detail_df = pd.DataFrame(rows)
    if detail_df.empty:
        raise SystemExit("No rows were audited. Check experiment selection and input paths.")

    detail_df["issue_flag"] = (
        detail_df["stale_flag"]
        | detail_df["low_inside_flag"]
        | detail_df["missing_yolk_flag"]
        | detail_df["missing_source_artifact_flag"]
        | detail_df["yolk_empty_flag"]
    )

    summary_df = (
        detail_df.groupby("experiment_date", dropna=False)
        .agg(
            n_sampled=("experiment_date", "size"),
            n_issues=("issue_flag", "sum"),
            n_stale=("stale_flag", "sum"),
            n_low_inside=("low_inside_flag", "sum"),
            n_missing_yolk=("missing_yolk_flag", "sum"),
            n_missing_source=("missing_source_artifact_flag", "sum"),
            n_yolk_empty=("yolk_empty_flag", "sum"),
            mean_inside_ratio=("yolk_inside_embryo_ratio", "mean"),
            p10_inside_ratio=("yolk_inside_embryo_ratio", lambda s: float(np.nanpercentile(s, 10))),
        )
        .reset_index()
        .sort_values("experiment_date")
    )
    summary_df["issue_rate"] = summary_df["n_issues"] / summary_df["n_sampled"].clip(lower=1)

    detail_path = args.out_dir / "yolk_audit_samples.csv"
    issue_path = args.out_dir / "yolk_audit_issue_rows.csv"
    summary_path = args.out_dir / "yolk_audit_summary_by_experiment.csv"
    detail_df.to_csv(detail_path, index=False)
    detail_df[detail_df["issue_flag"]].to_csv(issue_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Wrote: {detail_path}")
    print(f"Wrote: {issue_path}")
    print(f"Wrote: {summary_path}")
    print("\nSummary (sorted by issue_rate desc):")
    print(summary_df.sort_values("issue_rate", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()

