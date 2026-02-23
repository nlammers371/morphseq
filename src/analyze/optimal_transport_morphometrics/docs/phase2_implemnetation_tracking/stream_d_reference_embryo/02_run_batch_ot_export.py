#!/usr/bin/env python3
"""Batch OT export for selected 24-48 hpf cohort transitions.

This module only runs and persists OT outputs (no reference/PCA logic).
It is resume-safe via (run_id, pair_id) upsert keys.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[6]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyze.utils.optimal_transport import MassMode, UOTConfig, UOTFrame, UOTFramePair
from analyze.utils.optimal_transport.backends.ott_backend import OTTBackend
from analyze.utils.optimal_transport.backends.pot_backend import POTBackend
from analyze.optimal_transport_morphometrics.uot_masks import frame_mask_io as fmio
from analyze.optimal_transport_morphometrics.uot_masks.run_transport import run_uot_pair
from analyze.optimal_transport_morphometrics.uot_masks.feature_compaction.storage import (
    build_pair_metrics_record,
    upsert_ot_pair_metrics_parquet,
    save_pair_artifacts,
)
from analyze.optimal_transport_morphometrics.uot_masks.feature_compaction.features import (
    extract_pair_feature_record,
    upsert_ot_feature_matrix_parquet,
)


DEFAULT_CSV = (
    PROJECT_ROOT
    / "results"
    / "mcolon"
    / "20251229_cep290_phenotype_extraction"
    / "final_data"
    / "embryo_data_with_labels.csv"
)
DEFAULT_TRANSITIONS = (
    Path(__file__).resolve().parent
    / "cohort_selection"
    / "cohort_transition_manifest.csv"
)
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "ot_24_48_exports"


def _make_config(epsilon: float, reg_m: float, max_support_points: int) -> UOTConfig:
    shape = (256, 576)
    return UOTConfig(
        epsilon=float(epsilon),
        marginal_relaxation=float(reg_m),
        downsample_factor=1,
        downsample_divisor=1,
        padding_px=16,
        mass_mode=MassMode.UNIFORM,
        align_mode="none",
        max_support_points=int(max_support_points),
        store_coupling=True,
        random_seed=42,
        metric="sqeuclidean",
        coord_scale=1.0 / max(shape),
        use_pair_frame=True,
        use_canonical_grid=True,
        canonical_grid_um_per_pixel=10.0,
        canonical_grid_shape_hw=shape,
        canonical_grid_align_mode="yolk",
        canonical_grid_center_mode="joint_centering",
    )


def _build_backend(name: str):
    if name.lower() == "ott":
        return OTTBackend(), "OTT"
    if name.lower() == "pot":
        return POTBackend(), "POT"
    raise ValueError(f"Unsupported backend: {name}")


def _build_key_set(rows: pd.DataFrame) -> set[Tuple[str, str]]:
    if rows.empty or "run_id" not in rows.columns or "pair_id" not in rows.columns:
        return set()
    return set(zip(rows["run_id"].astype(str), rows["pair_id"].astype(str)))


def _required_frame_keys(transitions: pd.DataFrame) -> set[Tuple[str, int]]:
    keys: set[Tuple[str, int]] = set()
    for row in transitions.itertuples(index=False):
        if getattr(row, "is_control_pair", False):
            src_emb = getattr(row, "src_embryo_id", None)
            tgt_emb = getattr(row, "tgt_embryo_id", None)
            if pd.notna(src_emb):
                keys.add((str(src_emb), int(row.frame_src)))
            if pd.notna(tgt_emb):
                keys.add((str(tgt_emb), int(row.frame_tgt)))
        else:
            keys.add((str(row.embryo_id), int(row.frame_src)))
            keys.add((str(row.embryo_id), int(row.frame_tgt)))
    return keys


def _load_frame_row_map(
    csv_path: Path,
    needed_keys: set[Tuple[str, int]],
) -> Dict[Tuple[str, int], pd.Series]:
    if not needed_keys:
        return {}
    df = pd.read_csv(csv_path, usecols=fmio.DEFAULT_USECOLS, low_memory=False)
    needed_df = pd.DataFrame(list(needed_keys), columns=["embryo_id", "frame_index"])
    merged = df.merge(needed_df, on=["embryo_id", "frame_index"], how="inner")
    if merged.empty:
        return {}
    out: Dict[Tuple[str, int], pd.Series] = {}
    for _, row in merged.iterrows():
        out[(str(row["embryo_id"]), int(row["frame_index"]))] = row
    return out


def _row_to_frame(row: pd.Series, data_root: Path | None) -> UOTFrame:
    mask = fmio.load_mask_from_rle_counts(row["mask_rle"], row["mask_height_px"], row["mask_width_px"])
    meta = row.to_dict()
    meta["um_per_pixel"] = fmio._compute_um_per_pixel(row)
    if data_root is not None:
        yolk = fmio._load_build02_aux_mask(data_root, row, mask.shape, keyword="yolk")
        if yolk is not None:
            meta["yolk_mask"] = yolk
    return UOTFrame(embryo_mask=mask, meta=meta)


def _prepare_transitions(df: pd.DataFrame, include_control: bool) -> pd.DataFrame:
    work = df.copy()
    if "analysis_use" in work.columns:
        work["analysis_use"] = work["analysis_use"].astype(bool)
    else:
        work["analysis_use"] = True
    if "is_control_pair" in work.columns:
        work["is_control_pair"] = work["is_control_pair"].astype(bool)
    else:
        work["is_control_pair"] = False

    if include_control:
        keep = (work["analysis_use"]) | (work["is_control_pair"])
    else:
        keep = work["analysis_use"]
    work = work[keep].copy()
    return work.reset_index(drop=True)


def run_export(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_root = output_root / "pair_artifacts"

    transitions_raw = pd.read_csv(args.transitions, low_memory=False)
    transitions = _prepare_transitions(transitions_raw, include_control=args.include_control)
    if args.limit > 0:
        transitions = transitions.iloc[: args.limit].copy()
    if transitions.empty:
        raise ValueError("No transitions to process after filtering.")

    backend, backend_name = _build_backend(args.backend)
    cfg = _make_config(epsilon=args.epsilon, reg_m=args.reg_m, max_support_points=args.max_support_points)
    run_id = args.run_id

    metrics_path = output_root / "ot_pair_metrics.parquet"
    features_path = output_root / "ot_feature_matrix.parquet"
    log_path = output_root / f"run_log_{run_id}.csv"

    existing = pd.read_parquet(metrics_path) if metrics_path.exists() else pd.DataFrame()
    done_keys = _build_key_set(existing)

    needed = _required_frame_keys(transitions)
    frame_row_map = _load_frame_row_map(args.csv, needed)

    data_root = Path(args.data_root).resolve() if args.data_root else None
    frame_cache: Dict[Tuple[str, int], UOTFrame] = {}

    metric_buffer: List[Dict] = []
    feat_buffer: List[Dict] = []
    log_rows: List[Dict] = []

    n_total = len(transitions)
    n_done = 0
    n_skip = 0
    n_fail = 0

    def flush() -> None:
        nonlocal metric_buffer, feat_buffer
        if metric_buffer:
            upsert_ot_pair_metrics_parquet(metrics_path, metric_buffer)
            metric_buffer = []
        if feat_buffer:
            upsert_ot_feature_matrix_parquet(features_path, feat_buffer)
            feat_buffer = []

    for idx, row in enumerate(transitions.itertuples(index=False), start=1):
        pair_id = str(row.pair_id)
        key = (run_id, pair_id)
        if args.resume and key in done_keys:
            n_skip += 1
            continue

        if bool(getattr(row, "is_control_pair", False)):
            src_emb = str(getattr(row, "src_embryo_id"))
            tgt_emb = str(getattr(row, "tgt_embryo_id"))
        else:
            src_emb = str(row.embryo_id)
            tgt_emb = str(row.embryo_id)

        src_key = (src_emb, int(row.frame_src))
        tgt_key = (tgt_emb, int(row.frame_tgt))

        if src_key not in frame_cache:
            src_row = frame_row_map.get(src_key)
            if src_row is None:
                n_fail += 1
                log_rows.append(
                    {
                        "run_id": run_id,
                        "pair_id": pair_id,
                        "status": "failed",
                        "error": f"Missing source row for {src_key}",
                    }
                )
                continue
            frame_cache[src_key] = _row_to_frame(src_row, data_root=data_root)
        if tgt_key not in frame_cache:
            tgt_row = frame_row_map.get(tgt_key)
            if tgt_row is None:
                n_fail += 1
                log_rows.append(
                    {
                        "run_id": run_id,
                        "pair_id": pair_id,
                        "status": "failed",
                        "error": f"Missing target row for {tgt_key}",
                    }
                )
                continue
            frame_cache[tgt_key] = _row_to_frame(tgt_row, data_root=data_root)

        src_frame = frame_cache[src_key]
        tgt_frame = frame_cache[tgt_key]
        pair = UOTFramePair(src=src_frame, tgt=tgt_frame)

        t0 = time.time()
        error_message = None
        success = True
        try:
            result = run_uot_pair(pair, config=cfg, backend=backend)
            runtime_sec = time.time() - t0
        except Exception as e:  # noqa: BLE001
            success = False
            runtime_sec = time.time() - t0
            error_message = str(e)
            result = None

        if not success or result is None:
            n_fail += 1
            fail_record = {
                "run_id": run_id,
                "pair_id": pair_id,
                "success": False,
                "error_message": error_message,
                "runtime_sec": runtime_sec,
                "backend": backend_name,
                "set_type": str(row.set_type) if hasattr(row, "set_type") else "",
                "set_rank": int(row.set_rank) if hasattr(row, "set_rank") and pd.notna(row.set_rank) else np.nan,
                "genotype": str(row.genotype) if hasattr(row, "genotype") else "",
                "bin_src_hpf": float(row.bin_src_hpf) if hasattr(row, "bin_src_hpf") else np.nan,
                "bin_tgt_hpf": float(row.bin_tgt_hpf) if hasattr(row, "bin_tgt_hpf") else np.nan,
                "is_control_pair": bool(getattr(row, "is_control_pair", False)),
            }
            metric_buffer.append(fail_record)
            log_rows.append({"run_id": run_id, "pair_id": pair_id, "status": "failed", "error": error_message})
        else:
            rec = build_pair_metrics_record(
                run_id=run_id,
                pair_id=pair_id,
                result=result,
                src_meta=src_frame.meta,
                tgt_meta=tgt_frame.meta,
                config=cfg,
                backend=backend_name,
                runtime_sec=runtime_sec,
                success=True,
                error_message=None,
            )
            rec.update(
                {
                    "set_type": str(row.set_type) if hasattr(row, "set_type") else "",
                    "set_rank": int(row.set_rank) if hasattr(row, "set_rank") and pd.notna(row.set_rank) else np.nan,
                    "genotype": str(row.genotype) if hasattr(row, "genotype") else "",
                    "bin_src_hpf": float(row.bin_src_hpf) if hasattr(row, "bin_src_hpf") else np.nan,
                    "bin_tgt_hpf": float(row.bin_tgt_hpf) if hasattr(row, "bin_tgt_hpf") else np.nan,
                    "is_control_pair": bool(getattr(row, "is_control_pair", False)),
                    "src_embryo_id_manifest": src_emb,
                    "tgt_embryo_id_manifest": tgt_emb,
                }
            )
            metric_buffer.append(rec)
            feat = extract_pair_feature_record(
                run_id=run_id,
                pair_id=pair_id,
                result=result,
                backend=backend_name,
                n_bands=args.n_dct_bands,
            )
            feat.update(
                {
                    "set_type": str(row.set_type) if hasattr(row, "set_type") else "",
                    "set_rank": int(row.set_rank) if hasattr(row, "set_rank") and pd.notna(row.set_rank) else np.nan,
                    "genotype": str(row.genotype) if hasattr(row, "genotype") else "",
                    "bin_src_hpf": float(row.bin_src_hpf) if hasattr(row, "bin_src_hpf") else np.nan,
                    "bin_tgt_hpf": float(row.bin_tgt_hpf) if hasattr(row, "bin_tgt_hpf") else np.nan,
                    "is_control_pair": bool(getattr(row, "is_control_pair", False)),
                }
            )
            feat_buffer.append(feat)

            save_pair_artifacts(
                result=result,
                artifact_root=artifact_root,
                pair_id=pair_id,
                float_dtype="float32",
                include_barycentric=True,
            )
            n_done += 1
            log_rows.append(
                {
                    "run_id": run_id,
                    "pair_id": pair_id,
                    "status": "ok",
                    "runtime_sec": runtime_sec,
                    "cost": float(result.cost),
                }
            )

        if (idx % args.flush_every) == 0:
            flush()
            print(
                f"[{idx}/{n_total}] done={n_done} skip={n_skip} fail={n_fail} "
                f"cache_frames={len(frame_cache)}"
            )

    flush()
    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    print(f"Finished run_id={run_id}")
    print(f"  transitions_total={n_total}")
    print(f"  successes={n_done} skipped={n_skip} failed={n_fail}")
    print(f"  metrics_parquet={metrics_path}")
    print(f"  feature_parquet={features_path}")
    print(f"  artifact_root={artifact_root}")
    print(f"  log_csv={log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch OT export for cohort transitions.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--transitions", type=Path, default=DEFAULT_TRANSITIONS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=str, default="phase2_24_48_ott_v1")
    parser.add_argument("--backend", type=str, choices=["ott", "pot"], default="ott")
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--reg-m", type=float, default=10.0)
    parser.add_argument("--max-support-points", type=int, default=3000)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "morphseq_playground",
    )
    parser.add_argument("--n-dct-bands", type=int, default=8)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-control", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=0, help="For smoke runs only.")
    return parser.parse_args()


if __name__ == "__main__":
    run_export(parse_args())
