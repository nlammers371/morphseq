#!/usr/bin/env python3
"""DCT compaction fidelity spike on the canonical Stream A comparison pair.

This script runs one OT solve (POT or OTT) on the same embryo pair used in
Stream A visual checks, then evaluates top-K DCT coefficient compaction of the
velocity field.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.fft import dctn, idctn

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle
from analyze.utils.coord.grids.canonical import CanonicalGridConfig, to_canonical_grid_mask
from analyze.utils.optimal_transport import (
    MassMode,
    UOTConfig,
    UOTFrame,
    UOTFramePair,
    WorkingGridConfig,
    lift_work_result_to_canonical,
    prepare_working_grid_pair,
    run_uot_on_working_grid,
)
from analyze.optimal_transport_morphometrics.uot_masks.feature_compaction.storage import (
    build_pair_id,
    build_pair_metrics_record,
    upsert_ot_pair_metrics_parquet,
    save_pair_artifacts,
)
from analyze.optimal_transport_morphometrics.uot_masks.feature_compaction.features import (
    extract_pair_feature_record,
    upsert_ot_feature_matrix_parquet,
)


CSV_PATH_DEFAULT = (
    PROJECT_ROOT
    / "results"
    / "mcolon"
    / "20251229_cep290_phenotype_extraction"
    / "final_data"
    / "embryo_data_with_labels.csv"
)
EMBRYO_A = "20251113_A05_e01"
EMBRYO_B = "20251113_E04_e01"
TARGET_HPF = 48.0
STAGE_TOL = 1.0
REG_M = 10.0
EPSILON = 1e-4
CANONICAL_GRID_SHAPE = (256, 576)
UM_PER_PX = 10.0
COORD_SCALE = 1.0 / max(CANONICAL_GRID_SHAPE)

STREAM_F_ROOT = (
    PROJECT_ROOT
    / "src"
    / "analyze"
    / "optimal_transport_morphometrics"
    / "docs"
    / "phase2_implemnetation_tracking"
    / "stream_f_feature_development"
)

DEFAULT_USECOLS = [
    "experiment_date",
    "experiment_id",
    "well",
    "time_int",
    "embryo_id",
    "frame_index",
    "mask_rle",
    "mask_height_px",
    "mask_width_px",
    "image_id",
    "snip_id",
    "relative_time_s",
    "raw_time_s",
    "Height (um)",
    "Height (px)",
    "Width (um)",
    "Width (px)",
]


def find_frame_at_stage(csv_path: Path, embryo_id: str, target_hpf: float, tolerance_hpf: float) -> Tuple[int, float]:
    df = pd.read_csv(csv_path, usecols=["embryo_id", "frame_index", "predicted_stage_hpf"])
    subset = df[
        (df["embryo_id"] == embryo_id)
        & (df["predicted_stage_hpf"] >= target_hpf - tolerance_hpf)
        & (df["predicted_stage_hpf"] <= target_hpf + tolerance_hpf)
    ].copy()
    if subset.empty:
        raise ValueError(
            f"No frame found near {target_hpf} +/- {tolerance_hpf} hpf for embryo_id={embryo_id}"
        )
    subset["dist"] = (subset["predicted_stage_hpf"] - target_hpf).abs()
    row = subset.loc[subset["dist"].idxmin()]
    return int(row["frame_index"]), float(row["predicted_stage_hpf"])


def make_config(epsilon: float, reg_m: float, max_support_points: int) -> UOTConfig:
    return UOTConfig(
        epsilon=epsilon,
        marginal_relaxation=reg_m,
        max_support_points=int(max_support_points),
        store_coupling=True,
        random_seed=42,
        metric="sqeuclidean",
        coord_scale=COORD_SCALE,
    )

def make_working_grid_config() -> WorkingGridConfig:
    return WorkingGridConfig(
        downsample_factor=1,
        padding_px=16,
        mass_mode=MassMode.UNIFORM,
    )


def _ensure_2d(mask: np.ndarray) -> np.ndarray:
    if mask.ndim > 2:
        mask = mask.squeeze()
        if mask.ndim > 2:
            mask = mask[..., 0]
    return mask


def _load_mask_from_rle_counts(rle_counts: str, height_px: int, width_px: int) -> np.ndarray:
    rle_data = {"counts": rle_counts, "size": [int(height_px), int(width_px)]}
    mask = decode_mask_rle(rle_data)
    mask = _ensure_2d(mask)
    return mask.astype(np.uint8)


def _extract_time_stub(row: pd.Series) -> str | None:
    for key in ("time_int", "frame_index"):
        if key in row and pd.notnull(row[key]):
            try:
                return f"{int(row[key]):04d}"
            except (TypeError, ValueError):
                continue
    return None


def _compute_um_per_pixel(row: pd.Series) -> float:
    if "Height (um)" in row and "Height (px)" in row:
        height_um = float(row["Height (um)"])
        height_px = float(row["Height (px)"])
        if height_px > 0:
            return height_um / height_px
    if "Width (um)" in row and "Width (px)" in row:
        width_um = float(row["Width (um)"])
        width_px = float(row["Width (px)"])
        if width_px > 0:
            return width_um / width_px
    return float("nan")


def _load_yolk_mask(data_root: Path, row: pd.Series, mask_shape: Tuple[int, int]) -> np.ndarray | None:
    seg_root = data_root / "segmentation"
    if not seg_root.exists():
        return None

    date = str(row.get("experiment_date", ""))
    well = row.get("well", None)
    time_stub = _extract_time_stub(row)
    if not date or well is None or time_stub is None:
        return None

    stub = f"{well}_t{time_stub}"
    for p in seg_root.iterdir():
        if not p.is_dir() or "yolk" not in p.name:
            continue
        date_dir = p / date
        if not date_dir.exists():
            continue
        candidates = sorted(date_dir.glob(f"*{stub}*"))
        if not candidates:
            continue
        arr_raw = cv2.imread(str(candidates[0]), cv2.IMREAD_UNCHANGED)
        if arr_raw is None:
            continue
        arr = _ensure_2d(arr_raw)
        if arr.max() >= 255:
            arr = (arr > 127).astype(np.uint8)
        else:
            arr = (arr > 0).astype(np.uint8)
        if arr.shape != mask_shape:
            arr = cv2.resize(arr, (mask_shape[1], mask_shape[0]), interpolation=cv2.INTER_NEAREST)
            arr = (arr > 0).astype(np.uint8)
        return arr
    return None


def _load_frame_from_csv(csv_path: Path, embryo_id: str, frame_index: int, data_root: Path) -> UOTFrame:
    df = pd.read_csv(csv_path, usecols=DEFAULT_USECOLS, low_memory=False)
    subset = df[(df["embryo_id"] == embryo_id) & (df["frame_index"] == frame_index)]
    if subset.empty:
        raise ValueError(f"No mask found for embryo_id={embryo_id} frame_index={frame_index}")

    row = subset.iloc[0]
    mask = _load_mask_from_rle_counts(row["mask_rle"], row["mask_height_px"], row["mask_width_px"])
    meta = row.to_dict()
    meta["um_per_pixel"] = _compute_um_per_pixel(row)

    yolk_mask = _load_yolk_mask(data_root, row, mask.shape)
    if yolk_mask is not None:
        meta["yolk_mask"] = yolk_mask

    return UOTFrame(embryo_mask=mask, meta=meta)


def _support_mask(result) -> np.ndarray:
    vel_mag = np.linalg.norm(result.velocity_canon_px_per_step_yx, axis=-1)
    return (
        (result.mass_created_canon > 0)
        | (result.mass_destroyed_canon > 0)
        | (vel_mag > 0)
    )


def _dct_reconstruct_topk(
    velocity_hw2: np.ndarray,
    k_freq: int,
) -> Tuple[np.ndarray, float]:
    """Reconstruct velocity from top-k spatial frequencies by combined power."""
    vy = velocity_hw2[..., 0]
    vx = velocity_hw2[..., 1]

    cy = dctn(vy, type=2, norm="ortho")
    cx = dctn(vx, type=2, norm="ortho")

    total_freq = cy.size
    k_freq = int(max(1, min(total_freq, k_freq)))

    score = (cy ** 2) + (cx ** 2)
    flat_score = score.ravel()

    keep_flat = np.zeros_like(flat_score, dtype=bool)
    if k_freq >= flat_score.size:
        keep_flat[:] = True
    else:
        idx = np.argpartition(flat_score, -k_freq)[-k_freq:]
        keep_flat[idx] = True
    keep_mask = keep_flat.reshape(score.shape)

    cy_k = np.where(keep_mask, cy, 0.0)
    cx_k = np.where(keep_mask, cx, 0.0)

    vy_rec = idctn(cy_k, type=2, norm="ortho")
    vx_rec = idctn(cx_k, type=2, norm="ortho")
    rec = np.stack([vy_rec, vx_rec], axis=-1).astype(np.float32)

    total_energy = float(np.sum(score))
    kept_energy = float(np.sum(score[keep_mask]))
    energy_retained = 0.0 if total_energy <= 0 else kept_energy / total_energy
    return rec, energy_retained


def _divergence_and_curl(velocity_hw2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vy = velocity_hw2[..., 0]
    vx = velocity_hw2[..., 1]
    dvy_dy, dvy_dx = np.gradient(vy)
    dvx_dy, dvx_dx = np.gradient(vx)
    divergence = dvy_dy + dvx_dx
    curl = dvx_dy - dvy_dx
    return divergence, curl


def _evaluate_reconstruction(
    original_hw2: np.ndarray,
    reconstructed_hw2: np.ndarray,
    support_mask: np.ndarray,
) -> Dict[str, float]:
    if not np.any(support_mask):
        raise ValueError("Support mask is empty; cannot evaluate reconstruction fidelity.")

    orig = original_hw2[support_mask]
    rec = reconstructed_hw2[support_mask]
    diff = rec - orig

    rmse = float(np.sqrt(np.mean(diff ** 2)))
    orig_rms = float(np.sqrt(np.mean(orig ** 2)))
    rmse_rel = float(rmse / max(orig_rms, 1e-12))
    endpoint_err = np.linalg.norm(diff, axis=1)
    endpoint_rmse = float(np.sqrt(np.mean(endpoint_err ** 2)))
    endpoint_orig = np.linalg.norm(orig, axis=1)
    endpoint_orig_rms = float(np.sqrt(np.mean(endpoint_orig ** 2)))
    endpoint_rmse_rel = float(endpoint_rmse / max(endpoint_orig_rms, 1e-12))
    endpoint_p95 = float(np.percentile(endpoint_err, 95))

    dot = np.sum(orig * rec, axis=1)
    denom = np.linalg.norm(orig, axis=1) * np.linalg.norm(rec, axis=1)
    valid = denom > 1e-12
    cosine = float(np.mean(dot[valid] / denom[valid])) if np.any(valid) else 1.0

    div_o, curl_o = _divergence_and_curl(original_hw2)
    div_r, curl_r = _divergence_and_curl(reconstructed_hw2)
    div_rmse = float(np.sqrt(np.mean((div_r[support_mask] - div_o[support_mask]) ** 2)))
    curl_rmse = float(np.sqrt(np.mean((curl_r[support_mask] - curl_o[support_mask]) ** 2)))

    return {
        "rmse_support": rmse,
        "rmse_rel_support": rmse_rel,
        "endpoint_rmse_support": endpoint_rmse,
        "endpoint_rmse_rel_support": endpoint_rmse_rel,
        "endpoint_p95_support": endpoint_p95,
        "cosine_similarity_support": cosine,
        "divergence_rmse_support": div_rmse,
        "curl_rmse_support": curl_rmse,
    }


def _parse_ratios(ratio_csv: str) -> List[float]:
    vals = []
    for token in ratio_csv.split(","):
        t = token.strip()
        if not t:
            continue
        vals.append(float(t))
    if not vals:
        raise ValueError("No coefficient ratios parsed from input.")
    return vals


def _get_backend(name: str):
    name = name.lower()
    if name == "pot":
        from analyze.utils.optimal_transport.backends.pot_backend import POTBackend

        return POTBackend(), "POT"
    if name == "ott":
        from analyze.utils.optimal_transport.backends.ott_backend import OTTBackend

        return OTTBackend(), "OTT"
    raise ValueError(f"Unsupported backend: {name}")


def run_spike(args):
    csv_path = Path(args.csv).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    frame_a, stage_a = find_frame_at_stage(csv_path, args.embryo_a, args.target_hpf, args.stage_tol)
    frame_b, stage_b = find_frame_at_stage(csv_path, args.embryo_b, args.target_hpf, args.stage_tol)

    data_root = PROJECT_ROOT / "morphseq_playground"
    src_frame = _load_frame_from_csv(csv_path, args.embryo_a, frame_a, data_root=data_root)
    tgt_frame = _load_frame_from_csv(csv_path, args.embryo_b, frame_b, data_root=data_root)
    pair = UOTFramePair(src=src_frame, tgt=tgt_frame)

    backend, backend_name = _get_backend(args.backend)
    solver_cfg = make_config(args.epsilon, args.reg_m, args.max_support_points)
    working_cfg = make_working_grid_config()
    canonical_cfg = CanonicalGridConfig(
        reference_um_per_pixel=UM_PER_PX,
        grid_shape_hw=CANONICAL_GRID_SHAPE,
        align_mode="yolk",
    )

    run_id = args.run_id or f"dct_spike_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    pair_id = build_pair_id(args.embryo_a, args.embryo_b, frame_a, frame_b)

    t0 = time.time()
    # --- Canonicalize (coord owns geometry) ---
    um_src = float(src_frame.meta.get("um_per_pixel", float("nan"))) if src_frame.meta else float("nan")
    um_tgt = float(tgt_frame.meta.get("um_per_pixel", float("nan"))) if tgt_frame.meta else float("nan")
    yolk_src = src_frame.meta.get("yolk_mask") if src_frame.meta else None
    yolk_tgt = tgt_frame.meta.get("yolk_mask") if tgt_frame.meta else None

    src_can = to_canonical_grid_mask(
        np.asarray(src_frame.embryo_mask),
        um_per_px=um_src,
        yolk_mask=yolk_src,
        cfg=canonical_cfg,
    )
    tgt_can = to_canonical_grid_mask(
        np.asarray(tgt_frame.embryo_mask),
        um_per_px=um_tgt,
        yolk_mask=yolk_tgt,
        cfg=canonical_cfg,
    )

    # --- Prepare work grid + solve (solver is math-only) ---
    pair_work = prepare_working_grid_pair(src_can, tgt_can, working_cfg)
    result_work = run_uot_on_working_grid(pair_work, config=solver_cfg, backend=backend)
    result_canon = lift_work_result_to_canonical(result_work, pair_work)
    runtime_sec = time.time() - t0

    cfg_record = {
        "epsilon": float(solver_cfg.epsilon),
        "marginal_relaxation": float(solver_cfg.marginal_relaxation),
        "metric": str(solver_cfg.metric),
        "coord_scale": float(solver_cfg.coord_scale),
        "max_support_points": int(solver_cfg.max_support_points),
        "mass_mode": working_cfg.mass_mode.value,
        "downsample_factor": int(working_cfg.downsample_factor),
        "padding_px": int(working_cfg.padding_px),
        "use_canonical_grid": True,
        "canonical_grid_align_mode": str(canonical_cfg.align_mode),
        "canonical_grid_um_per_pixel": float(canonical_cfg.reference_um_per_pixel),
        "canonical_grid_shape_hw": tuple(canonical_cfg.grid_shape_hw),
    }

    record = build_pair_metrics_record(
        run_id=run_id,
        pair_id=pair_id,
        result=result_work,
        src_meta=src_frame.meta,
        tgt_meta=tgt_frame.meta,
        config=cfg_record,
        backend=backend_name,
        runtime_sec=runtime_sec,
        success=True,
        error_message=None,
    )
    metrics_path = output_root / "ot_pair_metrics.parquet"
    upsert_ot_pair_metrics_parquet(metrics_path, [record])

    artifact_root = output_root / "pair_artifacts"
    artifact_paths = save_pair_artifacts(
        result_work=result_work,
        result_canon=result_canon,
        artifact_root=artifact_root,
        pair_id=pair_id,
        float_dtype="float32",
        include_barycentric=True,
    )

    velocity = np.asarray(result_canon.velocity_canon_px_per_step_yx, dtype=np.float32)
    support_mask = _support_mask(result_canon)
    total_freq = velocity.shape[0] * velocity.shape[1]
    ratios = _parse_ratios(args.ratios)

    rows = []
    for ratio in ratios:
        k_freq = int(max(1, round(total_freq * ratio)))
        rec, energy_retained = _dct_reconstruct_topk(velocity, k_freq=k_freq)
        stats = _evaluate_reconstruction(velocity, rec, support_mask)
        rows.append(
            {
                "run_id": run_id,
                "pair_id": pair_id,
                "backend": backend_name,
                "epsilon": args.epsilon,
                "reg_m": args.reg_m,
                "k_freq": k_freq,
                "k_ratio": ratio,
                "energy_retained": energy_retained,
                **stats,
            }
        )

    df = pd.DataFrame(rows).sort_values("k_freq").reset_index(drop=True)
    csv_out = output_root / f"dct_sweep_metrics_{run_id}.csv"
    df.to_csv(csv_out, index=False)

    # Also keep/update a "latest" convenience path.
    latest_csv_out = output_root / "dct_sweep_metrics.csv"
    df.to_csv(latest_csv_out, index=False)

    criteria = (
        (df["cosine_similarity_support"] >= args.min_cosine)
        & (df["rmse_rel_support"] <= args.max_rmse_rel)
        & (df["rmse_support"] <= args.max_rmse)
    )
    if criteria.any():
        recommended = df.loc[criteria].iloc[0].to_dict()
    else:
        recommended = df.iloc[-1].to_dict()

    summary = {
        "run_id": run_id,
        "pair_id": pair_id,
        "backend": backend_name,
        "csv_path": str(csv_path),
        "embryo_a": args.embryo_a,
        "embryo_b": args.embryo_b,
        "frame_a": frame_a,
        "frame_b": frame_b,
        "stage_a_hpf": stage_a,
        "stage_b_hpf": stage_b,
        "support_pixels": int(support_mask.sum()),
        "total_freq": total_freq,
        "thresholds": {
            "min_cosine": args.min_cosine,
            "max_rmse": args.max_rmse,
            "max_rmse_rel": args.max_rmse_rel,
        },
        "max_support_points": int(args.max_support_points),
        "recommended": recommended,
        "outputs": {
            "ot_pair_metrics_parquet": str(metrics_path),
            "ot_feature_matrix_parquet": str(output_root / "ot_feature_matrix.parquet"),
            "dct_sweep_metrics_csv": str(csv_out),
            "dct_sweep_metrics_latest_csv": str(output_root / "dct_sweep_metrics.csv"),
            "artifact_paths": {k: str(v) for k, v in artifact_paths.items()},
        },
    }
    feature_record = extract_pair_feature_record(
        run_id=run_id,
        pair_id=pair_id,
        result_work=result_work,
        result_canon=result_canon,
        backend=backend_name,
        n_bands=args.n_dct_bands,
    )
    feature_matrix_path = output_root / "ot_feature_matrix.parquet"
    upsert_ot_feature_matrix_parquet(feature_matrix_path, [feature_record])

    summary_out = output_root / f"dct_sweep_summary_{run_id}.json"
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Also keep/update a "latest" convenience path.
    latest_summary_out = output_root / "dct_sweep_summary.json"
    with open(latest_summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Run ID: {run_id}")
    print(f"Pair: {pair_id}")
    print(
        "Loaded stages (hpf): "
        f"{args.embryo_a}={stage_a:.2f}, {args.embryo_b}={stage_b:.2f}"
    )
    print(f"OT solve runtime: {runtime_sec:.2f}s")
    print(f"Wrote metrics parquet: {metrics_path}")
    print(f"Wrote feature matrix parquet: {feature_matrix_path}")
    print(f"Wrote DCT sweep CSV: {csv_out}")
    print(
        "Recommended K: "
        f"k={int(recommended['k_freq'])} "
        f"(ratio={float(recommended['k_ratio']):.4f}, "
        f"cos={float(recommended['cosine_similarity_support']):.4f}, "
        f"rmse={float(recommended['rmse_support']):.4f}, "
        f"rmse_rel={float(recommended['rmse_rel_support']):.4f})"
    )
    print(f"Wrote summary JSON: {summary_out}")


def parse_args():
    parser = argparse.ArgumentParser(description="DCT compaction spike for UOT velocity fields.")
    parser.add_argument("--csv", type=Path, default=CSV_PATH_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=STREAM_F_ROOT / "dct_spike_results")
    parser.add_argument("--backend", type=str, default="pot", choices=["pot", "ott"])
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    parser.add_argument("--reg-m", type=float, default=REG_M)
    parser.add_argument("--embryo-a", type=str, default=EMBRYO_A)
    parser.add_argument("--embryo-b", type=str, default=EMBRYO_B)
    parser.add_argument("--target-hpf", type=float, default=TARGET_HPF)
    parser.add_argument("--stage-tol", type=float, default=STAGE_TOL)
    parser.add_argument(
        "--ratios",
        type=str,
        default="0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.75,1.0",
    )
    parser.add_argument("--min-cosine", type=float, default=0.98)
    parser.add_argument(
        "--max-rmse-rel",
        type=float,
        default=0.10,
        help="Maximum relative RMSE on support (RMSE / RMS(original)); used for K recommendation.",
    )
    parser.add_argument(
        "--max-rmse",
        type=float,
        default=float("inf"),
        help="Optional absolute RMSE cap on support for K recommendation.",
    )
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--max-support-points", type=int, default=5000)
    parser.add_argument("--n-dct-bands", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run_spike(parse_args())
