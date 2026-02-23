#!/usr/bin/env python3
"""
Step-by-step canonical alignment debug for problematic Phase 0 samples.

Focuses on selected problematic samples from:
results/mcolon/20260215_roi_discovery_via_ot_feature_maps/scripts/output/phase0_qc_fix_rerun_alignedmasks

Outputs:
- Per-sample diagnostic figures (raw vs aligned vs reference overlay)
- metrics CSV for quick triage

Usage:
  PYTHON=/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python
  "$PYTHON" results/mcolon/20260216_canonical_aligner_debug/debug_phase0_samples_012_013.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr


MORPHSEQ_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MORPHSEQ_ROOT))
sys.path.insert(0, str(MORPHSEQ_ROOT / "src"))
sys.path.insert(0, str(MORPHSEQ_ROOT / "results/mcolon/20260215_roi_discovery_via_ot_feature_maps"))

from analyze.optimal_transport_morphometrics.uot_masks import frame_mask_io as fmio
from analyze.optimal_transport_morphometrics.uot_masks.preprocess import preprocess_pair_canonical
from analyze.utils.coord.grids.canonical import CanonicalAligner, CanonicalGridConfig
from analyze.utils.optimal_transport import UOTConfig, UOTFrame


RUN_DIR = (
    MORPHSEQ_ROOT
    / "results/mcolon/20260215_roi_discovery_via_ot_feature_maps/scripts/output/phase0_qc_fix_rerun_alignedmasks"
)
DATA_CSV = (
    MORPHSEQ_ROOT
    / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
)
DATA_ROOT = MORPHSEQ_ROOT / "morphseq_playground"
OUT_DIR = Path(__file__).resolve().parent / "debug_results" / "phase0_samples_012_013"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_EMBRYO_ID = "20250512_B09_e01"
REFERENCE_FRAME_INDEX = 113
SAMPLE_IDS = ["sample_012", "sample_013", "sample_019"]


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 1.0


def load_row_mask(df: pd.DataFrame, embryo_id: str, frame_index: int):
    row = df[(df["embryo_id"] == embryo_id) & (df["frame_index"] == frame_index)]
    if len(row) == 0:
        raise ValueError(f"Missing row for embryo_id={embryo_id}, frame_index={frame_index}")
    row = row.iloc[0]
    mask = fmio.load_mask_from_rle_counts(
        rle_counts=row["mask_rle"],
        height_px=int(row["mask_height_px"]),
        width_px=int(row["mask_width_px"]),
    )
    yolk = fmio._load_build02_aux_mask(DATA_ROOT, row, mask.shape, keyword="yolk")
    um_per_px = float(fmio._compute_um_per_pixel(row))
    return row, mask.astype(np.uint8), None if yolk is None else yolk.astype(np.uint8), um_per_px


def draw_yolk_back(ax, meta: dict, color_yolk: str, color_back: str, prefix: str):
    yyx = meta.get("yolk_yx_final", None)
    byx = meta.get("back_yx_final", None)
    if yyx is not None and byx is not None:
        ax.plot(
            [yyx[1], byx[1]],
            [yyx[0], byx[0]],
            color="white",
            linewidth=1.6,
            alpha=0.8,
            zorder=9,
        )
    if yyx is not None:
        ax.scatter(
            yyx[1], yyx[0],
            s=72, marker="o", color=color_yolk, edgecolor="black", linewidth=0.5,
            label=f"{prefix} yolk", zorder=10,
        )
        ax.text(yyx[1] + 2, yyx[0] - 2, f"{prefix}:Y", color=color_yolk, fontsize=8, weight="bold")
    if byx is not None:
        ax.scatter(
            byx[1], byx[0],
            s=72, marker="s", color=color_back, edgecolor="black", linewidth=0.5,
            label=f"{prefix} back", zorder=10,
        )
        ax.text(byx[1] + 2, byx[0] - 2, f"{prefix}:B", color=color_back, fontsize=8, weight="bold")


def draw_centroid(ax, mask: np.ndarray, color: str, label: str):
    cy, cx = _center_of_mass_bool(mask.astype(bool))
    if np.isfinite(cy) and np.isfinite(cx):
        ax.scatter(cx, cy, s=60, marker="+", color=color, linewidth=1.4, zorder=10, label=label)
        ax.text(cx + 2, cy + 2, label, color=color, fontsize=7)


def _parse_args():
    parser = argparse.ArgumentParser(description="Debug canonical aligner behavior for selected Phase 0 samples")
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--reference-embryo-id", type=str, default=REFERENCE_EMBRYO_ID)
    parser.add_argument("--reference-frame-index", type=int, default=REFERENCE_FRAME_INDEX)
    parser.add_argument("--samples", type=str, default=",".join(SAMPLE_IDS))
    return parser.parse_args()


def _safe_yolk_back_angle(yolk, back) -> float:
    if yolk is None or back is None:
        return np.nan
    dy = float(back[0] - yolk[0])
    dx = float(back[1] - yolk[1])
    return float(np.degrees(np.arctan2(dy, dx)))


def _center_of_mass_bool(mask: np.ndarray):
    ys, xs = np.nonzero(mask.astype(bool))
    if len(ys) == 0:
        return (np.nan, np.nan)
    return (float(ys.mean()), float(xs.mean()))


def main():
    args = _parse_args()
    sample_ids = [s.strip() for s in args.samples.split(",") if s.strip()]
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent / "debug_results" / f"phase0_samples_{'_'.join([s.split('_')[-1] for s in sample_ids])}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load run artifacts
    zarr_path = args.run_dir / "feature_dataset" / "features.zarr"
    z = zarr.open(str(zarr_path), mode="r")
    mask_ref_saved = np.asarray(z["mask_ref"]).astype(np.uint8)
    target_saved = np.asarray(z["optional/target_masks_canonical"]).astype(np.uint8)
    metadata = pd.read_parquet(args.run_dir / "feature_dataset" / "metadata.parquet").reset_index(drop=True)
    align_debug_path = args.run_dir / "feature_dataset" / "alignment_debug.parquet"
    align_debug_df = None
    if align_debug_path.exists():
        align_debug_df = pd.read_parquet(align_debug_path)

    # Load CSV subset needed for mask/yolk resolution
    usecols = [
        "embryo_id",
        "frame_index",
        "experiment_date",
        "well",
        "time_int",
        "genotype",
        "predicted_stage_hpf",
        "mask_rle",
        "mask_height_px",
        "mask_width_px",
        "Height (um)",
        "Height (px)",
        "Width (um)",
        "Width (px)",
    ]
    df = pd.read_csv(DATA_CSV, usecols=usecols)

    # Reference (raw)
    _, ref_raw_mask, ref_raw_yolk, ref_um = load_row_mask(
        df, args.reference_embryo_id, args.reference_frame_index
    )
    if ref_raw_yolk is None or ref_raw_yolk.sum() == 0:
        raise RuntimeError("Reference yolk is missing; cannot debug yolk-based alignment.")

    # Same alignment settings used by OT pipeline
    uot_cfg = UOTConfig(
        epsilon=1e-4,
        marginal_relaxation=10.0,
        max_support_points=3000,
        use_canonical_grid=True,
        output_grid="canonical",
        canonical_grid_shape_hw=(256, 576),
        canonical_grid_um_per_pixel=10.0,
        canonical_grid_align_mode="yolk",
        canonical_grid_center_mode="joint_centering",
        downsample_factor=1,
        downsample_divisor=1,
        padding_px=16,
        align_mode="none",
    )
    canonical_cfg = CanonicalGridConfig(
        reference_um_per_pixel=10.0,
        grid_shape_hw=(256, 576),
        align_mode="yolk",
        downsample_factor=1,
        allow_flip=uot_cfg.canonical_grid_allow_flip,
        anchor_mode=uot_cfg.canonical_grid_anchor_mode,
        anchor_frac_yx=uot_cfg.canonical_grid_anchor_frac_yx,
        clipping_threshold=uot_cfg.canonical_grid_clipping_threshold,
        error_on_clip=uot_cfg.canonical_grid_error_on_clip,
    )
    aligner = CanonicalAligner.from_config(canonical_cfg)

    # Also get aligned reference/yolk from direct aligner (for plotting and overlap metrics)
    ref_aligned_mask, ref_aligned_yolk, ref_align_meta = aligner.align(
        mask=ref_raw_mask.astype(bool),
        yolk=ref_raw_yolk.astype(bool),
        original_um_per_px=ref_um,
        use_pca=True,
        use_yolk=True,
        return_debug=True,
    )

    rows = []
    for sample_id in sample_ids:
        m = metadata[metadata["sample_id"] == sample_id]
        if len(m) != 1:
            raise RuntimeError(f"Expected exactly one metadata row for {sample_id}, got {len(m)}")
        m = m.iloc[0]
        sample_index = int(m["sample_index"])
        embryo_id = str(m["embryo_id"])
        frame_index = int(m["frame_index"])
        genotype = str(m["genotype"])
        total_cost = float(m["total_cost_C"])

        _, tgt_raw_mask, tgt_raw_yolk, tgt_um = load_row_mask(df, embryo_id, frame_index)
        if tgt_raw_yolk is None or tgt_raw_yolk.sum() == 0:
            raise RuntimeError(f"Target yolk missing for {sample_id} ({embryo_id}, frame {frame_index})")
        raw_mask_b = tgt_raw_mask.astype(bool)
        raw_yolk_b = tgt_raw_yolk.astype(bool)
        raw_yolk_inside_ratio = float((raw_mask_b & raw_yolk_b).sum() / max(raw_yolk_b.sum(), 1))
        raw_yolk_iou = mask_iou(raw_mask_b, raw_yolk_b)

        # Exact OT preprocessing path
        src_frame = UOTFrame(
            embryo_mask=ref_raw_mask,
            meta={"um_per_pixel": ref_um, "yolk_mask": ref_raw_yolk},
        )
        tgt_frame = UOTFrame(
            embryo_mask=tgt_raw_mask,
            meta={"um_per_pixel": tgt_um, "yolk_mask": tgt_raw_yolk},
        )
        src_canon_pre, tgt_canon_pre, pre_meta = preprocess_pair_canonical(
            src_frame, tgt_frame, uot_cfg, canonical_cfg
        )

        # Direct aligner target/yolk for richer diagnostics
        tgt_aligned_mask, tgt_aligned_yolk, tgt_align_meta = aligner.align(
            mask=tgt_raw_mask.astype(bool),
            yolk=tgt_raw_yolk.astype(bool),
            original_um_per_px=tgt_um,
            use_pca=True,
            use_yolk=True,
            return_debug=True,
        )

        saved_target = target_saved[sample_index].astype(bool)
        src_canon_pre_b = src_canon_pre.astype(bool)
        tgt_canon_pre_b = tgt_canon_pre.astype(bool)
        tgt_aligned_b = tgt_aligned_mask.astype(bool)

        # Metrics
        iou_saved_vs_pre = mask_iou(saved_target, tgt_canon_pre_b)
        iou_pre_vs_direct = mask_iou(tgt_canon_pre_b, tgt_aligned_b)
        iou_tgt_vs_ref = mask_iou(tgt_canon_pre_b, src_canon_pre_b)
        iou_saved_ref_vs_pre_src = mask_iou(mask_ref_saved.astype(bool), src_canon_pre_b)
        yolk_inside_ratio = float(
            (tgt_aligned_yolk.astype(bool) & tgt_aligned_b).sum() / max(tgt_aligned_yolk.sum(), 1)
        )
        yolk_iou_vs_ref = mask_iou(tgt_aligned_yolk.astype(bool), ref_aligned_yolk.astype(bool))
        yolk_to_ref_overlap_ratio = float(
            (tgt_aligned_yolk.astype(bool) & ref_aligned_yolk.astype(bool)).sum() / max(ref_aligned_yolk.sum(), 1)
        )

        yolk_pt = tgt_align_meta.get("yolk_yx_final", None)
        back = tgt_align_meta.get("back_yx_final", None)
        yb_angle_deg = _safe_yolk_back_angle(yolk_pt, back)
        yolk_left_of_back = None
        back_below_yolk = None
        if yolk_pt is not None and back is not None:
            yolk_left_of_back = bool(yolk_pt[1] < back[1])
            back_below_yolk = bool(back[0] > yolk_pt[0])

        tgt_mask_com = _center_of_mass_bool(tgt_aligned_b)
        tgt_yolk_com = _center_of_mass_bool(tgt_aligned_yolk.astype(bool))
        yolk_left_of_body_centroid = (
            bool(tgt_yolk_com[1] < tgt_mask_com[1])
            if np.isfinite(tgt_yolk_com[1]) and np.isfinite(tgt_mask_com[1])
            else None
        )

        align_row = None
        if align_debug_df is not None and "sample_id" in align_debug_df.columns:
            mr = align_debug_df[align_debug_df["sample_id"] == sample_id]
            if len(mr) == 1:
                align_row = mr.iloc[0]

        rows.append(
            {
                "sample_id": sample_id,
                "sample_index": sample_index,
                "embryo_id": embryo_id,
                "frame_index": frame_index,
                "genotype": genotype,
                "total_cost_C": total_cost,
                "rotation_deg": float(tgt_align_meta.get("rotation_deg", np.nan)),
                "flip": bool(tgt_align_meta.get("flip", False)),
                "retained_ratio": float(tgt_align_meta.get("retained_ratio", np.nan)),
                "raw_yolk_inside_target_ratio": raw_yolk_inside_ratio,
                "raw_yolk_mask_iou": raw_yolk_iou,
                "iou_saved_target_vs_preprocess_target": iou_saved_vs_pre,
                "iou_preprocess_target_vs_direct_aligner_target": iou_pre_vs_direct,
                "iou_target_vs_reference_on_canonical": iou_tgt_vs_ref,
                "iou_saved_ref_vs_preprocess_ref": iou_saved_ref_vs_pre_src,
                "yolk_inside_target_ratio": yolk_inside_ratio,
                "yolk_iou_vs_reference_yolk": yolk_iou_vs_ref,
                "yolk_overlap_ratio_vs_reference_yolk": yolk_to_ref_overlap_ratio,
                "yolk_left_of_back": yolk_left_of_back,
                "back_below_yolk": back_below_yolk,
                "yolk_back_angle_deg": yb_angle_deg,
                "yolk_left_of_body_centroid": yolk_left_of_body_centroid,
                "target_area_px": int(tgt_canon_pre_b.sum()),
                "reference_area_px": int(src_canon_pre_b.sum()),
                "align_debug_total_cost_C": float(align_row["total_cost_C"]) if align_row is not None and "total_cost_C" in align_row else np.nan,
                "align_debug_tgt_rotation_deg": float(align_row["tgt_rotation_deg"]) if align_row is not None and "tgt_rotation_deg" in align_row else np.nan,
                "align_debug_tgt_flip": bool(align_row["tgt_flip"]) if align_row is not None and "tgt_flip" in align_row else None,
                "align_debug_overlap_iou_src_tgt": float(align_row["overlap_iou_src_tgt"]) if align_row is not None and "overlap_iou_src_tgt" in align_row else np.nan,
            }
        )

        # Visualization
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))

        # Raw target
        axes[0, 0].imshow(tgt_raw_mask, cmap="gray", origin="upper")
        axes[0, 0].imshow(np.where(tgt_raw_yolk.astype(bool), 1.0, np.nan), cmap="Blues", alpha=0.35, origin="upper")
        axes[0, 0].set_title(f"Raw Target + Raw Yolk\n{embryo_id} f{frame_index}")
        axes[0, 0].axis("off")

        # Canonical target
        axes[0, 1].imshow(tgt_aligned_mask, cmap="gray", origin="upper")
        axes[0, 1].imshow(np.where(tgt_aligned_yolk.astype(bool), 1.0, np.nan), cmap="Blues", alpha=0.35, origin="upper")
        axes[0, 1].contour(ref_aligned_yolk.astype(float), levels=[0.5], colors=["#66ff66"], linewidths=0.9, alpha=0.9)
        draw_centroid(axes[0, 1], tgt_aligned_yolk.astype(bool), "#4A90E2", "tgt_yolk_COM")
        draw_centroid(axes[0, 1], ref_aligned_yolk.astype(bool), "#66ff66", "ref_yolk_COM")
        draw_yolk_back(axes[0, 1], tgt_align_meta, "red", "cyan", "tgt")
        axes[0, 1].set_title(
            "Aligned Target (direct aligner)\n"
            f"rot={tgt_align_meta.get('rotation_deg', np.nan):.2f}, "
            f"flip={tgt_align_meta.get('flip', False)}"
        )
        axes[0, 1].legend(loc="lower right", fontsize=7, framealpha=0.75)
        axes[0, 1].axis("off")

        # Saved OT target
        axes[0, 2].imshow(saved_target, cmap="gray", origin="upper")
        axes[0, 2].set_title(f"Saved OT Target Mask\nIoU vs preprocess={iou_saved_vs_pre:.4f}")
        axes[0, 2].axis("off")

        # Reference aligned
        axes[1, 0].imshow(src_canon_pre_b, cmap="gray", origin="upper")
        axes[1, 0].imshow(np.where(ref_aligned_yolk.astype(bool), 1.0, np.nan), cmap="Greens", alpha=0.25, origin="upper")
        axes[1, 0].contour(tgt_aligned_yolk.astype(float), levels=[0.5], colors=["#4A90E2"], linewidths=0.9, alpha=0.9)
        draw_centroid(axes[1, 0], ref_aligned_yolk.astype(bool), "#66ff66", "ref_yolk_COM")
        draw_centroid(axes[1, 0], tgt_aligned_yolk.astype(bool), "#4A90E2", "tgt_yolk_COM")
        draw_yolk_back(axes[1, 0], ref_align_meta, "magenta", "yellow", "ref")
        axes[1, 0].set_title("Aligned Reference + Yolk (target yolk contour in blue)")
        axes[1, 0].legend(loc="lower right", fontsize=7, framealpha=0.75)
        axes[1, 0].axis("off")

        # Ref/target overlay mismatch
        target_only = tgt_canon_pre_b & (~src_canon_pre_b)
        ref_only = src_canon_pre_b & (~tgt_canon_pre_b)
        axes[1, 1].imshow(np.where(src_canon_pre_b, 1.0, np.nan), cmap="gray", alpha=0.25, origin="upper")
        axes[1, 1].imshow(np.where(target_only, 1.0, np.nan), cmap="Blues", alpha=0.60, origin="upper")
        axes[1, 1].imshow(np.where(ref_only, 1.0, np.nan), cmap="Purples", alpha=0.50, origin="upper")
        axes[1, 1].contour(src_canon_pre_b.astype(float), levels=[0.5], colors=["white"], linewidths=0.6, alpha=0.9)
        axes[1, 1].contour(tgt_canon_pre_b.astype(float), levels=[0.5], colors=["#4A90E2"], linewidths=0.9, alpha=0.9)
        axes[1, 1].set_title(
            "Canonical Overlay\n"
            f"IoU(tgt,ref)={iou_tgt_vs_ref:.4f}; yolk_in={yolk_inside_ratio:.3f}"
        )
        axes[1, 1].axis("off")

        # Text summary
        axes[1, 2].axis("off")
        debug_iou_line = (
            f"align_debug_iou_src_tgt: {float(align_row['overlap_iou_src_tgt']):.6f}"
            if align_row is not None and "overlap_iou_src_tgt" in align_row
            else "align_debug_iou_src_tgt: n/a"
        )
        text = (
            f"sample_id: {sample_id}\n"
            f"genotype: {genotype}\n"
            f"total_cost_C: {total_cost:.4f}\n"
            f"rotation_deg: {tgt_align_meta.get('rotation_deg', np.nan):.4f}\n"
            f"flip: {tgt_align_meta.get('flip', False)}\n"
            f"retained_ratio: {tgt_align_meta.get('retained_ratio', np.nan):.4f}\n"
            f"raw_yolk_inside_target_ratio: {raw_yolk_inside_ratio:.4f}\n"
            f"raw_yolk_mask_iou: {raw_yolk_iou:.4f}\n"
            f"yolk_left_of_back: {yolk_left_of_back}\n"
            f"back_below_yolk: {back_below_yolk}\n"
            f"yolk_back_angle_deg: {yb_angle_deg:.2f}\n"
            f"yolk_left_of_body_centroid: {yolk_left_of_body_centroid}\n"
            f"yolk_inside_target_ratio: {yolk_inside_ratio:.4f}\n"
            f"yolk_iou_vs_ref: {yolk_iou_vs_ref:.4f}\n"
            f"yolk_overlap_ratio_vs_ref: {yolk_to_ref_overlap_ratio:.4f}\n"
            f"saved_vs_preprocess_iou: {iou_saved_vs_pre:.6f}\n"
            f"preprocess_vs_direct_iou: {iou_pre_vs_direct:.6f}\n"
            f"saved_ref_vs_pre_src_iou: {iou_saved_ref_vs_pre_src:.6f}\n"
            f"{debug_iou_line}"
        )
        axes[1, 2].text(0.02, 0.98, text, va="top", ha="left", family="monospace", fontsize=9)

        fig.suptitle(
            f"Canonical Alignment Debug - {sample_id} ({embryo_id}, frame {frame_index})",
            fontsize=13,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(out_dir / f"{sample_id}_alignment_debug.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    metrics_df = pd.DataFrame(rows)
    metrics_path = out_dir / f"sample_{'_'.join([s.split('_')[-1] for s in sample_ids])}_alignment_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Wrote: {metrics_path}")
    for sample_id in sample_ids:
        print(f"Wrote: {out_dir / f'{sample_id}_alignment_debug.png'}")
    print("\\nSummary:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
