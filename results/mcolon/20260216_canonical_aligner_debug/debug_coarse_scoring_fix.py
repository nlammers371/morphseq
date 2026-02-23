#!/usr/bin/env python3
"""
Debug script: Two-Stage Canonical Alignment + Reference-Centric Registration.

Validates the Stage 1 + Stage 2 split for the coarse-scoring bug fix.

For reference 20250512_B09_e01 f113 and problem embryos
  sample_001 (20251205_F11_e01) and sample_015 (20251017_combined_H07_e01):

  1. Stage 1 → tgt_can_pre  (embryo_canonical_alignment)
  2. Stage 2 → tgt_can_post + reg_meta  (embryo_src_tgt_register)
  3. Plot IoU curve [-180°, 180°] with gating thresholds marked
  4. 2-panel overlay: pre vs post against src contour + yolk marker
  5. Print: iou_before, iou_after, applied, angle_deg, best_iou,
            best_angle_deg, hit_boundary, tgt_pivot_source
  6. Assert: if applied → iou_after >= iou_before + 0.02
  7. If rotate_then_pivot_translate + applied →
         assert tgt_pivot_yx + translate_dyx ≈ src_pivot_yx (within 1px)

Output: debug_results/coarse_scoring_fix/

Usage:
    PYTHONPATH=src \\
      /net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python \\
      results/mcolon/20260216_canonical_aligner_debug/debug_coarse_scoring_fix.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

MORPHSEQ_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MORPHSEQ_ROOT))
sys.path.insert(0, str(MORPHSEQ_ROOT / "src"))

from analyze.optimal_transport_morphometrics.uot_masks import frame_mask_io as fmio
from analyze.optimal_transport_morphometrics.uot_masks.uot_grid import (
    CanonicalAligner,
    CanonicalGridConfig,
    embryo_src_tgt_register,
    _apply_pivot_rotation,
    _iou,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_CSV = (
    MORPHSEQ_ROOT
    / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
)
DATA_ROOT = MORPHSEQ_ROOT / "morphseq_playground"

OUT_DIR = Path(__file__).resolve().parent / "debug_results" / "coarse_scoring_fix"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Reference embryo
REFERENCE_EMBRYO_ID = "20250512_B09_e01"
REFERENCE_FRAME = 113

# Problem embryos (sample_001, sample_015)
TARGETS = [
    ("20251205_F11_e01",           50,  "sample_001"),
    ("20251017_combined_H07_e01",  31,  "sample_015"),
]

CANONICAL_CFG = CanonicalGridConfig(
    reference_um_per_pixel=10.0,
    grid_shape_hw=(256, 576),
    align_mode="yolk",
    downsample_factor=1,
)

# Stage 2 gating thresholds
MIN_IOU_ABSOLUTE = 0.25
MIN_IOU_IMPROVEMENT = 0.02
ANGLE_STEP_DEG = 1.0

USECOLS = [
    "embryo_id", "frame_index", "genotype", "predicted_stage_hpf",
    "mask_rle", "mask_height_px", "mask_width_px",
    "Height (um)", "Height (px)", "Width (um)", "Width (px)",
    "experiment_date", "well", "time_int",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_embryo(df: pd.DataFrame, embryo_id: str, frame_index: int):
    """Load mask, yolk, and um/px for a given embryo+frame."""
    row = df[(df["embryo_id"] == embryo_id) & (df["frame_index"] == frame_index)]
    if len(row) == 0:
        raise ValueError(f"No row for embryo_id={embryo_id!r}, frame_index={frame_index}")
    row = row.iloc[0]
    mask = fmio.load_mask_from_rle_counts(
        row["mask_rle"], int(row["mask_height_px"]), int(row["mask_width_px"])
    )
    yolk = fmio._load_build02_aux_mask(DATA_ROOT, row, mask.shape, keyword="yolk")
    um_per_px = float(fmio._compute_um_per_pixel(row))
    return mask.astype(np.uint8), yolk, um_per_px


def compute_iou_sweep(
    tgt_mask: np.ndarray,
    src_mask: np.ndarray,
    pivot_yx: tuple,
    angle_step_deg: float = 1.0,
) -> tuple:
    """Return (angles, ious) arrays for a full ±180° sweep."""
    angles = np.arange(-180.0, 180.0 + angle_step_deg, angle_step_deg)
    ious = []
    for a in angles:
        rotated = _apply_pivot_rotation(tgt_mask, pivot_yx, float(a))
        ious.append(_iou(rotated, src_mask))
    return angles, np.array(ious)


def overlay_panel(ax, tgt_mask, tgt_yolk, src_mask, title, iou_val, yolk_com_yx=None):
    """Draw tgt_mask (gray) with src contour (red) and optional yolk marker."""
    ax.imshow(tgt_mask, cmap="gray", origin="upper", vmin=0, vmax=1)
    if tgt_yolk is not None and tgt_yolk.sum() > 0:
        ax.imshow(
            np.where(tgt_yolk.astype(bool), 1.0, np.nan),
            cmap="Blues", alpha=0.5, origin="upper",
        )
    ax.contour(src_mask > 0.5, levels=[0.5], colors=["red"], linewidths=1.2)
    if yolk_com_yx is not None:
        ax.plot(yolk_com_yx[1], yolk_com_yx[0], "o", color="cyan", ms=6, zorder=10, label="tgt yolk COM")
    ax.set_title(f"{title}\nIoU = {iou_val:.4f}", fontsize=10)
    ax.axis("off")
    patches = [
        mpatches.Patch(color="gray", label="Target (canonical)"),
        mpatches.Patch(edgecolor="red", facecolor="none", label="Source contour"),
    ]
    if tgt_yolk is not None:
        patches.append(mpatches.Patch(color="steelblue", alpha=0.6, label="Target yolk"))
    ax.legend(handles=patches, loc="lower right", fontsize=7)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading CSV …")
    df = pd.read_csv(DATA_CSV, usecols=USECOLS)

    aligner = CanonicalAligner.from_config(CANONICAL_CFG)

    # --- Stage 1: Reference ---
    print(f"\nStage 1 — Reference: {REFERENCE_EMBRYO_ID} f{REFERENCE_FRAME}")
    ref_raw, ref_yolk_raw, ref_um = load_embryo(df, REFERENCE_EMBRYO_ID, REFERENCE_FRAME)
    print(f"  um/px={ref_um:.4f}  yolk_px={ref_yolk_raw.sum() if ref_yolk_raw is not None else 0}")

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        src_can, src_yolk_can, src_meta = aligner.embryo_canonical_alignment(
            ref_raw.astype(bool),
            ref_um,
            yolk=ref_yolk_raw.astype(bool) if ref_yolk_raw is not None else None,
            use_pca=True,
        )
    src_yolk_com_yx = src_meta.get("yolk_com_yx")
    print(f"  Canonical: {src_can.sum()} px, yolk_com_yx={src_yolk_com_yx}")

    all_pass = True

    for embryo_id, frame_index, label in TARGETS:
        print(f"\n{'='*60}")
        print(f"Target: {embryo_id} f{frame_index} — {label}")

        tgt_raw, tgt_yolk_raw, tgt_um = load_embryo(df, embryo_id, frame_index)
        print(f"  um/px={tgt_um:.4f}  yolk_px={tgt_yolk_raw.sum() if tgt_yolk_raw is not None else 0}")

        # --- Stage 1: Target canonical alignment ---
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            tgt_can_pre, tgt_yolk_can, tgt_meta = aligner.embryo_canonical_alignment(
                tgt_raw.astype(bool),
                tgt_um,
                yolk=tgt_yolk_raw.astype(bool) if tgt_yolk_raw is not None else None,
                use_pca=True,
            )
        tgt_yolk_com_yx = tgt_meta.get("yolk_com_yx")
        print(f"  Stage 1 done: {tgt_can_pre.sum()} px, yolk_com_yx={tgt_yolk_com_yx}")

        # --- IoU sweep for plotting ---
        pivot_for_sweep = (
            tgt_yolk_com_yx if tgt_yolk_com_yx is not None
            else (tgt_can_pre.shape[0] / 2, tgt_can_pre.shape[1] / 2)
        )
        print(f"  Computing IoU sweep (pivot={pivot_for_sweep}) …")
        angles, ious = compute_iou_sweep(
            tgt_can_pre, src_can, pivot_for_sweep, ANGLE_STEP_DEG
        )
        iou_at_0 = float(ious[np.argmin(np.abs(angles))])

        # --- Stage 2: Registration ---
        tgt_can_post, reg_meta = embryo_src_tgt_register(
            src_can, tgt_can_pre,
            src_yolk_com_yx=src_yolk_com_yx,
            tgt_yolk_com_yx=tgt_yolk_com_yx,
            mode="rotate_only",
            angle_step_deg=ANGLE_STEP_DEG,
            min_iou_absolute=MIN_IOU_ABSOLUTE,
            min_iou_improvement=MIN_IOU_IMPROVEMENT,
        )

        # Print results
        print(f"\n  --- Stage 2 reg_meta ---")
        for k in [
            "applied", "mode", "best_angle_deg", "best_iou", "hit_boundary",
            "angle_deg", "iou_before", "iou_after", "tgt_pivot_yx", "tgt_pivot_source",
        ]:
            print(f"    {k}: {reg_meta.get(k)}")

        # --- Assertions ---
        iou_before = reg_meta["iou_before"]
        iou_after = reg_meta["iou_after"]
        applied = reg_meta["applied"]

        if applied:
            ok = iou_after >= iou_before + MIN_IOU_IMPROVEMENT - 1e-6
            status = "PASS" if ok else "FAIL"
            print(f"\n  {status}: applied=True → iou_after({iou_after:.4f}) >= iou_before({iou_before:.4f}) + {MIN_IOU_IMPROVEMENT}")
            if not ok:
                all_pass = False
        else:
            print(f"\n  NOTE: applied=False (iou_before={iou_before:.4f}, best_iou={reg_meta['best_iou']:.4f})")

        if reg_meta.get("mode") == "rotate_then_pivot_translate" and applied:
            tpy, tpx = reg_meta["tgt_pivot_yx"]
            spy, spx = reg_meta["src_pivot_yx"]
            dy, dx = reg_meta["translate_dyx"]
            err = np.sqrt((tpy + dy - spy) ** 2 + (tpx + dx - spx) ** 2)
            ok2 = err < 1.0
            status2 = "PASS" if ok2 else "FAIL"
            print(f"  {status2}: tgt_pivot + translate_dyx ≈ src_pivot (err={err:.3f}px)")
            if not ok2:
                all_pass = False

        # --- Plot 1: IoU sweep curve ---
        fig_curve, ax_curve = plt.subplots(figsize=(10, 4))
        ax_curve.plot(angles, ious, color="royalblue", linewidth=1.2)
        ax_curve.axhline(MIN_IOU_ABSOLUTE, color="orange", linestyle="--",
                         linewidth=1.0, label=f"min_iou_absolute={MIN_IOU_ABSOLUTE}")
        ax_curve.axhline(iou_at_0 + MIN_IOU_IMPROVEMENT, color="green", linestyle="--",
                         linewidth=1.0, label=f"iou_at_0° + {MIN_IOU_IMPROVEMENT} = {iou_at_0 + MIN_IOU_IMPROVEMENT:.3f}")
        ax_curve.axvline(reg_meta["best_angle_deg"], color="red", linestyle=":",
                         linewidth=1.0, label=f"best_angle={reg_meta['best_angle_deg']:.1f}°")
        ax_curve.set_xlabel("Rotation angle (°)")
        ax_curve.set_ylabel("IoU with src")
        ax_curve.set_title(
            f"{label} — {embryo_id} f{frame_index}\n"
            f"IoU sweep about tgt_pivot ({reg_meta['tgt_pivot_source']})",
            fontsize=11,
        )
        ax_curve.legend(fontsize=9)
        ax_curve.grid(True, alpha=0.3)
        fig_curve.tight_layout()
        safe_id = embryo_id.replace("/", "_")
        curve_path = OUT_DIR / f"{safe_id}_f{frame_index}_iou_curve.png"
        fig_curve.savefig(curve_path, dpi=150, bbox_inches="tight")
        plt.close(fig_curve)
        print(f"  Saved IoU curve → {curve_path}")

        # --- Plot 2: 2-panel overlay (pre vs post) ---
        fig_ov, axes = plt.subplots(1, 2, figsize=(16, 5))

        overlay_panel(
            axes[0],
            tgt_can_pre,
            tgt_yolk_can,
            src_can,
            title="Stage 1 only (pre-registration)",
            iou_val=iou_before,
            yolk_com_yx=tgt_yolk_com_yx,
        )

        # Propagate yolk sidecar if registration was applied
        tgt_yolk_post = None
        if tgt_yolk_can is not None and applied:
            tgt_yolk_post = _apply_pivot_rotation(
                tgt_yolk_can, reg_meta["tgt_pivot_yx"], reg_meta["angle_deg"]
            )
            tgt_yolk_post = (tgt_yolk_post > 0.5).astype(np.uint8)
        elif tgt_yolk_can is not None:
            tgt_yolk_post = tgt_yolk_can

        overlay_panel(
            axes[1],
            tgt_can_post,
            tgt_yolk_post,
            src_can,
            title=(
                f"Stage 1 + Stage 2 (post-registration)\n"
                f"applied={applied}, angle={reg_meta['angle_deg']:+.1f}°"
            ),
            iou_val=iou_after,
            yolk_com_yx=tgt_yolk_com_yx,
        )

        fig_ov.suptitle(
            f"{label} — {embryo_id} f{frame_index}\n"
            f"ΔIoU = {iou_after - iou_before:+.4f}  |  "
            f"tgt_pivot_source={reg_meta['tgt_pivot_source']}  |  "
            f"hit_boundary={reg_meta['hit_boundary']}",
            fontsize=12, fontweight="bold",
        )
        fig_ov.tight_layout(rect=[0, 0, 1, 0.9])
        ov_path = OUT_DIR / f"{safe_id}_f{frame_index}_overlay.png"
        fig_ov.savefig(ov_path, dpi=150, bbox_inches="tight")
        plt.close(fig_ov)
        print(f"  Saved overlay    → {ov_path}")

    print(f"\n{'='*60}")
    if all_pass:
        print("All assertions PASSED.")
    else:
        print("Some assertions FAILED — see above.")
    print(f"Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
