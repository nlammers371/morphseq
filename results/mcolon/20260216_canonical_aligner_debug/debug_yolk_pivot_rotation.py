#!/usr/bin/env python3
"""
Yolk-pivot fine rotation debug script.

Loads a reference embryo (20250512_B09_e01, frame 113) and 3 problem embryos,
then shows IoU before and after the yolk-pivot fine rotation sweep.

For each problem embryo:
  - Left panel: aligned target (pre-pivot) overlaid on reference mask contour + IoU
  - Right panel: aligned target (post-pivot) overlaid on reference mask contour + IoU + pivot angle

Usage:
    PYTHONPATH=src /net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python \
      results/mcolon/20260216_canonical_aligner_debug/debug_yolk_pivot_rotation.py
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
from analyze.utils.coord.grids.canonical import CanonicalAligner, CanonicalGridConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_CSV = (
    MORPHSEQ_ROOT
    / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
)
DATA_ROOT = MORPHSEQ_ROOT / "morphseq_playground"

OUT_DIR = Path(__file__).resolve().parent / "debug_results" / "yolk_pivot_rotation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Reference embryo (canonical "ground truth" for this experiment set)
REFERENCE_EMBRYO_ID = "20250512_B09_e01"
REFERENCE_FRAME = 113

# Problem embryos to evaluate
TARGETS = [
    ("20251106_E09_e01", 20,  "sample_012 (O13 / cep290_homozygous)"),
    ("20251113_C04_e01", 14,  "sample_013 (cep290_homozygous)"),
    ("20251205_F06_e01", 71,  "sample_019 (O19 / cep290_homozygous)"),
]

# Canonical grid settings matching the pipeline
CANONICAL_CFG = CanonicalGridConfig(
    reference_um_per_pixel=10.0,
    grid_shape_hw=(256, 576),
    align_mode="yolk",
    downsample_factor=1,
)

# Sweep parameters
ANGLE_RANGE_DEG = 15.0
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
    row = df[(df["embryo_id"] == embryo_id) & (df["frame_index"] == frame_index)]
    if len(row) == 0:
        raise ValueError(f"No row found for embryo_id={embryo_id}, frame_index={frame_index}")
    row = row.iloc[0]
    mask = fmio.load_mask_from_rle_counts(
        row["mask_rle"], int(row["mask_height_px"]), int(row["mask_width_px"])
    )
    yolk = fmio._load_build02_aux_mask(DATA_ROOT, row, mask.shape, keyword="yolk")
    um_per_px = float(fmio._compute_um_per_pixel(row))
    return mask.astype(np.uint8), yolk, um_per_px, row


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0.5
    b = mask_b > 0.5
    intersection = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return intersection / (union + 1e-6)


def overlay_panel(
    ax,
    target_mask: np.ndarray,
    target_yolk: np.ndarray | None,
    reference_mask: np.ndarray,
    title: str,
    iou: float,
):
    """Draw target mask (gray) with reference mask contour (red) and yolk overlay (blue)."""
    ax.imshow(target_mask, cmap="gray", origin="upper", vmin=0, vmax=1)
    if target_yolk is not None and target_yolk.sum() > 0:
        ax.imshow(
            np.where(target_yolk.astype(bool), 1.0, np.nan),
            cmap="Blues", alpha=0.5, origin="upper",
        )
    # Reference mask contour in red
    ax.contour(reference_mask > 0.5, levels=[0.5], colors=["red"], linewidths=1.2)
    ax.set_title(f"{title}\nIoU = {iou:.4f}", fontsize=10)
    ax.axis("off")
    # Legend
    target_patch = mpatches.Patch(color="gray", label="Target (aligned)")
    ref_patch = mpatches.Patch(edgecolor="red", facecolor="none", label="Reference contour")
    yolk_patch = mpatches.Patch(color="steelblue", alpha=0.6, label="Target yolk")
    ax.legend(handles=[target_patch, ref_patch, yolk_patch], loc="lower right", fontsize=7)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading CSV from {DATA_CSV}")
    df = pd.read_csv(DATA_CSV, usecols=USECOLS)

    aligner = CanonicalAligner.from_config(CANONICAL_CFG)

    # Load and align reference embryo
    print(f"\nLoading reference: {REFERENCE_EMBRYO_ID} f{REFERENCE_FRAME}")
    ref_raw, ref_yolk, ref_um_per_px, ref_row = load_embryo(df, REFERENCE_EMBRYO_ID, REFERENCE_FRAME)
    print(f"  um/px: {ref_um_per_px:.4f}, yolk_px: {ref_yolk.sum() if ref_yolk is not None else 0}")

    ref_canonical, ref_yolk_aligned, ref_meta = aligner.align(
        mask=ref_raw.astype(bool),
        yolk=ref_yolk.astype(bool) if ref_yolk is not None else None,
        original_um_per_px=ref_um_per_px,
        use_pca=True,
        use_yolk=True,
    )
    print(f"  Reference aligned: {ref_canonical.sum()} pixels, "
          f"rot={ref_meta['rotation_deg']:.1f}°, flip={ref_meta['flip']}")

    # Process each target embryo
    all_pass = True
    for embryo_id, frame_index, label in TARGETS:
        print(f"\n{'='*60}")
        print(f"Processing: {embryo_id} f{frame_index} — {label}")

        raw_mask, raw_yolk, um_per_px, row = load_embryo(df, embryo_id, frame_index)
        print(f"  um/px: {um_per_px:.4f}, yolk_px: {raw_yolk.sum() if raw_yolk is not None else 0}")

        # Align WITHOUT reference (baseline = pre-pivot)
        aligned_pre, yolk_pre, meta_pre = aligner.align(
            mask=raw_mask.astype(bool),
            yolk=raw_yolk.astype(bool) if raw_yolk is not None else None,
            original_um_per_px=um_per_px,
            use_pca=True,
            use_yolk=True,
            reference_mask=None,  # no pivot
        )
        iou_pre = compute_iou(aligned_pre, ref_canonical)

        # Apply yolk-pivot sweep manually (same params as aligner)
        aligned_post, yolk_post, pivot_angle = aligner._yolk_pivot_rotate(
            aligned_pre,
            yolk_pre,
            ref_canonical,
            angle_range_deg=ANGLE_RANGE_DEG,
            angle_step_deg=ANGLE_STEP_DEG,
        )
        iou_post = compute_iou(aligned_post, ref_canonical)

        print(f"  IoU before pivot: {iou_pre:.4f}")
        print(f"  IoU after pivot:  {iou_post:.4f}  (pivot angle: {pivot_angle:+.1f}°)")

        # Verification
        if iou_post < iou_pre - 1e-6:
            print(f"  FAIL: post-pivot IoU ({iou_post:.4f}) < pre-pivot IoU ({iou_pre:.4f})")
            all_pass = False
        else:
            print(f"  PASS: post-pivot IoU >= pre-pivot IoU")

        # --- Plot ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        overlay_panel(
            axes[0],
            aligned_pre,
            yolk_pre,
            ref_canonical,
            title="Pre-pivot (coarse alignment only)",
            iou=iou_pre,
        )

        overlay_panel(
            axes[1],
            aligned_post,
            yolk_post,
            ref_canonical,
            title=f"Post-pivot (sweep ±{ANGLE_RANGE_DEG:.0f}°, step {ANGLE_STEP_DEG:.0f}°)\nbest angle: {pivot_angle:+.1f}°",
            iou=iou_post,
        )

        fig.suptitle(
            f"{label}\n{embryo_id}  f{frame_index}  |  "
            f"ΔIoU = {iou_post - iou_pre:+.4f}",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.93])

        safe_id = embryo_id.replace("/", "_")
        out_path = OUT_DIR / f"{safe_id}_f{frame_index}_yolk_pivot.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"  Wrote: {out_path}")

    print(f"\n{'='*60}")
    if all_pass:
        print("All verifications PASSED: post-pivot IoU >= pre-pivot IoU for all embryos.")
    else:
        print("Some verifications FAILED — see above.")
    print(f"Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
