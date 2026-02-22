#!/usr/bin/env python3
"""
Diagnose back-detection failure for 20251113_C04_e01 frame 14.

Shows all 4 candidate orientations and their scores, plus yolk/back positions.
"""

from __future__ import annotations

import sys
from pathlib import Path

MORPHSEQ_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MORPHSEQ_ROOT))
sys.path.insert(0, str(MORPHSEQ_ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import cv2

from analyze.optimal_transport_morphometrics.uot_masks import frame_mask_io as fmio
from analyze.utils.coord.grids.canonical import CanonicalAligner, CanonicalGridConfig
from analyze.utils.masks.qc import qc_mask

OUTPUT_DIR = Path(__file__).parent / "debug_results" / "c04_alignment_candidates"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_CSV = MORPHSEQ_ROOT / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
DATA_ROOT = MORPHSEQ_ROOT / "morphseq_playground"

CANONICAL_GRID_HW = (256, 576)
CANONICAL_UM_PER_PX = 10.0

TARGET_EMBRYO_ID = "20251113_C04_e01"
TARGET_FRAME_INDEX = 14
REF_EMBRYO_ID = "20251113_A05_e01"
REF_FRAME_INDEX = 95


def load_mask_yolk(row: pd.Series):
    mask = fmio.load_mask_from_rle_counts(
        rle_counts=row["mask_rle"],
        height_px=int(row["mask_height_px"]),
        width_px=int(row["mask_width_px"]),
    )
    yolk = fmio._load_build02_aux_mask(DATA_ROOT, row, mask.shape, keyword="yolk")
    um_per_px = fmio._compute_um_per_pixel(row)
    return mask, yolk, float(um_per_px)


def show_all_candidates(mask, yolk, um_per_px, aligner, embryo_id, frame_idx, save_path):
    """Show all 4 candidate orientations and their scores."""
    scale = um_per_px / aligner.target_res
    angle_deg, centroid_xy, pca_used = aligner._pca_angle_deg(mask)
    cx, cy = centroid_xy
    target_angle = 0.0 if aligner.is_landscape else 90.0
    rotation_needed = angle_deg - target_angle

    print(f"\n=== {embryo_id} frame {frame_idx} ===")
    print(f"  Raw shape: {mask.shape}, yolk pixels: {yolk.sum() if yolk is not None else 0}")
    print(f"  PCA angle: {angle_deg:.1f}°, rotation_needed: {rotation_needed:.1f}°, scale: {scale:.4f}")

    candidates = []
    rot_options = [0, 180]
    flip_options = [False, True]

    fig, axes = plt.subplots(2, 4, figsize=(20, 12))

    for idx, (rot_add, do_flip) in enumerate([(r, f) for r in rot_options for f in flip_options]):
        M = cv2.getRotationMatrix2D((cx, cy), rotation_needed + rot_add, scale)
        M[0, 2] += (aligner.W / 2) - cx
        M[1, 2] += (aligner.H / 2) - cy
        mask_w = aligner._warp(mask, M)
        yolk_w = aligner._warp(yolk, M) if yolk is not None else None
        if do_flip:
            mask_w = cv2.flip(mask_w, 1)
            if yolk_w is not None:
                yolk_w = cv2.flip(yolk_w, 1)

        # Compute features
        yolk_feature_mask = yolk_w if (yolk_w is not None and yolk_w.sum() > 0) else mask_w
        yolk_yx = aligner._center_of_mass(yolk_feature_mask)
        back_yx = aligner._compute_back_direction(mask_w, yolk_mask=yolk_w)

        # Use the SAME scorer as canonical.py _coarse_candidate_select:
        # when yolk present → score = -yolk_yx[1]  (wants yolk on LEFT = small x)
        if yolk_w is not None and yolk_w.sum() > 0:
            score = -yolk_yx[1]
        else:
            yolk_cost = yolk_yx[1] + yolk_yx[0]
            back_score = back_yx[1] + back_yx[0]
            score = (aligner.back_weight * back_score) - (aligner.yolk_weight * yolk_cost)

        candidates.append((score, rot_add, do_flip, yolk_yx, back_yx, mask_w, yolk_w))

        row_idx = 0 if idx < 2 else 1
        col_idx = (idx % 2) * 2

        ax_mask = axes[row_idx][col_idx]
        ax_overlay = axes[row_idx][col_idx + 1]

        # Mask with markers
        ax_mask.imshow(mask_w, cmap="gray", interpolation="nearest")
        ax_mask.plot(yolk_yx[1], yolk_yx[0], 'r+', markersize=15, markeredgewidth=2, label=f"yolk ({yolk_yx[1]:.0f},{yolk_yx[0]:.0f})")
        ax_mask.plot(back_yx[1], back_yx[0], 'b*', markersize=15, markeredgewidth=2, label=f"back ({back_yx[1]:.0f},{back_yx[0]:.0f})")
        ax_mask.set_title(
            f"rot+{rot_add}° flip={do_flip}\nscore={score:.1f}  yolk_x={yolk_yx[1]:.0f}",
            fontsize=7
        )
        ax_mask.legend(fontsize=6)
        ax_mask.axis("off")

        # Yolk overlay
        overlay = np.zeros((*mask_w.shape, 3), dtype=np.float32)
        overlay[mask_w > 0.5] = [0.7, 0.7, 0.7]
        if yolk_w is not None and yolk_w.sum() > 0:
            overlay[yolk_w > 0.5] = [1.0, 0.5, 0.0]  # orange = yolk
        ax_overlay.imshow(overlay, interpolation="nearest")
        ax_overlay.plot(yolk_yx[1], yolk_yx[0], 'r+', markersize=15, markeredgewidth=2)
        ax_overlay.plot(back_yx[1], back_yx[0], 'b*', markersize=15, markeredgewidth=2)
        ax_overlay.set_title(f"Yolk overlay (orange)\ngray=embryo", fontsize=7)
        ax_overlay.axis("off")

        print(f"  [{idx}] rot+{rot_add}° flip={do_flip}: score={score:.2f}  "
              f"yolk_yx=({yolk_yx[0]:.1f},{yolk_yx[1]:.1f}) back_yx=({back_yx[0]:.1f},{back_yx[1]:.1f})")

    best = max(candidates, key=lambda x: x[0])
    print(f"  WINNER: rot+{best[1]}° flip={best[2]} score={best[0]:.2f}")

    fig.suptitle(
        f"{embryo_id} frame {frame_idx} — All 4 orientation candidates\n"
        f"red+ = yolk COM, blue* = back point  |  "
        f"Score = -yolk_x  [wants: yolk on LEFT; matches canonical.py _coarse_candidate_select]",
        fontsize=9
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")
    return best


def main():
    print("=" * 70)
    print(f"BACK-DETECTION FAILURE DIAGNOSIS: {TARGET_EMBRYO_ID} frame {TARGET_FRAME_INDEX}")
    print("=" * 70)

    df = pd.read_csv(DATA_CSV, low_memory=False)

    cfg = CanonicalGridConfig(
        reference_um_per_pixel=CANONICAL_UM_PER_PX,
        grid_shape_hw=CANONICAL_GRID_HW,
        align_mode="yolk",
    )
    aligner = CanonicalAligner.from_config(cfg)

    # Load reference for comparison
    ref_row = df[(df["embryo_id"] == REF_EMBRYO_ID) & (df["frame_index"] == REF_FRAME_INDEX)].iloc[0]
    ref_raw, ref_yolk, ref_um = load_mask_yolk(ref_row)
    ref_qc, _ = qc_mask(ref_raw.astype(bool))

    print(f"\n--- Reference: {REF_EMBRYO_ID} frame {REF_FRAME_INDEX} ---")
    best_ref = show_all_candidates(
        ref_qc, ref_yolk.astype(bool) if ref_yolk is not None and ref_yolk.sum() > 0 else None,
        ref_um, aligner, REF_EMBRYO_ID, REF_FRAME_INDEX,
        OUTPUT_DIR / "ref_A05_e01_candidates.png"
    )

    # Load the failing target
    tgt_row = df[(df["embryo_id"] == TARGET_EMBRYO_ID) & (df["frame_index"] == TARGET_FRAME_INDEX)].iloc[0]
    tgt_raw, tgt_yolk, tgt_um = load_mask_yolk(tgt_row)
    tgt_qc, _ = qc_mask(tgt_raw.astype(bool))

    print(f"\n--- Target: {TARGET_EMBRYO_ID} frame {TARGET_FRAME_INDEX} ---")
    best_tgt = show_all_candidates(
        tgt_qc, tgt_yolk.astype(bool) if tgt_yolk is not None and tgt_yolk.sum() > 0 else None,
        tgt_um, aligner, TARGET_EMBRYO_ID, TARGET_FRAME_INDEX,
        OUTPUT_DIR / f"tgt_{TARGET_EMBRYO_ID.replace('/', '_')}_candidates.png"
    )

    # Also plot raw masks with yolk overlay for both
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    def plot_raw_with_yolk(ax0, ax1, mask, yolk, label):
        ax0.imshow(mask, cmap="gray", interpolation="nearest")
        ax0.set_title(f"{label}\nRaw mask {mask.shape}", fontsize=8)
        ax0.axis("off")

        overlay = np.zeros((*mask.shape, 3), dtype=np.float32)
        overlay[mask > 0.5] = [0.7, 0.7, 0.7]
        if yolk is not None and yolk.sum() > 0:
            overlay[yolk > 0.5] = [1.0, 0.5, 0.0]
            yc, xc = np.where(yolk > 0.5)
            ax1.text(0.5, 0.02, f"yolk pixels: {yolk.sum()}", transform=ax1.transAxes,
                     ha='center', va='bottom', fontsize=7, color='orange')
        ax1.imshow(overlay, interpolation="nearest")
        ax1.set_title(f"{label}\nYolk=orange overlay", fontsize=8)
        ax1.axis("off")

    plot_raw_with_yolk(axes[0][0], axes[0][1], ref_qc, ref_yolk, f"REF {REF_EMBRYO_ID}")
    plot_raw_with_yolk(axes[1][0], axes[1][1], tgt_qc, tgt_yolk, f"TGT {TARGET_EMBRYO_ID}")

    fig.suptitle("Raw masks with yolk overlay", fontsize=10)
    plt.tight_layout()
    raw_path = OUTPUT_DIR / "raw_masks_yolk_overlay.png"
    plt.savefig(raw_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved raw overlay: {raw_path.name}")

    print(f"\nDone. Results in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
