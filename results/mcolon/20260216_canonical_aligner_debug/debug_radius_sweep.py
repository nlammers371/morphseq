#!/usr/bin/env python3
"""
Sweep back_sample_radius_k and show how the back vector + orientation changes.

For each embryo, for each radius_k value:
  - Show the canonical mask with the sampling disk + back point marked
  - Show which orientation is selected

Embryos:
  - 20251113_A05_e01 frame 95  (reference, works)
  - 20251205_B04_e01 frame 70  (fails)
  - one extra WT that worked: 20251205_G10_e01 frame 69
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

OUTPUT_DIR = Path(__file__).parent / "debug_results" / "radius_sweep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_CSV = MORPHSEQ_ROOT / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
DATA_ROOT = MORPHSEQ_ROOT / "morphseq_playground"

CANONICAL_GRID_HW = (256, 576)
CANONICAL_UM_PER_PX = 10.0

EMBRYOS = [
    ("20251113_A05_e01", 95,  "REF — works"),
    ("20251205_B04_e01", 70,  "TGT — fails"),
    ("20251205_G10_e01", 69,  "WT   — works"),
    ("20251113_C04_e01", 14,  "TGT — C04 curl"),
]

# radius_k values to sweep
RADIUS_K_VALUES = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]


def load_mask_yolk(row: pd.Series):
    mask = fmio.load_mask_from_rle_counts(
        rle_counts=row["mask_rle"],
        height_px=int(row["mask_height_px"]),
        width_px=int(row["mask_width_px"]),
    )
    yolk = fmio._load_build02_aux_mask(DATA_ROOT, row, mask.shape, keyword="yolk")
    um_per_px = fmio._compute_um_per_pixel(row)
    return mask, yolk, float(um_per_px)


def apply_pca_rotation(mask, yolk, um_per_px, aligner):
    """Apply only PCA rotation (no orientation decision) to get a work-space mask."""
    scale = um_per_px / aligner.target_res
    angle_deg, (cx, cy), _ = aligner._pca_angle_deg(mask)
    target_angle = 0.0 if aligner.is_landscape else 90.0
    rotation_needed = angle_deg - target_angle
    M = cv2.getRotationMatrix2D((cx, cy), rotation_needed, scale)
    M[0, 2] += (aligner.W / 2) - cx
    M[1, 2] += (aligner.H / 2) - cy
    mask_w = aligner._warp(mask, M)
    yolk_w = aligner._warp(yolk, M) if yolk is not None else None
    return mask_w, yolk_w


def compute_back_for_radius(aligner, mask_w, yolk_w, radius_k):
    """Manually compute back point for a given radius_k."""
    yolk_com_y, yolk_com_x = aligner._center_of_mass(yolk_w if yolk_w is not None and yolk_w.sum() > 0 else mask_w)
    if yolk_w is None or yolk_w.sum() == 0:
        return (yolk_com_y, yolk_com_x), 0.0, 0, "no_yolk"

    yolk_area = float(yolk_w.sum())
    r_yolk = np.sqrt(yolk_area / np.pi)
    r_sample = radius_k * r_yolk

    ys, xs = np.where(mask_w > 0.5)
    if ys.size == 0:
        return (yolk_com_y, yolk_com_x), r_sample, 0, "empty_mask"

    dy = ys.astype(np.float64) - yolk_com_y
    dx = xs.astype(np.float64) - yolk_com_x
    in_disk = (dy**2 + dx**2) <= r_sample**2
    n_in = int(in_disk.sum())

    if n_in == 0:
        return (yolk_com_y, yolk_com_x), r_sample, 0, "empty_disk"

    back_y = float(ys[in_disk].mean())
    back_x = float(xs[in_disk].mean())
    return (back_y, back_x), r_sample, n_in, "ok"


def orientation_winner(aligner, mask, yolk, um_per_px, radius_k):
    """Run full candidate selection with a given radius_k, return winning (rot_add, flip, score)."""
    aligner.back_sample_radius_k = radius_k
    scale = um_per_px / aligner.target_res
    angle_deg, (cx, cy), _ = aligner._pca_angle_deg(mask)
    target_angle = 0.0 if aligner.is_landscape else 90.0
    rotation_needed = angle_deg - target_angle

    best_score = -np.inf
    best_rot = 0
    best_flip = False
    for rot_add in [0, 180]:
        for do_flip in [False, True]:
            M = cv2.getRotationMatrix2D((cx, cy), rotation_needed + rot_add, scale)
            M[0, 2] += (aligner.W / 2) - cx
            M[1, 2] += (aligner.H / 2) - cy
            mask_w = aligner._warp(mask, M)
            yolk_w = aligner._warp(yolk, M) if yolk is not None else None
            if do_flip:
                mask_w = cv2.flip(mask_w, 1)
                if yolk_w is not None:
                    yolk_w = cv2.flip(yolk_w, 1)
            yolk_feature = yolk_w if (yolk_w is not None and yolk_w.sum() > 0) else mask_w
            yolk_yx = aligner._center_of_mass(yolk_feature)
            back_yx = aligner._compute_back_direction(mask_w, yolk_mask=yolk_w)
            if yolk_w is not None and yolk_w.sum() > 0:
                score = -yolk_yx[1]  # minimize yolk x (wants yolk on left)
            else:
                score = (aligner.back_weight * (back_yx[1] + back_yx[0])) - (aligner.yolk_weight * (yolk_yx[1] + yolk_yx[0]))
            if score > best_score:
                best_score = score
                best_rot = rot_add
                best_flip = do_flip
    return best_rot, best_flip, best_score


def make_sweep_figure(embryo_id, frame_idx, label, mask, yolk, um_per_px, aligner):
    """One figure: rows = radius_k, columns = [mask+disk+back, orientation result]."""
    n_k = len(RADIUS_K_VALUES)

    # Get the PCA-rotated (but orientation-undecided) canonical frame for visualization
    mask_w_pca, yolk_w_pca = apply_pca_rotation(mask, yolk, um_per_px, aligner)

    yolk_feature = yolk_w_pca if (yolk_w_pca is not None and yolk_w_pca.sum() > 0) else mask_w_pca
    yolk_com_y, yolk_com_x = aligner._center_of_mass(yolk_feature)
    yolk_area = float(yolk_w_pca.sum()) if yolk_w_pca is not None and yolk_w_pca.sum() > 0 else 0.0
    r_yolk = np.sqrt(yolk_area / np.pi) if yolk_area > 0 else 0.0

    fig, axes = plt.subplots(n_k, 3, figsize=(18, n_k * 3.2))
    if n_k == 1:
        axes = axes[np.newaxis, :]

    print(f"\n  {label}: yolk r_yolk={r_yolk:.1f}px, yolk_area={yolk_area:.0f}px")

    for row_i, radius_k in enumerate(RADIUS_K_VALUES):
        back_yx, r_sample, n_in_disk, status = compute_back_for_radius(
            aligner, mask_w_pca, yolk_w_pca, radius_k
        )
        best_rot, best_flip, best_score = orientation_winner(aligner, mask, yolk, um_per_px, radius_k)

        print(f"    k={radius_k:.1f}: r_sample={r_sample:.1f}px n_in={n_in_disk:4d} "
              f"back=({back_yx[0]:.1f},{back_yx[1]:.1f}) → rot+{best_rot}° flip={best_flip} score={best_score:.2f}")

        # Col 0: PCA-rotated mask + sampling disk + yolk COM + back point
        ax0 = axes[row_i][0]
        overlay = np.zeros((*mask_w_pca.shape, 3), dtype=np.float32)
        overlay[mask_w_pca > 0.5] = [0.6, 0.6, 0.6]
        if yolk_w_pca is not None and yolk_w_pca.sum() > 0:
            overlay[yolk_w_pca > 0.5] = [1.0, 0.55, 0.0]

        # Draw sampling disk
        theta = np.linspace(0, 2 * np.pi, 200)
        disk_x = yolk_com_x + r_sample * np.cos(theta)
        disk_y = yolk_com_y + r_sample * np.sin(theta)

        ax0.imshow(overlay, interpolation="nearest")
        ax0.plot(disk_x, disk_y, 'c-', linewidth=1.2, label=f"disk r={r_sample:.0f}px")
        ax0.plot(yolk_com_x, yolk_com_y, 'r+', markersize=14, markeredgewidth=2, label="yolk COM")
        ax0.plot(back_yx[1], back_yx[0], 'b*', markersize=14, label=f"back ({status})")
        if n_in_disk > 0:
            # Highlight disk pixels
            ys, xs = np.where(mask_w_pca > 0.5)
            dy = ys - yolk_com_y
            dx = xs - yolk_com_x
            in_disk = (dy**2 + dx**2) <= r_sample**2
            if in_disk.any():
                ax0.scatter(xs[in_disk], ys[in_disk], s=0.3, c='yellow', alpha=0.4, zorder=2)

        ax0.set_title(f"k={radius_k} → r_sample={r_sample:.0f}px\nn_in_disk={n_in_disk}", fontsize=7)
        ax0.legend(fontsize=5, loc="upper right")
        ax0.axis("off")
        ax0.set_xlim(0, aligner.W)
        ax0.set_ylim(aligner.H, 0)

        # Col 1: Final canonical result at this radius_k
        aligner.back_sample_radius_k = radius_k
        mask_qc, _ = qc_mask(mask.astype(bool))
        yolk_bool = yolk.astype(bool) if yolk is not None and yolk.sum() > 0 else None
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            canonical, canonical_yolk, meta, _ = aligner.embryo_canonical_alignment(
                mask_qc, um_per_px, yolk=yolk_bool, use_pca=True
            )

        ax1 = axes[row_i][1]
        ax1.imshow(canonical, cmap="gray", interpolation="nearest")
        rot_deg = meta.get("rotation_deg", float("nan"))
        flip = meta.get("flip", "?")
        ax1.set_title(
            f"rot={rot_deg:.1f}° flip={flip}\n[+{best_rot}° flip={best_flip}]",
            fontsize=7
        )
        ax1.axis("off")

        # Col 2: Canonical with yolk overlay
        ax2 = axes[row_i][2]
        overlay2 = np.zeros((*canonical.shape, 3), dtype=np.float32)
        overlay2[canonical > 0.5] = [0.6, 0.6, 0.6]
        if canonical_yolk is not None and canonical_yolk.sum() > 0:
            overlay2[canonical_yolk > 0.5] = [1.0, 0.55, 0.0]
        ax2.imshow(overlay2, interpolation="nearest")
        back_final = meta.get("back_yx_final", (None, None))
        yolk_final = meta.get("yolk_yx_final", (None, None))
        if back_final[0] is not None:
            ax2.plot(back_final[1], back_final[0], 'b*', markersize=10)
        if yolk_final[0] is not None:
            ax2.plot(yolk_final[1], yolk_final[0], 'r+', markersize=10, markeredgewidth=2)
        ax2.set_title("Canonical+yolk\nred+=yolk, blue*=back", fontsize=7)
        ax2.axis("off")

    # Reset to default
    aligner.back_sample_radius_k = 1.5

    fig.suptitle(
        f"{label}  |  {embryo_id} frame {frame_idx}\n"
        f"yolk_area={yolk_area:.0f}px  r_yolk={r_yolk:.1f}px\n"
        f"Col0: PCA-rotated + sampling disk  |  Col1: Final canonical  |  Col2: +yolk overlay",
        fontsize=9, y=1.01
    )
    plt.tight_layout()
    tag = embryo_id.replace("_", "-")
    save_path = OUTPUT_DIR / f"radius_sweep_{tag}_f{frame_idx}.png"
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def main():
    print("=" * 70)
    print("RADIUS SWEEP: back_sample_radius_k effect on orientation")
    print("=" * 70)

    df = pd.read_csv(DATA_CSV, low_memory=False)

    cfg = CanonicalGridConfig(
        reference_um_per_pixel=CANONICAL_UM_PER_PX,
        grid_shape_hw=CANONICAL_GRID_HW,
        align_mode="yolk",
    )
    aligner = CanonicalAligner.from_config(cfg)

    for embryo_id, frame_idx, label in EMBRYOS:
        row = df[(df["embryo_id"] == embryo_id) & (df["frame_index"] == frame_idx)]
        if row.empty:
            print(f"SKIP: {embryo_id} frame {frame_idx} not found")
            continue
        row = row.iloc[0]
        mask, yolk, um_per_px = load_mask_yolk(row)
        mask_qc, _ = qc_mask(mask.astype(bool))
        yolk_bool = yolk.astype(bool) if yolk is not None and yolk.sum() > 0 else None

        print(f"\nProcessing {label}: {embryo_id} frame {frame_idx}")
        make_sweep_figure(embryo_id, frame_idx, label, mask_qc, yolk_bool, um_per_px, aligner)

    print(f"\nDone. Results in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
