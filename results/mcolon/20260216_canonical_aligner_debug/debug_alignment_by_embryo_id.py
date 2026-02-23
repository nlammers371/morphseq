#!/usr/bin/env python3
"""
Standalone canonical alignment debug for specific embryos by ID.

Loads each embryo directly from the CSV (with full columns including
experiment_date, well, time_int so yolk masks load correctly), runs
the CanonicalAligner, and plots head/back/yolk overlaid on the aligned
mask — without needing any pre-existing pipeline Zarr output.

Usage:
    PYTHON=/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python
    PYTHONPATH=src "$PYTHON" results/mcolon/20260216_canonical_aligner_debug/debug_alignment_by_embryo_id.py
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

OUT_DIR = Path(__file__).resolve().parent / "debug_results" / "alignment_by_embryo_id"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Embryos to debug: (embryo_id, frame_index, label)
TARGETS = [
    ("20251106_E09_e01", 20,  "sample_012 (O13 / cep290_homozygous)"),
    ("20251113_A05_e01", 95,  "reference (WT)"),
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

# CSV columns needed — must include experiment_date, well, time_int for yolk loading
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


def draw_landmarks(ax, meta: dict):
    yolk = meta.get("yolk_yx_final")
    back = meta.get("back_yx_final")
    if yolk is not None and back is not None:
        ax.plot(
            [yolk[1], back[1]], [yolk[0], back[0]],
            color="white", linewidth=2.0, alpha=0.85, zorder=9,
        )
    if yolk is not None:
        ax.scatter(yolk[1], yolk[0], s=100, marker="o", color="red",
                   edgecolor="black", linewidth=0.8, zorder=10, label="yolk")
        ax.text(yolk[1] + 3, yolk[0] - 4, "Y", color="red", fontsize=9, weight="bold", zorder=11)
    if back is not None:
        ax.scatter(back[1], back[0], s=100, marker="s", color="cyan",
                   edgecolor="black", linewidth=0.8, zorder=10, label="back")
        ax.text(back[1] + 3, back[0] - 4, "B", color="cyan", fontsize=9, weight="bold", zorder=11)

    back_dbg = meta.get("debug", {}).get("back_direction", {})
    yolk_com = back_dbg.get("yolk_com_yx")
    if yolk_com is not None:
        ax.scatter(yolk_com[1], yolk_com[0], s=40, marker="+", color="yellow",
                   linewidth=1.5, zorder=8, label="yolk COM")


def summary_text(meta: dict, row: pd.Series, yolk_raw, raw_mask, aligned_mask, aligned_yolk) -> str:
    yolk = meta.get("yolk_yx_final")
    back = meta.get("back_yx_final")
    flip = meta.get("flip", False)
    rot = meta.get("rotation_deg", float("nan"))
    retained = meta.get("retained_ratio", float("nan"))

    yb_angle = float("nan")
    yolk_left = None
    back_below = None
    if yolk is not None and back is not None:
        dy = back[0] - yolk[0]
        dx = back[1] - yolk[1]
        yb_angle = float(np.degrees(np.arctan2(dy, dx)))
        yolk_left = bool(yolk[1] < back[1])
        back_below = bool(back[0] > yolk[0])

    yolk_sum = yolk_raw.sum() if yolk_raw is not None else 0
    raw_yolk_in = float("nan")
    aligned_yolk_in = float("nan")
    if yolk_raw is not None and yolk_raw.sum() > 0:
        raw_yolk_in = float(np.logical_and(raw_mask > 0, yolk_raw > 0).sum() / max(int(yolk_raw.sum()), 1))
    if aligned_yolk is not None and aligned_yolk.sum() > 0:
        aligned_yolk_in = float(
            np.logical_and(aligned_mask > 0, aligned_yolk > 0).sum() / max(int(aligned_yolk.sum()), 1)
        )
    back_dbg = meta.get("debug", {}).get("back_direction", {})
    return (
        f"embryo_id: {row['embryo_id']}\n"
        f"frame_index: {row['frame_index']}\n"
        f"genotype: {row['genotype']}\n"
        f"stage_hpf: {row['predicted_stage_hpf']:.2f}\n"
        f"um_per_px (raw): {fmio._compute_um_per_pixel(row):.4f}\n"
        f"yolk_pixels (raw): {yolk_sum}\n"
        f"raw_yolk_inside_ratio: {raw_yolk_in:.4f}\n"
        f"aligned_yolk_inside_ratio: {aligned_yolk_in:.4f}\n"
        f"---\n"
        f"rotation_deg: {rot:.3f}\n"
        f"flip: {flip}\n"
        f"retained_ratio: {retained:.4f}\n"
        f"yolk_yx: {yolk}\n"
        f"back_yx: {back}\n"
        f"yb_angle_deg: {yb_angle:.2f}\n"
        f"yolk_left_of_back: {yolk_left}\n"
        f"back_below_yolk: {back_below}\n"
        f"--- back debug ---\n"
        f"selected: {back_dbg.get('selected', 'n/a')}\n"
        f"r_yolk: {back_dbg.get('r_yolk', 'n/a')}\n"
        f"r_sample: {back_dbg.get('r_sample', 'n/a')}\n"
        f"n_pixels_in_disk: {back_dbg.get('n_pixels_in_disk', 'n/a')}\n"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading CSV from {DATA_CSV}")
    df = pd.read_csv(DATA_CSV, usecols=USECOLS)

    aligner = CanonicalAligner.from_config(CANONICAL_CFG)

    for embryo_id, frame_index, label in TARGETS:
        print(f"\nProcessing: {embryo_id} f{frame_index} — {label}")

        raw_mask, raw_yolk, um_per_px, row = load_embryo(df, embryo_id, frame_index)

        if raw_yolk is None or raw_yolk.sum() == 0:
            print(f"  WARNING: yolk mask is missing or empty for {embryo_id} f{frame_index}")
        else:
            print(f"  yolk pixels: {raw_yolk.sum()}, um/px: {um_per_px:.4f}")

        aligned_mask, aligned_yolk, meta, _chain = aligner.embryo_canonical_alignment(
            raw_mask.astype(bool),
            um_per_px,
            yolk=raw_yolk.astype(bool) if raw_yolk is not None else None,
            use_pca=True,
            return_debug=True,
        )

        yolk_pt = meta.get("yolk_yx_final")
        back = meta.get("back_yx_final")
        print(f"  yolk_yx: {yolk_pt}")
        print(f"  back_yx: {back}")
        print(f"  rotation: {meta.get('rotation_deg', float('nan')):.2f} deg, flip: {meta.get('flip', False)}")

        # --- Plot ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Panel 1: raw mask + yolk
        axes[0].imshow(raw_mask, cmap="gray", origin="upper")
        if raw_yolk is not None and raw_yolk.sum() > 0:
            axes[0].imshow(
                np.where(raw_yolk.astype(bool), 1.0, np.nan),
                cmap="Blues", alpha=0.45, origin="upper",
            )
        axes[0].set_title(f"Raw mask + yolk\n{embryo_id} f{frame_index}", fontsize=10)
        axes[0].axis("off")

        # Panel 2: canonical aligned mask + yolk + head/back
        axes[1].imshow(aligned_mask, cmap="gray", origin="upper")
        if aligned_yolk is not None and aligned_yolk.sum() > 0:
            axes[1].imshow(
                np.where(aligned_yolk.astype(bool), 1.0, np.nan),
                cmap="Blues", alpha=0.45, origin="upper",
            )
        draw_landmarks(axes[1], meta)
        axes[1].legend(loc="lower right", fontsize=8, framealpha=0.75)
        axes[1].set_title(
            f"Canonical aligned\nrot={meta.get('rotation_deg', float('nan')):.1f}°, "
            f"flip={meta.get('flip', False)}",
            fontsize=10,
        )
        axes[1].axis("off")

        # Panel 3: text summary
        axes[2].axis("off")
        axes[2].text(
            0.03, 0.97,
            summary_text(meta, row, raw_yolk, raw_mask, aligned_mask, aligned_yolk),
            va="top", ha="left", family="monospace", fontsize=9,
            transform=axes[2].transAxes,
        )

        fig.suptitle(label, fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        safe_id = embryo_id.replace("/", "_")
        out_path = OUT_DIR / f"{safe_id}_f{frame_index}_alignment_debug.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"  Wrote: {out_path}")

    print(f"\nAll done. Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
