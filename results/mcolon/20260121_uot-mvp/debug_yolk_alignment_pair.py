#!/usr/bin/env python3
"""
Debug yolk-based alignment for two embryos/frames.

Creates overlays and marks head/back positions used by the aligner.
"""

from __future__ import annotations

import sys
from pathlib import Path
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add morphseq root to path
morphseq_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(morphseq_root))

from src.analyze.optimal_transport_morphometrics.uot_masks.frame_mask_io import load_mask_from_csv
from src.analyze.optimal_transport_morphometrics.uot_masks.uot_grid import CanonicalAligner


DEFAULT_CSV = Path(
    "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
)


def _overlay_rgb(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    overlay = np.zeros((*a.shape, 3), dtype=np.float32)
    overlay[a & ~b] = [1.0, 0.0, 0.0]  # red
    overlay[~a & b] = [0.0, 1.0, 0.0]  # green
    overlay[a & b] = [1.0, 1.0, 0.0]   # yellow
    return overlay


def _plot_overlay(
    ax,
    overlay: np.ndarray,
    title: str,
    meta_a: dict | None = None,
    meta_b: dict | None = None,
    show_back: bool = False,
):
    h, w = overlay.shape[:2]
    ax.imshow(overlay, origin="upper", extent=[0, w, h, 0])
    ax.set_title(title)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    if meta_a is not None:
        yolk_pt = meta_a.get("yolk_yx_final")
        back = meta_a.get("back_yx_final")
        if yolk_pt:
            ax.scatter([yolk_pt[1]], [yolk_pt[0]], s=40, c="cyan", marker="o", label="A yolk")
        if show_back and back:
            ax.scatter([back[1]], [back[0]], s=40, c="blue", marker="x", label="A back")
    if meta_b is not None:
        yolk_pt = meta_b.get("yolk_yx_final")
        back = meta_b.get("back_yx_final")
        if yolk_pt:
            ax.scatter([yolk_pt[1]], [yolk_pt[0]], s=40, c="magenta", marker="o", label="B yolk")
        if show_back and back:
            ax.scatter([back[1]], [back[0]], s=40, c="purple", marker="x", label="B back")

    if meta_a or meta_b:
        ax.legend(loc="lower right", fontsize=8, framealpha=0.6)


def run_pair(
    csv_path: Path,
    embryo_a: str,
    frame_a: int,
    embryo_b: str,
    frame_b: int,
    out_dir: Path,
    target_shape_hw: tuple[int, int],
    target_um_per_px: float,
    allow_flip: bool,
    data_root: Path | None = None,
) -> None:
    frame_a_obj = load_mask_from_csv(csv_path, embryo_a, frame_a, data_root=data_root)
    frame_b_obj = load_mask_from_csv(csv_path, embryo_b, frame_b, data_root=data_root)

    aligner = CanonicalAligner(
        target_shape_hw=target_shape_hw,
        target_um_per_pixel=target_um_per_px,
        allow_flip=allow_flip,
    )

    a_mask, a_yolk, a_meta = aligner.align(
        frame_a_obj.embryo_mask,
        frame_a_obj.meta.get("yolk_mask", None),
        original_um_per_px=float(frame_a_obj.meta.get("um_per_pixel", 1.0)),
        use_pca=True,
        use_yolk=True,
    )
    b_mask, b_yolk, b_meta = aligner.align(
        frame_b_obj.embryo_mask,
        frame_b_obj.meta.get("yolk_mask", None),
        original_um_per_px=float(frame_b_obj.meta.get("um_per_pixel", 1.0)),
        use_pca=True,
        use_yolk=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    overlay_masks = _overlay_rgb(a_mask, b_mask)
    overlay_yolks = _overlay_rgb(
        a_yolk if a_yolk is not None else np.zeros_like(a_mask),
        b_yolk if b_yolk is not None else np.zeros_like(b_mask),
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    _plot_overlay(axes[0, 0], overlay_masks, "Aligned Masks (A red, B green)")
    _plot_overlay(axes[0, 1], overlay_yolks, "Aligned Yolks (A red, B green)")
    _plot_overlay(axes[1, 0], overlay_masks, "Masks + Head/Back", a_meta, b_meta, show_back=True)
    _plot_overlay(axes[1, 1], overlay_yolks, "Yolks + Head", a_meta, b_meta, show_back=False)
    fig.tight_layout()
    fig.savefig(out_dir / "alignment_overlay.png", dpi=150)
    plt.close(fig)

    with open(out_dir / "alignment_meta_a.json", "w") as f:
        json.dump(a_meta, f, indent=2)
    with open(out_dir / "alignment_meta_b.json", "w") as f:
        json.dump(b_meta, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug yolk alignment for two embryos")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--embryo-a", required=True)
    parser.add_argument("--frame-a", type=int, required=True)
    parser.add_argument("--embryo-b", required=True)
    parser.add_argument("--frame-b", type=int, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("results/mcolon/20260121_uot-mvp/debug_alignment_pair"))
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--um-per-px", type=float, default=10.0)
    parser.add_argument("--no-flip", action="store_true")
    args = parser.parse_args()

    run_pair(
        args.csv,
        args.embryo_a,
        args.frame_a,
        args.embryo_b,
        args.frame_b,
        args.outdir / f"{args.embryo_a}_f{args.frame_a}_vs_{args.embryo_b}_f{args.frame_b}",
        (args.height, args.width),
        args.um_per_px,
        allow_flip=not args.no_flip,
        data_root=args.data_root,
    )


if __name__ == "__main__":
    main()
