#!/usr/bin/env python3
"""
Debug canonical alignment (PCA rotation + flip + anchoring).

Writes raw, pre-shift, and final aligned masks plus alignment metadata.
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


def save_mask(mask: np.ndarray, path: Path, title: str, display_mode: str = "image") -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    h, w = mask.shape[:2]
    if display_mode == "cartesian":
        ax.imshow(mask, cmap="gray", origin="lower", extent=[0, w, 0, h])
        ax.set_ylim(0, h)
    else:
        ax.imshow(mask, cmap="gray", origin="upper", extent=[0, w, h, 0])
        ax.set_ylim(h, 0)
    ax.set_title(title)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.axis("image")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_one(
    csv_path: Path,
    embryo_id: str,
    frame_index: int,
    out_dir: Path,
    target_shape_hw: tuple[int, int],
    target_um_per_px: float,
    allow_flip: bool,
    data_root: Path | None = None,
    display_mode: str = "image",
) -> None:
    frame = load_mask_from_csv(csv_path, embryo_id, frame_index, data_root=data_root)
    mask = frame.embryo_mask
    yolk = frame.meta.get("yolk_mask", None)
    um_per_px = frame.meta.get("um_per_pixel", np.nan)

    aligner = CanonicalAligner(
        target_shape_hw=target_shape_hw,
        target_um_per_pixel=target_um_per_px,
        allow_flip=allow_flip,
    )

    aligned_mask, aligned_yolk, meta = aligner.align(
        mask,
        yolk,
        original_um_per_px=float(um_per_px),
        use_pca=True,
        use_yolk=True,
        return_debug=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    save_mask(mask, out_dir / "raw_mask.png", f"{embryo_id} f{frame_index} (raw)", display_mode)
    if yolk is not None:
        save_mask(yolk, out_dir / "raw_yolk.png", f"{embryo_id} f{frame_index} (raw yolk)", display_mode)

    debug = meta.get("debug", {})
    pre_shift = debug.get("aligned_mask_pre_shift", None)
    if pre_shift is not None:
        save_mask(pre_shift, out_dir / "aligned_pre_shift.png", "Aligned (pre-shift)", display_mode)
    save_mask(aligned_mask, out_dir / "aligned_final.png", "Aligned (final)", display_mode)

    if aligned_yolk is not None:
        save_mask(aligned_yolk, out_dir / "aligned_yolk.png", "Aligned yolk", display_mode)

    meta_out = meta.copy()
    meta_out.pop("debug", None)
    with open(out_dir / "alignment_meta.json", "w") as f:
        json.dump(meta_out, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug canonical alignment outputs")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--embryo-id", type=str, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("results/mcolon/20260121_uot-mvp/debug_alignment"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--um-per-px", type=float, default=10.0)
    parser.add_argument("--no-flip", action="store_true")
    parser.add_argument("--display-mode", choices=["image", "cartesian"], default="image")
    args = parser.parse_args()

    run_one(
        args.csv,
        args.embryo_id,
        args.frame,
        args.outdir / f"{args.embryo_id}_f{args.frame}",
        (args.height, args.width),
        args.um_per_px,
        allow_flip=not args.no_flip,
        data_root=args.data_root,
        display_mode=args.display_mode,
    )


if __name__ == "__main__":
    main()
