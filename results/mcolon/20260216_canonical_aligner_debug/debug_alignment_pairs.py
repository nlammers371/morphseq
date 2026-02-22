#!/usr/bin/env python3
"""
Visualize canonical alignment for ref→target pairs (no OT transport).

For each sampled embryo pair, shows:
  Col 0: raw ref mask
  Col 1: raw target mask
  Col 2: canonical ref (src_canonical)
  Col 3: canonical target (tgt_canonical)
  Col 4: overlay of canonical src (red) + canonical tgt (blue) on canonical grid

Saves one PNG per pair to debug_results/alignment_pairs/.
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

from analyze.optimal_transport_morphometrics.uot_masks import frame_mask_io as fmio
from analyze.utils.coord.grids.canonical import CanonicalAligner, CanonicalGridConfig
from analyze.utils.masks.qc import qc_mask

OUTPUT_DIR = Path(__file__).parent / "debug_results" / "alignment_pairs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_CSV = MORPHSEQ_ROOT / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
DATA_ROOT = MORPHSEQ_ROOT / "morphseq_playground"

CANONICAL_GRID_HW = (256, 576)
CANONICAL_UM_PER_PX = 10.0

# Reference embryo (same as Phase 0 run)
REF_EMBRYO_ID = "20251113_A05_e01"
REF_FRAME_INDEX = 95

# Stage window to sample targets from
STAGE_LO, STAGE_HI = 47.0, 49.0
N_WT = 5
N_MUT = 5
SEED = 42


def load_mask_yolk(row: pd.Series):
    mask = fmio.load_mask_from_rle_counts(
        rle_counts=row["mask_rle"],
        height_px=int(row["mask_height_px"]),
        width_px=int(row["mask_width_px"]),
    )
    yolk = fmio._load_build02_aux_mask(DATA_ROOT, row, mask.shape, keyword="yolk")
    um_per_px = fmio._compute_um_per_pixel(row)
    return mask, yolk, float(um_per_px)


def canonicalize(mask, yolk, um_per_px, aligner):
    mask_qc, _ = qc_mask(mask.astype(bool))
    canonical, _, align_meta = aligner.align(
        mask_qc,
        yolk.astype(bool) if yolk is not None and yolk.sum() > 0 else None,
        original_um_per_px=um_per_px,
        use_pca=True,
        use_yolk=(yolk is not None and yolk.sum() > 0),
    )
    return canonical, align_meta


def plot_pair(ref_raw, ref_canonical, tgt_raw, tgt_canonical,
              ref_meta, tgt_meta, ref_id, tgt_id, genotype, save_path):
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))

    # Raw ref
    axes[0].imshow(ref_raw, cmap="gray", interpolation="nearest")
    axes[0].set_title(f"Raw ref\n{ref_raw.shape}", fontsize=8)
    axes[0].axis("off")

    # Raw target
    axes[1].imshow(tgt_raw, cmap="gray", interpolation="nearest")
    axes[1].set_title(f"Raw target ({genotype})\n{tgt_raw.shape}", fontsize=8)
    axes[1].axis("off")

    # Canonical ref
    axes[2].imshow(ref_canonical, cmap="Reds", interpolation="nearest", vmin=0, vmax=1)
    rot_r = ref_meta.get("rotation_deg", float("nan"))
    flip_r = ref_meta.get("flip", "?")
    ret_r = ref_meta.get("retained_ratio", float("nan"))
    axes[2].set_title(f"Canonical ref\nrot={rot_r:.1f}° flip={flip_r} ret={ret_r:.2f}", fontsize=8)
    axes[2].axis("off")

    # Canonical target
    axes[3].imshow(tgt_canonical, cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
    rot_t = tgt_meta.get("rotation_deg", float("nan"))
    flip_t = tgt_meta.get("flip", "?")
    ret_t = tgt_meta.get("retained_ratio", float("nan"))
    axes[3].set_title(f"Canonical target\nrot={rot_t:.1f}° flip={flip_t} ret={ret_t:.2f}", fontsize=8)
    axes[3].axis("off")

    # Overlay: src=red, tgt=blue, overlap=purple
    H, W = CANONICAL_GRID_HW
    overlay = np.zeros((H, W, 3), dtype=np.float32)
    src_bool = ref_canonical > 0
    tgt_bool = tgt_canonical > 0
    overlap = src_bool & tgt_bool
    src_only = src_bool & ~overlap
    tgt_only = tgt_bool & ~overlap
    overlay[src_only] = [0.9, 0.2, 0.2]    # red = ref only
    overlay[tgt_only] = [0.2, 0.4, 0.9]    # blue = target only
    overlay[overlap]  = [0.6, 0.2, 0.8]    # purple = overlap

    axes[4].imshow(overlay, interpolation="nearest")
    iou = float(overlap.sum()) / float(np.logical_or(src_bool, tgt_bool).sum() or 1)
    axes[4].set_title(f"Overlay (IoU={iou:.2f})\nref=red tgt=blue overlap=purple", fontsize=8)
    axes[4].axis("off")

    fig.suptitle(
        f"Ref: {ref_id}   →   Target: {tgt_id}\nGenotype: {genotype}",
        fontsize=9, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 70)
    print("CANONICAL ALIGNMENT PAIR DEBUG")
    print("=" * 70)

    df = pd.read_csv(DATA_CSV, low_memory=False)

    # Setup aligner
    cfg = CanonicalGridConfig(
        reference_um_per_pixel=CANONICAL_UM_PER_PX,
        grid_shape_hw=CANONICAL_GRID_HW,
        align_mode="yolk",
    )
    aligner = CanonicalAligner.from_config(cfg)

    # Load reference
    print(f"\nLoading reference: {REF_EMBRYO_ID} frame {REF_FRAME_INDEX}")
    ref_row = df[(df["embryo_id"] == REF_EMBRYO_ID) & (df["frame_index"] == REF_FRAME_INDEX)]
    if ref_row.empty:
        raise ValueError(f"Reference embryo not found: {REF_EMBRYO_ID} frame {REF_FRAME_INDEX}")
    ref_row = ref_row.iloc[0]
    ref_raw, ref_yolk, ref_um = load_mask_yolk(ref_row)
    ref_canonical, ref_meta = canonicalize(ref_raw, ref_yolk, ref_um, aligner)
    print(f"  Raw: {ref_raw.shape}, {ref_um:.3f} µm/px")
    print(f"  Canonical: {ref_canonical.shape}, rot={ref_meta.get('rotation_deg', '?'):.1f}°, "
          f"flip={ref_meta.get('flip', '?')}, retained={ref_meta.get('retained_ratio', '?'):.3f}")

    # Sample targets
    stage_df = df[
        (df["predicted_stage_hpf"] >= STAGE_LO) &
        (df["predicted_stage_hpf"] <= STAGE_HI) &
        (df["embryo_id"] != REF_EMBRYO_ID)
    ]
    rng = np.random.default_rng(SEED)

    samples = []
    for genotype, n in [("cep290_wildtype", N_WT), ("cep290_homozygous", N_MUT)]:
        gdf = stage_df[stage_df["genotype"] == genotype]
        per_embryo = gdf.groupby("embryo_id").first().reset_index()
        n = min(n, len(per_embryo))
        chosen = per_embryo.iloc[rng.choice(len(per_embryo), size=n, replace=False)]
        for _, row in chosen.iterrows():
            samples.append((row, genotype))

    print(f"\nProcessing {len(samples)} target pairs ({N_WT} WT + {N_MUT} mutant)...")
    print(f"Saving to: {OUTPUT_DIR}\n")

    success = 0
    for i, (tgt_row, genotype) in enumerate(samples):
        embryo_id = tgt_row["embryo_id"]
        frame_idx = int(tgt_row["frame_index"])
        tag = f"{i+1:02d}_{embryo_id}_f{frame_idx}_{genotype[:3]}"
        print(f"[{i+1}/{len(samples)}] {embryo_id} frame {frame_idx} ({genotype})")

        try:
            tgt_raw, tgt_yolk, tgt_um = load_mask_yolk(tgt_row)
            if tgt_yolk is None or tgt_yolk.sum() == 0:
                print(f"  SKIP: no yolk mask")
                continue

            tgt_canonical, tgt_meta = canonicalize(tgt_raw, tgt_yolk, tgt_um, aligner)

            print(f"  Raw: {tgt_raw.shape}, {tgt_um:.3f} µm/px")
            print(f"  Canonical: rot={tgt_meta.get('rotation_deg', '?'):.1f}°, "
                  f"flip={tgt_meta.get('flip', '?')}, retained={tgt_meta.get('retained_ratio', '?'):.3f}")

            save_path = OUTPUT_DIR / f"{tag}.png"
            plot_pair(
                ref_raw, ref_canonical,
                tgt_raw, tgt_canonical,
                ref_meta, tgt_meta,
                REF_EMBRYO_ID, embryo_id, genotype,
                save_path,
            )
            print(f"  Saved: {save_path.name}")
            success += 1

        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone. {success}/{len(samples)} pairs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
