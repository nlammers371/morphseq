#!/usr/bin/env python3
"""Debug why alignment returns 0 pixels for specific embryos"""

import sys
from pathlib import Path

MORPHSEQ_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(MORPHSEQ_ROOT))
sys.path.insert(0, str(MORPHSEQ_ROOT / "src"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analyze.optimal_transport_morphometrics.uot_masks import frame_mask_io as fmio
from analyze.utils.coord.grids.canonical import CanonicalAligner, CanonicalGridConfig

# Load data
data_csv = MORPHSEQ_ROOT / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
df = pd.read_csv(data_csv, low_memory=False)
data_root = MORPHSEQ_ROOT / "morphseq_playground"

# Test the failing embryo from s01b
embryo_id = "20251205_B08_e01"
frame_index = 70

row = df[(df["embryo_id"] == embryo_id) & (df["frame_index"] == frame_index)].iloc[0]

print(f"Debug: {embryo_id} frame {frame_index}")
print(f"  Stage: {row['predicted_stage_hpf']:.1f} hpf")
print(f"  Genotype: {row['genotype']}")

# Load mask
mask = fmio.load_mask_from_rle_counts(
    rle_counts=row["mask_rle"],
    height_px=int(row["mask_height_px"]),
    width_px=int(row["mask_width_px"]),
)
print(f"  Original mask: {mask.sum()} pixels, shape {mask.shape}")

# Load yolk
yolk = fmio._load_build02_aux_mask(data_root, row, mask.shape, keyword="yolk")
print(f"  Yolk mask: {yolk.sum()} pixels")

# Align
config = CanonicalGridConfig(reference_um_per_pixel=10.0, grid_shape_hw=(256, 576))
aligner = CanonicalAligner.from_config(config)

um_per_px = fmio._compute_um_per_pixel(row)
print(f"  Resolution: {um_per_px:.3f} um/px")

print(f"\n  Attempting alignment with use_yolk=True...")
try:
    aligned_yolk, _, _ = aligner.align(
        mask=mask.astype(bool),
        yolk=yolk.astype(bool),
        original_um_per_px=um_per_px,
        use_yolk=True,
    )
    print(f"  ✓ Aligned mask: {aligned_yolk.sum()} pixels")
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    aligned_yolk = None

print(f"\n  Attempting alignment with use_yolk=False...")
try:
    aligned_no_yolk, _, _ = aligner.align(
        mask=mask.astype(bool),
        yolk=yolk.astype(bool),
        original_um_per_px=um_per_px,
        use_yolk=False,
    )
    print(f"  ✓ Aligned mask: {aligned_no_yolk.sum()} pixels")
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    aligned_no_yolk = None

# Visualize
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(mask, cmap="gray")
axes[0].set_title("Original Mask")

axes[1].imshow(yolk, cmap="gray")
axes[1].set_title("Yolk Mask")

if aligned_yolk is not None:
    axes[2].imshow(aligned_yolk, cmap="gray")
    axes[2].set_title(f"Aligned (use_yolk=True)\n{aligned_yolk.sum()} pixels")
else:
    axes[2].text(0.5, 0.5, "FAILED", ha="center", va="center", transform=axes[2].transAxes)
    axes[2].set_title("Aligned (use_yolk=True)")

if aligned_no_yolk is not None:
    axes[3].imshow(aligned_no_yolk, cmap="gray")
    axes[3].set_title(f"Aligned (use_yolk=False)\n{aligned_no_yolk.sum()} pixels")
else:
    axes[3].text(0.5, 0.5, "FAILED", ha="center", va="center", transform=axes[3].transAxes)
    axes[3].set_title("Aligned (use_yolk=False)")

for ax in axes:
    ax.axis("off")

output_path = Path("scripts/output/mask_qc/debug_alignment_failure.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"\nSaved debug visualization: {output_path}")
