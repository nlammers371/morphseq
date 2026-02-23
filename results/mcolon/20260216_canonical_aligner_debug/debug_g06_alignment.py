#!/usr/bin/env python3
"""Debug alignment for 20251113_G06_e01 to understand orientation detection."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add morphseq root to path
MORPHSEQ_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MORPHSEQ_ROOT))
sys.path.insert(0, str(MORPHSEQ_ROOT / "src"))

from analyze.utils.coord.grids.canonical import CanonicalAligner, CanonicalGridConfig
from analyze.optimal_transport_morphometrics.uot_masks import frame_mask_io as fmio

# Load data
data_root = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground")
csv_path = MORPHSEQ_ROOT / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
df = pd.read_csv(csv_path)

# Find the specific embryo
row = df[(df['embryo_id'] == '20251113_G06_e01') & (df['frame_index'] == 14)].iloc[0]

print(f"=== Debugging {row['embryo_id']} frame {row['frame_index']} ===")
print(f"Genotype: {row['genotype']}")

# Decode mask
mask = fmio.load_mask_from_rle_counts(
    rle_counts=row["mask_rle"],
    height_px=int(row["mask_height_px"]),
    width_px=int(row["mask_width_px"]),
)
print(f"Original mask: {mask.sum()} pixels, shape {mask.shape}")

# Load yolk
yolk = fmio._load_build02_aux_mask(
    data_root,
    row,
    mask.shape,
    keyword="yolk",
)
print(f"Yolk mask: {yolk.sum()} pixels, shape {yolk.shape}")

# Get um_per_px
um_per_px = fmio._compute_um_per_pixel(row)
print(f"um_per_px: {um_per_px:.3f}\n")

# Initialize aligner (same as s01b)
config = CanonicalGridConfig(reference_um_per_pixel=10.0, grid_shape_hw=(256, 576))
aligner = CanonicalAligner.from_config(config)

# Align with debug output
print("Aligning with debug enabled...")
aligned_mask, aligned_yolk, meta = aligner.align(
    mask=mask.astype(bool),
    yolk=yolk.astype(bool),
    original_um_per_px=um_per_px,
    use_yolk=True,
    return_debug=True
)

print(f"\n=== Alignment Results ===")
print(f"Aligned mask: {aligned_mask.sum()} pixels")
print(f"Retained ratio: {meta['retained_ratio']:.4f}")
print(f"PCA angle: {meta['pca_angle_deg']:.1f}°")
print(f"Rotation needed: {meta['rotation_needed_deg']:.1f}°")
print(f"Final rotation: {meta['rotation_deg']:.1f}°")
print(f"Flip: {meta['flip']}")
print(f"Scale: {meta['scale']:.3f}")
print(f"Shift: ({meta['anchor_shift_xy'][0]:.1f}, {meta['anchor_shift_xy'][1]:.1f})")
print(f"Yolk (pre-shift): {meta['yolk_yx_pre_shift']}")
print(f"Back (pre-shift): {meta['back_yx_pre_shift']}")
print(f"Yolk (final): {meta['yolk_yx_final']}")
print(f"Back (final): {meta['back_yx_final']}")

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 10))

# Row 1: Original data
ax1 = plt.subplot(2, 4, 1)
ax1.imshow(mask, cmap='gray', origin='upper')
ax1.set_title('Original Mask', fontsize=10)
ax1.axis('off')

ax2 = plt.subplot(2, 4, 2)
ax2.imshow(yolk, cmap='gray', origin='upper')
ax2.set_title('Original Yolk', fontsize=10)
ax2.axis('off')

ax3 = plt.subplot(2, 4, 3)
ax3.imshow(mask, cmap='gray', origin='upper', alpha=0.7)
ax3.imshow(yolk, cmap='Blues', origin='upper', alpha=0.3)
ax3.set_title('Overlay (Mask + Yolk)', fontsize=10)
ax3.axis('off')

# Row 2: Aligned results
ax5 = plt.subplot(2, 4, 5)
ax5.imshow(aligned_mask, cmap='gray', origin='upper')
yolk_y, yolk_x = meta['yolk_yx_final']
back_y, back_x = meta['back_yx_final']
ax5.plot(yolk_x, yolk_y, 'ro', markersize=8, label='Yolk')
ax5.plot(back_x, back_y, 'bo', markersize=8, label='Back')
ax5.axvline(x=288, color='yellow', linestyle='--', alpha=0.5, label='Anchor X')
ax5.axhline(y=128, color='yellow', linestyle='--', alpha=0.5, label='Anchor Y')
ax5.set_title(f'Aligned Mask\n(flip={meta["flip"]}, rot={meta["rotation_deg"]:.1f}°)', fontsize=10)
ax5.legend(loc='upper right', fontsize=8)
ax5.set_xlim(0, 576)
ax5.set_ylim(256, 0)
ax5.grid(True, alpha=0.3)

ax6 = plt.subplot(2, 4, 6)
if aligned_yolk is not None:
    ax6.imshow(aligned_yolk, cmap='gray', origin='upper')
    ax6.set_title('Aligned Yolk', fontsize=10)
else:
    ax6.text(0.5, 0.5, 'No aligned yolk', ha='center', va='center')
    ax6.set_title('Aligned Yolk (None)', fontsize=10)
ax6.axis('off')

ax7 = plt.subplot(2, 4, 7)
ax7.imshow(aligned_mask, cmap='gray', origin='upper', alpha=0.7)
if aligned_yolk is not None:
    ax7.imshow(aligned_yolk, cmap='Blues', origin='upper', alpha=0.3)
ax7.plot(yolk_x, yolk_y, 'ro', markersize=8, label='Yolk')
ax7.plot(back_x, back_y, 'bo', markersize=8, label='Back')
ax7.set_title('Aligned Overlay', fontsize=10)
ax7.legend(loc='upper right', fontsize=8)
ax7.set_xlim(0, 576)
ax7.set_ylim(256, 0)

# Metadata text
ax8 = plt.subplot(2, 4, 8)
ax8.axis('off')
info_text = f"""Alignment Metadata:

PCA Angle: {meta['pca_angle_deg']:.1f}°
Rotation Needed: {meta['rotation_needed_deg']:.1f}°
Final Rotation: {meta['rotation_deg']:.1f}°
Flip: {meta['flip']}
Scale: {meta['scale']:.3f}

Yolk (pre): {meta['yolk_yx_pre_shift']}
Back (pre): {meta['back_yx_pre_shift']}

Yolk (final): ({yolk_y:.1f}, {yolk_x:.1f})
Back (final): ({back_y:.1f}, {back_x:.1f})

Shift: ({meta['anchor_shift_xy'][0]:.1f}, {meta['anchor_shift_xy'][1]:.1f})
Retained: {meta['retained_ratio']:.1%}

Embryo: {row['embryo_id']}
Frame: {row['frame_index']}
"""
ax8.text(0.05, 0.95, info_text, transform=ax8.transAxes,
         fontsize=9, verticalalignment='top', family='monospace')

# Pre-shift visualization (if available in debug)
if 'debug' in meta:
    ax4 = plt.subplot(2, 4, 4)
    pre_shift = meta['debug']['aligned_mask_pre_shift']
    ax4.imshow(pre_shift, cmap='gray', origin='upper')
    if meta['yolk_yx_pre_shift']:
        hy, hx = meta['yolk_yx_pre_shift']
        ax4.plot(hx, hy, 'ro', markersize=8, label='Yolk (pre)')
    if meta['back_yx_pre_shift']:
        by, bx = meta['back_yx_pre_shift']
        ax4.plot(bx, by, 'bo', markersize=8, label='Back (pre)')
    ax4.set_title('Pre-Shift Aligned', fontsize=10)
    ax4.legend(loc='upper right', fontsize=8)
    ax4.set_xlim(0, 576)
    ax4.set_ylim(256, 0)
    ax4.grid(True, alpha=0.3)

plt.suptitle(f'Alignment Debug: {row["embryo_id"]} frame {row["frame_index"]}', 
             fontsize=14, fontweight='bold')
plt.tight_layout()

output_path = Path('scripts/output/mask_qc/debug_g06_alignment.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved debug visualization to {output_path}")
plt.close()

# Check if yolk is to the left of back (correct orientation)
if yolk_x < back_x:
    print("\nOrientation appears CORRECT: Yolk (x={:.1f}) is LEFT of Back (x={:.1f})".format(yolk_x, back_x))
else:
    print("\nOrientation appears WRONG: Yolk (x={:.1f}) is RIGHT of Back (x={:.1f})".format(yolk_x, back_x))
    print("   Expected: Yolk on LEFT, Back on RIGHT")
