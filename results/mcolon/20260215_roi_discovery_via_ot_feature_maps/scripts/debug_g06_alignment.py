#!/usr/bin/env python3
"""Test specific embryo that user reported as touching edge."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add morphseq root to path
MORPHSEQ_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MORPHSEQ_ROOT))
sys.path.insert(0, str(MORPHSEQ_ROOT / "src"))

from analyze.utils.coord.grids.canonical import CanonicalAligner
from analyze.optimal_transport_morphometrics.uot_masks import frame_mask_io as fmio

# Load data
data_root = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground")
csv_path = MORPHSEQ_ROOT / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
df = pd.read_csv(csv_path)

# Find the specific embryo
row = df[(df['embryo_id'] == '20251113_G06_e01') & (df['frame_index'] == 14)].iloc[0]

print(f"Testing {row['embryo_id']} frame {row['frame_index']}")
print(f"Genotype: {row['genotype']}")

# Decode mask using the same function as s01b
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
print(f"um_per_px: {um_per_px:.3f}")

# Initialize aligner
aligner = CanonicalAligner(
    canonical_height=256,
    canonical_width=576,
    um_per_pixel=10.0,
    anchor_point_xy=(288, 128),
    error_on_clip=False
)

# Try to align
print("\nAttempting alignment...")
try:
    aligned_mask, aligned_yolk, meta = aligner.align(mask, yolk, um_per_px)
    print(f"✓ Alignment successful!")
    print(f"  Aligned mask: {aligned_mask.sum()} pixels")
    print(f"  Rotation: {meta['rotation_deg']:.1f}°")
    print(f"  Flip: {meta['flip']}")
    print(f"  Scale: {meta['scale']:.3f}")
    print(f"  Shift: ({meta['anchor_shift_xy'][0]:.1f}, {meta['anchor_shift_xy'][1]:.1f})")
    
    # Check edges
    touches_top = aligned_mask[0, :].any()
    touches_bottom = aligned_mask[-1, :].any()
    touches_left = aligned_mask[:, 0].any()
    touches_right = aligned_mask[:, -1].any()
    
    if any([touches_top, touches_bottom, touches_left, touches_right]):
        edges = []
        if touches_top: edges.append("top")
        if touches_bottom: edges.append("bottom")
        if touches_left: edges.append("left")
        if touches_right: edges.append("right")
        print(f"  ⚠️  Mask touches edges: {', '.join(edges)}")
    else:
        print(f"  ✓ Mask does not touch any edges")
        
except RuntimeError as e:
    print(f"❌ Alignment failed with RuntimeError:")
    print(f"   {e}")
except Exception as e:
    print(f"❌ Alignment failed with unexpected error:")
    print(f"   {type(e).__name__}: {e}")

# Initialize aligner
aligner = CanonicalAligner(
    canonical_height=256,
    canonical_width=576,
    um_per_pixel=10.0,
    anchor_point_xy=(288, 128),
    error_on_clip=False
)

# Try to align
print("\nAttempting alignment...")
try:
    aligned_mask, aligned_yolk, meta = aligner.align(mask, yolk, row['um_per_px'])
    print(f"✓ Alignment successful!")
    print(f"  Aligned mask: {aligned_mask.sum()} pixels")
    print(f"  Rotation: {meta['rotation_deg']:.1f}°")
    print(f"  Flip: {meta['flip']}")
    print(f"  Scale: {meta['scale']:.3f}")
    print(f"  Shift: ({meta['anchor_shift_xy'][0]:.1f}, {meta['anchor_shift_xy'][1]:.1f})")
    
    # Check edges
    touches_top = aligned_mask[0, :].any()
    touches_bottom = aligned_mask[-1, :].any()
    touches_left = aligned_mask[:, 0].any()
    touches_right = aligned_mask[:, -1].any()
    
    if any([touches_top, touches_bottom, touches_left, touches_right]):
        edges = []
        if touches_top: edges.append("top")
        if touches_bottom: edges.append("bottom")
        if touches_left: edges.append("left")
        if touches_right: edges.append("right")
        print(f"  ⚠️  Mask touches edges: {', '.join(edges)}")
    else:
        print(f"  ✓ Mask does not touch any edges")
        
except RuntimeError as e:
    print(f"❌ Alignment failed with RuntimeError:")
    print(f"   {e}")
except Exception as e:
    print(f"❌ Alignment failed with unexpected error:")
    print(f"   {type(e).__name__}: {e}")
