#!/usr/bin/env python3
"""Quick test: Check if OT gives non-zero costs with POT backend."""

import sys
from pathlib import Path

MORPHSEQ_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(MORPHSEQ_ROOT))
sys.path.insert(0, str(MORPHSEQ_ROOT / "src"))

ROI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROI_DIR))

import numpy as np
from p0_ot_maps import run_single_ot

# Create two simple test masks
print("Creating test masks...")
mask_ref = np.zeros((1000, 1000), dtype=np.uint8)
mask_ref[400:600, 400:600] = 1  # 200x200 square in center

mask_tgt = np.zeros((1000, 1000), dtype=np.uint8) 
mask_tgt[450:650, 450:650] = 1  # 200x200 square, shifted by 50 pixels

print(f"Reference mask: {mask_ref.sum()} pixels")
print(f"Target mask: {mask_tgt.sum()} pixels")
print(f"Overlap: {(mask_ref & mask_tgt).sum()} pixels")

print("\nRunning OT with POT backend...")
result = run_single_ot(
    mask_ref=mask_ref,
    mask_target=mask_tgt,
    sample_id="test",
    raw_um_per_px_ref=10.0,
    raw_um_per_px_tgt=10.0,
    yolk_ref=None,
    yolk_tgt=None,
)

print(f"\n✓ OT completed!")
print(f"Total cost: {result.total_cost_C:.6e}")
print(f"Cost density shape: {result.cost_density.shape}")
print(f"Cost density mean: {result.cost_density.mean():.6e}")
print(f"Cost density max: {result.cost_density.max():.6e}")
print(f"Nonzero pixels: {(result.cost_density > 0).sum()} / {result.cost_density.size}")
print(f"Nonzero %: {100 * (result.cost_density > 0).sum() / result.cost_density.size:.2f}%")

if result.total_cost_C > 0:
    print("\n✅ SUCCESS: Nonzero cost detected!")
else:
    print("\n❌ PROBLEM: Cost is still zero")
