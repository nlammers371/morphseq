#!/usr/bin/env python3
"""Debug script: Check single OT computation in detail."""

from __future__ import annotations
import sys
from pathlib import Path

MORPHSEQ_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(MORPHSEQ_ROOT))
sys.path.insert(0, str(MORPHSEQ_ROOT / "src"))

ROI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROI_DIR))

import numpy as np
import pandas as pd
from p0_ot_maps import run_single_ot

# Load data (copy from s02_compute_ot_features.py)
DATA_CSV = MORPHSEQ_ROOT / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"

from analyze.optimal_transport_morphometrics.uot_masks import frame_mask_io as fmio

# Load data (copy from s02_compute_ot_features.py)
DATA_CSV = MORPHSEQ_ROOT / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"

from analyze.optimal_transport_morphometrics.uot_masks import frame_mask_io as fmio

df = pd.read_csv(DATA_CSV)
print(f"Loaded CSV with columns: {list(df.columns)}")

# Get reference
ref_id = "20251112_H04_e01"
ref_frame = 39
ref_row = df[(df["embryo_id"] == ref_id) & (df["time_int"] == ref_frame)].iloc[0]

mask_ref = fmio.load_mask_from_rle_counts(
    rle_counts=ref_row["mask_rle"],
    height_px=int(ref_row["mask_height_px"]),
    width_px=int(ref_row["mask_width_px"]),
)
yolk_ref = None  # No yolk for now
um_per_px_ref = fmio._compute_um_per_pixel(ref_row)

print(f"Reference: {ref_id} frame {ref_frame}")
print(f"  Mask shape: {mask_ref.shape}")
print(f"  Mask pixels: {mask_ref.sum()}")
print(f"  Genotype: {ref_row['genotype']}")

# Get a wildtype target OR any target
print(f"\nUnique genotypes in data: {df['genotype'].unique()}")
wt_rows = df[df["genotype"].str.lower() == "wildtype"]
if len(wt_rows) == 0:
    print("No 'wildtype' rows, trying any genotype...")
    wt_rows = df  # Use all rows

print(f"\nAvailable samples: {len(wt_rows)} total")

# Pick one that's NOT the reference
wt_rows = wt_rows[wt_rows["embryo_id"] != ref_id]
if len(wt_rows) == 0:
    raise ValueError("No non-reference embryos found!")
    
tgt_row = wt_rows.iloc[0]

mask_tgt = fmio.load_mask_from_rle_counts(
    rle_counts=tgt_row["mask_rle"],
    height_px=int(tgt_row["mask_height_px"]),
    width_px=int(tgt_row["mask_width_px"]),
)
yolk_tgt = None  # No yolk for now
um_per_px_tgt = fmio._compute_um_per_pixel(tgt_row)

print(f"\nTarget: {tgt_row['embryo_id']} frame {tgt_row['time_int']}")
print(f"  Mask shape: {mask_tgt.shape}")
print(f"  Mask pixels: {mask_tgt.sum()}")

# Ru OT
print("\n" + "="*70)
print("Running OT with current p0_ot_maps.py::run_single_ot() config")
print("="*70)

result = run_single_ot(
    mask_ref=mask_ref,
    mask_target=mask_tgt,
    sample_id="debug_test",
    raw_um_per_px_ref=um_per_px_ref,
    raw_um_per_px_tgt=um_per_px_tgt,
    yolk_ref=yolk_ref,
    yolk_tgt=yolk_tgt,
)

print(f"\nResult:")
print(f"  Total cost (C): {result.total_cost_C}")
print(f"  cost_density shape: {result.cost_density.shape}")
print(f"  cost_density mean: {result.cost_density.mean()}")
print(f"  cost_density max: {result.cost_density.max()}")
print(f"  cost_density nonzero pixels: {(result.cost_density > 0).sum()}")
print(f"  cost_density nonzero %: {100 * (result.cost_density > 0).sum() / result.cost_density.size:.1f}%")

print("\n" + "="*70)
print("DIAGNOSIS:")
print("="*70)

if result.total_cost_C == 0:
    print("❌ ZERO COST - This is the bug!")
    print("\nPossible causes:")
    print("1. Masks are identical (unlikely - they're different embryos)")
    print("2. Relaxation parameters too high (OT prefers create/destroy over transport)")
    print("3. Canonical alignment makes them identical (orientation/size normalization)")
    print("4. Bug in cost extraction from UOTResult")
    
    if (result.cost_density == 0).all():
        print("\n⚠️  ALL cost_density pixels are zero!")
        print("This suggests the OT solver returned zero cost, not just a rasterization issue")
else:
    print(f"✅ NONZERO COST: {result.total_cost_C:.6e}")
    if (result.cost_density > 0).sum() < 1000:
        print(f"⚠️  But only {(result.cost_density > 0).sum()} nonzero pixels")
        print("This suggests sparse rasterization (downsampling artifact)")

