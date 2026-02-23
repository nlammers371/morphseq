"""
Check H07 flagging at early stages (predicted_stage_hpf < 50).

H07 appears to be too small at early stages and should be flagged.
This script validates that ALL frames < 50 hpf are properly flagged.

Usage:
    conda activate segmentations_grounded_sam
    python check_h07_early_stages.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
OUTPUT_FILE = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build04_output/qc_staged_20250711.csv")
TEST_EMBRYO = "20250711_H07_e01"
STAGE_THRESHOLD = 50.0  # Check frames below this stage

print("=" * 80)
print("H07 EARLY STAGE FLAGGING VALIDATION")
print("=" * 80)

# Load results
print(f"\n📊 Loading results from: {OUTPUT_FILE}")
df = pd.read_csv(OUTPUT_FILE)
print(f"✓ Loaded {len(df)} rows")

# Filter for H07
print(f"\n🔍 Filtering for {TEST_EMBRYO}...")
h07 = df[df['embryo_id'] == TEST_EMBRYO].copy()
print(f"✓ Found {len(h07)} frames")

if len(h07) == 0:
    print(f"❌ ERROR: {TEST_EMBRYO} not found in results!")
    exit(1)

# Filter for early stages (< 50 hpf)
print(f"\n🎯 Checking frames with predicted_stage_hpf < {STAGE_THRESHOLD}...")
h07_early = h07[h07['predicted_stage_hpf'] < STAGE_THRESHOLD].copy()
print(f"✓ Found {len(h07_early)} frames < {STAGE_THRESHOLD} hpf")

if len(h07_early) == 0:
    print(f"⚠️  No frames found below {STAGE_THRESHOLD} hpf")
    exit(0)

# Check flagging status
print(f"\n📋 Flagging summary for frames < {STAGE_THRESHOLD} hpf:")
n_flagged = h07_early['sa_outlier_flag'].sum()
n_total = len(h07_early)
pct_flagged = 100 * n_flagged / n_total

print(f"   Total frames: {n_total}")
print(f"   Flagged: {n_flagged} ({pct_flagged:.1f}%)")
print(f"   Not flagged: {n_total - n_flagged} ({100 - pct_flagged:.1f}%)")

# Show details of unflagged frames
if n_flagged < n_total:
    print(f"\n⚠️  {n_total - n_flagged} frames are NOT flagged!")
    print("\nUnflagged frames:")
    unflagged = h07_early[~h07_early['sa_outlier_flag']]

    # Show key columns
    cols = ['snip_id', 'frame_index', 'predicted_stage_hpf', 'surface_area_um', 'sa_outlier_flag']
    print(unflagged[cols].to_string(index=False))

    # Calculate SA statistics for unflagged frames
    print(f"\n📊 Surface area statistics (unflagged frames):")
    print(f"   Min SA: {unflagged['surface_area_um'].min():.0f} µm")
    print(f"   Max SA: {unflagged['surface_area_um'].max():.0f} µm")
    print(f"   Mean SA: {unflagged['surface_area_um'].mean():.0f} µm")
    print(f"   Median SA: {unflagged['surface_area_um'].median():.0f} µm")

    # Check reference values at these stages
    if 'ref_p5' in df.columns:
        print(f"\n📈 Reference p5 values at these stages:")
        print(f"   Min p5: {unflagged['ref_p5'].min():.0f} µm")
        print(f"   Max p5: {unflagged['ref_p5'].max():.0f} µm")

        # Calculate ratio
        unflagged['sa_to_p5_ratio'] = unflagged['surface_area_um'] / unflagged['ref_p5']
        print(f"\n📊 SA / p5 ratios:")
        print(f"   Min ratio: {unflagged['sa_to_p5_ratio'].min():.2f}")
        print(f"   Max ratio: {unflagged['sa_to_p5_ratio'].max():.2f}")
        print(f"   Mean ratio: {unflagged['sa_to_p5_ratio'].mean():.2f}")

        # Check how many are below various thresholds
        for k in [0.5, 0.6, 0.7, 0.8]:
            n_below = (unflagged['sa_to_p5_ratio'] < k).sum()
            print(f"   Frames with SA < {k}×p5: {n_below} / {len(unflagged)}")

else:
    print(f"\n✅ ALL {n_total} frames < {STAGE_THRESHOLD} hpf are flagged!")

# Overall H07 summary
print(f"\n" + "=" * 80)
print("OVERALL H07 SUMMARY")
print("=" * 80)

all_flagged = h07['sa_outlier_flag'].sum()
all_total = len(h07)
print(f"Total H07 frames: {all_total}")
print(f"Flagged: {all_flagged} ({100*all_flagged/all_total:.1f}%)")
print(f"Not flagged: {all_total - all_flagged} ({100*(all_total-all_flagged)/all_total:.1f}%)")

# Stage breakdown
print(f"\nFlagging by stage range:")
stage_bins = [0, 30, 40, 50, 60, 70, 80, 100, 150]
for i in range(len(stage_bins) - 1):
    low, high = stage_bins[i], stage_bins[i+1]
    mask = (h07['predicted_stage_hpf'] >= low) & (h07['predicted_stage_hpf'] < high)
    if mask.sum() > 0:
        n_in_bin = mask.sum()
        n_flagged_in_bin = h07.loc[mask, 'sa_outlier_flag'].sum()
        pct = 100 * n_flagged_in_bin / n_in_bin
        print(f"  {low:3.0f}-{high:3.0f} hpf: {n_flagged_in_bin:3d}/{n_in_bin:3d} flagged ({pct:5.1f}%)")

# Recommendation
print(f"\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

if n_flagged == n_total:
    print("✅ Current threshold (k_lower=0.7) is working correctly for early stages!")
else:
    unflagged_pct = 100 * (n_total - n_flagged) / n_total
    print(f"⚠️  Current threshold missing {unflagged_pct:.1f}% of early stage frames")

    # Estimate needed k_lower
    if 'sa_to_p5_ratio' in locals():
        max_unflagged_ratio = unflagged['sa_to_p5_ratio'].max()
        recommended_k = max_unflagged_ratio * 0.95  # 5% buffer
        print(f"\n💡 To catch all early frames, recommend k_lower ≤ {recommended_k:.2f}")
        print(f"   (Current: k_lower = 0.7)")

print("\n✓ Done!")
