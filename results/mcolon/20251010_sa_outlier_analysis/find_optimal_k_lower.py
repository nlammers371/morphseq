"""
Find optimal k_lower to catch all H07 early stage frames.

Loads the reference curves, calculates SA/p5 ratios for H07 unflagged frames,
and recommends k_lower value.

Usage:
    conda activate segmentations_grounded_sam
    python find_optimal_k_lower.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
OUTPUT_FILE = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build04_output/qc_staged_20250711.csv")
REF_FILE = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/metadata/sa_reference_curves.csv")
TEST_EMBRYO = "20250711_H07_e01"
STAGE_THRESHOLD = 50.0

print("=" * 80)
print("FINDING OPTIMAL k_lower FOR H07 EARLY STAGES")
print("=" * 80)

# Load reference
print(f"\n📚 Loading reference curves: {REF_FILE}")
ref_df = pd.read_csv(REF_FILE)
print(f"✓ Loaded {len(ref_df)} stage bins")
print(f"  Stage range: {ref_df['stage_hpf'].min():.1f} - {ref_df['stage_hpf'].max():.1f} hpf")

# Load results
print(f"\n📊 Loading results: {OUTPUT_FILE}")
df = pd.read_csv(OUTPUT_FILE)
print(f"✓ Loaded {len(df)} rows")

# Filter for H07 early stages
h07 = df[df['embryo_id'] == TEST_EMBRYO].copy()
h07_early = h07[h07['predicted_stage_hpf'] < STAGE_THRESHOLD].copy()

print(f"\n🎯 Analyzing {TEST_EMBRYO} frames < {STAGE_THRESHOLD} hpf...")
print(f"  Total frames: {len(h07_early)}")

# Interpolate reference p5 values at each frame's stage
h07_early['ref_p5'] = np.interp(h07_early['predicted_stage_hpf'], ref_df['stage_hpf'], ref_df['p5'])
h07_early['ref_p50'] = np.interp(h07_early['predicted_stage_hpf'], ref_df['stage_hpf'], ref_df['p50'])
h07_early['ref_p95'] = np.interp(h07_early['predicted_stage_hpf'], ref_df['stage_hpf'], ref_df['p95'])

# Calculate ratios
h07_early['sa_to_p5'] = h07_early['surface_area_um'] / h07_early['ref_p5']
h07_early['sa_to_p50'] = h07_early['surface_area_um'] / h07_early['ref_p50']

print(f"\n📊 SA / p5 ratio statistics:")
print(f"  Min: {h07_early['sa_to_p5'].min():.3f}")
print(f"  Max: {h07_early['sa_to_p5'].max():.3f}")
print(f"  Mean: {h07_early['sa_to_p5'].mean():.3f}")
print(f"  Median: {h07_early['sa_to_p5'].median():.3f}")

# Find frames that are currently NOT flagged
unflagged = h07_early[~h07_early['sa_outlier_flag']].copy()
print(f"\n⚠️  Unflagged frames: {len(unflagged)} / {len(h07_early)}")

if len(unflagged) > 0:
    print(f"\n📈 SA / p5 ratios for unflagged frames:")
    print(f"  Min: {unflagged['sa_to_p5'].min():.3f}")
    print(f"  Max: {unflagged['sa_to_p5'].max():.3f}")
    print(f"  Mean: {unflagged['sa_to_p5'].mean():.3f}")
    print(f"  Median: {unflagged['sa_to_p5'].median():.3f}")

    # Show worst offenders (highest ratios still unflagged)
    print(f"\n🔍 Top 10 unflagged frames (highest SA/p5 ratio):")
    worst = unflagged.nlargest(10, 'sa_to_p5')[['frame_index', 'predicted_stage_hpf', 'surface_area_um', 'ref_p5', 'sa_to_p5']]
    for idx, row in worst.iterrows():
        print(f"  Frame {int(row['frame_index']):3d} @ {row['predicted_stage_hpf']:5.1f} hpf: "
              f"SA={row['surface_area_um']:8.0f}, p5={row['ref_p5']:8.0f}, ratio={row['sa_to_p5']:.3f}")

# Test different k_lower values
print(f"\n" + "=" * 80)
print("TESTING DIFFERENT k_lower VALUES")
print("=" * 80)

k_values = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

for k in k_values:
    threshold = k * h07_early['ref_p5']
    would_flag = h07_early['surface_area_um'] < threshold
    n_flagged = would_flag.sum()
    pct = 100 * n_flagged / len(h07_early)

    marker = "✓" if n_flagged == len(h07_early) else " "
    print(f"{marker} k_lower = {k:.1f}: {n_flagged:3d}/{len(h07_early)} flagged ({pct:5.1f}%)")

# Recommend optimal k_lower
max_ratio = h07_early['sa_to_p5'].max()
recommended_k = max_ratio * 1.05  # 5% buffer above max ratio

print(f"\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print(f"\nMax SA/p5 ratio in early H07 frames: {max_ratio:.3f}")
print(f"Recommended k_lower: {recommended_k:.2f} (with 5% buffer)")
print(f"Current k_lower: 0.7")

if recommended_k > 0.7:
    print(f"\n✅ Current k_lower = 0.7 is SUFFICIENT (< {max_ratio:.3f})")
    print(f"   All early H07 frames should be flagged")
    print(f"   Check for other issues (e.g., reference interpolation)")
else:
    print(f"\n⚠️  Current k_lower = 0.7 is TOO LENIENT")
    print(f"   Recommended: k_lower ≤ {recommended_k:.2f}")

    # Find smallest k that catches all
    for k in [0.65, 0.6, 0.55, 0.5, 0.45, 0.4]:
        if k <= recommended_k:
            print(f"   → Suggest k_lower = {k:.2f}")
            break

print("\n✓ Done!")
