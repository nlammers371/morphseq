"""
Verify that preprocessing in test script matches batch processing pipeline.

Loads an embryo from the CSV and compares:
1. Centerline from batch processing (curvature_arrays CSV)
2. Centerline from test script preprocessing

Should be identical if preprocessing matches.
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle
from segmentation_sandbox.scripts.body_axis_analysis.centerline_extraction import extract_centerline
from segmentation_sandbox.scripts.utils.mask_cleaning import clean_embryo_mask

# Test embryo
snip_id = "20251017_combined_A02_e01_t0064"  # A02 embryo for testing

# Load from original CSV
metadata_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build06_output/df03_final_output_with_latents_20251017_combined.csv")
df = pd.read_csv(metadata_path)
embryo_row = df[df['snip_id'] == snip_id].iloc[0]

# Decode mask
mask = decode_mask_rle({
    'size': [int(embryo_row['mask_height_px']), int(embryo_row['mask_width_px'])],
    'counts': embryo_row['mask_rle']
})

um_per_pixel = embryo_row['height_um'] / int(embryo_row['mask_height_px'])

print(f"Testing embryo: {snip_id}")
print(f"Mask shape: {mask.shape}")
print(f"Mask area: {mask.sum():,} px")
print(f"um_per_pixel: {um_per_pixel:.4f}")
print()

# Method 1: Replicate batch processing pipeline
print("="*80)
print("METHOD 1: Batch Processing Pipeline (process_curvature_batch.py)")
print("="*80)

# Step 1: Clean mask
cleaned_mask, cleaning_stats = clean_embryo_mask(mask, verbose=False)
print(f"After cleaning: {cleaned_mask.sum():,} px ({cleaned_mask.sum()/mask.sum()*100:.1f}%)")

# Step 2: Extract centerline (which internally applies gaussian preprocessing)
spline_x_batch, spline_y_batch, curvature_batch, arc_length_batch = extract_centerline(
    cleaned_mask,
    method='geodesic',
    um_per_pixel=um_per_pixel,
    bspline_smoothing=5.0
)

print(f"Centerline points: {len(spline_x_batch)}")
print(f"Total length: {arc_length_batch[-1]:.2f} um" if len(arc_length_batch) > 0 else "FAILED")
print(f"Mean curvature: {np.mean(np.abs(curvature_batch)):.4f}" if len(curvature_batch) > 0 else "FAILED")
print()

# Method 2: Load from curvature arrays CSV
print("="*80)
print("METHOD 2: From curvature_arrays CSV (batch processing output)")
print("="*80)

arrays_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/body_axis/arrays/curvature_arrays_20251017_combined.csv")
if arrays_path.exists():
    arrays_df = pd.read_csv(arrays_path)
    arrays_row = arrays_df[arrays_df['snip_id'] == snip_id]

    if len(arrays_row) > 0:
        arrays_row = arrays_row.iloc[0]

        # Parse JSON arrays
        centerline_x_csv = np.array(json.loads(arrays_row['centerline_x']))
        centerline_y_csv = np.array(json.loads(arrays_row['centerline_y']))
        curvature_csv = np.array(json.loads(arrays_row['curvature_values']))
        arc_length_csv = np.array(json.loads(arrays_row['arc_length_values']))

        print(f"Centerline points: {len(centerline_x_csv)}")
        print(f"Total length: {arc_length_csv[-1]:.2f} um" if len(arc_length_csv) > 0 else "FAILED")
        print(f"Mean curvature: {np.mean(np.abs(curvature_csv)):.4f}" if len(curvature_csv) > 0 else "FAILED")
        print()

        # Compare
        print("="*80)
        print("COMPARISON")
        print("="*80)

        # Check if they match
        if len(centerline_x_csv) == len(spline_x_batch):
            max_diff_x = np.max(np.abs(centerline_x_csv - spline_x_batch))
            max_diff_y = np.max(np.abs(centerline_y_csv - spline_y_batch))
            print(f"✓ Same number of points: {len(centerline_x_csv)}")
            print(f"  Max difference in x: {max_diff_x:.6f} pixels")
            print(f"  Max difference in y: {max_diff_y:.6f} pixels")

            if max_diff_x < 0.001 and max_diff_y < 0.001:
                print("  → IDENTICAL! Preprocessing matches perfectly.")
            else:
                print("  → DIFFERENT! Preprocessing may differ.")
        else:
            print(f"✗ Different number of points: CSV={len(centerline_x_csv)}, Batch={len(spline_x_batch)}")
            print("  → Preprocessing likely differs significantly.")
    else:
        print(f"Embryo {snip_id} not found in curvature arrays CSV!")
else:
    print(f"Curvature arrays CSV not found: {arrays_path}")

print()
print("="*80)
print("VISUALIZATION")
print("="*80)

# Create visualization
output_dir = Path(__file__).parent / "preprocessing_param_sweep"
output_dir.mkdir(exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Plot 1: Original mask with batch processing centerline
ax = axes[0]
ax.imshow(mask, cmap='gray', alpha=0.7)
if len(spline_x_batch) > 0:
    ax.plot(spline_x_batch, spline_y_batch, 'r-', linewidth=2.5, label='Centerline')
    ax.plot(spline_x_batch[0], spline_y_batch[0], 'go', markersize=10, label='Head')
    ax.plot(spline_x_batch[-1], spline_y_batch[-1], 'bo', markersize=10, label='Tail')
ax.set_title(f"Original Mask + Batch Centerline\n{snip_id}", fontsize=12, fontweight='bold')
ax.legend()
ax.axis('off')

# Plot 2: Cleaned mask with centerline
ax = axes[1]
ax.imshow(cleaned_mask, cmap='gray', alpha=0.7)
if len(spline_x_batch) > 0:
    ax.plot(spline_x_batch, spline_y_batch, 'r-', linewidth=2.5, label='Centerline')
    ax.plot(spline_x_batch[0], spline_y_batch[0], 'go', markersize=10, label='Head')
    ax.plot(spline_x_batch[-1], spline_y_batch[-1], 'bo', markersize=10, label='Tail')
ax.set_title(f"Cleaned Mask + Centerline\nAfter 5-step cleaning", fontsize=12, fontweight='bold')
ax.legend()
ax.axis('off')

plt.tight_layout()
output_path = output_dir / f"verify_preprocessing_{snip_id.replace('/', '_')}.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved visualization: {output_path}")
plt.close()

print()
print("="*80)
print("CONCLUSION")
print("="*80)
print("If the preprocessing matches, both methods should produce identical centerlines.")
print("Small differences (<0.001 px) are acceptable due to floating point precision.")
print(f"\nVisualization saved to: {output_path}")
