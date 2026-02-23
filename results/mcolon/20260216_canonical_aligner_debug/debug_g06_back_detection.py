#!/usr/bin/env python3
"""Enhanced debug for G06 back detection - shows yolk ring diagnostics."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

MORPHSEQ_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MORPHSEQ_ROOT))
sys.path.insert(0, str(MORPHSEQ_ROOT / "src"))

from analyze.utils.coord.grids.canonical import CanonicalAligner, CanonicalGridConfig
from analyze.optimal_transport_morphometrics.uot_masks import frame_mask_io as fmio

# Load data
data_root = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground")
csv_path = MORPHSEQ_ROOT / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
df = pd.read_csv(csv_path)

row = df[(df['embryo_id'] == '20251113_G06_e01') & (df['frame_index'] == 14)].iloc[0]

# Load mask and yolk
mask = fmio.load_mask_from_rle_counts(
    rle_counts=row["mask_rle"],
    height_px=int(row["mask_height_px"]),
    width_px=int(row["mask_width_px"]),
)
yolk = fmio._load_build02_aux_mask(data_root, row, mask.shape, keyword="yolk")
um_per_px = fmio._compute_um_per_pixel(row)

print(f"=== 20251113_G06_e01 Back Detection Diagnostics ===\n")
print(f"Original mask: {mask.sum()} pixels")
print(f"Yolk mask: {yolk.sum()} pixels")

# Calculate yolk metrics
from scipy import ndimage
cy, cx = ndimage.center_of_mass(yolk)
yolk_area = float(yolk.sum())
yolk_radius = np.sqrt(max(yolk_area, 1.0) / np.pi)

# Ring parameters (from CanonicalAligner defaults)
yolk_ring_inner_k = 1.2
yolk_ring_outer_k = 3.0
r_in = yolk_ring_inner_k * yolk_radius
r_out = yolk_ring_outer_k * yolk_radius

print(f"\nYolk center: ({cy:.1f}, {cx:.1f})")
print(f"Yolk radius:  {yolk_radius:.1f} px")
print(f"Ring inner radius (1.2×): {r_in:.1f} px")
print(f"Ring outer radius (3.0×): {r_out:.1f} px")

# Check ring pixels
ys, xs = np.nonzero(mask > 0.5)
dy = ys - cy
dx = xs - cx
dist = np.sqrt(dx * dx + dy * dy)
ring = (dist >= r_in) & (dist <= r_out)
ring_pixels = ring.sum()

print(f"\nMask pixels in ring: {ring_pixels}")
print(f"Ring threshold: 200 pixels")
print(f"Ring method viable: {ring_pixels >= 200}")

# Visualize original orientation issue
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Original with yolk ring
ax = axes[0]
ax.imshow(mask, cmap='gray', origin='upper', alpha=0.7)
ax.imshow(yolk, cmap='Blues', origin='upper', alpha=0.3)

# Draw yolk rings
circle_inner = Circle((cx, cy), r_in, fill=False, edgecolor='yellow', linewidth=2, label='Inner ring')
circle_outer = Circle((cx, cy), r_out, fill=False, edgecolor='orange', linewidth=2, label='Outer ring')
ax.add_patch(circle_inner)
ax.add_patch(circle_outer)

# Show ring pixels
if ring_pixels > 0:
    ring_y = ys[ring]
    ring_x = xs[ring]
    ax.scatter(ring_x, ring_y, c='red', s=1, alpha=0.5, label=f'Ring pixels ({ring_pixels})')

ax.plot(cx, cy, 'g*', markersize=15, label='Yolk center')
ax.set_title(f'Original Mask + Yolk Ring\n(Ring pixels: {ring_pixels})', fontsize=12)
ax.legend(loc='upper right')
ax.axis('equal')

# Plot 2: Distance from yolk center
ax = axes[1]
if ys.size > 0:
    scatter = ax.scatter(xs, ys, c=dist, s=1, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, ax=ax, label='Distance from yolk center (px)')
ax.plot(cx, cy, 'r*', markersize=15, label='Yolk center')
ax.set_title('Distance from Yolk Center', fontsize=12)
ax.legend()
ax.axis('equal')
ax.invert_yaxis()

# Plot 3: Quantile-based back detection fallback
ax = axes[2]
ax.imshow(mask, cmap='gray', origin='upper', alpha=0.7)

# Simulate head detection (center of mass of yolk for head-end)
head_y, head_x = ndimage.center_of_mass(mask)
ax.plot(head_x, head_y, 'ro', markersize=10, label=f'Head (CoM)')

# Distance from head
dy_head = ys - head_y
dx_head = xs - head_x
dist_head = np.sqrt(dx_head * dx_head + dy_head * dy_head)
threshold = np.quantile(dist_head, 0.9)  # 90th percentile
far = dist_head >= threshold

if far.sum() > 0:
    far_y = ys[far]
    far_x = xs[far]
    back_y = far_y.mean()
    back_x = far_x.mean()
    
    ax.scatter(far_x, far_y, c='cyan', s=1, alpha=0.5, label=f'Far pixels (>90th %ile)')
    ax.plot(back_x, back_y, 'bo', markersize=10, label=f'Back (mean of far)')
    
    # Draw line
    ax.plot([head_x, back_x], [head_y, back_y], 'g--', linewidth=2, alpha=0.7)
    
    # Distance between head and back
    hb_dist = np.sqrt((back_x - head_x)**2 + (back_y - head_y)**2)
    ax.text(0.5, 0.95, f'Head-Back distance: {hb_dist:.1f} px', 
            transform=ax.transAxes, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.set_title('Quantile Fallback Method\n(90th percentile distance from head)', fontsize=12)
ax.legend()
ax.axis('equal')

plt.suptitle('G06 Back Detection Diagnostics', fontsize=14, fontweight='bold')
plt.tight_layout()

output_path = Path('scripts/output/mask_qc/g06_back_detection_debug.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved diagnostics to {output_path}")
