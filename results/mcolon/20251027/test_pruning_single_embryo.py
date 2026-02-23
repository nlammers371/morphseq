"""
Quick test of skeleton pruning on a single embryo.

Tests the pruning logic and visualizes results before running full analysis.
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt

from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle
from segmentation_sandbox.scripts.utils.mask_cleaning import clean_embryo_mask
from segmentation_sandbox.scripts.utils.bodyaxis_spline_utils import prune_skeleton_for_geodesic

# Test case: H07_e01_t0008 - known to have fin problem
test_snip_id = "20251017_part2_H07_e01_t0008"

# Load metadata
metadata_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build06_output/df03_final_output_with_latents_20251017_part2.csv")
df = pd.read_csv(metadata_path)

# Get test embryo
row = df[df['snip_id'] == test_snip_id].iloc[0]

# Decode mask
mask = decode_mask_rle({
    'size': [int(row['mask_height_px']), int(row['mask_width_px'])],
    'counts': row['mask_rle']
})

# Clean mask
cleaned_mask, _ = clean_embryo_mask(mask, verbose=False)

# Get skeleton
skeleton = skeletonize(cleaned_mask)

# Test different parameter combinations
# We'll vary each parameter to see its effect
# Format: (length_fraction, width_fraction, aspect_ratio, label)

# Total skeleton length for reference
total_skeleton_length = 0
skeleton_coords = np.argwhere(skeleton)
if len(skeleton_coords) > 1:
    from segmentation_sandbox.scripts.utils.bodyaxis_spline_utils import _sort_coords_along_path
    sorted_coords = _sort_coords_along_path(skeleton_coords)
    for i in range(len(sorted_coords) - 1):
        total_skeleton_length += np.linalg.norm(sorted_coords[i+1] - sorted_coords[i])
    print(f"Total skeleton length: {total_skeleton_length:.1f} px")

test_configs = [
    # Format: (length_frac, width_frac, aspect, max_angle, label)

    # Vary max angle (how sharp is "too sharp")
    (0.10, 0.50, 3.0, 45, 'Angle 45°'),
    (0.10, 0.50, 3.0, 60, 'Angle 60°'),
    (0.10, 0.50, 3.0, 75, 'Angle 75°'),

    # Vary length fraction
    (0.05, 0.50, 3.0, 60, '5% Length'),
    (0.15, 0.50, 3.0, 60, '15% Length'),

    # Combined variations
    (0.10, 0.40, 3.0, 60, '40% Width'),
    (0.10, 0.50, 5.0, 60, 'Aspect 5.0'),
]

# Create a 4x3 grid to show all variations
# Row 0: original images (3 cells)
# Rows 1-3: pruning results (9 cells, using 7)
fig, axes = plt.subplots(4, 3, figsize=(15, 18))

# Top row: original mask and skeleton
axes[0, 0].imshow(cleaned_mask, cmap='gray')
axes[0, 0].set_title('Cleaned Mask', fontsize=11, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(skeleton, cmap='gray')
axes[0, 1].set_title(f'Original Skeleton\n({skeleton.sum()} pixels)', fontsize=11, fontweight='bold')
axes[0, 1].axis('off')

# Distance transform
distance_map = distance_transform_edt(cleaned_mask)
axes[0, 2].imshow(distance_map, cmap='viridis')
axes[0, 2].set_title('Distance Transform\n(Local Width/2)', fontsize=11, fontweight='bold')
axes[0, 2].axis('off')

# Plot each pruning configuration
for idx, (length_frac, width_frac, aspect, max_angle, label) in enumerate(test_configs):
    row = (idx + 3) // 3  # Start from row 1
    col = (idx + 3) % 3

    pruned_skel, stats = prune_skeleton_for_geodesic(
        skeleton, cleaned_mask,
        min_branch_length_fraction=length_frac,
        min_width_fraction=width_frac,
        min_aspect_ratio=aspect,
        max_branch_angle=max_angle
    )

    pruned_coords = np.argwhere(pruned_skel)
    removed_coords = np.argwhere(skeleton & ~pruned_skel)

    # Show mask with kept (green) and removed (red) skeleton overlaid
    axes[row, col].imshow(cleaned_mask, cmap='gray', alpha=0.4)

    # Show removed skeleton (red)
    if len(removed_coords) > 0:
        axes[row, col].scatter(removed_coords[:, 1], removed_coords[:, 0],
                              c='red', s=8, alpha=0.8, label='Removed', marker='x')

    # Show kept skeleton (green)
    if len(pruned_coords) > 0:
        axes[row, col].scatter(pruned_coords[:, 1], pruned_coords[:, 0],
                              c='green', s=6, alpha=0.7, label='Kept')

    # Title with parameters and results
    title = f'{label}\n'
    title += f'L={length_frac:.0%}, W={width_frac:.0%}, A={aspect:.1f}, θ={max_angle}°\n'
    title += f'{stats["n_branches_removed"]}/{stats["n_branches_analyzed"]} br removed '
    title += f'({stats["n_branches_removed_by_angle"]}∠ {stats["n_branches_removed_by_size"]}□)'

    axes[row, col].set_title(title, fontsize=8)
    axes[row, col].axis('off')
    axes[row, col].legend(loc='upper right', fontsize=6)

plt.tight_layout()
output_path = Path(__file__).parent / 'test_pruning_single_embryo.png'
plt.savefig(output_path, dpi=200, bbox_inches='tight')
print(f"Saved: {output_path}")

# Print statistics
print(f"\nTest embryo: {test_snip_id}")
print(f"Skeleton pixels: {skeleton.sum()}")
print(f"Total skeleton length (approx): {total_skeleton_length:.1f} px")

print("\n" + "="*80)
print("ADAPTIVE PRUNING PARAMETER SWEEP")
print("="*80)

print("\nReference values:")
# Get one set of stats to show reference values
ref_stats = prune_skeleton_for_geodesic(skeleton, cleaned_mask, 0.10, 0.50, 3.0, 60.0)[1]
print(f"  Embryo major axis length: {ref_stats['embryo_length']:.1f} px")
print(f"  Median skeleton width: {ref_stats['median_width']:.1f} px")
print(f"  Total branches detected: {ref_stats['n_branches_analyzed']}")
print(f"  Total skeleton length: {total_skeleton_length:.1f} px")

print("\nParameter sweep results:")
print("-" * 100)
print(f"{'Config':<12} {'L%':<6} {'W%':<6} {'Asp':<5} {'Ang':<5} {'Branches':<12} {'By Angle':<10} {'By Size':<10} {'Pixels Removed':<15}")
print("-" * 100)

for length_frac, width_frac, aspect, max_angle, label in test_configs:
    pruned_skel, stats = prune_skeleton_for_geodesic(
        skeleton, cleaned_mask,
        min_branch_length_fraction=length_frac,
        min_width_fraction=width_frac,
        min_aspect_ratio=aspect,
        max_branch_angle=max_angle
    )

    branches_str = f"{stats['n_branches_removed']}/{stats['n_branches_analyzed']}"
    angle_str = f"{stats['n_branches_removed_by_angle']}"
    size_str = f"{stats['n_branches_removed_by_size']}"
    pixels_str = f"{stats['removed_fraction']*100:.0f}%"

    print(f"{label:<12} {length_frac:<6.0%} {width_frac:<6.0%} {aspect:<5.1f} {max_angle:<5}° "
          f"{branches_str:<12} {angle_str:<10} {size_str:<10} {pixels_str:<15}")

print("-" * 80)

print("\nKey Insights:")
print(f"  • 10% of embryo length = {ref_stats['embryo_length'] * 0.10:.1f} px")
print(f"  • 10% of total skeleton = {total_skeleton_length * 0.10:.1f} px")
print(f"  • Ratio (embryo/skeleton): {ref_stats['embryo_length'] / total_skeleton_length:.2f}x")
print(f"  • Using 10% of embryo = ~{(ref_stats['embryo_length'] * 0.10) / total_skeleton_length:.1%} of skeleton length")

print("\n" + "="*80)
print("Test complete!")
print("="*80)
