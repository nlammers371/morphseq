"""
Debug B05 Mask Cleaning - Step by Step Visualization

Shows what happens at each cleaning step to understand where the tail gets cut off.
"""

import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import morphology, measure

from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle


def debug_cleaning_steps(mask: np.ndarray):
    """
    Apply cleaning step-by-step and visualize each stage.
    """
    steps = []

    # Original
    steps.append(('0_Original', mask.copy()))

    # Step 1: Remove debris
    labeled = measure.label(mask)
    total_area = mask.sum()
    threshold = 0.10 * total_area

    if labeled.max() > 1:
        component_areas = [(i, np.sum(labeled == i)) for i in range(1, labeled.max() + 1)]
        keep_labels = [label for label, area in component_areas if area >= threshold]

        if len(keep_labels) == 0:
            keep_labels = [max(component_areas, key=lambda x: x[1])[0]]

        mask_filtered = np.zeros_like(mask, dtype=bool)
        for label in keep_labels:
            mask_filtered |= (labeled == label)

        steps.append(('1_After_Debris_Removal', mask_filtered.copy()))
        mask = mask_filtered

    # Step 2: Closing iterations
    props_temp = measure.regionprops(measure.label(mask))[0]
    perimeter_temp = props_temp.perimeter
    closing_radius = max(5, int(perimeter_temp / 100))

    closed = mask.copy()
    max_iterations = 5
    max_radius = 50

    for iteration in range(max_iterations):
        selem_close = morphology.disk(closing_radius)
        closed = morphology.binary_closing(closed, selem_close)

        steps.append((f'2_Closing_iter{iteration+1}_r{closing_radius}', closed.copy()))

        closed_labeled = measure.label(closed)
        n_components = closed_labeled.max()

        if n_components == 1:
            break

        closing_radius = min(closing_radius + 5, max_radius)
        if closing_radius >= max_radius:
            break

    mask = closed

    # Step 3: Fill holes
    filled = ndimage.binary_fill_holes(mask)
    steps.append(('3_After_Fill_Holes', filled.copy()))

    # Step 4: Opening - THIS IS WHERE THE PROBLEM HAPPENS
    props = measure.regionprops(measure.label(filled))[0]
    perimeter = props.perimeter
    # Use perimeter/150 for gentler smoothing (was /100)
    adaptive_radius = max(3, int(perimeter / 150))

    # Show intermediate steps of opening
    selem_open = morphology.disk(adaptive_radius)

    # Erosion step
    eroded = morphology.binary_erosion(filled, selem_open)
    steps.append((f'4a_After_Erosion_r{adaptive_radius}', eroded.copy()))

    # Dilation step
    opened = morphology.binary_dilation(eroded, selem_open)
    steps.append((f'4b_After_Dilation_r{adaptive_radius}', opened.copy()))

    # Step 5: Final component check
    final_labeled = measure.label(opened)
    if final_labeled.max() > 1:
        component_sizes = [(i, np.sum(final_labeled == i)) for i in range(1, final_labeled.max() + 1)]
        largest_label = max(component_sizes, key=lambda x: x[1])[0]
        final_mask = (final_labeled == largest_label)
        steps.append(('5_Keep_Largest_Component', final_mask.copy()))
    else:
        steps.append(('5_Final_Single_Component', opened.copy()))

    return steps


def main():
    """Main execution."""
    snip_id = '20251017_part2_B05_e01_t0037'

    csv_path = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build06_output/df03_final_output_with_latents_20251017_part2.csv")
    output_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251024")

    print("="*80)
    print(f"DEBUG: {snip_id}")
    print("="*80)

    # Load data
    df = pd.read_csv(csv_path)
    row = df[df['snip_id'] == snip_id].iloc[0]

    # Decode mask
    mask = decode_mask_rle({
        'size': [int(row['mask_height_px']), int(row['mask_width_px'])],
        'counts': row['mask_rle']
    })
    mask = np.ascontiguousarray(mask.astype(np.uint8))

    print(f"Original mask area: {mask.sum():,} px")

    # Debug cleaning steps
    steps = debug_cleaning_steps(mask)

    # Create visualization
    n_steps = len(steps)
    n_cols = 4
    n_rows = int(np.ceil(n_steps / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten()

    for i, (step_name, step_mask) in enumerate(steps):
        axes[i].imshow(step_mask, cmap='gray')

        # Count components
        labeled = measure.label(step_mask)
        n_comp = labeled.max()
        area = step_mask.sum()

        axes[i].set_title(f'{step_name}\nArea: {area:,} px, Components: {n_comp}',
                         fontsize=10)
        axes[i].axis('off')

        print(f"\n{step_name}:")
        print(f"  Area: {area:,} px")
        print(f"  Components: {n_comp}")

        if n_comp > 1:
            component_sizes = [np.sum(labeled == j) for j in range(1, n_comp + 1)]
            print(f"  Component sizes: {component_sizes}")

    # Hide unused subplots
    for i in range(len(steps), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    output_path = output_dir / "debug_b05_cleaning_steps.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n{'='*80}")
    print(f"Saved: {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()
