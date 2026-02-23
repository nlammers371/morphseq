"""
Visualize Real Examples of Low Extent Values

Shows actual mask examples for extent < 0.3 to understand what
these shapes look like.

Low extent means the embryo occupies a smaller proportion of its
bounding box - could indicate very elongated or curved shapes.
"""

import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure
import warnings
warnings.filterwarnings('ignore')

from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle


def load_morphology_results(results_csv: Path):
    """Load precomputed morphology metrics."""
    print(f"Loading morphology results: {results_csv.name}")
    df = pd.read_csv(results_csv)
    return df


def get_mask_from_snip_id(snip_id: str, dataset_dir: Path):
    """
    Load mask for a given snip_id.

    Args:
        snip_id: Embryo identifier
        dataset_dir: Directory containing CSV files

    Returns:
        mask: Binary mask array
    """
    # Determine which dataset (part1 or part2)
    if 'part1' in snip_id:
        csv_path = dataset_dir / "df03_final_output_with_latents_20251017_part1.csv"
    elif 'part2' in snip_id:
        csv_path = dataset_dir / "df03_final_output_with_latents_20251017_part2.csv"
    else:
        raise ValueError(f"Cannot determine dataset for {snip_id}")

    # Load CSV
    df = pd.read_csv(csv_path)
    row = df[df['snip_id'] == snip_id]

    if len(row) == 0:
        raise ValueError(f"snip_id {snip_id} not found in {csv_path.name}")

    row = row.iloc[0]

    # Decode mask
    mask = decode_mask_rle({
        'size': [int(row['mask_height_px']), int(row['mask_width_px'])],
        'counts': row['mask_rle']
    })

    return np.ascontiguousarray(mask.astype(np.uint8))


def plot_mask_grid_with_bbox(masks_data: list, title: str, output_path: Path, n_cols: int = 5):
    """
    Plot grid of masks with bounding boxes overlaid and extent values.

    Args:
        masks_data: List of dicts with keys: 'mask', 'snip_id', 'extent'
        title: Overall title for the figure
        output_path: Where to save the figure
        n_cols: Number of columns in grid
    """
    n_masks = len(masks_data)
    n_rows = int(np.ceil(n_masks / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))

    # Handle single row case
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    axes = axes.flatten()

    for i, data in enumerate(masks_data):
        ax = axes[i]
        mask = data['mask']
        snip_id = data['snip_id']
        extent = data['extent']

        # Get bounding box
        regions = measure.regionprops(mask)
        if len(regions) > 0:
            bbox = regions[0].bbox  # (min_row, min_col, max_row, max_col)
            min_row, min_col, max_row, max_col = bbox
        else:
            min_row = min_col = max_row = max_col = 0

        # Display mask
        ax.imshow(mask, cmap='gray')

        # Overlay bounding box
        from matplotlib.patches import Rectangle
        rect = Rectangle((min_col, min_row), max_col - min_col, max_row - min_row,
                         linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        # Title with snip_id and extent
        title_text = f"{snip_id}\nExtent: {extent:.3f}"
        ax.set_title(title_text, fontsize=9, fontweight='bold')
        ax.axis('off')

    # Hide unused subplots
    for i in range(n_masks, len(axes)):
        axes[i].axis('off')

    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def visualize_low_extent_examples(morphology_df: pd.DataFrame,
                                   dataset_dir: Path,
                                   output_dir: Path,
                                   n_samples: int = 15):
    """
    Visualize examples of masks with extent < 0.3.

    Also shows some normal extent examples for comparison.
    """
    print(f"\n{'='*80}")
    print("LOW EXTENT EXAMPLES (< 0.3)")
    print('='*80)

    # Group 1: Extent < 0.3 (low extent - elongated/curved?)
    group1 = morphology_df[morphology_df['extent'] < 0.3]

    # Group 2: Normal extent (0.4-0.6) for comparison
    group2 = morphology_df[(morphology_df['extent'] >= 0.4) & (morphology_df['extent'] <= 0.6)]

    print(f"\nGroup 1 (extent <0.3):     {len(group1)} masks")
    print(f"Group 2 (extent 0.4-0.6):  {len(group2)} masks")

    # Sample randomly
    np.random.seed(42)
    sample1 = group1.sample(n=min(n_samples, len(group1)))
    sample2 = group2.sample(n=min(n_samples, len(group2)))

    # Load masks for Group 1
    print("\nLoading masks for Group 1 (extent <0.3)...")
    masks_data1 = []
    for idx, row in sample1.iterrows():
        try:
            mask = get_mask_from_snip_id(row['snip_id'], dataset_dir)
            masks_data1.append({
                'mask': mask,
                'snip_id': row['snip_id'],
                'extent': row['extent']
            })
        except Exception as e:
            print(f"  Failed to load {row['snip_id']}: {e}")

    # Load masks for Group 2
    print("Loading masks for Group 2 (extent 0.4-0.6)...")
    masks_data2 = []
    for idx, row in sample2.iterrows():
        try:
            mask = get_mask_from_snip_id(row['snip_id'], dataset_dir)
            masks_data2.append({
                'mask': mask,
                'snip_id': row['snip_id'],
                'extent': row['extent']
            })
        except Exception as e:
            print(f"  Failed to load {row['snip_id']}: {e}")

    # Create output subdirectory
    output_subdir = output_dir / "metric_examples" / "extent"
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Plot Group 1 (low extent)
    if len(masks_data1) > 0:
        plot_mask_grid_with_bbox(
            masks_data1,
            title="Extent: <0.3 (Low - Elongated/Curved Shapes)",
            output_path=output_subdir / "group1_extent_lt_0.3.png",
            n_cols=5
        )

    # Plot Group 2 (normal extent)
    if len(masks_data2) > 0:
        plot_mask_grid_with_bbox(
            masks_data2,
            title="Extent: 0.4-0.6 (Normal Range)",
            output_path=output_subdir / "group2_extent_0.4_to_0.6.png",
            n_cols=5
        )

    print(f"\nOutputs saved to: {output_subdir}")

    # Print some statistics
    print(f"\n{'='*80}")
    print("EXTENT STATISTICS")
    print('='*80)
    print(f"\nLow extent group (<0.3):")
    print(f"  Min:  {group1['extent'].min():.4f}")
    print(f"  Max:  {group1['extent'].max():.4f}")
    print(f"  Mean: {group1['extent'].mean():.4f}")
    print(f"  Std:  {group1['extent'].std():.4f}")

    print(f"\nNormal extent group (0.4-0.6):")
    print(f"  Min:  {group2['extent'].min():.4f}")
    print(f"  Max:  {group2['extent'].max():.4f}")
    print(f"  Mean: {group2['extent'].mean():.4f}")
    print(f"  Std:  {group2['extent'].std():.4f}")


def main():
    """Main execution."""
    results_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251027")
    dataset_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build06_output")

    # Load morphology results
    morphology_csv = results_dir / "when_to_compute_opening_cleaning" / "morphology_metrics_500samples.csv"

    if not morphology_csv.exists():
        print(f"ERROR: {morphology_csv} not found!")
        return

    morphology_df = load_morphology_results(morphology_csv)

    print("="*80)
    print("VISUALIZE LOW EXTENT EXAMPLES")
    print("="*80)

    # Visualize low extent examples
    visualize_low_extent_examples(
        morphology_df,
        dataset_dir,
        results_dir,
        n_samples=15
    )

    print(f"\n{'='*80}")
    print("VISUALIZATION COMPLETE")
    print('='*80)
    print(f"\nOutputs in: {results_dir / 'metric_examples' / 'extent'}")


if __name__ == "__main__":
    main()
