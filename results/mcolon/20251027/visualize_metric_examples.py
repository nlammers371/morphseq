"""
Visualize Real Examples from Morphology Distributions

Shows actual mask examples for different metric ranges to validate
whether our detection thresholds make sense.

Comparisons:
1. Perimeter-area ratio: 7-10 (problematic?) vs <7 (normal?)
2. Solidity: <0.65 (problematic?) vs >0.65 (normal?)

Outputs organized in subdirectories to avoid clutter.
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


def plot_mask_grid(masks_data: list, title: str, output_path: Path, n_cols: int = 5):
    """
    Plot grid of masks with their metrics displayed.

    Args:
        masks_data: List of dicts with keys: 'mask', 'snip_id', 'metric_value', 'metric_name'
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
        metric_value = data['metric_value']
        metric_name = data['metric_name']

        # Display mask
        ax.imshow(mask, cmap='gray')

        # Title with full snip_id and metric value
        title_text = f"{snip_id}\n{metric_name}: {metric_value:.3f}"
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


def visualize_perimeter_area_ratio_comparison(morphology_df: pd.DataFrame,
                                                dataset_dir: Path,
                                                output_dir: Path,
                                                n_samples: int = 10):
    """
    Compare masks with different perimeter-area ratios.

    Group 1 (problematic?): ratio >= 8.5
    Group 2 (normal?): ratio < 8.5
    """
    print(f"\n{'='*80}")
    print("PERIMETER-AREA RATIO COMPARISON")
    print('='*80)

    # Filter to part1_sample only (exclude problem cases)
    normal_df = morphology_df[morphology_df['dataset'] == 'part1_sample'].copy()

    # Group 1: Ratio >= 8.5 (potentially problematic)
    group1 = normal_df[normal_df['perimeter_area_ratio'] >= 8.5]

    # Group 2: Ratio < 8.5 (normal)
    group2 = normal_df[normal_df['perimeter_area_ratio'] < 8.5]

    print(f"\nGroup 1 (ratio >=8.5): {len(group1)} masks")
    print(f"Group 2 (ratio <8.5):  {len(group2)} masks")

    # Sample randomly
    np.random.seed(42)
    sample1 = group1.sample(n=min(n_samples, len(group1)))
    sample2 = group2.sample(n=min(n_samples, len(group2)))

    # Load masks for Group 1
    print("\nLoading masks for Group 1 (ratio 7-10)...")
    masks_data1 = []
    for idx, row in sample1.iterrows():
        try:
            mask = get_mask_from_snip_id(row['snip_id'], dataset_dir)
            masks_data1.append({
                'mask': mask,
                'snip_id': row['snip_id'],
                'metric_value': row['perimeter_area_ratio'],
                'metric_name': 'Perim/Area Ratio'
            })
        except Exception as e:
            print(f"  Failed to load {row['snip_id']}: {e}")

    # Load masks for Group 2
    print("Loading masks for Group 2 (ratio <7)...")
    masks_data2 = []
    for idx, row in sample2.iterrows():
        try:
            mask = get_mask_from_snip_id(row['snip_id'], dataset_dir)
            masks_data2.append({
                'mask': mask,
                'snip_id': row['snip_id'],
                'metric_value': row['perimeter_area_ratio'],
                'metric_name': 'Perim/Area Ratio'
            })
        except Exception as e:
            print(f"  Failed to load {row['snip_id']}: {e}")

    # Create output subdirectory
    output_subdir = output_dir / "metric_examples" / "perimeter_area_ratio"
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Plot Group 1
    if len(masks_data1) > 0:
        plot_mask_grid(
            masks_data1,
            title="Perimeter-Area Ratio: ≥8.5 (Potentially Problematic?)",
            output_path=output_subdir / "group1_ratio_gte_8.5.png",
            n_cols=5
        )

    # Plot Group 2
    if len(masks_data2) > 0:
        plot_mask_grid(
            masks_data2,
            title="Perimeter-Area Ratio: <8.5 (Normal?)",
            output_path=output_subdir / "group2_ratio_lt_8.5.png",
            n_cols=5
        )

    print(f"\nOutputs saved to: {output_subdir}")


def visualize_solidity_comparison(morphology_df: pd.DataFrame,
                                   dataset_dir: Path,
                                   output_dir: Path,
                                   n_samples: int = 10):
    """
    Compare masks with different solidity values.

    Group 1 (problematic?): solidity < 0.5
    Group 2 (normal?): solidity >= 0.5
    """
    print(f"\n{'='*80}")
    print("SOLIDITY COMPARISON")
    print('='*80)

    # Filter to part1_sample only (exclude problem cases)
    normal_df = morphology_df[morphology_df['dataset'] == 'part1_sample'].copy()

    # Group 1: Solidity < 0.5 (potentially problematic)
    group1 = normal_df[normal_df['solidity'] < 0.5]

    # Group 2: Solidity >= 0.5 (normal)
    group2 = normal_df[normal_df['solidity'] >= 0.5]

    print(f"\nGroup 1 (solidity <0.5):  {len(group1)} masks")
    print(f"Group 2 (solidity >=0.5): {len(group2)} masks")

    # Sample randomly
    np.random.seed(42)
    sample1 = group1.sample(n=min(n_samples, len(group1)))
    sample2 = group2.sample(n=min(n_samples, len(group2)))

    # Load masks for Group 1
    print("\nLoading masks for Group 1 (solidity <0.65)...")
    masks_data1 = []
    for idx, row in sample1.iterrows():
        try:
            mask = get_mask_from_snip_id(row['snip_id'], dataset_dir)
            masks_data1.append({
                'mask': mask,
                'snip_id': row['snip_id'],
                'metric_value': row['solidity'],
                'metric_name': 'Solidity'
            })
        except Exception as e:
            print(f"  Failed to load {row['snip_id']}: {e}")

    # Load masks for Group 2
    print("Loading masks for Group 2 (solidity >0.65)...")
    masks_data2 = []
    for idx, row in sample2.iterrows():
        try:
            mask = get_mask_from_snip_id(row['snip_id'], dataset_dir)
            masks_data2.append({
                'mask': mask,
                'snip_id': row['snip_id'],
                'metric_value': row['solidity'],
                'metric_name': 'Solidity'
            })
        except Exception as e:
            print(f"  Failed to load {row['snip_id']}: {e}")

    # Create output subdirectory
    output_subdir = output_dir / "metric_examples" / "solidity"
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Plot Group 1
    if len(masks_data1) > 0:
        plot_mask_grid(
            masks_data1,
            title="Solidity: <0.5 (Potentially Problematic?)",
            output_path=output_subdir / "group1_solidity_lt_0.5.png",
            n_cols=5
        )

    # Plot Group 2
    if len(masks_data2) > 0:
        plot_mask_grid(
            masks_data2,
            title="Solidity: ≥0.5 (Normal?)",
            output_path=output_subdir / "group2_solidity_gte_0.5.png",
            n_cols=5
        )

    print(f"\nOutputs saved to: {output_subdir}")


def main():
    """Main execution."""
    results_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251027")
    dataset_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build06_output")

    # Load morphology results
    morphology_csv = results_dir / "when_to_compute_opening_cleaning" / "morphology_metrics_500samples.csv"

    if not morphology_csv.exists():
        print(f"ERROR: {morphology_csv} not found!")
        print("Please run analyze_morphology_distributions.py first.")
        return

    morphology_df = load_morphology_results(morphology_csv)

    print("="*80)
    print("VISUALIZE METRIC EXAMPLES FROM DISTRIBUTIONS")
    print("="*80)

    # Visualize perimeter-area ratio comparison
    visualize_perimeter_area_ratio_comparison(
        morphology_df,
        dataset_dir,
        results_dir,
        n_samples=10
    )

    # Visualize solidity comparison
    visualize_solidity_comparison(
        morphology_df,
        dataset_dir,
        results_dir,
        n_samples=10
    )

    print(f"\n{'='*80}")
    print("VISUALIZATION COMPLETE")
    print('='*80)
    print(f"\nOutputs organized in: {results_dir / 'metric_examples'}")


if __name__ == "__main__":
    main()
