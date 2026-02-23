"""
Visualize curvature metrics to validate that they capture real embryo curvature.

This script:
1. Creates histograms of all curvature statistics
2. Displays top 5 highest and lowest curvature embryos with their masks and splines
3. Annotates each embryo with its snip_id and metric values
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import sys
from typing import Dict, Tuple, List
import warnings

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from segmentation_sandbox.scripts.body_axis_analysis import extract_centerline
from segmentation_sandbox.scripts.body_axis_analysis.curvature_metrics import get_baseline_coordinates
from segmentation_sandbox.scripts.utils.mask_cleaning import clean_embryo_mask
from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load curvature summary with simple metrics and full build metadata."""
    # Load the summary with simple metrics (now included in standard output)
    summary_path = project_root / "morphseq_playground/metadata/body_axis/summary/curvature_metrics_summary_20251017_combined.csv"
    build_path = project_root / "morphseq_playground/metadata/build06_output/df03_final_output_with_latents_20251017_combined.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    summary_df = pd.read_csv(summary_path)
    build_df = pd.read_csv(build_path)

    # Merge to get mask information (um_per_pixel already in summary_df)
    merged_df = summary_df.merge(build_df[['snip_id', 'exported_mask_path', 'mask_rle',
                                           'mask_height_px', 'mask_width_px']],
                                 on='snip_id', how='left')

    return summary_df, merged_df


def load_mask_and_spline(row: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load mask and compute centerline spline for a given embryo."""
    # Try to load from exported path first
    mask_path = row['exported_mask_path']
    if pd.notna(mask_path) and Path(mask_path).exists():
        mask = np.load(mask_path)
    else:
        # Decode from RLE using proper mask_utils
        rle_dict = {
            'counts': row['mask_rle'],
            'size': [int(row['mask_height_px']), int(row['mask_width_px'])]
        }
        mask = decode_mask_rle(rle_dict)

    # Ensure mask is 2D - more aggressive handling
    while mask.ndim > 2:
        mask = mask.squeeze()
        if mask.ndim > 2:
            # If squeeze didn't work, take first slice along last axis
            mask = mask[..., 0]

    # Final check - if still not 2D, force it
    if mask.ndim > 2:
        mask = mask.reshape(mask.shape[-2:])
    elif mask.ndim < 2:
        raise ValueError(f"Mask has only {mask.ndim} dimension(s), cannot process")

    # Convert to boolean
    mask = mask.astype(bool)

    # Clean mask
    mask_cleaned = clean_embryo_mask(mask)

    # Ensure cleaned mask is still 2D (clean_embryo_mask might return tuple)
    if isinstance(mask_cleaned, tuple):
        mask_cleaned = mask_cleaned[0]  # Take just the mask, not the stats

    # Double-check mask_cleaned is 2D
    while mask_cleaned.ndim > 2:
        mask_cleaned = mask_cleaned.squeeze()
        if mask_cleaned.ndim > 2:
            mask_cleaned = mask_cleaned[..., 0]

    if mask_cleaned.ndim > 2:
        mask_cleaned = mask_cleaned.reshape(mask_cleaned.shape[-2:])

    # Extract centerline
    um_per_pixel = row['um_per_pixel'] if 'um_per_pixel' in row else 1.0
    try:
        spline_x, spline_y, curvature, arc_length = extract_centerline(
            mask_cleaned,
            method='geodesic',
            um_per_pixel=um_per_pixel,
            return_intermediate=False
        )
        return mask, spline_x, spline_y
    except Exception as e:
        print(f"Error processing {row['snip_id']}: {e}")
        print(f"  Mask shape: {mask.shape}, cleaned shape: {mask_cleaned.shape}")
        return mask, None, None


def plot_histograms(summary_df: pd.DataFrame, output_dir: Path):
    """Create histograms for all curvature metrics including simple metrics."""
    # Original metrics
    original_metrics = {
        'mean_curvature_per_um': 'Mean Curvature (1/μm)',
        'std_curvature_per_um': 'Std Curvature (1/μm)',
        'max_curvature_per_um': 'Max Curvature (1/μm)',
        'total_length_um': 'Total Length (μm)'
    }

    # Simple metrics
    simple_metrics = {
        'baseline_deviation_um': 'Baseline Deviation (μm)',
        'arc_length_ratio': 'Arc-Length Ratio',
        'keypoint_deviation_mid_um': 'Midpoint Deviation (μm)',
        'max_baseline_deviation_um': 'Max Baseline Deviation (μm)'
    }

    # Plot original metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (metric, label) in enumerate(original_metrics.items()):
        ax = axes[idx]
        data = summary_df[metric].dropna()

        ax.hist(data, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Distribution of {label}', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add statistics
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
        ax.axvline(median_val, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_val:.4f}')
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'curvature_histograms_original.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved original metric histograms to {output_dir / 'curvature_histograms_original.png'}")

    # Plot simple metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (metric, label) in enumerate(simple_metrics.items()):
        ax = axes[idx]
        data = summary_df[metric].dropna()

        ax.hist(data, bins=50, edgecolor='black', alpha=0.7, color='coral')
        ax.set_xlabel(label, fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Distribution of {label}', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add statistics
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
        ax.axvline(median_val, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_val:.4f}')
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'curvature_histograms_simple.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved simple metric histograms to {output_dir / 'curvature_histograms_simple.png'}")


def plot_embryo_comparison(merged_df: pd.DataFrame, output_dir: Path,
                          metric: str = 'mean_curvature_per_um', n_top: int = 5):
    """Plot top and bottom embryos by a given metric."""
    # Filter successful analyses
    success_df = merged_df[merged_df['success'] == True].copy()

    # Sort by metric
    success_df = success_df.sort_values(metric, ascending=False)

    # Get top and bottom n
    top_embryos = success_df.head(n_top)
    bottom_embryos = success_df.tail(n_top)

    # Create figure: 2 rows (top/bottom), n_top columns
    fig, axes = plt.subplots(2, n_top, figsize=(4 * n_top, 8))

    # If only one embryo, axes won't be 2D
    if n_top == 1:
        axes = axes.reshape(2, 1)

    for row_idx, (embryo_set, title_prefix) in enumerate([(top_embryos, 'Highest'),
                                                           (bottom_embryos, 'Lowest')]):
        for col_idx, (_, row) in enumerate(embryo_set.iterrows()):
            # Load mask and spline
            mask, spline_x, spline_y = load_mask_and_spline(row)

            # Get the axis for this embryo
            ax = axes[row_idx, col_idx]

            # Plot mask with spline overlay
            ax.imshow(mask, cmap='gray', alpha=0.7)

            if spline_x is not None and spline_y is not None:
                # Plot centerline
                ax.plot(spline_x, spline_y, 'r-', linewidth=2.5, label='Centerline')

                # Plot baseline (head-to-tail straight line)
                baseline_x, baseline_y = get_baseline_coordinates(spline_x, spline_y)
                ax.plot(baseline_x, baseline_y, 'c--', linewidth=2, label='Baseline', alpha=0.8)

                # Plot head and tail markers
                ax.plot(spline_x[0], spline_y[0], 'go', markersize=10, label='Head',
                       markeredgecolor='white', markeredgewidth=1.5)
                ax.plot(spline_x[-1], spline_y[-1], 'bo', markersize=10, label='Tail',
                       markeredgecolor='white', markeredgewidth=1.5)
                ax.legend(fontsize=8, loc='upper right')

            # Title with embryo info
            title = f'{title_prefix} #{col_idx+1}\n{row["snip_id"]}'
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.axis('off')

            # Add text annotation with metrics (include simple metrics if available)
            metric_text = f"Mean κ: {row['mean_curvature_per_um']:.6f} 1/μm\n"
            metric_text += f"Max κ: {row['max_curvature_per_um']:.6f} 1/μm\n"
            metric_text += f"Length: {row['total_length_um']:.1f} μm\n"

            # Add simple metrics if available
            if 'baseline_deviation_um' in row and pd.notna(row['baseline_deviation_um']):
                metric_text += f"Base dev: {row['baseline_deviation_um']:.1f} μm\n"
            if 'arc_length_ratio' in row and pd.notna(row['arc_length_ratio']):
                metric_text += f"Arc ratio: {row['arc_length_ratio']:.3f}\n"

            metric_text += f"Geno: {row['genotype']}\n"
            metric_text += f"Stage: {row['predicted_stage_hpf']:.1f} hpf"

            ax.text(0.02, 0.98, metric_text,
                   transform=ax.transAxes,
                   fontsize=8,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    output_file = output_dir / f'curvature_comparison_{metric}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved comparison plot to {output_file}")


def main():
    """Main execution function."""
    # Setup output directory
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Loading data...")
    summary_df, merged_df = load_data()

    print(f"Loaded {len(summary_df)} curvature records")
    print(f"Success rate: {summary_df['success'].sum() / len(summary_df) * 100:.1f}%")

    # Create histograms
    print("\nGenerating histograms...")
    plot_histograms(summary_df, output_dir)

    # Create comparison plots for original metrics
    original_metrics = ['mean_curvature_per_um', 'std_curvature_per_um',
                       'max_curvature_per_um', 'total_length_um']

    print("\nGenerating comparison plots for original metrics...")
    for metric in original_metrics:
        print(f"  - {metric}...")
        plot_embryo_comparison(merged_df, output_dir, metric=metric, n_top=5)

    # Create comparison plots for simple metrics
    simple_metrics = ['baseline_deviation_um', 'arc_length_ratio',
                     'keypoint_deviation_mid_um', 'max_baseline_deviation_um']

    print("\nGenerating comparison plots for simple metrics...")
    for metric in simple_metrics:
        if metric in merged_df.columns:
            print(f"  - {metric}...")
            plot_embryo_comparison(merged_df, output_dir, metric=metric, n_top=5)
        else:
            print(f"  - {metric}... SKIPPED (not found in data)")

    print("\n✓ All visualizations complete!")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
