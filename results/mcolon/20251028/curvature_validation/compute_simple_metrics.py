"""
Compute Simple Curvature Metrics from Existing Spline Arrays

This script reads the curvature arrays CSV (with centerline_x and centerline_y),
computes simple curvature metrics for each embryo, and outputs an enhanced summary CSV
with the new metrics added.

Usage:
    python results/mcolon/20251028/curvature_validation/compute_simple_metrics.py

Output:
    morphseq_playground/metadata/body_axis/summary/curvature_metrics_summary_with_simple_20251017_combined.csv

New metrics added:
    - baseline_deviation_um: Mean distance from straight head-tail line
    - max_baseline_deviation_um: Maximum distance from baseline
    - baseline_deviation_std_um: Std dev of distances
    - arc_length_ratio: Arc length / chord length ratio
    - arc_length_um: Total spline length (redundant but included for clarity)
    - chord_length_um: Straight-line head-tail distance
    - keypoint_deviation_q1_um: Deviation at 25% point
    - keypoint_deviation_mid_um: Deviation at 50% point (midpoint)
    - keypoint_deviation_q3_um: Deviation at 75% point
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import sys
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from segmentation_sandbox.scripts.body_axis_analysis.curvature_metrics import (
    compute_all_simple_metrics
)


def load_data(project_root: Path):
    """Load the arrays CSV and summary CSV."""
    arrays_path = project_root / "morphseq_playground/metadata/body_axis/arrays/curvature_arrays_20251017_combined.csv"
    summary_path = project_root / "morphseq_playground/metadata/body_axis/summary/curvature_metrics_summary_20251017_combined.csv"

    print(f"Loading arrays from: {arrays_path}")
    arrays_df = pd.read_csv(arrays_path)

    print(f"Loading summary from: {summary_path}")
    summary_df = pd.read_csv(summary_path)

    print(f"Loaded {len(arrays_df)} array records")
    print(f"Loaded {len(summary_df)} summary records")

    return arrays_df, summary_df


def compute_metrics_for_embryo(row: pd.Series) -> dict:
    """
    Compute simple curvature metrics for a single embryo.

    Args:
        row: DataFrame row with centerline_x, centerline_y, um_per_pixel

    Returns:
        dict with computed metrics (or None values if computation fails)
    """
    try:
        # Parse JSON arrays
        centerline_x = np.array(json.loads(row['centerline_x']))
        centerline_y = np.array(json.loads(row['centerline_y']))
        um_per_pixel = row['um_per_pixel']

        # Validate arrays
        if len(centerline_x) != len(centerline_y):
            raise ValueError(f"Mismatched array lengths: x={len(centerline_x)}, y={len(centerline_y)}")

        if len(centerline_x) < 3:
            raise ValueError(f"Insufficient points: {len(centerline_x)}")

        # Compute all simple metrics
        metrics = compute_all_simple_metrics(centerline_x, centerline_y, um_per_pixel)

        return metrics

    except Exception as e:
        print(f"Error processing {row['snip_id']}: {e}")
        # Return None values for all expected metrics
        return {
            'baseline_deviation_um': None,
            'max_baseline_deviation_um': None,
            'baseline_deviation_std_um': None,
            'arc_length_ratio': None,
            'arc_length_um': None,
            'chord_length_um': None,
            'keypoint_deviation_q1_um': None,
            'keypoint_deviation_mid_um': None,
            'keypoint_deviation_q3_um': None
        }


def main():
    """Main processing function."""
    print("=" * 80)
    print("Computing Simple Curvature Metrics")
    print("=" * 80)
    print()

    # Load data
    arrays_df, summary_df = load_data(project_root)

    # Filter to successful analyses only
    print(f"\nFiltering to successful analyses...")
    arrays_df = arrays_df[arrays_df['success'] == True].copy()
    print(f"Processing {len(arrays_df)} successful embryos")

    # Compute metrics for each embryo
    print("\nComputing simple metrics for each embryo...")
    metrics_list = []

    for _, row in tqdm(arrays_df.iterrows(), total=len(arrays_df), desc="Processing embryos"):
        metrics = compute_metrics_for_embryo(row)
        metrics['snip_id'] = row['snip_id']  # Add snip_id for merging
        metrics_list.append(metrics)

    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics_list)

    # Merge with original summary
    print("\nMerging with original summary...")
    enhanced_summary = summary_df.merge(metrics_df, on='snip_id', how='left')

    # Report statistics
    print("\n" + "=" * 80)
    print("Summary Statistics for New Metrics")
    print("=" * 80)

    metric_columns = [
        'baseline_deviation_um',
        'max_baseline_deviation_um',
        'arc_length_ratio',
        'keypoint_deviation_mid_um'
    ]

    for col in metric_columns:
        if col in enhanced_summary.columns:
            data = enhanced_summary[col].dropna()
            if len(data) > 0:
                print(f"\n{col}:")
                print(f"  Mean: {data.mean():.4f}")
                print(f"  Median: {data.median():.4f}")
                print(f"  Std: {data.std():.4f}")
                print(f"  Min: {data.min():.4f}")
                print(f"  Max: {data.max():.4f}")
                print(f"  Count: {len(data)}")

    # Save output
    output_path = project_root / "morphseq_playground/metadata/body_axis/summary/curvature_metrics_summary_with_simple_20251017_combined.csv"
    print(f"\nSaving enhanced summary to: {output_path}")
    enhanced_summary.to_csv(output_path, index=False)

    print(f"\n✓ Successfully saved {len(enhanced_summary)} records")
    print(f"✓ Added {len(metric_columns)} new metric columns")

    # Show column list
    print("\nNew columns added:")
    new_cols = [col for col in metrics_df.columns if col != 'snip_id']
    for col in sorted(new_cols):
        print(f"  - {col}")

    print("\n" + "=" * 80)
    print("Processing complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
