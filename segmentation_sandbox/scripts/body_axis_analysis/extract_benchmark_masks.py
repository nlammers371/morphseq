#!/usr/bin/env python3
"""
Extract sample masks for geodesic speedup benchmarking.

Uses the 1000-embryo comparison CSV to load 5 diverse masks:
1. Simple case (agreement, low curvature)
2. Moderate case (agreement, moderate curvature)
3. Challenging case (disagreement, high curvature)
4. High aspect ratio case
5. High solidity/non-convex case

Masks are saved as .npy files for benchmarking.
"""

import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
from segmentation_sandbox.scripts.utils.mask_utils import decode_mask_rle
from segmentation_sandbox.scripts.utils.mask_cleaning import clean_embryo_mask


def extract_benchmark_masks(
    comparison_csv: Path,
    metadata_csv: Path,
    output_dir: Path,
    n_masks: int = 5
):
    """
    Extract diverse masks from comparison results for benchmarking.

    Selects masks that represent different morphological profiles:
    - Simple (low extent, low eccentricity)
    - Moderate (mid-range metrics)
    - Challenging (high extent, high eccentricity, high disagreement)
    - etc.

    Args:
        comparison_csv: Path to pca_vs_geodesic_comparison_1000embryos.csv
        metadata_csv: Path to df03_final_output_with_latents_*.csv (with mask_rle)
        output_dir: Directory to save .npy masks
        n_masks: Number of masks to extract (default=5)

    Returns:
        List of (snip_id, mask_path) tuples
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading comparison CSV: {comparison_csv}")
    df_comparison = pd.read_csv(comparison_csv)
    print(f"Loaded {len(df_comparison)} embryos from comparison")

    print(f"\nLoading metadata CSV: {metadata_csv}")
    df_metadata = pd.read_csv(metadata_csv)
    print(f"Loaded {len(df_metadata)} embryos from metadata")

    # Merge on snip_id to get both comparison metrics and mask_rle
    df = df_comparison.merge(df_metadata, on='snip_id', how='inner')
    print(f"Merged: {len(df)} embryos with both metrics and masks")

    # Define selection criteria for diverse masks
    selections = []

    # Sort by different metrics to get diverse examples
    # 1. Smallest masks (simple cases)
    if len(df) > 0:
        simple = df.nsmallest(1, 'area')
        if len(simple) > 0:
            selections.append(("simple", simple.iloc[0]))
            print(f"✓ Selected simple case (smallest): {simple.iloc[0]['snip_id']}")

    # 2. Moderate size
    if len(df) > 1:
        df_sorted = df.sort_values('area')
        mid_idx = len(df_sorted) // 2
        moderate = df_sorted.iloc[mid_idx:mid_idx+1]
        if len(moderate) > 0:
            selections.append(("moderate", moderate.iloc[0]))
            print(f"✓ Selected moderate case (median size): {moderate.iloc[0]['snip_id']}")

    # 3. Highest disagreement (largest Hausdorff distance)
    if len(df) > 2:
        challenging = df.nlargest(1, 'hausdorff_distance')
        if len(challenging) > 0:
            selections.append(("challenging", challenging.iloc[0]))
            print(f"✓ Selected challenging case (high disagreement): {challenging.iloc[0]['snip_id']}")

    # 4. Highest eccentricity (most elongated)
    if len(df) > 3:
        elongated = df.nlargest(1, 'eccentricity')
        if len(elongated) > 0:
            selections.append(("elongated", elongated.iloc[0]))
            print(f"✓ Selected elongated case (high eccentricity): {elongated.iloc[0]['snip_id']}")

    # 5. Lowest solidity (most non-convex)
    if len(df) > 4:
        nonconvex = df.nsmallest(1, 'solidity')
        if len(nonconvex) > 0:
            selections.append(("nonconvex", nonconvex.iloc[0]))
            print(f"✓ Selected non-convex case (low solidity): {nonconvex.iloc[0]['snip_id']}")

    # Save masks
    saved_masks = []
    for case_name, row in selections:
        try:
            # Decode mask
            mask = decode_mask_rle({
                'size': [int(row['mask_height_px']), int(row['mask_width_px'])],
                'counts': row['mask_rle']
            })

            # Clean mask
            cleaned_mask, _ = clean_embryo_mask(mask, verbose=False)

            # Save
            snip_id = row['snip_id']
            mask_path = output_dir / f"{snip_id}_{case_name}.npy"
            np.save(mask_path, cleaned_mask)

            saved_masks.append((snip_id, mask_path))
            print(f"  → Saved: {mask_path.name} ({cleaned_mask.shape}, {cleaned_mask.sum()} pixels)")

        except Exception as e:
            print(f"  ✗ Error saving {case_name}: {e}")

    return saved_masks


def main():
    """Extract benchmark masks and report paths."""
    # Paths
    comparison_csv = Path(
        "/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251027/"
        "pca_vs_geodesic_comparison_1000embryos.csv"
    )
    metadata_dir = Path(
        "/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build06_output"
    )
    output_dir = Path(__file__).parent / "benchmark_masks"

    if not comparison_csv.exists():
        print(f"✗ Comparison CSV not found: {comparison_csv}")
        return

    if not metadata_dir.exists():
        print(f"✗ Metadata directory not found: {metadata_dir}")
        return

    # Find metadata CSV (handle both part1 and part2)
    metadata_csvs = list(metadata_dir.glob("df03_final_output_with_latents_*.csv"))
    if not metadata_csvs:
        print(f"✗ No metadata CSVs found in: {metadata_dir}")
        return

    # Extract masks from each metadata CSV and combine
    print("=" * 70)
    print("Extracting Benchmark Masks")
    print("=" * 70)

    all_saved_masks = []
    for metadata_csv in sorted(metadata_csvs):
        print(f"\nProcessing: {metadata_csv.name}")
        saved_masks = extract_benchmark_masks(comparison_csv, metadata_csv, output_dir)
        all_saved_masks.extend(saved_masks)

    # Report
    print("\n" + "=" * 70)
    print("Benchmark Masks Ready")
    print("=" * 70)
    print(f"\nSaved {len(all_saved_masks)} masks to: {output_dir}")
    print("\nMasks:")
    for snip_id, mask_path in all_saved_masks:
        print(f"  • {snip_id}")
        print(f"    Path: {mask_path}")

    print("\nUsage in geodesic_speedup.py:")
    for snip_id, mask_path in all_saved_masks:
        print(f"  python -m segmentation_sandbox.scripts.body_axis_analysis.geodesic_speedup \\")
        print(f"      --mask {mask_path} \\")
        print(f"      --clean --preprocess gaussian_blur --repeats 5")


if __name__ == "__main__":
    main()
