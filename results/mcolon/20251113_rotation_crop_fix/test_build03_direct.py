#!/usr/bin/env python3
"""
Test build03 extraction directly at both 6.5 and 7.8 μm/px scales.
No custom logic - just calls build03's export_embryo_snips() function.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import skimage.io as io
import matplotlib.pyplot as plt

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.build.build03A_process_images import export_embryo_snips


if __name__ == "__main__":
    root = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground")
    output_dir = Path(__file__).parent / "build03_direct_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load SAM2 metadata
    metadata_files = [
        root / "sam2_pipeline_files" / "sam2_expr_files" / "sam2_metadata_20251106.csv",
        root / "sam2_pipeline_files" / "sam2_expr_files" / "sam2_metadata_20250512.csv",
    ]

    available_files = [f for f in metadata_files if f.exists()]
    if not available_files:
        print(f"❌ ERROR: No metadata files found!")
        exit(1)

    df_meta = pd.concat([pd.read_csv(f) for f in available_files], ignore_index=True)

    # Test snip IDs
    test_snip_ids = [
        "20251106_A02_e01_t0049",
        "20251106_A03_e01_t0085",
        "20250512_A03_e01_t0112",
        "20251106_H12_e01_t0093",
    ]

    # Filter to test snips
    test_df = df_meta[df_meta["snip_id"].isin(test_snip_ids)].copy()

    # Add missing columns that build03 expects
    if "experiment_date" not in test_df.columns:
        test_df["experiment_date"] = test_df["experiment_id"]

    if "region_label" not in test_df.columns:
        test_df["region_label"] = test_df["embryo_id"].str.extract(r'_e(\d+)$')[0].astype(int)

    print("="*80)
    print("BUILD03 DIRECT TEST - COMPARING 6.5 vs 7.8 μm/px")
    print("="*80)
    print(f"Testing {len(test_df)} embryos")
    print()

    # Compute px_mean and px_std for noise generation
    px_mean = 100  # Default
    px_std = 50    # Default

    results = []

    for scale, label in [(6.5, "OLD"), (7.8, "NEW")]:
        print(f"\n{'='*80}")
        print(f"{label}: Processing at {scale} μm/px")
        print(f"{'='*80}")

        # Create a copy of test_df for this scale
        test_df_copy = test_df.copy()

        # Process each embryo
        for idx in test_df_copy.index:
            row_idx = test_df.index.get_loc(idx)
            snip_id = test_df_copy.loc[idx, "snip_id"]

            try:
                print(f"  Processing {snip_id}...")

                # Call build03's export_embryo_snips directly
                out_of_frame = export_embryo_snips(
                    r=row_idx,
                    root=root,
                    stats_df=test_df_copy,
                    dl_rad_um=50,
                    outscale=scale,
                    outshape=[576, 256],
                    px_mean=px_mean,
                    px_std=px_std
                )

                print(f"    ✓ Complete - out_of_frame: {out_of_frame}")

                results.append({
                    "snip_id": snip_id,
                    "scale": scale,
                    "out_of_frame": out_of_frame
                })

            except Exception as e:
                print(f"    ❌ Failed: {e}")
                results.append({
                    "snip_id": snip_id,
                    "scale": scale,
                    "out_of_frame": None,
                    "error": str(e)
                })

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_dir / "extraction_results.csv", index=False)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    # Pivot to compare scales
    pivot = df_results.pivot(index='snip_id', columns='scale', values='out_of_frame')
    print(pivot)

    print(f"\n{'='*80}")
    print(f"✓ Results saved to {output_dir / 'extraction_results.csv'}")
    print(f"✓ Extracted snips saved to {root / 'training_data' / 'bf_embryo_snips'}")
    print(f"{'='*80}")
