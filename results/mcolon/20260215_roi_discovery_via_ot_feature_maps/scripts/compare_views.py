#!/usr/bin/env python3
"""
Create side-by-side comparison of full vs cropped mask views
"""

import sys
from pathlib import Path

# Add morphseq to path
MORPHSEQ_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(MORPHSEQ_ROOT))
sys.path.insert(0, str(MORPHSEQ_ROOT / "src"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analyze.optimal_transport_morphometrics.uot_masks import frame_mask_io as fmio
from analyze.utils.coord.grids.canonical import CanonicalAligner, CanonicalGridConfig

def main():
    print("Creating full vs cropped comparison...")
    
    # Load data
    data_csv = MORPHSEQ_ROOT / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
    df = pd.read_csv(data_csv)
    data_root = MORPHSEQ_ROOT / "morphseq_playground"
    
    # Test with a few different embryos
    test_embryos = [
        ("20251112_H04_e01", 39),  # reference
        ("20250512_B09_e01", 115),  # WT
        ("20250512_B03_e01", 110),  # Mutant
    ]
    
    config = CanonicalGridConfig(reference_um_per_pixel=10.0, grid_shape_hw=(256, 576))
    aligner = CanonicalAligner.from_config(config)
    
    # Crop region for embryos
    crop_rows = slice(60, 210)
    crop_cols = slice(170, 550)
    
    fig, axes = plt.subplots(len(test_embryos), 2, figsize=(12, 4 * len(test_embryos)))
    
    for i, (embryo_id, frame_index) in enumerate(test_embryos):
        print(f"Processing {embryo_id} frame {frame_index}...")
        
        row = df[(df["embryo_id"] == embryo_id) & (df["frame_index"] == frame_index)]
        if row.empty:
            print(f"  No data found")
            continue
        
        row = row.iloc[0]
        
        # Load and align mask
        mask = fmio.load_mask_from_rle_counts(
            rle_counts=row["mask_rle"],
            height_px=int(row["mask_height_px"]),
            width_px=int(row["mask_width_px"]),
        )
        
        try:
            yolk = fmio._load_build02_aux_mask(data_root, row, mask.shape, keyword="yolk")
            if yolk is None or yolk.sum() == 0:
                print(f"  Missing yolk mask - skipping")
                continue
            
            um_per_px = fmio._compute_um_per_pixel(row)
            aligned_mask, _, _ = aligner.align(
                mask=mask.astype(bool),
                yolk=yolk.astype(bool),
                original_um_per_px=um_per_px,
                use_yolk=True,
            )
            aligned_mask = aligned_mask.astype(np.uint8)
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
        
        # Full view
        axes[i, 0].imshow(aligned_mask, cmap="gray", origin="upper")
        axes[i, 0].set_title(f"Full View\n{embryo_id} ({row['genotype']})\n{aligned_mask.sum()} pixels")
        axes[i, 0].axis("off")
        
        # Cropped view
        cropped = aligned_mask[crop_rows, crop_cols]
        axes[i, 1].imshow(cropped, cmap="gray", origin="upper", vmin=0, vmax=1)
        axes[i, 1].set_title(f"Cropped View (Embryo Region)\n{cropped.sum()} pixels\nShape: {cropped.shape}")
        axes[i, 1].axis("off")
        
        # Add red box on full view to show crop region
        from matplotlib.patches import Rectangle
        start_row, start_col = crop_rows.start, crop_cols.start
        height = crop_rows.stop - crop_rows.start
        width = crop_cols.stop - crop_cols.start
        rect = Rectangle((start_col, start_row), width, height, 
                        linewidth=2, edgecolor='red', facecolor='none')
        axes[i, 0].add_patch(rect)
    
    plt.tight_layout()
    output_path = Path("scripts/output/mask_qc/full_vs_cropped_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved comparison to {output_path}")

if __name__ == "__main__":
    main()
