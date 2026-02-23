#!/usr/bin/env python3
"""
Quick test to visualize a single mask properly 
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
    # Load reference embryo
    data_csv = MORPHSEQ_ROOT / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
    df = pd.read_csv(data_csv)
    data_root = MORPHSEQ_ROOT / "morphseq_playground"
    
    embryo_id = "20251112_H04_e01"
    frame_index = 39
    
    row = df[(df["embryo_id"] == embryo_id) & (df["frame_index"] == frame_index)].iloc[0]
    
    # Load and align mask
    mask = fmio.load_mask_from_rle_counts(
        rle_counts=row["mask_rle"],
        height_px=int(row["mask_height_px"]),
        width_px=int(row["mask_width_px"]),
    )
    yolk = fmio._load_build02_aux_mask(data_root, row, mask.shape, keyword="yolk")
    um_per_px = fmio._compute_um_per_pixel(row)
    
    config = CanonicalGridConfig(reference_um_per_pixel=10.0, grid_shape_hw=(256, 576))
    aligner = CanonicalAligner.from_config(config)
    aligned_mask, _, _ = aligner.align(
        mask=mask.astype(bool),
        yolk=yolk.astype(bool),
        original_um_per_px=um_per_px,
        use_yolk=True,
    )
    aligned_mask = aligned_mask.astype(np.uint8)
    
    # Create a test visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original mask
    axes[0].imshow(mask, cmap="gray", origin="upper") 
    axes[0].set_title(f"Original Mask\n{mask.sum()} pixels")
    axes[0].axis("off")
    
    # Aligned mask - full view
    axes[1].imshow(aligned_mask, cmap="gray", origin="upper")
    axes[1].set_title(f"Aligned Mask (Full)\n{aligned_mask.sum()} pixels")
    axes[1].axis("off")
    
    # Aligned mask - zoomed to non-zero region 
    rows, cols = np.where(aligned_mask > 0)
    if len(rows) > 0:
        row_min, row_max = rows.min(), rows.max()
        col_min, col_max = cols.min(), cols.max()
        
        # Add some padding
        padding = 20
        row_min = max(0, row_min - padding)
        row_max = min(aligned_mask.shape[0] - 1, row_max + padding)
        col_min = max(0, col_min - padding)
        col_max = min(aligned_mask.shape[1] - 1, col_max + padding)
        
        cropped = aligned_mask[row_min:row_max+1, col_min:col_max+1]
        axes[2].imshow(cropped, cmap="gray", origin="upper")
        axes[2].set_title(f"Aligned Mask (Zoomed)\nRows {row_min}-{row_max}, Cols {col_min}-{col_max}")
    else:
        axes[2].text(0.5, 0.5, "No mask pixels", ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_title("Aligned Mask (Zoomed)")
    axes[2].axis("off")
    
    plt.tight_layout()
    output_path = Path("scripts/output/mask_qc/test_single_mask.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved test visualization to {output_path}")
    print(f"Mask occupies roughly rows {row_min}-{row_max}, cols {col_min}-{col_max}")
    print(f"That's a {row_max-row_min}x{col_max-col_min} region out of {aligned_mask.shape}")

if __name__ == "__main__":
    main()
