#!/usr/bin/env python3
"""
Debug mask loading and alignment to understand why masks appear black
"""

import sys
from pathlib import Path

# Add morphseq to path
MORPHSEQ_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(MORPHSEQ_ROOT))
sys.path.insert(0, str(MORPHSEQ_ROOT / "src"))

import pandas as pd
import numpy as np
from analyze.optimal_transport_morphometrics.uot_masks import frame_mask_io as fmio
from analyze.utils.coord.grids.canonical import CanonicalAligner, CanonicalGridConfig

def main():
    print("=== Debugging Mask Loading and Alignment ===")
    
    # Load data
    data_csv = MORPHSEQ_ROOT / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
    print(f"Loading data from: {data_csv}")
    df = pd.read_csv(data_csv)
    
    data_root = MORPHSEQ_ROOT / "morphseq_playground"
    print(f"Data root: {data_root}")
    
    # Check reference embryo
    embryo_id = "20251112_H04_e01"
    frame_index = 39
    
    row = df[(df["embryo_id"] == embryo_id) & (df["frame_index"] == frame_index)]
    if row.empty:
        print(f"ERROR: No data for {embryo_id} frame {frame_index}")
        return
    
    row = row.iloc[0]
    print(f"\nReference embryo: {embryo_id} frame {frame_index}")
    print(f"  Genotype: {row['genotype']}")
    print(f"  Stage: {row['predicted_stage_hpf']:.1f} hpf")
    print(f"  Original mask shape: {row['mask_height_px']}x{row['mask_width_px']}")
    
    # Load original mask
    mask = fmio.load_mask_from_rle_counts(
        rle_counts=row["mask_rle"],
        height_px=int(row["mask_height_px"]),
        width_px=int(row["mask_width_px"]),
    )
    print(f"  Original mask pixels: {mask.sum()} / {mask.size} ({100 * mask.sum() / mask.size:.1f}%)")
    print(f"  Original mask dtype: {mask.dtype}, unique values: {np.unique(mask)}")
    
    # Load yolk mask
    try:
        yolk = fmio._load_build02_aux_mask(data_root, row, mask.shape, keyword="yolk")
        if yolk is not None:
            print(f"  Yolk pixels: {yolk.sum()} / {yolk.size} ({100 * yolk.sum() / yolk.size:.1f}%)")
            print(f"  Yolk dtype: {yolk.dtype}, unique values: {np.unique(yolk)}")
        else:
            print(f"  Yolk: None")
            return
    except Exception as e:
        print(f"  Yolk error: {e}")
        return
    
    # Align to canonical
    um_per_px = fmio._compute_um_per_pixel(row)
    print(f"  Original um/px: {um_per_px}")
    
    config = CanonicalGridConfig(reference_um_per_pixel=10.0, grid_shape_hw=(256, 576))
    aligner = CanonicalAligner.from_config(config)
    aligned_mask, _, _ = aligner.align(
        mask=mask.astype(bool),
        yolk=yolk.astype(bool),
        original_um_per_px=um_per_px,
        use_yolk=True,
    )
    aligned_mask = aligned_mask.astype(np.uint8)
    
    print(f"  Aligned mask shape: {aligned_mask.shape}")
    print(f"  Aligned mask pixels: {aligned_mask.sum()} / {aligned_mask.size} ({100 * aligned_mask.sum() / aligned_mask.size:.1f}%)")
    print(f"  Aligned mask dtype: {aligned_mask.dtype}, unique values: {np.unique(aligned_mask)}")
    
    # Now check a few other embryos
    print("\n=== Checking Other Embryos ===")
    
    # Get a few WT and mutant samples
    stage_window = (47, 49)
    target_stage = 48.0
    
    wt_sample = df[
        (df["genotype"] == "cep290_wildtype") &
        (df["predicted_stage_hpf"] >= stage_window[0]) &
        (df["predicted_stage_hpf"] <= stage_window[1])
    ].groupby("embryo_id").apply(lambda g: g.iloc[0]).head(3)
    
    mut_sample = df[
        (df["genotype"] == "cep290_homozygous") &
        (df["predicted_stage_hpf"] >= stage_window[0]) &
        (df["predicted_stage_hpf"] <= stage_window[1])
    ].groupby("embryo_id").apply(lambda g: g.iloc[0]).head(3)
    
    for label, sample_df in [("WT", wt_sample), ("Mutant", mut_sample)]:
        print(f"\n{label} samples:")
        for idx, row in sample_df.iterrows():
            print(f"  {row['embryo_id']} frame {row['frame_index']}:")
            print(f"    Stage: {row['predicted_stage_hpf']:.1f} hpf")
            
            # Load mask
            mask = fmio.load_mask_from_rle_counts(
                rle_counts=row["mask_rle"],
                height_px=int(row["mask_height_px"]),
                width_px=int(row["mask_width_px"]),
            )
            
            # Try to load yolk
            try:
                yolk = fmio._load_build02_aux_mask(data_root, row, mask.shape, keyword="yolk")
                if yolk is None or yolk.sum() == 0:
                    print(f"    ERROR: Missing/empty yolk mask - skipping alignment")
                    continue
                
                # Align
                um_per_px = fmio._compute_um_per_pixel(row)
                aligned_mask, _, _ = aligner.align(
                    mask=mask.astype(bool),
                    yolk=yolk.astype(bool),
                    original_um_per_px=um_per_px,
                    use_yolk=True,
                )
                aligned_mask = aligned_mask.astype(np.uint8)
                
                print(f"    Original: {mask.sum()}/{mask.size} ({100 * mask.sum() / mask.size:.1f}%)")
                print(f"    Aligned: {aligned_mask.sum()}/{aligned_mask.size} ({100 * aligned_mask.sum() / aligned_mask.size:.1f}%)")
                
            except Exception as e:
                print(f"    ERROR: {e}")

if __name__ == "__main__":
    main()
