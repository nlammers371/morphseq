#!/usr/bin/env python3
"""
Debug mask loading pipeline step by step
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

def debug_single_mask(embryo_id: str, frame_index: int, df: pd.DataFrame, data_root: Path):
    """Debug a single embryo mask through the entire pipeline"""
    print(f"\n{'='*60}")
    print(f"DEBUGGING: {embryo_id} frame {frame_index}")
    print('='*60)
    
    # Step 1: Find the row
    row = df[(df["embryo_id"] == embryo_id) & (df["frame_index"] == frame_index)]
    if row.empty:
        print("❌ ERROR: No row found in dataframe")
        return False
    
    row = row.iloc[0]
    print(f"✅ Found row: genotype={row['genotype']}, stage={row['predicted_stage_hpf']:.1f} hpf")
    
    # Step 2: Load original mask from RLE
    print(f"\nStep 2: Loading mask from RLE...")
    print(f"  RLE length: {len(row['mask_rle'])} chars")
    print(f"  Expected shape: {row['mask_height_px']}x{row['mask_width_px']}")
    
    try:
        mask = fmio.load_mask_from_rle_counts(
            rle_counts=row["mask_rle"],
            height_px=int(row["mask_height_px"]),
            width_px=int(row["mask_width_px"]),
        )
        print(f"✅ Loaded mask: shape={mask.shape}, dtype={mask.dtype}")
        print(f"  Pixel stats: {mask.sum()}/{mask.size} pixels ({100*mask.sum()/mask.size:.1f}% coverage)")
        print(f"  Value range: {mask.min()}-{mask.max()}")
        
        if mask.sum() == 0:
            print("❌ ERROR: Mask is completely empty!")
            return False
            
    except Exception as e:
        print(f"❌ ERROR loading mask: {e}")
        return False
    
    # Step 3: Load yolk mask
    print(f"\nStep 3: Loading yolk mask...")
    try:
        yolk = fmio._load_build02_aux_mask(data_root, row, mask.shape, keyword="yolk")
        
        if yolk is None:
            print("❌ ERROR: Yolk mask is None")
            return False
        
        print(f"✅ Loaded yolk: shape={yolk.shape}, dtype={yolk.dtype}")
        print(f"  Pixel stats: {yolk.sum()}/{yolk.size} pixels ({100*yolk.sum()/yolk.size:.1f}% coverage)")
        
        if yolk.sum() == 0:
            print("❌ ERROR: Yolk mask is completely empty!")
            return False
            
    except Exception as e:
        print(f"❌ ERROR loading yolk: {e}")
        return False
    
    # Step 4: Canonical alignment
    print(f"\nStep 4: Canonical alignment...")
    try:
        um_per_px = fmio._compute_um_per_pixel(row)
        print(f"  Original resolution: {um_per_px:.3f} um/px")
        
        config = CanonicalGridConfig(reference_um_per_pixel=10.0, grid_shape_hw=(256, 576))
        aligner = CanonicalAligner.from_config(config)
        print(f"  Target resolution: {config.reference_um_per_pixel} um/px")
        print(f"  Target shape: {config.grid_shape_hw}")
        
        aligned_mask, _, _ = aligner.align(
            mask=mask.astype(bool),
            yolk=yolk.astype(bool),
            original_um_per_px=um_per_px,
            use_yolk=True,
        )
        aligned_mask = aligned_mask.astype(np.uint8)
        
        print(f"✅ Aligned mask: shape={aligned_mask.shape}, dtype={aligned_mask.dtype}")
        print(f"  Pixel stats: {aligned_mask.sum()}/{aligned_mask.size} pixels ({100*aligned_mask.sum()/aligned_mask.size:.1f}% coverage)")
        
        if aligned_mask.sum() == 0:
            print("❌ ERROR: Aligned mask is completely empty!")
            return False
        
        # Find bounding box
        rows, cols = np.where(aligned_mask > 0)
        if len(rows) > 0:
            row_min, row_max = rows.min(), rows.max()
            col_min, col_max = cols.min(), cols.max()
            print(f"  Bounding box: rows {row_min}-{row_max}, cols {col_min}-{col_max}")
            print(f"  Size: {row_max-row_min}x{col_max-col_min}")
        
    except Exception as e:
        print(f"❌ ERROR in alignment: {e}")
        return False
    
    # Step 5: Create debug visualization
    print(f"\nStep 5: Creating debug visualization...")
    try:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original mask
        axes[0].imshow(mask, cmap="gray", origin="upper")
        axes[0].set_title(f"Original Mask\n{mask.sum()} pixels")
        axes[0].axis("off")
        
        # Yolk mask
        axes[1].imshow(yolk, cmap="gray", origin="upper")
        axes[1].set_title(f"Yolk Mask\n{yolk.sum()} pixels")
        axes[1].axis("off")
        
        # Aligned mask - full
        axes[2].imshow(aligned_mask, cmap="gray", origin="upper")
        axes[2].set_title(f"Aligned (Full)\n{aligned_mask.sum()} pixels")
        axes[2].axis("off")
        
        # Aligned mask - cropped to embryo region
        crop_rows = slice(60, 210)
        crop_cols = slice(170, 550)
        cropped = aligned_mask[crop_rows, crop_cols]
        axes[3].imshow(cropped, cmap="gray", origin="upper", vmin=0, vmax=1)
        axes[3].set_title(f"Aligned (Cropped)\n{cropped.sum()} pixels")
        axes[3].axis("off")
        
        plt.suptitle(f"Debug: {embryo_id} frame {frame_index} ({row['genotype']})", fontsize=12)
        plt.tight_layout()
        
        output_path = Path(f"scripts/output/mask_qc/debug_{embryo_id}_f{frame_index}.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"✅ Saved debug image: {output_path}")
        
    except Exception as e:
        print(f"❌ ERROR creating visualization: {e}")
        return False
    
    print(f"✅ SUCCESS: All steps completed for {embryo_id}")
    return True

def main():
    print("=== COMPREHENSIVE MASK LOADING DEBUG ===")
    
    # Load data
    data_csv = MORPHSEQ_ROOT / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
    print(f"Loading data from: {data_csv}")
    
    try:
        df = pd.read_csv(data_csv)
        print(f"✅ Loaded {len(df)} rows")
    except Exception as e:
        print(f"❌ ERROR loading CSV: {e}")
        return
    
    data_root = MORPHSEQ_ROOT / "morphseq_playground"
    print(f"Data root: {data_root}")
    print(f"Yolk path exists: {(data_root / 'segmentation/yolk_v1_0050_predictions').exists()}")
    
    # Test specific embryos that should work
    test_cases = [
        ("20251112_H04_e01", 39),   # Reference embryo
        ("20250512_B09_e01", 115),  # WT sample
        ("20250512_B03_e01", 110),  # Mutant sample
    ]
    
    success_count = 0
    for embryo_id, frame_index in test_cases:
        success = debug_single_mask(embryo_id, frame_index, df, data_root)
        if success:
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {success_count}/{len(test_cases)} embryos processed successfully")
    print("="*60)
    
    if success_count == 0:
        print("❌ NO embryos processed successfully - there's a fundamental issue!")
    elif success_count < len(test_cases):
        print("⚠️  Some embryos failed - check individual debug output")
    else:
        print("✅ All embryos processed - issue might be in the batch visualization")

if __name__ == "__main__":
    main()
