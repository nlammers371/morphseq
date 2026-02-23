#!/usr/bin/env python3
"""
SIMPLE VERIFICATION: Show WT vs Mutant masks side-by-side
No fancy processing - just load, align, and display clearly
"""

import sys
from pathlib import Path

MORPHSEQ_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(MORPHSEQ_ROOT))
sys.path.insert(0, str(MORPHSEQ_ROOT / "src"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
from analyze.optimal_transport_morphometrics.uot_masks import frame_mask_io as fmio
from analyze.utils.coord.grids.canonical import CanonicalAligner, CanonicalGridConfig

def load_and_align_mask(row, data_root, aligner):
    """Load mask, align to canonical, return aligned mask"""
    # Load original mask
    mask = fmio.load_mask_from_rle_counts(
        rle_counts=row["mask_rle"],
        height_px=int(row["mask_height_px"]),
        width_px=int(row["mask_width_px"]),
    )
    
    # Load yolk
    yolk = fmio._load_build02_aux_mask(data_root, row, mask.shape, keyword="yolk")
    if yolk is None or yolk.sum() == 0:
        return None
    
    # Align
    um_per_px = fmio._compute_um_per_pixel(row)
    aligned, _, _ = aligner.align(
        mask=mask.astype(bool),
        yolk=yolk.astype(bool),
        original_um_per_px=um_per_px,
        use_yolk=True,
    )
    return aligned.astype(np.uint8)

def main():
    print("\n" + "="*70)
    print("SIMPLE VERIFICATION: WT vs MUTANT MASKS")
    print("="*70 + "\n")
    
    # Load data
    data_csv = MORPHSEQ_ROOT / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
    df = pd.read_csv(data_csv)
    data_root = MORPHSEQ_ROOT / "morphseq_playground"
    
    # Setup aligner
    config = CanonicalGridConfig(reference_um_per_pixel=10.0, grid_shape_hw=(256, 576))
    aligner = CanonicalAligner.from_config(config)
    
    # Get stage-matched samples
    stage_df = df[(df["predicted_stage_hpf"] >= 47) & (df["predicted_stage_hpf"] <= 49)]
    
    # Sample WT and mutants
    wt_embryos = stage_df[stage_df["genotype"] == "cep290_wildtype"].groupby("embryo_id").first().reset_index().head(3)
    mut_embryos = stage_df[stage_df["genotype"] == "cep290_homozygous"].groupby("embryo_id").first().reset_index().head(3)
    
    print(f"Found {len(wt_embryos)} WT embryos")
    print(f"Found {len(mut_embryos)} MUTANT embryos\n")
    
    # Load masks
    wt_masks = []
    wt_labels = []
    mut_masks = []
    mut_labels = []
    
    print("Loading WT masks...")
    for _, row in wt_embryos.iterrows():
        mask = load_and_align_mask(row, data_root, aligner)
        if mask is not None and mask.sum() > 0:
            wt_masks.append(mask)
            wt_labels.append(f"{row['embryo_id']}\n{row['predicted_stage_hpf']:.1f} hpf\n{mask.sum()} px")
            print(f"  ✓ {row['embryo_id']}: {mask.sum()} pixels")
    
    print(f"\nLoading MUTANT masks...")
    for _, row in mut_embryos.iterrows():
        mask = load_and_align_mask(row, data_root, aligner)
        if mask is not None and mask.sum() > 0:
            mut_masks.append(mask)
            mut_labels.append(f"{row['embryo_id']}\n{row['predicted_stage_hpf']:.1f} hpf\n{mask.sum()} px")
            print(f"  ✓ {row['embryo_id']}: {mask.sum()} pixels")
    
    if len(wt_masks) == 0 or len(mut_masks) == 0:
        print("\n❌ ERROR: No masks loaded!")
        return
    
    print(f"\n✓ Loaded {len(wt_masks)} WT masks and {len(mut_masks)} MUTANT masks")
    
    # Create side-by-side comparison
    n_samples = min(len(wt_masks), len(mut_masks))
    fig, axes = plt.subplots(2, n_samples, figsize=(5*n_samples, 10))
    
    if n_samples == 1:
        axes = axes.reshape(2, 1)
    
    # Crop to embryo region
    crop_rows = slice(60, 210)
    crop_cols = slice(170, 550)
    
    for i in range(n_samples):
        # WT row
        wt_cropped = wt_masks[i][crop_rows, crop_cols]
        axes[0, i].imshow(wt_cropped, cmap="gray", origin="upper", vmin=0, vmax=1)
        
        # Add outline
        outline = wt_cropped.astype(bool) ^ binary_erosion(wt_cropped.astype(bool), iterations=1)
        axes[0, i].contour(outline, colors="cyan", linewidths=2, levels=[0.5])
        
        axes[0, i].set_title(f"WILDTYPE\n{wt_labels[i]}", fontsize=11, fontweight="bold", color="green")
        axes[0, i].axis("off")
        
        # Mutant row
        mut_cropped = mut_masks[i][crop_rows, crop_cols]
        axes[1, i].imshow(mut_cropped, cmap="gray", origin="upper", vmin=0, vmax=1)
        
        # Add outline
        outline = mut_cropped.astype(bool) ^ binary_erosion(mut_cropped.astype(bool), iterations=1)
        axes[1, i].contour(outline, colors="red", linewidths=2, levels=[0.5])
        
        axes[1, i].set_title(f"HOMOZYGOUS MUTANT\n{mut_labels[i]}", fontsize=11, fontweight="bold", color="red")
        axes[1, i].axis("off")
    
    plt.suptitle("CEP290 Phenotype Comparison: WT (top) vs Homozygous Mutant (bottom)\nLook for curvature differences", 
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    output_path = Path("scripts/output/mask_qc/SIMPLE_VERIFICATION.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"\n{'='*70}")
    print(f"✓ SAVED: {output_path}")
    print(f"{'='*70}")
    print("\nLOOK FOR:")
    print("  - WT embryos (TOP ROW, GREEN): Should be relatively STRAIGHT")
    print("  - MUTANT embryos (BOTTOM ROW, RED): Should be more CURVED/BENT")
    print("\nIf you see clear masks with different shapes, the pipeline is working!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
