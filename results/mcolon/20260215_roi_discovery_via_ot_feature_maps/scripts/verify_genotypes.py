#!/usr/bin/env python3
"""
Verify genotype selection and visualize phenotype differences
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
    print("=== Verifying Genotype Selection and Phenotypes ===")
    
    # Load data
    data_csv = MORPHSEQ_ROOT / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
    df = pd.read_csv(data_csv)
    data_root = MORPHSEQ_ROOT / "morphseq_playground"
    
    # Check available genotypes in the 47-49 hpf window
    stage_window = df[
        (df["predicted_stage_hpf"] >= 47) &
        (df["predicted_stage_hpf"] <= 49)
    ]
    
    print("\nGenotype counts in 47-49 hpf window:")
    genotype_counts = stage_window["genotype"].value_counts()
    print(genotype_counts)
    print()
    
    # Sample a few from each genotype for visual comparison
    config = CanonicalGridConfig(reference_um_per_pixel=10.0, grid_shape_hw=(256, 576))
    aligner = CanonicalAligner.from_config(config)
    
    # Crop region for better visibility
    crop_rows = slice(60, 210)
    crop_cols = slice(170, 550)
    
    # Get samples from each genotype
    samples_per_genotype = 3
    
    genotype_samples = {}
    for genotype in genotype_counts.index:
        print(f"Sampling {genotype}...")
        genotype_df = stage_window[stage_window["genotype"] == genotype]
        
        # Sample diverse embryos (one frame per embryo)
        per_embryo = genotype_df.groupby("embryo_id").apply(lambda g: g.iloc[0]).head(samples_per_genotype)
        
        masks = []
        labels = []
        
        for _, row in per_embryo.iterrows():
            try:
                print(f"  Processing {row['embryo_id']} frame {row['frame_index']} (stage {row['predicted_stage_hpf']:.1f})")
                
                # Load mask
                mask = fmio.load_mask_from_rle_counts(
                    rle_counts=row["mask_rle"],
                    height_px=int(row["mask_height_px"]),
                    width_px=int(row["mask_width_px"]),
                )
                
                # Load yolk
                yolk = fmio._load_build02_aux_mask(data_root, row, mask.shape, keyword="yolk")
                if yolk is None or yolk.sum() == 0:
                    print(f"    SKIP: No yolk mask")
                    continue
                
                # Align to canonical
                um_per_px = fmio._compute_um_per_pixel(row)
                aligned_mask, _, _ = aligner.align(
                    mask=mask.astype(bool),
                    yolk=yolk.astype(bool),
                    original_um_per_px=um_per_px,
                    use_yolk=True,
                )
                aligned_mask = aligned_mask.astype(np.uint8)
                
                masks.append(aligned_mask)
                labels.append(f"{row['embryo_id']}\nFrame {row['frame_index']}\nStage {row['predicted_stage_hpf']:.1f} hpf")
                
            except Exception as e:
                print(f"    ERROR: {e}")
                continue
        
        genotype_samples[genotype] = (masks, labels)
        print(f"  Got {len(masks)} valid masks for {genotype}")
    
    # Create visualization comparing genotypes
    n_genotypes = len(genotype_samples)
    if n_genotypes == 0:
        print("ERROR: No samples loaded")
        return
    
    fig, axes = plt.subplots(n_genotypes, samples_per_genotype, figsize=(4*samples_per_genotype, 4*n_genotypes))
    if n_genotypes == 1:
        axes = axes.reshape(1, -1)
    if samples_per_genotype == 1:
        axes = axes.reshape(-1, 1)
    
    for row_idx, (genotype, (masks, labels)) in enumerate(genotype_samples.items()):
        for col_idx in range(samples_per_genotype):
            ax = axes[row_idx, col_idx]
            
            if col_idx >= len(masks):
                ax.axis("off")
                continue
            
            mask = masks[col_idx]
            cropped_mask = mask[crop_rows, crop_cols]
            
            # Display with high contrast
            ax.imshow(cropped_mask, cmap="gray", origin="upper", vmin=0, vmax=1)
            
            # Add outline for better visibility
            from scipy.ndimage import binary_erosion
            outline = cropped_mask.astype(bool) ^ binary_erosion(cropped_mask.astype(bool), iterations=1)
            ax.contour(outline, colors="cyan", linewidths=2, levels=[0.5])
            
            ax.set_title(labels[col_idx], fontsize=9)
            ax.axis("off")
        
        # Add genotype label on the left
        axes[row_idx, 0].text(-0.1, 0.5, f"{genotype}\n({genotype_counts[genotype]} total)", 
                             transform=axes[row_idx, 0].transAxes, 
                             rotation=90, 
                             ha="center", va="center", 
                             fontsize=12, fontweight="bold",
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.suptitle("Genotype Verification: Visual Comparison of CEP290 Phenotypes", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    output_path = Path("scripts/output/mask_qc/genotype_verification.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"\nSaved genotype comparison to {output_path}")
    
    # Print summary statistics
    print("\n=== Summary ===")
    print(f"Total frames in 47-49 hpf window: {len(stage_window)}")
    for genotype, count in genotype_counts.items():
        pct = 100 * count / len(stage_window)
        print(f"  {genotype}: {count} frames ({pct:.1f}%)")
    
    # Check for curvature differences (simple analysis)
    print(f"\nNote: Look for visual differences in embryo shape:")
    print(f"  - cep290_wildtype should appear relatively straight")
    print(f"  - cep290_homozygous should appear more curved/bent")
    print(f"  - This visualization shows {samples_per_genotype} examples from each genotype")

if __name__ == "__main__":
    main()
