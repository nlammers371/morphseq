#!/usr/bin/env python3
"""
Comprehensive CanonicalAligner Debugging

Tests alignment on many embryos to identify failure patterns.
Saves detailed diagnostics and visualizations for failed cases.
"""

import sys
from pathlib import Path

MORPHSEQ_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MORPHSEQ_ROOT))
sys.path.insert(0, str(MORPHSEQ_ROOT / "src"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import traceback

from analyze.optimal_transport_morphometrics.uot_masks import frame_mask_io as fmio
from analyze.utils.coord.grids.canonical import CanonicalAligner, CanonicalGridConfig

OUTPUT_DIR = Path(__file__).parent / "debug_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def test_embryo_alignment(
    embryo_id: str,
    frame_index: int,
    row: pd.Series,
    data_root: Path,
    aligner: CanonicalAligner,
) -> Tuple[bool, str, Dict]:
    """Test alignment for a single embryo frame"""
    
    try:
        # Load mask
        mask = fmio.load_mask_from_rle_counts(
            rle_counts=row["mask_rle"],
            height_px=int(row["mask_height_px"]),
            width_px=int(row["mask_width_px"]),
        )
        
        # Load yolk
        yolk = fmio._load_build02_aux_mask(data_root, row, mask.shape, keyword="yolk")
        if yolk is None or yolk.sum() == 0:
            return False, "NO_YOLK", {}
        
        # Compute resolution
        um_per_px = fmio._compute_um_per_pixel(row)
        
        # Attempt alignment
        aligned, aligned_yolk, meta = aligner.align(
            mask=mask.astype(bool),
            yolk=yolk.astype(bool),
            original_um_per_px=um_per_px,
            use_yolk=True,
        )
        
        # Check result
        if aligned.sum() == 0:
            return False, f"EMPTY_OUTPUT (should have raised error!)", meta
        
        # Success
        info = {
            "original_pixels": int(mask.sum()),
            "yolk_pixels": int(yolk.sum()),
            "aligned_pixels": int(aligned.sum()),
            "retained_ratio": meta.get("retained_ratio", 0),
            "rotation_deg": meta.get("rotation_deg", 0),
            "flip": meta.get("flip", False),
            "clipped": meta.get("clipped", False),
            "fit_impossible": meta.get("fit_impossible", False),
        }
        return True, "SUCCESS", info
        
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        return False, f"{error_type}: {error_msg[:200]}", {}

def save_failure_visualization(
    embryo_id: str,
    frame_index: int,
    row: pd.Series,
    data_root: Path,
    error_msg: str,
):
    """Create detailed visualization for failed alignment"""
    
    try:
        mask = fmio.load_mask_from_rle_counts(
            rle_counts=row["mask_rle"],
            height_px=int(row["mask_height_px"]),
            width_px=int(row["mask_width_px"]),
        )
        yolk = fmio._load_build02_aux_mask(data_root, row, mask.shape, keyword="yolk")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(mask, cmap="gray")
        axes[0].set_title(f"Original Mask\n{mask.sum()} pixels")
        axes[0].axis("off")
        
        if yolk is not None:
            axes[1].imshow(yolk, cmap="gray")
            axes[1].set_title(f"Yolk Mask\n{yolk.sum()} pixels")
        else:
            axes[1].text(0.5, 0.5, "NO YOLK", ha="center", va="center", transform=axes[1].transAxes)
            axes[1].set_title("Yolk Mask")
        axes[1].axis("off")
        
        plt.suptitle(
            f"FAILED: {embryo_id} frame {frame_index}\n"
            f"Genotype: {row['genotype']}, Stage: {row['predicted_stage_hpf']:.1f} hpf\n"
            f"Error: {error_msg[:100]}",
            fontsize=10
        )
        plt.tight_layout()
        
        output_path = OUTPUT_DIR / f"FAILED_{embryo_id}_f{frame_index}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        
    except Exception as e:
        print(f"  Could not create visualization: {e}")

def main():
    print("="*80)
    print("COMPREHENSIVE CANONICAL ALIGNER DEBUG")
    print("="*80)
    print()
    
    # Load data
    data_csv = MORPHSEQ_ROOT / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
    print(f"Loading data from: {data_csv}")
    df = pd.read_csv(data_csv, low_memory=False)
    print(f"Loaded {len(df)} total frames\n")
    
    data_root = MORPHSEQ_ROOT / "morphseq_playground"
    
    # Setup aligner
    config = CanonicalGridConfig(reference_um_per_pixel=10.0, grid_shape_hw=(256, 576))
    aligner = CanonicalAligner.from_config(config)
    print(f"Aligner config: {config.grid_shape_hw} grid at {config.reference_um_per_pixel} um/px\n")
    
    # Test on stage-matched samples (47-49 hpf)
    stage_df = df[
        (df["predicted_stage_hpf"] >= 47) & 
        (df["predicted_stage_hpf"] <= 49)
    ]
    print(f"Testing on {len(stage_df)} frames in 47-49 hpf window")
    
    # Sample diverse embryos
    test_samples = []
    for genotype in ["cep290_wildtype", "cep290_homozygous"]:
        genotype_df = stage_df[stage_df["genotype"] == genotype]
        per_embryo = genotype_df.groupby("embryo_id").first().reset_index()
        
        # Test first 20 embryos of each genotype
        n_test = min(20, len(per_embryo))
        for i in range(n_test):
            row = per_embryo.iloc[i]
            test_samples.append((
                row["embryo_id"],
                int(row["frame_index"]),
                genotype,
                row
            ))
    
    print(f"Testing {len(test_samples)} embryos\n")
    print("-"*80)
    
    # Run tests
    results = []
    failures = []
    
    for i, (embryo_id, frame_index, genotype, row) in enumerate(test_samples):
        print(f"[{i+1}/{len(test_samples)}] Testing {embryo_id} frame {frame_index} ({genotype})...")
        
        success, msg, info = test_embryo_alignment(
            embryo_id, frame_index, row, data_root, aligner
        )
        
        results.append({
            "embryo_id": embryo_id,
            "frame_index": frame_index,
            "genotype": genotype,
            "stage_hpf": row["predicted_stage_hpf"],
            "success": success,
            "message": msg,
            **info
        })
        
        if not success:
            print(f"  ❌ FAILED: {msg}")
            failures.append((embryo_id, frame_index, row, msg))
            save_failure_visualization(embryo_id, frame_index, row, data_root, msg)
        else:
            retained = info.get("retained_ratio", 0)
            aligned_pix = info.get("aligned_pixels", 0)
            print(f"  ✓ SUCCESS: {aligned_pix} pixels, retained={retained:.2%}")
    
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    
    # Analysis
    results_df = pd.DataFrame(results)
    
    success_count = results_df["success"].sum()
    failure_count = len(results_df) - success_count
    
    print(f"\nTotal tested: {len(results_df)}")
    print(f"  ✓ Successful: {success_count} ({100*success_count/len(results_df):.1f}%)")
    print(f"  ❌ Failed: {failure_count} ({100*failure_count/len(results_df):.1f}%)")
    
    if failure_count > 0:
        print(f"\nFailure breakdown by error type:")
        failure_df = results_df[~results_df["success"]]
        for msg, count in failure_df["message"].value_counts().items():
            print(f"  {msg}: {count}")
        
        print(f"\nFailure breakdown by genotype:")
        for genotype, count in failure_df["genotype"].value_counts().items():
            print(f"  {genotype}: {count}")
    
    # Save results
    results_path = OUTPUT_DIR / "alignment_test_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved detailed results to: {results_path}")
    
    if failure_count > 0:
        print(f"Saved {failure_count} failure visualizations to: {OUTPUT_DIR}/")
    
    print()
    print("="*80)

if __name__ == "__main__":
    main()
