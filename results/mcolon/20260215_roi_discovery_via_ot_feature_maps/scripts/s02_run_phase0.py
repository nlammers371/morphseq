#!/usr/bin/env python3
"""
Phase 0 Script 2: Run Full Phase 0 Pipeline

Loads CEP290 data with selected reference mask, samples WT and mutant embryos,
and runs the complete Phase 0 1D S-bin localization pipeline.

Usage:
    python scripts/s02_run_phase0.py \\
        --reference-embryo-id 20251113_A05_e01 \\
        --reference-frame-index 95 \\
        --n-wt 10 \\
        --n-mut 10 \\
        --output-dir scripts/output/phase0_run_001

Outputs:
    <output_dir>/
        - feature_dataset/  (Zarr + Parquet + JSON)
        - qc/              (QC plots and tables)
        - visualizations/   (A1-A6, B1-B3, C1, D1, E1 figures)
        - results/         (AUROC, interval, nulls JSONs)
        - summary.json     (full run summary)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add morphseq root to path (for both src/ and segmentation_sandbox/)
MORPHSEQ_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(MORPHSEQ_ROOT))
sys.path.insert(0, str(MORPHSEQ_ROOT / "src"))

# Add ROI discovery module to path
ROI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROI_DIR))

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Import UOT infrastructure
from analyze.optimal_transport_morphometrics.uot_masks import frame_mask_io as fmio

# Import Phase 0 pipeline
from run_phase0 import run_phase0
from roi_config import Phase0RunConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Data path
DATA_CSV = MORPHSEQ_ROOT / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
DEFAULT_DATA_ROOT = MORPHSEQ_ROOT / "morphseq_playground"


def load_embryo_frame(
    csv_path: Path,
    data_root: Path,
    embryo_id: str,
    frame_index: int,
) -> Tuple[np.ndarray, np.ndarray, float, pd.Series]:
    """
    Load a single embryo mask at specified frame.

    Returns
    -------
    mask : (H, W) uint8 at raw resolution
    um_per_pixel : float
    metadata : Series
    """
    usecols = [
        "embryo_id",
        "genotype",
        "experiment_date",
        "well",
        "time_int",
        "frame_index",
        "mask_rle",
        "mask_height_px",
        "mask_width_px",
        "predicted_stage_hpf",
        "Height (um)",
        "Height (px)",
        "Width (um)",
        "Width (px)",
    ]
    
    df = pd.read_csv(csv_path, usecols=usecols)
    
    row = df[
        (df["embryo_id"] == embryo_id) &
        (df["frame_index"] == frame_index)
    ]
    
    if len(row) == 0:
        raise ValueError(f"No data found for {embryo_id} frame {frame_index}")
    
    row = row.iloc[0]
    
    mask = fmio.load_mask_from_rle_counts(
        rle_counts=row["mask_rle"],
        height_px=int(row["mask_height_px"]),
        width_px=int(row["mask_width_px"]),
    )
    yolk = fmio._load_build02_aux_mask(
        data_root,
        row,
        mask.shape,
        keyword="yolk",
    )
    
    um_per_px = fmio._compute_um_per_pixel(row)
    if not np.isfinite(um_per_px):
        raise ValueError(f"Missing um_per_pixel for {embryo_id} frame {frame_index}")
    
    return mask, yolk, um_per_px, row


def sample_embryos_in_window(
    csv_path: Path,
    data_root: Path,
    stage_lo_hpf: float,
    stage_hi_hpf: float,
    target_stage_hpf: float,
    n_wt: int,
    n_mut: int,
    exclude_embryo_id: str = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    Sample WT and mutant embryos from specified stage window.

    Returns
    -------
    metadata : DataFrame
        Columns: sample_id, embryo_id, snip_id, genotype, predicted_stage_hpf, frame_index
    raw_masks : List[np.ndarray]
        Raw resolution masks (not yet on canonical grid)
    yolk_masks : List[np.ndarray]
        Raw resolution yolk masks
    um_per_pixel : (N,) array
    """
    logger.info(f"Loading embryos from {stage_lo_hpf}-{stage_hi_hpf} hpf window")
    
    usecols = [
        "embryo_id",
        "genotype",
        "experiment_date",
        "well",
        "time_int",
        "frame_index",
        "mask_rle",
        "mask_height_px",
        "mask_width_px",
        "predicted_stage_hpf",
        "snip_id",
        "Height (um)",
        "Height (px)",
        "Width (um)",
        "Width (px)",
    ]
    
    df = pd.read_csv(csv_path, usecols=usecols)
    
    # Filter to stage window
    df = df[
        (df["predicted_stage_hpf"] >= stage_lo_hpf) &
        (df["predicted_stage_hpf"] <= stage_hi_hpf)
    ]
    
    # Separate WT and mutant
    df_wt = df[df["genotype"] == "cep290_wildtype"].copy()
    df_mut = df[df["genotype"] == "cep290_homozygous"].copy()
    
    # Exclude reference embryo if specified
    if exclude_embryo_id:
        df_wt = df_wt[df_wt["embryo_id"] != exclude_embryo_id]
        df_mut = df_mut[df_mut["embryo_id"] != exclude_embryo_id]
    
    # Select one frame per embryo (closest to target stage)
    def select_frame(group_df):
        group_df["stage_diff"] = (group_df["predicted_stage_hpf"] - target_stage_hpf).abs()
        return group_df.sort_values("stage_diff").iloc[0]
    
    df_wt = df_wt.groupby("embryo_id").apply(select_frame).reset_index(drop=True)
    df_mut = df_mut.groupby("embryo_id").apply(select_frame).reset_index(drop=True)
    
    logger.info(f"Available: {len(df_wt)} WT, {len(df_mut)} mutant embryos")
    
    # Random sample
    rng = np.random.default_rng(seed)
    
    if len(df_wt) < n_wt:
        logger.warning(f"Requested {n_wt} WT but only {len(df_wt)} available")
        n_wt = len(df_wt)
    
    if len(df_mut) < n_mut:
        logger.warning(f"Requested {n_mut} mutant but only {len(df_mut)} available")
        n_mut = len(df_mut)
    
    wt_idx = rng.choice(len(df_wt), size=n_wt, replace=False)
    mut_idx = rng.choice(len(df_mut), size=n_mut, replace=False)
    
    df_wt_sampled = df_wt.iloc[wt_idx].reset_index(drop=True)
    df_mut_sampled = df_mut.iloc[mut_idx].reset_index(drop=True)
    
    # Combine
    df_combined = pd.concat([df_wt_sampled, df_mut_sampled], ignore_index=True)
    
    # Decode masks
    raw_masks = []
    yolk_masks = []
    um_per_px_list = []
    valid_indices = []
    
    for idx, row in df_combined.iterrows():
        try:
            mask = fmio.load_mask_from_rle_counts(
                rle_counts=row["mask_rle"],
                height_px=int(row["mask_height_px"]),
                width_px=int(row["mask_width_px"]),
            )
            yolk = fmio._load_build02_aux_mask(
                data_root,
                row,
                mask.shape,
                keyword="yolk",
            )
            if yolk is None or yolk.sum() == 0:
                logger.warning(f"Missing yolk mask for {row['embryo_id']}, skipping")
                continue
            um_px = fmio._compute_um_per_pixel(row)
            if not np.isfinite(um_px):
                logger.warning(f"Missing um_per_pixel for {row['embryo_id']}, skipping")
                continue
            
            raw_masks.append(mask)
            yolk_masks.append(yolk)
            um_per_px_list.append(um_px)
            valid_indices.append(idx)
        except Exception as e:
            logger.warning(f"Failed to decode mask for {row['embryo_id']}: {e}")
    
    df_combined = df_combined.loc[valid_indices].reset_index(drop=True)
    
    # Create sample_id column
    df_combined["sample_id"] = [f"sample_{i:03d}" for i in range(len(df_combined))]
    
    # Select columns for metadata
    metadata = df_combined[[
        "sample_id", "embryo_id", "snip_id", "genotype",
        "predicted_stage_hpf", "frame_index"
    ]].copy()
    
    um_per_pixel = np.array(um_per_px_list, dtype=np.float32)
    
    logger.info(f"Loaded {len(raw_masks)} embryos with yolk: {n_wt} WT + {n_mut} mutant")
    
    return metadata, raw_masks, yolk_masks, um_per_pixel


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Run Phase 0 pipeline on CEP290 data")
    parser.add_argument("--reference-embryo-id", required=True, help="Reference WT embryo ID")
    parser.add_argument("--reference-frame-index", type=int, required=True, help="Reference frame index")
    parser.add_argument("--stage-window", default="47-49", help="Stage window in hpf (e.g., '47-49')")
    parser.add_argument("--n-wt", type=int, default=10, help="Number of WT samples")
    parser.add_argument("--n-mut", type=int, default=10, help="Number of mutant samples")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Root for Build02 yolk masks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Load data and print summary without running OT")
    args = parser.parse_args()
    
    # Parse stage window
    stage_lo, stage_hi = map(float, args.stage_window.split("-"))
    target_stage = (stage_lo + stage_hi) / 2.0
    
    # Set output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path(__file__).parent / f"output/phase0_run_{timestamp}"
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("Phase 0 Script 2: Run Full Pipeline")
    logger.info("=" * 70)
    logger.info(f"Reference: {args.reference_embryo_id} frame {args.reference_frame_index}")
    logger.info(f"Stage window: {stage_lo}-{stage_hi} hpf")
    logger.info(f"Target samples: {args.n_wt} WT + {args.n_mut} mutant")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 70)
    
    # 1. Load reference mask
    logger.info("\n[1/5] Loading reference mask...")
    mask_ref_raw, yolk_ref_raw, um_per_px_ref, ref_metadata = load_embryo_frame(
        csv_path=DATA_CSV,
        data_root=args.data_root,
        embryo_id=args.reference_embryo_id,
        frame_index=args.reference_frame_index,
    )
    logger.info(f"Reference: {ref_metadata['embryo_id']} @ {ref_metadata['predicted_stage_hpf']:.2f} hpf")
    logger.info(f"Raw shape: {mask_ref_raw.shape}, {um_per_px_ref:.3f} µm/px")
    
    # NOTE: Do not pre-canonicalize the reference here. Phase 0 derives the canonical
    # reference template from the OT pipeline outputs to avoid double-canonicalization.
    
    # 2. Sample target embryos
    logger.info("\n[2/5] Sampling target embryos...")
    metadata, target_masks_raw, yolk_masks_raw, um_per_px_targets = sample_embryos_in_window(
        DATA_CSV,
        data_root=args.data_root,
        stage_lo_hpf=stage_lo,
        stage_hi_hpf=stage_hi,
        target_stage_hpf=target_stage,
        n_wt=args.n_wt,
        n_mut=args.n_mut,
        exclude_embryo_id=args.reference_embryo_id,
        seed=args.seed,
    )
    
    # 3. Build label array
    logger.info("\n[3/5] Building label array...")
    y = (metadata["genotype"] == "cep290_homozygous").astype(np.int32).values
    logger.info(f"Labels: {(y==0).sum()} WT (0), {(y==1).sum()} mutant (1)")
    metadata = metadata.copy()
    metadata["target_id"] = metadata.apply(
        lambda row: f"{row['embryo_id']}|frame_{int(row['frame_index'])}",
        axis=1,
    )
    metadata["source_id"] = f"{args.reference_embryo_id}|frame_{int(args.reference_frame_index)}"
    
    if args.dry_run:
        logger.info("\n[DRY RUN] Data loaded successfully. Exiting.")
        logger.info(f"Metadata:\n{metadata.head()}")
        return
    
    # 4. Run Phase 0 pipeline
    logger.info("\n[4/5] Running Phase 0 pipeline...")
    logger.info("This may take 10-30 minutes depending on OT computation time.")
    
    config = Phase0RunConfig()
    
    try:
        results = run_phase0(
            mask_ref=mask_ref_raw,
            target_masks=target_masks_raw,
            y=y,
            metadata_df=metadata,
            config=config,
            raw_um_per_px_ref=um_per_px_ref,
            raw_um_per_px_targets=um_per_px_targets,
            yolk_ref=yolk_ref_raw,
            yolk_targets=yolk_masks_raw,
            source_id=f"{args.reference_embryo_id}|frame_{int(args.reference_frame_index)}",
            out_dir=args.output_dir,
        )
        
        logger.info("\n[5/5] Phase 0 complete!")
        logger.info("=" * 70)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"QC: {results.get('n_qc_passed', 0)}/{results.get('n_total', 0)} samples passed")
        logger.info(f"AUROC peak: {results.get('auroc_peak', 0.0):.3f}")
        logger.info(f"Selected interval: {results.get('selected_interval', 'N/A')}")
        logger.info(f"Permutation p-value: {results.get('permutation_p', 1.0):.4f}")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Phase 0 pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
