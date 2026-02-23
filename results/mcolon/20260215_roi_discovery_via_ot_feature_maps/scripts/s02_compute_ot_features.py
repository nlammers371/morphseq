#!/usr/bin/env python3
"""
Phase 0 Step 1: Compute Optimal Transport Features

Loads reference + target embryos, computes OT transport plans,
and saves feature dataset with QC metrics.

Usage:
    python scripts/s02_compute_ot_features.py \
        --reference-embryo-id 20251112_H04_e01 \
        --reference-frame-index 39 \
        --n-wt 10 \
        --n-mut 10 \
        --output-dir scripts/output/phase0_run_001

Outputs:
    <output_dir>/
        - feature_dataset/  (Zarr with OT feature maps X, metadata, QC)
        - qc/              (QC plots and outlier diagnostics)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add morphseq root to path
MORPHSEQ_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(MORPHSEQ_ROOT))
sys.path.insert(0, str(MORPHSEQ_ROOT / "src"))

# Add ROI discovery module to path
ROI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROI_DIR))

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

# Import UOT infrastructure
from analyze.optimal_transport_morphometrics.uot_masks import frame_mask_io as fmio
from analyze.utils.coord.grids.canonical import CanonicalAligner, CanonicalGridConfig

# Import Phase 0 modules
from roi_config import Phase0RunConfig
from p0_ot_maps import generate_ot_maps
from p0_qc import run_qc_suite

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
    """Load a single embryo mask at specified frame."""
    usecols = [
        "embryo_id", "genotype", "experiment_date", "well", "time_int",
        "frame_index", "mask_rle", "mask_height_px", "mask_width_px",
        "predicted_stage_hpf", "Height (um)", "Height (px)",
        "Width (um)", "Width (px)",
    ]
    
    df = pd.read_csv(csv_path, usecols=usecols)
    row = df[(df["embryo_id"] == embryo_id) & (df["frame_index"] == frame_index)]
    
    if len(row) == 0:
        raise ValueError(f"No data found for {embryo_id} frame {frame_index}")
    
    row = row.iloc[0]
    
    mask = fmio.load_mask_from_rle_counts(
        rle_counts=row["mask_rle"],
        height_px=int(row["mask_height_px"]),
        width_px=int(row["mask_width_px"]),
    )
    yolk = fmio._load_build02_aux_mask(data_root, row, mask.shape, keyword="yolk")
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
    """Sample WT and mutant embryos from specified stage window."""
    logger.info(f"Loading embryos from {stage_lo_hpf}-{stage_hi_hpf} hpf window")
    
    usecols = [
        "embryo_id", "genotype", "experiment_date", "well", "time_int",
        "frame_index", "mask_rle", "mask_height_px", "mask_width_px",
        "predicted_stage_hpf", "snip_id", "Height (um)", "Height (px)",
        "Width (um)", "Width (px)",
    ]
    
    df = pd.read_csv(csv_path, usecols=usecols)
    df = df[(df["predicted_stage_hpf"] >= stage_lo_hpf) & (df["predicted_stage_hpf"] <= stage_hi_hpf)]
    
    df_wt = df[df["genotype"] == "cep290_wildtype"].copy()
    df_mut = df[df["genotype"] == "cep290_homozygous"].copy()
    
    if exclude_embryo_id:
        df_wt = df_wt[df_wt["embryo_id"] != exclude_embryo_id]
        df_mut = df_mut[df_mut["embryo_id"] != exclude_embryo_id]
    
    def select_frame(group_df):
        group_df["stage_diff"] = (group_df["predicted_stage_hpf"] - target_stage_hpf).abs()
        return group_df.sort_values("stage_diff").iloc[0]
    
    df_wt = df_wt.groupby("embryo_id").apply(select_frame).reset_index(drop=True)
    df_mut = df_mut.groupby("embryo_id").apply(select_frame).reset_index(drop=True)
    
    logger.info(f"Available: {len(df_wt)} WT, {len(df_mut)} mutant embryos")
    
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
    df_combined = pd.concat([df_wt_sampled, df_mut_sampled], ignore_index=True)
    
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
            yolk = fmio._load_build02_aux_mask(data_root, row, mask.shape, keyword="yolk")
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
    df_combined["sample_id"] = [f"sample_{i:03d}" for i in range(len(df_combined))]
    
    metadata = df_combined[[
        "sample_id", "embryo_id", "snip_id", "genotype",
        "predicted_stage_hpf", "frame_index"
    ]].copy()
    
    um_per_pixel = np.array(um_per_px_list, dtype=np.float32)
    
    logger.info(f"Loaded {len(raw_masks)} embryos with yolk: {n_wt} WT + {n_mut} mutant")
    
    return metadata, raw_masks, yolk_masks, um_per_pixel


def transform_reference_to_canonical(
    mask: np.ndarray,
    yolk: np.ndarray,
    um_per_pixel: float,
) -> np.ndarray:
    """Transform reference mask to canonical grid (256×576 at 10 µm/px)."""
    config = CanonicalGridConfig(reference_um_per_pixel=10.0, grid_shape_hw=(256, 576))
    aligner = CanonicalAligner.from_config(config)
    
    if yolk is None or yolk.sum() == 0:
        raise ValueError("Reference yolk mask is missing or empty")
    
    canonical_mask, _, _ = aligner.align(
        mask=mask.astype(bool),
        yolk=yolk.astype(bool),
        original_um_per_px=um_per_pixel,
        use_yolk=True,
    )
    return canonical_mask.astype(np.uint8)


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Compute OT features for Phase 0")
    parser.add_argument("--reference-embryo-id", required=True)
    parser.add_argument("--reference-frame-index", type=int, required=True)
    parser.add_argument("--stage-window", default="47-49")
    parser.add_argument("--n-wt", type=int, default=10)
    parser.add_argument("--n-mut", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    stage_lo, stage_hi = map(float, args.stage_window.split("-"))
    target_stage = (stage_lo + stage_hi) / 2.0
    
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path(__file__).parent / f"output/phase0_run_{timestamp}"
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("Phase 0 Step 1: Compute OT Features")
    logger.info("=" * 70)
    logger.info(f"Reference: {args.reference_embryo_id} frame {args.reference_frame_index}")
    logger.info(f"Stage window: {stage_lo}-{stage_hi} hpf")
    logger.info(f"Target samples: {args.n_wt} WT + {args.n_mut} mutant")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 70)
    
    # Load reference mask
    logger.info("\n[1/4] Loading reference mask...")
    mask_ref_raw, yolk_ref_raw, um_per_px_ref, ref_metadata = load_embryo_frame(
        DATA_CSV, args.data_root,
        args.reference_embryo_id, args.reference_frame_index,
    )
    # Keep two representations:
    # 1) raw reference mask for OT (run_single_ot canonicalizes internally)
    # 2) canonical reference mask for QC overlays + dataset mask_ref contract
    mask_ref_canonical = transform_reference_to_canonical(mask_ref_raw, yolk_ref_raw, um_per_px_ref)
    logger.info(f"Reference: {ref_metadata['embryo_id']} @ {ref_metadata['predicted_stage_hpf']:.2f} hpf")
    
    # Sample target embryos
    logger.info("\n[2/4] Sampling target embryos...")
    metadata, target_masks_raw, yolk_masks_raw, um_per_px_targets = sample_embryos_in_window(
        DATA_CSV, args.data_root,
        stage_lo, stage_hi, target_stage,
        args.n_wt, args.n_mut,
        exclude_embryo_id=args.reference_embryo_id,
        seed=args.seed,
    )
    
    # Build label array
    y = (metadata["genotype"] == "cep290_homozygous").astype(np.int32).values
    logger.info(f"Labels: {(y==0).sum()} WT (0), {(y==1).sum()} mutant (1)")
    
    # Compute OT transport plans
    logger.info("\n[3/4] Computing OT transport plans...")
    logger.info("This may take 10-30 minutes depending on sample size.")
    
    config = Phase0RunConfig()
    sample_ids = metadata["sample_id"].tolist()
    
    source_id = f"{args.reference_embryo_id}|frame_{int(args.reference_frame_index)}"
    metadata = metadata.copy()
    metadata["target_id"] = metadata.apply(
        lambda row: f"{row['embryo_id']}|frame_{int(row['frame_index'])}",
        axis=1,
    )
    metadata["source_id"] = source_id
    metadata["source_embryo_id"] = str(args.reference_embryo_id)
    metadata["source_frame_index"] = int(args.reference_frame_index)

    X, total_cost_C, mask_ref_aligned_ot, target_masks_canonical, alignment_debug_df = generate_ot_maps(
        mask_ref_raw,
        target_masks_raw,
        sample_ids,
        raw_um_per_px_ref=um_per_px_ref,
        raw_um_per_px_targets=um_per_px_targets,
        yolk_ref=yolk_ref_raw,
        yolk_targets=yolk_masks_raw,
        feature_set=config.feature_set,
        source_id=source_id,
        target_ids=metadata["target_id"].tolist(),
        return_aligned_masks=True,
        collect_debug=True,
        strict_debug_ids=True,
        return_debug_df=True,
    )
    
    logger.info(f"Generated OT feature maps: X.shape = {X.shape}")
    logger.info(f"Mean transport cost: {total_cost_C.mean():.3f}")

    # Use the exact aligned source mask from OT (not an independently reconstructed transform).
    mask_ref_for_outputs = mask_ref_aligned_ot.astype(np.uint8)
    
    # QC and filtering
    logger.info("\n[4/4] Running QC suite...")
    qc_dir = args.output_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    
    outlier_flag, qc_stats = run_qc_suite(
        X, y, total_cost_C, mask_ref_for_outputs, metadata,
        sample_ids, out_dir=qc_dir,
        iqr_multiplier=config.dataset.iqr_multiplier,
        target_masks_canonical=target_masks_canonical,
        alignment_debug_df=alignment_debug_df,
    )
    
    n_outliers = outlier_flag.sum()
    logger.info(f"QC complete: {n_outliers}/{len(outlier_flag)} samples flagged as outliers")
    
    # Save feature dataset
    logger.info("\nSaving feature dataset...")
    from roi_feature_dataset import Phase0FeatureDatasetBuilder
    
    dataset_dir = args.output_dir / "feature_dataset"
    builder = Phase0FeatureDatasetBuilder(
        out_dir=dataset_dir,
        feature_set=config.feature_set,
        config=config.dataset,
        stage_window=f"{stage_lo}-{stage_hi}",
        reference_mask_id=source_id,
    )
    
    # Build without S-coordinate (we'll do that in next script)
    builder.build(
        X=X,
        y=y,
        mask_ref=mask_ref_for_outputs,
        metadata_df=metadata,
        total_cost_C=total_cost_C,
        target_masks_canonical=target_masks_canonical,
        alignment_debug_df=alignment_debug_df,
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE: OT Features Computed")
    logger.info("=" * 70)
    logger.info(f"Feature dataset: {dataset_dir}")
    logger.info(f"QC diagnostics: {qc_dir}")
    logger.info(f"Samples: {len(metadata)} total, {n_outliers} outliers")
    logger.info(f"Mean cost: {total_cost_C.mean():.3f}")
    logger.info("\nNext step: Run s03_visualize_differences.py to create heat maps")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
