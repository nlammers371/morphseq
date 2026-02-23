#!/usr/bin/env python3
"""
Phase 0 Script 1b: Visualize WT/Mutant/Reference Masks

Loads CEP290 embryo masks, aligns them to canonical grid using yolk masks,
and renders separate grids for WT, mutant, and the chosen reference.

Usage:
    python scripts/s01b_visualize_masks.py \
        --reference-embryo-id 20251112_H04_e01 \
        --reference-frame-index 39 \
        --n-wt 10 \
        --n-mut 10 \
        --output-dir scripts/output/mask_qc
"""

from __future__ import annotations

import argparse
import sys
from math import ceil
from pathlib import Path
from typing import List, Tuple

# Add morphseq root to path (for both src/ and segmentation_sandbox/)
MORPHSEQ_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(MORPHSEQ_ROOT))
sys.path.insert(0, str(MORPHSEQ_ROOT / "src"))

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion

from analyze.optimal_transport_morphometrics.uot_masks import frame_mask_io as fmio
from analyze.utils.coord.grids.canonical import CanonicalAligner, CanonicalGridConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_CSV = MORPHSEQ_ROOT / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
DEFAULT_DATA_ROOT = MORPHSEQ_ROOT / "morphseq_playground"


def _load_row(df: pd.DataFrame, embryo_id: str, frame_index: int) -> pd.Series:
    row = df[
        (df["embryo_id"] == embryo_id) &
        (df["frame_index"] == frame_index)
    ]
    if row.empty:
        raise ValueError(f"No row found for {embryo_id} frame {frame_index}")
    return row.iloc[0]


def _decode_mask_and_yolk(row: pd.Series, data_root: Path) -> Tuple[np.ndarray, np.ndarray, float]:
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
    return mask, yolk, um_per_px


def _align_to_canonical(mask: np.ndarray, yolk: np.ndarray, um_per_px: float) -> np.ndarray:
    if yolk is None or yolk.sum() == 0:
        raise ValueError("Missing yolk mask for alignment")
    config = CanonicalGridConfig(reference_um_per_pixel=10.0, grid_shape_hw=(256, 576))
    aligner = CanonicalAligner.from_config(config)
    aligned_mask, _, _ = aligner.align(
        mask=mask.astype(bool),
        yolk=yolk.astype(bool),
        original_um_per_px=um_per_px,
        use_yolk=True,
    )
    return aligned_mask.astype(np.uint8)


def _plot_grid(
    masks: List[np.ndarray],
    labels: List[str],
    title: str,
    output_path: Path,
    ncols: int = 5,
) -> None:
    n = len(masks)
    logger.info(f"_plot_grid called with {n} masks for '{title}'")
    if n == 0:
        logger.warning(f"No masks to plot for {title}")
        return
    
    # Log mask statistics
    for i, mask in enumerate(masks[:3]):  # Log first 3
        logger.info(f"  Mask {i}: {mask.sum()} pixels, shape {mask.shape}, dtype {mask.dtype}, range [{mask.min()}, {mask.max()}]")
    
    ncols = min(ncols, n)
    nrows = ceil(n / ncols)

    # Show full canonical grid (256×576) to verify alignment and edge distances
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    for idx in range(nrows * ncols):
        ax = axes.flat[idx]
        if idx >= n:
            ax.axis("off")
            continue
        
        mask = masks[idx]
        
        # Create outline for better visibility
        outline = mask.astype(bool) ^ binary_erosion(mask.astype(bool), iterations=1)
        
        # Display FULL canonical grid (no cropping)
        ax.imshow(mask, cmap="gray", origin="upper", vmin=0, vmax=1)
        ax.contour(outline, colors="cyan", linewidths=1.0, levels=[0.5])
        
        # Show pixel count for verification
        pixel_count = mask.sum()
        ax.set_title(f"{labels[idx]}\n{pixel_count} pixels", fontsize=8)
        ax.axis("off")

    fig.suptitle(f"{title} (full canonical grid: {masks[0].shape[0]}×{masks[0].shape[1]})", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {title} grid to {output_path}")


def _sample_by_stage(
    df: pd.DataFrame,
    genotype: str,
    stage_lo: float,
    stage_hi: float,
    target_stage: float,
    n_samples: int,
    seed: int,
    exclude_embryo_id: str = None,
) -> pd.DataFrame:
    subset = df[
        (df["genotype"] == genotype) &
        (df["predicted_stage_hpf"] >= stage_lo) &
        (df["predicted_stage_hpf"] <= stage_hi)
    ].copy()
    if exclude_embryo_id:
        subset = subset[subset["embryo_id"] != exclude_embryo_id]

    def select_frame(group_df: pd.DataFrame) -> pd.Series:
        group_df = group_df.copy()
        group_df["stage_diff"] = (group_df["predicted_stage_hpf"] - target_stage).abs()
        return group_df.sort_values("stage_diff").iloc[0]

    per_embryo = subset.groupby("embryo_id").apply(select_frame).reset_index(drop=True)
    if len(per_embryo) < n_samples:
        logger.warning(f"Requested {n_samples} {genotype} but only {len(per_embryo)} available")
        n_samples = len(per_embryo)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(per_embryo), size=n_samples, replace=False)
    return per_embryo.iloc[idx].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize WT/mutant/reference masks")
    parser.add_argument("--reference-embryo-id", required=True)
    parser.add_argument("--reference-frame-index", type=int, required=True)
    parser.add_argument("--stage-window", default="47-49")
    parser.add_argument("--n-wt", type=int, default=10)
    parser.add_argument("--n-mut", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("scripts/output/mask_qc"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_lo, stage_hi = map(float, args.stage_window.split("-"))
    target_stage = (stage_lo + stage_hi) / 2.0

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
    df = pd.read_csv(DATA_CSV, usecols=usecols)

    # Reference
    ref_row = _load_row(df, args.reference_embryo_id, args.reference_frame_index)
    ref_mask, ref_yolk, ref_um = _decode_mask_and_yolk(ref_row, args.data_root)
    ref_canon = _align_to_canonical(ref_mask, ref_yolk, ref_um)

    # Sample WT and mutant
    wt_df = _sample_by_stage(
        df,
        genotype="cep290_wildtype",
        stage_lo=stage_lo,
        stage_hi=stage_hi,
        target_stage=target_stage,
        n_samples=args.n_wt,
        seed=args.seed,
        exclude_embryo_id=args.reference_embryo_id,
    )
    mut_df = _sample_by_stage(
        df,
        genotype="cep290_homozygous",
        stage_lo=stage_lo,
        stage_hi=stage_hi,
        target_stage=target_stage,
        n_samples=args.n_mut,
        seed=args.seed,
        exclude_embryo_id=args.reference_embryo_id,
    )

    def build_set(sample_df: pd.DataFrame) -> Tuple[List[np.ndarray], List[str]]:
        masks = []
        labels = []
        logger.info(f"Processing {len(sample_df)} embryos...")
        # FIX: iterate over dataframe directly instead of using itertuples
        # itertuples() + _asdict() + pd.Series() conversion was corrupting data
        for idx, row in sample_df.iterrows():
            logger.info(f"  Loading {row['embryo_id']} frame {row['frame_index']}...")
            mask, yolk, um = _decode_mask_and_yolk(row, args.data_root)
            logger.info(f"    Original mask: {mask.sum()} pixels")
            if yolk is None or yolk.sum() == 0:
                logger.warning(f"    Missing yolk mask, skipping")
                continue
            logger.info(f"    Yolk mask: {yolk.sum()} pixels")
            aligned = _align_to_canonical(mask, yolk, um)
            logger.info(f"    Aligned mask: {aligned.sum()} pixels")
            masks.append(aligned)
            labels.append(f"{row['embryo_id']}\nframe {row['frame_index']}")
        logger.info(f"Successfully loaded {len(masks)} masks")
        return masks, labels

    wt_masks, wt_labels = build_set(wt_df)
    mut_masks, mut_labels = build_set(mut_df)

    logger.info(f"Plotting reference mask: {ref_canon.sum()} pixels, shape {ref_canon.shape}")
    _plot_grid([ref_canon], [f"{args.reference_embryo_id}\nframe {args.reference_frame_index}"],
               "Reference (canonical)", args.output_dir / "reference_mask.png", ncols=1)
    
    logger.info(f"Plotting {len(wt_masks)} WT masks")
    if len(wt_masks) > 0:
        logger.info(f"  Example WT mask: {wt_masks[0].sum()} pixels, shape {wt_masks[0].shape}")
    _plot_grid(wt_masks, wt_labels, "WT masks (canonical)", args.output_dir / "wt_masks.png")
    
    logger.info(f"Plotting {len(mut_masks)} mutant masks")
    if len(mut_masks) > 0:
        logger.info(f"  Example mutant mask: {mut_masks[0].sum()} pixels, shape {mut_masks[0].shape}")
    _plot_grid(mut_masks, mut_labels, "Mutant masks (canonical)", args.output_dir / "mutant_masks.png")


if __name__ == "__main__":
    main()
