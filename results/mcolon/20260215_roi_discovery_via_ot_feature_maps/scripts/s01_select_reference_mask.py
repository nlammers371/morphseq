#!/usr/bin/env python3
"""
Phase 0 Script 1: Select WT Reference Mask

Loads CEP290 embryo data, filters to 47-49 hpf developmental window,
transforms WT masks to canonical grid (256×576), and ranks by IoU overlap.
Visualizes top-3 candidates for user selection.

Usage:
    python scripts/s01_select_reference_mask.py

Outputs:
    scripts/output/reference_mask_candidates/
        - top_candidate_0.png
        - top_candidate_1.png
        - top_candidate_2.png
        - candidates_summary.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add morphseq root to path (for both src/ and segmentation_sandbox/)
MORPHSEQ_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(MORPHSEQ_ROOT))
sys.path.insert(0, str(MORPHSEQ_ROOT / "src"))

import logging
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.ndimage import binary_erosion

# Import UOT infrastructure
from analyze.optimal_transport_morphometrics.uot_masks import frame_mask_io as fmio
from analyze.utils.coord.grids.canonical import CanonicalAligner, CanonicalGridConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



# Data path
DATA_CSV = MORPHSEQ_ROOT / "results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv"
DEFAULT_DATA_ROOT = MORPHSEQ_ROOT / "morphseq_playground"
OUTPUT_DIR = Path(__file__).parent / "output/reference_mask_candidates"


def load_wt_masks_at_stage(
    csv_path: Path,
    data_root: Path,
    stage_lo_hpf: float = 47.0,
    stage_hi_hpf: float = 49.0,
    target_stage_hpf: float = 48.0,
) -> Tuple[pd.DataFrame, List[np.ndarray], List[np.ndarray], List[float]]:
    """
    Load WT embryo masks from CEP290 dataset at specified developmental stage.

    For each WT embryo, selects the frame closest to target_stage_hpf within
    the [stage_lo_hpf, stage_hi_hpf] window.

    Returns
    -------
    metadata : DataFrame
        Columns: embryo_id, frame_index, predicted_stage_hpf, genotype,
                 mask_height_px, mask_width_px, um_per_pixel
    raw_masks : List[np.ndarray]
        Raw resolution masks (not yet on canonical grid)
    yolk_masks : List[np.ndarray]
        Raw resolution yolk masks
    um_per_pixel : List[float]
        Physical resolution for each mask
    """
    logger.info(f"Loading CSV from {csv_path}")
    
    # Load minimal columns for efficiency
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
    logger.info(f"Loaded {len(df):,} rows")
    
    # Filter to WT and stage window
    df_wt = df[
        (df["genotype"] == "cep290_wildtype") &
        (df["predicted_stage_hpf"] >= stage_lo_hpf) &
        (df["predicted_stage_hpf"] <= stage_hi_hpf)
    ].copy()
    
    logger.info(f"Filtered to {len(df_wt):,} WT rows in {stage_lo_hpf}-{stage_hi_hpf} hpf window")
    
    # Select one frame per embryo (closest to target stage)
    df_wt["stage_diff"] = (df_wt["predicted_stage_hpf"] - target_stage_hpf).abs()
    df_wt = df_wt.sort_values("stage_diff").groupby("embryo_id").first().reset_index()
    
    logger.info(f"Selected {len(df_wt)} unique WT embryos (one frame each)")
    
    # Compute um_per_pixel with fallback
    df_wt["um_per_pixel"] = df_wt.apply(fmio._compute_um_per_pixel, axis=1)
    
    # Decode RLE masks
    raw_masks = []
    yolk_masks = []
    um_per_px_list = []
    valid_indices = []
    
    for idx, row in df_wt.iterrows():
        try:
            mask = fmio.load_mask_from_rle_counts(
                rle_counts=row["mask_rle"],
                height_px=int(row["mask_height_px"]),
                width_px=int(row["mask_width_px"]),
            )
            um_per_px = float(row["um_per_pixel"])
            if not np.isfinite(um_per_px):
                logger.warning(f"Missing um_per_pixel for {row['embryo_id']}, skipping")
                continue
            yolk = fmio._load_build02_aux_mask(
                data_root,
                row,
                mask.shape,
                keyword="yolk",
            )
            if yolk is None or yolk.sum() == 0:
                logger.warning(f"Missing yolk mask for {row['embryo_id']}, skipping")
                continue
            raw_masks.append(mask)
            yolk_masks.append(yolk)
            um_per_px_list.append(um_per_px)
            valid_indices.append(idx)
        except Exception as e:
            logger.warning(f"Failed to decode mask for {row['embryo_id']}: {e}")
    
    df_wt = df_wt.loc[valid_indices].reset_index(drop=True)
    logger.info(f"Successfully decoded {len(raw_masks)} masks with yolk")
    
    return df_wt, raw_masks, yolk_masks, um_per_px_list


def transform_to_canonical(
    raw_masks: List[np.ndarray],
    yolk_masks: List[np.ndarray],
    um_per_pixel: List[float],
) -> List[np.ndarray]:
    """
    Transform raw masks to canonical grid (256×576 at 10 µm/px).

    Uses CanonicalAligner with yolk-based alignment and PCA orientation.
    """
    config = CanonicalGridConfig(
        reference_um_per_pixel=10.0,
        grid_shape_hw=(256, 576),
    )
    aligner = CanonicalAligner.from_config(config)
    
    canonical_masks = []
    for i, (mask, yolk, um_px) in enumerate(zip(raw_masks, yolk_masks, um_per_pixel)):
        try:
            # Align to canonical grid with yolk-based alignment
            aligned_mask, _, _ = aligner.align(
                mask=mask.astype(bool),
                yolk=yolk.astype(bool),
                original_um_per_px=um_px,
                use_yolk=True,
            )
            aligned_mask = aligned_mask.astype(np.uint8)
            if aligned_mask.sum() == 0:
                logger.warning(f"Aligned mask is empty for index {i}, skipping")
                continue
            canonical_masks.append(aligned_mask)
        except Exception as e:
            logger.warning(f"Failed to align mask {i}: {e}")
            # Create empty mask as fallback
            continue
    
    logger.info(f"Transformed {len(canonical_masks)} masks to canonical grid")
    return canonical_masks


def compute_pairwise_iou(masks: List[np.ndarray]) -> np.ndarray:
    """
    Compute pairwise IoU (Intersection over Union) for a list of binary masks.

    Returns
    -------
    iou_matrix : (N, N) array
        IoU[i, j] = IoU between masks[i] and masks[j]
    """
    n = len(masks)
    iou_matrix = np.zeros((n, n), dtype=np.float32)
    
    for i in range(n):
        for j in range(i, n):
            intersection = np.logical_and(masks[i], masks[j]).sum()
            union = np.logical_or(masks[i], masks[j]).sum()
            
            if union > 0:
                iou = intersection / union
            else:
                iou = 0.0
            
            iou_matrix[i, j] = iou
            iou_matrix[j, i] = iou
    
    return iou_matrix


def rank_by_iou(iou_matrix: np.ndarray) -> np.ndarray:
    """
    Rank masks by mean IoU with all other masks.

    Returns
    -------
    ranked_indices : (N,) array
        Indices sorted by mean IoU (highest first)
    """
    # Exclude self-IoU (diagonal = 1.0) from mean
    n = iou_matrix.shape[0]
    mean_iou = np.zeros(n)
    
    for i in range(n):
        other_iou = np.concatenate([iou_matrix[i, :i], iou_matrix[i, i+1:]])
        mean_iou[i] = other_iou.mean() if len(other_iou) > 0 else 0.0
    
    ranked_indices = np.argsort(mean_iou)[::-1]  # Descending order
    return ranked_indices


def visualize_candidate(
    mask: np.ndarray,
    metadata: pd.Series,
    mean_iou: float,
    rank: int,
    output_path: Path,
):
    """
    Visualize a single reference mask candidate.

    Shows the canonical grid mask with embryo outline and metadata annotation.
    """
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(12, 5))

    mask_bin = (mask > 0).astype(np.uint8)

    # Full-frame view
    ax_full.imshow(mask_bin, cmap="gray", origin="upper", vmin=0, vmax=1, interpolation="nearest")

    # Draw embryo outline
    outline = mask_bin.astype(bool) ^ binary_erosion(mask_bin.astype(bool), iterations=2)
    ax_full.contour(outline, colors="cyan", linewidths=1.5, levels=[0.5])
    
    # Annotate
    title = (
        f"Rank {rank+1} Reference Candidate\n"
        f"Embryo: {metadata['embryo_id']} | Frame: {metadata['frame_index']} | "
        f"Stage: {metadata['predicted_stage_hpf']:.2f} hpf\n"
        f"Mean IoU: {mean_iou:.3f} | Area: {mask.sum():,} px"
    )
    ax_full.set_title(title, fontsize=11, fontweight="bold")
    ax_full.set_xlabel("X (canonical grid)", fontsize=9)
    ax_full.set_ylabel("Y (canonical grid)", fontsize=9)
    ax_full.grid(False)

    # Add scale bar (10 µm/px → 50 px = 500 µm)
    scale_bar_px = 50
    scale_bar_um = scale_bar_px * 10
    ax_full.add_patch(Rectangle((500, 230), scale_bar_px, 8, facecolor="white", edgecolor="black", linewidth=1))
    ax_full.text(525, 245, f"{scale_bar_um} µm", fontsize=8, ha="center", va="top", color="white", fontweight="bold")

    # Zoomed view around mask bounding box
    ys, xs = np.where(mask_bin > 0)
    if ys.size > 0:
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        pad = 10
        y0p = max(y0 - pad, 0)
        y1p = min(y1 + pad, mask_bin.shape[0] - 1)
        x0p = max(x0 - pad, 0)
        x1p = min(x1 + pad, mask_bin.shape[1] - 1)
        zoom = mask_bin[y0p:y1p + 1, x0p:x1p + 1]
        ax_full.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0 + 1,
                y1 - y0 + 1,
                edgecolor="yellow",
                facecolor="none",
                linewidth=1,
            )
        )
        ax_zoom.imshow(zoom, cmap="gray", origin="upper", vmin=0, vmax=1, interpolation="nearest")
        ax_zoom.set_title("Zoomed mask", fontsize=10)
    else:
        ax_zoom.text(0.5, 0.5, "Empty mask", ha="center", va="center", fontsize=10)
    ax_zoom.set_xlabel("X (zoom)", fontsize=9)
    ax_zoom.set_ylabel("Y (zoom)", fontsize=9)
    ax_zoom.grid(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved candidate {rank+1} visualization to {output_path}")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Select WT reference mask using yolk alignment")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Root for Build02 yolk masks")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Phase 0 Script 1: WT Reference Mask Selection")
    logger.info("=" * 60)
    
    # 1. Load WT masks at 47-49 hpf (with yolk masks)
    metadata, raw_masks, yolk_masks, um_per_px = load_wt_masks_at_stage(
        DATA_CSV,
        data_root=args.data_root,
        stage_lo_hpf=47.0,
        stage_hi_hpf=49.0,
        target_stage_hpf=48.0,
    )
    
    if len(raw_masks) < 3:
        logger.error(
            f"Only {len(raw_masks)} WT masks found with yolk. Need at least 3 for ranking. "
            f"Check --data-root (current: {args.data_root})."
        )
        return
    
    # 2. Transform to canonical grid
    canonical_masks = transform_to_canonical(raw_masks, yolk_masks, um_per_px)
    
    # 3. Compute pairwise IoU
    logger.info("Computing pairwise IoU...")
    iou_matrix = compute_pairwise_iou(canonical_masks)
    
    # 4. Rank by mean IoU
    ranked_indices = rank_by_iou(iou_matrix)
    
    # Compute mean IoU for each mask
    n = len(canonical_masks)
    mean_ious = np.array([
        np.concatenate([iou_matrix[i, :i], iou_matrix[i, i+1:]]).mean()
        for i in range(n)
    ])
    
    # 5. Visualize top 3 candidates
    logger.info("Visualizing top 3 candidates...")
    top_k = min(3, len(ranked_indices))
    
    summary_rows = []
    for rank in range(top_k):
        idx = ranked_indices[rank]
        visualize_candidate(
            mask=canonical_masks[idx],
            metadata=metadata.iloc[idx],
            mean_iou=mean_ious[idx],
            rank=rank,
            output_path=OUTPUT_DIR / f"top_candidate_{rank}.png",
        )
        
        summary_rows.append({
            "rank": rank + 1,
            "embryo_id": metadata.iloc[idx]["embryo_id"],
            "frame_index": metadata.iloc[idx]["frame_index"],
            "predicted_stage_hpf": metadata.iloc[idx]["predicted_stage_hpf"],
            "mean_iou": mean_ious[idx],
            "mask_area_px": canonical_masks[idx].sum(),
        })
    
    # 6. Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "candidates_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved summary to {summary_path}")
    
    # 7. Report statistics
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total WT embryos in 47-49 hpf window: {len(raw_masks)}")
    logger.info(f"Top-3 candidates ranked by mean IoU:")
    for _, row in summary_df.iterrows():
        logger.info(
            f"  Rank {row['rank']}: {row['embryo_id']} frame {row['frame_index']} "
            f"(stage {row['predicted_stage_hpf']:.2f} hpf, IoU {row['mean_iou']:.3f})"
        )
    
    logger.info("\nNext steps:")
    logger.info("  1. Review visualizations in scripts/output/reference_mask_candidates/")
    logger.info("  2. Select one reference by embryo_id + frame_index")
    logger.info("  3. Pass to s02_run_phase0.py with --reference-embryo-id and --reference-frame-index")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
