#!/usr/bin/env python3
"""
Debug script: Verify OT cost computation and canonical grid alignment.

For each sample, shows:
- Left: Reference mask with cost overlay (colored by transport cost)
- Right: Target mask

This verifies:
1. Both masks are on same canonical grid (256x576)
2. Cost is being computed on correct mask regions
3. Masks are properly aligned

Usage:
    python scripts/s02b_debug_cost_alignment.py \
        --feature-dir scripts/output/phase0_run_20260216_235308/feature_dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add morphseq root to path
MORPHSEQ_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(MORPHSEQ_ROOT))
sys.path.insert(0, str(MORPHSEQ_ROOT / "src"))

# Add ROI discovery module to path
ROI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROI_DIR))

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    import zarr
except ImportError:
    zarr = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_feature_dataset(dataset_dir: Path):
    """Load Phase 0 feature dataset from Zarr."""
    logger.info(f"Loading feature dataset from {dataset_dir}")
    
    if zarr is None:
        raise ImportError("zarr is required. pip install zarr")
    
    zarr_path = dataset_dir / "features.zarr"
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr dataset not found: {zarr_path}")
    
    store = zarr.open(str(zarr_path), mode="r")
    
    X = np.array(store["X"])  # (N, H, W, C)
    y = np.array(store["y"])  # (N,)
    mask_ref = np.array(store["mask_ref"])  # (H, W)
    
    # Load QC data
    qc_group = store.get("qc")
    outlier_flag = np.array(qc_group["outlier_flag"]) if qc_group else np.zeros(len(X), dtype=bool)
    total_cost_C = np.array(qc_group["total_cost_C"]) if qc_group else np.zeros(len(X))
    
    # Load metadata
    metadata_path = dataset_dir / "metadata.parquet"
    metadata = pd.read_parquet(metadata_path) if metadata_path.exists() else None
    
    logger.info(f"Loaded: X.shape={X.shape}, y.shape={y.shape}, mask_ref.shape={mask_ref.shape}")
    logger.info(f"Samples: {(y==0).sum()} WT, {(y==1).sum()} mutant")
    logger.info(f"Outliers: {outlier_flag.sum()}/{len(outlier_flag)}")
    
    return X, y, mask_ref, outlier_flag, total_cost_C, metadata


def plot_cost_debug_simple(
    mask_ref: np.ndarray,
    cost_map: np.ndarray,
    sample_info: dict,
    output_path: Path,
):
    """
    Create simple 2-panel debug plot:
    - Left: Reference mask with cost overlay (colored)
    - Right: Cost map alone
    
    Args:
        mask_ref: (H, W) reference mask on canonical grid
        cost_map: (H, W) cost density map on canonical grid
        sample_info: dict with sample metadata
        output_path: where to save
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    h, w = mask_ref.shape
    
    # ==== LEFT: Reference with cost overlay ====
    ax = axes[0]
    
    # Show reference mask as grayscale background
    ax.imshow(mask_ref, cmap="gray", alpha=0.4, extent=[0, w, h, 0], origin="upper")
    
    # Overlay cost map (mask out zeros)
    cost_masked = cost_map.copy()
    cost_masked[cost_map == 0] = np.nan
    
    if not np.all(np.isnan(cost_masked)):
        im = ax.imshow(
            cost_masked,
            cmap="hot",
            alpha=0.8,
            extent=[0, w, h, 0],
            origin="upper",
            interpolation="nearest",
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cost")
        
        # Statistics
        valid_cost = cost_map[cost_map > 0]
        if len(valid_cost) > 0:
            cost_mean = valid_cost.mean()
            cost_max = valid_cost.max()
            cost_nonzero_pct = 100 * len(valid_cost) / cost_map.size
            ax.text(
                0.02, 0.98,
                f"Cost: mean={cost_mean:.2e}, max={cost_max:.2e}\n"
                f"Nonzero: {cost_nonzero_pct:.1f}% of pixels",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
            )
    
    ax.set_title(f"Reference Mask (Canonical {h}×{w})\nWith Cost Overlay")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    
    # ==== RIGHT: Cost map alone ====
    ax = axes[1]
    
    if not np.all(np.isnan(cost_masked)):
        im = ax.imshow(
            cost_masked,
            cmap="hot",
            extent=[0, w, h, 0],
            origin="upper",
            interpolation="nearest",
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cost")
    else:
        ax.text(0.5, 0.5, "All costs are zero", transform=ax.transAxes, ha="center", va="center")
    
    ax.set_title(f"Cost Map Alone\n{sample_info['label']}: {sample_info['embryo_id']}")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    
    # Overall title
    fig.suptitle(
        f"Sample {sample_info['sample_id']} | Total Cost: {sample_info['total_cost']:.3e} | "
        f"Outlier: {sample_info['outlier']}",
        fontsize=12,
        fontweight="bold",
    )
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    logger.info(f"Saved: {output_path.name}")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Debug OT cost alignment")
    parser.add_argument("--feature-dir", type=Path, required=True,
                        help="Path to feature dataset directory")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of samples to plot (default: all)")
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.feature_dir.parent / "debug_cost_alignment"
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("Debug: OT Cost Alignment Check")
    logger.info("=" * 70)
    logger.info(f"Feature dataset: {args.feature_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 70)
    
    # Load feature dataset
    X, y, mask_ref, outlier_flag, total_cost_C, metadata = load_feature_dataset(args.feature_dir)
    
    N = len(X)
    n_samples = args.n_samples if args.n_samples is not None else N
    n_samples = min(n_samples, N)
    
    logger.info(f"\nGenerating debug plots for {n_samples}/{N} samples...")
    
    # Generate plots
    for i in range(n_samples):
        cost_map = X[i, :, :, 0]  # First channel is cost
        label = "WT" if y[i] == 0 else "mutant"
        
        sample_info = {
            "sample_id": i,
            "embryo_id": metadata.iloc[i]["embryo_id"] if metadata is not None else f"sample_{i}",
            "label": label,
            "total_cost": total_cost_C[i],
            "outlier": bool(outlier_flag[i]),
        }
        
        output_path = args.output_dir / f"debug_sample_{i:03d}_{label}.png"
        
        plot_cost_debug_simple(
            mask_ref=mask_ref,
            cost_map=cost_map,
            sample_info=sample_info,
            output_path=output_path,
        )
    
    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE: Debug plots generated")
    logger.info("=" * 70)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Generated {n_samples} debug plots")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
