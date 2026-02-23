#!/usr/bin/env python3
"""
Plot per-outlier QC diagnostics from a built Phase 0 feature dataset.

Usage:
    PYTHONPATH=src /net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python \
      results/mcolon/20260215_roi_discovery_via_ot_feature_maps/scripts/s02c_plot_outlier_diagnostics.py \
      --feature-dir results/mcolon/20260215_roi_discovery_via_ot_feature_maps/scripts/output/phase0_rerun_simplified/feature_dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import logging
import numpy as np
import pandas as pd

try:
    import zarr
except ImportError:
    zarr = None

# Add morphseq root to path
MORPHSEQ_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(MORPHSEQ_ROOT))
sys.path.insert(0, str(MORPHSEQ_ROOT / "src"))

# Add ROI discovery module to path
ROI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROI_DIR))

from p0_qc import compute_iqr_outliers, plot_qc4_outlier_diagnostics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot outlier alignment/mask diagnostics")
    parser.add_argument("--feature-dir", type=Path, required=True, help="Path to feature_dataset directory")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional custom output dir")
    parser.add_argument(
        "--iqr-multiplier",
        type=float,
        default=None,
        help="Optional override: recompute outliers from total_cost_C with this IQR multiplier",
    )
    args = parser.parse_args()

    if zarr is None:
        raise ImportError("zarr is required to read features.zarr")

    feature_dir = args.feature_dir
    zarr_path = feature_dir / "features.zarr"
    if not zarr_path.exists():
        raise FileNotFoundError(f"Missing Zarr store: {zarr_path}")

    store = zarr.open(str(zarr_path), mode="r")
    X = np.array(store["X"])
    mask_ref = np.array(store["mask_ref"])
    outlier_flag = np.array(store["qc/outlier_flag"])
    total_cost_C = np.array(store["qc/total_cost_C"])
    target_masks_canonical = np.array(store["optional/target_masks_canonical"])
    if args.iqr_multiplier is not None:
        outlier_flag, stats = compute_iqr_outliers(total_cost_C, multiplier=float(args.iqr_multiplier))
        logger.info(
            "Recomputed outliers with iqr_multiplier=%.3f: %d/%d (bounds=[%.4f, %.4f])",
            float(args.iqr_multiplier),
            int(outlier_flag.sum()),
            int(len(outlier_flag)),
            float(stats["lower_bound"]),
            float(stats["upper_bound"]),
        )

    metadata_path = feature_dir / "metadata.parquet"
    alignment_path = feature_dir / "alignment_debug.parquet"
    metadata_df = pd.read_parquet(metadata_path)
    alignment_debug_df = pd.read_parquet(alignment_path) if alignment_path.exists() else None

    sample_ids = metadata_df["sample_id"].astype(str).tolist()
    out_dir = args.output_dir if args.output_dir is not None else (feature_dir.parent / "qc_outlier_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loaded feature dataset: {feature_dir}")
    logger.info(f"Outliers: {int(outlier_flag.sum())}/{len(outlier_flag)}")
    summary_df = plot_qc4_outlier_diagnostics(
        X=X,
        total_cost_C=total_cost_C,
        mask_ref=mask_ref,
        sample_ids=sample_ids,
        metadata_df=metadata_df,
        outlier_flag=outlier_flag,
        target_masks_canonical=target_masks_canonical,
        alignment_debug_df=alignment_debug_df,
        out_dir=out_dir,
    )

    if not summary_df.empty:
        logger.info("Outlier summary:")
        for _, row in summary_df.iterrows():
            logger.info(
                "  %s | %s | C=%.4f | %s",
                row["sample_id"],
                row["embryo_id"],
                row["total_cost_C"],
                row["diagnostic_label"],
            )
    logger.info(f"Done. Outputs: {out_dir}")


if __name__ == "__main__":
    main()
