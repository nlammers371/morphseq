#!/usr/bin/env python3
"""
Re-generate QC and viz plots from a cached phase0 feature_dataset.

Usage:
    python scripts/reviz_run003.py

Loads feature_dataset from phase0_run_003 and re-runs Step 2 (QC) and
Step 3 (visualization) with the fixed origin= contract.
"""

from __future__ import annotations
import sys
from pathlib import Path

MORPHSEQ_ROOT = Path(__file__).resolve().parents[4]
ROI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MORPHSEQ_ROOT))
sys.path.insert(0, str(MORPHSEQ_ROOT / "src"))
sys.path.insert(0, str(ROI_DIR))

import json
import logging
import numpy as np
import pandas as pd
import zarr

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RUN_DIR = Path(__file__).resolve().parent / "output" / "phase0_run_003"
DATASET_DIR = RUN_DIR / "feature_dataset"

# ── load cached data ─────────────────────────────────────────────────────────
logger.info("Loading feature dataset from %s", DATASET_DIR)

z = zarr.open(str(DATASET_DIR / "features.zarr"), "r")

X = z["X"][:]                                    # (N, H, W, C)
total_cost_C = z["qc/total_cost_C"][:]           # (N,)
mask_ref_canonical = z["mask_ref"][:]            # (H, W)
S_map_cached = z["optional/S_map_ref"][:]        # (H, W)

if "optional/target_masks_canonical" in z:
    aligned_target_masks = z["optional/target_masks_canonical"][:]  # (N, H, W)
else:
    aligned_target_masks = None
    logger.warning("target_masks_canonical not in zarr — QC-2 overlays will be skipped")

metadata_df = pd.read_parquet(DATASET_DIR / "metadata.parquet")
y = z["y"][:]  # (N,) int — labels from zarr
sample_ids = metadata_df["sample_id"].tolist()

alignment_debug_df = None
debug_path = DATASET_DIR / "alignment_debug.parquet"
if debug_path.exists():
    alignment_debug_df = pd.read_parquet(debug_path)
    logger.info("Loaded alignment debug: %d rows", len(alignment_debug_df))

with open(DATASET_DIR / "manifest.json") as f:
    manifest = json.load(f)
logger.info("Manifest: %s", {k: v for k, v in manifest.items() if k != "sample_ids"})

logger.info("X shape: %s, mask_ref: %s, N=%d", X.shape, mask_ref_canonical.shape, len(y))

# ── re-run QC (step 2) ────────────────────────────────────────────────────────
from p0_qc import run_qc_suite

qc_dir = RUN_DIR / "qc"
logger.info("Re-running QC suite → %s", qc_dir)

outlier_flag, qc_stats = run_qc_suite(
    X, y, total_cost_C, mask_ref_canonical, metadata_df, sample_ids,
    out_dir=qc_dir,
    iqr_multiplier=2.0,
    target_masks_canonical=aligned_target_masks,
    alignment_debug_df=alignment_debug_df,
)
logger.info("QC done: %d outliers flagged", qc_stats["n_outliers"])

# ── re-run viz (step 3) ───────────────────────────────────────────────────────
from p0_viz import plot_cost_density_suite, plot_s_map

viz_dir = RUN_DIR / "viz"
viz_dir.mkdir(parents=True, exist_ok=True)
logger.info("Re-running viz suite → %s", viz_dir)

# Use cached S_map (avoid re-running centerline computation)
plot_s_map(S_map_cached, mask_ref_canonical, save_path=viz_dir / "s_map_ref.png")
logger.info("Saved s_map_ref.png")

plot_cost_density_suite(
    X, y, mask_ref_canonical, outlier_flag,
    sigma_grid=(1.0, 2.0, 4.0),
    save_dir=viz_dir,
)
logger.info("Saved cost density suite")

logger.info("Done — check %s", RUN_DIR)
