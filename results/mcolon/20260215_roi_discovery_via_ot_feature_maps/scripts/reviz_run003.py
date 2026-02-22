#!/usr/bin/env python3
"""
Re-generate QC and viz plots from a cached phase0 feature_dataset.

Usage:
    python scripts/reviz_run003.py [run_name]

    run_name: e.g. phase0_run_003 (default) or phase0_run_004

Loads feature_dataset via Phase0Loader and re-runs Step 2 (QC) and
Step 3 (visualization) with the enforced coordinate contract.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

run_name = sys.argv[1] if len(sys.argv) > 1 else "phase0_run_003"
RUN_DIR = Path(__file__).resolve().parent / "output" / run_name
logger.info("Run: %s", run_name)

# ── load cached data via Phase0Loader ────────────────────────────────────────
from io.phase0 import Phase0Loader

loader = Phase0Loader(RUN_DIR)
logger.info("%s", loader)

X = loader.get_X()
y = loader.get_y()
total_cost_C = loader.get_total_cost_C()
mask_ref_canonical = loader.get_mask_ref()
sample_ids = loader.sample_ids
metadata_df = loader.metadata_df

S_map_cached = loader.S_map
if S_map_cached is None:
    logger.warning("S_map not in dataset — s_map_ref.png will be skipped")

aligned_target_masks = loader.target_masks
if aligned_target_masks is None:
    logger.warning("target_masks_canonical not in zarr — QC-2 overlays will be skipped")

alignment_debug_df = loader.alignment_debug
if alignment_debug_df is not None:
    logger.info("Loaded alignment debug: %d rows", len(alignment_debug_df))

manifest_path = RUN_DIR / "feature_dataset" / "manifest.json"
if manifest_path.exists():
    with open(manifest_path) as f:
        manifest = json.load(f)
    logger.info("Manifest: %s", {k: v for k, v in manifest.items() if k != "sample_ids"})

logger.info("X shape: %s, mask_ref: %s, N=%d", X.shape, mask_ref_canonical.shape, len(y))

# ── re-run QC (step 2) ────────────────────────────────────────────────────────
from viz import run_qc_suite

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
from viz import plot_cost_density_suite, plot_s_map

viz_dir = RUN_DIR / "viz"
viz_dir.mkdir(parents=True, exist_ok=True)
logger.info("Re-running viz suite → %s", viz_dir)

if S_map_cached is not None:
    plot_s_map(S_map_cached, mask_ref_canonical, save_path=viz_dir / "s_map_ref.png")
    logger.info("Saved s_map_ref.png")

plot_cost_density_suite(
    X, y, mask_ref_canonical, outlier_flag,
    sigma_grid=(1.0, 2.0, 4.0),
    save_dir=viz_dir,
)
logger.info("Saved cost density suite")

logger.info("Done — check %s", RUN_DIR)
