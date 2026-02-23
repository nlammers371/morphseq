"""
Example driver script for ROI discovery.

This script demonstrates the end-to-end pipeline:
1. Build a FeatureDataset from OT results (or load existing)
2. Run the ROI discovery sweep
3. Run null tests
4. Generate plots and report

Usage:
    python run_roi_discovery.py

Before running:
    - Ensure OT results exist from the UOT pipeline
      (see results/mcolon/20260121_uot-mvp/)
    - A FeatureDataset must be built first (see build_dataset() below)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

# Add morphseq root to path for imports
MORPHSEQ_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MORPHSEQ_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from roi_config import ROIRunConfig, FeatureSet
from roi_api import fit, plot, report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Path to OT pair results from the UOT MVP pipeline
OT_RESULTS_CSV = (
    MORPHSEQ_ROOT
    / "results"
    / "mcolon"
    / "20251229_cep290_phenotype_extraction"
    / "final_data"
    / "embryo_data_with_labels.csv"
)

# Output paths
SCRIPT_DIR = Path(__file__).parent
DATASET_DIR = SCRIPT_DIR / "roi_feature_dataset_cep290"


# ---------------------------------------------------------------------------
# Step 0: Build FeatureDataset (run once, then reuse)
# ---------------------------------------------------------------------------

def build_dataset_from_ot_results():
    """
    Build a FeatureDataset from existing OT transport results.

    This function demonstrates how to extract canonical-grid feature maps
    from the UOT pipeline output and package them into the FeatureDataset
    contract.

    NOTE: This is a TEMPLATE. The actual implementation will depend on
    the specific format of your OT results. The key requirement is to
    produce:
        X:      (N, 512, 512, C) float32 feature maps
        y:      (N,) int labels (0=WT, 1=cep290)
        mask:   (512, 512) reference mask
        meta:   DataFrame with embryo_id, genotype columns
        costs:  (N,) total OT cost per sample
    """
    import pandas as pd
    from roi_feature_dataset import FeatureDatasetBuilder

    logger.info("Building FeatureDataset from OT results...")
    logger.info(f"  Source: {OT_RESULTS_CSV}")

    # ------------------------------------------------------------------
    # TODO: Replace this section with actual data loading from your
    # OT pipeline outputs. The code below is a structural template
    # showing the expected data shapes and types.
    #
    # In practice, you would:
    # 1. Load OT results (UOTResult objects or saved arrays)
    # 2. Extract canonical-grid maps (cost, velocity, mass creation)
    # 3. Stack into (N, 512, 512, C) array
    # 4. Build label vector from genotype metadata
    # ------------------------------------------------------------------

    # Check if source data exists
    if not OT_RESULTS_CSV.exists():
        logger.warning(
            f"Source data not found: {OT_RESULTS_CSV}\n"
            "Cannot build dataset without OT results.\n"
            "Run the UOT pipeline first "
            "(see results/mcolon/20260121_uot-mvp/)."
        )
        return None

    # Load metadata to get embryo info
    df = pd.read_csv(OT_RESULTS_CSV)
    logger.info(f"  Loaded metadata: {len(df)} rows")

    # Identify unique embryos and their genotypes
    if "embryo_id" in df.columns and "genotype" in df.columns:
        embryo_info = df.groupby("embryo_id")["genotype"].first().reset_index()
        logger.info(f"  Unique embryos: {len(embryo_info)}")
        logger.info(f"  Genotype distribution:\n{embryo_info['genotype'].value_counts()}")
    else:
        logger.warning("  Required columns (embryo_id, genotype) not found")
        return None

    # ------------------------------------------------------------------
    # TEMPLATE: This section shows the expected structure.
    # Replace with actual feature extraction from OT results.
    # ------------------------------------------------------------------
    logger.info(
        "\n"
        "  FeatureDataset builder is ready. To complete the build:\n"
        "  1. Extract canonical-grid OT maps for each embryo\n"
        "     (use transport_maps.rasterize_*_to_canonical)\n"
        "  2. Stack into X array: (N, 512, 512, C)\n"
        "  3. Call builder.build(X, y, mask_ref, metadata_df, costs)\n"
        "\n"
        "  See roi_feature_dataset.py for the builder API.\n"
    )

    return None


# ---------------------------------------------------------------------------
# Step 1: Run ROI discovery
# ---------------------------------------------------------------------------

def run_discovery():
    """Run the full ROI discovery pipeline."""
    dataset_dir = str(DATASET_DIR)

    if not DATASET_DIR.exists():
        logger.error(
            f"FeatureDataset not found at {DATASET_DIR}\n"
            "Run build_dataset_from_ot_results() first."
        )
        return None

    # Run with defaults
    result = fit(
        dataset_dir=dataset_dir,
        genotype="cep290",
        features="cost",
        learn_res=128,
        roi_size="medium",
        smoothness="medium",
        null="both",
        n_permute=100,
        n_boot=200,
    )

    # Generate plots
    plot(result["out_dir"])

    # Print report
    report(result["out_dir"])

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ROI Discovery via OT Feature Maps")
    parser.add_argument(
        "--build", action="store_true",
        help="Build FeatureDataset from OT results",
    )
    parser.add_argument(
        "--run", action="store_true",
        help="Run ROI discovery pipeline",
    )
    parser.add_argument(
        "--report", type=str, default=None,
        help="Path to run directory for report generation",
    )

    args = parser.parse_args()

    if args.build:
        build_dataset_from_ot_results()
    elif args.run:
        run_discovery()
    elif args.report:
        plot(args.report)
        report(args.report)
    else:
        # Default: show help
        parser.print_help()
        print("\nExample workflow:")
        print("  python run_roi_discovery.py --build   # Build dataset first")
        print("  python run_roi_discovery.py --run     # Run discovery pipeline")
        print("  python run_roi_discovery.py --report <dir>  # View results")
