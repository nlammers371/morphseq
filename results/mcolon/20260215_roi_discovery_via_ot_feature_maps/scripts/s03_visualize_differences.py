#!/usr/bin/env python3
"""
Phase 0 Step 2: Visualize WT vs Mutant Differences

Loads pre-computed OT features, builds S-coordinate map,
and creates heat maps showing phenotype differences.

Usage:
    python scripts/s03_visualize_differences.py \
        --feature-dir scripts/output/phase0_run_001/feature_dataset \
        --output-dir scripts/output/phase0_run_001

Outputs:
    <output_dir>/viz/
        - cost_density_wt.png
        - cost_density_mutant.png
        - cost_density_diff.png
        - s_map_ref.png
        - sbin_heatmap_wt.png
        - sbin_heatmap_mutant.png
        - sbin_heatmap_diff.png
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr

from roi_config import Phase0RunConfig
from p0_s_coordinate import build_s_coordinate
from p0_viz import plot_cost_density_suite, plot_s_map
from p0_sbin_features import build_sbin_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_feature_dataset(feature_dir: Path):
    """Load pre-computed OT feature dataset from Zarr."""
    logger.info(f"Loading feature dataset from {feature_dir}")
    
    zarr_path = feature_dir / "features.zarr"
    if not zarr_path.exists():
        raise FileNotFoundError(f"Feature dataset not found: {zarr_path}")
    
    z = zarr.open(str(zarr_path), mode="r")
    
    # Load arrays
    X = z["X"][:]
    mask_ref = z["mask_ref"][:]
    y = z["y"][:]
    
    # Load QC data
    outlier_flag = z["qc/outlier_flag"][:] if "qc/outlier_flag" in z else np.zeros(len(y), dtype=bool)
    total_cost_C = z["qc/total_cost_C"][:] if "qc/total_cost_C" in z else np.zeros(len(y))
    
    # Load metadata
    metadata_path = feature_dir / "metadata.parquet"
    if metadata_path.exists():
        metadata = pd.read_parquet(metadata_path)
    else:
        metadata = pd.DataFrame({
            "sample_id": [f"sample_{i:03d}" for i in range(len(y))],
            "genotype": ["cep290_homozygous" if label==1 else "cep290_wildtype" for label in y],
        })
    
    logger.info(f"Loaded: X.shape={X.shape}, y.shape={y.shape}, mask_ref.shape={mask_ref.shape}")
    logger.info(f"Samples: {(y==0).sum()} WT, {(y==1).sum()} mutant")
    logger.info(f"Outliers: {outlier_flag.sum()}/{len(outlier_flag)}")
    
    return X, y, mask_ref, outlier_flag, total_cost_C, metadata


def create_cost_density_heatmaps(
    X: np.ndarray,
    y: np.ndarray,
    mask_ref: np.ndarray,
    outlier_flag: np.ndarray,
    viz_dir: Path,
    sigma_grid: float = 2.0,
):
    """Create cost density heat maps for WT vs mutant."""
    logger.info("Creating cost density heat maps...")
    
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Use visualization suite (wrap sigma in list if it's a scalar)
    sigma_list = [sigma_grid] if isinstance(sigma_grid, (int, float)) else sigma_grid
    plot_cost_density_suite(
        X, y, mask_ref, outlier_flag,
        sigma_grid=sigma_list,
        save_dir=viz_dir,
    )
    
    logger.info(f"Saved cost density maps to {viz_dir}")


def create_sbin_heatmaps(
    X: np.ndarray,
    y: np.ndarray,
    S_map_ref: np.ndarray,
    metadata: pd.DataFrame,
    outlier_flag: np.ndarray,
    viz_dir: Path,
    K: int = 10,
    feature_set=None,
    tangent_ref=None,
    normal_ref=None,
):
    """Create S-bin aggregated heat maps."""
    logger.info(f"Creating S-bin heat maps (K={K} bins)...")
    
    # Build S-bin features
    sbin_df = build_sbin_features(
        X, y, S_map_ref, metadata, outlier_flag,
        feature_set=feature_set,
        K=K,
        tangent_ref=tangent_ref,
        normal_ref=normal_ref,
    )
    
    # Separate by genotype
    sbin_wt = sbin_df[sbin_df["label_int"] == 0]
    sbin_mut = sbin_df[sbin_df["label_int"] == 1]
    
    # Aggregate by bin
    wt_by_bin = sbin_wt.groupby("k_bin")["cost_mean"].mean()
    mut_by_bin = sbin_mut.groupby("k_bin")["cost_mean"].mean()
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # WT heatmap
    axes[0].bar(wt_by_bin.index, wt_by_bin.values, color="steelblue", alpha=0.7)
    axes[0].set_xlabel("S-bin (0=head, 1=tail)")
    axes[0].set_ylabel("Mean transport cost")
    axes[0].set_title(f"WT (n={len(sbin_wt['sample_id'].unique())})")
    axes[0].grid(True, alpha=0.3)
    
    # Mutant heatmap
    axes[1].bar(mut_by_bin.index, mut_by_bin.values, color="coral", alpha=0.7)
    axes[1].set_xlabel("S-bin (0=head, 1=tail)")
    axes[1].set_ylabel("Mean transport cost")
    axes[1].set_title(f"Mutant (n={len(sbin_mut['sample_id'].unique())})")
    axes[1].grid(True, alpha=0.3)
    
    # Difference (mutant - WT)
    diff = mut_by_bin - wt_by_bin
    colors = ["coral" if d > 0 else "steelblue" for d in diff.values]
    axes[2].bar(diff.index, diff.values, color=colors, alpha=0.7)
    axes[2].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[2].set_xlabel("S-bin (0=head, 1=tail)")
    axes[2].set_ylabel("Δ Cost (mutant - WT)")
    axes[2].set_title("Phenotype Difference")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = viz_dir / "sbin_comparison.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved S-bin comparison to {output_path}")
    
    # Save table
    table_path = viz_dir.parent / "features_sbins.parquet"
    sbin_df.to_parquet(table_path)
    logger.info(f"Saved S-bin features to {table_path}")
    
    return sbin_df


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Visualize WT vs mutant differences")
    parser.add_argument("--feature-dir", type=Path, required=True,
                        help="Path to feature_dataset directory")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (defaults to parent of feature-dir)")
    parser.add_argument("--sigma", type=float, default=2.0,
                        help="Gaussian smoothing sigma for cost density")
    parser.add_argument("--k-bins", type=int, default=10,
                        help="Number of S-bins along A-P axis")
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.feature_dir.parent
    
    viz_dir = args.output_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("Phase 0 Step 2: Visualize Differences")
    logger.info("=" * 70)
    logger.info(f"Feature dataset: {args.feature_dir}")
    logger.info(f"Output: {viz_dir}")
    logger.info("=" * 70)
    
    # Load pre-computed features
    logger.info("\n[1/4] Loading feature dataset...")
    X, y, mask_ref, outlier_flag, total_cost_C, metadata = load_feature_dataset(args.feature_dir)
    
    # Build S-coordinate
    logger.info("\n[2/4] Building S-coordinate map...")
    config = Phase0RunConfig()
    S_map_ref, tangent_ref, normal_ref, s_info = build_s_coordinate(
        mask_ref, config=config.s_coord,
    )
    logger.info(f"S-coordinate range: [{S_map_ref[mask_ref > 0].min():.3f}, {S_map_ref[mask_ref > 0].max():.3f}]")
    
    # Visualize S-map
    plot_s_map(S_map_ref, mask_ref, save_path=viz_dir / "s_map_ref.png")
    logger.info(f"Saved S-map visualization to {viz_dir / 's_map_ref.png'}")
    
    # Create cost density heatmaps
    logger.info("\n[3/4] Creating cost density heat maps...")
    create_cost_density_heatmaps(
        X, y, mask_ref, outlier_flag, viz_dir,
        sigma_grid=args.sigma,
    )
    
    # Create S-bin heatmaps
    logger.info("\n[4/4] Creating S-bin heat maps...")
    sbin_df = create_sbin_heatmaps(
        X, y, S_map_ref, metadata, outlier_flag, viz_dir,
        K=args.k_bins,
        feature_set=config.feature_set,
        tangent_ref=tangent_ref,
        normal_ref=normal_ref,
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE: Visualizations Created")
    logger.info("=" * 70)
    logger.info(f"Visualizations: {viz_dir}")
    logger.info(f"Key files:")
    logger.info(f"  - s_map_ref.png: S-coordinate map")
    logger.info(f"  - cost_density_*.png: Cost density heat maps")
    logger.info(f"  - sbin_comparison.png: S-bin phenotype comparison")
    logger.info("\nNext step: Run s04_run_classification.py for AUROC analysis")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
