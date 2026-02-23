#!/usr/bin/env python3
"""
Phase 0 Step 3: Run Classification and Bootstrap Analysis

Loads pre-computed S-bin features and runs AUROC localization
with bootstrap stability testing.

Usage:
    python scripts/s04_run_classification.py \
        --sbin-features scripts/output/phase0_run_001/features_sbins.parquet \
        --output-dir scripts/output/phase0_run_001

Outputs:
    <output_dir>/results/
        - auroc_by_sbin.json
        - auroc_curve.png
        - selected_interval.json
        - bootstrap_stability.json
        - permutation_nulls.json
"""

from __future__ import annotations

import argparse
import json
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
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_sbin_features(sbin_path: Path) -> pd.DataFrame:
    """Load S-bin feature table."""
    logger.info(f"Loading S-bin features from {sbin_path}")
    
    if not sbin_path.exists():
        raise FileNotFoundError(f"S-bin features not found: {sbin_path}")
    
    df = pd.read_parquet(sbin_path)
    
    logger.info(f"Loaded {len(df)} S-bin rows")
    logger.info(f"Unique samples: {df['sample_id'].nunique()}")
    logger.info(f"  WT: {df[df['label_int']==0]['sample_id'].nunique()}")
    logger.info(f"  Mutant: {df[df['label_int']==1]['sample_id'].nunique()}")
    logger.info(f"S-bins: {df['k_bin'].nunique()}")
    
    return df


def compute_auroc_by_sbin(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute AUROC for each S-bin independently.
    
    Returns
    -------
    s_centers : (K,) array of S-bin centers
    aurocs : (K,) array of AUROC values
    """
    logger.info("Computing AUROC for each S-bin...")
    
    # Get unique bins
    bins = sorted(df["k_bin"].unique())
    K = len(bins)
    
    s_centers = np.zeros(K)
    aurocs = np.zeros(K)
    
    for i, k in enumerate(bins):
        bin_data = df[df["k_bin"] == k]
        
        # Use mean S position as center
        s_centers[i] = (bin_data["S_lo"].mean() + bin_data["S_hi"].mean()) / 2.0
        
        # Get cost and labels
        y_bin = bin_data["label_int"].values
        cost_bin = bin_data["cost_mean"].values
        
        # Compute AUROC (handle edge case where all same label)
        if len(np.unique(y_bin)) < 2:
            aurocs[i] = 0.5
        else:
            try:
                aurocs[i] = roc_auc_score(y_bin, cost_bin)
            except:
                aurocs[i] = 0.5
    
    logger.info(f"AUROC range: [{aurocs.min():.3f}, {aurocs.max():.3f}]")
    logger.info(f"Peak AUROC: {aurocs.max():.3f} at S={s_centers[aurocs.argmax()]:.3f}")
    
    return s_centers, aurocs


def plot_auroc_curve(
    s_centers: np.ndarray,
    aurocs: np.ndarray,
    save_path: Path,
):
    """Plot AUROC vs S-coordinate."""
    logger.info(f"Plotting AUROC curve to {save_path}")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(s_centers, aurocs, 'o-', linewidth=2, markersize=8, color="steelblue")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Chance")
    
    # Highlight peak
    peak_idx = aurocs.argmax()
    ax.plot(s_centers[peak_idx], aurocs[peak_idx], 'r*', markersize=20, 
            label=f"Peak: {aurocs[peak_idx]:.3f} @ S={s_centers[peak_idx]:.2f}")
    
    ax.set_xlabel("S-coordinate (0=head, 1=tail)", fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("Phase 0 ROI Localization: AUROC vs Anterior-Posterior Position", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_ylim([0.3, 1.0])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def select_best_interval(
    s_centers: np.ndarray,
    aurocs: np.ndarray,
    min_width: float = 0.1,
    max_width: float = 0.3,
) -> dict:
    """
    Select best contiguous S-interval with high AUROC.
    
    Returns
    -------
    dict with keys: s_lo, s_hi, mean_auroc, peak_auroc, n_bins
    """
    logger.info("Selecting best S-interval...")
    
    K = len(s_centers)
    best_score = 0
    best_interval = None
    
    # Try all contiguous intervals
    for i in range(K):
        for j in range(i + 1, K + 1):
            interval_s = s_centers[i:j]
            interval_auroc = aurocs[i:j]
            
            s_width = interval_s[-1] - interval_s[0]
            
            # Check width constraint
            if s_width < min_width or s_width > max_width:
                continue
            
            # Score: mean AUROC weighted by peak
            mean_auroc = interval_auroc.mean()
            peak_auroc = interval_auroc.max()
            score = mean_auroc * 0.7 + peak_auroc * 0.3
            
            if score > best_score:
                best_score = score
                best_interval = {
                    "s_lo": float(interval_s[0]),
                    "s_hi": float(interval_s[-1]),
                    "mean_auroc": float(mean_auroc),
                    "peak_auroc": float(peak_auroc),
                    "n_bins": len(interval_s),
                    "score": float(score),
                }
    
    if best_interval is None:
        logger.warning("No valid interval found, using peak bin")
        peak_idx = aurocs.argmax()
        best_interval = {
            "s_lo": float(s_centers[peak_idx]),
            "s_hi": float(s_centers[peak_idx]),
            "mean_auroc": float(aurocs[peak_idx]),
            "peak_auroc": float(aurocs[peak_idx]),
            "n_bins": 1,
            "score": float(aurocs[peak_idx]),
        }
    
    logger.info(f"Selected interval: S=[{best_interval['s_lo']:.3f}, {best_interval['s_hi']:.3f}]")
    logger.info(f"  Mean AUROC: {best_interval['mean_auroc']:.3f}")
    logger.info(f"  Peak AUROC: {best_interval['peak_auroc']:.3f}")
    logger.info(f"  N bins: {best_interval['n_bins']}")
    
    return best_interval


def run_bootstrap_stability(
    df: pd.DataFrame,
    selected_interval: dict,
    n_bootstrap: int = 100,
    seed: int = 42,
) -> dict:
    """
    Bootstrap resampling to test interval stability.
    
    Returns
    -------
    dict with bootstrap AUROC distribution
    """
    logger.info(f"Running bootstrap stability test (n={n_bootstrap})...")
    
    # Filter to selected interval
    df_roi = df[
        (df["S_lo"] >= selected_interval["s_lo"]) &
        (df["S_hi"] <= selected_interval["s_hi"])
    ]
    
    # Aggregate by sample
    sample_features = df_roi.groupby("sample_id").agg({
        "cost_mean": "mean",
        "label_int": "first",
    }).reset_index()
    
    y = sample_features["label_int"].values
    X = sample_features["cost_mean"].values.reshape(-1, 1)
    
    rng = np.random.default_rng(seed)
    bootstrap_aurocs = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = rng.choice(len(y), size=len(y), replace=True)
        y_boot = y[idx]
        X_boot = X[idx]
        
        # Compute AUROC
        if len(np.unique(y_boot)) < 2:
            continue
        
        try:
            auroc = roc_auc_score(y_boot, X_boot.ravel())
            bootstrap_aurocs.append(auroc)
        except:
            continue
    
    bootstrap_aurocs = np.array(bootstrap_aurocs)
    
    # Handle case where bootstrap failed to produce any valid AUROCs
    if len(bootstrap_aurocs) == 0:
        logger.warning("Bootstrap failed: no valid AUROC values computed")
        logger.warning("This can happen with small sample sizes or no signal")
        results = {
            "n_bootstrap": 0,
            "mean": np.nan,
            "std": np.nan,
            "ci_5": np.nan,
            "ci_95": np.nan,
            "median": np.nan,
        }
        return results
    
    results = {
        "n_bootstrap": len(bootstrap_aurocs),
        "mean": float(bootstrap_aurocs.mean()),
        "std": float(bootstrap_aurocs.std()),
        "ci_5": float(np.percentile(bootstrap_aurocs, 5)),
        "ci_95": float(np.percentile(bootstrap_aurocs, 95)),
        "median": float(np.median(bootstrap_aurocs)),
    }
    
    logger.info(f"Bootstrap AUROC: {results['mean']:.3f} ± {results['std']:.3f}" if not np.isnan(results['mean']) 
                else "Bootstrap AUROC: failed (no valid resamples)")
    if not np.isnan(results['ci_5']):
        logger.info(f"90% CI: [{results['ci_5']:.3f}, {results['ci_95']:.3f}]")
    
    return results


def run_permutation_test(
    df: pd.DataFrame,
    selected_interval: dict,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Permutation test to compute null distribution.
    
    Returns
    -------
    dict with null distribution and p-value
    """
    logger.info(f"Running permutation test (n={n_permutations})...")
    
    # Filter to selected interval
    df_roi = df[
        (df["S_lo"] >= selected_interval["s_lo"]) &
        (df["S_hi"] <= selected_interval["s_hi"])
    ]
    
    # Aggregate by sample
    sample_features = df_roi.groupby("sample_id").agg({
        "cost_mean": "mean",
        "label_int": "first",
    }).reset_index()
    
    y = sample_features["label_int"].values
    X = sample_features["cost_mean"].values
    
    # Compute observed AUROC
    auroc_observed = roc_auc_score(y, X)
    
    # Permutation null
    rng = np.random.default_rng(seed)
    null_aurocs = []
    
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        
        try:
            auroc_null = roc_auc_score(y_perm, X)
            null_aurocs.append(auroc_null)
        except:
            continue
    
    null_aurocs = np.array(null_aurocs)
    
    # Handle case where permutation failed
    if len(null_aurocs) == 0:
        logger.warning("Permutation test failed: no valid null AUROCs computed")
        results = {
            "auroc_observed": float(auroc_observed),
            "n_permutations": 0,
            "null_mean": np.nan,
            "null_std": np.nan,
            "p_value": np.nan,
        }
        return results
    
    # Compute p-value (two-tailed)
    p_value = np.mean(np.abs(null_aurocs - 0.5) >= np.abs(auroc_observed - 0.5))
    
    results = {
        "auroc_observed": float(auroc_observed),
        "n_permutations": len(null_aurocs),
        "null_mean": float(null_aurocs.mean()),
        "null_std": float(null_aurocs.std()),
        "p_value": float(p_value),
    }
    
    logger.info(f"Observed AUROC: {auroc_observed:.3f}")
    logger.info(f"Null mean: {null_aurocs.mean():.3f} ± {null_aurocs.std():.3f}")
    logger.info(f"P-value: {p_value:.4f}")
    
    return results


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Run Phase 0 classification analysis")
    parser.add_argument("--sbin-features", type=Path, required=True,
                        help="Path to features_sbins.parquet")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=100)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.sbin_features.parent
    
    results_dir = args.output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("Phase 0 Step 3: Classification & Bootstrap")
    logger.info("=" * 70)
    logger.info(f"S-bin features: {args.sbin_features}")
    logger.info(f"Output: {results_dir}")
    logger.info("=" * 70)
    
    # Load features
    logger.info("\n[1/5] Loading S-bin features...")
    df = load_sbin_features(args.sbin_features)
    
    # Filter out QC outliers
    if "qc_outlier_flag" in df.columns:
        n_before = len(df)
        df = df[~df["qc_outlier_flag"]]
        logger.info(f"Filtered out {n_before - len(df)} outlier rows")
    
    # Compute AUROC by S-bin
    logger.info("\n[2/5] Computing AUROC by S-bin...")
    s_centers, aurocs = compute_auroc_by_sbin(df)
    
    # Save AUROC data
    auroc_data = {
        "s_centers": s_centers.tolist(),
        "aurocs": aurocs.tolist(),
    }
    with open(results_dir / "auroc_by_sbin.json", "w") as f:
        json.dump(auroc_data, f, indent=2)
    
    # Plot AUROC curve
    plot_auroc_curve(s_centers, aurocs, results_dir / "auroc_curve.png")
    
    # Select best interval
    logger.info("\n[3/5] Selecting best S-interval...")
    selected_interval = select_best_interval(s_centers, aurocs)
    
    with open(results_dir / "selected_interval.json", "w") as f:
        json.dump(selected_interval, f, indent=2)
    
    # Bootstrap stability
    logger.info("\n[4/5] Bootstrap stability test...")
    bootstrap_results = run_bootstrap_stability(
        df, selected_interval,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    
    with open(results_dir / "bootstrap_stability.json", "w") as f:
        json.dump(bootstrap_results, f, indent=2)
    
    # Permutation test
    logger.info("\n[5/5] Permutation test...")
    permutation_results = run_permutation_test(
        df, selected_interval,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )
    
    with open(results_dir / "permutation_nulls.json", "w") as f:
        json.dump(permutation_results, f, indent=2)
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE: Classification Analysis")
    logger.info("=" * 70)
    logger.info(f"Results saved to: {results_dir}")
    logger.info(f"\nKey findings:")
    logger.info(f"  Peak AUROC: {aurocs.max():.3f} at S={s_centers[aurocs.argmax()]:.2f}")
    logger.info(f"  Selected interval: S=[{selected_interval['s_lo']:.2f}, {selected_interval['s_hi']:.2f}]")
    logger.info(f"  Bootstrap AUROC: {bootstrap_results['mean']:.3f} ± {bootstrap_results['std']:.3f}")
    logger.info(f"  Permutation p-value: {permutation_results['p_value']:.4f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
