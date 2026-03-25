#!/usr/bin/env python3
"""
Phase 0 Step 8: Nulls + Stability (separate, slow stage).

This script is intentionally separated from the core Phase 0 pipeline so we can:
  - iterate quickly on OT/QC/S-bin features without paying the null/bootstrap cost
  - tune QC (IQR multiplier) without recomputing OT maps

Usage:
    PYTHON=/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python
    KMP_SHM_DISABLE=1 "$PYTHON" scripts/s05_run_nulls_and_stability.py \\
        --run-dir scripts/output/phase0_run_001 \\
        --iqr-multiplier 2.5 \\
        --n-boot 50 \\
        --n-permute 200

Outputs (written to <run_dir>/nulls/):
    - selected_interval.json
    - interval_search.csv (optional; only if recomputed here)
    - perm_null_auroc.json + .png
    - perm_null_interval.json + .png
    - bootstrap_interval.json + auroc_vs_sbin_with_ci.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import importlib.util

# POT (and some NumPy/OpenMP builds) may fail on locked-down systems unless this is set
# in the process environment before numerical libraries initialize.
os.environ.setdefault("KMP_SHM_DISABLE", "1")

# Add morphseq root to path (for both src/ and segmentation_sandbox/)
MORPHSEQ_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(MORPHSEQ_ROOT))
sys.path.insert(0, str(MORPHSEQ_ROOT / "src"))

# Add ROI discovery module to path
ROI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROI_DIR))

import logging

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

_PHASE0_LOADER_PATH = ROI_DIR / "io" / "phase0.py"
_spec = importlib.util.spec_from_file_location("phase0_loader", _PHASE0_LOADER_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not import Phase0Loader from {_PHASE0_LOADER_PATH}")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Phase0Loader = _mod.Phase0Loader

from roi_config import Phase0RunConfig
from p0_classification import compute_auroc_per_bin
from p0_interval_search import search_all_intervals, select_best_interval
from p0_nulls import (
    run_bootstrap_stability,
    run_permutation_null_auroc_max,
    run_permutation_null_auroc_max_over_features,
    run_permutation_null_interval,
)
from viz import plot_auroc_vs_sbin, plot_bootstrap_interval_stability, plot_permutation_null
from viz.qc import compute_iqr_outliers

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_manifest_iqr_multiplier(run_dir: Path) -> float | None:
    manifest_path = run_dir / "feature_dataset" / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
        qc_rules = manifest.get("qc_rules", {}) if isinstance(manifest, dict) else {}
        mult = qc_rules.get("iqr_multiplier")
        return float(mult) if mult is not None else None
    except Exception:
        return None


def _infer_interval_feature_cols(sbin_df: pd.DataFrame, interval_df: pd.DataFrame | None) -> list[str]:
    # If the interval_search.csv was produced by the updated pipeline, trust its declaration.
    if interval_df is not None and "feature_cols" in interval_df.columns and len(interval_df) > 0:
        raw = interval_df["feature_cols"].iloc[0]
        if isinstance(raw, str) and raw.strip():
            cols = [c.strip() for c in raw.split("+") if c.strip()]
            cols = [c for c in cols if c in sbin_df.columns]
            if cols:
                return cols

    # Otherwise, infer from columns present in the S-bin table.
    dyn = ["disp_mag_mean", "disp_par_mean", "disp_perp_mean"]
    if all(c in sbin_df.columns for c in dyn):
        return dyn
    return ["cost_mean"]


def _json_sanitize(d: dict) -> dict:
    """Drop large arrays / non-serializables from p0_nulls outputs."""
    out = {}
    for k, v in d.items():
        if k == "null_distribution":
            continue
        if isinstance(v, (np.ndarray,)):
            if v.ndim == 0:
                out[k] = float(v)
            continue
        out[k] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0: run nulls + bootstrap stability (separate stage)")
    parser.add_argument("--run-dir", type=Path, required=True, help="Phase 0 run directory (contains feature_dataset/)")
    parser.add_argument("--sbin-features", type=Path, default=None, help="Path to features_sbins.parquet (default: <run_dir>/features_sbins.parquet)")
    parser.add_argument("--interval-search", type=Path, default=None, help="Path to interval_search.csv (default: <run_dir>/interval_search.csv)")
    parser.add_argument("--recompute-interval-search", action="store_true",
                        help="Recompute interval_search.csv from sbin features (fast-ish, but still CV-heavy).")
    parser.add_argument("--iqr-multiplier", type=float, default=None,
                        help="IQR multiplier for outlier flagging (overrides manifest/config).")
    parser.add_argument("--n-permute", type=int, default=None,
                        help="Number of permutations (default: config.nulls.n_permute).")
    parser.add_argument("--n-boot", type=int, default=None,
                        help="Number of bootstrap iterations (default: config.nulls.n_boot).")
    parser.add_argument("--n-folds", type=int, default=None,
                        help="CV folds for interval search/nulls/bootstrap (default: config.classification.n_cv_folds).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: config.nulls.random_seed).")
    args = parser.parse_args()

    run_dir = args.run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    config = Phase0RunConfig()
    n_permute = int(args.n_permute) if args.n_permute is not None else int(config.nulls.n_permute)
    n_boot = int(args.n_boot) if args.n_boot is not None else int(config.nulls.n_boot)
    n_folds = int(args.n_folds) if args.n_folds is not None else int(config.classification.n_cv_folds)
    seed = int(args.seed) if args.seed is not None else int(config.nulls.random_seed)

    sbin_path = args.sbin_features or (run_dir / "features_sbins.parquet")
    interval_path = args.interval_search or (run_dir / "interval_search.csv")

    nulls_dir = run_dir / "nulls"
    nulls_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Phase 0 Step 8: Nulls + Stability (separate stage)")
    logger.info("=" * 70)
    logger.info("Run dir: %s", run_dir)
    logger.info("S-bin features: %s", sbin_path)
    logger.info("Interval search: %s", interval_path)
    logger.info("n_permute=%d, n_boot=%d, n_folds=%d, seed=%d", n_permute, n_boot, n_folds, seed)
    logger.info("Output: %s", nulls_dir)
    logger.info("=" * 70)

    if not sbin_path.exists():
        raise FileNotFoundError(f"Missing sbin features: {sbin_path}")
    sbin_df = pd.read_parquet(sbin_path)

    loader = Phase0Loader(run_dir)
    total_cost_C = loader.get_total_cost_C()
    sample_ids = loader.sample_ids

    # Recompute outlier flags using the requested IQR multiplier (or manifest default).
    manifest_mult = _load_manifest_iqr_multiplier(run_dir)
    iqr_mult = float(args.iqr_multiplier) if args.iqr_multiplier is not None else (
        float(manifest_mult) if manifest_mult is not None else float(config.dataset.iqr_multiplier)
    )
    outlier_flag, qc_stats = compute_iqr_outliers(total_cost_C, multiplier=iqr_mult)
    logger.info("QC (IQR×%.2f): retained %d/%d, outliers=%d",
                iqr_mult, qc_stats["n_retained"], qc_stats["n_total"], qc_stats["n_outliers"])

    if len(sample_ids) != len(outlier_flag):
        raise ValueError(f"sample_ids length {len(sample_ids)} != outlier_flag length {len(outlier_flag)}")
    sample_to_outlier = dict(zip(sample_ids, outlier_flag))
    sbin_df = sbin_df.copy()
    sbin_df["qc_outlier_flag"] = sbin_df["sample_id"].map(sample_to_outlier).fillna(False).astype(bool)

    K = int(sbin_df["k_bin"].nunique())
    if K <= 0:
        raise ValueError("Could not infer K from sbin_df (k_bin).")

    # Interval search + selection (needed for interval-based nulls and stability).
    if args.recompute_interval_search:
        logger.info("Recomputing interval search (K=%d, folds=%d)...", K, n_folds)
        interval_feature_cols = _infer_interval_feature_cols(sbin_df, interval_df=None)
        interval_df = search_all_intervals(
            sbin_df, feature_cols=interval_feature_cols, K=K, n_folds=n_folds,
        )
        interval_df["feature_cols"] = "+".join(interval_feature_cols)
        interval_df["exclude_outliers"] = True
        interval_df.to_csv(nulls_dir / "interval_search.csv", index=False)
    else:
        if not interval_path.exists():
            raise FileNotFoundError(
                f"Missing interval_search.csv: {interval_path}. "
                "Run the core pipeline first, or pass --recompute-interval-search."
            )
        interval_df = pd.read_csv(interval_path)

    interval_feature_cols = _infer_interval_feature_cols(sbin_df, interval_df=interval_df)
    selected = select_best_interval(interval_df, config.interval, K=K)
    selected["feature_cols"] = list(interval_feature_cols)
    selected["exclude_outliers"] = True
    with open(nulls_dir / "selected_interval.json", "w") as f:
        json.dump(selected, f, indent=2)

    # AUROC per bin (filtered) for observed max AUROC.
    max_auroc_feature_cols = [c for c in interval_feature_cols if c in sbin_df.columns]
    if not max_auroc_feature_cols:
        max_auroc_feature_cols = ["cost_mean"]

    observed_max = -np.inf
    observed_max_by_col: dict[str, float] = {}
    for col in max_auroc_feature_cols:
        auroc_df = compute_auroc_per_bin(sbin_df, col, exclude_outliers=True)
        m = float(auroc_df["auroc"].max())
        observed_max_by_col[col] = m
        if np.isfinite(m) and m > observed_max:
            observed_max = m
    if not np.isfinite(observed_max):
        observed_max = 0.5

    primary_feature_col = max(observed_max_by_col, key=observed_max_by_col.get) if observed_max_by_col else "cost_mean"
    auroc_primary_df = compute_auroc_per_bin(sbin_df, primary_feature_col, exclude_outliers=True)

    # Permutation null: max AUROC across bins
    if len(max_auroc_feature_cols) == 1:
        perm_auroc = run_permutation_null_auroc_max(
            sbin_df, observed_max,
            feature_col=max_auroc_feature_cols[0],
            n_permute=n_permute,
            random_seed=seed,
        )
    else:
        perm_auroc = run_permutation_null_auroc_max_over_features(
            sbin_df, observed_max,
            feature_cols=max_auroc_feature_cols,
            n_permute=n_permute,
            random_seed=seed,
        )
    with open(nulls_dir / "perm_null_auroc.json", "w") as f:
        json.dump(_json_sanitize(perm_auroc), f, indent=2, default=str)
    plot_permutation_null(perm_auroc, save_path=nulls_dir / "perm_null_auroc.png")

    # Permutation null: best interval AUROC (expensive, so cap permutations here too)
    perm_interval = run_permutation_null_interval(
        sbin_df, float(selected["auroc"]),
        feature_cols=interval_feature_cols,
        K=K, n_folds=n_folds,
        n_permute=min(n_permute, 50),
        random_seed=seed,
    )
    with open(nulls_dir / "perm_null_interval.json", "w") as f:
        json.dump(_json_sanitize(perm_interval), f, indent=2, default=str)
    plot_permutation_null(perm_interval, save_path=nulls_dir / "perm_null_interval.png")

    # Bootstrap stability (interval distribution)
    boot = run_bootstrap_stability(
        sbin_df,
        feature_col=primary_feature_col,
        feature_cols_interval=interval_feature_cols,
        K=K, n_folds=n_folds,
        n_boot=n_boot,
        random_seed=seed,
    )
    with open(nulls_dir / "bootstrap_interval.json", "w") as f:
        json.dump(_json_sanitize(boot), f, indent=2, default=str)

    # Plots
    plot_auroc_vs_sbin(
        {primary_feature_col: auroc_primary_df},
        K=K,
        bootstrap_result=boot,
        save_path=nulls_dir / "auroc_vs_sbin_with_ci.png",
    )
    plot_bootstrap_interval_stability(boot, K=K, save_path=nulls_dir / "bootstrap_interval.png")

    logger.info("Done. Null/stability outputs in: %s", nulls_dir)


if __name__ == "__main__":
    main()
