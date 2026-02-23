"""
Phase 0 Orchestrator: Run the full Phase 0 pipeline.

Follows the implementation order from PHASE0_SPEC.md:
  0) FeatureDataset contract + validator
  1) Generate OT maps
  2) QC + filtering (GATE)
  3) Compute S_map_ref
  4) Build features_sbins.parquet
  5) AUROC localization
  6) Enable dynamics + rerun (if V1)
  7) Interval search + sanity checks
  8) Nulls + bootstrap stability

Each step has a gate checkpoint. The orchestrator logs all
intermediate results and creates a summary report.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from roi_config import Phase0RunConfig, Phase0FeatureSet

logger = logging.getLogger(__name__)


def run_phase0(
    mask_ref: np.ndarray,
    target_masks: List[np.ndarray],
    y: np.ndarray,
    metadata_df: pd.DataFrame,
    config: Phase0RunConfig = Phase0RunConfig(),
    raw_um_per_px_ref: float = 7.8,
    raw_um_per_px_targets: np.ndarray = None,
    yolk_ref: Optional[np.ndarray] = None,
    yolk_targets: Optional[List[np.ndarray]] = None,
    source_id: Optional[str] = None,
    uot_config=None,
    backend=None,
    out_dir=None,
) -> Dict:
    """
    Run the full Phase 0 pipeline.

    Parameters
    ----------
    mask_ref : (H_ref, W_ref) uint8 — fixed WT reference mask at raw resolution
    target_masks : list of (H_i, W_i) uint8 — raw resolution masks (OT aligns to canonical)
    y : (N,) int — labels (0=WT, 1=cep290)
    metadata_df : DataFrame with sample_id, embryo_id, snip_id columns
    config : Phase0RunConfig
    raw_um_per_px_ref : float — physical resolution of reference mask
    raw_um_per_px_targets : (N,) array — physical resolution per target mask
    yolk_ref : (H_ref, W_ref) uint8, optional — reference yolk mask
    yolk_targets : list of (H_i, W_i) uint8, optional — yolk masks per target
    source_id : str, optional — stable source identifier (e.g., embryo_id|frame_index)
    uot_config : UOTConfig, optional
    backend : UOTBackend, optional
    out_dir : Path, optional — override config.out_dir

    Returns
    -------
    dict with all intermediate results and gate statuses.
    """
    if out_dir is None:
        out_dir = Path(config.out_dir or f"results/phase0_{config.genotype}")
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"config": config, "out_dir": str(out_dir), "gates": {}}
    N = len(target_masks)
    
    # Default um_per_px if not provided
    if raw_um_per_px_targets is None:
        raw_um_per_px_targets = np.full(N, 7.8, dtype=np.float32)
    
    sample_ids = metadata_df.get("sample_id", pd.Series([f"s_{i}" for i in range(N)])).tolist()

    # =====================================================================
    # Step 1: Generate OT maps
    # =====================================================================
    logger.info("=" * 60)
    logger.info("PHASE 0 STEP 1: Generate OT maps")
    logger.info("=" * 60)

    from p0_ot_maps import generate_ot_maps
    target_ids = None
    if {"embryo_id", "frame_index"}.issubset(metadata_df.columns):
        target_ids = metadata_df.apply(
            lambda row: f"{row['embryo_id']}|frame_{int(row['frame_index'])}",
            axis=1,
        ).tolist()
    elif "target_id" in metadata_df.columns:
        target_ids = metadata_df["target_id"].astype(str).tolist()

    X, total_cost_C, aligned_ref_mask, aligned_target_masks, alignment_debug_df = generate_ot_maps(
        mask_ref, target_masks, sample_ids,
        raw_um_per_px_ref=raw_um_per_px_ref,
        raw_um_per_px_targets=raw_um_per_px_targets,
        yolk_ref=yolk_ref,
        yolk_targets=yolk_targets,
        feature_set=config.feature_set,
        source_id=source_id,
        target_ids=target_ids,
        return_aligned_masks=True,
        collect_debug=True,
        return_debug_df=True,
        uot_config=uot_config, backend=backend,
    )
    results["X_shape"] = X.shape
    results["total_cost_mean"] = float(total_cost_C.mean())
    results["mask_ref_canonical_shape"] = tuple(aligned_ref_mask.shape) if aligned_ref_mask is not None else None

    if aligned_ref_mask is None:
        raise ValueError("generate_ot_maps did not return aligned_ref_mask (canonical reference template).")
    mask_ref_canonical = aligned_ref_mask.astype(np.uint8)
    results["mask_ref_canonical"] = mask_ref_canonical

    # =====================================================================
    # Step 2: QC + Filtering (GATE)
    # =====================================================================
    logger.info("=" * 60)
    logger.info("PHASE 0 STEP 2: QC + Filtering")
    logger.info("=" * 60)

    from p0_qc import run_qc_suite
    qc_dir = out_dir / "qc"
    outlier_flag, qc_stats = run_qc_suite(
        X, y, total_cost_C, mask_ref_canonical, metadata_df, sample_ids,
        out_dir=qc_dir,
        iqr_multiplier=config.dataset.iqr_multiplier,
        target_masks_canonical=aligned_target_masks,
        alignment_debug_df=alignment_debug_df,
    )
    results["qc_stats"] = qc_stats
    results["gates"]["qc_passed"] = True  # User must visually confirm

    # =====================================================================
    # Step 3: Visualizations (cost density + optionally displacement)
    # =====================================================================
    logger.info("=" * 60)
    logger.info("PHASE 0 STEP 3: Visualizations")
    logger.info("=" * 60)

    from p0_viz import plot_cost_density_suite, plot_displacement_suite
    viz_dir = out_dir / "viz"

    plot_cost_density_suite(
        X, y, mask_ref_canonical, outlier_flag,
        sigma_grid=config.viz_sigma_grid,
        save_dir=viz_dir,
    )

    if config.feature_set == Phase0FeatureSet.V1_DYNAMICS:
        plot_displacement_suite(
            X, y, mask_ref_canonical, outlier_flag,
            stride=config.quiver_stride,
            save_dir=viz_dir,
        )

    # =====================================================================
    # Step 4: S coordinate
    # =====================================================================
    logger.info("=" * 60)
    logger.info("PHASE 0 STEP 4: S coordinate")
    logger.info("=" * 60)

    from p0_s_coordinate import build_s_coordinate
    from p0_viz import plot_s_map

    S_map_ref, tangent_ref, normal_ref, s_info = build_s_coordinate(
        mask_ref_canonical, config=config.s_coord,
    )
    results["s_coordinate_info"] = s_info

    plot_s_map(S_map_ref, mask_ref_canonical, save_path=viz_dir / "s_map_ref.png")

    # =====================================================================
    # Step 5: Build FeatureDataset + S-bin features
    # =====================================================================
    logger.info("=" * 60)
    logger.info("PHASE 0 STEP 5: Build S-bin feature table")
    logger.info("=" * 60)

    from roi_feature_dataset import Phase0FeatureDatasetBuilder
    dataset_dir = out_dir / "feature_dataset"
    builder = Phase0FeatureDatasetBuilder(
        out_dir=dataset_dir,
        feature_set=config.feature_set,
        config=config.dataset,
        stage_window=config.stage_window,
        reference_mask_id=source_id or "",
    )
    builder.build(
        X=X, y=y, mask_ref=mask_ref_canonical, metadata_df=metadata_df,
        total_cost_C=total_cost_C,
        target_masks_canonical=aligned_target_masks,
        alignment_debug_df=alignment_debug_df,
        S_map_ref=S_map_ref, tangent_ref=tangent_ref, normal_ref=normal_ref,
    )

    from p0_sbin_features import build_sbin_features
    K = config.s_bins.K
    sbin_path = out_dir / "features_sbins.parquet"
    sbin_df = build_sbin_features(
        X, y, S_map_ref, metadata_df, outlier_flag,
        feature_set=config.feature_set, K=K,
        tangent_ref=tangent_ref, normal_ref=normal_ref,
        save_path=sbin_path,
    )
    results["sbin_shape"] = sbin_df.shape

    # =====================================================================
    # Step 6: AUROC localization + logistic
    # =====================================================================
    logger.info("=" * 60)
    logger.info("PHASE 0 STEP 6: Classification + AUROC")
    logger.info("=" * 60)

    from p0_classification import run_phase0_classification, compute_auroc_per_bin
    from p0_viz import plot_auroc_vs_sbin, plot_coefficient_profile

    class_results = run_phase0_classification(sbin_df, n_folds=config.classification.n_cv_folds)
    results["classification"] = {
        k: v for k, v in class_results.items()
        if not isinstance(v, pd.DataFrame)
    }

    # Plot AUROC per bin
    auroc_dfs = {}
    for key, val in class_results.items():
        if isinstance(val, pd.DataFrame) and "auroc" in val.columns:
            auroc_dfs[key] = val

    if auroc_dfs:
        plot_auroc_vs_sbin(auroc_dfs, K=K, save_path=viz_dir / "auroc_vs_sbin.png")

    # Plot coefficient profile
    logistic_cost = class_results.get("logistic_cost_filtered", {})
    if isinstance(logistic_cost, dict) and "coef_mean" in logistic_cost:
        plot_coefficient_profile(
            logistic_cost["coef_mean"], K=K,
            save_path=viz_dir / "logistic_coef_profile.png",
        )

    # =====================================================================
    # Step 7: Interval search + sanity checks
    # =====================================================================
    logger.info("=" * 60)
    logger.info("PHASE 0 STEP 7: Interval search")
    logger.info("=" * 60)

    from p0_interval_search import search_all_intervals, select_best_interval, run_sanity_checks
    from p0_viz import plot_interval_results

    interval_df = search_all_intervals(
        sbin_df, feature_cols=["cost_mean"], K=K,
        n_folds=config.classification.n_cv_folds,
    )
    interval_df.to_csv(out_dir / "interval_search.csv", index=False)

    selected = select_best_interval(interval_df, config.interval, K=K)
    results["selected_interval"] = selected

    sanity = run_sanity_checks(
        sbin_df, selected, feature_cols=["cost_mean"],
        n_folds=config.classification.n_cv_folds, K=K,
    )
    results["sanity_checks"] = sanity

    plot_interval_results(interval_df, selected, K=K,
                          save_path=viz_dir / "interval_search_results.png")

    # =====================================================================
    # Step 8: Nulls + bootstrap stability
    # =====================================================================
    logger.info("=" * 60)
    logger.info("PHASE 0 STEP 8: Nulls + Stability")
    logger.info("=" * 60)

    from p0_nulls import run_permutation_null_auroc_max, run_permutation_null_interval, run_bootstrap_stability
    from p0_viz import plot_permutation_null, plot_bootstrap_interval_stability

    # Observed max AUROC
    auroc_cost_df = compute_auroc_per_bin(sbin_df, "cost_mean", exclude_outliers=True)
    observed_max = float(auroc_cost_df["auroc"].max())

    # 8.1 Permutation nulls
    perm_auroc = run_permutation_null_auroc_max(
        sbin_df, observed_max,
        n_permute=config.nulls.n_permute,
        random_seed=config.nulls.random_seed,
    )
    results["perm_null_auroc"] = {
        k: v for k, v in perm_auroc.items() if k != "null_distribution"
    }
    plot_permutation_null(perm_auroc, save_path=viz_dir / "perm_null_auroc.png")

    perm_interval = run_permutation_null_interval(
        sbin_df, selected["auroc"],
        K=K, n_folds=config.classification.n_cv_folds,
        n_permute=min(config.nulls.n_permute, 50),  # Interval search is expensive
        random_seed=config.nulls.random_seed,
    )
    results["perm_null_interval"] = {
        k: v for k, v in perm_interval.items() if k != "null_distribution"
    }
    plot_permutation_null(perm_interval, save_path=viz_dir / "perm_null_interval.png")

    # 8.2 Bootstrap stability
    boot = run_bootstrap_stability(
        sbin_df, K=K, n_folds=config.classification.n_cv_folds,
        n_boot=config.nulls.n_boot,
        random_seed=config.nulls.random_seed,
    )
    results["bootstrap"] = {
        k: v for k, v in boot.items()
        if not isinstance(v, np.ndarray) or v.ndim == 0
    }

    # Re-plot AUROC with bootstrap CIs
    plot_auroc_vs_sbin(
        {"cost_mean": auroc_cost_df}, K=K,
        bootstrap_result=boot,
        save_path=viz_dir / "auroc_vs_sbin_with_ci.png",
    )
    plot_bootstrap_interval_stability(boot, K=K, save_path=viz_dir / "bootstrap_interval.png")

    # =====================================================================
    # Save summary
    # =====================================================================
    summary_path = out_dir / "phase0_summary.json"
    # Filter to JSON-serializable values
    summary = {}
    for k, v in results.items():
        if isinstance(v, (str, int, float, bool, list, dict)):
            try:
                json.dumps(v)
                summary[k] = v
            except (TypeError, ValueError):
                summary[k] = str(v)
        elif isinstance(v, tuple):
            summary[k] = list(v)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"Phase 0 complete. Results in: {out_dir}")
    plt.close("all")

    return results


__all__ = ["run_phase0"]
