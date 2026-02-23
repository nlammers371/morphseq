"""
Biologist-facing API for ROI discovery.

Phase 1 API (minimal; no compare()):
    fit()  — run full pipeline (sweep + nulls)
    plot() — visualize ROI results
    report() — print summary statistics

Follows the front-end pattern from PLAN.md Section G.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from roi_config import (
    LAMBDA_PRESETS,
    MU_PRESETS,
    FeatureSet,
    NullConfig,
    NullMode,
    ROISizePreset,
    SmoothnessPreset,
    SweepConfig,
    TrainerConfig,
    ROIRunConfig,
)

logger = logging.getLogger(__name__)


def fit(
    dataset_dir: str,
    genotype: str = "cep290",
    features: str = "cost",
    learn_res: int = 128,
    roi_size: str = "medium",
    smoothness: str = "medium",
    class_balance: str = "morphseq_balance_method",
    null: str = "both",
    n_permute: int = 100,
    n_boot: int = 200,
    out_dir: Optional[str] = None,
    random_seed: int = 42,
    run_config: Optional[ROIRunConfig] = None,
) -> Dict:
    """
    Run ROI discovery pipeline end-to-end.

    Parameters
    ----------
    dataset_dir : str
        Path to validated FeatureDataset directory.
    genotype : str
        Target genotype to compare against WT.
    features : str
        "cost", "cost+disp", or "all_ot".
    learn_res : int
        Resolution for weight learning (128 or 256).
    roi_size : str
        "small", "medium", or "large" — maps to λ presets.
    smoothness : str
        "low", "medium", or "high" — maps to μ presets.
    class_balance : str
        Balance method (only "morphseq_balance_method" supported).
    null : str
        "permute", "bootstrap", "both", or "none".
    n_permute : int
        Number of permutations for NULL 1.
    n_boot : int
        Number of bootstrap iterations for NULL 3.
    out_dir : str, optional
        Output directory. Auto-generated if None.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        run_id, out_dir, sweep_result, nulls_result, config
    """
    from roi_loader import FeatureLoader
    from roi_sweep import run_sweep, save_sweep_result
    from roi_nulls import run_nulls, save_nulls_result

    # Resolve config
    if run_config is not None:
        genotype = run_config.genotype
        features = run_config.features.value
        roi_size = run_config.roi_size.value
        smoothness = run_config.smoothness.value
        learn_res = run_config.trainer.learn_res
        null = run_config.nulls.null_mode.value
        n_permute = run_config.nulls.n_permute
        n_boot = run_config.nulls.n_boot
        random_seed = run_config.trainer.random_seed
        out_dir = out_dir or run_config.out_dir

    roi_size_enum = ROISizePreset(roi_size)
    smoothness_enum = SmoothnessPreset(smoothness)
    null_mode = NullMode(null)

    lambda_values = list(LAMBDA_PRESETS[roi_size_enum])
    mu_values = list(MU_PRESETS[smoothness_enum])

    trainer_config = TrainerConfig(
        learn_res=learn_res,
        random_seed=random_seed,
    )
    sweep_config = SweepConfig(
        lambda_values=tuple(lambda_values),
        mu_values=tuple(mu_values),
    )
    null_config = NullConfig(
        null_mode=null_mode,
        n_permute=n_permute,
        n_boot=n_boot,
        random_seed=random_seed,
    )

    # Generate run_id
    import hashlib
    import time
    run_id = hashlib.md5(
        f"{genotype}_{features}_{learn_res}_{roi_size}_{smoothness}_{time.time()}".encode()
    ).hexdigest()[:12]

    if out_dir is None:
        out_dir = str(Path(dataset_dir).parent / f"roi_run_{run_id}")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"ROI discovery run: {run_id}")
    logger.info(f"  genotype={genotype}, features={features}")
    logger.info(f"  learn_res={learn_res}, roi_size={roi_size}, smoothness={smoothness}")
    logger.info(f"  null={null}, n_permute={n_permute}, n_boot={n_boot}")

    # Load data
    loader = FeatureLoader(dataset_dir)
    X, y, groups = loader.load_full()
    mask_ref = loader.load_mask_ref()
    channel_names = loader.get_channel_names()

    logger.info(f"  Loaded: {X.shape[0]} samples, {X.shape[-1]} channels")

    # 1) Sweep
    logger.info("Step 1: λ/μ sweep...")
    sweep_result = run_sweep(
        X, y, mask_ref, groups,
        lambda_values=lambda_values,
        mu_values=mu_values,
        sweep_config=sweep_config,
        trainer_config=trainer_config,
        channel_names=channel_names,
    )
    save_sweep_result(sweep_result, out_path / "sweep")

    # 2) Nulls
    nulls_result = None
    if null_mode != NullMode.NONE:
        logger.info("Step 2: Null testing...")
        nulls_result = run_nulls(
            X, y, mask_ref, groups,
            observed_auroc=sweep_result.selected_auroc,
            selected_lam=sweep_result.selected_lam,
            selected_mu=sweep_result.selected_mu,
            selection_rule=sweep_result.selection_rule,
            lambda_values=lambda_values,
            mu_values=mu_values,
            null_config=null_config,
            sweep_config=sweep_config,
            trainer_config=trainer_config,
            channel_names=channel_names,
        )
        save_nulls_result(nulls_result, out_path / "nulls")

    # 3) Save run config
    run_config = {
        "run_id": run_id,
        "genotype": genotype,
        "features": features,
        "learn_res": learn_res,
        "roi_size": roi_size,
        "smoothness": smoothness,
        "null": null,
        "n_permute": n_permute,
        "n_boot": n_boot,
        "n_samples": int(X.shape[0]),
        "n_channels": int(X.shape[-1]),
        "random_seed": random_seed,
    }
    with open(out_path / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    logger.info(f"Run complete: {out_path}")
    return {
        "run_id": run_id,
        "out_dir": str(out_path),
        "sweep_result": sweep_result,
        "nulls_result": nulls_result,
        "config": run_config,
    }


def plot(
    run_dir: str,
    style: str = "filled_contours",
    overlays: Optional[List[str]] = None,
    save: bool = True,
) -> None:
    """
    Plot ROI discovery results.

    Parameters
    ----------
    run_dir : str
        Path to a completed ROI run directory.
    style : str
        "filled_contours" (default) or "heatmap".
    overlays : list of str, optional
        ["outline", "optional_S_isolines"]
    save : bool
        If True, save plots to run_dir/plots/.

    Note
    ----
    This is a placeholder for the plotting infrastructure.
    The actual plotting will integrate with the existing visualization
    tools in src/analyze/viz/ and
    src/analyze/optimal_transport_morphometrics/uot_masks/viz.py.

    For Phase 1, we generate:
    1. Weight map heatmap (w_full magnitude)
    2. ROI overlay on reference mask
    3. Sweep Pareto front plot
    4. Null distribution histogram
    5. Bootstrap IoU distribution
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_path = Path(run_dir)
    plot_dir = run_path / "plots"
    plot_dir.mkdir(exist_ok=True)

    overlays = overlays or ["outline"]

    # Load sweep results
    sweep_path = run_path / "sweep"
    if (sweep_path / "sweep_table.csv").exists():
        sweep_df = pd.read_csv(sweep_path / "sweep_table.csv")

        # Plot 1: Sweep Pareto front
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        scatter = ax.scatter(
            sweep_df["area_fraction_mean"],
            sweep_df["auroc_mean"],
            c=np.log10(sweep_df["lam"]),
            cmap="viridis",
            s=80,
            edgecolors="k",
            linewidths=0.5,
        )
        plt.colorbar(scatter, ax=ax, label="log10(λ)")
        ax.set_xlabel("Area Fraction (complexity)")
        ax.set_ylabel("AUROC")
        ax.set_title("λ/μ Sweep: AUROC vs ROI Complexity")

        # Mark selected point
        if (sweep_path / "selection.json").exists():
            with open(sweep_path / "selection.json") as f:
                sel = json.load(f)
            sel_row = sweep_df[
                (np.isclose(sweep_df["lam"], sel["selected_lam"]))
                & (np.isclose(sweep_df["mu"], sel["selected_mu"]))
            ]
            if len(sel_row) > 0:
                ax.scatter(
                    sel_row["area_fraction_mean"],
                    sel_row["auroc_mean"],
                    marker="*", s=300, c="red", zorder=5,
                    label=f"Selected (λ={sel['selected_lam']:.2e})",
                )
                ax.legend()

        if save:
            fig.savefig(plot_dir / "sweep_pareto.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    # Plot 2: Null distribution
    nulls_path = run_path / "nulls"
    if (nulls_path / "null_aurocs.npy").exists():
        null_aurocs = np.load(nulls_path / "null_aurocs.npy")

        with open(nulls_path / "nulls_summary.json") as f:
            nulls_summary = json.load(f)

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.hist(null_aurocs[np.isfinite(null_aurocs)], bins=30, alpha=0.7,
                color="steelblue", edgecolor="k", label="Null distribution")

        if "permutation" in nulls_summary:
            obs = nulls_summary["permutation"]["observed_auroc"]
            pval = nulls_summary["permutation"]["pvalue"]
            ax.axvline(obs, color="red", linewidth=2, linestyle="--",
                       label=f"Observed AUROC={obs:.4f}\np={pval:.4f}")

        ax.set_xlabel("AUROC (selection-aware null)")
        ax.set_ylabel("Count")
        ax.set_title("NULL 1: Label Permutation Test")
        ax.legend()

        if save:
            fig.savefig(plot_dir / "null_permutation.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    # Plot 3: Bootstrap IoU
    if (nulls_path / "bootstrap_ious.npy").exists():
        ious = np.load(nulls_path / "bootstrap_ious.npy")

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.hist(ious, bins=30, alpha=0.7, color="darkorange", edgecolor="k")
        ax.axvline(np.mean(ious), color="red", linewidth=2,
                   label=f"Mean IoU={np.mean(ious):.4f}±{np.std(ious):.4f}")
        ax.set_xlabel("IoU with Reference ROI")
        ax.set_ylabel("Count")
        ax.set_title("NULL 3: Bootstrap ROI Stability")
        ax.legend()

        if save:
            fig.savefig(plot_dir / "bootstrap_iou.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    logger.info(f"Plots saved to {plot_dir}")


def report(run_dir: str) -> Dict:
    """
    Generate a summary report for a completed ROI run.

    Returns a dict with:
        auroc, selected_lam, selected_mu,
        area_fraction, n_components, boundary_fraction,
        permutation_pvalue, bootstrap_iou_mean, bootstrap_iou_std

    Also prints a human-readable summary.
    """
    run_path = Path(run_dir)
    result = {}

    # Load sweep selection
    sel_path = run_path / "sweep" / "selection.json"
    if sel_path.exists():
        with open(sel_path) as f:
            sel = json.load(f)
        result["auroc"] = sel["selected_auroc"]
        result["selected_lam"] = sel["selected_lam"]
        result["selected_mu"] = sel["selected_mu"]
        result.update(sel.get("selected_complexity", {}))
        result["selection_rule"] = sel["selection_rule"]

    # Load nulls
    nulls_path = run_path / "nulls" / "nulls_summary.json"
    if nulls_path.exists():
        with open(nulls_path) as f:
            nulls = json.load(f)
        if "permutation" in nulls:
            result["permutation_pvalue"] = nulls["permutation"]["pvalue"]
            result["null_auroc_mean"] = nulls["permutation"]["null_mean"]
        if "bootstrap" in nulls:
            result["bootstrap_iou_mean"] = nulls["bootstrap"]["iou_mean"]
            result["bootstrap_iou_std"] = nulls["bootstrap"]["iou_std"]

    # Print summary
    print("=" * 60)
    print("ROI Discovery Report")
    print("=" * 60)
    if "auroc" in result:
        print(f"  AUROC:              {result['auroc']:.4f}")
    if "selected_lam" in result:
        print(f"  Selected λ:         {result['selected_lam']:.2e}")
    if "selected_mu" in result:
        print(f"  Selected μ:         {result['selected_mu']:.2e}")
    if "area_fraction" in result:
        print(f"  ROI Area Fraction:  {result['area_fraction']:.4f}")
    if "n_components" in result:
        print(f"  ROI Components:     {result['n_components']:.1f}")
    if "boundary_fraction" in result:
        print(f"  Boundary Fraction:  {result['boundary_fraction']:.4f}")
    if "permutation_pvalue" in result:
        print(f"  Permutation p:      {result['permutation_pvalue']:.4f}")
    if "bootstrap_iou_mean" in result:
        print(
            f"  Bootstrap IoU:      "
            f"{result['bootstrap_iou_mean']:.4f}"
            f"±{result['bootstrap_iou_std']:.4f}"
        )
    print("=" * 60)

    return result


__all__ = ["fit", "plot", "report"]
