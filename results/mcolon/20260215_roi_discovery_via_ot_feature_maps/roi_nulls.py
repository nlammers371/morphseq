"""
Null distributions + stability testing for ROI discovery.

NULL 1 (required): Label permutation significance (selection-aware).
NULL 3 (required): Bootstrap stability at FIXED (λ,μ).

Follows the permutation testing patterns from
src/analyze/difference_detection/permutation_utils.py
(compute_pvalue, PermutationResult).

See PLAN.md Section F for specification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from roi_config import NullConfig, SweepConfig, TrainerConfig
from roi_trainer import TrainResult, extract_roi, train

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class PermutationNullResult:
    """Result from NULL 1: label permutation test."""
    observed_auroc: float
    null_aurocs: np.ndarray         # (n_permute,)
    pvalue: float
    selection_rule: str
    observed_lam: float
    observed_mu: float


@dataclass
class BootstrapStabilityResult:
    """Result from NULL 3: bootstrap stability at fixed (λ,μ)."""
    lam: float
    mu: float
    iou_distribution: np.ndarray    # (n_boot,) pairwise IoU with reference ROI
    iou_mean: float
    iou_std: float
    area_fraction_distribution: np.ndarray
    n_components_distribution: np.ndarray
    boundary_fraction_distribution: np.ndarray
    reference_roi: np.ndarray       # (H, W) bool — the "reference" ROI from full data


@dataclass
class NullsResult:
    """Combined null testing results."""
    permutation: Optional[PermutationNullResult]
    bootstrap: Optional[BootstrapStabilityResult]


# ---------------------------------------------------------------------------
# NULL 1: Label permutation (selection-aware)
# ---------------------------------------------------------------------------

def run_permutation_null(
    X: np.ndarray,
    y: np.ndarray,
    mask_ref: np.ndarray,
    groups: np.ndarray,
    observed_auroc: float,
    observed_lam: float,
    observed_mu: float,
    selection_rule: str,
    lambda_values: List[float],
    mu_values: List[float],
    n_permute: int = 100,
    sweep_config: Optional[SweepConfig] = None,
    trainer_config: Optional[TrainerConfig] = None,
    random_seed: int = 42,
    channel_names: Optional[Tuple[str, ...]] = None,
) -> PermutationNullResult:
    """
    NULL 1: Selection-aware label permutation test.

    For each permutation:
    1. Permute labels at the embryo_id level.
    2. Run the same sweep + same deterministic selection rule.
    3. Record the selected AUROC.

    The p-value is computed against this selection-aware null.

    Parameters
    ----------
    X, y, mask_ref, groups : data arrays
    observed_auroc : the real AUROC from the real sweep
    observed_lam, observed_mu : selected (λ,μ) from real sweep
    selection_rule : name of selection rule used
    lambda_values, mu_values : sweep grid
    n_permute : number of permutations
    sweep_config, trainer_config : configs
    random_seed : base seed

    Returns
    -------
    PermutationNullResult
    """
    from roi_sweep import run_sweep

    sweep_config = sweep_config or SweepConfig()
    trainer_config = trainer_config or TrainerConfig()

    rng = np.random.default_rng(random_seed)
    null_aurocs = []

    # Get unique embryo IDs and their labels
    unique_groups = np.unique(groups)
    group_to_label = {}
    for g in unique_groups:
        mask = groups == g
        group_to_label[g] = y[mask][0]  # all samples from same embryo share label

    for perm_i in range(n_permute):
        logger.info(f"Permutation {perm_i + 1}/{n_permute}")

        # Permute labels at embryo_id level
        shuffled_labels = rng.permutation(list(group_to_label.values()))
        perm_label_map = dict(zip(unique_groups, shuffled_labels))
        y_perm = np.array([perm_label_map[g] for g in groups])

        # Run full sweep with permuted labels (selection-aware)
        try:
            sweep_result = run_sweep(
                X, y_perm, mask_ref, groups,
                lambda_values=lambda_values,
                mu_values=mu_values,
                sweep_config=sweep_config,
                trainer_config=trainer_config,
                channel_names=channel_names,
            )
            null_aurocs.append(sweep_result.selected_auroc)
        except Exception as e:
            logger.warning(f"Permutation {perm_i} failed: {e}")
            null_aurocs.append(float("nan"))

    null_aurocs = np.array(null_aurocs)
    valid = np.isfinite(null_aurocs)

    # p-value: (k+1)/(n+1) formula (matches permutation_utils.py convention)
    k = np.sum(null_aurocs[valid] >= observed_auroc)
    n_valid = int(valid.sum())
    pvalue = (k + 1) / (n_valid + 1)

    logger.info(
        f"Permutation null: observed AUROC={observed_auroc:.4f}, "
        f"null mean={np.nanmean(null_aurocs):.4f}, p={pvalue:.4f}"
    )

    return PermutationNullResult(
        observed_auroc=observed_auroc,
        null_aurocs=null_aurocs,
        pvalue=pvalue,
        selection_rule=selection_rule,
        observed_lam=observed_lam,
        observed_mu=observed_mu,
    )


# ---------------------------------------------------------------------------
# NULL 3: Bootstrap stability at fixed (λ,μ)
# ---------------------------------------------------------------------------

def run_bootstrap_stability(
    X: np.ndarray,
    y: np.ndarray,
    mask_ref: np.ndarray,
    groups: np.ndarray,
    lam: float,
    mu: float,
    n_boot: int = 200,
    roi_quantile: float = 0.9,
    trainer_config: Optional[TrainerConfig] = None,
    random_seed: int = 42,
    channel_names: Optional[Tuple[str, ...]] = None,
) -> BootstrapStabilityResult:
    """
    NULL 3: Bootstrap stability at FIXED (λ,μ).

    Resamples embryos with replacement within each class, fits model
    at fixed (λ,μ), and computes ROI stability metrics.

    Parameters
    ----------
    X, y, mask_ref, groups : data arrays
    lam, mu : fixed penalty parameters
    n_boot : number of bootstrap iterations
    roi_quantile : threshold for ROI extraction
    trainer_config : training config
    random_seed : base seed

    Returns
    -------
    BootstrapStabilityResult
    """
    from sklearn.utils.class_weight import compute_class_weight

    trainer_config = trainer_config or TrainerConfig()
    rng = np.random.default_rng(random_seed)

    # First: fit reference ROI on full data
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    cw = {int(c): float(w) for c, w in zip(classes, weights)}

    ref_result = train(X, y, mask_ref, class_weights=cw, lam=lam, mu=mu, config=trainer_config, channel_names=channel_names)
    ref_roi, _ = extract_roi(ref_result.w_full, mask_ref, quantile=roi_quantile)

    # Bootstrap
    ious = []
    areas = []
    ncomps = []
    bfracs = []

    unique_groups = np.unique(groups)

    # Validate: all samples from each embryo must share the same label
    for g in unique_groups:
        labels_for_g = np.unique(y[groups == g])
        assert len(labels_for_g) == 1, (
            f"Embryo '{g}' has mixed labels {labels_for_g}. "
            f"All samples from one embryo must share the same class label."
        )

    class_0_groups = unique_groups[[y[groups == g][0] == 0 for g in unique_groups]]
    class_1_groups = unique_groups[[y[groups == g][0] == 1 for g in unique_groups]]

    for boot_i in range(n_boot):
        if (boot_i + 1) % 50 == 0:
            logger.info(f"Bootstrap {boot_i + 1}/{n_boot}")

        # Resample embryos with replacement within each class
        boot_groups_0 = rng.choice(class_0_groups, size=len(class_0_groups), replace=True)
        boot_groups_1 = rng.choice(class_1_groups, size=len(class_1_groups), replace=True)
        boot_groups_all = np.concatenate([boot_groups_0, boot_groups_1])

        # Collect samples for bootstrapped embryos
        boot_idx = []
        for g in boot_groups_all:
            boot_idx.extend(np.where(groups == g)[0])
        boot_idx = np.array(boot_idx)

        X_boot = X[boot_idx]
        y_boot = y[boot_idx]

        # Fold-local class weights
        boot_classes = np.unique(y_boot)
        if len(boot_classes) < 2:
            continue  # skip degenerate bootstrap
        boot_weights = compute_class_weight("balanced", classes=boot_classes, y=y_boot)
        boot_cw = {int(c): float(w) for c, w in zip(boot_classes, boot_weights)}

        try:
            boot_result = train(
                X_boot, y_boot, mask_ref,
                class_weights=boot_cw,
                lam=lam, mu=mu,
                config=trainer_config,
                channel_names=channel_names,
            )
            boot_roi, boot_stats = extract_roi(
                boot_result.w_full, mask_ref, quantile=roi_quantile,
            )

            # IoU with reference ROI
            intersection = (ref_roi & boot_roi).sum()
            union = (ref_roi | boot_roi).sum()
            iou = float(intersection) / float(union) if union > 0 else 0.0

            ious.append(iou)
            areas.append(boot_stats["area_fraction"])
            ncomps.append(boot_stats["n_components"])
            bfracs.append(boot_stats["boundary_fraction"])

        except Exception as e:
            logger.warning(f"Bootstrap {boot_i} failed: {e}")

    ious = np.array(ious)
    areas = np.array(areas)
    ncomps = np.array(ncomps)
    bfracs = np.array(bfracs)

    logger.info(
        f"Bootstrap stability: IoU={np.mean(ious):.4f}±{np.std(ious):.4f} "
        f"(n={len(ious)} successful)"
    )

    return BootstrapStabilityResult(
        lam=lam,
        mu=mu,
        iou_distribution=ious,
        iou_mean=float(np.mean(ious)) if len(ious) > 0 else 0.0,
        iou_std=float(np.std(ious)) if len(ious) > 0 else 0.0,
        area_fraction_distribution=areas,
        n_components_distribution=ncomps,
        boundary_fraction_distribution=bfracs,
        reference_roi=ref_roi,
    )


# ---------------------------------------------------------------------------
# Combined runner
# ---------------------------------------------------------------------------

def run_nulls(
    X: np.ndarray,
    y: np.ndarray,
    mask_ref: np.ndarray,
    groups: np.ndarray,
    observed_auroc: float,
    selected_lam: float,
    selected_mu: float,
    selection_rule: str,
    lambda_values: List[float],
    mu_values: List[float],
    null_config: Optional[NullConfig] = None,
    sweep_config: Optional[SweepConfig] = None,
    trainer_config: Optional[TrainerConfig] = None,
    channel_names: Optional[Tuple[str, ...]] = None,
) -> NullsResult:
    """Run NULL 1 + NULL 3 based on null_config.null_mode."""
    from roi_config import NullMode

    null_config = null_config or NullConfig()

    do_permute = null_config.null_mode in (NullMode.PERMUTE, NullMode.BOTH)
    do_bootstrap = null_config.null_mode in (NullMode.BOOTSTRAP, NullMode.BOTH)

    perm_result = None
    boot_result = None

    if do_permute:
        logger.info("Running NULL 1: label permutation test...")
        perm_result = run_permutation_null(
            X, y, mask_ref, groups,
            observed_auroc=observed_auroc,
            observed_lam=selected_lam,
            observed_mu=selected_mu,
            selection_rule=selection_rule,
            lambda_values=lambda_values,
            mu_values=mu_values,
            n_permute=null_config.n_permute,
            sweep_config=sweep_config,
            trainer_config=trainer_config,
            random_seed=null_config.random_seed,
            channel_names=channel_names,
        )

    if do_bootstrap:
        logger.info("Running NULL 3: bootstrap stability...")
        boot_result = run_bootstrap_stability(
            X, y, mask_ref, groups,
            lam=selected_lam,
            mu=selected_mu,
            n_boot=null_config.n_boot,
            roi_quantile=null_config.boot_roi_quantile,
            trainer_config=trainer_config,
            random_seed=null_config.random_seed + 1000,
            channel_names=channel_names,
        )

    return NullsResult(
        permutation=perm_result,
        bootstrap=boot_result,
    )


def save_nulls_result(result: NullsResult, out_dir: str | Path) -> None:
    """Save null testing results to disk."""
    import json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    if result.permutation is not None:
        p = result.permutation
        np.save(out_dir / "null_aurocs.npy", p.null_aurocs)
        summary["permutation"] = {
            "observed_auroc": p.observed_auroc,
            "pvalue": p.pvalue,
            "null_mean": float(np.nanmean(p.null_aurocs)),
            "null_std": float(np.nanstd(p.null_aurocs)),
            "n_permutations": len(p.null_aurocs),
            "selection_rule": p.selection_rule,
            "selected_lam": p.observed_lam,
            "selected_mu": p.observed_mu,
        }

    if result.bootstrap is not None:
        b = result.bootstrap
        np.save(out_dir / "bootstrap_ious.npy", b.iou_distribution)
        np.save(out_dir / "bootstrap_reference_roi.npy", b.reference_roi)
        summary["bootstrap"] = {
            "lam": b.lam,
            "mu": b.mu,
            "iou_mean": b.iou_mean,
            "iou_std": b.iou_std,
            "area_fraction_mean": float(np.mean(b.area_fraction_distribution)),
            "n_components_mean": float(np.mean(b.n_components_distribution)),
            "boundary_fraction_mean": float(np.mean(b.boundary_fraction_distribution)),
            "n_bootstrap": len(b.iou_distribution),
        }

    with open(out_dir / "nulls_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Null results saved to {out_dir}")


__all__ = [
    "run_permutation_null",
    "run_bootstrap_stability",
    "run_nulls",
    "save_nulls_result",
    "PermutationNullResult",
    "BootstrapStabilityResult",
    "NullsResult",
]
