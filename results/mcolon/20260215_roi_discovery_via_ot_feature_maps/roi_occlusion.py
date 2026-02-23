"""Phase 2.0 occlusion validation with OOB bootstrap evaluation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

from roi_perturbation import apply_perturbation, compute_spatial_baseline
from roi_resampling import iter_bootstrap_groups
from roi_trainer import compute_logits, extract_roi, train

logger = logging.getLogger(__name__)


@dataclass
class OcclusionEvalResult:
    observed_gap: float
    gap_per_sample: np.ndarray
    delete_gap: np.ndarray
    preserve_gap: np.ndarray
    auroc_orig: float
    auroc_delete: float
    auroc_preserve: float


@dataclass
class BootstrapOcclusionResult:
    bootstrap_gaps: np.ndarray
    ci95: Tuple[float, float]
    frac_positive: float
    q_sensitivity: Dict[str, Dict[str, float]]
    n_successful: int
    n_skipped_empty_oob: int


def evaluate_occlusion(
    X: np.ndarray,
    y: np.ndarray,
    w_full: np.ndarray,
    b: float,
    roi_mask: np.ndarray,
    baseline: np.ndarray,
) -> OcclusionEvalResult:
    z_orig = compute_logits(X, w_full, b)
    z_delete = compute_logits(apply_perturbation(X, 1.0 - roi_mask, baseline), w_full, b)
    z_preserve = compute_logits(apply_perturbation(X, roi_mask, baseline), w_full, b)

    delete_gap = z_orig - z_delete
    preserve_gap = z_orig - z_preserve
    gap_per_sample = delete_gap - preserve_gap
    observed_gap = float(np.mean(gap_per_sample))

    if np.unique(y).size < 2:
        logger.warning("Single-class evaluation set: AUROC metrics are undefined and set to NaN")
        auroc_orig = float("nan")
        auroc_delete = float("nan")
        auroc_preserve = float("nan")
    else:
        auroc_orig = float(roc_auc_score(y, z_orig))
        auroc_delete = float(roc_auc_score(y, z_delete))
        auroc_preserve = float(roc_auc_score(y, z_preserve))

    return OcclusionEvalResult(
        observed_gap=observed_gap,
        gap_per_sample=gap_per_sample,
        delete_gap=delete_gap,
        preserve_gap=preserve_gap,
        auroc_orig=auroc_orig,
        auroc_delete=auroc_delete,
        auroc_preserve=auroc_preserve,
    )


def bootstrap_occlusion(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    mask_ref: np.ndarray,
    channel_names: Sequence[str],
    lam: float,
    mu: float,
    n_boot: int = 200,
    roi_quantiles: Tuple[float, ...] = (0.85, 0.90, 0.95),
    random_seed: int = 42,
) -> BootstrapOcclusionResult:
    gaps = []
    q_gaps: Dict[float, list] = {q: [] for q in roi_quantiles}
    skipped_empty = 0

    for i, sample in enumerate(iter_bootstrap_groups(groups, y, n_boot=n_boot, random_seed=random_seed)):
        if sample.oob_is_empty:
            skipped_empty += 1
            logger.info("Bootstrap %d skipped: empty OOB", i)
            continue

        inbag_idx = np.concatenate([np.where(groups == g)[0] for g in sample.inbag_group_ids])
        oob_idx = np.concatenate([np.where(groups == g)[0] for g in sample.oob_group_ids])

        X_in, y_in = X[inbag_idx], y[inbag_idx]
        X_oob, y_oob = X[oob_idx], y[oob_idx]

        classes = np.unique(y_in)
        if classes.size < 2:
            continue
        weights = compute_class_weight("balanced", classes=classes, y=y_in)
        cw = {int(c): float(w) for c, w in zip(classes, weights)}

        fit_result = train(
            X_in,
            y_in,
            mask_ref,
            class_weights=cw,
            lam=lam,
            mu=mu,
            channel_names=channel_names,
        )

        baseline = compute_spatial_baseline(
            X_in,
            y_in,
            channel_names=channel_names,
        )

        for q in roi_quantiles:
            roi_mask, _ = extract_roi(fit_result.w_full, mask_ref, quantile=q)
            eval_result = evaluate_occlusion(X_oob, y_oob, fit_result.w_full, fit_result.b, roi_mask, baseline)
            q_gaps[q].append(eval_result.observed_gap)
            if np.isclose(q, 0.9):
                gaps.append(eval_result.observed_gap)

            if sample.oob_single_class:
                logger.info("Bootstrap %d has single-class OOB; AUROC metrics are NaN by design", i)

    gaps_np = np.array(gaps, dtype=float)
    if gaps_np.size == 0:
        ci = (float("nan"), float("nan"))
        frac_pos = float("nan")
    else:
        ci = (float(np.quantile(gaps_np, 0.025)), float(np.quantile(gaps_np, 0.975)))
        frac_pos = float(np.mean(gaps_np > 0))

    q_sensitivity = {}
    for q, vals in q_gaps.items():
        arr = np.array(vals, dtype=float)
        q_sensitivity[f"{q:.2f}"] = {
            "mean_gap": float(np.nanmean(arr)) if arr.size else float("nan"),
            "n": int(arr.size),
        }

    return BootstrapOcclusionResult(
        bootstrap_gaps=gaps_np,
        ci95=ci,
        frac_positive=frac_pos,
        q_sensitivity=q_sensitivity,
        n_successful=int(gaps_np.size),
        n_skipped_empty_oob=skipped_empty,
    )


def save_occlusion_result(result: BootstrapOcclusionResult, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "bootstrap_gaps.npy", result.bootstrap_gaps)
    summary = {
        "ci95": list(result.ci95),
        "frac_positive": result.frac_positive,
        "q_sensitivity": result.q_sensitivity,
        "n_successful": result.n_successful,
        "n_skipped_empty_oob": result.n_skipped_empty_oob,
    }
    with open(out / "occlusion_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


__all__ = [
    "OcclusionEvalResult",
    "BootstrapOcclusionResult",
    "evaluate_occlusion",
    "bootstrap_occlusion",
    "save_occlusion_result",
]
