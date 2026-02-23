"""
λ/μ sweep + deterministic selection for ROI discovery.

Phase 1: coarse grid sweep to identify a sensible operating region.
Selection via Pareto knee (recommended) or ε-best rule.

See PLAN.md Section E for specification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from roi_config import SelectionRule, SweepConfig, TrainerConfig
from roi_trainer import TrainResult, compute_logits, extract_roi, train

logger = logging.getLogger(__name__)


@dataclass
class SweepEntry:
    """One row of the sweep table."""
    lam: float
    mu: float
    fold: int
    auroc: float
    area_fraction: float
    n_components: int
    boundary_fraction: float
    train_result: Optional[TrainResult] = None


@dataclass
class SweepResult:
    """Full sweep output."""
    sweep_table: pd.DataFrame
    selected_lam: float
    selected_mu: float
    selected_auroc: float
    selected_complexity: Dict
    selection_rule: str
    selection_metadata: Dict


def run_sweep(
    X: np.ndarray,
    y: np.ndarray,
    mask_ref: np.ndarray,
    groups: np.ndarray,
    lambda_values: List[float],
    mu_values: List[float],
    sweep_config: Optional[SweepConfig] = None,
    trainer_config: Optional[TrainerConfig] = None,
    channel_names: Optional[Tuple[str, ...]] = None,
) -> SweepResult:
    """
    Run λ×μ sweep with cross-validation.

    For each (λ, μ) pair, trains on CV folds and records AUROC + complexity.

    Parameters
    ----------
    X : ndarray, (N, 512, 512, C)
    y : ndarray, (N,)
    mask_ref : ndarray, (512, 512)
    groups : ndarray, (N,) — embryo_id for GroupKFold
    lambda_values, mu_values : lists of floats
    sweep_config : SweepConfig
    trainer_config : TrainerConfig

    Returns
    -------
    SweepResult with full table + selected (λ, μ).
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.utils.class_weight import compute_class_weight

    sweep_config = sweep_config or SweepConfig()
    trainer_config = trainer_config or TrainerConfig()

    n_folds = sweep_config.n_cv_folds
    gkf = GroupKFold(n_splits=n_folds)
    folds = list(gkf.split(X, y, groups))

    entries: List[Dict] = []
    total_combos = len(lambda_values) * len(mu_values)

    for combo_i, (lam, mu) in enumerate(product(lambda_values, mu_values)):
        logger.info(
            f"Sweep [{combo_i + 1}/{total_combos}]: λ={lam:.2e}, μ={mu:.2e}"
        )

        fold_aurocs = []
        fold_areas = []
        fold_ncomp = []
        fold_bfrac = []

        for fold_i, (train_idx, val_idx) in enumerate(folds):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Fold-local class weights
            classes = np.unique(y_train)
            weights = compute_class_weight("balanced", classes=classes, y=y_train)
            cw = {int(c): float(w) for c, w in zip(classes, weights)}

            # Train
            result = train(
                X_train, y_train, mask_ref,
                class_weights=cw,
                lam=lam, mu=mu,
                config=trainer_config,
                channel_names=channel_names,
            )

            # Predict on validation
            w_full = result.w_full
            logits_val = compute_logits(X_val, w_full, result.b)

            # AUROC
            try:
                probs = 1.0 / (1.0 + np.exp(-logits_val))
                auroc = roc_auc_score(y_val, probs)
            except ValueError:
                auroc = 0.5  # single class in val fold

            # ROI complexity
            roi_mask, roi_stats = extract_roi(
                w_full, mask_ref, quantile=sweep_config.roi_quantile,
            )

            fold_aurocs.append(auroc)
            fold_areas.append(roi_stats["area_fraction"])
            fold_ncomp.append(roi_stats["n_components"])
            fold_bfrac.append(roi_stats["boundary_fraction"])

        entries.append({
            "lam": lam,
            "mu": mu,
            "auroc_mean": np.mean(fold_aurocs),
            "auroc_std": np.std(fold_aurocs),
            "area_fraction_mean": np.mean(fold_areas),
            "n_components_mean": np.mean(fold_ncomp),
            "boundary_fraction_mean": np.mean(fold_bfrac),
            "auroc_folds": fold_aurocs,
            "area_folds": fold_areas,
        })

    sweep_df = pd.DataFrame(entries)
    logger.info(f"Sweep complete: {len(sweep_df)} (λ,μ) combinations evaluated")

    # Deterministic selection
    selected_lam, selected_mu, sel_meta = _select_lambda_mu(
        sweep_df, sweep_config.selection_rule, sweep_config,
    )

    # Get the row for the selected (λ,μ)
    sel_row = sweep_df[
        (sweep_df["lam"] == selected_lam) & (sweep_df["mu"] == selected_mu)
    ].iloc[0]

    return SweepResult(
        sweep_table=sweep_df,
        selected_lam=selected_lam,
        selected_mu=selected_mu,
        selected_auroc=float(sel_row["auroc_mean"]),
        selected_complexity={
            "area_fraction": float(sel_row["area_fraction_mean"]),
            "n_components": float(sel_row["n_components_mean"]),
            "boundary_fraction": float(sel_row["boundary_fraction_mean"]),
        },
        selection_rule=sweep_config.selection_rule.value,
        selection_metadata=sel_meta,
    )


def _select_lambda_mu(
    sweep_df: pd.DataFrame,
    rule: SelectionRule,
    config: SweepConfig,
) -> Tuple[float, float, Dict]:
    """
    Deterministic selection of (λ, μ) from sweep table.

    Option A (PARETO_KNEE): Knee on Pareto front of AUROC vs complexity.
    Option B (EPSILON_BEST): Smallest complexity within ε of best AUROC.
    """
    if rule == SelectionRule.PARETO_KNEE:
        return _select_pareto_knee(sweep_df, config.pareto_beta)
    elif rule == SelectionRule.EPSILON_BEST:
        return _select_epsilon_best(sweep_df, config.epsilon_auroc)
    else:
        raise ValueError(f"Unknown selection rule: {rule}")


def _select_pareto_knee(
    df: pd.DataFrame,
    beta: float,
) -> Tuple[float, float, Dict]:
    """
    Find the knee point on the Pareto front (AUROC vs area_fraction).

    Uses a simple score: score = AUROC - beta * area_fraction
    Higher score = better trade-off.

    beta controls the AUROC-vs-simplicity trade-off:
        beta=1.0 — equal weight to AUROC and complexity
        beta>1.0 — prefer simpler (smaller) ROIs
        beta<1.0 — prefer higher AUROC even if ROI is larger
    """
    df = df.copy()
    df["pareto_score"] = df["auroc_mean"] - beta * df["area_fraction_mean"]

    best_idx = df["pareto_score"].idxmax()
    best = df.loc[best_idx]

    return (
        float(best["lam"]),
        float(best["mu"]),
        {
            "rule": "pareto_knee",
            "beta": beta,
            "pareto_score": float(best["pareto_score"]),
            "best_auroc_in_sweep": float(df["auroc_mean"].max()),
        },
    )


def _select_epsilon_best(
    df: pd.DataFrame,
    epsilon: float,
) -> Tuple[float, float, Dict]:
    """
    Select smallest-complexity (λ,μ) within ε of best AUROC.
    """
    best_auroc = df["auroc_mean"].max()
    threshold = best_auroc - epsilon

    candidates = df[df["auroc_mean"] >= threshold].copy()
    # Among candidates, pick smallest area_fraction
    best_idx = candidates["area_fraction_mean"].idxmin()
    best = candidates.loc[best_idx]

    return (
        float(best["lam"]),
        float(best["mu"]),
        {
            "rule": "epsilon_best",
            "epsilon": epsilon,
            "best_auroc_in_sweep": float(best_auroc),
            "auroc_threshold": float(threshold),
            "n_candidates": len(candidates),
        },
    )


def save_sweep_result(result: SweepResult, out_dir: str | Path) -> None:
    """Save sweep results to disk."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save sweep table (drop list columns for Parquet compatibility)
    table_df = result.sweep_table.drop(columns=["auroc_folds", "area_folds"], errors="ignore")
    table_df.to_parquet(out_dir / "sweep_table.parquet", index=False)
    table_df.to_csv(out_dir / "sweep_table.csv", index=False)

    # Save selection metadata
    import json
    selection_info = {
        "selected_lam": result.selected_lam,
        "selected_mu": result.selected_mu,
        "selected_auroc": result.selected_auroc,
        "selected_complexity": result.selected_complexity,
        "selection_rule": result.selection_rule,
        "selection_metadata": result.selection_metadata,
    }
    with open(out_dir / "selection.json", "w") as f:
        json.dump(selection_info, f, indent=2)

    logger.info(
        f"Sweep saved to {out_dir}: "
        f"selected λ={result.selected_lam:.2e}, μ={result.selected_mu:.2e}, "
        f"AUROC={result.selected_auroc:.4f}"
    )


__all__ = [
    "run_sweep",
    "SweepEntry",
    "SweepResult",
    "save_sweep_result",
]
