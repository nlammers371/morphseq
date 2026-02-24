"""Weights & Biases logging utilities for the three-model comparison (§11.2).

All functions are safe to call without an active wandb run — they check
`wandb.run` and silently skip if no run is active. This allows the same
evaluation code to work in notebooks (no wandb) and training scripts (with wandb).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .evaluate import EvalResult, ComparisonResult


def _wandb_available() -> bool:
    """Check if wandb is importable and a run is active."""
    try:
        import wandb
        return wandb.run is not None
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Single-model logging
# ---------------------------------------------------------------------------

def log_eval_results(
    results: EvalResult,
    model_name: str,
    step: int,
    prefix: str = "eval",
) -> None:
    """Log evaluation results for a single model to W&B.

    Logs:
        - Overall metrics as scalars: {prefix}/{model_name}/nll, etc.
        - Per-horizon metrics: {prefix}/{model_name}/horizon_{k}/nll, etc.
        - Calibration: {prefix}/{model_name}/calibration_90
        - Mode diagnostics (if available)

    Args:
        results: EvalResult from run_evaluation().
        model_name: Model identifier (e.g., "kernel", "phi0", "full").
        step: Training step or epoch number.
        prefix: Metric prefix for organization.
    """
    if not _wandb_available():
        return

    import wandb

    log_dict: Dict[str, Any] = {}
    base = f"{prefix}/{model_name}"

    # Overall metrics
    for name, value in results.metrics.items():
        log_dict[f"{base}/{name}"] = value

    log_dict[f"{base}/calibration_90"] = results.calibration
    log_dict[f"{base}/n_samples"] = results.n_samples
    log_dict[f"{base}/tier"] = results.tier

    # Per-horizon metrics
    for k, horizon_metrics in results.per_horizon.items():
        for name, value in horizon_metrics.items():
            log_dict[f"{base}/horizon_{k}/{name}"] = value

    # Mode diagnostics
    for name, value in results.mode_diagnostics.items():
        log_dict[f"{base}/mode_diag/{name}"] = value

    wandb.log(log_dict, step=step)


# ---------------------------------------------------------------------------
# Three-model comparison (spec §11.2)
# ---------------------------------------------------------------------------

def log_comparison(
    comparison: ComparisonResult,
    step: int,
    prefix: str = "eval",
) -> None:
    """Log three-model comparison to W&B.

    Creates:
        - Scalar metrics for each model
        - A W&B Table with the comparison summary
        - Bar charts comparing key metrics across models

    Args:
        comparison: ComparisonResult with kernel, phi0, and optionally full results.
        step: Training step or epoch number.
        prefix: Metric prefix.
    """
    if not _wandb_available():
        return

    import wandb

    # Log individual model results
    log_eval_results(comparison.kernel, "kernel", step, prefix)
    log_eval_results(comparison.phi0, "phi0", step, prefix)
    if comparison.full is not None:
        log_eval_results(comparison.full, "full", step, prefix)

    # Create comparison table
    models = [
        ("kernel", comparison.kernel),
        ("phi0", comparison.phi0),
    ]
    if comparison.full is not None:
        models.append(("full", comparison.full))

    columns = ["model", "tier", "nll", "mse", "rmse", "calibration_90", "n_samples"]
    table_data = []
    for name, result in models:
        table_data.append([
            name,
            result.tier,
            result.metrics.get("nll", float("nan")),
            result.metrics.get("mse", float("nan")),
            result.metrics.get("rmse", float("nan")),
            result.calibration,
            result.n_samples,
        ])

    table = wandb.Table(columns=columns, data=table_data)
    wandb.log({f"{prefix}/comparison_table": table}, step=step)

    # Log deltas (value of learned dynamics, value of modes)
    kernel_nll = comparison.kernel.metrics.get("nll", float("nan"))
    phi0_nll = comparison.phi0.metrics.get("nll", float("nan"))

    wandb.log({
        f"{prefix}/delta/kernel_to_phi0_nll": kernel_nll - phi0_nll,
    }, step=step)

    if comparison.full is not None:
        full_nll = comparison.full.metrics.get("nll", float("nan"))
        wandb.log({
            f"{prefix}/delta/phi0_to_full_nll": phi0_nll - full_nll,
            f"{prefix}/delta/kernel_to_full_nll": kernel_nll - full_nll,
        }, step=step)


# ---------------------------------------------------------------------------
# Horizon breakdown logging
# ---------------------------------------------------------------------------

def log_horizon_comparison(
    comparison: ComparisonResult,
    step: int,
    prefix: str = "eval",
) -> None:
    """Log per-horizon metric comparison as a W&B table.

    Creates a table with one row per (model, horizon_k) combination.

    Args:
        comparison: ComparisonResult.
        step: Training step.
        prefix: Metric prefix.
    """
    if not _wandb_available():
        return

    import wandb

    models = [
        ("kernel", comparison.kernel),
        ("phi0", comparison.phi0),
    ]
    if comparison.full is not None:
        models.append(("full", comparison.full))

    columns = ["model", "horizon_k", "nll", "mse", "rmse", "n_samples"]
    table_data = []
    for name, result in models:
        for k, hm in sorted(result.per_horizon.items()):
            table_data.append([
                name,
                k,
                hm.get("nll", float("nan")),
                hm.get("mse", float("nan")),
                hm.get("rmse", float("nan")),
                hm.get("n_samples", 0),
            ])

    table = wandb.Table(columns=columns, data=table_data)
    wandb.log({f"{prefix}/horizon_comparison": table}, step=step)


# ---------------------------------------------------------------------------
# Print summary (no wandb required)
# ---------------------------------------------------------------------------

def print_eval_summary(results: EvalResult, model_name: str) -> None:
    """Print a human-readable evaluation summary to stdout.

    Args:
        results: EvalResult from run_evaluation().
        model_name: Display name for the model.
    """
    print(f"\n{'='*60}")
    print(f"  {model_name} — Tier: {results.tier} ({results.n_samples} samples)")
    print(f"{'='*60}")

    for name, value in sorted(results.metrics.items()):
        print(f"  {name:<25} {value:.4f}")
    print(f"  {'calibration_90%':<25} {results.calibration:.4f}")

    if results.per_horizon:
        print(f"\n  Per-horizon breakdown:")
        for k in sorted(results.per_horizon):
            hm = results.per_horizon[k]
            n = hm.get("n_samples", "?")
            print(f"    k={k}: NLL={hm.get('nll', float('nan')):.4f}  "
                  f"MSE={hm.get('mse', float('nan')):.4f}  (n={n})")

    if results.mode_diagnostics:
        print(f"\n  Mode diagnostics:")
        for name, value in sorted(results.mode_diagnostics.items()):
            print(f"    {name:<25} {value:.4f}")
