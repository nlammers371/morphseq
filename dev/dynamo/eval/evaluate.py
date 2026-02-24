"""Evaluation orchestration: run predictions, compute metrics, aggregate.

Runs a predictor over a DataLoader, computes per-sample metrics, then
aggregates by prediction horizon and optional test tier label. Designed
to produce the same output format for all three models (kernel baseline,
phi0-only, full model) so they can be directly compared.

Model spec reference: §11 (evaluation stack), §10 (test set design).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..data.dataset import FragmentBatch, FragmentDataset, fragment_collate_fn, worker_init_fn
from .predictions import Predictor, PredictionResult
from .metrics import (
    compute_sample_metrics,
    calibration_fraction,
    mode_diagnostics,
)


# ---------------------------------------------------------------------------
# Evaluation result container
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Aggregated evaluation results.

    Attributes:
        metrics: Overall aggregated metrics {name: scalar value}.
        per_horizon: Metrics broken out by prediction horizon {k: {name: value}}.
        calibration: Calibration fraction at 90% level.
        n_samples: Total number of evaluated samples.
        tier: Test tier label (for logging).
        mode_diagnostics: Aggregated mode utilization stats (if available).
    """
    metrics: Dict[str, float]
    per_horizon: Dict[int, Dict[str, float]]
    calibration: float
    n_samples: int
    tier: str = "all"
    mode_diagnostics: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Horizon binning
# ---------------------------------------------------------------------------

def _bin_horizon_k(horizon_dt: Tensor, delta_t: Tensor) -> Tensor:
    """Convert continuous horizon_dt to integer horizon k (in frame units).

    k = round(horizon_dt / delta_t), clamped to [1, inf).

    Args:
        horizon_dt: (B,) time gap to target in seconds.
        delta_t: (B,) experiment-level median Δt in seconds.

    Returns:
        (B,) integer horizon k.
    """
    k = torch.round(horizon_dt / delta_t.clamp(min=1e-6)).long()
    return k.clamp(min=1)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_evaluation(
    predictor: Predictor,
    dataset: FragmentDataset,
    n_batches: int = 50,
    batch_size: int = 32,
    tier: str = "all",
    device: Optional[torch.device] = None,
) -> EvalResult:
    """Run evaluation loop and aggregate metrics.

    Args:
        predictor: Any object implementing the Predictor protocol.
        dataset: FragmentDataset to evaluate on.
        n_batches: Number of batches to evaluate (controls total samples).
        batch_size: Samples per batch.
        tier: Test tier label for logging (e.g., "tier1_novel", "tier2_within").
        device: Device to run on. None = CPU.

    Returns:
        EvalResult with aggregated metrics.
    """
    # Override epoch_length to control evaluation size
    eval_length = n_batches * batch_size
    dataset._epoch_length = eval_length

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=fragment_collate_fn,
        num_workers=0,  # Eval is typically small enough for main process
        shuffle=False,
    )

    # Accumulators
    all_nll: List[Tensor] = []
    all_mse: List[Tensor] = []
    all_rmse: List[Tensor] = []
    all_energy: List[Tensor] = []
    all_horizon_k: List[Tensor] = []
    all_means: List[Tensor] = []
    all_covs: List[Tensor] = []
    all_targets: List[Tensor] = []

    # Mode diagnostic accumulators
    all_v_norm: List[Tensor] = []
    all_residual_norm: List[Tensor] = []
    all_loading_norm: List[Tensor] = []
    all_rate: List[Tensor] = []

    for batch in loader:
        if device is not None:
            batch = _batch_to_device(batch, device)

        prediction = predictor.predict(batch)
        sample_metrics = compute_sample_metrics(prediction, batch.target)

        # Compute horizon k
        horizon_k = _bin_horizon_k(batch.horizon_dt, batch.delta_t)

        all_nll.append(sample_metrics["nll"])
        all_mse.append(sample_metrics["mse"])
        all_rmse.append(sample_metrics["rmse"])
        all_horizon_k.append(horizon_k)
        all_means.append(prediction.predicted_mean)
        all_covs.append(prediction.predicted_cov_diag)
        all_targets.append(batch.target)

        if "energy_distance" in sample_metrics:
            all_energy.append(sample_metrics["energy_distance"])

        # Mode diagnostics
        diag = mode_diagnostics(prediction)
        if "v_norm" in diag:
            all_v_norm.append(diag["v_norm"])
        if "residual_norm" in diag:
            all_residual_norm.append(diag["residual_norm"])
        if "loading_norm" in diag:
            all_loading_norm.append(diag["loading_norm"])
        if "rate" in diag:
            all_rate.append(diag["rate"])

    # Concatenate
    nll = torch.cat(all_nll)
    mse_vals = torch.cat(all_mse)
    rmse_vals = torch.cat(all_rmse)
    horizon_k = torch.cat(all_horizon_k)
    means = torch.cat(all_means)
    covs = torch.cat(all_covs)
    targets = torch.cat(all_targets)

    n_total = len(nll)

    # Overall metrics
    overall: Dict[str, float] = {
        "nll": nll.mean().item(),
        "mse": mse_vals.mean().item(),
        "rmse": rmse_vals.mean().item(),
    }
    if all_energy:
        overall["energy_distance"] = torch.cat(all_energy).mean().item()

    # Calibration
    cal = calibration_fraction(means, covs, targets, level=0.90).item()

    # Per-horizon breakdown
    per_horizon: Dict[int, Dict[str, float]] = {}
    unique_k = horizon_k.unique().tolist()
    for k in sorted(unique_k):
        mask = horizon_k == k
        per_horizon[int(k)] = {
            "nll": nll[mask].mean().item(),
            "mse": mse_vals[mask].mean().item(),
            "rmse": rmse_vals[mask].mean().item(),
            "n_samples": int(mask.sum().item()),
        }
        if all_energy:
            energy = torch.cat(all_energy)
            per_horizon[int(k)]["energy_distance"] = energy[mask].mean().item()

    # Mode diagnostic aggregation
    mode_diag: Dict[str, float] = {}
    if all_v_norm:
        v = torch.cat(all_v_norm)
        mode_diag["v_norm_mean"] = v.mean().item()
        mode_diag["v_norm_std"] = v.std().item()
        mode_diag["v_norm_median"] = v.median().item()
    if all_residual_norm:
        r = torch.cat(all_residual_norm)
        mode_diag["residual_norm_mean"] = r.mean().item()
    if all_loading_norm:
        ln = torch.cat(all_loading_norm)
        mode_diag["loading_norm_mean"] = ln.mean().item()
    if all_rate:
        rate = torch.cat(all_rate)
        mode_diag["rate_mean"] = rate.mean().item()
        mode_diag["rate_std"] = rate.std().item()

    return EvalResult(
        metrics=overall,
        per_horizon=per_horizon,
        calibration=cal,
        n_samples=n_total,
        tier=tier,
        mode_diagnostics=mode_diag,
    )


# ---------------------------------------------------------------------------
# Three-model comparison (spec §11.2)
# ---------------------------------------------------------------------------

@dataclass
class ComparisonResult:
    """Side-by-side results from the three evaluation models.

    Attributes:
        kernel: EvalResult from kernel baseline.
        phi0: EvalResult from phi0-only model.
        full: EvalResult from full model (may be None during early development).
    """
    kernel: EvalResult
    phi0: EvalResult
    full: Optional[EvalResult] = None

    def summary_table(self) -> str:
        """Format a readable comparison table."""
        models = [("Kernel", self.kernel), ("φ₀-only", self.phi0)]
        if self.full is not None:
            models.append(("Full", self.full))

        lines = [f"{'Metric':<25}" + "".join(f"{name:>15}" for name, _ in models)]
        lines.append("-" * len(lines[0]))

        metric_names = ["nll", "mse", "rmse"]
        if "energy_distance" in self.kernel.metrics:
            metric_names.append("energy_distance")

        for m in metric_names:
            row = f"{m:<25}"
            for _, result in models:
                val = result.metrics.get(m, float("nan"))
                row += f"{val:>15.4f}"
            lines.append(row)

        # Calibration row
        row = f"{'calibration_90%':<25}"
        for _, result in models:
            row += f"{result.calibration:>15.4f}"
        lines.append(row)

        # Per-horizon breakdown
        all_horizons = sorted(set().union(*(r.per_horizon.keys() for _, r in models)))
        for k in all_horizons:
            lines.append(f"\n  Horizon k={k}:")
            for m in ["nll", "mse"]:
                row = f"    {m:<21}"
                for _, result in models:
                    val = result.per_horizon.get(k, {}).get(m, float("nan"))
                    row += f"{val:>15.4f}"
                lines.append(row)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _batch_to_device(batch: FragmentBatch, device: torch.device) -> FragmentBatch:
    """Move all batch tensors to the specified device."""
    return FragmentBatch(
        context=batch.context.to(device),
        context_mask=batch.context_mask.to(device),
        target=batch.target.to(device),
        time_deltas=batch.time_deltas.to(device),
        horizon_dt=batch.horizon_dt.to(device),
        delta_t=batch.delta_t.to(device),
        temperature=batch.temperature.to(device),
        class_idx=batch.class_idx.to(device),
        embryo_idx=batch.embryo_idx.to(device),
    )
