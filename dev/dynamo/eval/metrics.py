"""Quantitative evaluation metrics (model spec §11.1).

All metric functions operate on tensors and return per-sample values.
Aggregation (by horizon, by test tier) is handled separately in evaluate.py.

Metrics implemented:
    - Gaussian NLL (primary metric per spec)
    - MSE (simple displacement error)
    - Calibration fraction (what fraction of targets fall within predicted CI)
    - Energy distance (distributional metric when forward samples are available)
    - Mode utilization diagnostics (§11.1 bullet 4)
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import Tensor

from .predictions import PredictionResult


# ---------------------------------------------------------------------------
# Per-sample metrics
# ---------------------------------------------------------------------------

def gaussian_nll(
    predicted_mean: Tensor,
    predicted_cov_diag: Tensor,
    target: Tensor,
) -> Tensor:
    """Negative log-likelihood under diagonal Gaussian.

    Implements the Euler-Maruyama transition likelihood from spec §7.3:
        -log N(target; mean, diag(var))

    Args:
        predicted_mean: (B, D) predicted target locations.
        predicted_cov_diag: (B, D) diagonal variance (must be > 0).
        target: (B, D) observed target locations.

    Returns:
        (B,) per-sample NLL values.
    """
    D = target.shape[-1]
    diff = target - predicted_mean
    # Clamp variance away from zero for numerical stability
    var = predicted_cov_diag.clamp(min=1e-8)

    nll = (
        0.5 * D * math.log(2.0 * math.pi)
        + 0.5 * var.log().sum(dim=-1)
        + 0.5 * (diff.pow(2) / var).sum(dim=-1)
    )
    return nll  # (B,)


def mse(
    predicted_mean: Tensor,
    target: Tensor,
) -> Tensor:
    """Per-sample mean squared error.

    Args:
        predicted_mean: (B, D) predicted target locations.
        target: (B, D) observed target locations.

    Returns:
        (B,) per-sample MSE values.
    """
    return (predicted_mean - target).pow(2).mean(dim=-1)


def per_dim_mse(
    predicted_mean: Tensor,
    target: Tensor,
) -> Tensor:
    """Per-sample, per-dimension squared error.

    Args:
        predicted_mean: (B, D) predicted target locations.
        target: (B, D) observed target locations.

    Returns:
        (B, D) per-sample, per-dimension SE values.
    """
    return (predicted_mean - target).pow(2)


def calibration_fraction(
    predicted_mean: Tensor,
    predicted_cov_diag: Tensor,
    target: Tensor,
    level: float = 0.90,
) -> Tensor:
    """Fraction of targets within the predicted confidence ellipsoid.

    Under a diagonal Gaussian, the squared Mahalanobis distance follows
    a chi-squared distribution with D degrees of freedom. We check whether
    each sample falls within the `level`-quantile ellipsoid.

    Args:
        predicted_mean: (B, D) predicted means.
        predicted_cov_diag: (B, D) diagonal variances.
        target: (B, D) observed targets.
        level: Confidence level (default 0.90).

    Returns:
        Scalar: fraction of samples inside the ellipsoid.
    """
    D = target.shape[-1]
    var = predicted_cov_diag.clamp(min=1e-8)
    diff = target - predicted_mean

    # Squared Mahalanobis distance
    mahal_sq = (diff.pow(2) / var).sum(dim=-1)  # (B,)

    # Chi-squared quantile (approximate via normal for large D)
    # For exact: scipy.stats.chi2.ppf(level, D), but we avoid the scipy dep
    # Use Wilson-Hilferty approximation: chi2_D(p) ≈ D * (1 - 2/(9D) + z_p * sqrt(2/(9D)))^3
    z_p = (math.sqrt(2.0) * torch.erfinv(torch.tensor(2.0 * level - 1.0))).item()
    term = 1.0 - 2.0 / (9.0 * D) + z_p * math.sqrt(2.0 / (9.0 * D))
    chi2_threshold = D * (term ** 3)

    inside = (mahal_sq <= chi2_threshold).float()
    return inside.mean()


def energy_distance(
    samples: Tensor,
    target: Tensor,
) -> Tensor:
    """Energy distance between predicted sample distribution and observed target.

    For a single observed point y and predicted samples {x_1, ..., x_n}:
        ED = 2 * E[||x - y||] - E[||x - x'||]

    This is a proper metric on distributions. Lower is better.

    Args:
        samples: (B, N, D) forward-simulated predictions.
        target: (B, D) observed targets.

    Returns:
        (B,) per-sample energy distance.
    """
    B, N, D = samples.shape

    # E[||x - y||]: mean distance from samples to target
    target_expanded = target.unsqueeze(1)  # (B, 1, D)
    dist_to_target = (samples - target_expanded).norm(dim=-1).mean(dim=1)  # (B,)

    # E[||x - x'||]: mean pairwise distance between samples
    # Use random pairs for efficiency when N is large
    if N <= 100:
        # Exact pairwise
        diff = samples.unsqueeze(2) - samples.unsqueeze(1)  # (B, N, N, D)
        pairwise = diff.norm(dim=-1)  # (B, N, N)
        # Exclude diagonal (self-distances)
        mask = ~torch.eye(N, dtype=torch.bool, device=samples.device).unsqueeze(0)
        mean_pairwise = (pairwise * mask).sum(dim=(1, 2)) / (N * (N - 1))  # (B,)
    else:
        # Random pairing approximation
        perm = torch.randperm(N, device=samples.device)
        half = N // 2
        mean_pairwise = (samples[:, :half] - samples[:, perm[:half]]).norm(dim=-1).mean(dim=1)

    return 2.0 * dist_to_target - mean_pairwise


# ---------------------------------------------------------------------------
# Compute all metrics from a PredictionResult
# ---------------------------------------------------------------------------

def compute_sample_metrics(
    prediction: PredictionResult,
    target: Tensor,
) -> Dict[str, Tensor]:
    """Compute all per-sample metrics from a prediction result.

    Args:
        prediction: PredictionResult from a Predictor.
        target: (B, D) observed target locations.

    Returns:
        Dict of metric_name → (B,) tensor.
    """
    metrics: Dict[str, Tensor] = {}

    metrics["nll"] = gaussian_nll(
        prediction.predicted_mean, prediction.predicted_cov_diag, target
    )
    metrics["mse"] = mse(prediction.predicted_mean, target)
    metrics["rmse"] = metrics["mse"].sqrt()

    if prediction.forward_samples is not None:
        metrics["energy_distance"] = energy_distance(
            prediction.forward_samples, target
        )

    return metrics


# ---------------------------------------------------------------------------
# Mode utilization diagnostics (spec §11.1)
# ---------------------------------------------------------------------------

def mode_diagnostics(prediction: PredictionResult) -> Dict[str, Tensor]:
    """Extract mode utilization diagnostics from a PredictionResult.

    Returns dict with available diagnostics:
        - v_norm: (B,) ||v_e|| local correction norms
        - residual_norm: (B,) solve residual norms
        - loading_magnitudes: (B, M) per-mode |c_{e,m}|
        - rate: (B,) R_e values

    Args:
        prediction: PredictionResult with mode diagnostics populated.

    Returns:
        Dict of diagnostic_name → tensor. Empty dict if no diagnostics available.
    """
    diag: Dict[str, Tensor] = {}

    if prediction.local_correction_norm is not None:
        diag["v_norm"] = prediction.local_correction_norm

    if prediction.residual_norm is not None:
        diag["residual_norm"] = prediction.residual_norm

    if prediction.mode_loadings is not None:
        diag["loading_magnitudes"] = prediction.mode_loadings.abs()
        diag["loading_norm"] = prediction.mode_loadings.norm(dim=-1)

    if prediction.rate is not None:
        diag["rate"] = prediction.rate

    return diag
