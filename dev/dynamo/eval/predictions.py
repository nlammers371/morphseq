"""Prediction interface and dummy predictors for eval pipeline testing.

Defines the common prediction result format and protocol that all models
(kernel baseline, phi0-only, full model) must implement. Includes simple
predictors for end-to-end pipeline validation before any model is trained.

Model spec references: §11 (evaluation), §15.2 (build sequence step 2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

import torch
from torch import Tensor

from ..data.dataset import FragmentBatch


# ---------------------------------------------------------------------------
# Prediction result
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """Standardized output from any predictor.

    All tensors are on the same device as the input batch.

    Attributes:
        predicted_mean: (B, D) predicted target location.
        predicted_cov_diag: (B, D) diagonal covariance (variance per dim).
        forward_samples: (B, n_samples, D) optional forward-simulated trajectories.
        horizon_k: (B,) integer horizon used for each sample (in frame units).
        diffusion_D: Global diffusion coefficient (scalar), if applicable.

        Mode diagnostics (populated only by models with modes):
        mode_loadings: (B, M) inferred c_e per embryo.
        local_correction_norm: (B,) ||v_e|| per embryo.
        residual_norm: (B,) ||residual|| after closed-form solve.
        rate: (B,) inferred R_e per embryo.
    """
    predicted_mean: Tensor               # (B, D)
    predicted_cov_diag: Tensor           # (B, D) diagonal variance
    forward_samples: Optional[Tensor] = None  # (B, n_samples, D)
    horizon_k: Optional[Tensor] = None   # (B,) int
    diffusion_D: Optional[float] = None

    # Mode diagnostics (optional)
    mode_loadings: Optional[Tensor] = None       # (B, M)
    local_correction_norm: Optional[Tensor] = None  # (B,)
    residual_norm: Optional[Tensor] = None       # (B,)
    rate: Optional[Tensor] = None                # (B,)


# ---------------------------------------------------------------------------
# Predictor protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Predictor(Protocol):
    """Interface that all models/baselines must implement."""

    def predict(self, batch: FragmentBatch) -> PredictionResult:
        """Produce predictions for a batch of fragments.

        Args:
            batch: A FragmentBatch from the data pipeline.

        Returns:
            PredictionResult with at least predicted_mean and predicted_cov_diag.
        """
        ...


# ---------------------------------------------------------------------------
# Dummy predictors for pipeline testing
# ---------------------------------------------------------------------------

class PersistencePredictor:
    """Predict that the target equals the last observed context frame.

    This is a trivial baseline: the "prediction" is simply that the embryo
    stays where it was last seen. Variance is set to a fixed isotropic value
    scaled by the horizon time gap.
    """

    def __init__(self, noise_scale: float = 0.1) -> None:
        self.noise_scale = noise_scale

    def predict(self, batch: FragmentBatch) -> PredictionResult:
        B = batch.context.shape[0]
        D = batch.context.shape[2]

        # Last real context frame for each sample
        lengths = batch.context_mask.sum(dim=1).long()  # (B,)
        last_idx = lengths - 1
        last_frame = batch.context[torch.arange(B, device=batch.context.device), last_idx]  # (B, D)

        # Variance scales with horizon_dt (longer horizon → more uncertainty)
        var = (self.noise_scale ** 2) * batch.horizon_dt.unsqueeze(-1).expand(B, D)

        return PredictionResult(
            predicted_mean=last_frame,
            predicted_cov_diag=var,
        )


class LinearExtrapolationPredictor:
    """Predict by linearly extrapolating the last observed velocity.

    Uses the last two context frames to estimate velocity, then extrapolates
    forward by horizon_dt. Variance is isotropic and scales with horizon.
    """

    def __init__(self, noise_scale: float = 0.1) -> None:
        self.noise_scale = noise_scale

    def predict(self, batch: FragmentBatch) -> PredictionResult:
        B = batch.context.shape[0]
        D = batch.context.shape[2]
        device = batch.context.device

        lengths = batch.context_mask.sum(dim=1).long()  # (B,)
        last_idx = lengths - 1
        prev_idx = torch.clamp(last_idx - 1, min=0)

        last_frame = batch.context[torch.arange(B, device=device), last_idx]
        prev_frame = batch.context[torch.arange(B, device=device), prev_idx]

        # Velocity from last two frames, normalized by delta_t
        velocity = (last_frame - prev_frame) / batch.delta_t.unsqueeze(-1).clamp(min=1e-6)

        # Extrapolate
        mean = last_frame + velocity * batch.horizon_dt.unsqueeze(-1)

        var = (self.noise_scale ** 2) * batch.horizon_dt.unsqueeze(-1).expand(B, D)

        return PredictionResult(
            predicted_mean=mean,
            predicted_cov_diag=var,
        )


class GaussianNoisePredictor:
    """Random Gaussian predictions centered on global data mean.

    Useful only for verifying that metrics worsen with random predictions.
    """

    def __init__(self, mean: Optional[Tensor] = None, std: float = 1.0) -> None:
        self._mean = mean  # (D,) or None
        self._std = std

    def predict(self, batch: FragmentBatch) -> PredictionResult:
        B = batch.context.shape[0]
        D = batch.context.shape[2]
        device = batch.context.device

        if self._mean is not None:
            center = self._mean.to(device).unsqueeze(0).expand(B, D)
        else:
            # Use batch mean as fallback
            mask = batch.context_mask.unsqueeze(-1)  # (B, L, 1)
            center = (batch.context * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        mean = center + torch.randn(B, D, device=device) * self._std
        var = torch.full((B, D), self._std ** 2, device=device)

        return PredictionResult(
            predicted_mean=mean,
            predicted_cov_diag=var,
        )
