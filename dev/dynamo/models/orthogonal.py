"""Stage 2 model: orthogonal modes applied to baseline potential (spec §3.4).

Modes are antisymmetric matrices S_m applied to ∇φ₀, producing drift fields
orthogonal to the baseline gradient. The total drift is:

    f(z; c_e, R_e) = R_e * [-βI + Σ_m c_{e,m} S_m] ∇φ₀(z)

Key properties:
- Exact tempo-mode separation: modes cannot affect developmental tempo
- No separate mode networks: each mode is d(d-1)/2 parameters
- Frobenius normalization ||S_m||_F = 1 for interpretable c-space distances
- Closed-form c_e solve remains linear (design matrix columns: S_m ∇φ₀ Δt)

Model spec references: §3.4 (orthogonal modes), §4.1 (c_e solve),
    §4.2 (R_e solve), §5.2 (S_m normalization), §7.1 (staged training).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ..data.dataset import FragmentBatch
from ..eval.predictions import PredictionResult
from ..inference.closed_form import solve_loadings, solve_rate
from .potential import PotentialNetwork


class AntisymmetricMatrix(nn.Module):
    """Learnable antisymmetric matrix with Frobenius normalization.

    Parameterized by d(d-1)/2 unconstrained upper-triangular entries.
    The matrix S is constructed as S = (U - U^T) / ||U - U^T||_F, ensuring
    ||S||_F = 1 at all times.

    Args:
        dim: Matrix dimension d.
        init_scale: Scale of random initialization for unconstrained entries.
    """

    def __init__(self, dim: int, init_scale: float = 0.01) -> None:
        super().__init__()
        n_entries = dim * (dim - 1) // 2
        self.dim = dim
        # Initialize at small random values (NOT zero — normalization would divide by zero)
        self.raw_entries = nn.Parameter(
            torch.randn(n_entries) * init_scale
        )
        # Pre-compute upper-triangular indices
        rows, cols = torch.triu_indices(dim, dim, offset=1)
        self.register_buffer("_rows", rows)
        self.register_buffer("_cols", cols)

    def forward(self) -> Tensor:
        """Return the Frobenius-normalized antisymmetric matrix (d, d)."""
        S_raw = torch.zeros(self.dim, self.dim,
                            device=self.raw_entries.device,
                            dtype=self.raw_entries.dtype)
        S_raw[self._rows, self._cols] = self.raw_entries
        S_raw = S_raw - S_raw.T  # Antisymmetrize
        # Frobenius normalization
        norm = S_raw.norm(p="fro").clamp(min=1e-8)
        return S_raw / norm


class OrthogonalModesModel(nn.Module):
    """Stage 2 model: orthogonal modes on frozen baseline potential.

    Drift: f(z; c_e, R_e) = R_e * [-βI + Σ_m c_{e,m} S_m] ∇φ₀(z)

    φ₀ is loaded from a Stage 1 checkpoint and frozen. Learnable parameters
    are S_m entries, class-level priors c_{0,p}, D, and optionally β_T.

    Implements the Predictor protocol for the eval pipeline.

    Args:
        phi0: Pre-trained PotentialNetwork (will be frozen).
        input_dim: Latent space dimension d.
        n_modes: Number of orthogonal modes M.
        n_classes: Number of perturbation classes.
        init_log_beta: Initial log(β) from Stage 1 checkpoint.
        init_log_D: Initial log(D) from Stage 1 checkpoint.
        lambda_c: L2 regularization on mode loadings.
        n_forward_samples: Euler-Maruyama samples for predict().
        rate_clamp_min: Floor on R_e.
        n_alternations: Number of c_e/R_e alternating solve iterations.
        s_init_scale: Scale of random initialization for S_m entries.
        normalize_rate: Whether to apply batch-level R_e normalization
            (rate identifiability constraint, spec §6.6).
        log_beta_T: Initial log(β_T) for temperature dependence.
            None = no temperature correction.
        T_ref: Reference temperature for Arrhenius term (°C).
        alpha_0: Hessian smoothness penalty weight (carried forward from Stage 1
            for monitoring, but phi0 is frozen so R0 doesn't affect training).
        hessian_n_points: Points subsampled for R0 computation.
    """

    def __init__(
        self,
        phi0: PotentialNetwork,
        input_dim: int,
        n_modes: int = 5,
        n_classes: int = 1,
        init_log_beta: float = 0.0,
        init_log_D: float = -2.0,
        lambda_c: float = 1.0,
        n_forward_samples: int = 50,
        rate_clamp_min: float = 1e-6,
        n_alternations: int = 2,
        s_init_scale: float = 0.01,
        normalize_rate: bool = True,
        log_beta_T: Optional[float] = None,
        T_ref: float = 28.5,
        alpha_0: float = 0.01,
        hessian_n_points: int = 64,
    ) -> None:
        super().__init__()

        # Frozen baseline potential
        self.phi0 = phi0
        for p in self.phi0.parameters():
            p.requires_grad_(False)

        self.input_dim = input_dim
        self.n_modes = n_modes
        self.n_classes = n_classes
        self.lambda_c = lambda_c
        self.n_forward_samples = n_forward_samples
        self.rate_clamp_min = rate_clamp_min
        self.n_alternations = n_alternations
        self.normalize_rate = normalize_rate
        self.alpha_0 = alpha_0
        self.hessian_n_points = hessian_n_points

        # Global parameters (β frozen from Stage 1, D continues training)
        self.log_beta = nn.Parameter(
            torch.tensor(init_log_beta, dtype=torch.float32)
        )
        self.log_beta.requires_grad_(False)  # Freeze β with φ₀
        self.log_D = nn.Parameter(
            torch.tensor(init_log_D, dtype=torch.float32)
        )

        # Antisymmetric mode matrices
        self.S_modules = nn.ModuleList([
            AntisymmetricMatrix(input_dim, init_scale=s_init_scale)
            for _ in range(n_modes)
        ])

        # Class-level priors c_{0,p} (learnable, spec §4.3)
        self.class_priors = nn.Parameter(
            torch.zeros(n_classes, n_modes)
        )

        # Temperature dependence (optional, spec §3.5)
        self.T_ref = T_ref
        if log_beta_T is not None:
            self.log_beta_T = nn.Parameter(
                torch.tensor(log_beta_T, dtype=torch.float32)
            )
        else:
            self.log_beta_T = None

    @property
    def beta(self) -> Tensor:
        return self.log_beta.exp()

    @property
    def D(self) -> Tensor:
        return self.log_D.exp()

    @property
    def beta_T(self) -> Optional[Tensor]:
        if self.log_beta_T is not None:
            return self.log_beta_T.exp()
        return None

    def get_S_matrices(self) -> Tensor:
        """Return stacked normalized antisymmetric matrices (M, d, d)."""
        return torch.stack([s_mod() for s_mod in self.S_modules], dim=0)

    def _temperature_factor(self, temperature: Tensor) -> Tensor:
        """Compute Arrhenius temperature correction factor (B,).

        R_e = λ_e * exp(-β_T * (T_ref - T_e))

        Args:
            temperature: (B,) incubation temperatures in °C.

        Returns:
            (B,) multiplicative correction factor. 1.0 if no temp dependence.
        """
        if self.beta_T is None:
            return torch.ones_like(temperature)
        # Replace NaN temperatures with T_ref (no correction)
        temp = torch.where(
            torch.isnan(temperature), torch.full_like(temperature, self.T_ref),
            temperature,
        )
        return torch.exp(-self.beta_T * (self.T_ref - temp))

    def _compute_mode_design_matrix(
        self,
        grad_phi0: Tensor,
        R_e: Tensor,
        dt: Tensor,
        S_matrices: Tensor,
    ) -> Tensor:
        """Build the mode design matrix H for the c_e solve.

        H[b, t, m, :] = R_e[b] * S_m @ grad_phi0[b,t,:] * dt[b,t]

        Args:
            grad_phi0: (B, T, d) gradient of phi0 at context departure points.
            R_e: (B,) current rate estimates.
            dt: (B, T) time steps per transition.
            S_matrices: (M, d, d) normalized antisymmetric matrices.

        Returns:
            (B, T, M, d) design matrix.
        """
        B, T, d = grad_phi0.shape
        M = S_matrices.shape[0]

        # S_m @ grad_phi0: (M, d, d) x (B, T, d) -> (B, T, M, d)
        # Reshape for batch matmul: grad (B*T, d, 1), S (M, d, d)
        g = grad_phi0.reshape(B * T, d, 1)  # (B*T, d, 1)
        # Broadcast: (M, d, d) @ (B*T, 1, d, 1) -> need einsum
        # H_{b,t,m,i} = S_m[i,j] * grad[b,t,j]
        H = torch.einsum("mij,btj->btmi", S_matrices, grad_phi0)  # (B, T, M, d)

        # Scale by R_e and dt
        H = H * R_e[:, None, None, None] * dt[:, :, None, None]

        return H

    def _infer_parameters(
        self,
        batch: FragmentBatch,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Infer R_e and c_e from context via alternating closed-form solves.

        Returns:
            z_last: (B, d) last valid context frame.
            R_e: (B,) inferred rate.
            c_e: (B, M) inferred mode loadings.
            grad_phi0_ctx: (B, T, d) gradients at context departure points.
        """
        B, L_max, d = batch.context.shape

        # Context transition pairs
        z_from = batch.context[:, :-1, :]  # (B, L_max-1, d)
        z_to = batch.context[:, 1:, :]     # (B, L_max-1, d)
        displacements = z_to - z_from      # (B, L_max-1, d)
        dt_ctx = batch.time_deltas         # (B, L_max-1)

        # Valid transition mask
        trans_mask = batch.context_mask[:, :-1] & batch.context_mask[:, 1:]

        # Gradients of phi0 at departure points
        z_flat = z_from.reshape(-1, d)
        grad_flat = self.phi0.gradient(z_flat)
        grad_phi0 = grad_flat.reshape(B, L_max - 1, d)

        # S matrices
        S_matrices = self.get_S_matrices()  # (M, d, d)

        # Initial R_e from baseline-only solve (no modes)
        f_hat_baseline = -self.beta * grad_phi0
        R_e = solve_rate(displacements, f_hat_baseline, dt_ctx, trans_mask,
                         clamp_min=self.rate_clamp_min)

        # Apply temperature correction
        temp_factor = self._temperature_factor(batch.temperature)
        R_e = R_e / temp_factor.clamp(min=1e-8)  # Remove temp effect to get λ_e

        # Rate identifiability: normalize so mean(λ_e) = 1
        if self.normalize_rate and self.training:
            R_e = R_e / R_e.mean().clamp(min=1e-8)

        # Re-apply temperature correction
        R_e = R_e * temp_factor

        # Class priors for this batch
        class_prior = self.class_priors[batch.class_idx.clamp(min=0)]  # (B, M)

        # Alternating c_e / R_e solve
        c_e = torch.zeros(B, self.n_modes, device=batch.context.device)
        for _ in range(self.n_alternations):
            # Build design matrix
            H = self._compute_mode_design_matrix(
                grad_phi0, R_e, dt_ctx, S_matrices
            )

            # Baseline contribution
            baseline = R_e[:, None, None] * (-self.beta * grad_phi0) * dt_ctx.unsqueeze(-1)
            residuals = displacements - baseline  # (B, T, d)

            # Solve for c_e
            c_e = solve_loadings(
                residuals, H, dt_ctx, trans_mask,
                lambda_c=self.lambda_c,
                class_prior=class_prior,
                D=self.D.item(),
            )

            # Recompute total drift direction for R_e solve
            # f_hat = [-βI + Σ c_m S_m] @ ∇φ₀
            mode_contrib = torch.einsum("bm,mij->bij", c_e, S_matrices)  # (B, d, d)
            drift_matrix = -self.beta * torch.eye(d, device=batch.context.device).unsqueeze(0) + mode_contrib
            f_hat_total = torch.einsum("bij,btj->bti", drift_matrix, grad_phi0)

            R_e = solve_rate(displacements, f_hat_total, dt_ctx, trans_mask,
                             clamp_min=self.rate_clamp_min)

            # Re-apply temperature + identifiability
            temp_factor = self._temperature_factor(batch.temperature)
            R_e_lambda = R_e / temp_factor.clamp(min=1e-8)
            if self.normalize_rate and self.training:
                R_e_lambda = R_e_lambda / R_e_lambda.mean().clamp(min=1e-8)
            R_e = R_e_lambda * temp_factor

        # Last valid context frame
        lengths = batch.context_mask.sum(dim=1).long()
        last_idx = (lengths - 1).clamp(min=0)
        z_last = batch.context[torch.arange(B, device=batch.context.device), last_idx]

        return z_last, R_e, c_e, grad_phi0

    def compute_drift(
        self,
        z: Tensor,
        R_e: Tensor,
        c_e: Tensor,
    ) -> Tensor:
        """Compute full drift at given points.

        f(z) = R_e * [-βI + Σ c_m S_m] ∇φ₀(z)

        Args:
            z: (B, d) or (B, N, d) input points.
            R_e: (B,) rate parameters.
            c_e: (B, M) mode loadings.

        Returns:
            Same shape as z: drift vectors.
        """
        squeeze = False
        if z.dim() == 2:
            z = z.unsqueeze(1)
            squeeze = True

        B, N, d = z.shape
        z_flat = z.reshape(B * N, d)
        grad_phi0 = self.phi0.gradient(z_flat).reshape(B, N, d)

        S_matrices = self.get_S_matrices()
        mode_contrib = torch.einsum("bm,mij->bij", c_e, S_matrices)
        drift_matrix = -self.beta * torch.eye(d, device=z.device).unsqueeze(0) + mode_contrib
        f = torch.einsum("bij,bnj->bni", drift_matrix, grad_phi0)
        f = R_e[:, None, None] * f

        if squeeze:
            f = f.squeeze(1)
        return f

    def forward(self, batch: FragmentBatch) -> Dict[str, Tensor]:
        """Training forward pass: teacher-forced multi-target NLL loss.

        Infers R_e and c_e from context (alternating closed-form), then scores
        each of M target transitions from observed predecessors.

        Args:
            batch: FragmentBatch from the data pipeline.

        Returns:
            Dict with keys: loss, nll, R_e, c_e, beta, D, S_norms,
                hessian_penalty, c_prior_loss.
        """
        B, _, d = batch.context.shape
        M_targets = batch.targets.shape[1]

        z_last, R_e, c_e, _ = self._infer_parameters(batch)

        # Score targets from their observed predecessors (teacher forcing)
        predecessors = batch.predecessors          # (B, M_targets, d)
        targets = batch.targets                    # (B, M_targets, d)
        horizon_dts = batch.horizon_dts            # (B, M_targets)

        # Compute drift at each predecessor
        pred_flat = predecessors.reshape(B * M_targets, d)
        grad_phi0_pred = self.phi0.gradient(pred_flat).reshape(B, M_targets, d)

        S_matrices = self.get_S_matrices()
        mode_contrib = torch.einsum("bm,mij->bij", c_e, S_matrices)
        drift_matrix = -self.beta * torch.eye(d, device=batch.context.device).unsqueeze(0) + mode_contrib
        drift_at_pred = torch.einsum("bij,bmj->bmi", drift_matrix, grad_phi0_pred)
        drift_at_pred = R_e[:, None, None] * drift_at_pred

        predicted = predecessors + drift_at_pred * horizon_dts.unsqueeze(-1)

        # Single-step NLL (spec §7.3)
        diff = targets - predicted
        D = self.D
        four_D_dt = 4.0 * D * horizon_dts
        nll_per_target = (
            0.5 * d * torch.log(math.pi * four_D_dt.clamp(min=1e-10))
            + (diff ** 2).sum(dim=-1) / four_D_dt.clamp(min=1e-10)
        )

        nll = nll_per_target.mean(dim=1)  # (B,)
        nll_mean = nll.mean()

        # Hessian penalty (monitoring only — phi0 is frozen)
        z_all = batch.context[batch.context_mask]
        n_pts = min(self.hessian_n_points, z_all.shape[0])
        idx = torch.randperm(z_all.shape[0], device=z_all.device)[:n_pts]
        r0 = self.phi0.hessian_penalty(z_all[idx])

        # Class prior regularization: encourage c_e to stay near class prior
        class_prior = self.class_priors[batch.class_idx.clamp(min=0)]
        c_prior_loss = ((c_e - class_prior) ** 2).mean()

        loss = nll_mean

        return {
            "loss": loss,
            "nll": nll.detach(),
            "hessian_penalty": r0.detach(),
            "R_e": R_e.detach(),
            "c_e": c_e.detach(),
            "beta": self.beta.detach(),
            "D": D.detach(),
            "mean_c_norm": c_e.norm(dim=-1).mean().detach(),
            "c_prior_loss": c_prior_loss.detach(),
        }

    @torch.no_grad()
    def predict(self, batch: FragmentBatch) -> PredictionResult:
        """Prediction for eval pipeline (Predictor protocol).

        Infers R_e and c_e from context, then produces predicted mean,
        diagonal covariance, and forward samples.

        Args:
            batch: FragmentBatch from the data pipeline.

        Returns:
            PredictionResult with mode_loadings populated.
        """
        B, _, d = batch.context.shape
        z_last, R_e, c_e, _ = self._infer_parameters(batch)

        # Drift at last context frame
        drift = self.compute_drift(z_last, R_e, c_e)  # (B, d)
        horizon_dt = batch.horizon_dts[:, 0]

        # Mean prediction: single Euler step
        predicted_mean = z_last + drift * horizon_dt.unsqueeze(-1)

        # Diagonal covariance: 2D * dt
        var = 2.0 * self.D * horizon_dt.unsqueeze(-1)
        predicted_cov_diag = var.expand(B, d)

        # Forward samples
        noise_std = (2.0 * self.D * horizon_dt).sqrt()
        noise = torch.randn(B, self.n_forward_samples, d,
                            device=batch.context.device)
        forward_samples = (
            predicted_mean.unsqueeze(1)
            + noise * noise_std.unsqueeze(-1).unsqueeze(-1)
        )

        return PredictionResult(
            predicted_mean=predicted_mean,
            predicted_cov_diag=predicted_cov_diag,
            forward_samples=forward_samples,
            rate=R_e,
            diffusion_D=self.D.item(),
            mode_loadings=c_e,
        )
