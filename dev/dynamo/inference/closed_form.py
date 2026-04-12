"""Closed-form solvers for per-embryo inference.

All solvers are batched and differentiable — gradients flow through into
network parameters via torch autograd.

Model spec references: §4.1 (c_e solve), §4.2 (R_e solve).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def solve_loadings(
    residuals: Tensor,
    design_matrix: Tensor,
    dt: Tensor,
    mask: Tensor,
    lambda_c: float = 1.0,
    class_prior: Optional[Tensor] = None,
    D: Optional[float] = None,
) -> Tensor:
    """Closed-form ridge solve for mode loadings c_e (spec §4.1).

    Given observed residuals (displacement minus baseline drift) and a design
    matrix of mode contributions, solves the regularized least-squares system:

        c_e* = (H^T H + lambda_c I)^{-1} (H^T r + lambda_c c_0)

    With optional heteroscedasticity correction (weighting by 1/sqrt(2D*dt)).

    Fully differentiable — gradients flow through design_matrix into potential
    network weights via torch.linalg.solve.

    Args:
        residuals: (B, T, d) observed displacement minus baseline drift.
        design_matrix: (B, T, M, d) mode contributions per transition per mode.
            H[b, t, m, :] = R_e * S_m @ grad_phi0(z_t) * dt_t.
        dt: (B, T) time step per transition.
        mask: (B, T) boolean mask for valid transitions.
        lambda_c: L2 regularization strength on c_e.
        class_prior: (B, M) per-embryo class-level prior c_{0,p(e)}.
            If None, defaults to zero (standard ridge).
        D: Global diffusion coefficient for heteroscedasticity correction.
            If None, no correction applied.

    Returns:
        (B, M) inferred mode loadings c_e per embryo.
    """
    B, T, M, d = design_matrix.shape
    mask_f = mask.float()  # (B, T)

    # Flatten spatial dimensions: H -> (B, T*d, M), r -> (B, T*d)
    H = design_matrix.permute(0, 1, 3, 2).reshape(B, T * d, M)  # (B, T*d, M)
    r = residuals.reshape(B, T * d)  # (B, T*d)

    # Build per-element weight mask: repeat mask across d dimensions
    w = mask_f.unsqueeze(-1).expand(B, T, d).reshape(B, T * d)  # (B, T*d)

    # Heteroscedasticity correction: weight by 1/sqrt(2D*dt)
    if D is not None and D > 0:
        dt_rep = dt.unsqueeze(-1).expand(B, T, d).reshape(B, T * d)  # (B, T*d)
        hetero_weight = 1.0 / (2.0 * D * dt_rep).clamp(min=1e-10).sqrt()
        w = w * hetero_weight

    # Apply weights
    H_w = H * w.unsqueeze(-1)  # (B, T*d, M)
    r_w = r * w                # (B, T*d)

    # Normal equations: (H^T H + lambda_c I) c = H^T r + lambda_c c_0
    HtH = torch.bmm(H_w.transpose(1, 2), H_w)  # (B, M, M)
    Htr = torch.bmm(H_w.transpose(1, 2), r_w.unsqueeze(-1)).squeeze(-1)  # (B, M)

    # Regularization
    reg = lambda_c * torch.eye(M, device=HtH.device, dtype=HtH.dtype).unsqueeze(0)
    A = HtH + reg  # (B, M, M)

    # Add diagonal jitter for numerical stability (spec: §15.3)
    A = A + 1e-6 * torch.eye(M, device=A.device, dtype=A.dtype).unsqueeze(0)

    rhs = Htr  # (B, M)
    if class_prior is not None:
        rhs = rhs + lambda_c * class_prior  # (B, M)

    # Solve via Cholesky (symmetric positive definite)
    c_e = torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)  # (B, M)

    return c_e


def solve_rate(
    displacements: Tensor,
    drift_direction: Tensor,
    dt: Tensor,
    mask: Tensor,
    clamp_min: float = 1e-6,
) -> Tensor:
    """Closed-form R_e solve via scalar projection (spec §4.2).

    Given observed displacements and predicted drift directions (before R_e
    scaling), finds the optimal scalar R_e that projects displacements onto
    the drift:

        R_e* = sum_t(delta_t^T f_hat_t dt) / sum_t(||f_hat_t||^2 dt^2)

    Fully differentiable — gradients flow through drift_direction into the
    potential network weights.

    Args:
        displacements: (B, T, d) observed z_{t+1} - z_t.
        drift_direction: (B, T, d) predicted drift (before R_e scaling).
        dt: (B, T) time step per transition.
        mask: (B, T) boolean mask for valid transitions.
        clamp_min: Floor on R_e to prevent division issues.

    Returns:
        (B,) inferred R_e per embryo.
    """
    mask_f = mask.float()

    # Dot product per transition: sum over d dimension
    dot = (displacements * drift_direction).sum(dim=-1)  # (B, T)

    # Squared norm of drift direction per transition
    f_sq = (drift_direction ** 2).sum(dim=-1)  # (B, T)

    # Weighted sums over time dimension
    numerator = (dot * dt * mask_f).sum(dim=1)        # (B,)
    denominator = (f_sq * dt ** 2 * mask_f).sum(dim=1)  # (B,)

    # Scalar projection with numerical safety
    R_e = numerator / denominator.clamp(min=1e-10)
    R_e = R_e.clamp(min=clamp_min)

    return R_e
