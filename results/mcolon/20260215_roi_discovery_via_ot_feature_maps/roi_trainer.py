"""
JAX trainer for weight-map regularized logistic regression.

Model:
    params: w_low (learn_res x learn_res x C), b (scalar)
    w_full = bilinear_upsample(w_low -> 512 x 512 x C)
    logits = <X, w_full> + b
    loss = class_weighted_logistic + λ L1(w_low) + μ TV(w_low)

Follows the JAX/Optax pattern from
src/analyze/utils/optimal_transport/backends/ott_backend.py
(optional JAX import, graceful fallback).

See PLAN.md Section D for full specification.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, value_and_grad
    from jax.image import resize as jax_resize
    import optax
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False

from roi_config import TrainerConfig
from roi_tv import build_tv_edges

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """Result from a single training run."""
    w_low: np.ndarray           # (learn_res, learn_res, C)
    w_full: np.ndarray          # (512, 512, C)
    b: float
    channel_names: Tuple[str, ...]
    objective_log: List[Dict]   # per-step objective breakdown
    converged: bool
    n_steps: int
    runtime_sec: float
    config: Dict


def compute_logits(
    X: np.ndarray,
    w_full: np.ndarray,
    b: float,
) -> np.ndarray:
    """Compute linear logits z = <X, w_full> + b for batched feature maps."""
    if X.ndim != 4:
        raise ValueError(f"X must be 4D (N,H,W,C), got shape {X.shape}")
    if w_full.ndim != 3:
        raise ValueError(f"w_full must be 3D (H,W,C), got shape {w_full.shape}")
    if X.shape[1:] != w_full.shape:
        raise ValueError(
            f"Shape mismatch: X has spatial/channels {X.shape[1:]}, w_full has {w_full.shape}"
        )
    return np.sum(X * w_full[None, :, :, :], axis=(1, 2, 3)) + float(b)


def _check_jax():
    if not _JAX_AVAILABLE:
        raise ImportError(
            "JAX + Optax required for ROI trainer. "
            "Install with: pip install jax optax"
        )


def _upsample_bilinear(w_low: "jnp.ndarray", target_hw: Tuple[int, int]) -> "jnp.ndarray":
    """Bilinear upsample w_low to target resolution."""
    # w_low: (H_low, W_low, C) -> (H_out, W_out, C)
    H_out, W_out = target_hw
    C = w_low.shape[-1]
    return jax_resize(w_low, (H_out, W_out, C), method="bilinear")


def _build_objective_fn(
    X: "jnp.ndarray",            # (N, H, W, C) float32
    y: "jnp.ndarray",            # (N,) int
    sample_weights: "jnp.ndarray",  # (N,) float32
    mask_low: "jnp.ndarray",     # (H_low, W_low) bool
    edges_src: "jnp.ndarray",    # (E,)
    edges_tgt: "jnp.ndarray",    # (E,)
    output_hw: Tuple[int, int],
    lam: float,
    mu: float,
):
    """
    Build the JIT-compiled objective function.

    Returns a function f(params) -> (total_loss, aux_dict).
    """
    lam_jnp = jnp.float32(lam)
    mu_jnp = jnp.float32(mu)

    def objective(params):
        w_low = params["w_low"]   # (H_low, W_low, C)
        b = params["b"]           # scalar

        # Upsample to full resolution
        w_full = _upsample_bilinear(w_low, output_hw)  # (H, W, C)

        # Compute logits: <X, w_full> + b
        # X: (N, H, W, C), w_full: (H, W, C) -> dot product over (H, W, C)
        logits = jnp.sum(X * w_full[None, :, :, :], axis=(1, 2, 3)) + b  # (N,)

        # Class-weighted logistic loss
        y_float = y.astype(jnp.float32)
        log_sigmoid = jax.nn.log_sigmoid
        per_sample_loss = -(
            y_float * log_sigmoid(logits)
            + (1.0 - y_float) * log_sigmoid(-logits)
        )
        logistic_loss = jnp.sum(per_sample_loss * sample_weights) / jnp.sum(sample_weights)

        # L1 on w_low (only inside mask)
        w_low_masked = w_low * mask_low[:, :, None]
        l1_raw = jnp.sum(jnp.abs(w_low_masked))

        # TV on w_low (mask-aware edges)
        H_low, W_low, C = w_low.shape
        w_flat = w_low.reshape(H_low * W_low, C)
        diffs = w_flat[edges_src] - w_flat[edges_tgt]  # (E, C)
        tv_raw = jnp.sum(jnp.abs(diffs))

        # Total objective
        l1_weighted = lam_jnp * l1_raw
        tv_weighted = mu_jnp * tv_raw
        total = logistic_loss + l1_weighted + tv_weighted

        aux = {
            "logistic_loss_raw": logistic_loss,
            "l1_raw": l1_raw,
            "tv_raw": tv_raw,
            "l1_weighted": l1_weighted,
            "tv_weighted": tv_weighted,
            "total_objective": total,
        }

        return total, aux

    return objective


def train(
    X: np.ndarray,
    y: np.ndarray,
    mask_ref: np.ndarray,
    class_weights: Dict[int, float],
    lam: float,
    mu: float,
    config: Optional[TrainerConfig] = None,
    channel_names: Optional[Sequence[str]] = None,
) -> TrainResult:
    """
    Train weight-map logistic regression with L1 + TV regularization.

    Parameters
    ----------
    X : ndarray, shape (N, 512, 512, C)
        Feature maps on canonical grid.
    y : ndarray, shape (N,)
        Binary labels (0/1).
    mask_ref : ndarray, shape (512, 512)
        Reference mask on canonical grid.
    class_weights : dict
        {0: weight_for_class_0, 1: weight_for_class_1}
        Computed from training fold via sklearn balanced weights.
    lam : float
        L1 penalty strength.
    mu : float
        TV penalty strength.
    config : TrainerConfig, optional
        Training hyperparameters.

    Returns
    -------
    TrainResult with trained weights and objective log.
    """
    _check_jax()

    config = config or TrainerConfig()
    N, H_out, W_out, C = X.shape
    learn_res = config.learn_res

    logger.info(
        f"Training: N={N}, C={C}, learn_res={learn_res}, "
        f"λ={lam:.2e}, μ={mu:.2e}"
    )

    # Downsample mask to learn_res for L1/TV computation.
    # order=0 (nearest-neighbor) preserves binary mask semantics —
    # bilinear would create fractional values at mask edges.
    from scipy.ndimage import zoom
    mask_low_np = zoom(
        mask_ref.astype(np.float32),
        (learn_res / H_out, learn_res / W_out),
        order=0,
    ) > 0.5
    mask_low_np = mask_low_np.astype(np.float32)

    # Build TV edges at learn_res
    edges_src_np, edges_tgt_np = build_tv_edges(mask_low_np.astype(bool))

    # Compute per-sample weights from class weights
    sample_weights_np = np.array(
        [class_weights.get(int(yi), 1.0) for yi in y],
        dtype=np.float32,
    )

    # Move to JAX
    X_jnp = jnp.array(X, dtype=jnp.float32)
    y_jnp = jnp.array(y, dtype=jnp.int32)
    sample_weights_jnp = jnp.array(sample_weights_np, dtype=jnp.float32)
    mask_low_jnp = jnp.array(mask_low_np, dtype=jnp.float32)
    edges_src_jnp = jnp.array(edges_src_np, dtype=jnp.int32)
    edges_tgt_jnp = jnp.array(edges_tgt_np, dtype=jnp.int32)

    # Build objective
    objective_fn = _build_objective_fn(
        X_jnp, y_jnp, sample_weights_jnp,
        mask_low_jnp, edges_src_jnp, edges_tgt_jnp,
        output_hw=(H_out, W_out),
        lam=lam, mu=mu,
    )

    # Initialize params
    rng = jax.random.PRNGKey(config.random_seed)
    params = {
        "w_low": jnp.zeros((learn_res, learn_res, C), dtype=jnp.float32),
        "b": jnp.float32(0.0),
    }

    # Optimizer
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(params)

    # JIT-compiled train step
    @jit
    def train_step(params, opt_state):
        (loss, aux), grads = value_and_grad(objective_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, aux

    # Training loop
    objective_log = []
    converged = False
    prev_loss = float("inf")
    t_start = time.time()

    for step in range(config.max_steps):
        params, opt_state, loss, aux = train_step(params, opt_state)

        if step % config.log_every == 0 or step == config.max_steps - 1:
            log_entry = {
                "step": step,
                **{k: float(v) for k, v in aux.items()},
            }
            objective_log.append(log_entry)

            if step % (config.log_every * 5) == 0:
                logger.info(
                    f"  step {step:5d}: total={float(loss):.6f}, "
                    f"logistic={float(aux['logistic_loss_raw']):.6f}, "
                    f"L1={float(aux['l1_weighted']):.6f}, "
                    f"TV={float(aux['tv_weighted']):.6f}"
                )

        # Convergence check
        current_loss = float(loss)
        if abs(prev_loss - current_loss) < config.convergence_tol:
            converged = True
            logger.info(f"Converged at step {step} (Δloss < {config.convergence_tol})")
            break
        prev_loss = current_loss

    runtime = time.time() - t_start

    # Extract final weights
    w_low_np = np.array(params["w_low"])
    w_full_np = np.array(
        _upsample_bilinear(params["w_low"], (H_out, W_out))
    )
    b_np = float(params["b"])

    resolved_channel_names = tuple(channel_names) if channel_names is not None else tuple(
        f"channel_{i}" for i in range(C)
    )

    return TrainResult(
        w_low=w_low_np,
        w_full=w_full_np,
        b=b_np,
        channel_names=resolved_channel_names,
        objective_log=objective_log,
        converged=converged,
        n_steps=step + 1,
        runtime_sec=runtime,
        config={
            "learn_res": learn_res,
            "output_res": H_out,
            "lam": lam,
            "mu": mu,
            "learning_rate": config.learning_rate,
            "max_steps": config.max_steps,
            "class_weights": class_weights,
            "channel_names": resolved_channel_names,
        },
    )


def extract_roi(
    w_full: np.ndarray,
    mask_ref: np.ndarray,
    quantile: float = 0.9,
) -> Tuple[np.ndarray, Dict]:
    """
    Extract ROI from trained weight map via quantile thresholding.

    Uses magnitude-based thresholding to identify "active" regions where
    the model places large weights. This shows WHERE the model focuses,
    but not necessarily WHERE discrimination happens (which requires
    considering sign cancellation — see test_p1_05_trainer.py).

    For interpretable ROIs, use moderate-to-strong TV regularization
    (mu >= 1e-3) to suppress oscillating weights. This forces the model
    to learn smooth, same-sign patterns where magnitude ≈ discriminative power.

    Parameters
    ----------
    w_full : ndarray, shape (H, W, C)
        Full-resolution weight map.
    mask_ref : ndarray, shape (H, W)
        Reference mask.
    quantile : float
        Threshold |w| at this quantile (within mask).

    Returns
    -------
    roi_mask : ndarray, shape (H, W), bool
        Binary ROI mask.
    roi_stats : dict
        area_fraction, n_components, boundary_fraction

    See Also
    --------
    tests/test_p1_05_trainer.py : Magnitude vs signed weight distinction
    tests/TESTPLAN.md : Testing principle for discriminative power
    """
    from scipy.ndimage import label as ndimage_label
    from roi_tv import compute_boundary_fraction

    # Magnitude across channels
    w_mag = np.sqrt(np.sum(w_full ** 2, axis=-1))  # (H, W)

    # Only consider pixels inside the reference mask
    mask_bool = mask_ref.astype(bool)
    w_mag_masked = w_mag * mask_bool

    # Threshold at quantile
    values_in_mask = w_mag_masked[mask_bool]
    if len(values_in_mask) == 0:
        roi_mask = np.zeros_like(mask_bool)
        return roi_mask, {"area_fraction": 0.0, "n_components": 0, "boundary_fraction": 0.0}

    threshold = np.quantile(values_in_mask, quantile)
    roi_mask = (w_mag_masked > threshold) & mask_bool

    # Compute stats
    mask_area = mask_bool.sum()
    roi_area = roi_mask.sum()
    area_fraction = float(roi_area) / float(mask_area) if mask_area > 0 else 0.0

    labeled, n_components = ndimage_label(roi_mask)
    boundary_frac = compute_boundary_fraction(roi_mask, mask_ref)

    roi_stats = {
        "area_fraction": area_fraction,
        "n_components": n_components,
        "boundary_fraction": boundary_frac,
        "threshold": float(threshold),
        "quantile": quantile,
    }

    return roi_mask, roi_stats


__all__ = [
    "train",
    "extract_roi",
    "compute_logits",
    "TrainResult",
]
