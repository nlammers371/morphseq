"""Phase 2.5a trainer: learn a global soft ROI mask with a fixed classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from roi_mask_objective import dual_objective
from roi_mask_param import jitter_mask, mask_from_param, tv_loss, upsample_mask


@dataclass
class MaskTrainResult:
    mask_low: np.ndarray
    mask_full: np.ndarray
    objective_log: List[Dict[str, float]]


def train_mask_fixed_model(
    X: np.ndarray,
    w_full: np.ndarray,
    b: float,
    baseline: np.ndarray,
    mask_ref: np.ndarray,
    learn_res: int = 128,
    n_steps: int = 500,
    learning_rate: float = 1e-2,
    l1_weight: float = 1e-3,
    tv_weight: float = 1e-3,
    entropy_weight: float = 0.0,
    temperature: float = 1.0,
    jitter_px: int = 2,
    random_seed: int = 42,
) -> MaskTrainResult:
    """Optimize a global mask m for a fixed (w,b) classifier using the dual objective."""
    X_j = jnp.array(X, dtype=jnp.float32)
    w_j = jnp.array(w_full, dtype=jnp.float32)
    baseline_j = jnp.array(baseline, dtype=jnp.float32)

    mask_ref_low = jax.image.resize(
        jnp.array(mask_ref, dtype=jnp.float32)[:, :, None],
        (learn_res, learn_res, 1),
        method="nearest",
    )[:, :, 0]

    params = {"mask_param": jnp.zeros((learn_res, learn_res), dtype=jnp.float32)}
    opt = optax.adam(learning_rate)
    state = opt.init(params)
    rng = jax.random.PRNGKey(random_seed)

    @jax.jit
    def step(params, state, rng):
        rng, shift_rng = jax.random.split(rng)
        shift = jax.random.randint(shift_rng, (2,), minval=-jitter_px, maxval=jitter_px + 1)

        def loss_fn(p):
            m_low = mask_from_param(p["mask_param"], temperature=temperature) * mask_ref_low
            m_full = upsample_mask(m_low, w_j.shape[:2]) * jnp.array(mask_ref, dtype=jnp.float32)
            m_full = jitter_mask(m_full, shift[0], shift[1])

            dual = dual_objective(X_j, m_full, baseline_j, w_j, b)
            l1 = jnp.mean(jnp.abs(m_low))
            tv = tv_loss(m_low) / jnp.float32(m_low.size)
            entropy = jnp.mean(m_low * (1.0 - m_low))
            total = -(dual - l1_weight * l1 - tv_weight * tv - entropy_weight * entropy)
            return total, {"dual": dual, "l1": l1, "tv": tv, "entropy": entropy}

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, state = opt.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        return params, state, rng, loss, aux

    logs: List[Dict[str, float]] = []
    for step_i in range(n_steps):
        params, state, rng, loss, aux = step(params, state, rng)
        if step_i % 25 == 0 or step_i == n_steps - 1:
            logs.append({"step": float(step_i), "loss": float(loss), **{k: float(v) for k, v in aux.items()}})

    mask_low = np.array(mask_from_param(params["mask_param"], temperature=temperature) * mask_ref_low)
    mask_full = np.array(upsample_mask(jnp.array(mask_low), w_full.shape[:2]) * jnp.array(mask_ref, dtype=jnp.float32))

    return MaskTrainResult(mask_low=mask_low, mask_full=mask_full, objective_log=logs)
