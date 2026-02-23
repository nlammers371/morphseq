"""Perturbation primitives and fold-safe baselines for Phase 2 occlusion validation."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def compute_spatial_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    channel_names: Sequence[str],
    wt_label: int = 0,
    baseline_policy: Optional[Dict[str, str]] = None,
) -> np.ndarray:
    """Compute per-pixel baseline B(H,W,C) using train/inbag samples only."""
    if X_train.ndim != 4:
        raise ValueError(f"X_train must be 4D (N,H,W,C), got shape={X_train.shape}")

    baseline_policy = baseline_policy or {name: "spatial" for name in channel_names}
    wt_mask = y_train == wt_label
    if not np.any(wt_mask):
        raise ValueError("No WT samples available in training data for baseline estimation")

    X_wt = X_train[wt_mask]
    baseline = np.zeros(X_train.shape[1:], dtype=np.float32)

    for c, channel_name in enumerate(channel_names):
        policy = baseline_policy.get(channel_name, "spatial")
        if policy != "spatial":
            raise ValueError(f"Unsupported baseline policy '{policy}' for channel '{channel_name}'")
        baseline[:, :, c] = X_wt[:, :, :, c].mean(axis=0)

    return baseline


def apply_perturbation(X: np.ndarray, mask: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Apply preserve-style perturbation P(X,m)=m*X + (1-m)*B."""
    if X.ndim != 4:
        raise ValueError(f"X must be 4D (N,H,W,C), got {X.shape}")
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D (H,W), got {mask.shape}")
    if baseline.ndim != 3:
        raise ValueError(f"baseline must be 3D (H,W,C), got {baseline.shape}")

    if X.shape[1:3] != mask.shape:
        raise ValueError(f"mask shape mismatch: X has {X.shape[1:3]}, mask has {mask.shape}")
    if X.shape[1:] != baseline.shape:
        raise ValueError(f"baseline shape mismatch: X has {X.shape[1:]}, baseline has {baseline.shape}")

    m = mask.astype(np.float32)
    return m[None, :, :, None] * X + (1.0 - m[None, :, :, None]) * baseline[None, :, :, :]


__all__ = ["compute_spatial_baseline", "apply_perturbation"]
