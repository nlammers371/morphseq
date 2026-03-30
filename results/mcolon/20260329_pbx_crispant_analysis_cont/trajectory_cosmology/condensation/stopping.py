"""
stopping.py
-----------
Stopping metrics and heuristics for the condensation dynamical system.

Philosophy: instrument broadly, stop conservatively, learn empirically.
Do not hard-code one universal threshold. Track several candidate signals,
normalize by a fixed reference scale, and observe what stable convergence
actually looks like in this system before committing to a rule.

Three responsibilities:
  1. compute_*  — pure metric functions (arrays in, scalars out)
  2. log_metrics — assemble one row of diagnostics per iteration
  3. evaluate_stopping_heuristics — multi-metric patience-based decision

All displacement metrics are normalized by a reference spatial scale
computed once from the initial positions. This makes thresholds portable
across AlignedUMAP, PCA, and any other 2D initialization.

Reference scale:
  s_x = RMS distance of all valid initial positions from the global mean
  Computed once via reference_scale_from_positions(x0, mask).
  Does not change during the run.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Reference scale
# ---------------------------------------------------------------------------

def reference_scale_from_positions(
    x0: np.ndarray,
    mask: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """Compute RMS radius of the initial valid point cloud.

    s_x = sqrt( mean_{i,t: mask} ||x0[i,t] - mu||^2 )

    Parameters
    ----------
    x0 : (N_e, T, 2)
    mask : (N_e, T) bool

    Returns
    -------
    s_x : float — the reference spatial scale for this run
    """
    valid = x0[mask]          # (n_obs, 2)
    if len(valid) == 0:
        return 1.0
    mu = valid.mean(axis=0)   # (2,)
    sq_dist = ((valid - mu) ** 2).sum(axis=1)
    return float(np.sqrt(sq_dist.mean()) + eps)


# ---------------------------------------------------------------------------
# Per-iteration metrics
# ---------------------------------------------------------------------------

def displacement_metrics(
    x_prev: np.ndarray,
    x_curr: np.ndarray,
    mask: np.ndarray,
    reference_scale: float,
) -> dict[str, float]:
    """Displacement between consecutive position arrays, normalized by scale.

    Parameters
    ----------
    x_prev, x_curr : (N_e, T, 2)
    mask : (N_e, T) bool
    reference_scale : float — from reference_scale_from_positions()

    Returns
    -------
    dict with keys:
      disp_max_abs   — max absolute displacement (raw)
      disp_max_rel   — max displacement / reference_scale
      disp_rms_abs   — RMS displacement (raw)
      disp_rms_rel   — RMS displacement / reference_scale
    """
    delta = x_curr - x_prev                       # (N_e, T, 2)
    dist = np.linalg.norm(delta, axis=-1)         # (N_e, T)
    valid_dist = dist[mask]

    if len(valid_dist) == 0:
        return dict(disp_max_abs=0.0, disp_max_rel=0.0,
                    disp_rms_abs=0.0, disp_rms_rel=0.0)

    max_abs = float(valid_dist.max())
    rms_abs = float(np.sqrt((valid_dist ** 2).mean()))

    return dict(
        disp_max_abs=max_abs,
        disp_max_rel=max_abs / reference_scale,
        disp_rms_abs=rms_abs,
        disp_rms_rel=rms_abs / reference_scale,
    )


def relative_energy_change(
    prev_energy: float,
    curr_energy: float,
    eps: float = 1e-8,
) -> float:
    """Relative change in total energy between iterations.

    |E_curr - E_prev| / (|E_prev| + eps)

    Treat as supportive, not sovereign — coherence recomputation means
    energy is not a clean monotone signal.
    """
    return abs(curr_energy - prev_energy) / (abs(prev_energy) + eps)


def coherence_change_metric(
    C_prev: np.ndarray,
    C_curr: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """Relative Frobenius change in the coherence field across all time bins.

    ||C_curr - C_prev||_F / (||C_prev||_F + eps)

    This measures whether "who travels with whom" is still changing.
    When this is small, the memory field has stabilized.

    Parameters
    ----------
    C_prev, C_curr : (N_e, N_e, T)
    """
    diff_norm = float(np.linalg.norm(C_curr - C_prev))
    prev_norm = float(np.linalg.norm(C_prev))
    return diff_norm / (prev_norm + eps)


# ---------------------------------------------------------------------------
# Per-iteration log assembly
# ---------------------------------------------------------------------------

def log_metrics(
    iteration: int,
    x_prev: np.ndarray,
    x_curr: np.ndarray,
    mask: np.ndarray,
    reference_scale: float,
    energy_terms: dict[str, float],
    prev_total_energy: float | None = None,
    C_prev: np.ndarray | None = None,
    C_curr: np.ndarray | None = None,
) -> dict:
    """Assemble one diagnostics row for the iteration log.

    Returns a flat dict suitable for appending to a list and converting to
    a DataFrame at the end of a run.

    Columns produced:
      iter, spatial_scale_ref,
      energy_total, energy_attract, energy_repel, energy_elastic, energy_fidelity,
      energy_change_rel,
      disp_max_abs, disp_max_rel, disp_rms_abs, disp_rms_rel,
      coherence_change_rel  (if C_prev/C_curr provided)
    """
    row: dict = {"iter": iteration, "spatial_scale_ref": reference_scale}

    # energy terms
    row["energy_total"] = energy_terms.get("total", float("nan"))
    for key in ("attract", "repel", "elastic", "fidelity"):
        row[f"energy_{key}"] = energy_terms.get(key, float("nan"))

    # relative energy change
    if prev_total_energy is not None:
        row["energy_change_rel"] = relative_energy_change(
            prev_total_energy, row["energy_total"]
        )
    else:
        row["energy_change_rel"] = float("nan")

    # displacement
    row.update(displacement_metrics(x_prev, x_curr, mask, reference_scale))

    # coherence drift
    if C_prev is not None and C_curr is not None:
        row["coherence_change_rel"] = coherence_change_metric(C_prev, C_curr)
    else:
        row["coherence_change_rel"] = float("nan")

    return row


# ---------------------------------------------------------------------------
# Stopping heuristic evaluator
# ---------------------------------------------------------------------------

@dataclass
class StoppingConfig:
    """Thresholds for the multi-metric patience-based stopping heuristic.

    All displacement thresholds are relative (normalized by reference_scale).
    Start with loose values and tighten after observing convergence behavior.

    Attributes
    ----------
    patience : int
        Number of consecutive iterations all active criteria must be satisfied.
    disp_rms_rel_threshold : float or None
        Stop signal if relative RMS displacement < threshold. None = inactive.
    disp_max_rel_threshold : float or None
        Stop signal if relative max displacement < threshold. None = inactive.
    energy_change_rel_threshold : float or None
        Stop signal if relative energy change < threshold. None = inactive.
    coherence_change_rel_threshold : float or None
        Stop signal if relative coherence change < threshold. None = inactive.
    require_all : bool
        If True, ALL active criteria must be satisfied simultaneously for
        patience counting. If False, ANY single criterion suffices.
        Default True (conservative).
    """
    patience: int = 10
    disp_rms_rel_threshold: float | None = 1e-3
    disp_max_rel_threshold: float | None = None    # inactive by default
    energy_change_rel_threshold: float | None = 1e-4
    coherence_change_rel_threshold: float | None = None   # inactive by default
    require_all: bool = True


class StoppingMonitor:
    """Stateful monitor that evaluates stopping heuristics across iterations.

    Usage
    -----
    monitor = StoppingMonitor(config)
    ...
    for iteration in loop:
        row = log_metrics(...)
        stop, reason = monitor.update(row)
        if stop:
            break
    """

    def __init__(self, config: StoppingConfig):
        self.config = config
        self._satisfied_streak = 0

    def update(self, metrics_row: dict) -> tuple[bool, str]:
        """Evaluate stopping heuristics for one iteration.

        Parameters
        ----------
        metrics_row : dict — output of log_metrics()

        Returns
        -------
        (should_stop, reason_string)
        """
        cfg = self.config
        signals: list[bool] = []
        active_names: list[str] = []

        def _check(key: str, threshold: float | None) -> None:
            if threshold is None:
                return
            val = metrics_row.get(key, float("nan"))
            active_names.append(key)
            signals.append(not np.isnan(val) and val < threshold)

        _check("disp_rms_rel", cfg.disp_rms_rel_threshold)
        _check("disp_max_rel", cfg.disp_max_rel_threshold)
        _check("energy_change_rel", cfg.energy_change_rel_threshold)
        _check("coherence_change_rel", cfg.coherence_change_rel_threshold)

        if not signals:
            return False, "no active criteria"

        satisfied = all(signals) if cfg.require_all else any(signals)

        if satisfied:
            self._satisfied_streak += 1
        else:
            self._satisfied_streak = 0

        if self._satisfied_streak >= cfg.patience:
            reason = (
                f"all criteria satisfied for {cfg.patience} consecutive iterations "
                f"({', '.join(active_names)})"
            )
            return True, reason

        return False, f"streak={self._satisfied_streak}/{cfg.patience}"
