"""
diagnostics.py
--------------
Computes interpretable quantities from a CondensationResult.

No plotting here — all outputs are DataFrames or arrays.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .condensation.state import CondensationResult


def bundle_centroids(
    result: CondensationResult,
    labels: np.ndarray,
    time_values: np.ndarray,
) -> pd.DataFrame:
    """Mean condensed position per condition per time bin.

    Parameters
    ----------
    result : CondensationResult
    labels : (N_e,) — genotype label per embryo
    time_values : (T,) — time bin centers in hpf

    Returns
    -------
    DataFrame with columns: condition, time_bin_center, x, y, n_embryos
    """
    positions = result.positions   # (N_e, T, 2)
    mask = result.mask             # (N_e, T)
    conditions = np.unique(labels)
    rows = []

    for cond in conditions:
        cond_idx = np.where(labels == cond)[0]
        for t, hpf in enumerate(time_values):
            obs = mask[cond_idx, t]
            if obs.sum() == 0:
                continue
            pts = positions[cond_idx[obs], t, :]
            rows.append({
                "condition": cond,
                "time_bin_center": hpf,
                "x": pts[:, 0].mean(),
                "y": pts[:, 1].mean(),
                "n_embryos": obs.sum(),
            })

    return pd.DataFrame(rows)


def bundle_width(
    result: CondensationResult,
    labels: np.ndarray,
    time_values: np.ndarray,
) -> pd.DataFrame:
    """Within-condition spread (mean distance from centroid) per time bin.

    Returns
    -------
    DataFrame with columns: condition, time_bin_center, spread, n_embryos
    """
    centroids = bundle_centroids(result, labels, time_values)
    positions = result.positions
    mask = result.mask
    rows = []

    for _, row in centroids.iterrows():
        cond = row["condition"]
        t = np.searchsorted(time_values, row["time_bin_center"])
        cond_idx = np.where(labels == cond)[0]
        obs = mask[cond_idx, t]
        if obs.sum() < 2:
            continue
        pts = positions[cond_idx[obs], t, :]
        centroid = np.array([row["x"], row["y"]])
        spread = np.linalg.norm(pts - centroid, axis=1).mean()
        rows.append({
            "condition": cond,
            "time_bin_center": row["time_bin_center"],
            "spread": spread,
            "n_embryos": obs.sum(),
        })

    return pd.DataFrame(rows)


def bundle_divergence_time(
    result: CondensationResult,
    labels: np.ndarray,
    time_values: np.ndarray,
    reference_condition: str = "inj_ctrl",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Estimate when each condition diverges from the reference bundle.

    Divergence time = first time bin where the distance between condition
    centroid and reference centroid exceeds `threshold`.

    Returns
    -------
    DataFrame with columns: condition, divergence_time_hpf
    """
    centroids = bundle_centroids(result, labels, time_values)
    ref = centroids[centroids["condition"] == reference_condition].set_index("time_bin_center")
    rows = []

    for cond in centroids["condition"].unique():
        if cond == reference_condition:
            continue
        cond_df = centroids[centroids["condition"] == cond].set_index("time_bin_center")
        shared_times = sorted(set(ref.index) & set(cond_df.index))
        div_time = None
        for hpf in shared_times:
            r = ref.loc[hpf]
            c = cond_df.loc[hpf]
            dist = np.sqrt((r["x"] - c["x"]) ** 2 + (r["y"] - c["y"]) ** 2)
            if dist > threshold:
                div_time = hpf
                break
        rows.append({"condition": cond, "divergence_time_hpf": div_time})

    return pd.DataFrame(rows)


def coherence_persistence_matrix(
    result: CondensationResult,
    labels: np.ndarray,
    n_iter_window: int = 50,
) -> np.ndarray:
    """Fraction of iterations in the last window where each embryo pair
    ended up in the same bundle (within threshold distance).

    Useful diagnostic for whether filament structure is stable or drifting.

    Not yet implemented — placeholder for iteration-level snapshots.
    """
    raise NotImplementedError(
        "Requires saving position snapshots during dynamics. "
        "Set save_snapshots=True in run_condensation (future feature)."
    )


def loss_curve_df(result: CondensationResult) -> pd.DataFrame:
    """Return the loss history as a tidy DataFrame."""
    return pd.DataFrame(result.loss_history)
