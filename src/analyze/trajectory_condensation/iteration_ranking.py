"""
iteration_ranking.py
--------------------
Scoring and selection logic for saved condensation iterations.

For visualization of ranking results, see viz/iteration_choice_plots.py.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .space_density_metrics import summarize_iteration_geometry


def score_saved_iterations(
    position_history: np.ndarray,
    snapshot_iters: list[int] | np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    time_values: np.ndarray,
    metrics_df: pd.DataFrame,
    *,
    objective: str = "balanced",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    snapshot_iters = [int(x) for x in snapshot_iters]
    metric_lookup = metrics_df.set_index("iter") if "iter" in metrics_df.columns else pd.DataFrame()

    for idx, iter_idx in enumerate(snapshot_iters):
        geom = summarize_iteration_geometry(position_history[idx], mask, labels, time_values)
        row = {"iter": iter_idx, "snapshot_index": idx, **geom}
        if not metric_lookup.empty and iter_idx in metric_lookup.index:
            metric_row = metric_lookup.loc[iter_idx]
            for col in [
                "energy_total",
                "energy_change_rel",
                "disp_rms_rel",
                "disp_max_rel",
                "coherence_change_rel",
                "mu",
            ]:
                if col in metric_row.index:
                    row[col] = float(metric_row[col])
        rows.append(row)

    scores = pd.DataFrame(rows)
    if scores.empty:
        return scores

    def normalize(series: pd.Series, *, larger_better: bool) -> pd.Series:
        s = series.astype(float)
        valid = s[np.isfinite(s)]
        if valid.empty:
            return pd.Series(np.nan, index=s.index)
        lo = float(valid.min())
        hi = float(valid.max())
        if hi - lo < 1e-12:
            out = pd.Series(1.0, index=s.index)
        else:
            out = (s - lo) / (hi - lo)
        if not larger_better:
            out = 1.0 - out
        out[~np.isfinite(s)] = np.nan
        return out

    scores["score_separation"] = normalize(scores["centroid_separation_median"], larger_better=True)
    scores["score_compactness"] = normalize(scores["within_over_between"], larger_better=False)
    scores["score_stability"] = normalize(scores["centroid_shift_mean"], larger_better=False)
    scores["score_crowding"] = normalize(scores["crowding_score"], larger_better=True)
    scores["score_density_level"] = normalize(scores["density_knn_mean"], larger_better=True)
    scores["score_density_uniformity"] = normalize(scores["density_knn_cv"], larger_better=False)
    scores["score_dispersion"] = normalize(scores.get("disp_rms_rel", pd.Series(np.nan, index=scores.index)), larger_better=False)
    scores["score_energy"] = normalize(scores.get("energy_change_rel", pd.Series(np.nan, index=scores.index)), larger_better=False)

    if objective != "balanced":
        raise ValueError(f"Unsupported selection objective {objective!r}")

    component_cols = [
        "score_separation",
        "score_compactness",
        "score_stability",
        "score_crowding",
        "score_density_level",
        "score_density_uniformity",
        "score_dispersion",
        "score_energy",
    ]
    scores["geometry_score"] = scores[component_cols].mean(axis=1, skipna=True)
    scores["selection_score"] = scores["geometry_score"]
    scores = scores.sort_values(["geometry_score", "iter"], ascending=[False, True]).reset_index(drop=True)
    scores["rank"] = np.arange(1, len(scores) + 1, dtype=int)
    return scores


def select_evenly_distributed(scores: pd.DataFrame, k: int) -> pd.DataFrame:
    """Return k rows whose geometry_score values are most evenly spaced across [min, max]."""
    if len(scores) <= k:
        return scores
    targets = np.linspace(scores["geometry_score"].min(), scores["geometry_score"].max(), k)
    chosen_idx = []
    remaining = set(scores.index)
    for t in targets:
        idx = min(remaining, key=lambda i: abs(scores.loc[i, "geometry_score"] - t))
        chosen_idx.append(idx)
        remaining.discard(idx)
    return scores.loc[chosen_idx].sort_values("geometry_score", ascending=False)


def save_ranking_outputs(
    scores: pd.DataFrame,
    *,
    output_dir: Path,
    config_payload: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_dir / "iteration_scores.csv", index=False)
    (output_dir / "iteration_geometry_manifest.json").write_text(json.dumps(config_payload, indent=2))
