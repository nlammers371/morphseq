"""
iteration_ranking.py
--------------------
Ranking and rendering helpers for saved condensation iterations.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import plotting
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


def plot_iteration_scores(scores: pd.DataFrame, output_path: Path, *, title: str = "") -> None:
    if scores.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(scores["iter"], scores["geometry_score"], color="#2166AC", lw=2.0, label="geometry_score")
    top = scores.sort_values("rank").head(min(5, len(scores)))
    axes[0].scatter(top["iter"], top["geometry_score"], color="#B2182B", s=35, zorder=3, label="Top-K")
    axes[0].set_ylabel("geometry_score", fontsize=9)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    metric_cols = [
        ("score_separation", "sep"),
        ("score_compactness", "compact"),
        ("score_stability", "stable"),
        ("score_density_uniformity", "density_cv"),
    ]
    for col, label in metric_cols:
        if col in scores.columns:
            axes[1].plot(scores["iter"], scores[col], lw=1.5, label=label)
    axes[1].set_xlabel("iteration", fontsize=9)
    axes[1].set_ylabel("component score", fontsize=9)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8, ncol=4)

    if title:
        fig.suptitle(title, fontsize=10, y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def render_selected_iteration_bundle(
    *,
    positions: np.ndarray,
    mask: np.ndarray,
    time_values: np.ndarray,
    labels: np.ndarray,
    color_map: dict[str, str],
    output_dir: Path,
    title_prefix: str,
    snapshot_iter: int,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, _ = plotting.plot_trajectories(
        positions, mask, time_values,
        labels=labels, color_map=color_map,
        title=f"{title_prefix} | iter {snapshot_iter}",
    )
    fig.savefig(output_dir / "plot_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    snap_indices = np.linspace(0, len(time_values) - 1, min(6, len(time_values)), dtype=int)
    snapshot_times = [float(time_values[i]) for i in snap_indices]
    fig, _ = plotting.plot_panels(
        positions, mask, time_values,
        labels=labels, color_map=color_map,
        snapshot_times=snapshot_times,
        title=f"{title_prefix} | iter {snapshot_iter}",
    )
    fig.savefig(output_dir / "plot_panels.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, _ = plotting.plot_stacked_3d(
        positions, mask, time_values,
        labels=labels, color_map=color_map,
        title=f"{title_prefix} | iter {snapshot_iter}",
    )
    fig.savefig(output_dir / "plot_stacked_3d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        output_dir / "positions_iter.npz",
        positions=positions,
        mask=mask,
        time_values=time_values,
        labels=labels,
        snapshot_iter=int(snapshot_iter),
    )
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def save_ranking_outputs(
    scores: pd.DataFrame,
    *,
    output_dir: Path,
    config_payload: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_dir / "iteration_scores.csv", index=False)
    (output_dir / "iteration_geometry_manifest.json").write_text(json.dumps(config_payload, indent=2))
