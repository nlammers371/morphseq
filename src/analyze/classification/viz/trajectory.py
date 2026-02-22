from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    # Preferred import path requested for this workflow.
    from src.analyze.viz.plotting import plot_feature_over_time
except Exception:  # pragma: no cover - fallback for environments with PYTHONPATH=src
    from analyze.viz.plotting import plot_feature_over_time


def _color_lookup_for_values(values: Iterable[object]) -> dict[object, tuple[float, float, float, float]]:
    vals = [v for v in values if pd.notna(v)]
    uniq = list(dict.fromkeys(vals))
    if not uniq:
        return {}
    palette = plt.cm.tab20(np.linspace(0, 1, max(3, len(uniq))))
    return {val: palette[i] for i, val in enumerate(uniq)}


def save_pca_scatter(
    stage_table: pd.DataFrame,
    *,
    color_col: str,
    output_path: Path,
    title: str,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if "PC1" not in stage_table.columns or "PC2" not in stage_table.columns:
        raise ValueError("stage_table must contain PC1 and PC2 columns for PCA scatter")
    if color_col not in stage_table.columns:
        raise ValueError(f"Missing color_col='{color_col}' in stage_table")

    x = stage_table["PC1"].to_numpy(dtype=float)
    y = stage_table["PC2"].to_numpy(dtype=float)
    color_values = stage_table[color_col]

    cmap = _color_lookup_for_values(color_values.tolist())

    fig, ax = plt.subplots(figsize=(7, 5))
    for val in sorted(color_values.dropna().unique().tolist(), key=lambda z: str(z)):
        mask = color_values == val
        ax.scatter(
            x[mask.to_numpy()],
            y[mask.to_numpy()],
            label=str(val),
            s=28,
            alpha=0.85,
            color=cmap.get(val, "tab:blue"),
            edgecolor="none",
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def save_wrong_rate_null_diagnostics(
    stage_table: pd.DataFrame,
    *,
    output_path: Path,
    title: str = "Wrong-Rate Null Diagnostics",
) -> Path:
    """Visualize observed wrong-rate against permutation-null statistics."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    required = [
        "wrong_frac",
        "wrong_rate_window_null_mean",
        "wrong_rate_window_z",
        "qval_wrong_rate_window_perm",
    ]
    for col in required:
        if col not in stage_table.columns:
            raise ValueError(f"Missing required column '{col}' for null diagnostics plot")

    obs = stage_table["wrong_frac"].to_numpy(dtype=float)
    null_mean = stage_table["wrong_rate_window_null_mean"].to_numpy(dtype=float)
    z = stage_table["wrong_rate_window_z"].to_numpy(dtype=float)
    q = stage_table["qval_wrong_rate_window_perm"].to_numpy(dtype=float)

    sig_col = (
        stage_table["is_wrong_significant_in_window_perm"].to_numpy(dtype=bool)
        if "is_wrong_significant_in_window_perm" in stage_table.columns
        else np.zeros(len(stage_table), dtype=bool)
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    ax.scatter(
        null_mean[~sig_col],
        obs[~sig_col],
        s=18,
        alpha=0.6,
        color="tab:blue",
        label="not significant",
    )
    ax.scatter(
        null_mean[sig_col],
        obs[sig_col],
        s=22,
        alpha=0.85,
        color="tab:red",
        label="significant",
    )
    lims = [
        float(np.nanmin(np.r_[obs, null_mean])),
        float(np.nanmax(np.r_[obs, null_mean])),
    ]
    ax.plot(lims, lims, color="black", lw=1, linestyle="--")
    ax.set_xlabel("null mean wrong-rate (window perm)")
    ax.set_ylabel("observed wrong-rate")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    ax.hist(z, bins=30, alpha=0.85, color="tab:purple")
    ax.axvline(0.0, color="black", lw=1, linestyle="--")
    ax.set_xlabel("wrong_rate_window_z")
    ax.set_ylabel("count")
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.hist(q, bins=np.linspace(0, 1, 21), alpha=0.85, color="tab:green")
    ax.axvline(0.10, color="black", lw=1, linestyle="--", label="q=0.10")
    ax.axvline(0.05, color="black", lw=1, linestyle=":", label="q=0.05")
    ax.set_xlabel("qval_wrong_rate_window_perm")
    ax.set_ylabel("count")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def save_rolling_window_significance_counts(
    rolling_df: pd.DataFrame,
    *,
    output_path: Path,
    title: str = "Rolling-Window Wrong-Rate Significance",
) -> Path:
    """Plot number of significant embryos per rolling window."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    required = {"window_center_hpf", "is_wrong_significant_in_window_global_perm"}
    missing = required - set(rolling_df.columns)
    if missing:
        raise ValueError(f"Missing required rolling significance columns: {sorted(missing)}")

    agg = (
        rolling_df.groupby("window_center_hpf", as_index=False)
        .agg(
            n_sig_global=("is_wrong_significant_in_window_global_perm", "sum"),
            n_sig_window=("is_wrong_significant_in_window_perm", "sum"),
            min_q_global=("qval_wrong_rate_window_global_perm", "min"),
            median_z=("wrong_rate_window_z", "median"),
        )
        .sort_values("window_center_hpf")
    )

    x = agg["window_center_hpf"].to_numpy(dtype=float)
    n_sig_global = agg["n_sig_global"].to_numpy(dtype=float)
    n_sig_window = agg["n_sig_window"].to_numpy(dtype=float)
    min_q = agg["min_q_global"].to_numpy(dtype=float)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(x, n_sig_global, marker="o", color="tab:red", label="n significant (global FDR)")
    ax1.plot(x, n_sig_window, marker="o", color="tab:orange", label="n significant (within-window FDR)")
    ax1.set_xlabel("window center (hpf)")
    ax1.set_ylabel("number of embryos")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper left", fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(x, min_q, marker="s", color="tab:blue", alpha=0.8, label="min global q")
    ax2.set_ylabel("min global q-value")
    ax2.set_ylim(0, 1)
    ax2.axhline(0.10, color="black", lw=1, linestyle="--")
    ax2.axhline(0.05, color="black", lw=1, linestyle=":")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def save_rolling_destination_significance_counts(
    rolling_df: pd.DataFrame,
    *,
    output_path: Path,
    title: str = "Rolling Destination-Confusion Significance",
) -> Path:
    """Plot destination-confusion significance counts by rolling window."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    required = {
        "window_center_hpf",
        "is_dest_confusion_significant_global_perm",
        "is_dest_confusion_significant_perm",
        "qval_dest_confusion_global_perm",
    }
    missing = required - set(rolling_df.columns)
    if missing:
        raise ValueError(f"Missing required destination rolling columns: {sorted(missing)}")

    agg = (
        rolling_df.groupby("window_center_hpf", as_index=False)
        .agg(
            n_sig_global=("is_dest_confusion_significant_global_perm", "sum"),
            n_sig_window=("is_dest_confusion_significant_perm", "sum"),
            min_q_global=("qval_dest_confusion_global_perm", "min"),
            median_z=("dest_confusion_z", "median"),
        )
        .sort_values("window_center_hpf")
    )

    x = agg["window_center_hpf"].to_numpy(dtype=float)
    n_sig_global = agg["n_sig_global"].to_numpy(dtype=float)
    n_sig_window = agg["n_sig_window"].to_numpy(dtype=float)
    min_q = agg["min_q_global"].to_numpy(dtype=float)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(x, n_sig_global, marker="o", color="tab:red", label="n significant (global FDR)")
    ax1.plot(x, n_sig_window, marker="o", color="tab:orange", label="n significant (within-window FDR)")
    ax1.set_xlabel("window center (hpf)")
    ax1.set_ylabel("number of embryos")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper left", fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(x, min_q, marker="s", color="tab:blue", alpha=0.8, label="min global q")
    ax2.set_ylabel("min global q-value")
    ax2.set_ylim(0, 1)
    ax2.axhline(0.10, color="black", lw=1, linestyle="--")
    ax2.axhline(0.05, color="black", lw=1, linestyle=":")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_cluster_feature_trends(
    *,
    raw_df: pd.DataFrame,
    stage_table: pd.DataFrame,
    cluster_col: str,
    output_path: Path,
    features: list[str],
    time_col: str,
    embryo_id_col: str,
    group_color_by: str,
    facet_col_override: str | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if cluster_col not in stage_table.columns:
        raise ValueError(f"Missing cluster column '{cluster_col}' in stage_table")
    for col in [embryo_id_col] + features + [time_col]:
        if col not in raw_df.columns:
            raise ValueError(f"Missing required raw_df column '{col}'")

    cluster_df = stage_table[["embryo_id", cluster_col]].copy()
    cluster_df["embryo_id"] = cluster_df["embryo_id"].astype(str)
    merged = raw_df.copy()
    merged["embryo_id"] = merged[embryo_id_col].astype(str)
    merged = merged.merge(cluster_df, on="embryo_id", how="inner")

    facet_col = facet_col_override if (facet_col_override and facet_col_override in merged.columns) else cluster_col

    color_by = group_color_by if group_color_by in merged.columns else cluster_col
    if color_by == facet_col:
        color_by = cluster_col

    # Preferred route: one faceted figure, rows=features and cols=facet_col.
    try:
        plot_feature_over_time(
            df=merged,
            features=features,
            time_col=time_col,
            id_col=embryo_id_col,
            color_by=color_by,
            facet_col=facet_col,
            show_individual=False,
            show_error_band=True,
            trend_statistic="median",
            trend_smooth_sigma=1.5,
            bin_width=0.5,
            backend="matplotlib",
            output_path=output_path,
            title=f"Feature trends faceted by {facet_col} ({cluster_col})",
        )
        return output_path
    except Exception:
        # Fallback: split by facet groups and save a combined png panel.
        vals = sorted(merged[facet_col].dropna().unique().tolist(), key=lambda z: str(z))
        n_cols = min(3, max(1, len(vals)))
        n_rows = int(np.ceil(len(vals) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), squeeze=False)
        axes_flat = axes.ravel()
        for i, val in enumerate(vals):
            sub = merged[merged[facet_col] == val]
            ax = axes_flat[i]
            for feat in features:
                grp = sub.groupby(time_col, as_index=False)[feat].median()
                ax.plot(grp[time_col].to_numpy(dtype=float), grp[feat].to_numpy(dtype=float), label=feat)
            ax.set_title(f"{facet_col}={val}")
            ax.set_xlabel(time_col)
            ax.grid(alpha=0.25)
            if i == 0:
                ax.legend(loc="best", fontsize=8)
        for j in range(len(vals), len(axes_flat)):
            axes_flat[j].axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=220)
        plt.close(fig)
        return output_path
