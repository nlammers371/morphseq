"""
Penetrance Bin-Width Analysis Utilities

Reusable helpers extracted from the curvature temporal analysis workflow.
They support:
1. Building a global WT IQR band.
2. Marking penetrant timepoints relative to that band.
3. Binning data by developmental time and computing embryo-level penetrance.
4. Summarising/plotting the effect of different temporal bin widths.

Example usage lives in src/analyze/penetrance/README.md.
"""

from __future__ import annotations

import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_METRIC = "baseline_deviation_normalized"
ALT_METRIC = "normalized_baseline_deviation"
DEFAULT_TIME_COL = "predicted_stage_hpf"

DEFAULT_GENOTYPE_ORDER: Sequence[Tuple[str, str]] = (
    ("wt", "WT"),
    ("het", "Het"),
    ("homo", "Homo"),
)

DEFAULT_STYLE_MAP = {
    "wt": {"color": "#1f77b4", "marker": "^"},
    "het": {"color": "#ff7f0e", "marker": "o"},
    "homo": {"color": "#d62728", "marker": "s"},
}


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def ensure_metric_column(
    df: pd.DataFrame,
    *,
    metric: str = DEFAULT_METRIC,
    alt_metric: str = ALT_METRIC,
) -> pd.DataFrame:
    """
    Guarantee the preferred metric column exists (copy from fallback if needed).

    Returns a DataFrame (copy-on-write) that always contains `metric`.
    Raises KeyError if neither column exists.
    """
    if metric in df.columns:
        return df

    if alt_metric in df.columns:
        df = df.copy()
        df[metric] = df[alt_metric]
        print(
            f"NOTE: '{metric}' not found. Using '{alt_metric}' values instead."
        )
        return df

    raise KeyError(
        f"Required metric column not found. Expected '{metric}' or '{alt_metric}'."
    )


# ---------------------------------------------------------------------------
# Thresholding
# ---------------------------------------------------------------------------

def compute_global_iqr_bounds(
    wt_df: pd.DataFrame,
    *,
    metric: str = DEFAULT_METRIC,
    k: float = 1.5,
) -> Dict[str, float]:
    """
    Compute pooled WT IQR bounds.
    Returns dict with keys: low, high, median, mean, q1, q3, iqr, n_samples, k.
    """
    values = wt_df[metric].to_numpy()

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1

    return {
        "low": q1 - k * iqr,
        "high": q3 + k * iqr,
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "n_samples": len(values),
        "k": k,
    }


def mark_penetrant_global(
    df: pd.DataFrame,
    wt_bounds: Dict[str, float],
    *,
    metric: str = DEFAULT_METRIC,
    penetrant_col: str = "penetrant",
) -> pd.DataFrame:
    """
    Add a binary column marking rows outside the WT bounds.
    """
    df = df.copy()
    mask = (df[metric] < wt_bounds["low"]) | (df[metric] > wt_bounds["high"])
    df[penetrant_col] = mask.astype(int)
    return df


# ---------------------------------------------------------------------------
# Time binning + penetrance computation
# ---------------------------------------------------------------------------

def bin_data_by_time(
    df: pd.DataFrame,
    *,
    bin_width: float = 2.0,
    time_col: str = DEFAULT_TIME_COL,
    bin_col: str = "time_bin",
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Append a `time_bin` column representing uniform bin centers.
    Returns (df_with_bins, bin_centers).
    """
    df = df.copy()
    min_time = df[time_col].min()
    max_time = df[time_col].max()

    bin_edges = np.arange(
        np.floor(min_time / bin_width) * bin_width,
        np.ceil(max_time / bin_width) * bin_width + bin_width,
        bin_width,
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    df[bin_col] = pd.cut(
        df[time_col],
        bins=bin_edges,
        labels=bin_centers,
        include_lowest=True,
        right=False,
    ).astype(float)

    return df, bin_centers


def compute_penetrance_by_time(
    df_binned: pd.DataFrame,
    time_bins: np.ndarray,
    *,
    embryo_col: str = "embryo_id",
    metric_col: str = "penetrant",
    bin_col: str = "time_bin",
) -> List[Dict[str, float]]:
    """
    Compute embryo-level penetrance per time bin (with binomial standard error).
    """
    results: List[Dict[str, float]] = []

    # group per bin to avoid repeated filtering for empty bins
    grouped = df_binned.dropna(subset=[bin_col]).groupby(bin_col)

    for time_bin in time_bins:
        if time_bin not in grouped.groups:
            continue

        bin_df = grouped.get_group(time_bin).dropna(subset=[metric_col])
        if bin_df.empty:
            continue

        embryos = bin_df[embryo_col].unique()
        penetrant_embryos = bin_df.loc[bin_df[metric_col] == 1, embryo_col].unique()

        n_embryos = len(embryos)
        n_pen = len(penetrant_embryos)
        penetrance = n_pen / n_embryos if n_embryos else 0.0
        se = (
            np.sqrt(penetrance * (1 - penetrance) / n_embryos)
            if n_embryos
            else 0.0
        )

        results.append(
            {
                "time_bin": float(time_bin),
                "embryo_penetrance": float(penetrance),
                "n_embryos": int(n_embryos),
                "n_penetrant": int(n_pen),
                "se": float(se),
            }
        )

    return results


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def compute_summary_stats(
    temporal_results: Dict[float, Dict[str, List[Dict[str, float]]]],
    binwidths: Iterable[float],
    *,
    genotype_order: Sequence[Tuple[str, str]] = DEFAULT_GENOTYPE_ORDER,
) -> pd.DataFrame:
    """
    Build a summary DataFrame for reporting/interpretation.
    """
    rows = []
    for bw in binwidths:
        results = temporal_results.get(bw, {})
        for key, label in genotype_order:
            data = results.get(key, [])
            if not data:
                continue
            pens = np.array([entry["embryo_penetrance"] for entry in data], dtype=float)
            ses = np.array([entry["se"] for entry in data], dtype=float)
            rows.append(
                {
                    "binwidth_hpf": bw,
                    "genotype": label,
                    "n_time_bins": len(data),
                    "mean_penetrance_%": pens.mean() * 100,
                    "std_penetrance_%": pens.std(ddof=0) * 100,
                    "min_penetrance_%": pens.min() * 100,
                    "max_penetrance_%": pens.max() * 100,
                    "range_penetrance_%": (pens.max() - pens.min()) * 100,
                    "mean_se_%": ses.mean() * 100,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _resolve_style(
    genotype_order: Sequence[Tuple[str, str]],
    style_map: Optional[Dict[str, Dict[str, str]]] = None,
):
    styles = {}
    style_map = style_map or DEFAULT_STYLE_MAP
    marker_cycle = ["^", "o", "s", "D", "v"]
    color_cycle = plt.cm.tab10(np.linspace(0, 1, len(genotype_order)))

    for idx, (key, label) in enumerate(genotype_order):
        base = style_map.get(key, {})
        color = base.get("color", color_cycle[idx % len(color_cycle)])
        marker = base.get("marker", marker_cycle[idx % len(marker_cycle)])
        styles[key] = {"color": color, "marker": marker, "label": label}
    return styles


def plot_temporal_by_binwidth(
    temporal_results: Dict[float, Dict[str, List[Dict[str, float]]]],
    binwidths: Sequence[float],
    *,
    genotype_order: Sequence[Tuple[str, str]] = DEFAULT_GENOTYPE_ORDER,
    style_map: Optional[Dict[str, Dict[str, str]]] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """
    Create a subplot per bin width with WT/Het/Homo (or custom) penetrance curves.
    """
    styles = _resolve_style(genotype_order, style_map)
    n_cols = len(binwidths)
    figsize = figsize or (5 * n_cols, 4.5)

    fig, axes = plt.subplots(1, n_cols, figsize=figsize, sharey=True)
    axes = np.atleast_1d(axes)

    for ax, bw in zip(axes, binwidths):
        results = temporal_results.get(bw, {})
        for key, _ in genotype_order:
            data = results.get(key, [])
            if not data:
                continue
            times = [entry["time_bin"] for entry in data]
            pens = [entry["embryo_penetrance"] * 100 for entry in data]
            ses = [entry["se"] * 100 for entry in data]
            style = styles[key]
            ax.errorbar(
                times,
                pens,
                yerr=ses,
                color=style["color"],
                marker=style["marker"],
                linewidth=2,
                markersize=6,
                label=style["label"],
                alpha=0.8,
                capsize=3,
            )

        ax.set_title(f"{bw:.1f} hpf bins", fontweight="bold")
        ax.set_xlabel("Time (hpf)")
        ax.set_ylabel("Penetrance (%)")
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Temporal Penetrance Across Bin Widths", y=1.02, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_genotype_smoothing(
    temporal_results: Dict[float, Dict[str, List[Dict[str, float]]]],
    binwidths: Sequence[float],
    *,
    genotype_order: Sequence[Tuple[str, str]] = DEFAULT_GENOTYPE_ORDER,
    style_map: Optional[Dict[str, Dict[str, str]]] = None,
    figsize: Tuple[float, float] = (16, 5),
) -> plt.Figure:
    """
    Plot one subplot per genotype; overlay curves for each bin width.
    """
    styles = _resolve_style(genotype_order, style_map)
    fig, axes = plt.subplots(1, len(genotype_order), figsize=figsize, sharey=True)
    axes = np.atleast_1d(axes)

    color_cycle = ["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"]
    linestyle_cycle = ["-", "--", ":", "-."]

    for ax, (geno_key, label) in zip(axes, genotype_order):
        for idx, bw in enumerate(binwidths):
            data = temporal_results.get(bw, {}).get(geno_key, [])
            if not data:
                continue
            times = [entry["time_bin"] for entry in data]
            pens = [entry["embryo_penetrance"] * 100 for entry in data]
            ses = [entry["se"] * 100 for entry in data]

            ax.errorbar(
                times,
                pens,
                yerr=ses,
                color=color_cycle[idx % len(color_cycle)],
                linestyle=linestyle_cycle[idx % len(linestyle_cycle)],
                linewidth=2,
                markersize=5,
                marker=styles[geno_key]["marker"],
                label=f"{bw:.1f} hpf",
                alpha=0.75,
                capsize=3,
            )

        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Time (hpf)")
        ax.set_ylabel("Penetrance (%)")
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Bin width", fontsize=9)

    fig.suptitle("Smoothing Effect by Bin Width", y=1.05, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_wt_focus(
    temporal_results: Dict[float, Dict[str, List[Dict[str, float]]]],
    binwidths: Sequence[float],
    *,
    genotype_key: str = "wt",
    label: str = "WT",
    marker_cycle: Optional[Sequence[str]] = None,
    color_cycle: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (10, 5),
) -> plt.Figure:
    """
    Single-axis plot highlighting one genotype (default WT) across bin widths.
    """
    marker_cycle = marker_cycle or ["^", "o", "s", "D"]
    color_cycle = color_cycle or ["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"]

    fig, ax = plt.subplots(figsize=figsize)

    for idx, bw in enumerate(binwidths):
        data = temporal_results.get(bw, {}).get(genotype_key, [])
        if not data:
            continue
        times = [entry["time_bin"] for entry in data]
        pens = [entry["embryo_penetrance"] * 100 for entry in data]
        ses = [entry["se"] * 100 for entry in data]

        ax.errorbar(
            times,
            pens,
            yerr=ses,
            color=color_cycle[idx % len(color_cycle)],
            marker=marker_cycle[idx % len(marker_cycle)],
            linewidth=2.5,
            markersize=6 + idx,
            label=f"{bw:.1f} hpf bins (n={len(data)})",
            alpha=0.85,
            capsize=4,
        )

    ax.set_title(f"{label} Penetrance vs Bin Width", fontweight="bold")
    ax.set_xlabel("Time (hpf)")
    ax.set_ylabel("Penetrance (%)")
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


__all__ = [
    "ensure_metric_column",
    "compute_global_iqr_bounds",
    "mark_penetrant_global",
    "bin_data_by_time",
    "compute_penetrance_by_time",
    "compute_summary_stats",
    "plot_temporal_by_binwidth",
    "plot_genotype_smoothing",
    "plot_wt_focus",
    "DEFAULT_GENOTYPE_ORDER",
]
