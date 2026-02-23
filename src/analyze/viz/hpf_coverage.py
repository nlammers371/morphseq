"""
Utilities for assessing experiment coverage across predicted_stage_hpf.
"""

from typing import Optional, Tuple

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_experiment_time_coverage(
    df: pd.DataFrame,
    experiment_col: str = "experiment_id",
    hpf_col: str = "predicted_stage_hpf",
    embryo_col: str = "embryo_id",
    bin_width: float = 0.5,
    min_embryos_per_bin: int = 1,
    hpf_min: Optional[float] = None,
    hpf_max: Optional[float] = None,
) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Compute experiment coverage across HPF bins.

    Returns
    -------
    bins_mid : np.ndarray
        Bin midpoints.
    cover_df : pd.DataFrame
        (experiments x bins) boolean coverage matrix.
    cov_count : np.ndarray
        Count of experiments covering each bin.
    """
    d = df[[experiment_col, hpf_col, embryo_col]].dropna().copy()
    d[hpf_col] = pd.to_numeric(d[hpf_col], errors="coerce")
    d = d.dropna(subset=[hpf_col])

    if hpf_min is None:
        hpf_min = float(d[hpf_col].min())
    if hpf_max is None:
        hpf_max = float(d[hpf_col].max())

    start = np.floor(hpf_min / bin_width) * bin_width
    end = np.ceil(hpf_max / bin_width) * bin_width
    edges = np.arange(start, end + bin_width, bin_width)

    d["hpf_bin"] = pd.cut(d[hpf_col], bins=edges, include_lowest=True, right=False)

    counts = (
        d.groupby([experiment_col, "hpf_bin"])[embryo_col]
        .nunique()
        .unstack(fill_value=0)
    )

    cover_df = (counts >= min_embryos_per_bin)

    full_bins = pd.IntervalIndex.from_breaks(edges, closed="left")
    cover_df = cover_df.reindex(columns=full_bins, fill_value=False)

    cov_count = cover_df.sum(axis=0).to_numpy()

    bins_mid = np.array([(iv.left + iv.right) / 2 for iv in cover_df.columns])

    return bins_mid, cover_df, cov_count


def longest_interval_where(
    cov_count: np.ndarray,
    bins_mid: np.ndarray,
    min_experiments: int = 5
) -> Tuple[Optional[float], Optional[float], int]:
    """
    Find the longest contiguous interval where cov_count >= min_experiments.

    Returns (hpf_start, hpf_end, n_bins) or (None, None, 0) if none found.
    """
    ok = cov_count >= min_experiments
    if not ok.any():
        return None, None, 0

    idx = np.where(ok)[0]
    splits = np.where(np.diff(idx) > 1)[0] + 1
    runs = np.split(idx, splits)

    best = max(runs, key=len)
    hpf_start = bins_mid[best[0]]
    hpf_end = bins_mid[best[-1]]
    return float(hpf_start), float(hpf_end), int(len(best))


def plot_hpf_overlap_quick(
    bins_mid: np.ndarray,
    cov_count: np.ndarray,
    cover_df: Optional[pd.DataFrame] = None,
    min_experiments: int = 5,
    show_heatmap: bool = True,
    max_experiments_heatmap: int = 80,
    title_prefix: str = "HPF overlap",
    coverage_plot_path: Optional[Path] = None,
    heatmap_path: Optional[Path] = None,
    show: bool = True,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Quick coverage plot + optional heatmap.
    """
    hpf_start, hpf_end, n_bins = longest_interval_where(
        cov_count, bins_mid, min_experiments=min_experiments
    )

    plt.figure()
    plt.plot(bins_mid, cov_count)
    plt.axhline(min_experiments)
    if hpf_start is not None:
        plt.axvspan(hpf_start, hpf_end, alpha=0.2)
        plt.title(
            f"{title_prefix}: longest window with >= {min_experiments} experiments\n"
            f"~[{hpf_start:.2f}, {hpf_end:.2f}] (bins={n_bins})"
        )
    else:
        plt.title(f"{title_prefix}: no window with >= {min_experiments} experiments")
    plt.xlabel("predicted_stage_hpf (binned)")
    plt.ylabel("# experiments covering bin")
    plt.tight_layout()
    if coverage_plot_path is not None:
        plt.savefig(coverage_plot_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

    if show_heatmap and cover_df is not None:
        hm = cover_df
        if hm.shape[0] > max_experiments_heatmap:
            order = hm.sum(axis=1).sort_values(ascending=False).index[:max_experiments_heatmap]
            hm = hm.loc[order]

        plt.figure()
        plt.imshow(hm.to_numpy().astype(int), aspect="auto", interpolation="nearest")
        plt.yticks(range(hm.shape[0]), hm.index.astype(str))
        xticks = np.linspace(0, len(bins_mid) - 1, num=min(10, len(bins_mid))).astype(int)
        plt.xticks(
            ticks=xticks,
            labels=[f"{bins_mid[i]:.1f}" for i in xticks],
            rotation=0,
        )
        plt.xlabel("predicted_stage_hpf (bin mids)")
        plt.ylabel("experiment")
        plt.title("Experiment x HPF-bin coverage (1=covered)")
        plt.tight_layout()
        if heatmap_path is not None:
            plt.savefig(heatmap_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()

    return hpf_start, hpf_end


# Backward-compatible alias
experiment_hpf_coverage = plot_experiment_time_coverage
