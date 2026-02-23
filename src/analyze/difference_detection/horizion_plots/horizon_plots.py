"""
Plotting helpers for horizon-style heatmaps showing forward-time relationships.

These utilities consolidate the matplotlib logic used in the exploratory
comparison notebooks so that the same styling can be reused by scripts and
notebooks alike.

Horizon plots are designed for time matrix data where analysis proceeds forward in time.
Expected data structure:
- Rows represent START times (ordered low-to-high, lowest at top)
- Columns represent TARGET times (ordered low-to-high, lowest at left)
- Data fills the UPPER TRIANGLE: only where start_time < target_time
- The lower triangle is empty (impossible predictions into the past)

This creates a clear visualization of how information/patterns at early timepoints
predict later timepoints, with empty space below the diagonal representing
undefined/impossible comparisons.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from .time_matrix import align_matrix_times

MatrixLike = Union[pd.DataFrame, Dict[str, pd.DataFrame]]
NestedMatrixDict = Mapping[str, Mapping[str, pd.DataFrame]]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def compute_shared_colorscale(
    matrices: MatrixLike,
    clip_percentiles: Optional[Tuple[float, float]] = (5.0, 95.0),
) -> Tuple[Optional[float], Optional[float]]:
    """
    Determine colour scale bounds across one or many matrices.

    Parameters
    ----------
    matrices
        Either a single dataframe, a mapping of label -> dataframe, or the
        nested mapping used by :func:`plot_horizon_grid`.
    clip_percentiles
        Optional percentile range for robust scaling.  Pass ``None`` to use the
        absolute min/max.  A boolean ``True``/``False`` is also accepted for
        backwards compatibility (``True`` => (5, 95), ``False`` => ``None``).
    """

    if isinstance(clip_percentiles, bool):
        clip_percentiles = (5.0, 95.0) if clip_percentiles else None

    values = []
    for matrix in _iter_matrices(matrices):
        arr = matrix.to_numpy().ravel()
        arr = arr[~np.isnan(arr)]
        if arr.size:
            values.append(arr)

    if not values:
        return None, None

    all_values = np.concatenate(values)
    if clip_percentiles is None:
        return float(np.min(all_values)), float(np.max(all_values))

    low, high = np.percentile(all_values, clip_percentiles)
    return float(low), float(high)


# ---------------------------------------------------------------------------
# Single heatmap
# ---------------------------------------------------------------------------


def plot_single_horizon(
    matrix: pd.DataFrame,
    metric: str = "mae",
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    xlabel: str = "Target Time (hpf)",
    ylabel: str = "Start Time (hpf)",
    annotate: bool = False,
    add_colorbar: bool = True,
) -> plt.Axes:
    """
    Render a single horizon heatmap.

    Expected matrix structure for forward-time analysis:
    - Rows (index): Start times, ordered low-to-high (earliest at top)
    - Columns (columns): Target times, ordered low-to-high (earliest at left)
    - Upper triangle populated: matrix[i, j] only defined where start_time[i] < target_time[j]
    - Lower triangle empty: represents impossible predictions (predicting into the past)

    This ensures the visualization shows forward-time relationships with the upper-right
    triangle filled (how well early observations predict later observations).
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    if matrix.empty:
        ax.set_axis_off()
        ax.set_title(title or "No data")
        return ax

    im = ax.imshow(
        matrix.values,
        origin="upper",
        cmap=cmap,
        interpolation="nearest",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(range(len(matrix.columns)))
    # Convert column labels to integers if they are numeric
    col_labels = [int(c) if isinstance(c, (int, float)) and c == int(c) else c for c in matrix.columns]
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix.index)))
    # Convert row labels to integers if they are numeric
    row_labels = [int(r) if isinstance(r, (int, float)) and r == int(r) else r for r in matrix.index]
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10)

    if add_colorbar:
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.08)
        cbar.ax.set_ylabel(metric.upper(), rotation=270, labelpad=15, fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    if annotate:
        for i, j in product(range(matrix.shape[0]), range(matrix.shape[1])):
            value = matrix.iat[i, j]
            if np.isnan(value):
                continue
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="black", fontsize=8)

    return ax


# ---------------------------------------------------------------------------
# Horizon grids
# ---------------------------------------------------------------------------


def plot_horizon_grid(
    matrices: NestedMatrixDict,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    metric: str = "mae",
    cmap: str = "viridis",
    clip_percentiles: Union[Tuple[float, float], bool, None] = (5.0, 95.0),
    annotate: bool = False,
    loeo_highlight: Optional[Mapping[str, str]] = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot a grid of horizon heatmaps (rows = conditions, columns = groups).
    """

    row_keys = list(matrices.keys())
    if len(row_labels) != len(row_keys):
        raise ValueError("row_labels length must match number of rows in matrices")

    if not matrices:
        raise ValueError("No matrices provided for plotting.")

    first_row = next(iter(matrices.values()))
    col_keys = list(first_row.keys())
    if len(col_labels) != len(col_keys):
        raise ValueError("col_labels length must match number of columns in matrices")

    vmin, vmax = compute_shared_colorscale(matrices, clip_percentiles=clip_percentiles)

    n_rows = len(row_keys)
    n_cols = len(col_keys)
    if figsize is None:
        figsize = (6 * n_cols, 5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for r_idx, row_key in enumerate(row_keys):
        for c_idx, col_key in enumerate(col_keys):
            ax = axes[r_idx, c_idx]
            matrix = matrices.get(row_key, {}).get(col_key)

            if matrix is None or matrix.empty:
                ax.set_axis_off()
                ax.set_title(f"{col_labels[c_idx]} – No data")
                continue

            plot_single_horizon(
                matrix,
                metric=metric,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                ax=ax,
                title=f"{row_labels[r_idx]} → {col_labels[c_idx]}",
                annotate=annotate,
                add_colorbar=False,
            )

            if loeo_highlight and loeo_highlight.get(row_key) == col_key:
                _add_highlight_border(ax, color="red", linewidth=2.0)

    if title:
        fig.suptitle(title, fontsize=18, y=0.98)

    # Adjust layout first to make room for colorbar
    if title:
        fig.subplots_adjust(right=0.88, top=0.93, bottom=0.1, left=0.08, wspace=0.3, hspace=0.3)
    else:
        fig.subplots_adjust(right=0.88, bottom=0.1, left=0.08, wspace=0.3, hspace=0.3)

    # Create a single shared colourbar with sufficient padding to avoid overlap
    if vmin is not None and vmax is not None:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(cmap))
        sm.set_clim(vmin, vmax)
        # Place colorbar to the right of the rightmost axes
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label=metric.upper())
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.set_ylabel(metric.upper(), fontsize=9)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_best_condition_map(
    matrices: NestedMatrixDict,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    metric: str = "mae",
    mode: str = "min",
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Visualise which row condition performs best for every cell.

    ``mode`` determines whether the smallest (default) or largest value is
    considered "best".  Cells with missing data for all conditions are left
    blank.
    """

    if mode not in {"min", "max"}:
        raise ValueError("mode must be either 'min' or 'max'")

    row_keys = list(matrices.keys())
    col_keys = list(next(iter(matrices.values())).keys())

    if len(row_labels) != len(row_keys) or len(col_labels) != len(col_keys):
        raise ValueError("row_labels/col_labels must match the matrices structure.")

    if figsize is None:
        figsize = (6 * len(col_keys), 5 * len(row_keys))

    fig, axes = plt.subplots(1, len(col_keys), figsize=figsize, squeeze=False)
    axes = axes[0]

    tab_cmap = plt.cm.get_cmap("tab10", len(row_keys))
    colors = [tab_cmap(i) for i in range(len(row_keys))]
    legend_handles = [mpatches.Patch(color=colors[i], label=row_labels[i]) for i in range(len(row_labels))]

    for c_idx, col_key in enumerate(col_keys):
        ax = axes[c_idx]

        per_condition = {
            row_key: matrices[row_key].get(col_key)
            for row_key in row_keys
            if col_key in matrices[row_key]
        }
        if not per_condition:
            ax.set_axis_off()
            ax.set_title(f"{col_labels[c_idx]} – No data")
            continue

        aligned = align_matrix_times(per_condition, time_axis="both")
        stacked = np.stack([df.to_numpy() for df in aligned.values()])

        with np.errstate(invalid="ignore"):
            if mode == "min":
                scores = np.where(np.isnan(stacked), np.inf, stacked)
                winner_idx = np.argmin(scores, axis=0)
                invalid_mask = np.all(np.isnan(stacked), axis=0)
            else:
                scores = np.where(np.isnan(stacked), -np.inf, stacked)
                winner_idx = np.argmax(scores, axis=0)
                invalid_mask = np.all(np.isnan(stacked), axis=0)

        best_map = np.full_like(stacked[0], fill_value=np.nan, dtype=float)
        for idx in range(len(aligned)):
            best_map[winner_idx == idx] = idx
        best_map[invalid_mask] = np.nan

        masked = np.ma.masked_invalid(best_map)
        im = ax.imshow(masked, origin="lower", aspect="auto", cmap=tab_cmap, vmin=-0.5, vmax=len(row_keys) - 0.5)

        reference_matrix = next(iter(aligned.values()))
        ax.set_xticks(range(reference_matrix.shape[1]))
        ax.set_xticklabels(reference_matrix.columns, rotation=45, ha="right")
        ax.set_yticks(range(reference_matrix.shape[0]))
        ax.set_yticklabels(reference_matrix.index)
        ax.set_title(f"Best {mode} {metric.upper()} – {col_labels[c_idx]}")

    axes[0].set_ylabel("Start Time (hpf)")
    axes[len(axes) // 2].set_xlabel("Target Time (hpf)")

    fig.legend(handles=legend_handles, loc="upper center", ncol=len(row_labels), bbox_to_anchor=(0.5, 1.02))
    if title:
        fig.suptitle(title, fontsize=18, y=1.05)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _iter_matrices(matrices: MatrixLike) -> Iterable[pd.DataFrame]:
    if isinstance(matrices, pd.DataFrame):
        yield matrices
    elif isinstance(matrices, Mapping):
        for value in matrices.values():
            yield from _iter_matrices(value)
    else:
        raise TypeError(f"Unsupported matrix container type: {type(matrices)!r}")


def _add_highlight_border(ax: plt.Axes, color: str = "red", linewidth: float = 2.0) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    rect = mpatches.Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        linewidth=linewidth,
        edgecolor=color,
        facecolor="none",
        transform=ax.transData,
    )
    ax.add_patch(rect)


__all__ = [
    "compute_shared_colorscale",
    "plot_single_horizon",
    "plot_horizon_grid",
    "plot_best_condition_map",
]
