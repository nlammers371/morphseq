"""
Pairwise coordinate heatmap for per-embryo phenotypic fingerprinting.

For a single embryo, renders a square matrix where only the upper-right triangle
is filled:
  - rows    = positive class of each pairwise classifier
  - columns = negative class of each pairwise classifier
  - upper-right triangle cell (row=A, col=B, col_index > row_index):
              time-averaged signed margin (class_signed_margin) on the A-vs-B classifier
  - diagonal and lower-left triangle: NaN / masked

Interpretation: a cell (row=A, col=B) shows how strongly the embryo looks like
class A (positive) vs class B (negative) according to the A-vs-B classifier.
Only the upper triangle is shown to avoid redundancy (the lower triangle would
be the negated mirror).

This figure grounds the geometric all-pairs phenotypic comparison and lets you
read off, for any embryo, where it falls on every phenotypic axis simultaneously.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze.viz.plotting.faceting_engine import (
    ColorbarSpec,
    HeatmapStyle,
    build_heatmap_figure,
    render,
)

__all__ = ["plot_pairwise_coordinate_heatmap"]

# Required columns in the input long-form DataFrame
_REQUIRED_COLS = {"embryo_id", "positive_label", "negative_label", "time_bin", "class_signed_margin"}


def _validate_input(df: pd.DataFrame) -> None:
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required columns: {sorted(missing)}. "
            f"Required: {sorted(_REQUIRED_COLS)}"
        )


def _build_upper_triangle_long(
    df: pd.DataFrame,
    *,
    time_bins: Optional[list] = None,
    label_order: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Aggregate margins and fill the upper-right triangle of the square matrix.

    Layout convention:
      - rows    = positive class  (A)
      - columns = negative class  (B)
      - upper-right triangle: cell (A, B) where col_index > row_index =  mean(class_signed_margin)
      - diagonal:             cell (A, A)                               =  NaN (masked)
      - lower-left triangle:  cell (B, A) where col_index < row_index  =  NaN (masked)

    The upper triangle shows: "how strongly does this embryo look like A vs B?"
    for every ordered pair (A, B) in the comparison set.

    Returns
    -------
    long_df : pd.DataFrame
        Columns: positive_label, negative_label, value.
        Only upper-triangle cells have finite values; diagonal and lower triangle are NaN.
    label_order : list[str]
        Ordered list of all genotype labels (shared row/col axis).
    """
    if time_bins is not None:
        df = df[df["time_bin"].isin(time_bins)].copy()

    # Aggregate class_signed_margin over time bins for each (positive_label, negative_label) pair
    agg = (
        df.groupby(["positive_label", "negative_label"])["class_signed_margin"]
        .mean()
        .reset_index(name="value")
    )

    all_labels_set = set(agg["positive_label"].unique()) | set(agg["negative_label"].unique())
    if label_order is not None:
        order = label_order
    else:
        order = sorted(all_labels_set, key=str)

    order_idx = {label: i for i, label in enumerate(order)}

    # Keep only upper-triangle entries: row_index < col_index
    # i.e. positive_label has a lower index than negative_label in label_order
    upper_rows = []
    for _, row in agg.iterrows():
        pos = row["positive_label"]
        neg = row["negative_label"]
        if pos not in order_idx or neg not in order_idx:
            continue
        pi = order_idx[pos]
        ni = order_idx[neg]
        if pi < ni:
            # Already upper triangle
            upper_rows.append({"positive_label": pos, "negative_label": neg, "value": row["value"]})
        elif ni < pi:
            # Swap: the original (pos, neg) with pi > ni is lower-triangle;
            # place it in upper triangle as (neg, pos) with negated sign
            upper_rows.append({"positive_label": neg, "negative_label": pos, "value": -row["value"]})
        # pi == ni: diagonal, skip

    upper_df = pd.DataFrame(upper_rows) if upper_rows else pd.DataFrame(
        columns=["positive_label", "negative_label", "value"]
    )

    # Collapse any duplicates by mean (shouldn't happen after dedup above, but safe)
    if not upper_df.empty:
        upper_df = (
            upper_df.groupby(["positive_label", "negative_label"])["value"]
            .mean()
            .reset_index()
        )

    # Fill in all cells of the full square matrix with NaN by default,
    # then overlay the upper-triangle values
    all_pairs = pd.DataFrame(
        [
            {"positive_label": r, "negative_label": c, "value": np.nan}
            for r in order
            for c in order
        ]
    )

    if not upper_df.empty:
        # Merge upper triangle values into the full grid
        all_pairs = all_pairs.merge(
            upper_df.rename(columns={"value": "value_upper"}),
            on=["positive_label", "negative_label"],
            how="left",
        )
        mask = all_pairs["value_upper"].notna()
        all_pairs.loc[mask, "value"] = all_pairs.loc[mask, "value_upper"]
        all_pairs = all_pairs.drop(columns=["value_upper"])

    return all_pairs, order


def plot_pairwise_coordinate_heatmap(
    df: pd.DataFrame,
    embryo_id: str,
    *,
    label_order: Optional[list[str]] = None,
    positive_labels: Optional[list[str]] = None,
    negative_labels: Optional[list[str]] = None,
    time_bins: Optional[list] = None,
    vmin: float = -1.0,
    vcenter: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "RdBu_r",
    title: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> "matplotlib.figure.Figure":
    """Render the pairwise phenotypic coordinate heatmap for a single embryo.

    Only the upper-right triangle is filled: rows = positive class, columns =
    negative class, cell value = time-averaged signed margin class_signed_margin.
    The diagonal and lower triangle are NaN / masked.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form data with columns:
        ``embryo_id``, ``positive_label``, ``negative_label``, ``time_bin``, ``class_signed_margin``.
        Typically comes from ``run_classification(save_contrast_coordinates=True)``
        → ``analysis.layers["raw_contrast_scores_long"]``.
    embryo_id : str
        The embryo to visualise.
    positive_labels : list[str] | None
        If provided, restricts the row axis to these genotype labels.  A comparison
        is kept if either its positive_label or negative_label is in this set
        (so the upper-triangle flip is handled automatically).
    negative_labels : list[str] | None
        If provided, restricts the column axis to these genotype labels.  Same
        union logic as ``positive_labels``.
    label_order : list[str] | None
        Explicit display order for the axis labels that survive filtering.
        Must be a subset of the labels present after filtering.  If None,
        sorted alphabetically.
    time_bins : list | None
        Subset of ``time_bin`` values to include in the average.  If None,
        all supported time bins for this embryo are used.
    vmin, vcenter, vmax : float
        Colormap range.  Defaults are -0.5 / 0 / 0.5.
    cmap : str
        Matplotlib colormap name.  A diverging blue-white-red map is recommended.
    title : str | None
        Figure title.  Defaults to ``"Pairwise phenotypic coordinates: <embryo_id>"``.
    output_path : str | Path | None
        If provided, saves to this path (PNG at 150 dpi) and closes the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _validate_input(df)

    embryo_df = df[df["embryo_id"].astype(str) == str(embryo_id)].copy()
    if embryo_df.empty:
        raise ValueError(f"No rows found for embryo_id={embryo_id!r} in the provided DataFrame.")

    # --- Resolve the axis label set ---
    # positive_labels / negative_labels define which labels appear on each axis.
    # A row is kept if its positive_label is in the requested positive set OR
    # its negative_label is in the requested negative set — the union ensures
    # no valid comparison is dropped just because the data stores it in the
    # opposite triangle orientation.
    pos_set = set(positive_labels) if positive_labels is not None else None
    neg_set = set(negative_labels) if negative_labels is not None else None

    if pos_set is not None or neg_set is not None:
        mask = pd.Series(True, index=embryo_df.index)
        if pos_set is not None:
            mask = mask & (
                embryo_df["positive_label"].isin(pos_set) |
                embryo_df["negative_label"].isin(pos_set)
            )
        if neg_set is not None:
            mask = mask & (
                embryo_df["negative_label"].isin(neg_set) |
                embryo_df["positive_label"].isin(neg_set)
            )
        embryo_df = embryo_df[mask].copy()

    if embryo_df.empty:
        raise ValueError(
            f"No rows remain after filtering positive_labels={positive_labels!r} / "
            f"negative_labels={negative_labels!r} for embryo_id={embryo_id!r}."
        )

    # The effective axis = union of both filter sets (or all present labels if unfiltered)
    present = set(embryo_df["positive_label"].unique()) | set(embryo_df["negative_label"].unique())
    axis_set = set()
    if pos_set is not None:
        axis_set |= pos_set & present
    if neg_set is not None:
        axis_set |= neg_set & present
    if not axis_set:
        axis_set = present  # no filters applied

    if label_order is not None:
        # Caller-supplied order must be a subset of the axis set
        unknown = set(label_order) - present
        if unknown:
            raise ValueError(
                f"label_order contains labels not present in the filtered data: {sorted(unknown)}"
            )
        effective_order = [g for g in label_order if g in axis_set]
    else:
        effective_order = sorted(axis_set, key=str)

    long_df, row_col_order = _build_upper_triangle_long(
        embryo_df,
        time_bins=time_bins,
        label_order=effective_order,
    )

    n = len(row_col_order)
    fig_size = max(5, n * 1.1)

    heatmap_style = HeatmapStyle(
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        vcenter=vcenter,
        missing_color="#e0e0e0",
    )
    colorbar = ColorbarSpec(label="Signed margin (class_signed_margin)", shrink=0.7)

    fig_title = title or f"Pairwise phenotypic coordinates: {embryo_id}"

    fig_data = build_heatmap_figure(
        long_df,
        heatmap_row="positive_label",
        heatmap_col="negative_label",
        value_col="value",
        heatmap_row_order=row_col_order,
        heatmap_col_order=row_col_order,
        heatmap_style=heatmap_style,
        colorbar=colorbar,
        title=fig_title,
        x_label="Negative class (B)",
        y_label="Positive class (A)",
    )

    fig = render(
        fig_data,
        backend="matplotlib",
    )

    # Resize to approximate square after render (before saving)
    fig.set_size_inches(fig_size + 1.5, fig_size)

    if output_path is not None:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig
