"""
Generic heatmap figure builder.

Converts long-form (tidy) DataFrames into heatmap FigureData ready for render().

Two public functions:
- resolve_axis_order()      — global axis ordering helper (documented contract)
- prepare_heatmap_panel()   — single-panel pivot + sig_mask + annotations
- build_heatmap_figure()    — full FigureData assembly with faceting
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .ir import (
    ColorbarSpec,
    FacetSpec,
    FigureData,
    HeatmapData,
    HeatmapStyle,
    SubplotData,
)
from .utils import iter_facet_cells

__all__ = [
    "resolve_axis_order",
    "prepare_heatmap_panel",
    "build_heatmap_figure",
]


# ---------------------------------------------------------------------------
# Axis ordering helper
# ---------------------------------------------------------------------------

def resolve_axis_order(
    df: pd.DataFrame,
    col: str,
    explicit_order: Optional[Sequence[Any]] = None,
) -> list:
    """Determine axis label order for a heatmap dimension.

    Parameters
    ----------
    df : pd.DataFrame
        The full (unfiltered) DataFrame.
    col : str
        Column name whose unique values define the axis.
    explicit_order : sequence, optional
        If provided, use this order exactly. All values must exist in df[col].

    Returns
    -------
    list
        Ordered list of unique values for this axis.

    Raises
    ------
    ValueError
        If explicit_order contains values not present in df[col].
    """
    actual = set(df[col].dropna().unique())

    if explicit_order is not None:
        explicit = list(explicit_order)
        missing = set(explicit) - actual
        if missing:
            raise ValueError(
                f"explicit_order for column {col!r} contains values not found in data: {sorted(missing)}"
            )
        return explicit

    # Auto-infer: sort numerically if possible, else alphabetically
    vals = list(actual)
    try:
        return sorted(vals, key=float)
    except (TypeError, ValueError):
        return sorted(vals, key=str)


# ---------------------------------------------------------------------------
# Single-panel builder
# ---------------------------------------------------------------------------

def prepare_heatmap_panel(
    df: pd.DataFrame,
    *,
    heatmap_row: str,
    heatmap_col: str,
    value_col: str,
    row_order: list,
    col_order: list,
    sig_col: Optional[str] = None,
    sig_threshold: float = 0.05,
    annotation_fmt: Optional[str] = None,
) -> HeatmapData:
    """Build a HeatmapData from a tidy DataFrame slice.

    Independently testable — does not touch faceting or FigureData.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered slice for this panel. Must not have duplicate (heatmap_row, heatmap_col) pairs.
    heatmap_row, heatmap_col : str
        Column names for the y- and x-axes of the heatmap matrix.
    value_col : str
        Column whose values fill the matrix cells.
    row_order, col_order : list
        Globally pre-resolved label orderings (from resolve_axis_order()).
    sig_col : str, optional
        Column for significance. Numeric → compared <= sig_threshold. Bool → used directly.
        Any other dtype raises TypeError.
    sig_threshold : float
        Significance threshold when sig_col is numeric.
    annotation_fmt : str, optional
        Python format string for cell annotations (e.g. '{:.2f}'). If None, no annotations.

    Returns
    -------
    HeatmapData
        Pure data payload with values matrix, labels, optional sig_mask, optional annotations.

    Raises
    ------
    ValueError
        If df contains duplicate (heatmap_row, heatmap_col) pairs.
    TypeError
        If sig_col has an unsupported dtype (not numeric or bool).
    """
    # Validate uniqueness — no silent aggregation
    if df.duplicated(subset=[heatmap_row, heatmap_col]).any():
        dupes = df[df.duplicated(subset=[heatmap_row, heatmap_col], keep=False)]
        raise ValueError(
            f"Duplicate ({heatmap_row!r}, {heatmap_col!r}) pairs found in panel data. "
            f"Either filter the DataFrame or add a facet dimension to split them.\n"
            f"Duplicate rows:\n{dupes[[heatmap_row, heatmap_col, value_col]].head(10)}"
        )

    # Pivot — no aggregation semantics
    pivot = df.pivot(index=heatmap_row, columns=heatmap_col, values=value_col)

    # Reindex to global ordering, filling missing cells with NaN
    pivot = pivot.reindex(index=row_order, columns=col_order)

    values_2d = pivot.to_numpy(dtype=float)

    # --- Significance mask ---
    sig_mask_2d: Optional[np.ndarray] = None
    if sig_col is not None:
        sig_pivot = df.pivot(index=heatmap_row, columns=heatmap_col, values=sig_col)
        sig_pivot = sig_pivot.reindex(index=row_order, columns=col_order)
        raw = sig_pivot.to_numpy()

        if raw.dtype == bool or (hasattr(raw, 'dtype') and raw.dtype.kind == 'b'):
            sig_mask_2d = raw.astype(bool)
        elif np.issubdtype(raw.dtype, np.number) or raw.dtype == object:
            # Try numeric conversion (handles object arrays with numeric values)
            try:
                numeric_raw = raw.astype(float)
                sig_mask_2d = numeric_raw <= sig_threshold
            except (ValueError, TypeError):
                raise TypeError(
                    f"sig_col {sig_col!r} has unsupported dtype {raw.dtype!r}; "
                    f"expected numeric or bool."
                )
        else:
            raise TypeError(
                f"sig_col {sig_col!r} has unsupported dtype {raw.dtype!r}; "
                f"expected numeric or bool."
            )

        # Never mark missing cells as significant
        sig_mask_2d = sig_mask_2d & ~np.isnan(values_2d)

    # --- Annotations ---
    annotations_2d: Optional[np.ndarray] = None
    if annotation_fmt is not None:
        annotations_2d = np.empty(values_2d.shape, dtype=object)
        for r in range(values_2d.shape[0]):
            for c in range(values_2d.shape[1]):
                v = values_2d[r, c]
                annotations_2d[r, c] = annotation_fmt.format(v) if np.isfinite(v) else ''

    return HeatmapData(
        values=values_2d,
        row_labels=[str(v) for v in row_order],
        col_labels=[str(v) for v in col_order],
        sig_mask=sig_mask_2d,
        annotations=annotations_2d,
    )


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------

def build_heatmap_figure(
    df: pd.DataFrame,
    *,
    # Within-panel axes
    heatmap_row: str,
    heatmap_col: str,
    value_col: str,
    # Panel faceting
    facet_row: Optional[str] = None,
    facet_col: Optional[str] = None,
    # Explicit ordering (if None, inferred from data)
    heatmap_row_order: Optional[Sequence[Any]] = None,
    heatmap_col_order: Optional[Sequence[Any]] = None,
    facet_row_order: Optional[Sequence[Any]] = None,
    facet_col_order: Optional[Sequence[Any]] = None,
    # Significance overlay
    sig_col: Optional[str] = None,
    sig_threshold: float = 0.05,
    # Annotations
    annotation_fmt: Optional[str] = None,
    # Figure-level style
    heatmap_style: Optional[HeatmapStyle] = None,
    colorbar: Optional[ColorbarSpec] = None,
    # Labels
    title: str = '',
    subtitle: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
) -> FigureData:
    """Build a FigureData with heatmap panels from a long-form DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form (tidy) data. One row per (heatmap_row, heatmap_col) cell per facet combination.
    heatmap_row : str
        Column whose unique values become y-axis labels in each heatmap panel.
    heatmap_col : str
        Column whose unique values become x-axis labels in each heatmap panel.
    value_col : str
        Column whose values fill the matrix cells (e.g. 'auroc_obs').
    facet_row, facet_col : str, optional
        Columns for panel grid layout.
    heatmap_row_order, heatmap_col_order : sequence, optional
        Explicit ordering for heatmap axes. Inferred from data if None.
    facet_row_order, facet_col_order : sequence, optional
        Explicit ordering for facet axes. Inferred from data if None.
    sig_col : str, optional
        Column for significance marking. Numeric → <= sig_threshold. Bool → used directly.
    sig_threshold : float
        Threshold for numeric sig_col.
    annotation_fmt : str, optional
        Format string for cell text annotations (e.g. '{:.2f}'). Respects heatmap_style.show_annotations.
    heatmap_style : HeatmapStyle, optional
        Shared visual style for all panels. Defaults to HeatmapStyle().
    colorbar : ColorbarSpec, optional
        Colorbar configuration. Pass ColorbarSpec() to enable.
    title : str
        Figure title.
    x_label, y_label : str, optional
        Axis labels for heatmap panels.

    Returns
    -------
    FigureData
        Ready for render().

    Raises
    ------
    ValueError
        - Axis roles are not distinct
        - value_col overlaps with an axis column
        - Required columns are missing
        - explicit_order contains values not in data
    """
    # --- Validate axis roles are distinct ---
    roles: dict[str, str] = {'heatmap_row': heatmap_row, 'heatmap_col': heatmap_col}
    if facet_row:
        roles['facet_row'] = facet_row
    if facet_col:
        roles['facet_col'] = facet_col

    all_cols = list(roles.values())
    if len(all_cols) != len(set(all_cols)):
        dupes = [v for v in all_cols if all_cols.count(v) > 1]
        raise ValueError(
            f"Axis roles must all map to distinct columns, but got duplicates: {dupes}. "
            f"Assignments: {roles}"
        )

    if value_col in set(all_cols):
        raise ValueError(
            f"value_col={value_col!r} cannot also be an axis column. "
            f"Axis columns: {sorted(set(all_cols))}"
        )

    # --- Validate required columns exist ---
    needed = set(all_cols) | {value_col}
    if sig_col:
        needed.add(sig_col)
    missing_cols = needed - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    # --- Resolve global axis orders (before iterating panels) ---
    hm_row_order = resolve_axis_order(df, heatmap_row, heatmap_row_order)
    hm_col_order = resolve_axis_order(df, heatmap_col, heatmap_col_order)

    # --- Determine facet values ---
    row_vals: list[Any] = [None]
    col_vals: list[Any] = [None]
    if facet_row:
        row_vals_seq = facet_row_order if facet_row_order is not None else None
        row_vals = resolve_axis_order(df, facet_row, row_vals_seq)
    if facet_col:
        col_vals_seq = facet_col_order if facet_col_order is not None else None
        col_vals = resolve_axis_order(df, facet_col, col_vals_seq)

    # Determine annotation_fmt to pass: use it if style.show_annotations or explicitly set
    eff_annotation_fmt = annotation_fmt
    hm_style = heatmap_style or HeatmapStyle()
    if hm_style.show_annotations and eff_annotation_fmt is None:
        eff_annotation_fmt = '{:.2f}'

    # --- Iterate facet cells and build panels ---
    subplots: list[SubplotData] = []

    for row_val, col_val, filter_dict, subplot_key in iter_facet_cells(
        facet_row, facet_col, row_vals, col_vals
    ):
        # Filter data for this panel
        mask = pd.Series(True, index=df.index)
        for k, v in filter_dict.items():
            mask &= df[k].astype(str) == str(v)
        cell_df = df.loc[mask].copy()

        # Build title for this panel
        panel_title: Optional[str] = None
        if filter_dict:
            parts = []
            if facet_row and row_val is not None:
                parts.append(str(row_val))
            if facet_col and col_val is not None:
                parts.append(str(col_val))
            if parts:
                panel_title = " | ".join(parts)

        hm_data = prepare_heatmap_panel(
            cell_df,
            heatmap_row=heatmap_row,
            heatmap_col=heatmap_col,
            value_col=value_col,
            row_order=hm_row_order,
            col_order=hm_col_order,
            sig_col=sig_col,
            sig_threshold=sig_threshold,
            annotation_fmt=eff_annotation_fmt,
        )

        subplots.append(SubplotData(
            heatmap=hm_data,
            key=subplot_key,
            title=panel_title,
            x_label=x_label or heatmap_col,
            y_label=y_label or heatmap_row,
        ))

    return FigureData(
        title=title,
        subtitle=subtitle,
        subplots=subplots,
        heatmap_style=hm_style,
        colorbar=colorbar,
    )
