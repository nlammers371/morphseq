"""
Classification heatmap visualizations using the generic faceting engine.

This module wraps the generic heatmap primitive with classification-friendly defaults.
- plot_auroc_heatmaps: faceted heatmap of AUROC scores over time
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd

from analyze.viz.plotting.faceting_engine import (
    FacetSpec,
    StyleSpec,
    default_style,
    render,
)
from analyze.viz.plotting.faceting_engine.ir import ColorbarSpec, HeatmapStyle
from analyze.viz.plotting.faceting_engine.heatmap import build_heatmap_figure

from ..engine.analysis import ClassificationAnalysis

__all__ = ["plot_auroc_heatmaps"]


def _as_scores(results: Union[ClassificationAnalysis, pd.DataFrame]) -> pd.DataFrame:
    """Extract scores DataFrame from ClassificationAnalysis or accept plain DataFrame."""
    if isinstance(results, ClassificationAnalysis):
        return results.scores.copy()
    return results.copy()


def _infer_facets(
    df: pd.DataFrame,
    heatmap_row: str,
    heatmap_col: str,
    facet_row: Optional[str],
    facet_col: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """Infer default facet axes if not explicitly set.

    Classification default logic:
    - If both feature_set and negative_label vary AND are not already used → facet_row=feature_set, facet_col=negative_label
    - If only one varies and is not used → facet on the one that varies
    - If neither varies → no faceting
    """
    if facet_row is not None or facet_col is not None:
        return facet_row, facet_col  # caller was explicit; don't override

    in_use = {heatmap_row, heatmap_col}
    candidates = {
        'feature_set': 'facet_row',
        'negative_label': 'facet_col',
    }

    inferred_row: Optional[str] = None
    inferred_col: Optional[str] = None

    for col, role in candidates.items():
        if col not in df.columns or col in in_use:
            continue
        n_unique = df[col].nunique()
        if n_unique > 1:
            if role == 'facet_row':
                inferred_row = col
            else:
                inferred_col = col

    return inferred_row, inferred_col


def plot_auroc_heatmaps(
    results: Union[ClassificationAnalysis, pd.DataFrame],
    *,
    # Heatmap matrix axes
    heatmap_row: str = 'positive_label',
    heatmap_col: str = 'time_bin_center',
    # Panel faceting (None = auto-infer from data)
    facet_row: Optional[str] = None,
    facet_col: Optional[str] = None,
    # Explicit ordering (None = sorted from data)
    heatmap_row_order: Optional[Sequence[Any]] = None,
    heatmap_col_order: Optional[Sequence[Any]] = None,
    facet_row_order: Optional[Sequence[Any]] = None,
    facet_col_order: Optional[Sequence[Any]] = None,
    # Column names
    auroc_col: str = 'auroc_obs',
    pval_col: str = 'pval',
    # Significance
    show_significance: bool = True,
    sig_threshold: float = 0.01,
    # Colormap / scale
    cmap: str = 'BuPu',
    vcenter: Optional[float] = None,
    vmin: float = 0.4,
    vmax: float = 1.0,
    missing_color: str = '#d9d9d9',
    # Annotations
    show_annotations: bool = False,
    annotation_fmt: str = '{:.2f}',
    # Labels
    title: str = 'AUROC Heatmap',
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    colorbar_label: str = 'AUROC',
    # Output
    backend: str = 'matplotlib',
    output_path: Optional[Union[str, Path]] = None,
    style: Optional[StyleSpec] = None,
) -> Any:
    """
    Plot AUROC scores as a faceted heatmap.

    Each panel shows a matrix of (heatmap_row × heatmap_col) cells coloured
    by AUROC value. Significant cells (p ≤ sig_threshold) receive a black border.
    Missing combinations are shown in grey.

    Parameters
    ----------
    results : ClassificationAnalysis | pd.DataFrame
        Classification results. If ClassificationAnalysis, uses .scores.
    heatmap_row : str
        Column for heatmap y-axis (rows within each panel). Default: 'positive_label'.
    heatmap_col : str
        Column for heatmap x-axis (columns within each panel). Default: 'time_bin_center'.
    facet_row, facet_col : str | None
        Columns for the panel grid. If both are None, auto-inferred from data
        (feature_set → rows, negative_label → cols, if they vary).
    auroc_col : str
        Column for AUROC values. Default: 'auroc_obs'.
    pval_col : str
        Column for p-values used in significance overlay. Default: 'pval'.
    show_significance : bool
        Draw black border on cells where p ≤ sig_threshold.
    sig_threshold : float
        Significance threshold. Default: 0.01.
    cmap : str
        Matplotlib colormap name. Default: 'RdBu_r'.
    vcenter : float | None
        If set, uses TwoSlopeNorm to centre the colormap at this value (e.g. 0.5 = chance).
    vmin, vmax : float
        Color scale bounds. Default: 0.3, 1.0.
    show_annotations : bool
        Show AUROC value as text inside each cell. Default: False.
    annotation_fmt : str
        Format string for annotations. Default: '{:.2f}'.
    backend : str
        'matplotlib', 'plotly', or 'both'. Default: 'matplotlib'.
    output_path : str | Path | None
        If provided, save figure to this path.
    style : StyleSpec | None
        Faceting engine style overrides.

    Returns
    -------
    matplotlib.figure.Figure | plotly.graph_objects.Figure | dict
    """
    df = _as_scores(results)

    # --- Validate required columns ---
    required = {heatmap_row, heatmap_col, auroc_col}
    if show_significance:
        required.add(pval_col)
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {sorted(missing_cols)}. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )

    # --- Auto-infer facet axes ---
    eff_facet_row, eff_facet_col = _infer_facets(df, heatmap_row, heatmap_col, facet_row, facet_col)

    # --- Auto-generate subtitle from constant context columns ---
    # Show negative_label when it's not already a facet/heatmap axis and is constant
    subtitle: Optional[str] = None
    axis_cols = {heatmap_row, heatmap_col, eff_facet_row, eff_facet_col} - {None}
    subtitle_parts = []
    for context_col in ('negative_label', 'positive_label'):
        if context_col in df.columns and context_col not in axis_cols:
            unique_vals = df[context_col].dropna().unique()
            if len(unique_vals) == 1:
                subtitle_parts.append(f"{context_col}: {unique_vals[0]}")
    if subtitle_parts:
        subtitle = "  |  ".join(subtitle_parts)

    # --- Build style objects ---
    heatmap_style = HeatmapStyle(
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        vcenter=vcenter,
        missing_color=missing_color,
        show_annotations=show_annotations,
    )
    colorbar = ColorbarSpec(label=colorbar_label)

    # --- Resolve annotation_fmt (only passed if annotations enabled) ---
    eff_annotation_fmt = annotation_fmt if show_annotations else None

    # --- Build FigureData ---
    fig_data = build_heatmap_figure(
        df,
        heatmap_row=heatmap_row,
        heatmap_col=heatmap_col,
        value_col=auroc_col,
        facet_row=eff_facet_row,
        facet_col=eff_facet_col,
        heatmap_row_order=heatmap_row_order,
        heatmap_col_order=heatmap_col_order,
        facet_row_order=facet_row_order,
        facet_col_order=facet_col_order,
        sig_col=pval_col if show_significance else None,
        sig_threshold=sig_threshold,
        annotation_fmt=eff_annotation_fmt,
        heatmap_style=heatmap_style,
        colorbar=colorbar,
        title=title,
        subtitle=subtitle,
        x_label=x_label or heatmap_col,
        y_label=y_label or heatmap_row,
    )

    facet = FacetSpec(sharex=False, sharey=False)
    style = style or default_style()

    return render(fig_data, backend=backend, facet=facet, style=style, output_path=output_path)
