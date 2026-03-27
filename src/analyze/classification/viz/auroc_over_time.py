"""
AUROC-over-time plotting using the generic faceting engine.

This follows the same overall pattern as `analyze.viz.plotting.feature_over_time`:
- Build a backend-agnostic IR (`FigureData` / `SubplotData` / `TraceData`)
- Render with the faceting engine (matplotlib/plotly/both)

Key difference: the input is *already aggregated* (one row per time bin per comparison),
so we do not compute trajectories or trend lines here. We just plot observed AUROC,
optional null bands, and optional significance markers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple

import numpy as np
import pandas as pd

from analyze.viz.styling import (
    STANDARD_PALETTE,
    get_known_genotype_color,
    resolve_color_lookup,
)
from analyze.viz.plotting.faceting_engine import (
    FigureData,
    SubplotData,
    TraceData,
    TraceStyle,
    FacetSpec,
    StyleSpec,
    render,
    default_style,
    iter_facet_cells,
)

from ..results import MulticlassOVRResults

__all__ = ["plot_aurocs_over_time"]


def _as_df(results: Union[MulticlassOVRResults, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(results, MulticlassOVRResults):
        return results.comparisons.copy()
    return results.copy()


def plot_aurocs_over_time(
    results: Union[MulticlassOVRResults, pd.DataFrame],
    *,
    time_col: str = "time_bin_center",
    auroc_col: str = "auroc_obs",
    curve_col: str = "positive",
    facet_row: Optional[str] = None,
    facet_col: Optional[str] = None,
    layout: Optional[FacetSpec] = None,
    # Styling/colors
    color_lookup: Optional[Dict[Any, str]] = None,
    color_palette: Optional[list[str]] = None,
    # Optional overlays
    show_null_band: bool = False,
    null_mean_col: str = "auroc_null_mean",
    null_std_col: str = "auroc_null_std",
    null_band_alpha: float = 0.12,
    show_significance: bool = True,
    pval_col: str = "pval",
    sig_threshold: float = 0.01,
    sig_marker_size: float = 8.0,
    # Baseline
    show_chance_line: bool = True,
    chance_y: float = 0.5,
    chance_label: str = "Chance (0.5)",
    # Labels/ranges
    title: str = "AUROC over time",
    x_label: str = "Hours Post Fertilization (hpf)",
    y_label: str = "AUROC",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Tuple[float, float] = (0.3, 1.05),
    # Output
    backend: str = "plotly",
    output_path: Optional[Union[str, Path]] = None,
    style: Optional[StyleSpec] = None,
) -> Any:
    """
    Plot AUROC curves over time for one or more comparisons, optionally faceted.

    Parameters
    ----------
    results : MulticlassOVRResults | pd.DataFrame
        Either the results object (preferred) or a comparisons DataFrame.
    time_col, auroc_col : str
        Column names for x and y.
    curve_col : str
        Column that identifies which curve a row belongs to (e.g. 'positive').
    facet_row, facet_col : str | None
        Optional faceting columns (e.g. facet_col='negative').
    show_null_band : bool
        If True and null columns exist, draws mean±std band.
    show_significance : bool
        If True and pval column exists, draws open-circle markers where p ≤ sig_threshold.
    show_chance_line : bool
        Adds an AUROC=0.5 baseline line.
    backend : 'plotly' | 'matplotlib' | 'both'

    Returns
    -------
    plotly Figure | matplotlib Figure | dict (both)
    """
    df = _as_df(results)

    required = {time_col, auroc_col, curve_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in AUROC table.")

    df = df[df[time_col].notna() & df[auroc_col].notna()].copy()
    df[curve_col] = df[curve_col].astype(str)
    df = df.sort_values([curve_col, time_col])

    # Determine facet values
    row_vals = [None]
    col_vals = [None]
    if facet_row:
        row_vals = sorted(df[facet_row].dropna().astype(str).unique().tolist()) or [None]
        df[facet_row] = df[facet_row].astype(str)
    if facet_col:
        col_vals = sorted(df[facet_col].dropna().astype(str).unique().tolist()) or [None]
        df[facet_col] = df[facet_col].astype(str)

    # Color mapping for curves
    unique_curves = sorted(df[curve_col].dropna().astype(str).unique().tolist())
    colors = resolve_color_lookup(
        unique_curves,
        color_lookup=color_lookup,
        palette=color_palette or STANDARD_PALETTE,
        default_resolver=get_known_genotype_color,
        enforce_distinct=True,
        warn_on_collision=True,
    )

    facet = layout or FacetSpec()
    style = style or default_style()

    legend_tracker: set[str] = set()
    subplots: list[SubplotData] = []

    for row_val, col_val, filter_dict, subplot_key in iter_facet_cells(
        facet_row, facet_col, row_vals, col_vals
    ):
        mask = pd.Series(True, index=df.index)
        for k, v in filter_dict.items():
            mask &= (df[k].astype(str) == str(v))
        dsub = df.loc[mask].copy()

        traces: list[TraceData] = []

        # Chance line (one legend entry across the whole figure)
        if show_chance_line:
            if xlim is not None:
                x0, x1 = float(xlim[0]), float(xlim[1])
            else:
                x0, x1 = float(dsub[time_col].min()), float(dsub[time_col].max())
            show_legend = chance_label not in legend_tracker
            if show_legend:
                legend_tracker.add(chance_label)
            traces.append(
                TraceData(
                    x=np.array([x0, x1], dtype=float),
                    y=np.array([chance_y, chance_y], dtype=float),
                    style=TraceStyle(color="#7a7a7a", alpha=0.7, width=2.5, linestyle=":", zorder=0),
                    label=chance_label,
                    show_legend=show_legend,
                )
            )

        # Curves
        for curve in unique_curves:
            dcurve = dsub[dsub[curve_col] == curve].copy()
            if dcurve.empty:
                continue

            color = colors.get(curve, STANDARD_PALETTE[0])
            show_legend = curve not in legend_tracker
            if show_legend:
                legend_tracker.add(curve)

            x = dcurve[time_col].to_numpy(dtype=float)
            y = dcurve[auroc_col].to_numpy(dtype=float)
            traces.append(
                TraceData(
                    x=x,
                    y=y,
                    style=TraceStyle(color=color, alpha=1.0, width=2.5, linestyle="-", zorder=3),
                    label=curve,
                    show_legend=show_legend,
                )
            )

            if show_null_band and (null_mean_col in dcurve.columns) and (null_std_col in dcurve.columns):
                m = dcurve[null_mean_col].to_numpy(dtype=float)
                s = dcurve[null_std_col].to_numpy(dtype=float)
                if np.isfinite(m).all() and np.isfinite(s).all():
                    traces.append(
                        TraceData(
                            x=x,
                            y=m,
                            band_lower=m - s,
                            band_upper=m + s,
                            style=TraceStyle(color=color, alpha=null_band_alpha, width=0.0, linestyle="-", zorder=1),
                            show_legend=False,
                            render_as="band",
                        )
                    )

            if show_significance and (pval_col in dcurve.columns):
                p = dcurve[pval_col].to_numpy(dtype=float)
                sig_mask = np.isfinite(p) & (p <= float(sig_threshold))
                if sig_mask.any():
                    traces.append(
                        TraceData(
                            x=x[sig_mask],
                            y=y[sig_mask],
                            style=TraceStyle(
                                color=color,
                                alpha=1.0,
                                width=0.0,
                                linestyle="-",
                                zorder=4,
                                marker="o",
                                marker_size=float(sig_marker_size),
                                marker_facecolor="none",
                                marker_edgecolor=color,
                                marker_edgewidth=2.0,
                            ),
                            show_legend=False,
                            render_as="scatter",
                        )
                    )

        # Significance legend handle (one per figure), only if we are actually drawing sig markers.
        if show_significance:
            sig_label = f"p ≤ {sig_threshold:g}"
            if sig_label not in legend_tracker:
                legend_tracker.add(sig_label)
                traces.append(
                    TraceData(
                        x=np.array([], dtype=float),
                        y=np.array([], dtype=float),
                        style=TraceStyle(
                            color="#000000",
                            alpha=1.0,
                            width=0.0,
                            linestyle="-",
                            zorder=5,
                            marker="o",
                            marker_size=float(sig_marker_size),
                            marker_facecolor="none",
                            marker_edgecolor="#000000",
                            marker_edgewidth=2.0,
                        ),
                        label=sig_label,
                        show_legend=True,
                        render_as="scatter",
                    )
                )

        panel_title = None
        if facet_row or facet_col:
            parts = []
            if facet_row and row_val is not None:
                parts.append(f"{facet_row}={row_val}")
            if facet_col and col_val is not None:
                parts.append(f"{facet_col}={col_val}")
            if parts:
                panel_title = ", ".join(parts)

        subplots.append(
            SubplotData(
                traces=traces,
                key=subplot_key,
                title=panel_title,
                x_label=x_label,
                y_label=y_label,
                xlim=xlim,
                ylim=ylim,
            )
        )

    fig_data = FigureData(title=title, subplots=subplots, legend_title=curve_col)
    return render(fig_data, backend=backend, facet=facet, style=style, output_path=output_path)
