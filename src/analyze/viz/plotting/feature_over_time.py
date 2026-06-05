"""
Plot feature over time with optional faceting.

100% DOMAIN-AGNOSTIC: No trajectory_analysis imports.
Optional label and color lookups can be supplied by the caller, but the
default behavior stays palette-first.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Any, Union, Dict, Set, Tuple

# Generic imports ONLY
from analyze.utils.data_processing import get_trajectories_for_group, get_global_axis_ranges
from analyze.utils.stats import compute_trend_line
from analyze.viz.styling import (
    STANDARD_PALETTE,
    ColorPreset,
    resolve_color_lookup,
)

# Engine imports
from .faceting_engine import (
    FigureData, SubplotData, TraceData, TraceStyle,
    FacetSpec, StyleSpec, render, default_style,
    iter_facet_cells, compute_error_band, resolve_linestyle,
)


@dataclass(frozen=True)
class IdTraceStyle:
    """Optional per-ID trace emphasis layered on top of group-level plotting."""

    color: Optional[str] = None
    alpha: float = 1.0
    width: float = 2.8
    linestyle: str = "-"
    zorder: int = 8
    label: Optional[str] = None


EmbryoTraceStyle = IdTraceStyle


def _coerce_embryo_trace_style(value: Any) -> IdTraceStyle:
    """Normalize style config from a dataclass or plain dict."""
    if isinstance(value, IdTraceStyle):
        return value
    if isinstance(value, dict):
        return IdTraceStyle(
            color=value.get("color"),
            alpha=float(value.get("alpha", 1.0)),
            width=float(value.get("width", 2.8)),
            linestyle=str(value.get("linestyle", "-")),
            zorder=int(value.get("zorder", 8)),
            label=value.get("label"),
        )
    raise TypeError(
        "id_style_lookup values must be IdTraceStyle instances or dicts with "
        "keys like color/alpha/width/zorder/label."
    )


def _build_color_lookup(
    df: pd.DataFrame,
    color_by: Optional[str],
    label_map: Optional[Dict[Any, str]] = None,
    color_lookup: Optional[Dict[Any, str]] = None,
    palette: Optional[List[str]] = None,
    color_preset: Optional[ColorPreset] = None,
    color_mode: str = "auto",
) -> Dict[Any, str]:
    """Build or use provided color lookup (auto genotype-aware by default).
    
    Private helper. If color_lookup provided, use it.
    Otherwise, resolve via preset or the requested color mode.
    """
    if color_by is None or color_by not in df.columns:
        return {}
    
    unique_vals = list(df[color_by].dropna().unique())
    return resolve_color_lookup(
        unique_vals,
        color_lookup=color_lookup,
        palette=palette or STANDARD_PALETTE,
        color_preset=color_preset,
        color_mode=color_mode,
        label_map=label_map,
    )


def _plot_features_over_time_subplot(
    df: pd.DataFrame,
    filter_dict: Dict[str, Any],
    x_col: str,
    y_col: str,
    line_by: str,
    color_lookup: Dict[Any, str],
    color_by: Optional[str],
    subplot_key: Tuple[Any, Any],
    legend_tracker: Set[str],
    *,
    show_individual: bool = True,
    show_trend: bool = True,
    show_error_band: bool = False,
    error_type: str = 'iqr',
    trend_statistic: str = 'median',
    trend_smooth_sigma: float = 1.5,
    trend_linestyle: str = 'dotted',
    bin_width: float = 0.5,
    smooth_method: Optional[str] = 'gaussian',
    smooth_params: Optional[Dict] = None,
    highlight_ids: Optional[Set[str]] = None,
    id_style_lookup: Optional[Dict[str, IdTraceStyle]] = None,
    label_map: Optional[Dict[Any, str]] = None,
    style: Optional[StyleSpec] = None,
) -> SubplotData:
    """Plot Features Over Time Subplot (internal IR builder for one facet cell)."""
    style = style or default_style()
    # Determine color groups
    if color_by and color_by in df.columns:
        mask = pd.Series(True, index=df.index)
        for k, v in filter_dict.items():
            mask &= (df[k] == v)
        groups_raw = list(df.loc[mask, color_by].dropna().unique())
        if color_preset is not None and color_preset.order:
            groups = [v for v in color_preset.order if v in groups_raw]
            groups.extend([v for v in groups_raw if v not in groups])
        else:
            groups = sorted(groups_raw)
    else:
        groups = [None]
    
    traces: List[TraceData] = []
    highlight_ids = highlight_ids or set()
    id_style_lookup = id_style_lookup or {}
    
    for group_val in groups:
        group_filter = filter_dict.copy()
        if color_by and color_by in df.columns:
            group_filter[color_by] = group_val

        trajectories, _, _ = get_trajectories_for_group(
            df, group_filter,
            time_col=x_col, metric_col=y_col, embryo_id_col=line_by,
            smooth_method=smooth_method, smooth_params=smooth_params,
        )

        if not trajectories:
            continue

        label_value = label_map.get(str(group_val), str(group_val)) if label_map else str(group_val)
        color_key = label_value if label_map else group_val
        color = color_lookup.get(color_key, STANDARD_PALETTE[0])
        
        # Individual traces
        if show_individual:
            for traj in trajectories:
                traces.append(TraceData(
                    x=traj['times'], y=traj['metrics'],
                    style=TraceStyle(
                        color=color,
                        alpha=style.individual_alpha,
                        width=style.individual_width,
                        zorder=2,
                    ),
                    show_legend=False,
                    hover_meta={'header': f"ID: {traj['embryo_id']}", 'detail': f"<b>{y_col}:</b> %{{y:.3f}}"},
                ))
        
        # Legend: show once per group across all subplots
        label = label_value if group_val is not None else trend_statistic
        legend_key = f"{y_col}_{group_val}" if group_val is not None else f"{y_col}_agg"
        show_legend = legend_key not in legend_tracker
        if show_legend:
            legend_tracker.add(legend_key)

        # Aggregated data
        all_times = np.concatenate([t['times'] for t in trajectories])
        all_metrics = np.concatenate([t['metrics'] for t in trajectories])

        # Error band
        if show_error_band:
            band_t, band_c, band_e = compute_error_band(
                all_times, all_metrics, bin_width,
                statistic=trend_statistic, error_type=error_type,
            )
            if band_t is not None:
                traces.append(TraceData(
                    x=band_t, y=band_c,
                    band_lower=band_c - band_e,
                    band_upper=band_c + band_e,
                    style=TraceStyle(color=color, alpha=style.band_alpha, width=0, zorder=3),
                    render_as='band',
                    show_legend=False,
                ))
        
        # Trend line
        trend_t, trend_v = compute_trend_line(
            all_times, all_metrics, bin_width,
            statistic=trend_statistic, smooth_sigma=trend_smooth_sigma,
        )
        if show_trend and trend_t is not None and len(trend_t) > 0:
            mpl_ls, _ = resolve_linestyle(trend_linestyle)
            traces.append(TraceData(
                x=np.array(trend_t), y=np.array(trend_v),
                style=TraceStyle(color=color, alpha=style.trend_alpha, width=style.trend_width, linestyle=mpl_ls, zorder=5),
                label=label,
                legend_group=legend_key,
                show_legend=show_legend,
                hover_meta={'header': f"{trend_statistic.capitalize()}: {label}", 'detail': f"<b>{y_col}:</b> %{{y:.3f}}"},
            ))

        # Emphasized embryo traces are drawn last so they sit on top of the
        # normal individual traces and group-level summaries.
        for traj in trajectories:
            embryo_id = str(traj['embryo_id'])
            if embryo_id not in highlight_ids and embryo_id not in id_style_lookup:
                continue

            style_override = id_style_lookup.get(embryo_id, IdTraceStyle())
            highlight_color = style_override.color or color
            mpl_ls, _ = resolve_linestyle(style_override.linestyle)

            highlight_label = style_override.label
            show_highlight_legend = False
            legend_group = None
            if highlight_label:
                legend_group = f"highlight_{highlight_label}"
                if legend_group not in legend_tracker:
                    legend_tracker.add(legend_group)
                    show_highlight_legend = True

            traces.append(
                TraceData(
                    x=traj['times'],
                    y=traj['metrics'],
                    style=TraceStyle(
                        color=highlight_color,
                        alpha=float(style_override.alpha),
                        width=float(style_override.width),
                        linestyle=mpl_ls,
                        zorder=int(style_override.zorder),
                    ),
                    label=highlight_label,
                    legend_group=legend_group,
                    show_legend=show_highlight_legend,
                    hover_meta={'header': f"ID: {embryo_id}", 'detail': f"<b>{y_col}:</b> %{{y:.3f}}"},
                )
            )
    
    return SubplotData(
        key=subplot_key,
        traces=traces,
        x_label=x_col,
        y_label=y_col,
    )


def plot_feature_over_time(
    df: pd.DataFrame,
    features: Optional[Union[str, List[str]]] = None,  # Can be single feature or list of features
    time_col: str = 'predicted_stage_hpf',
    id_col: str = 'embryo_id',
    color_by: Optional[str] = None,
    color_lookup: Optional[Dict[Any, str]] = None,  # ← USER PROVIDES domain-specific colors
    label_map: Optional[Dict[Any, str]] = None,
    color_preset: Optional[ColorPreset] = None,
    color_mode: str = 'auto',
    # Faceting (consistent API)
    facet_row: Optional[str] = None,
    facet_col: Optional[str] = None,
    layout: Optional[FacetSpec] = None,
    # Display
    show_individual: bool = True,
    show_trend: bool = True,
    show_error_band: bool = False,
    error_type: str = 'iqr',
    trend_statistic: str = 'median',
    trend_smooth_sigma: float = 1.5,
    trend_linestyle: str = 'dotted',  # 'solid', 'dashed', 'dotted' (or '-', '--', ':')
    bin_width: float = 0.5,
    smooth_method: Optional[str] = 'gaussian',
    smooth_params: Optional[Dict] = None,
    # Output
    backend: str = 'plotly',
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    style: Optional[StyleSpec] = None,
    color_palette: Optional[List[str]] = None,  # Generic fallback
    feature: Optional[Union[str, List[str]]] = None,  # Backward-compatible alias for features
    # Label/tick visibility controls (API-level; avoids needing to import StyleSpec)
    repeat_xlabels: bool = False,
    repeat_ylabels: bool = False,
    repeat_xticklabels: bool = True,
    repeat_yticklabels: bool = True,
    # Legend placement: any matplotlib loc string (e.g. 'upper right', 'lower left'),
    # or 'outside' to place the legend to the right of the axes.
    legend_loc: str = 'upper right',
    # Manual axis limits (blanket applied to all subplots)
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    include_ids: Optional[List[str]] = None,
    exclude_ids: Optional[List[str]] = None,
    highlight_ids: Optional[List[str]] = None,
    id_style_lookup: Optional[Dict[str, Union[IdTraceStyle, Dict[str, Any]]]] = None,
) -> Any:
    """Plot feature(s) over time, optionally faceted.
    
    100% DOMAIN-AGNOSTIC: Defaults are genotype-aware via the shared resolver,
    but callers can supply an explicit `color_lookup` or `color_preset`.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with time-series data
    features : str or List[str]
        Column name(s) for y-axis metric(s)
    feature : str or List[str], optional
        Alias for `features` (backward-compatible keyword)
    time_col : str, default='predicted_stage_hpf'
        Column name for x-axis (time)
    id_col : str, default='embryo_id'
        Column name for trajectory identifiers
    color_by : str, optional
        Column to color traces by
    color_lookup : Dict[Any, str], optional
        Pre-built mapping from values in color_by column to hex colors.
        Use this to inject domain-specific coloring (e.g., genotype colors).
    color_preset : ColorPreset, optional
        Explicit reusable color preset object. This is the preferred path for
        project palettes and talk figures.
    color_mode : str, default='auto'
        Fallback color strategy when no preset is supplied. Use 'auto' or
        'genotype' for genotype-aware defaults, or 'palette' for generic
        palette-first behavior.
    facet_row : str, optional
        Column to facet by rows
    facet_col : str, optional
        Column to facet by columns
    layout : FacetSpec, optional
        Layout specification (row_order, col_order, sharex, sharey, etc.)
    show_individual : bool, default=True
        Whether to show individual trajectory lines
    show_error_band : bool, default=False
        Whether to show error band around trend line
    error_type : str, default='iqr'
        Error measure ('sd'/'se' for mean, 'iqr'/'mad' for median)
    trend_statistic : str, default='median'
        Central tendency statistic ('mean' or 'median')
    trend_smooth_sigma : float, default=1.5
        Gaussian smoothing sigma for trend line
    bin_width : float, default=0.5
        Bin width for aggregating trend line
    smooth_method : str, optional, default='gaussian'
        Smoothing method for individual trajectories
    smooth_params : dict, optional
        Parameters for smoothing method
    backend : str, default='plotly'
        Rendering backend ('plotly', 'matplotlib', or 'both')
    output_path : str or Path, optional
        Path to save figure
    title : str, optional
        Figure title
    style : StyleSpec, optional
        Style specification
    color_palette : List[str], optional
        Fallback palette if color_lookup not provided. Uses STANDARD_PALETTE if None.
    repeat_xlabels, repeat_ylabels : bool
        Repeat axis *titles* (e.g. "curvature") on every subplot. Default False.
    repeat_xticklabels, repeat_yticklabels : bool
        Repeat tick-label *numbers* on every subplot (useful with shared axes). Default True.
    xlim, ylim : (float, float), optional
        If provided, apply these limits to all subplots (useful for consistent scaling).
    include_ids, exclude_ids : list[str], optional
        Optional embryo ID filters applied before trajectory extraction.
    highlight_ids : list[str], optional
        Embryo IDs to emphasize with a second drawing pass.
    id_style_lookup : dict[str, IdTraceStyle | dict], optional
        Per-embryo styling overrides for highlighted traces. Supported fields:
        `color`, `alpha`, `width`, `linestyle`, `zorder`, `label`.
    
    Returns
    -------
    Figure
        Plotly or matplotlib figure (or dict if backend='both')
    """
    if feature is not None:
        if features is not None and features != feature:
            raise ValueError("Provide only one of `features` or `feature` (alias).")
        features = feature
    if features is None:
        raise ValueError("Missing required argument: `features` (alias: `feature`).")

    layout = layout or FacetSpec(row_order=None, col_order=None)
    style = style or default_style()
    # Plot-level API knobs. These are independent of axis sharing; sharing controls scaling,
    # while these control axis-title and tick-label visibility.
    style.repeat_xlabels = bool(repeat_xlabels)
    style.repeat_ylabels = bool(repeat_ylabels)
    style.repeat_xticklabels = bool(repeat_xticklabels)
    style.repeat_yticklabels = bool(repeat_yticklabels)
    style.legend_loc = legend_loc

    # Handle multi-feature: if features is a list, treat each as a row facet (no fake column)
    if isinstance(features, (list, tuple)):
        feature_list = list(features)
        facet_row_for_filter = None
    else:
        feature_list = [features]
        facet_row_for_filter = facet_row

    # Optional embryo-level filtering is separate from group-level coloring.
    if include_ids is not None:
        include_id_set = {str(x) for x in include_ids}
        df = df[df[id_col].astype(str).isin(include_id_set)].copy()
    if exclude_ids is not None:
        exclude_id_set = {str(x) for x in exclude_ids}
        df = df[~df[id_col].astype(str).isin(exclude_id_set)].copy()

    normalized_highlight_ids: Set[str] = {str(x) for x in highlight_ids} if highlight_ids is not None else set()
    normalized_id_styles: Dict[str, IdTraceStyle] = {}
    if id_style_lookup:
        normalized_id_styles = {str(k): _coerce_embryo_trace_style(v) for k, v in id_style_lookup.items()}
        normalized_highlight_ids.update(normalized_id_styles.keys())

    # Build color lookup (explicit lookup wins, then preset, then mode)
    color_lookup = _build_color_lookup(
        df,
        color_by,
        label_map,
        color_lookup,
        color_palette,
        color_preset=color_preset,
        color_mode=color_mode,
    )

    # Determine facet values
    if isinstance(features, (list, tuple)):
        # Multi-feature mode: rows are feature names
        row_vals = feature_list
    else:
        row_vals = (layout.row_order if layout.row_order
                    else sorted(df[facet_row].dropna().unique()) if facet_row
                    else [None])

    col_vals = (layout.col_order if layout.col_order
                else sorted(df[facet_col].dropna().unique()) if facet_col
                else [None])
    
    # Compile subplots using engine's grid iterator
    legend_tracker: Set[str] = set()
    subplots = []
    
    for row_val, col_val, filter_dict, subplot_key in iter_facet_cells(
        facet_row_for_filter, facet_col, row_vals, col_vals
    ):
        # In multi-feature mode, row_val is the feature name
        if isinstance(features, (list, tuple)):
            current_feature = row_val
        else:
            current_feature = feature_list[0]

        subplot = _plot_features_over_time_subplot(
            df=df,
            filter_dict=filter_dict,
            x_col=time_col,
            y_col=current_feature,
            line_by=id_col,
            color_lookup=color_lookup,
            color_by=color_by,
            subplot_key=subplot_key,
            legend_tracker=legend_tracker,
            show_individual=show_individual,
            show_trend=show_trend,
            show_error_band=show_error_band,
            error_type=error_type,
            trend_statistic=trend_statistic,
            trend_smooth_sigma=trend_smooth_sigma,
            trend_linestyle=trend_linestyle,
            bin_width=bin_width,
            smooth_method=smooth_method,
            smooth_params=smooth_params,
            highlight_ids=normalized_highlight_ids,
            id_style_lookup=normalized_id_styles,
            label_map=label_map,
            style=style,
        )
        if xlim is not None:
            subplot.xlim = tuple(float(v) for v in xlim)
        if ylim is not None:
            subplot.ylim = tuple(float(v) for v in ylim)
        subplots.append(subplot)

    # Title
    if isinstance(features, (list, tuple)):
        feature_title = ', '.join(feature_list)
    else:
        feature_title = features

    # Assemble FigureData with facet labels
    # For multi-feature mode, use feature names as row labels
    if isinstance(features, (list, tuple)):
        row_labels = feature_list
    else:
        row_labels = [str(v) for v in row_vals if v is not None] if facet_row else None

    col_labels = [str(v) for v in col_vals if v is not None] if facet_col else None

    fig_data = FigureData(
        title=title or f"{feature_title} over {time_col}",
        subplots=subplots,
        legend_title=color_by,
        row_labels=row_labels,
        col_labels=col_labels,
    )

    # Render via engine
    return render(fig_data, backend=backend, facet=layout, style=style, output_path=output_path)
