"""
Matplotlib renderer for faceted plots.
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from typing import Any, Dict, Optional, Tuple

# Matplotlib's built-in diverging colormaps (both forward and reversed _r variants).
# Used to warn when vcenter is set with a non-diverging colormap.
_DIVERGING_CMAPS = {
    'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
    'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic', 'berlin',
    'managua', 'vanimo',
}
_DIVERGING_CMAPS |= {f'{c}_r' for c in _DIVERGING_CMAPS}

from ..ir import FigureData, HeatmapData, HeatmapStyle, ColorbarSpec, TraceData, TraceStyle, FacetSpec
from ..style.defaults import StyleSpec
from ..utils import calculate_grid_map, compute_figure_size


# ---------------------------------------------------------------------------
# Heatmap helpers
# ---------------------------------------------------------------------------

def _resolve_heatmap_range(data: FigureData) -> Tuple[Optional[float], Optional[float]]:
    """Compute shared vmin/vmax across all heatmap panels from data."""
    all_vals = []
    for sub in data.subplots:
        if sub.heatmap is not None:
            finite = sub.heatmap.values[np.isfinite(sub.heatmap.values)]
            if len(finite) > 0:
                all_vals.append(finite)
    if not all_vals:
        return None, None
    combined = np.concatenate(all_vals)
    return float(np.nanmin(combined)), float(np.nanmax(combined))


def _build_norm(
    hm_style: HeatmapStyle,
    vmin: Optional[float],
    vmax: Optional[float],
) -> Optional[mcolors.Normalize]:
    """Build matplotlib Normalize object from HeatmapStyle."""
    effective_vmin = hm_style.vmin if hm_style.vmin is not None else vmin
    effective_vmax = hm_style.vmax if hm_style.vmax is not None else vmax

    if hm_style.vcenter is not None:
        v0 = effective_vmin if effective_vmin is not None else 0.0
        v1 = effective_vmax if effective_vmax is not None else 1.0
        if not (v0 < hm_style.vcenter < v1):
            raise ValueError(
                f"HeatmapStyle.vcenter={hm_style.vcenter} must satisfy vmin < vcenter < vmax, "
                f"but got vmin={v0}, vmax={v1}."
            )
        if hm_style.cmap not in _DIVERGING_CMAPS:
            warnings.warn(
                f"vcenter is set but cmap={hm_style.cmap!r} is not a recognized diverging colormap. "
                f"TwoSlopeNorm will be applied, but the result may look misleading. "
                f"Consider a diverging colormap such as 'RdBu_r', 'coolwarm', or 'PuOr'. "
                f"Pass vcenter=None to suppress this warning.",
                UserWarning,
                stacklevel=3,
            )
        return mcolors.TwoSlopeNorm(vcenter=hm_style.vcenter, vmin=v0, vmax=v1)
    elif effective_vmin is not None and effective_vmax is not None:
        return mcolors.Normalize(vmin=effective_vmin, vmax=effective_vmax)
    return None


def _render_heatmap_panel(
    ax: plt.Axes,
    hm: HeatmapData,
    hm_style: HeatmapStyle,
    norm: Optional[mcolors.Normalize],
) -> Any:
    """Render a single heatmap panel onto ax. Returns the AxesImage for the colorbar."""
    cmap = plt.get_cmap(hm_style.cmap).copy()
    cmap.set_bad(color=hm_style.missing_color)

    im = ax.imshow(
        hm.values,
        aspect='auto',
        interpolation='nearest',
        cmap=cmap,
        norm=norm,
    )

    # Categorical tick labels
    ax.set_xticks(range(len(hm.col_labels)))
    ax.set_xticklabels(hm.col_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(hm.row_labels)))
    ax.set_yticklabels(hm.row_labels, fontsize=9)

    # Remove spines — they look odd alongside the matrix
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Significance overlay: border on significant cells, with optional halo for contrast
    if hm.sig_mask is not None:
        for r in range(hm.sig_mask.shape[0]):
            for c in range(hm.sig_mask.shape[1]):
                if hm.sig_mask[r, c]:
                    # Halo (wider, drawn first so main border sits on top)
                    if hm_style.sig_halo_color is not None:
                        ax.add_patch(mpatches.Rectangle(
                            (c - 0.5, r - 0.5), 1, 1,
                            linewidth=hm_style.sig_halo_width,
                            edgecolor=hm_style.sig_halo_color,
                            facecolor='none',
                            zorder=3,
                        ))
                    # Main border
                    ax.add_patch(mpatches.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        linewidth=hm_style.sig_border_width,
                        edgecolor=hm_style.sig_border_color,
                        facecolor='none',
                        zorder=4,
                    ))

    # Optional text annotations
    if hm_style.show_annotations and hm.annotations is not None:
        for r in range(hm.annotations.shape[0]):
            for c in range(hm.annotations.shape[1]):
                txt = str(hm.annotations[r, c])
                if txt:
                    ax.text(
                        c, r, txt,
                        ha='center', va='center',
                        fontsize=hm_style.annotation_fontsize,
                        color='black',
                    )

    return im


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------

def render_matplotlib(
    data: FigureData,
    facet: FacetSpec,
    style: StyleSpec,
) -> plt.Figure:
    """Render FigureData to Matplotlib figure."""
    # Validate invariants for heatmap figures
    has_heatmaps = any(sub.heatmap is not None for sub in data.subplots)
    has_traces = any(len(sub.traces) > 0 for sub in data.subplots)

    if has_heatmaps and has_traces:
        raise ValueError(
            "Mixed trace/heatmap figures are not supported. "
            "All subplots must be the same geometry (all traces or all heatmaps)."
        )
    if has_heatmaps and data.heatmap_style is None:
        raise ValueError(
            "FigureData.heatmap_style must be set when any subplot contains heatmap data."
        )

    n_rows, n_cols, positions = calculate_grid_map(data, facet)

    if has_heatmaps:
        # Heatmap figures: use constrained_layout for proper colorbar placement
        # Extra width reserved for colorbar
        height, width = compute_figure_size(n_rows, n_cols, style)
        figsize = (width / 100.0 + 1.5, height / 100.0)
        # Disable axis sharing for heatmaps (each panel has its own categorical axes)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=figsize,
            squeeze=False, sharex=False, sharey=False,
            constrained_layout=True,
        )
    else:
        height, width = compute_figure_size(n_rows, n_cols, style)
        figsize = (width / 100.0, height / 100.0)
        try:
            sharex = "all" if facet.sharex else False
            sharey = "row" if facet.sharey else False
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False, sharex=sharex, sharey=sharey)
        except TypeError:
            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=figsize, squeeze=False,
                sharex=bool(facet.sharex), sharey=bool(facet.sharey),
            )

    legend_entries: Dict[str, Dict[str, Any]] = {}  # label → {'kind': str, 'style': TraceStyle}
    last_im = None  # track last AxesImage for colorbar

    # Build heatmap norm once, shared across all panels
    hm_norm = None
    if has_heatmaps:
        shared_vmin, shared_vmax = _resolve_heatmap_range(data)
        hm_norm = _build_norm(data.heatmap_style, shared_vmin, shared_vmax)

    for idx, sub in enumerate(data.subplots):
        pos = positions.get(idx)
        if pos is None:
            continue

        ax = axes[pos['row'] - 1][pos['col'] - 1]

        if sub.heatmap is not None:
            # Heatmap panel
            im = _render_heatmap_panel(ax, sub.heatmap, data.heatmap_style, hm_norm)
            last_im = im
            has_data = True
        else:
            # Trace panel (existing logic)
            has_data = False

            for trace in sub.traces:
                has_data = True

                if trace.render_as == 'band' and trace.band_lower is not None:
                    ax.fill_between(
                        trace.x, trace.band_lower, trace.band_upper,
                        color=trace.style.color, alpha=trace.style.alpha,
                        zorder=trace.style.zorder,
                    )
                elif trace.render_as == 'scatter':
                    edge = trace.style.marker_edgecolor or trace.style.color
                    ax.scatter(
                        trace.x,
                        trace.y,
                        s=float(trace.style.marker_size) ** 2,
                        marker=trace.style.marker,
                        facecolors=trace.style.marker_facecolor,
                        edgecolors=edge,
                        linewidths=float(trace.style.marker_edgewidth),
                        alpha=trace.style.alpha,
                        zorder=trace.style.zorder,
                    )
                else:
                    ax.plot(
                        trace.x, trace.y,
                        color=trace.style.color, alpha=trace.style.alpha,
                        linewidth=trace.style.width, linestyle=trace.style.linestyle,
                        zorder=trace.style.zorder,
                    )

                # Collect legend based on show_legend (NOT linewidth heuristic)
                if trace.show_legend and trace.label and trace.label not in legend_entries:
                    kind = 'scatter' if trace.render_as == 'scatter' else 'line'
                    legend_entries[trace.label] = {'kind': kind, 'style': trace.style}

        if not has_data and sub.heatmap is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='lightgray')

        if sub.xlim:
            ax.set_xlim(sub.xlim)
        if sub.ylim:
            ax.set_ylim(sub.ylim)
        if sub.title:
            ax.set_title(sub.title, fontweight='bold', fontsize=11)

        show_x = bool(pos['show_x'] or style.repeat_xlabels)
        show_y = bool(pos['show_y'] or style.repeat_ylabels)
        if show_x and sub.x_label:
            if style.axis_label_fontsize is None:
                ax.set_xlabel(sub.x_label)
            else:
                ax.set_xlabel(sub.x_label, fontsize=style.axis_label_fontsize)
        if show_y and sub.y_label:
            if style.axis_label_fontsize is None:
                ax.set_ylabel(sub.y_label)
            else:
                ax.set_ylabel(sub.y_label, fontsize=style.axis_label_fontsize)

        if not has_heatmaps:
            # Tick label visibility — only meaningful for trace-based panels
            if getattr(style, "repeat_xticklabels", False):
                ax.tick_params(axis="x", which="both", labelbottom=True)
            if getattr(style, "repeat_yticklabels", False):
                ax.tick_params(axis="y", which="both", labelleft=True)

            if style.show_grid:
                ax.grid(True, alpha=style.grid_alpha, linestyle='--', linewidth=0.5)

    # Shared colorbar for heatmap figures
    if last_im is not None and data.colorbar is not None:
        cb_spec: ColorbarSpec = data.colorbar
        fig.colorbar(
            last_im,
            ax=axes.ravel().tolist(),
            label=cb_spec.label,
            shrink=cb_spec.shrink,
            aspect=cb_spec.aspect,
            pad=cb_spec.pad,
        )

    # Unified legend (trace figures only)
    if legend_entries:
        handles = []
        for lbl, entry in legend_entries.items():
            kind = entry['kind']
            tstyle: TraceStyle = entry['style']
            if kind == 'scatter':
                edge = tstyle.marker_edgecolor or tstyle.color
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=edge,
                        marker=tstyle.marker,
                        markersize=float(tstyle.marker_size),
                        markerfacecolor=tstyle.marker_facecolor,
                        markeredgewidth=float(tstyle.marker_edgewidth),
                        linestyle='None',
                        label=lbl,
                    )
                )
            else:
                handles.append(Line2D([0], [0], color=tstyle.color, linewidth=style.trend_width, label=lbl))
        rightmost_ax = axes[0, -1]
        legend_loc = getattr(style, 'legend_loc', 'upper right')
        if legend_loc == 'outside':
            fig.legend(handles=handles, loc='upper left',
                       bbox_to_anchor=(1.01, 1.0), bbox_transform=rightmost_ax.transAxes,
                       fontsize=style.legend_fontsize, frameon=True, framealpha=0.9)
            plt.subplots_adjust(right=0.82)
        else:
            rightmost_ax.legend(handles=handles, loc=legend_loc,
                                fontsize=style.legend_fontsize, frameon=True, framealpha=0.9)

    if data.row_labels:
        if has_heatmaps:
            # Heatmap figures use constrained_layout — draw row-strip labels as
            # right-side annotations on the rightmost axis in each row.
            for idx, label in enumerate(data.row_labels):
                ax_right = axes[idx, -1]
                ax_right.annotate(
                    label,
                    xy=(1.02, 0.5), xycoords='axes fraction',
                    rotation=270, va='center', ha='left',
                    fontsize=11, fontweight='bold',
                    annotation_clip=False,
                )
        elif n_rows > 1:
            for idx, label in enumerate(data.row_labels):
                y_pos = 1 - (idx + 0.5) / n_rows
                fig.text(0.02, y_pos, label, rotation=90, va='center', ha='right',
                         fontsize=12, fontweight='bold', transform=fig.transFigure)

    if data.col_labels:
        if has_heatmaps:
            # Heatmap figures: draw col-strip labels above the top row axes.
            for idx, label in enumerate(data.col_labels):
                ax_top = axes[0, idx]
                ax_top.annotate(
                    label,
                    xy=(0.5, 1.02), xycoords='axes fraction',
                    va='bottom', ha='center',
                    fontsize=11, fontweight='bold',
                    annotation_clip=False,
                )
        elif n_cols > 1:
            for idx, label in enumerate(data.col_labels):
                ax = axes[0, idx]
                ax.set_title(label, fontsize=12, fontweight='bold')

    if data.subtitle:
        fig.suptitle(
            f"{data.title}\n{data.subtitle}",
            fontsize=14, fontweight='bold',
            linespacing=1.6,
        )
    else:
        fig.suptitle(data.title, fontsize=14, fontweight='bold')

    if not has_heatmaps:
        plt.tight_layout(rect=[0.03, 0, 1, 0.96])

    return fig
