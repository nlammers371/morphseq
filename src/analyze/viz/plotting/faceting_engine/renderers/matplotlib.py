"""
Matplotlib renderer for faceted plots.
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Dict, Any

from ..ir import FigureData, TraceData, TraceStyle, FacetSpec
from ..style.defaults import StyleSpec
from ..utils import calculate_grid_map


def render_matplotlib(
    data: FigureData,
    facet: FacetSpec,
    style: StyleSpec,
) -> plt.Figure:
    """Render FigureData to Matplotlib figure."""
    n_rows, n_cols, positions = calculate_grid_map(data, facet)
    figsize = (5 * n_cols, 4.5 * n_rows)
    # Apply axis sharing. We interpret:
    # - sharex=True: share x across all subplots (common time axis)
    # - sharey=True: share y within each row (so different rows can autoscale independently)
    #
    # Some older matplotlib versions may not accept string share modes; fall back to booleans.
    try:
        sharex = "all" if facet.sharex else False
        sharey = "row" if facet.sharey else False
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False, sharex=sharex, sharey=sharey)
    except TypeError:
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=figsize, squeeze=False, sharex=bool(facet.sharex), sharey=bool(facet.sharey)
        )
    
    legend_entries: Dict[str, Dict[str, Any]] = {}  # label → {'kind': str, 'style': TraceStyle}
    
    for idx, sub in enumerate(data.subplots):
        pos = positions.get(idx)
        if pos is None:
            continue
        
        ax = axes[pos['row'] - 1][pos['col'] - 1]
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
        
        if not has_data:
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

        # Tick label visibility (numbers) — independent of axis title visibility.
        if getattr(style, "repeat_xticklabels", False):
            ax.tick_params(axis="x", which="both", labelbottom=True)
        if getattr(style, "repeat_yticklabels", False):
            ax.tick_params(axis="y", which="both", labelleft=True)
        
        if style.show_grid:
            ax.grid(True, alpha=style.grid_alpha, linestyle='--', linewidth=0.5)
    
    # Unified legend
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

    if data.row_labels and n_rows > 1:
        for idx, label in enumerate(data.row_labels):
            y_pos = 1 - (idx + 0.5) / n_rows
            fig.text(0.02, y_pos, label, rotation=90, va='center', ha='right',
                     fontsize=12, fontweight='bold', transform=fig.transFigure)


    if data.col_labels and n_cols > 1:
        for idx, label in enumerate(data.col_labels):
            ax = axes[0, idx]
            ax.set_title(label, fontsize=12, fontweight='bold')

    fig.suptitle(data.title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0.03, 0, 1, 0.96])

    return fig
