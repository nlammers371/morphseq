"""
Matplotlib renderer for faceted plots.
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Dict, Any

from ..ir import FigureData, TraceData, FacetSpec
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
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    
    legend_entries = {}  # label → color
    
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
            else:
                ax.plot(
                    trace.x, trace.y,
                    color=trace.style.color, alpha=trace.style.alpha,
                    linewidth=trace.style.width, linestyle=trace.style.linestyle,
                    zorder=trace.style.zorder,
                )
            
            # Collect legend based on show_legend (NOT linewidth heuristic)
            if trace.show_legend and trace.label and trace.label not in legend_entries:
                legend_entries[trace.label] = trace.style.color
        
        if not has_data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='lightgray')
        
        if sub.xlim:
            ax.set_xlim(sub.xlim)
        if sub.ylim:
            ax.set_ylim(sub.ylim)
        if sub.title:
            ax.set_title(sub.title, fontweight='bold', fontsize=11)
        if pos['show_x'] and sub.x_label:
            ax.set_xlabel(sub.x_label, fontsize=10)
        if pos['show_y'] and sub.y_label:
            ax.set_ylabel(sub.y_label, fontsize=10)
        
        if style.show_grid:
            ax.grid(True, alpha=style.grid_alpha, linestyle='--', linewidth=0.5)
    
    # Unified legend
    if legend_entries:
        handles = [
            Line2D([0], [0], color=c, linewidth=style.trend_width, label=lbl)
            for lbl, c in legend_entries.items()
        ]
        rightmost_ax = axes[0, -1]
        fig.legend(handles=handles, loc='upper left',
                   bbox_to_anchor=(1.01, 1.0), bbox_transform=rightmost_ax.transAxes,
                   fontsize=style.legend_fontsize, frameon=True, framealpha=0.9)
        plt.subplots_adjust(right=0.82)
    
    if data.row_labels and n_rows > 1:
        for idx, label in enumerate(data.row_labels):
            y_pos = 1 - (idx + 0.5) / n_rows
            fig.text(0.02, y_pos, label, rotation=90, va='center', ha='right',
                     fontsize=12, fontweight='bold', transform=fig.transFigure)

    
    fig.suptitle(data.title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0.03, 0, 0.82 if legend_entries else 1, 0.96])

    if data.col_labels and n_cols > 1:
        for idx, label in enumerate(data.col_labels):
            ax = axes[0, idx]
            pos = ax.get_position()
            x_pos = (pos.x0 + pos.x1) / 2
            y_pos = pos.y1 + 0.01
            fig.text(x_pos, y_pos, label, va='bottom', ha='center',
                     fontsize=12, fontweight='bold', transform=fig.transFigure)
    
    return fig
