"""
Plotly renderer for faceted plots.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any

from ..ir import FigureData, TraceData, FacetSpec
from ..style.defaults import StyleSpec
from analyze.viz.styling.color_utils import to_rgba_string
from ..utils import calculate_grid_map, compute_figure_size


def render_plotly(
    data: FigureData,
    facet: FacetSpec,
    style: StyleSpec,
) -> go.Figure:
    """Render FigureData to Plotly figure."""
    n_rows, n_cols, positions = calculate_grid_map(data, facet)
    height, width = compute_figure_size(n_rows, n_cols, style)
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        vertical_spacing=style.vertical_spacing,
        horizontal_spacing=style.horizontal_spacing,
    )
    
    for idx, sub in enumerate(data.subplots):
        pos = positions.get(idx)
        if pos is None:
            continue
        
        for trace in sub.traces:
            if trace.render_as == 'band' and trace.band_lower is not None:
                _add_band_trace(fig, trace, pos['row'], pos['col'], style)
            else:
                _add_line_trace(fig, trace, pos['row'], pos['col'])
        
        # Axis config
        x_title = sub.x_label if pos['show_x'] else None
        y_title = sub.y_label if pos['show_y'] else None
        
        fig.update_xaxes(range=sub.xlim, title_text=x_title, row=pos['row'], col=pos['col'])
        fig.update_yaxes(range=sub.ylim, title_text=y_title, row=pos['row'], col=pos['col'])
    
    _add_facet_labels(fig, data, n_rows, n_cols)
    
    extra_left = 35 if data.row_labels and n_rows > 1 else 0
    extra_top = 35 if data.col_labels and n_cols > 1 else 0
    
    legend_config = dict(x=1.02, y=1)
    if data.legend_title:
        legend_config['title'] = dict(text=data.legend_title)
    
    fig.update_layout(
        title_text=data.title,
        height=height,
        width=width,
        hovermode='closest',
        template='plotly_white',
        legend=legend_config,
        margin=dict(l=style.margin_left + extra_left, r=style.margin_right,
                    t=style.margin_top + extra_top, b=style.margin_bottom),
    )
    
    return fig


def _add_line_trace(fig, trace: TraceData, row: int, col: int):
    """Add a line trace."""
    line_dash = 'dash' if trace.style.linestyle == '--' else 'solid'
    
    hovertemplate = None
    if trace.hover_meta:
        header = trace.hover_meta.get('header', '')
        detail = trace.hover_meta.get('detail', '')
        # Don't use f-string for the whole template - detail already contains Plotly template syntax
        hovertemplate = '<b>' + header + '</b><br><b>Time:</b> %{x:.2f}<br>' + detail + '<extra></extra>'
    
    fig.add_trace(
        go.Scatter(
            x=trace.x, y=trace.y,
            mode='lines',
            line=dict(color=trace.style.color, width=trace.style.width, dash=line_dash),
            opacity=trace.style.alpha,
            name=trace.label,
            legendgroup=trace.legend_group,
            showlegend=trace.show_legend,
            hovertemplate=hovertemplate,
        ),
        row=row, col=col
    )


def _add_band_trace(fig, trace: TraceData, row: int, col: int, style: StyleSpec):
    """Add a fill band trace (upper + lower with fill)."""
    # Upper bound (invisible)
    fig.add_trace(
        go.Scatter(
            x=trace.x, y=trace.band_upper,
            mode='lines', line=dict(width=0),
            showlegend=False, hoverinfo='skip',
        ),
        row=row, col=col
    )
    
    # Lower bound with fill — uses to_rgba_string (robust)
    fill_color = to_rgba_string(trace.style.color, style.band_alpha)
    fig.add_trace(
        go.Scatter(
            x=trace.x, y=trace.band_lower,
            mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor=fill_color,
            showlegend=False, hoverinfo='skip',
        ),
        row=row, col=col
    )


def _add_facet_labels(fig, data: FigureData, n_rows: int, n_cols: int):
    """Add row/column facet strip labels as annotations."""
    if data.row_labels and n_rows > 1:
        for idx, label in enumerate(data.row_labels, start=1):
            y_pos = 1 - (idx - 0.5) / n_rows
            fig.add_annotation(
                text=f"<b>{label}</b>",
                xref="paper", yref="paper",
                x=-0.08, y=y_pos,
                showarrow=False, xanchor="center", yanchor="middle",
                textangle=-90, font=dict(size=13),
            )
    
    if data.col_labels and n_cols > 1:
        for idx, label in enumerate(data.col_labels, start=1):
            x_pos = (idx - 0.5) / n_cols
            fig.add_annotation(
                text=f"<b>{label}</b>",
                xref="paper", yref="paper",
                x=x_pos, y=1.04,
                showarrow=False, xanchor="center", yanchor="bottom",
                font=dict(size=13),
            )
