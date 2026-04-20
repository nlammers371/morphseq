"""
Faceting Engine - Generic faceted plotting infrastructure.

Usage:
    from analyze.viz.plotting.faceting_engine import (
        FigureData, SubplotData, TraceData, TraceStyle,
        FacetSpec, StyleSpec, render,
    )
"""

from pathlib import Path
from typing import Optional, Union, Any

from .ir import (
    TraceData, TraceStyle, SubplotData, FigureData, FacetSpec, resolve_linestyle,
    HeatmapData, HeatmapStyle, ColorbarSpec,
)
from .style.defaults import StyleSpec, default_style, paper_style, presentation_style, dense_facet_style
from .stats import validate_error_type, compute_error_band, compute_linear_fit
from .utils import iter_facet_cells, calculate_grid_map, compute_figure_size
from .heatmap import resolve_axis_order, prepare_heatmap_panel, build_heatmap_figure


def render(
    fig_data: FigureData,
    backend: str = 'plotly',
    facet: Optional[FacetSpec] = None,
    style: Optional[StyleSpec] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Any:
    """Render FigureData to the specified backend."""
    from .renderers.plotly import render_plotly
    from .renderers.matplotlib import render_matplotlib
    
    style = style or default_style()
    facet = facet or FacetSpec()
    out_path = Path(output_path) if output_path is not None else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    if backend in ('plotly', 'both'):
        fig_plotly = render_plotly(fig_data, facet, style)
        if out_path is not None and backend == 'plotly':
            fig_plotly.write_html(str(out_path))
        results['plotly'] = fig_plotly
    
    if backend in ('matplotlib', 'both'):
        fig_mpl = render_matplotlib(fig_data, facet, style)
        if out_path is not None and backend == 'matplotlib':
            fig_mpl.savefig(str(out_path), dpi=150, bbox_inches='tight')
        results['matplotlib'] = fig_mpl
    
    if backend == 'both' and out_path is not None:
        results['plotly'].write_html(str(out_path.with_suffix('.html')))
        results['matplotlib'].savefig(str(out_path.with_suffix('.png')), dpi=150, bbox_inches='tight')
    
    return results if backend == 'both' else results.get(backend)


__all__ = [
    # Trace-based IR
    'TraceData', 'TraceStyle', 'SubplotData', 'FigureData',
    'FacetSpec',
    'resolve_linestyle',
    # Heatmap IR
    'HeatmapData', 'HeatmapStyle', 'ColorbarSpec',
    # Style
    'StyleSpec', 'default_style', 'paper_style', 'presentation_style', 'dense_facet_style',
    # Stats
    'validate_error_type', 'compute_error_band', 'compute_linear_fit',
    # Layout utilities
    'iter_facet_cells', 'calculate_grid_map', 'compute_figure_size',
    # Heatmap builder
    'resolve_axis_order', 'prepare_heatmap_panel', 'build_heatmap_figure',
    # Render
    'render',
]
