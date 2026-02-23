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

from .ir import TraceData, TraceStyle, SubplotData, FigureData, FacetSpec
from .style.defaults import StyleSpec, default_style, paper_style
from .stats import validate_error_type, compute_error_band, compute_linear_fit
from .utils import iter_facet_cells, calculate_grid_map, compute_figure_size


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
    
    results = {}
    
    if backend in ('plotly', 'both'):
        fig_plotly = render_plotly(fig_data, facet, style)
        if output_path and backend == 'plotly':
            fig_plotly.write_html(str(output_path))
        results['plotly'] = fig_plotly
    
    if backend in ('matplotlib', 'both'):
        fig_mpl = render_matplotlib(fig_data, facet, style)
        if output_path and backend == 'matplotlib':
            fig_mpl.savefig(str(output_path), dpi=150, bbox_inches='tight')
        results['matplotlib'] = fig_mpl
    
    if backend == 'both' and output_path:
        path = Path(output_path)
        results['plotly'].write_html(str(path.with_suffix('.html')))
        results['matplotlib'].savefig(str(path.with_suffix('.png')), dpi=150, bbox_inches='tight')
    
    return results if backend == 'both' else results.get(backend)


__all__ = [
    'TraceData', 'TraceStyle', 'SubplotData', 'FigureData',
    'FacetSpec',
    'StyleSpec', 'default_style', 'paper_style',
    'validate_error_type', 'compute_error_band', 'compute_linear_fit',
    'iter_facet_cells', 'calculate_grid_map', 'compute_figure_size',
    'render',
]
