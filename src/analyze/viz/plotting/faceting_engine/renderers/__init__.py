"""Renderers module for faceted plotting."""

from .plotly import render_plotly
from .matplotlib import render_matplotlib

__all__ = ['render_plotly', 'render_matplotlib']
