"""
Intermediate Representation (IR) & Layout Configuration.

Pure dataclasses. No imports beyond numpy and typing.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any


# --- Visual Styling ---

@dataclass
class TraceStyle:
    """Visual style for a trace (separates style from data)."""
    color: str
    alpha: float = 1.0
    width: float = 1.0
    linestyle: str = '-'  # '-' solid, '--' dashed, ':' dotted
    zorder: int = 1


# --- Data Content ---

@dataclass
class TraceData:
    """Represents a single line or band on a plot."""
    x: np.ndarray
    y: np.ndarray
    style: TraceStyle
    # Legend control
    label: Optional[str] = None
    legend_group: Optional[str] = None
    show_legend: bool = False
    # Hover metadata (Plotly uses; MPL ignores)
    hover_meta: Optional[dict] = None
    # Band support (for error bands / fill_between)
    band_lower: Optional[np.ndarray] = None
    band_upper: Optional[np.ndarray] = None
    render_as: str = 'line'  # 'line' or 'band'


@dataclass
class SubplotData:
    """A single cell. 'key' is just metadata; engine trusts list order by default."""
    traces: List[TraceData]
    key: Tuple[Any, Any] = (None, None)  # (row_val, col_val) - optional metadata
    title: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None


# --- Layout Configuration ---

@dataclass
class FacetSpec:
    """Rules for arranging subplots. Optional."""
    wrap: Optional[int] = None      # Columns for 1D wrapping
    sharex: bool = True
    sharey: bool = True
    # If None, engine infers order from FigureData.subplots list
    row_order: Optional[List[Any]] = None
    col_order: Optional[List[Any]] = None


@dataclass
class FigureData:
    """Represents the complete figure, agnostic of backend."""
    title: str
    subplots: List[SubplotData]
    legend_title: Optional[str] = None
    # Labels for facet strips (optional overrides)
    row_labels: Optional[List[str]] = None
    col_labels: Optional[List[str]] = None
