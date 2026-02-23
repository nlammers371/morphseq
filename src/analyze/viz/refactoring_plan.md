# Centralize Faceting Engine (Option A) - Final PRD

A clean refactor of `shared.py` into a modular faceting engine. This doc defines
the architecture, responsibilities, and concrete code for each component.

---

## Dataflow Contract (KISS) - Vertical Slice Approach

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PLOT MODULE (Vertical Slice)                         │
│  feature_over_time.py, proportions.py, etc.                             │
│                                                                          │
│  • Validates user args                                                   │
│  • Determines facet values from DataFrame                                │
│  • Calls engine.iter_facet_cells() for grid iteration                   │
│  • Builds SubplotData (domain-specific compilation logic)                │
│  • Assembles FigureData                                                 │
│  • Calls engine.render()                                                │
│                                                                          │
│  Owns: API + DataFrame→IR compilation + domain imports                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              ENGINE                                      │
│  faceting_engine/                                                        │
│                                                                          │
│  • iter_facet_cells() - generic grid iteration                          │
│  • plan_layout(FigureData, FacetSpec) → LayoutPlan                      │
│  • compute_figure_size(n_rows, n_cols, StyleSpec) → (height, width)     │
│  • render_plotly(FigureData, LayoutPlan, StyleSpec) → go.Figure         │
│  • render_matplotlib(FigureData, LayoutPlan, StyleSpec) → plt.Figure    │
│                                                                          │
│  NO DataFrame imports. NO domain imports. Pure IR → rendered figure.    │
└─────────────────────────────────────────────────────────────────────────┘
```

**Two boxes. Vertical slices. Colocation wins.**

---

## Directory Layout (Vertical Slice)

```
src/analyze/viz/plotting/
├── feature_over_time.py     # API + Compilation (Self-contained)
├── proportions.py           # API + Compilation (Self-contained)
│
└── faceting_engine/         # Generic Engine (Protected)
    ├── __init__.py          # render() + exports
    ├── ir.py                # TraceData, SubplotData, FigureData
    ├── layout.py            # FacetSpec, LayoutPlan, plan_layout()
    ├── utils.py             # iter_facet_cells() (generic grid iteration)
    ├── stats.py             # compute_error_band, compute_linear_fit
    ├── renderers/
    │   ├── __init__.py
    │   ├── matplotlib.py
    │   └── plotly.py
    └── style/
        ├── __init__.py
        ├── colors.py        # STANDARD_PALETTE, normalize_color, to_rgba_string
        └── defaults.py      # StyleSpec
```

**Key: NO `adapters/` folder. Plot modules own their compilation logic.**

---

## Phase 1: Engine Components

### 1a: `ir.py` - Intermediate Representation

```python
"""
Intermediate Representation (IR) for faceted plotting.

Pure dataclasses. No imports beyond numpy and typing.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any


@dataclass
class TraceStyle:
    """Visual style for a trace (separates style from data)."""
    color: str
    alpha: float = 1.0
    width: float = 1.0
    linestyle: str = '-'  # '-' solid, '--' dashed, ':' dotted
    zorder: int = 1


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


# SubplotKey is a simple tuple for hashability
SubplotKey = Tuple[Any, Any]  # (row_val, col_val)


@dataclass
class SubplotData:
    """Represents a single grid cell (one set of axes)."""
    key: SubplotKey  # (row_val, col_val) tuple
    traces: List[TraceData]
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None
    title: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None


@dataclass
class FigureData:
    """Represents the complete figure, agnostic of backend."""
    title: str
    subplots: List[SubplotData]
    legend_title: Optional[str] = None
    # Labels for facet strips (optional overrides)
    row_labels: Optional[List[str]] = None
    col_labels: Optional[List[str]] = None
```

**Key decisions:**
- `SubplotKey = Tuple[Any, Any]` — hashable, simple, no dataclass complexity
- No width/height/n_rows/n_cols on FigureData — layout computes these
- `hover_meta: dict` instead of separate header/detail fields

---

### 1b: `style/colors.py` - Color Utilities (NO pandas)

```python
"""
Color utilities for faceted plotting.

NO pandas imports. Engine stays pure.
"""

from typing import Sequence, Dict, Any, Optional, List
import matplotlib.colors as mcolors


STANDARD_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def normalize_color(color: Any) -> str:
    """Convert any color format to hex string."""
    try:
        return mcolors.to_hex(color)
    except (ValueError, TypeError):
        return str(color)


def to_rgba_string(color: Any, alpha: float = 1.0) -> str:
    """Convert any color to rgba() string for Plotly fill.
    
    Uses mcolors.to_rgba for robust parsing of all formats.
    """
    try:
        r, g, b, _ = mcolors.to_rgba(color)
        return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha})"
    except (ValueError, TypeError):
        return f"rgba(128,128,128,{alpha})"


def create_color_lookup(
    unique_values: Sequence[Any],
    palette: Optional[List[str]] = None,
) -> Dict[Any, str]:
    """Create value→color mapping from unique values.
    
    NOTE: Caller is responsible for extracting unique values
    and ordering. This function just assigns colors.
    """
    palette = palette or STANDARD_PALETTE
    return {v: palette[i % len(palette)] for i, v in enumerate(unique_values)}
```

**Key decisions:**
- `to_rgba_string` uses `mcolors.to_rgba()` — handles hex, named, tuples, rgb strings
- `create_color_lookup` takes `Sequence[Any]`, not `pd.Series` — no pandas in engine
- Caller (adapter/plot module) extracts unique values and passes them in

---

### 1c: `style/defaults.py` - StyleSpec

```python
"""
Style specification for faceted plots.
"""

from dataclasses import dataclass


@dataclass
class StyleSpec:
    """Backend-agnostic style configuration."""
    # Sizing (per-cell)
    height_per_row: int = 350
    width_per_col: int = 400
    min_height: int = 400
    min_width: int = 600
    
    # Trace styling (matches trajectory_analysis defaults)
    individual_alpha: float = 0.2
    individual_width: float = 0.8
    trend_alpha: float = 1.0
    trend_width: float = 2.2
    band_alpha: float = 0.2
    
    # Spacing (fractions)
    vertical_spacing: float = 0.08
    horizontal_spacing: float = 0.05
    
    # Margins (Plotly)
    margin_left: int = 80
    margin_right: int = 120
    margin_top: int = 80
    margin_bottom: int = 60
    
    # Legend
    legend_fontsize: int = 12
    
    # Grid
    show_grid: bool = True
    grid_alpha: float = 0.3


def default_style() -> StyleSpec:
    return StyleSpec()


def paper_style() -> StyleSpec:
    return StyleSpec(
        height_per_row=300,
        width_per_col=350,
        trend_width=2.0,
        legend_fontsize=10,
    )
```

---

### 1d: `layout.py` - Facet Layout (decoupled from sizing)

```python
"""
Facet layout planning.

FacetSpec = layout behavior only (wrap, sharex, sharey, ordering, drop_empty)
LayoutPlan = computed grid positions + label suppression
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

from .ir import FigureData, SubplotKey


@dataclass
class FacetSpec:
    """Layout behavior specification (NOT data slicing).
    
    NOTE: row_order/col_order are VALUES, not column names.
    Plot module determines which values exist and passes them here.
    """
    row_order: Optional[List[Any]] = None
    col_order: Optional[List[Any]] = None
    wrap: Optional[int] = None  # facet_wrap mode
    sharex: bool = True
    sharey: bool = True
    drop_empty: bool = False


@dataclass
class SubplotPosition:
    """Position info for a subplot."""
    row: int  # 1-based
    col: int  # 1-based
    show_x_label: bool
    show_y_label: bool


@dataclass
class LayoutPlan:
    """Computed layout for rendering."""
    n_rows: int
    n_cols: int
    positions: Dict[int, SubplotPosition]  # subplot_index → position
    row_labels: List[str]
    col_labels: List[str]


def plan_layout(
    fig_data: FigureData,
    facet: Optional[FacetSpec] = None,
) -> LayoutPlan:
    """Compute layout from figure data and facet spec.
    
    NOTE: Does NOT compute figure size. That's compute_figure_size().
    """
    facet = facet or FacetSpec()
    
    # Collect unique row/col values from subplots
    row_vals = []
    col_vals = []
    for sub in fig_data.subplots:
        row_val, col_val = sub.key
        if row_val is not None and row_val not in row_vals:
            row_vals.append(row_val)
        if col_val is not None and col_val not in col_vals:
            col_vals.append(col_val)
    
    # Apply ordering if specified
    if facet.row_order:
        row_vals = [v for v in facet.row_order if v in row_vals]
    if facet.col_order:
        col_vals = [v for v in facet.col_order if v in col_vals]
    
    # Handle single subplot case
    if not row_vals:
        row_vals = [None]
    if not col_vals:
        col_vals = [None]
    
    n_rows = len(row_vals)
    n_cols = len(col_vals)
    
    # Compute positions by subplot INDEX (not key)
    positions: Dict[int, SubplotPosition] = {}
    for idx, sub in enumerate(fig_data.subplots):
        row_val, col_val = sub.key
        row_idx = (row_vals.index(row_val) + 1) if row_val in row_vals else 1
        col_idx = (col_vals.index(col_val) + 1) if col_val in col_vals else 1
        
        # Label suppression
        show_x = (row_idx == n_rows) if facet.sharex else True
        show_y = (col_idx == 1) if facet.sharey else True
        
        positions[idx] = SubplotPosition(
            row=row_idx, col=col_idx,
            show_x_label=show_x, show_y_label=show_y,
        )
    
    # Row/col labels for facet strips
    row_labels = fig_data.row_labels or [str(v) for v in row_vals if v is not None]
    col_labels = fig_data.col_labels or [str(v) for v in col_vals if v is not None]
    
    return LayoutPlan(
        n_rows=n_rows,
        n_cols=n_cols,
        positions=positions,
        row_labels=row_labels,
        col_labels=col_labels,
    )


def compute_figure_size(
    n_rows: int,
    n_cols: int,
    style: "StyleSpec",
) -> tuple[int, int]:
    """Compute (height, width) from grid dimensions and style."""
    from .style.defaults import StyleSpec  # avoid circular
    height = max(style.min_height, n_rows * style.height_per_row)
    width = max(style.min_width, n_cols * style.width_per_col)
    return height, width
```

**Key decisions:**
- `FacetSpec` has NO `row_by/col_by` — those are DataFrame column names (plot module concern)
- `LayoutPlan.positions` keyed by `int` (subplot index) not SubplotKey — no hash collisions
- `compute_figure_size()` is separate from `plan_layout()` — layout is reusable, sizing is style-dependent

---

### 1e: `stats.py` - Generic Statistics

Move existing implementations verbatim from shared.py (lines 280-440).
No placeholders — actual working code.

```python
"""
Generic statistical utilities for plotting.
"""

import numpy as np
from typing import Tuple, Optional

VALID_ERROR_TYPES = {
    'mean': ['sd', 'se'],
    'median': ['iqr', 'mad'],
}


def validate_error_type(trend_statistic: str, error_type: str) -> None:
    """Validate error_type is compatible with trend_statistic."""
    valid = VALID_ERROR_TYPES.get(trend_statistic, [])
    if error_type not in valid:
        raise ValueError(
            f"error_type='{error_type}' incompatible with '{trend_statistic}'. "
            f"Valid: {valid}"
        )


def compute_error_band(
    times: np.ndarray,
    metrics: np.ndarray,
    bin_width: float,
    statistic: str = 'median',
    error_type: str = 'iqr',
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute binned central tendency ± error.
    
    (Implementation moved verbatim from shared.py)
    """
    # ... existing implementation unchanged ...


def compute_linear_fit(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """Compute linear regression fit.
    
    (Implementation moved verbatim from shared.py)
    """
    # ... existing implementation unchanged ...
```

---

### 1f: `renderers/plotly.py`

```python
"""
Plotly renderer for faceted plots.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..ir import FigureData, TraceData
from ..layout import LayoutPlan, compute_figure_size
from ..style.defaults import StyleSpec
from ..style.colors import to_rgba_string


def render_plotly(
    data: FigureData,
    layout: LayoutPlan,
    style: StyleSpec,
) -> go.Figure:
    """Render FigureData to Plotly figure."""
    height, width = compute_figure_size(layout.n_rows, layout.n_cols, style)
    
    fig = make_subplots(
        rows=layout.n_rows,
        cols=layout.n_cols,
        vertical_spacing=style.vertical_spacing,
        horizontal_spacing=style.horizontal_spacing,
    )
    
    for idx, sub in enumerate(data.subplots):
        pos = layout.positions.get(idx)
        if pos is None:
            continue
        
        for trace in sub.traces:
            if trace.render_as == 'band' and trace.band_lower is not None:
                _add_band_trace(fig, trace, pos.row, pos.col, style)
            else:
                _add_line_trace(fig, trace, pos.row, pos.col)
        
        # Axis config
        x_title = sub.x_label if pos.show_x_label else None
        y_title = sub.y_label if pos.show_y_label else None
        
        fig.update_xaxes(range=sub.xlim, title_text=x_title, row=pos.row, col=pos.col)
        fig.update_yaxes(range=sub.ylim, title_text=y_title, row=pos.row, col=pos.col)
    
    _add_facet_labels(fig, layout)
    
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
        margin=dict(l=style.margin_left, r=style.margin_right,
                    t=style.margin_top, b=style.margin_bottom),
    )
    
    return fig


def _add_line_trace(fig, trace: TraceData, row: int, col: int):
    """Add a line trace."""
    line_dash = 'dash' if trace.style.linestyle == '--' else 'solid'
    
    hovertemplate = None
    if trace.hover_meta:
        header = trace.hover_meta.get('header', '')
        detail = trace.hover_meta.get('detail', '')
        hovertemplate = f'<b>{header}</b><br><b>Time:</b> %{{x:.2f}}<br>{detail}<extra></extra>'
    
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


def _add_facet_labels(fig, layout: LayoutPlan):
    """Add row/column facet strip labels as annotations."""
    if layout.row_labels and layout.n_rows > 1:
        for idx, label in enumerate(layout.row_labels, start=1):
            y_pos = 1 - (idx - 0.5) / layout.n_rows
            fig.add_annotation(
                text=f"<b>{label}</b>",
                xref="paper", yref="paper",
                x=-0.06, y=y_pos,
                showarrow=False, xanchor="center", yanchor="middle",
                textangle=-90, font=dict(size=13),
            )
    
    if layout.col_labels and layout.n_cols > 1:
        for idx, label in enumerate(layout.col_labels, start=1):
            x_pos = (idx - 0.5) / layout.n_cols
            fig.add_annotation(
                text=f"<b>{label}</b>",
                xref="paper", yref="paper",
                x=x_pos, y=1.02,
                showarrow=False, xanchor="center", yanchor="bottom",
                font=dict(size=13),
            )
```

---

### 1g: `renderers/matplotlib.py`

```python
"""
Matplotlib renderer for faceted plots.
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ..ir import FigureData, TraceData
from ..layout import LayoutPlan
from ..style.defaults import StyleSpec


def render_matplotlib(
    data: FigureData,
    layout: LayoutPlan,
    style: StyleSpec,
) -> plt.Figure:
    """Render FigureData to Matplotlib figure."""
    figsize = (5 * layout.n_cols, 4.5 * layout.n_rows)
    fig, axes = plt.subplots(layout.n_rows, layout.n_cols, figsize=figsize, squeeze=False)
    
    legend_entries = {}  # label → color
    
    for idx, sub in enumerate(data.subplots):
        pos = layout.positions.get(idx)
        if pos is None:
            continue
        
        ax = axes[pos.row - 1][pos.col - 1]
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
        if pos.show_x_label and sub.x_label:
            ax.set_xlabel(sub.x_label, fontsize=10)
        if pos.show_y_label and sub.y_label:
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
    
    if layout.row_labels and layout.n_rows > 1:
        for idx, label in enumerate(layout.row_labels):
            y_pos = 1 - (idx + 0.5) / layout.n_rows
            fig.text(0.02, y_pos, label, rotation=90, va='center', ha='right',
                     fontsize=12, fontweight='bold', transform=fig.transFigure)
    
    fig.suptitle(data.title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0.03, 0, 0.82 if legend_entries else 1, 0.96])
    
    return fig
```

---

### 1h: `__init__.py` - Engine Front Door

```python
"""
Faceting Engine - Generic faceted plotting infrastructure.

Usage:
    from src.analyze.viz.plotting.faceting_engine import (
        FigureData, SubplotData, TraceData, TraceStyle, SubplotKey,
        FacetSpec, StyleSpec, render,
    )
"""

from pathlib import Path
from typing import Optional, Union, Any

from .ir import TraceData, TraceStyle, SubplotData, SubplotKey, FigureData
from .layout import FacetSpec, LayoutPlan, plan_layout, compute_figure_size
from .style.defaults import StyleSpec, default_style, paper_style
from .style.colors import STANDARD_PALETTE, normalize_color, to_rgba_string, create_color_lookup
from .stats import validate_error_type, compute_error_band, compute_linear_fit


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
    layout = plan_layout(fig_data, facet)
    
    results = {}
    
    if backend in ('plotly', 'both'):
        fig_plotly = render_plotly(fig_data, layout, style)
        if output_path and backend == 'plotly':
            fig_plotly.write_html(str(output_path))
        results['plotly'] = fig_plotly
    
    if backend in ('matplotlib', 'both'):
        fig_mpl = render_matplotlib(fig_data, layout, style)
        if output_path and backend == 'matplotlib':
            fig_mpl.savefig(str(output_path), dpi=150, bbox_inches='tight')
        results['matplotlib'] = fig_mpl
    
    if backend == 'both' and output_path:
        path = Path(output_path)
        results['plotly'].write_html(str(path.with_suffix('.html')))
        results['matplotlib'].savefig(str(path.with_suffix('.png')), dpi=150, bbox_inches='tight')
    
    return results if backend == 'both' else results.get(backend)


__all__ = [
    'TraceData', 'TraceStyle', 'SubplotData', 'SubplotKey', 'FigureData',
    'FacetSpec', 'LayoutPlan', 'plan_layout', 'compute_figure_size',
    'StyleSpec', 'default_style', 'paper_style',
    'STANDARD_PALETTE', 'normalize_color', 'to_rgba_string', 'create_color_lookup',
    'validate_error_type', 'compute_error_band', 'compute_linear_fit',
    'render',
]
```

---

## Phase 1f: `utils.py` - Generic Grid Iteration (NEW)

```python
"""
Generic grid iteration utilities.

This is part of the engine because it's about grids, not biology.
"""

from typing import Iterator, Tuple, Dict, Any, Optional, List

from .ir import SubplotKey


def iter_facet_cells(
    facet_row: Optional[str],
    facet_col: Optional[str],
    row_vals: List[Any],
    col_vals: List[Any],
) -> Iterator[Tuple[Any, Any, Dict[str, Any], SubplotKey]]:
    """Iterate over facet grid cells, yielding filter dicts.
    
    Generic grid iteration. Plot modules extract row_vals/col_vals
    from their DataFrame and pass them here.
    
    Parameters
    ----------
    facet_row, facet_col : str or None
        Column names (for building filter_dict keys)
    row_vals, col_vals : list
        Values to iterate over
    
    Yields
    ------
    (row_val, col_val, filter_dict, subplot_key)
        filter_dict can be used for DataFrame filtering
        subplot_key is used for IR assembly
    """
    for row_val in row_vals:
        for col_val in col_vals:
            filter_dict = {}
            if facet_row and row_val is not None:
                filter_dict[facet_row] = row_val
            if facet_col and col_val is not None:
                filter_dict[facet_col] = col_val
            
            subplot_key: SubplotKey = (row_val, col_val)
            yield row_val, col_val, filter_dict, subplot_key
```

---

## Phase 2: Plot Module (Vertical Slice)

### `feature_over_time.py` - Complete Self-Contained Module

```python
"""
Generic time-series plotting.

100% domain-agnostic. NO trajectory_analysis imports.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Set

# Generic imports ONLY
from src.analyze.utils.data_processing import get_trajectories_for_group, get_global_axis_ranges
from src.analyze.utils.stats import compute_trend_line

# Engine imports
from ..faceting_engine import (
    TraceData, TraceStyle, SubplotData, SubplotKey,
    create_color_lookup, compute_error_band, STANDARD_PALETTE,
)


def _build_color_lookup(
    df: pd.DataFrame,
    color_by: Optional[str],
    palette: Optional[List[str]] = None,
) -> Dict[Any, str]:
    """Build generic color lookup (NO domain logic).
    
    Private helper. Just assigns colors from palette.
    """
    if color_by is None or color_by not in df.columns:
        return {}
    
    unique_vals = list(df[color_by].dropna().unique())
    return create_color_lookup(unique_vals, palette or STANDARD_PALETTE)


def build_subplot_ir(
    df: pd.DataFrame,
    filter_dict: Dict[str, Any],
    x_col: str,
    y_col: str,
    line_by: str,
    color_lookup: Dict[Any, str],
    color_by: Optional[str],
    subplot_key: SubplotKey,
    legend_tracker: Set[str],
    *,
    show_individual: bool = True,
    show_error_band: bool = False,
    error_type: str = 'iqr',
    trend_statistic: str = 'median',
    trend_smooth_sigma: float = 1.5,
    bin_width: float = 0.5,
    smooth_method: Optional[str] = 'gaussian',
    smooth_params: Optional[Dict] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
) -> SubplotData:
    """Build SubplotData (IR) for one facet cell."""
    
    # Determine color groups
    if color_by:
        mask = pd.Series(True, index=df.index)
        for k, v in filter_dict.items():
            mask &= (df[k] == v)
        groups = sorted(df.loc[mask, color_by].dropna().unique())
    else:
        groups = [None]
    
    traces: List[TraceData] = []
    
    for group_val in groups:
        group_filter = filter_dict.copy()
        if color_by:
            group_filter[color_by] = group_val
        
        trajectories, _, _ = get_trajectories_for_group(
            df, group_filter,
            time_col=x_col, metric_col=y_col, embryo_id_col=line_by,
            smooth_method=smooth_method, smooth_params=smooth_params,
        )
        
        if not trajectories:
            continue
        
        color = color_lookup.get(group_val, STANDARD_PALETTE[0])
        
        # Individual traces
        if show_individual:
            for traj in trajectories:
                traces.append(TraceData(
                    x=traj['times'], y=traj['metrics'],
                    style=TraceStyle(
                        color=color,
                        alpha=INDIVIDUAL_TRACE_ALPHA,
                        width=INDIVIDUAL_TRACE_LINEWIDTH,
                        zorder=2,
                    ),
                    show_legend=False,
                    hover_meta={'header': f"ID: {traj['embryo_id']}", 'detail': f"{y_col}: {{y:.3f}}"},
                ))
        
        # Aggregated data
        all_times = np.concatenate([t['times'] for t in trajectories])
        all_metrics = np.concatenate([t['metrics'] for t in trajectories])
        
        # Legend: show once per group across all subplots
        label = str(group_val) if group_val else trend_statistic
        legend_key = f"{y_col}_{group_val}"
        show_legend = legend_key not in legend_tracker
        if show_legend:
            legend_tracker.add(legend_key)
        
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
                    style=TraceStyle(color=color, alpha=0.2, width=0, zorder=3),
                    render_as='band',
                    show_legend=False,
                ))
        
        # Trend line
        trend_t, trend_v = compute_trend_line(
            all_times, all_metrics, bin_width,
            statistic=trend_statistic, smooth_sigma=trend_smooth_sigma,
        )
        if trend_t is not None:
            traces.append(TraceData(
                x=trend_t, y=trend_v,
                style=TraceStyle(color=color, alpha=1.0, width=2.2, zorder=5),  # from StyleSpec.trend_width
                label=label,
                legend_group=legend_key,
                show_legend=show_legend,
                hover_meta={'header': f"{trend_statistic.capitalize()}: {label}", 'detail': f"{y_col}: {{y:.3f}}"},
            ))
    
    return SubplotData(
        key=subplot_key,
        traces=traces,
        x_label=x_label or x_col,
        y_label=y_label or y_col,
    )
```

---

## Phase 3: Plot Module

### `feature_over_time.py` - Refactored Public API

```python
"""
"""
Plot feature over time with optional faceting.

100% DOMAIN-AGNOSTIC: No trajectory_analysis imports.
Caller provides color_lookup with domain-specific logic.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Any, Union, Dict, Set

# Generic imports ONLY
from src.analyze.utils.data_processing import get_trajectories_for_group, get_global_axis_ranges
from src.analyze.utils.stats import compute_trend_line

# Engine imports
from .faceting_engine import (
    FigureData, SubplotData, TraceData, TraceStyle, SubplotKey,
    FacetSpec, StyleSpec, render, default_style,
    iter_facet_cells, create_color_lookup, compute_error_band, STANDARD_PALETTE,
)


def _build_color_lookup(
    df: pd.DataFrame,
    color_by: Optional[str],
    color_lookup: Optional[Dict[Any, str]] = None,
    palette: Optional[List[str]] = None,
) -> Dict[Any, str]:
    """Build or use provided color lookup (NO domain logic).
    
    Private helper. If color_lookup provided, use it.
    Otherwise, auto-assign from palette.
    """
    if color_lookup is not None:
        return color_lookup
    
    if color_by is None or color_by not in df.columns:
        return {}
    
    unique_vals = list(df[color_by].dropna().unique())
    return create_color_lookup(unique_vals, palette or STANDARD_PALETTE)


def _build_subplot_ir(
    df: pd.DataFrame,
    filter_dict: Dict[str, Any],
    x_col: str,
    y_col: str,
    line_by: str,
    color_lookup: Dict[Any, str],
    color_by: Optional[str],
    subplot_key: SubplotKey,
    legend_tracker: Set[str],
    *,
    show_individual: bool = True,
    show_error_band: bool = False,
    error_type: str = 'iqr',
    trend_statistic: str = 'median',
    trend_smooth_sigma: float = 1.5,
    bin_width: float = 0.5,
    smooth_method: Optional[str] = 'gaussian',
    smooth_params: Optional[Dict] = None,
) -> SubplotData:
    """Build SubplotData (IR) for one facet cell.
    
    Private helper. Lives here because it's trajectory-specific compilation.
    """
    # Determine color groups
    if color_by:
        mask = pd.Series(True, index=df.index)
        for k, v in filter_dict.items():
            mask &= (df[k] == v)
        groups = sorted(df.loc[mask, color_by].dropna().unique())
    else:
        groups = [None]
    
    traces: List[TraceData] = []
    
    for group_val in groups:
        group_filter = filter_dict.copy()
        if color_by:
            group_filter[color_by] = group_val
        
        trajectories, _, _ = get_trajectories_for_group(
            df, group_filter,
            time_col=x_col, metric_col=y_col, embryo_id_col=line_by,
            smooth_method=smooth_method, smooth_params=smooth_params,
        )
        
        if not trajectories:
            continue
        
        color = color_lookup.get(group_val, STANDARD_PALETTE[0])
        
        # Individual traces
        if show_individual:
            for traj in trajectories:
                traces.append(TraceData(
                    x=traj['times'], y=traj['metrics'],
                    style=TraceStyle(
                        color=color,
                        alpha=INDIVIDUAL_TRACE_ALPHA,
                        width=INDIVIDUAL_TRACE_LINEWIDTH,
                        zorder=2,
                    ),
                    show_legend=False,
                    hover_meta={'header': f"ID: {traj['embryo_id']}", 'detail': f"{y_col}: {{y:.3f}}"},
                ))
        
        # Aggregated data
        all_times = np.concatenate([t['times'] for t in trajectories])
        all_metrics = np.concatenate([t['metrics'] for t in trajectories])
        
        # Legend: show once per group across all subplots
        label = str(group_val) if group_val else trend_statistic
        legend_key = f"{y_col}_{group_val}"
        show_legend = legend_key not in legend_tracker
        if show_legend:
            legend_tracker.add(legend_key)
        
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
                    style=TraceStyle(color=color, alpha=0.2, width=0, zorder=3),
                    render_as='band',
                    show_legend=False,
                ))
        
        # Trend line
        trend_t, trend_v = compute_trend_line(
            all_times, all_metrics, bin_width,
            statistic=trend_statistic, smooth_sigma=trend_smooth_sigma,
        )
        if trend_t is not None:
            traces.append(TraceData(
                x=trend_t, y=trend_v,
                style=TraceStyle(color=color, alpha=1.0, width=2.2, zorder=5),  # from StyleSpec.trend_width
                label=label,
                legend_group=legend_key,
                show_legend=show_legend,
                hover_meta={'header': f"{trend_statistic.capitalize()}: {label}", 'detail': f"{y_col}: {{y:.3f}}"},
            ))
    
    return SubplotData(
        key=subplot_key,
        traces=traces,
        x_label=x_col,
        y_label=y_col,
    )


def plot_feature_over_time(
    df: pd.DataFrame,
    feature: str,
    time_col: str = 'predicted_stage_hpf',
    id_col: str = 'embryo_id',
    color_by: Optional[str] = None,
    color_lookup: Optional[Dict[Any, str]] = None,  # ← USER PROVIDES domain-specific colors
    # Faceting (consistent API)
    facet_row: Optional[str] = None,
    facet_col: Optional[str] = None,
    layout: Optional[FacetSpec] = None,
    # Display
    show_individual: bool = True,
    show_error_band: bool = False,
    error_type: str = 'iqr',
    trend_statistic: str = 'median',
    # Output
    backend: str = 'plotly',
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    style: Optional[StyleSpec] = None,
    color_palette: Optional[List[str]] = None,  # Generic fallback
) -> Any:
    """Plot a feature over time, optionally faceted.
    
    100% DOMAIN-AGNOSTIC: Caller provides color_lookup for domain-specific coloring.
    If color_lookup=None, auto-assigns colors from palette.
    
    Parameters
    ----------
    color_lookup : Dict[Any, str], optional
        Pre-built mapping from values in color_by column to hex colors.
        Use this to inject domain-specific coloring (e.g., genotype colors).
    color_palette : List[str], optional
        Fallback palette if color_lookup not provided. Uses STANDARD_PALETTE if None.
    """
    layout = layout or FacetSpec(row_order=None, col_order=None)
    style = style or default_style()
    
    # Build color lookup (generic or user-provided)
    color_lookup = _build_color_lookup(df, color_by, color_lookup, color_palette)
    
    # Determine facet values
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
        facet_row, facet_col, row_vals, col_vals
    ):
        subplot = _build_subplot_ir(
            df=df,
            filter_dict=filter_dict,
            x_col=time_col,
            y_col=feature,
            line_by=id_col,
            color_lookup=color_lookup,
            color_by=color_by,
            subplot_key=subplot_key,
            legend_tracker=legend_tracker,
            show_individual=show_individual,
            show_error_band=show_error_band,
            error_type=error_type,
            trend_statistic=trend_statistic,
        )
        subplots.append(subplot)
    
    # Assemble FigureData
    fig_data = FigureData(
        title=title or f"{feature} over {time_col}",
        subplots=subplots,
        legend_title=color_by,
    )
    
    # Render via engine
    return render(fig_data, backend=backend, facet=layout, style=style, output_path=output_path)
```

---

## Phase 4: Trajectory-Specific Wrapper (Optional)

**File:** `src/analyze/trajectory_analysis/viz/plotting/trajectories.py`

```python
"""
Trajectory-specific plotting wrappers.

Provides genotype-aware defaults on top of generic viz.plotting.
"""

import pandas as pd
from typing import Optional, Dict, Any, List
from pathlib import Path

from ....viz.plotting.feature_over_time import plot_feature_over_time as _generic_plot
from ....viz.plotting.faceting_engine import FacetSpec, StyleSpec
from ...config import GENOTYPE_SUFFIX_COLORS, GENOTYPE_SUFFIX_ORDER


def build_genotype_color_lookup(df: pd.DataFrame, color_by: str) -> Dict[Any, str]:
    """Build genotype-aware color lookup.
    
    Domain-specific logic: matches genotype suffixes to standard colors.
    """
    if color_by not in df.columns:
        return {}
    
    unique_vals = list(df[color_by].dropna().unique())
    lookup = {}
    
    for val in unique_vals:
        val_str = str(val)
        # Match genotype suffix
        for suffix in GENOTYPE_SUFFIX_ORDER:
            if val_str.endswith('_' + suffix) or val_str == suffix:
                lookup[val] = GENOTYPE_SUFFIX_COLORS[suffix]
                break
        else:
            # Fallback for unknown genotypes
            from ....viz.plotting.faceting_engine import STANDARD_PALETTE
            lookup[val] = STANDARD_PALETTE[len(lookup) % len(STANDARD_PALETTE)]
    
    return lookup


def plot_trajectories_faceted(
    df: pd.DataFrame,
    x_col: str = 'predicted_stage_hpf',
    y_col: str = 'baseline_deviation_normalized',
    line_by: str = 'embryo_id',
    row_by: Optional[str] = None,
    col_by: Optional[str] = None,
    color_by: Optional[str] = None,
    color_palette: Optional[Dict[Any, str]] = None,
    **kwargs
) -> Any:
    """Plot trajectories with genotype-aware coloring.
    
    THIN WRAPPER around generic plot_feature_over_time.
    Adds trajectory-specific defaults (genotype suffix coloring).
    
    If color_by='genotype' and no color_palette provided,
    automatically uses GENOTYPE_SUFFIX_COLORS.
    """
    # Build color lookup with genotype awareness
    if color_palette is None and color_by == 'genotype':
        color_palette = build_genotype_color_lookup(df, color_by)
    
    # Delegate to generic plotter
    return _generic_plot(
        df=df,
        feature=y_col,
        time_col=x_col,
        id_col=line_by,
        color_by=color_by,
        color_lookup=color_palette,  # ← Inject domain-specific colors
        facet_row=row_by,
        facet_col=col_by,
        **kwargs
    )
```

**Key advantage:** Domain logic stays in `trajectory_analysis/`, while `viz/plotting/` remains 100% reusable.

---

## Summary

| File | Purpose | Lines |
|------|---------|-------|
| `faceting_engine/ir.py` | IR dataclasses | ~60 |
| `faceting_engine/layout.py` | FacetSpec, LayoutPlan, plan_layout | ~80 |
| `faceting_engine/utils.py` | iter_facet_cells (generic grid) | ~30 |
| `faceting_engine/stats.py` | compute_error_band, linear_fit | ~100 |
| `faceting_engine/style/colors.py` | Palette, normalize, to_rgba | ~40 |
| `faceting_engine/style/defaults.py` | StyleSpec | ~50 |
| `faceting_engine/renderers/plotly.py` | Plotly renderer | ~100 |
| `faceting_engine/renderers/matplotlib.py` | MPL renderer | ~80 |
| `faceting_engine/__init__.py` | render() + exports | ~50 |
| `feature_over_time.py` | 100% Generic plotter | ~200 |
| `proportions.py` | 100% Generic plotter | ~200 |
| **Generic Utils** | | |
| `utils/stats.py` | compute_trend_line (moved) | ~90 |
| **Trajectory Wrappers** | | |
| `trajectory_analysis/viz/plotting/trajectories.py` | Thin wrapper (genotype colors) | ~80 |

**Total:** ~1040 lines across 12 files.

**Key advantage:** 
- `viz/plotting/` is 100% domain-agnostic and reusable
- `trajectory_analysis/viz/` provides thin wrappers with domain defaults
- Zero cross-module dependencies from generic → domain code

---

## Key Architectural Decision: Zero Domain Imports in viz/

**`src/analyze/viz/plotting/` has ZERO imports from `trajectory_analysis`.**

All domain-specific logic (genotype coloring, trajectory defaults) lives in thin wrappers:
- `trajectory_analysis/viz/plotting/trajectories.py` wraps generic plotters
- Wrapper injects domain logic via `color_lookup` parameter
- Generic plotters accept `color_lookup` from caller (no magic)

**Migration of generic functions:**
- ✅ `get_trajectories_for_group` → already in `utils/data_processing.py`
- ✅ `get_global_axis_ranges` → already in `utils/data_processing.py`
- 🔄 `compute_trend_line` → **MOVE to `utils/stats.py`**
- ✅ Styling constants → **USE StyleSpec defaults** (no imports)

---

## Changes from Previous Plan

1. **FacetSpec** — removed `row_by/col_by` (now purely layout: sharex, sharey, wrap, ordering)
2. **SubplotKey** — now `Tuple[Any, Any]` (hashable, simple)
3. **LayoutPlan.positions** — keyed by subplot index (int), not SubplotKey
4. **to_rgba_string** — simplified to `mcolors.to_rgba()` (handles all formats)
5. **colors.py** — no pandas import; `create_color_lookup(unique_values)` takes Sequence
6. **iter_facet_cells** — moved to `engine/utils.py` (generic grid iteration)
7. **stats.py** — implementations moved verbatim (no placeholders)
8. **compute_figure_size** — separate from `plan_layout()` (decoupled)
9. **NO adapters/ folder** — compilation logic lives in plot modules (vertical slice)
10. **Consistent API** — all plot functions take `facet_row`, `facet_col`, `layout: FacetSpec`
11. **compute_trend_line** — moved to `utils/stats.py` (was in trajectory_analysis)
12. **color_lookup parameter** — callers provide domain colors (no genotype magic in generic code)
13. **StyleSpec defaults** — updated to match trajectory constants (0.2, 2.2 instead of 0.3, 2.5)
14. **Zero trajectory_analysis imports** — viz/plotting is 100% domain-agnostic