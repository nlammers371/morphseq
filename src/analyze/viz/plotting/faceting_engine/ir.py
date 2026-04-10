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
    # Marker styling (used when render_as='scatter', or when a renderer supports markers)
    marker: str = 'o'
    marker_size: float = 7.0
    marker_facecolor: str = 'none'  # 'none' for hollow markers (matplotlib)
    marker_edgecolor: str | None = None
    marker_edgewidth: float = 2.0


_LINESTYLE_MAP = {
    'solid':    ('-',  'solid'),
    'dashed':   ('--', 'dash'),
    'dotted':   (':',  'dot'),
    # also accept matplotlib/plotly codes directly
    '-':        ('-',  'solid'),
    '--':       ('--', 'dash'),
    ':':        (':',  'dot'),
}


def resolve_linestyle(style: str) -> tuple[str, str]:
    """Return (matplotlib_style, plotly_dash) for a linestyle name or code.

    Parameters
    ----------
    style : str
        Linestyle name ('solid', 'dashed', 'dotted') or code ('-', '--', ':')

    Returns
    -------
    tuple[str, str]
        (matplotlib_linestyle, plotly_dash_value)

    Raises
    ------
    ValueError
        If linestyle is not recognized
    """
    key = style.strip().lower()
    if key not in _LINESTYLE_MAP:
        raise ValueError(
            f"Unknown linestyle {style!r}. "
            f"Use: 'solid', 'dashed', 'dotted' (or '-', '--', ':')."
        )
    return _LINESTYLE_MAP[key]


# --- Heatmap Visual Style (figure-level, shared across all panels) ---

@dataclass
class HeatmapStyle:
    """Visual style for heatmap panels. Lives on FigureData, shared across all panels."""
    cmap: str = 'BuPu'
    vmin: Optional[float] = None          # explicit bounds; if None, computed from data at render time
    vmax: Optional[float] = None
    vcenter: Optional[float] = None       # if set, use TwoSlopeNorm (e.g. 0.5 for AUROC = chance level)
    missing_color: str = '#d9d9d9'        # grey for NaN cells
    sig_border_color: str = '#000000'     # border color for significant cells
    sig_border_width: float = 2.5
    sig_halo_color: Optional[str] = None  # outer halo behind border (None = no halo)
    sig_halo_width: float = 4.5           # halo is wider, drawn first so border sits on top
    annotation_fontsize: float = 8.0
    show_annotations: bool = False


@dataclass
class HeatmapData:
    """Pure data payload for a single heatmap panel."""
    values: np.ndarray                    # (n_rows, n_cols), NaN for missing
    row_labels: List[str]                 # y-axis tick labels
    col_labels: List[str]                 # x-axis tick labels
    sig_mask: Optional[np.ndarray] = None # bool, same shape — True = significant (never True where values is NaN)
    annotations: Optional[np.ndarray] = None  # str array, same shape — cell text


@dataclass
class ColorbarSpec:
    """Configuration for a shared colorbar across all heatmap panels in a figure."""
    label: str = ''
    shrink: float = 0.8
    aspect: int = 30
    pad: float = 0.02


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
    """A single cell. 'key' is just metadata; engine trusts list order by default.

    Must contain exactly one panel geometry: either traces (non-empty) or heatmap (non-None),
    never both. An empty traces list with heatmap=None is also valid (empty panel).
    """
    traces: List[TraceData] = field(default_factory=list)
    heatmap: Optional[HeatmapData] = None  # if present, renderer uses heatmap path; traces must be empty
    key: Tuple[Any, Any] = (None, None)  # (row_val, col_val) - optional metadata
    title: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None

    def __post_init__(self) -> None:
        if self.heatmap is not None and len(self.traces) > 0:
            raise ValueError(
                "SubplotData cannot contain both traces and a heatmap. "
                "Use either traces (for line/scatter/band plots) or heatmap (for matrix plots), not both."
            )


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
    subtitle: Optional[str] = None        # secondary line below title (e.g. "negative: wik_ab")
    # Labels for facet strips (optional overrides)
    row_labels: Optional[List[str]] = None
    col_labels: Optional[List[str]] = None
    # Heatmap-specific figure-level config (required when any subplot contains a heatmap)
    heatmap_style: Optional[HeatmapStyle] = None
    colorbar: Optional[ColorbarSpec] = None
