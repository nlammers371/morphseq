"""
Style specification for faceted plots.
"""

from dataclasses import dataclass, fields, replace
from typing import Any


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
    legend_loc: str = 'upper right'  # any matplotlib loc string, or 'outside' to place right of axes

    # Text (Matplotlib)
    # If None, defer to backend defaults / rcParams.
    axis_label_fontsize: int | None = None

    # Axis labels
    # By default, when axes are shared the engine shows labels only on outer axes.
    # Set these to True to repeat labels on every subplot.
    repeat_xlabels: bool = False
    repeat_ylabels: bool = False

    # Tick-label numbers
    # When axes are shared, matplotlib/plotly typically hide inner tick labels by default.
    # Set these to True to show tick labels on every subplot.
    repeat_xticklabels: bool = True
    repeat_yticklabels: bool = True
    
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


def presentation_style() -> StyleSpec:
    return StyleSpec(
        height_per_row=320,
        width_per_col=360,
        min_height=420,
        min_width=640,
        individual_alpha=0.18,
        individual_width=0.8,
        trend_alpha=1.0,
        trend_width=2.4,
        band_alpha=0.18,
        legend_fontsize=10,
        legend_loc='outside',
        repeat_xlabels=False,
        repeat_ylabels=False,
        repeat_xticklabels=False,
        repeat_yticklabels=False,
    )


def dense_facet_style() -> StyleSpec:
    return StyleSpec(
        height_per_row=300,
        width_per_col=320,
        min_height=360,
        min_width=520,
        legend_fontsize=10,
        legend_loc='outside',
        repeat_xlabels=False,
        repeat_ylabels=False,
        repeat_xticklabels=False,
        repeat_yticklabels=False,
    )


def update_style(base: StyleSpec | None = None, **overrides: Any) -> StyleSpec:
    """Return a copy of StyleSpec with selected fields overridden.

    This is the ergonomic path for "start from a preset, tweak one or two
    fields" without requiring callers to know the full dataclass layout.
    """
    style = replace(base) if base is not None else StyleSpec()
    valid_fields = {f.name for f in fields(StyleSpec)}
    unknown = sorted(set(overrides) - valid_fields)
    if unknown:
        raise TypeError(
            "Unknown StyleSpec override(s): "
            + ", ".join(unknown)
            + ". Valid fields are: "
            + ", ".join(sorted(valid_fields))
        )
    for key, value in overrides.items():
        setattr(style, key, value)
    return style
