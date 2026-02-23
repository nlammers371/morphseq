"""
Proportion plotting functions for faceted layouts (generic).

Primary API:
    - plot_proportions: Proportion plots (preferred)

Usage
-----
Faceted mode (row/col are facet variables; colors are category values):
    >>> fig = plot_proportions(
    ...     df,
    ...     color_by_grouping='phenotype',
    ...     row_by='genotype_suffix',
    ...     col_by='pair',
    ... )

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

from analyze.viz.styling import (
    STANDARD_PALETTE,
    normalize_color,
    resolve_color_lookup,
)

DEFAULT_HEIGHT_PER_ROW = 350
DEFAULT_WIDTH_PER_COL = 400


def _ordered_unique(values: pd.Series) -> List:
    if hasattr(values, 'cat') and values.cat.ordered:
        present_cats = set(values.dropna().unique())
        return [c for c in values.cat.categories if c in present_cats]
    return list(pd.unique(values.dropna()))


def _make_color_lookup(values: pd.Series, palette: Optional[List[str]] = None) -> Dict:
    unique_vals = _ordered_unique(values)
    return resolve_color_lookup(
        unique_vals,
        palette=palette or STANDARD_PALETTE,
        enforce_distinct=True,
        warn_on_collision=False,
    )


# ==============================================================================
# Proportion Faceted Plot (Consistent API with plot_feature_over_time)
# ==============================================================================

def _plot_proportions_impl(
    df: pd.DataFrame,
    color_by_grouping: str,
    row_by: Optional[str] = None,
    col_by: Optional[str] = None,
    count_by: str = 'embryo_id',
    facet_order: Optional[Dict[str, List]] = None,
    color_order: Optional[List] = None,
    color_palette: Optional[Dict[str, str]] = None,
    normalize: bool = True,
    bar_mode: str = 'grouped',
    height_per_row: int = DEFAULT_HEIGHT_PER_ROW,
    width_per_col: int = DEFAULT_WIDTH_PER_COL,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    show_counts: bool = True,
) -> plt.Figure:
    """
    Plot proportion breakdown with faceted grid structure.

    API is consistent with plot_feature_over_time:
    - row_by/col_by define facet grid by column VALUES (not variable names)
    - color_by_grouping defines the categorical for bar colors
    """
    # Determine row and column values
    # When row_by/col_by is None, use a single row/column with a placeholder label
    if row_by is not None:
        row_values = sorted(df[row_by].dropna().unique())
        if facet_order and 'row_by' in facet_order:
            row_values = [v for v in facet_order['row_by'] if v in row_values]
    else:
        row_values = ['_all_']  # Placeholder for single row

    if col_by is not None:
        col_values = sorted(df[col_by].dropna().unique())
        if facet_order and 'col_by' in facet_order:
            col_values = [v for v in facet_order['col_by'] if v in col_values]
    else:
        col_values = ['_all_']  # Placeholder for single column

    # Determine color_by_grouping values
    # Use order-of-first-occurrence unless Categorical ordering is provided
    col_series = df[color_by_grouping]
    color_values = _ordered_unique(col_series)

    # Handle color_palette: can be None, a dict, or a list
    if color_palette is None:
        color_palette = _make_color_lookup(df[color_by_grouping])
    elif isinstance(color_palette, (list, tuple)):
        color_palette = resolve_color_lookup(
            color_values,
            palette=[normalize_color(c) for c in color_palette],
            enforce_distinct=True,
            warn_on_collision=False,
        )
    else:
        color_palette = resolve_color_lookup(
            color_values,
            color_lookup=color_palette,
            palette=STANDARD_PALETTE,
            enforce_distinct=True,
            warn_on_collision=True,
        )
    if color_order:
        color_values = [v for v in color_order if v in color_values]

    n_rows = len(row_values)
    n_cols = len(col_values)

    # Figure sizing
    fig_width = max(6, n_cols * (width_per_col / 100))
    fig_height = max(4, n_rows * (height_per_row / 100))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)

    # Legend tracking
    legend_handles = []
    legend_labels = []

    for r_idx, row_val in enumerate(row_values):
        for c_idx, col_val in enumerate(col_values):
            ax = axes[r_idx, c_idx]

            # Filter to this cell (skip filtering if using '_all_' placeholder)
            subset = df.copy()
            if row_by is not None and row_val != '_all_':
                subset = subset[subset[row_by] == row_val]
            if col_by is not None and col_val != '_all_':
                subset = subset[subset[col_by] == col_val]

            # Count unique count_by values per color_by_grouping category
            counts = subset.groupby(color_by_grouping, observed=True)[count_by].nunique()
            counts = counts.reindex(color_values, fill_value=0)

            total = counts.sum()
            if normalize and total > 0:
                proportions = counts / total
            else:
                proportions = counts

            # Draw bars
            n_bars = len(color_values)

            if bar_mode == 'grouped':
                bar_width = 0.8 / max(n_bars, 1)
                x_positions = np.linspace(-0.4 + bar_width/2, 0.4 - bar_width/2, n_bars) if n_bars > 0 else []

                for i, val in enumerate(color_values):
                    height = proportions.get(val, 0)
                    color = color_palette.get(val, STANDARD_PALETTE[0])

                    ax.bar(x_positions[i], height, width=bar_width,
                           color=color, edgecolor='white', linewidth=0.5)

                    # Add count annotation if requested
                    if show_counts and counts.get(val, 0) > 0:
                        ax.annotate(f'{int(counts.get(val, 0))}',
                                   xy=(x_positions[i], height),
                                   ha='center', va='bottom',
                                   fontsize=7, color='black')

                    # Track for legend (only once)
                    if r_idx == 0 and c_idx == 0:
                        legend_handles.append(plt.Rectangle((0, 0), 1, 1,
                                                           facecolor=color,
                                                           edgecolor='white',
                                                           linewidth=0.5))
                        legend_labels.append(val)

                # Y-axis limits for grouped mode
                max_val = proportions.max() if len(proportions) > 0 else 1
                ax.set_ylim(0, 1.15 if normalize else max_val * 1.2)

            else:  # stacked
                bottom = 0
                bar_width = 0.6

                for val in color_values:
                    height = proportions.get(val, 0)
                    if height > 0:
                        color = color_palette.get(val, STANDARD_PALETTE[0])
                        ax.bar(0, height, bottom=bottom, width=bar_width,
                              color=color, edgecolor='white', linewidth=0.5)

                        # Track for legend (only once)
                        if r_idx == 0 and c_idx == 0 and val not in legend_labels:
                            legend_handles.append(plt.Rectangle((0, 0), 1, 1,
                                                               facecolor=color,
                                                               edgecolor='white',
                                                               linewidth=0.5))
                            legend_labels.append(val)

                        bottom += height

                ax.set_ylim(0, 1.05 if normalize else total * 1.1)

            # Styling - tighter bar spacing
            ax.set_xlim(-0.5, 0.5)
            ax.set_xticks([])

            # Column titles on top row (skip if single column placeholder)
            if r_idx == 0 and col_by is not None and col_val != '_all_':
                ax.set_title(f'{col_val}', fontsize=10, fontweight='bold')

            # Row labels on left column (skip if single row placeholder)
            if c_idx == 0 and row_by is not None and row_val != '_all_':
                ax.set_ylabel(f'{row_val}', fontsize=10, fontweight='bold')

            # Y-axis formatting - set ticks on ALL facets for grid alignment
            if normalize:
                ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
                if c_idx == 0:
                    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
                else:
                    ax.set_yticklabels([])  # Hide labels but keep ticks for grid
            else:
                if c_idx != 0:
                    ax.set_yticklabels([])

            # Add horizontal grid lines across all facets
            ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5, color='lightgray')
            ax.set_axisbelow(True)

            # Clean up spines - keep bottom for baseline
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.spines['bottom'].set_color('gray')

    # Add single legend for all panels
    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc='center right',
            bbox_to_anchor=(0.98, 0.5),
            fontsize=9,
            frameon=True,
            framealpha=0.9,
            title=color_by_grouping,
            title_fontsize=10,
        )

    # Title
    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold')
    else:
        parts = []
        if col_by:
            parts.append(f'by {col_by}')
        if row_by:
            parts.append(f'and {row_by}')
        default_title = f'{color_by_grouping} Distribution {" ".join(parts)}'
        fig.suptitle(default_title, fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 0.82, 0.96])

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150, bbox_inches='tight')

    return fig


def plot_proportions(
    df: pd.DataFrame,
    color_by_grouping: str,
    row_by: Optional[str] = None,
    col_by: Optional[str] = None,
    count_by: str = 'embryo_id',
    facet_order: Optional[Dict[str, List]] = None,
    color_order: Optional[List] = None,
    color_palette: Optional[Dict] = None,
    normalize: bool = True,
    bar_mode: str = 'grouped',
    height_per_row: int = DEFAULT_HEIGHT_PER_ROW,
    width_per_col: int = DEFAULT_WIDTH_PER_COL,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    show_counts: bool = True,
) -> plt.Figure:
    """
    Proportion plotting with faceted grid:
    - row_by/col_by are column names whose VALUES define the grid
    - color_by_grouping defines bar categories
    """
    return _plot_proportions_impl(
        df=df,
        color_by_grouping=color_by_grouping,
        row_by=row_by,
        col_by=col_by,
        count_by=count_by,
        facet_order=facet_order,
        color_order=color_order,
        color_palette=color_palette,
        normalize=normalize,
        bar_mode=bar_mode,
        height_per_row=height_per_row,
        width_per_col=width_per_col,
        output_path=output_path,
        title=title,
        show_counts=show_counts,
    )


__all__ = [
    'plot_proportions',
]
