"""
Generic grid iteration and layout calculation utilities.

This is part of the engine because it's about grids, not biology.
"""

from typing import Iterator, Tuple, Dict, Any, Optional, List

from .ir import FigureData, FacetSpec
from .style.defaults import StyleSpec


def iter_facet_cells(
    facet_row: Optional[str],
    facet_col: Optional[str],
    row_vals: List[Any],
    col_vals: List[Any],
) -> Iterator[Tuple[Any, Any, Dict[str, Any], Tuple[Any, Any]]]:
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
            
            subplot_key: Tuple[Any, Any] = (row_val, col_val)
            yield row_val, col_val, filter_dict, subplot_key


def calculate_grid_map(fig_data: FigureData, facet: FacetSpec) -> Tuple[int, int, Dict[int, Dict[str, Any]]]:
    """Calculate grid layout from figure data.
    
    Returns
    -------
    n_rows : int
        Number of rows in grid
    n_cols : int
        Number of columns in grid
    positions : dict
        Map from subplot_index to position info dict with keys:
        {'row': int, 'col': int, 'show_x': bool, 'show_y': bool}
    """
    # 1. Wrap mode (1D list -> 2D grid, row-major)
    if facet.wrap:
        n_cols = facet.wrap
        n_rows = (len(fig_data.subplots) + n_cols - 1) // n_cols
        
        positions = {}
        for i in range(len(fig_data.subplots)):
            r = (i // n_cols) + 1
            c = (i % n_cols) + 1
            positions[i] = _make_position(r, c, n_rows, n_cols, facet)
            
        return n_rows, n_cols, positions

    # 2. 2D Grid based on keys
    # Collect unique values (preserving order from list if config is None)
    rows_seen = list(facet.row_order) if facet.row_order else []
    cols_seen = list(facet.col_order) if facet.col_order else []
    
    if not rows_seen or not cols_seen:
        for sub in fig_data.subplots:
            r_key, c_key = sub.key
            if r_key is not None and r_key not in rows_seen:
                rows_seen.append(r_key)
            if c_key is not None and c_key not in cols_seen:
                cols_seen.append(c_key)

    # Handle single subplot case
    if not rows_seen:
        rows_seen = [None]
    if not cols_seen:
        cols_seen = [None]

    n_rows = len(rows_seen)
    n_cols = len(cols_seen)
    
    positions = {}
    for i, sub in enumerate(fig_data.subplots):
        r_key, c_key = sub.key
        
        # Find index (1-based)
        r = (rows_seen.index(r_key) + 1) if r_key in rows_seen else 1
        c = (cols_seen.index(c_key) + 1) if c_key in cols_seen else 1
        
        positions[i] = _make_position(r, c, n_rows, n_cols, facet)

    # Infer row/col labels if not provided
    if fig_data.row_labels is None and len(rows_seen) > 1:
        fig_data.row_labels = [str(v) for v in rows_seen if v is not None]
    if fig_data.col_labels is None and len(cols_seen) > 1:
        fig_data.col_labels = [str(v) for v in cols_seen if v is not None]

    return n_rows, n_cols, positions


def _make_position(r: int, c: int, n_rows: int, n_cols: int, facet: FacetSpec) -> Dict[str, Any]:
    """Helper to create position dict with label visibility rules."""
    return {
        'row': r,
        'col': c,
        'show_x': (r == n_rows) if facet.sharex else True,
        'show_y': (c == 1) if facet.sharey else True,
    }


def compute_figure_size(n_rows: int, n_cols: int, style: StyleSpec) -> Tuple[int, int]:
    """Compute (height, width) from grid dimensions and style."""
    height = max(style.min_height, n_rows * style.height_per_row)
    width = max(style.min_width, n_cols * style.width_per_col)
    return height, width
