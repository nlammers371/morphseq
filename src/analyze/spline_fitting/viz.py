"""Spline visualization functions.

This module provides both augmentor functions (add to existing figures) and
convenience functions (create complete figures with one call).

Visualization Patterns:
    **Augmentor (Most Flexible)**:
    Build figures step by step by adding splines to existing plots.

    **Convenience (Quick One-Liner)**:
    Create complete scatter + spline figures in one call.

Example - Augmentor pattern:
    >>> from src.analyze.viz.plotting import plot_3d_scatter
    >>> from src.analyze.spline_fitting.viz import add_spline_to_fig
    >>>
    >>> fig = plot_3d_scatter(df, coords=['PC1', 'PC2', 'PC3'], color_by='phenotype')
    >>> add_spline_to_fig(fig, fitted_curve, color='red', width=5)
    >>> fig.show()

Example - Convenience pattern:
    >>> from src.analyze.spline_fitting.viz import plot_3d_with_spline
    >>>
    >>> fig = plot_3d_with_spline(
    ...     df, coords=['PC1', 'PC2', 'PC3'],
    ...     spline=fitted_curve,
    ...     color_by='phenotype'
    ... )
"""

import numpy as np
import plotly.graph_objects as go


# =============================================================================
# Augmentor Functions
# =============================================================================

def add_spline_to_fig(fig, spline_coords, color='red', width=4, name='Spline', showlegend=True):
    """Add spline curve to existing Plotly 3D figure.

    This is an augmentor function that modifies a figure in-place.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Existing 3D figure to add spline to.
    spline_coords : ndarray or pd.DataFrame, shape (n_points, 3)
        Spline coordinates. If DataFrame, uses first 3 columns.
    color : str, default='red'
        Line color.
    width : int, default=4
        Line width.
    name : str, default='Spline'
        Trace name for legend.
    showlegend : bool, default=True
        Whether to show in legend.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Modified figure (same object, modified in-place).

    Example
    -------
    >>> fig = plot_3d_scatter(df, coords=['PC1', 'PC2', 'PC3'])
    >>> add_spline_to_fig(fig, fitted_curve, color='blue', width=5)
    >>> fig.show()
    """
    # Convert to numpy if needed
    if hasattr(spline_coords, 'values'):
        coords = spline_coords.values
    else:
        coords = np.array(spline_coords)

    if coords.shape[1] < 3:
        raise ValueError(f"spline_coords must have at least 3 columns, got shape {coords.shape}")

    fig.add_trace(go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode='lines',
        line=dict(color=color, width=width),
        name=name,
        showlegend=showlegend
    ))

    return fig


def add_uncertainty_tube(fig, spline_df, coord_cols=None, color='red', opacity=0.2, name='Uncertainty'):
    """Add uncertainty tube around spline (from bootstrap SE).

    This is an augmentor function that adds uncertainty visualization to an
    existing figure.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Existing 3D figure.
    spline_df : pd.DataFrame
        Spline with uncertainty. Must have coord_cols and {col}_se columns.
    coord_cols : list of str, optional
        Coordinate columns. Defaults to first 3 PCA columns found.
    color : str, default='red'
        Tube color.
    opacity : float, default=0.2
        Tube opacity (0-1).
    name : str, default='Uncertainty'
        Trace name for legend.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Modified figure.

    Example
    -------
    >>> fig = plot_3d_scatter(df, coords=['PC1', 'PC2', 'PC3'])
    >>> add_spline_to_fig(fig, spline_df[['PC1', 'PC2', 'PC3']])
    >>> add_uncertainty_tube(fig, spline_df, coord_cols=['PC1', 'PC2', 'PC3'])
    """
    if coord_cols is None:
        coord_cols = [col for col in spline_df.columns if col.startswith('PCA_') and not col.endswith('_se')][:3]

    se_cols = [col + '_se' for col in coord_cols]

    # Check if SE columns exist
    missing = set(se_cols) - set(spline_df.columns)
    if missing:
        raise ValueError(f"Missing SE columns: {missing}. Bootstrap uncertainty not available.")

    # Upper bound
    upper = spline_df[coord_cols].values + spline_df[se_cols].values

    # Lower bound
    lower = spline_df[coord_cols].values - spline_df[se_cols].values

    # Add upper bound points
    fig.add_trace(go.Scatter3d(
        x=upper[:, 0],
        y=upper[:, 1],
        z=upper[:, 2],
        mode='markers',
        marker=dict(size=1, color=color, opacity=opacity),
        name=f'{name} +SE',
        showlegend=False
    ))

    # Add lower bound points
    fig.add_trace(go.Scatter3d(
        x=lower[:, 0],
        y=lower[:, 1],
        z=lower[:, 2],
        mode='markers',
        marker=dict(size=1, color=color, opacity=opacity),
        name=f'{name} -SE',
        showlegend=False
    ))

    return fig


# =============================================================================
# Convenience Functions
# =============================================================================

def plot_3d_with_spline(
    df,
    coords,
    spline=None,
    color_by=None,
    spline_color='red',
    spline_width=4,
    show_uncertainty=False,
    title=None,
    spline_group_by=None,
    color_palette=None,
    **scatter_kwargs
):
    """Create 3D scatter plot with optional spline overlay (convenience function).

    This is a convenience wrapper that creates a complete figure in one call.
    It matches the API of plot_3d_scatter from trajectory_analysis.viz.plotting
    plus spline-specific parameters.

    Parameters
    ----------
    df : pd.DataFrame
        Data points to scatter.
    coords : list of str, length 3
        Column names for x, y, z coordinates.
    spline : ndarray or pd.DataFrame, optional
        Spline coordinates to overlay. If None, only shows scatter.
        When ``spline_group_by`` is set, must be a DataFrame containing
        that grouping column plus the ``coords`` columns.
    color_by : str, optional
        Column to color scatter points by.
    spline_color : str, default='red'
        Spline line color (used when ``spline_group_by`` is None).
    spline_width : int, default=4
        Spline line width.
    show_uncertainty : bool, default=False
        If True and spline has SE columns, shows uncertainty tube.
    title : str, optional
        Plot title.
    spline_group_by : str, optional
        Column in ``spline`` DataFrame to group by.  Draws one spline per
        group with colors matching the scatter's ``color_by`` palette.
    color_palette : dict, optional
        Mapping of group values to color strings.  Passed through to
        ``plot_3d_scatter`` *and* used for per-group spline colours.
    **scatter_kwargs
        Additional arguments passed to plotly scatter creation.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Complete figure with scatter + spline.

    Example
    -------
    >>> fig = plot_3d_with_spline(
    ...     df, coords=['PC1', 'PC2', 'PC3'],
    ...     spline=fitted_curve,
    ...     color_by='phenotype',
    ...     title='Trajectory with Fitted Spline'
    ... )
    >>> fig.show()

    Notes
    -----
    For more control, use the augmentor pattern instead:
        >>> from src.analyze.viz.plotting import plot_3d_scatter
        >>> fig = plot_3d_scatter(df, coords=coords, color_by=color_by)
        >>> add_spline_to_fig(fig, spline, color='blue')
    """
    # Build scatter kwargs, passing color_palette if supported
    scatter_kw = dict(scatter_kwargs)
    if color_palette is not None:
        scatter_kw['color_palette'] = color_palette

    # Try to import the plotting module
    try:
        from analyze.viz.plotting import plot_3d_scatter
        fig = plot_3d_scatter(df, coords=coords, color_by=color_by, **scatter_kw)
    except ImportError:
        # Fallback: create basic scatter ourselves
        import plotly.express as px
        fig = px.scatter_3d(
            df, x=coords[0], y=coords[1], z=coords[2],
            color=color_by,
            title=title,
        )

    # Add spline(s)
    if spline is not None and spline_group_by is not None:
        # Per-group splines with matched colours
        from analyze.viz.plotting.plotting_3d import _build_color_lookup
        color_map = _build_color_lookup(df, color_by, color_palette=color_palette)
        for group_val, group_df in spline.groupby(spline_group_by):
            color = color_map.get(group_val, 'red')
            add_spline_to_fig(
                fig, group_df[coords].values,
                color=color, width=spline_width,
                name=f"{group_val} spline",
            )
    elif spline is not None:
        add_spline_to_fig(fig, spline, color=spline_color, width=spline_width)

        # Add uncertainty if requested and available
        if show_uncertainty:
            try:
                add_uncertainty_tube(fig, spline, coord_cols=coords, color=spline_color)
            except (ValueError, KeyError):
                pass

    if title:
        fig.update_layout(title=title)

    return fig
