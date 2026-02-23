"""
Generic 3D Plotly plotting utilities.

Provides interactive 3D scatter plots for visualizing embeddings, PCA spaces,
and trajectory data. Domain-agnostic and reusable across analysis modules.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List, Optional, Any

from analyze.viz.styling import STANDARD_PALETTE, resolve_color_lookup


# ==============================================================================
# Color Helpers
# ==============================================================================

def _build_color_lookup(
    df: pd.DataFrame,
    color_by: str,
    color_palette: Optional[Dict[str, str]] = None,
    color_order: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Build color lookup dictionary for a column.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing the color_by column
    color_by : str
        Column name to color by
    color_palette : dict, optional
        Custom color mapping {value: hex_color}
    color_order : list, optional
        Order for assigning colors from STANDARD_PALETTE

    Returns
    -------
    dict
        Mapping from column values to hex colors
    """
    # Get unique values
    unique_vals = df[color_by].dropna().unique()

    # Use color_order if provided, otherwise sort
    if color_order is not None:
        # Filter to values that exist in data
        ordered_vals = [v for v in color_order if v in unique_vals]
        # Add any values not in color_order
        ordered_vals.extend([v for v in unique_vals if v not in ordered_vals])
    else:
        ordered_vals = sorted(unique_vals)

    return resolve_color_lookup(
        ordered_vals,
        color_lookup=color_palette,
        palette=STANDARD_PALETTE,
        enforce_distinct=True,
        warn_on_collision=True,
    )


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert hex color to rgba string with given alpha."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r, g, b = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
        return f'rgba({r}, {g}, {b}, {alpha})'
    return f'rgba(100, 100, 100, {alpha})'


# ==============================================================================
# Main Plotting Function
# ==============================================================================

def plot_3d_scatter(
    df: pd.DataFrame,
    coords: List[str],
    # Coloring
    color_by: str = 'group',
    color_palette: Optional[Dict[str, str]] = None,
    color_order: Optional[List[str]] = None,
    color_continuous: bool = False,
    group_by: Optional[str] = None,
    colorscale: str = 'Viridis',
    colorbar_title: Optional[str] = None,
    # Filtering
    line_by: str = 'id',
    min_points_per_line: int = 20,
    filter_groups: Optional[List[Any]] = None,
    filter_by_col: Optional[str] = None,
    downsample_frac: Optional[Dict[str, float]] = None,
    # Trajectory lines (optional)
    show_lines: bool = False,
    x_col: Optional[str] = 'predicted_stage_hpf',
    line_opacity: float = 0.3,
    line_width: float = 1.5,
    # Mean trajectory (optional)
    show_mean: bool = False,
    mean_line_width: int = 6,
    # Display
    point_opacity: float = 0.65,
    point_size: int = 4,
    hover_cols: Optional[List[str]] = None,
    title: str = "3D Scatter Plot",
    output_path: Optional[Path] = None,
    # Axis labels
    axis_labels: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """
    Generic 3D scatter plot with optional trajectory lines.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing coordinate columns and grouping columns
    coords : list of str
        Exactly 3 column names for x, y, z coordinates
        e.g., ['PCA_1', 'PCA_2', 'PCA_3']
    color_by : str, default='group'
        Column to use for coloring points/lines
    color_palette : dict, optional
        Custom color mapping {value: hex_color}. If None, uses STANDARD_PALETTE.
        Ignored if color_continuous=True.
    color_order : list, optional
        Order for color assignment and legend. Ignored if color_continuous=True.
    color_continuous : bool, default=False
        If True, color by continuous numeric values instead of categorical groups.
        Uses colorscale for mapping values to colors.
    group_by : str, optional
        Optional column to split points into legend-clickable traces while keeping
        continuous coloring from color_by. Useful for toggling groups (e.g., experiment_id)
        on/off in Plotly when color_continuous=True.
    colorscale : str, default='Viridis'
        Plotly colorscale name for continuous coloring. Only used if color_continuous=True.
        Options: 'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Jet', etc.
    colorbar_title : str, optional
        Title for the colorbar when using continuous coloring. If None, uses color_by column name.
    line_by : str, default='id'
        Column identifying individual trajectories (for filtering and lines)
    min_points_per_line : int, default=20
        Minimum points required per line_by group to be included
    filter_groups : list, optional
        Only include these values from filter_by_col
        (or color_by if filter_by_col is None)
    filter_by_col : str, optional
        Column used with filter_groups for subset selection
        (e.g., 'experiment_id')
    downsample_frac : dict, optional
        Fraction of points to keep per color_by value, e.g., {'wt': 0.1}
    show_lines : bool, default=False
        Connect points per line_by group over time
    x_col : str, default='predicted_stage_hpf'
        Time column for hover display and trajectory sorting.
        Pass None to disable time in hover and require explicit x_col for line/mean trajectories.
    line_opacity : float, default=0.3
        Opacity for trajectory lines
    line_width : float, default=1.5
        Width for trajectory lines
    show_mean : bool, default=False
        Show mean trajectory per color_by group
    mean_line_width : int, default=6
        Width for mean trajectory lines
    point_opacity : float, default=0.65
        Opacity for scatter points
    point_size : int, default=4
        Size of scatter points
    hover_cols : list, optional
        Additional columns to show in hover tooltip.
        `embryo_id` is always shown by default when present.
    title : str, default="3D Scatter Plot"
        Plot title
    output_path : Path, optional
        If provided, save plot to this path (.html and/or .png)
    axis_labels : dict, optional
        Custom axis labels {coord_name: label}

    Returns
    -------
    go.Figure
        Plotly figure object

    Examples
    --------
    Basic usage (just points):
        >>> fig = plot_3d_scatter(df, ['PCA_1', 'PCA_2', 'PCA_3'], color_by='group')
        >>> fig.show()

    With trajectories:
        >>> fig = plot_3d_scatter(
        ...     df, ['PCA_1', 'PCA_2', 'PCA_3'],
        ...     color_by='phenotype',
        ...     show_lines=True,
        ...     x_col='time'
        ... )

    With mean trajectory per group:
        >>> fig = plot_3d_scatter(
        ...     df, ['PCA_1', 'PCA_2', 'PCA_3'],
        ...     color_by='phenotype',
        ...     show_mean=True,
        ...     x_col='time'
        ... )

    Color by continuous variable:
        >>> fig = plot_3d_scatter(
        ...     df, ['PCA_1', 'PCA_2', 'PCA_3'],
        ...     color_by='time',
        ...     color_continuous=True,
        ...     colorscale='Viridis',
        ...     colorbar_title='Time (hpf)'
        ... )

    Continuous color with experiment filtering:
        >>> fig = plot_3d_scatter(
        ...     df, ['PCA_1', 'PCA_2', 'PCA_3'],
        ...     color_by='predicted_stage_hpf',
        ...     color_continuous=True,
        ...     filter_by_col='experiment_id',
        ...     filter_groups=['20260122']
        ... )

    Continuous color with legend-clickable groups:
        >>> fig = plot_3d_scatter(
        ...     df, ['PCA_1', 'PCA_2', 'PCA_3'],
        ...     color_by='predicted_stage_hpf',
        ...     color_continuous=True,
        ...     group_by='experiment_id'
        ... )
    """
    # Validate inputs
    if len(coords) != 3:
        raise ValueError(f"coords must have exactly 3 elements, got {len(coords)}")

    x_coord, y_coord, z_coord = coords

    for col in coords:
        if col not in df.columns:
            raise ValueError(f"Coordinate column '{col}' not found in DataFrame")

    if color_by not in df.columns:
        raise ValueError(f"color_by column '{color_by}' not found in DataFrame")

    if group_by is not None and group_by not in df.columns:
        raise ValueError(f"group_by column '{group_by}' not found in DataFrame")

    if filter_by_col is not None and filter_by_col not in df.columns:
        raise ValueError(f"filter_by_col column '{filter_by_col}' not found in DataFrame")

    if show_lines and (x_col is None or x_col not in df.columns):
        raise ValueError(
            f"x_col '{x_col}' must exist in DataFrame when show_lines=True"
        )

    if show_mean and (x_col is None or x_col not in df.columns):
        raise ValueError(
            f"x_col '{x_col}' must exist in DataFrame when show_mean=True"
        )

    # Make a copy to avoid modifying original
    df_plot = df.copy()

    # Filter using the requested column (defaults to color_by for backward compatibility).
    filter_col = filter_by_col if filter_by_col is not None else color_by
    if filter_groups is not None:
        df_plot = df_plot[df_plot[filter_col].isin(filter_groups)].copy()

    # Filter by minimum points per line
    if line_by in df_plot.columns:
        line_counts = df_plot.groupby(line_by).size()
        valid_lines = line_counts[line_counts >= min_points_per_line].index
        removed_count = len(line_counts) - len(valid_lines)
        if removed_count > 0:
            print(f"Removed {removed_count} {line_by}s with fewer than {min_points_per_line} points.")
        df_plot = df_plot[df_plot[line_by].isin(valid_lines)].copy()

    # Check if any data remains
    if df_plot.empty:
        print("Warning: No data remaining after filtering.")
        fig = go.Figure()
        fig.update_layout(title="No Data Available")
        return fig

    # Build color lookup (only for categorical coloring)
    if not color_continuous:
        color_lookup = _build_color_lookup(df_plot, color_by, color_palette, color_order)

        # Get groups in order
        if color_order is not None:
            groups = [g for g in color_order if g in df_plot[color_by].unique()]
        else:
            groups = sorted(df_plot[color_by].unique())
    else:
        # For continuous coloring, optionally split into legend-clickable groups.
        color_lookup = None
        if group_by is not None:
            groups = sorted(df_plot[group_by].dropna().unique())
        else:
            groups = [None]  # Single "group" containing all data

    continuous_min = None
    continuous_max = None
    if color_continuous:
        continuous_min = df_plot[color_by].min()
        continuous_max = df_plot[color_by].max()
        if pd.isna(continuous_min) or pd.isna(continuous_max) or continuous_min == continuous_max:
            continuous_min = None
            continuous_max = None

    # Create figure
    fig = go.Figure()

    # Process each group
    for group_idx, group in enumerate(groups):
        if color_continuous:
            if group_by is not None:
                # Continuous coloring + legend groups.
                group_df = df_plot[df_plot[group_by] == group].copy()
            else:
                # Continuous coloring without grouping.
                group_df = df_plot.copy()
        else:
            # For categorical coloring, filter by group
            group_df = df_plot[df_plot[color_by] == group].copy()

        if group_df.empty:
            continue

        # Apply downsampling if specified (only for categorical)
        if not color_continuous and downsample_frac is not None and group in downsample_frac:
            frac = downsample_frac[group]
            if 0 < frac < 1:
                group_df = group_df.sample(frac=frac, random_state=42)

        # Set color (categorical) or color values (continuous)
        if color_continuous:
            color_values = group_df[color_by].values
            color = None  # Not used for continuous
        else:
            color = color_lookup.get(group, STANDARD_PALETTE[0])
            color_values = None

        n_points = len(group_df)
        n_lines = group_df[line_by].nunique() if line_by in group_df.columns else 0

        # Build hover text
        hover_texts = []
        for _, row in group_df.iterrows():
            if color_continuous:
                # Show the continuous value
                text_parts = [f"<b>{color_by}:</b> {row[color_by]:.2f}"]
                if group_by is not None and group_by in group_df.columns and group_by != color_by:
                    text_parts.append(f"<b>{group_by}:</b> {row[group_by]}")
            else:
                # Show the categorical group
                text_parts = [f"<b>{color_by}:</b> {group}"]

            # Always show embryo_id when available.
            if 'embryo_id' in group_df.columns:
                text_parts.append(f"<b>embryo_id:</b> {row['embryo_id']}")
            elif line_by in group_df.columns:
                text_parts.append(f"<b>{line_by}:</b> {row[line_by]}")
            if x_col and x_col in group_df.columns:
                text_parts.append(f"<b>{x_col}:</b> {row[x_col]:.2f}")
            text_parts.extend([
                f"<b>{x_coord}:</b> {row[x_coord]:.3f}",
                f"<b>{y_coord}:</b> {row[y_coord]:.3f}",
                f"<b>{z_coord}:</b> {row[z_coord]:.3f}",
            ])
            if hover_cols:
                for hcol in hover_cols:
                    if hcol in group_df.columns:
                        text_parts.append(f"<b>{hcol}:</b> {row[hcol]}")
            hover_texts.append("<br>".join(text_parts))

        # Build marker dict
        if color_continuous:
            marker_dict = dict(
                size=point_size,
                color=color_values,
                colorscale=colorscale,
                opacity=point_opacity,
                cmin=continuous_min,
                cmax=continuous_max,
                colorbar=dict(
                    title=colorbar_title if colorbar_title else color_by
                ),
                showscale=(group_idx == 0),
            )
            if group_by is not None:
                trace_name = f"{group} (n={n_lines})"
                legend_group = str(group)
            else:
                trace_name = f"Points (n={n_lines})"
                legend_group = "continuous"
        else:
            marker_dict = dict(
                size=point_size,
                color=color,
                opacity=point_opacity,
            )
            trace_name = f"{group} (n={n_lines})"
            legend_group = group

        # Add scatter points
        fig.add_trace(
            go.Scatter3d(
                x=group_df[x_coord],
                y=group_df[y_coord],
                z=group_df[z_coord],
                mode='markers',
                marker=marker_dict,
                name=trace_name,
                legendgroup=legend_group,
                showlegend=True,
                hovertemplate='%{text}<extra></extra>',
                text=hover_texts,
            )
        )

        # Add trajectory lines per line_by group
        if show_lines and line_by in group_df.columns and x_col in group_df.columns:
            for line_id in group_df[line_by].unique():
                line_df = group_df[group_df[line_by] == line_id].sort_values(x_col)

                if len(line_df) < 2:
                    continue

                # For continuous coloring, use a neutral gray for lines
                # For categorical, use group color with transparency
                if color_continuous:
                    line_color = f'rgba(128, 128, 128, {line_opacity})'
                else:
                    line_color = _hex_to_rgba(color, line_opacity)

                fig.add_trace(
                    go.Scatter3d(
                        x=line_df[x_coord],
                        y=line_df[y_coord],
                        z=line_df[z_coord],
                        mode='lines',
                        line=dict(
                            color=line_color,
                            width=line_width,
                        ),
                        name=f"{group} trajectory" if not color_continuous else "trajectory",
                        legendgroup=legend_group,
                        showlegend=False,
                        hoverinfo='skip',
                    )
                )

        # Add mean trajectory
        if show_mean and x_col in group_df.columns:
            # Skip mean trajectory for continuous coloring (doesn't make sense)
            if not color_continuous:
                # Bin by time and compute mean position
                group_df['_time_bin'] = pd.cut(group_df[x_col], bins=50, labels=False)
                mean_df = group_df.groupby('_time_bin').agg({
                    x_col: 'mean',
                    x_coord: 'mean',
                    y_coord: 'mean',
                    z_coord: 'mean',
                }).dropna().sort_values(x_col)

                if len(mean_df) >= 2:
                    fig.add_trace(
                        go.Scatter3d(
                            x=mean_df[x_coord],
                            y=mean_df[y_coord],
                            z=mean_df[z_coord],
                            mode='lines',
                            line=dict(
                                color=color,
                                width=mean_line_width,
                            ),
                            name=f"{group} mean",
                            legendgroup=group,
                            showlegend=False,
                            hoverinfo='skip',
                        )
                    )

    # Set axis labels
    if axis_labels is None:
        axis_labels = {}

    x_label = axis_labels.get(x_coord, x_coord)
    y_label = axis_labels.get(y_coord, y_coord)
    z_label = axis_labels.get(z_coord, z_coord)

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label,
        ),
        title=title,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(
            x=0.01,
            y=0.99,
            bordercolor="Black",
            borderwidth=1,
        ),
    )

    # Save if output_path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save HTML (interactive)
        html_path = output_path.with_suffix('.html')
        fig.write_html(str(html_path))
        print(f"Saved: {html_path}")

        # Try to save PNG (requires kaleido)
        try:
            png_path = output_path.with_suffix('.png')
            fig.write_image(str(png_path))
            print(f"Saved: {png_path}")
        except Exception as e:
            print(f"Could not save PNG (kaleido may not be installed): {e}")

    return fig
