import plotly.express as px
import plotly.graph_objects as go
import math


def format_2d_plotly(fig, axis_labels=None, font_size=14, marker_size=6,
                     theme="dark", dims=None, title="", show_gridlines=True):
    """
    Format a 2D Plotly figure (scatter plot) with consistent styling.

    Parameters:
      fig (go.Figure): A Plotly figure object.
      axis_labels (list): A list with two strings for the x- and y-axis titles.
      font_size (int): Global font size.
      marker_size (int): Marker size for traces.
      theme (str): "dark" or "light".
      dims (list): [height, width] for the figure (default: [600, 800]).
      title (str): Plot title.

    Returns:
      A formatted Plotly figure.
    """
    if dims is None:
        dims = [600, 1000]
    if theme == "dark":
        line_color = "white"
        text_color = "white"
        bk_color = "black"
    elif theme == "light":
        line_color = "black"
        text_color = "black"
        bk_color = "white"
    else:
        line_color = "black"
        text_color = "black"
        bk_color = "white"

    if axis_labels is None:
        axis_labels = ["", ""]

    # Update marker settings for all traces.
    fig.update_traces(marker=dict(size=marker_size,
                                  line=dict(color=line_color, width=1)))

    # Compute a slightly smaller tick font size.
    tick_font_size = font_size#int(font_size * 6 / 7)

    # Create a dictionary for axis formatting (for 2D, do not include 'showbackground').
    axis_format_dict = dict(
        showgrid=show_gridlines,
        zeroline=True,
        gridcolor=line_color,
        linecolor=line_color,
        zerolinecolor=line_color,
        tickfont=dict(size=tick_font_size)
    )

    # Create separate dictionaries for x- and y-axes with titles.
    xaxis_format = axis_format_dict.copy()
    xaxis_format["title"] = axis_labels[0]
    yaxis_format = axis_format_dict.copy()
    yaxis_format["title"] = axis_labels[1]

    # Check if axis ranges have been manually specified.
    x_range = fig.layout.xaxis.range if fig.layout.xaxis and fig.layout.xaxis.range else None
    y_range = fig.layout.yaxis.range if fig.layout.yaxis and fig.layout.yaxis.range else None

    if x_range is not None and y_range is not None:
        # Calculate extents.
        x_extent = x_range[1] - x_range[0]
        y_extent = y_range[1] - y_range[0]
        # Set up equal scaling using scaleanchor and scaleratio.
        fig.update_layout(
            xaxis=dict(**xaxis_format, range=x_range, scaleanchor="y", scaleratio=x_extent / y_extent),
            yaxis=dict(**yaxis_format, range=y_range)
        )
    else:
        fig.update_layout(
            xaxis=xaxis_format,
            yaxis=yaxis_format
        )

    # Set overall layout parameters.
    fig.update_layout(
        width=dims[1],
        height=dims[0],
        title=title,
        font=dict(color=text_color, family="Arial, sans-serif", size=font_size),
        plot_bgcolor=bk_color,
        paper_bgcolor=bk_color
    )

    try:
        fig.update_traces(
            error_x=dict(color="white", width=0),
            error_y=dict(color="white", width=0)
        )
    except:
        pass

    if any(hasattr(trace, "marker") and getattr(trace, "showlegend", True)
           for trace in fig.data):
        # Update the colorbar to be positioned in the bottom half of the y axis
        fig.update_layout(
            coloraxis_colorbar=dict(
                x=1,  # move the colorbar to the right edge
                y=0,  # position lower (bottom half)
                len=0.5,  # adjust the length as needed
                yanchor="bottom",
            )
        )
    else:
        fig.update_layout(width=dims[1], height=dims[0],
                          title=title,
                          coloraxis_colorbar=dict(
                              x=1,  # Increase x to move the colorbar rightwards
                              y=0.5,  # Center vertically (default is often around 0.5)
                              len=0.5  # Adjust the length if needed
                          ))

    return fig

def format_3d_plotly(fig, axis_labels=None, font_size=14, marker_size=6, show_gridlines=True,
                     aspectmode="data", eye=None, theme="dark", dims=None, title=""):

    if dims is None:
        dims = [600, 800]
    if theme == "dark":
        line_color = "white"
        text_color = "white"
        bk_color = "black"
    elif theme == "light":
        line_color = "black"
        text_color = "black"
        bk_color = "white"

    if axis_labels is None:
        axis_labels = ["", "", ""]

    if eye is None:
        eye = dict(x=1.5, y=1.5, z=1.5)

    fig.update_traces(marker=dict(size=marker_size, line=dict(color=line_color, width=1)))

    tick_font_size = int(font_size * 6 / 7)
    axis_format_dict = dict(showbackground=False,
                            showgrid=show_gridlines,
                            zeroline=True,
                            gridcolor=line_color,
                            linecolor=line_color,
                            zerolinecolor=line_color,
                            tickfont=dict(size=tick_font_size))

    dict_list = [axis_format_dict.copy(), axis_format_dict.copy(), axis_format_dict.copy()]
    for i, d in enumerate(dict_list):
        d["title"] = axis_labels[i]

    # check to see if axis ranges have been manually specified
    # Get the ranges

    x_range = fig.layout.scene.xaxis.range
    y_range = fig.layout.scene.yaxis.range
    z_range = fig.layout.scene.zaxis.range

    if x_range is not None:
        # Calculate the extents (difference between max and min)
        x_extent = x_range[1] - x_range[0]
        y_extent = y_range[1] - y_range[0]
        z_extent = z_range[1] - z_range[0]

        fig.update_layout(scene=dict(aspectmode="manual",
                                     aspectratio=dict(x=x_extent, y=y_extent, z=z_extent),
                                     xaxis=dict_list[0],
                                     yaxis=dict_list[1],
                                     zaxis=dict_list[2]
                                     ))

    else:
        fig.update_layout(scene=dict(aspectmode=aspectmode,
                                     xaxis=dict_list[0],
                                     yaxis=dict_list[1],
                                     zaxis=dict_list[2]
                          ))

    # Check if any trace with a marker has its legend shown


    fig.update_layout(
        font=dict(color=text_color, family="Arial, sans-serif", size=font_size),
        plot_bgcolor=bk_color,  # Background inside the plotting area
        paper_bgcolor=bk_color  # Background outside the plotting area (around the plot)
    )

    fig.update_layout(
        scene_camera=dict(
            eye=eye  # Adjust these values as needed.
        )
    )

    if any(hasattr(trace, "marker") and getattr(trace, "showlegend", True)
           for trace in fig.data):
        # Update the colorbar to be positioned in the bottom half of the y axis
        fig.update_layout(
            coloraxis_colorbar=dict(
                x=1,  # move the colorbar to the right edge
                y=0,  # position lower (bottom half)
                len=0.5,  # adjust the length as needed
                yanchor="bottom",
            )
        )
    else:
        fig.update_layout(width=dims[1], height=dims[0],
                      title=title,
                      coloraxis_colorbar=dict(
                          x=1,  # Increase x to move the colorbar rightwards
                          y=0.5,  # Center vertically (default is often around 0.5)
                          len=0.5  # Adjust the length if needed
                      ))

    return fig


def rotate_figure(fig, zoom_factor=1.0, z_rotation=0, elev_rotation=0):
    """
    Adjust the camera perspective of a Plotly 3D figure.

    Parameters:
      fig (go.Figure): Plotly figure object with a 3D scene.
      zoom_factor (float): Multiplicative factor to scale the camera's distance from the center.
                           Values < 1 zoom in; values > 1 zoom out.
      z_rotation (float): Rotation (in degrees) about the z-axis (i.e. in the x-y plane).
      elev_rotation (float): Rotation (in degrees) to change the elevation (i.e. the polar angle).
                             Positive values will tilt the camera; negative values will lower it.

    Returns:
      The updated Plotly figure.
    """

    # Get current camera eye position; if not set, use a default.
    try:
        current_eye = fig.layout.scene.camera.eye
        # Assume current_eye is a dict with keys "x", "y", and "z".
        x = current_eye.get("x", 1.25)
        y = current_eye.get("y", 1.25)
        z = current_eye.get("z", 1.25)
    except Exception:
        x, y, z = 1.25, 1.25, 1.25

    # Apply zoom: scale each coordinate by zoom_factor.
    # (A zoom_factor < 1 moves the camera closer (zoom in), > 1 moves it away.)
    x /= zoom_factor
    y /= zoom_factor
    z /= zoom_factor

    # Convert the eye position to spherical coordinates.
    r = math.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = math.atan2(y, x)  # azimuth angle in the x-y plane.
    # phi is the polar angle from the positive z-axis.
    phi = math.acos(z / r) if r != 0 else 0

    # Apply rotation about the z axis by adjusting the azimuth (theta).
    theta += math.radians(z_rotation)

    # Apply elevation rotation by adjusting the polar angle (phi).
    phi += math.radians(elev_rotation)
    # Clamp phi to be between 0 and pi.
    phi = max(0, min(math.pi, phi))

    # Convert back to Cartesian coordinates.
    new_x = r * math.sin(phi) * math.cos(theta)
    new_y = r * math.sin(phi) * math.sin(theta)
    new_z = r * math.cos(phi)

    # Update the camera in the figure.
    new_camera = {"eye": {"x": new_x, "y": new_y, "z": new_z}}
    fig.update_layout(scene_camera=new_camera)

    return fig

