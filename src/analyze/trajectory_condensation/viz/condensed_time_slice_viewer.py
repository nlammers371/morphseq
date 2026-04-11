"""
condensed_time_slice_viewer.py
-------------
Condensed trajectory time-slice HTML viewer.

Public API
----------
time_slice_html(positions, mask, time_values, ...) -> go.Figure
    Two-panel interactive figure with a time-bin slider:
      Left  — stacked 3D trajectory cloud (all time, dimmed) with the
              current bin's points highlighted.
      Right — 2D scatter of only the current time bin.
    Outputs a self-contained HTML file; no server required.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def time_slice_html(
    positions: np.ndarray,
    mask: np.ndarray,
    time_values: np.ndarray,
    labels: np.ndarray | None = None,
    color_map: dict[str, str] | None = None,
    embryo_ids: np.ndarray | None = None,
    output_path: str | Path | None = None,
    title: str = "Time slice",
    add_slice_panel: bool = True,
    width: int = 1500,
    height: int = 700,
    trajectory_trace_alpha: float = 0.16,
    current_time_marker_alpha: float = 1.0,
    slice_marker_alpha: float = 0.75,
    trajectory_marker_alpha: float = 0.15,
    current_time_marker_size: int = 6,
    slice_marker_size: int = 6,
    trajectory_marker_size: int = 2,
    alpha_bg: float | None = None,
    alpha_bg_marker: float | None = None,
    alpha_highlight: float | None = None,
    alpha_2d: float | None = None,
    marker_size_bg: int | None = None,
    marker_size_highlight: int | None = None,
    marker_size_2d: int | None = None,
) -> "go.Figure":
    """Build an interactive Plotly figure with a time-bin slider.

    Parameters
    ----------
    positions : (N_e, T, 2)
    mask : (N_e, T) bool
    time_values : (T,) float — hpf values
    labels : (N_e,) str, optional
    color_map : label → hex color string, auto-generated if None
    embryo_ids : (N_e,) str, optional — shown in hover tooltip
    output_path : if given, writes self-contained HTML and returns the Figure
    title : figure title
    add_slice_panel : if True (default), show a 2D cross-section panel on the
        right that updates with the slider. If False, only the 3D overview is
        shown, filling the full figure width.
    trajectory_trace_alpha : opacity of the background trajectory lines in the 3D overview
    current_time_marker_alpha : opacity of highlighted current-time points in the 3D overview
    slice_marker_alpha : opacity of points in the 2D slice panel
    trajectory_marker_alpha : opacity of small background trajectory markers in the 3D overview
    current_time_marker_size : size of highlighted current-time points in the 3D overview
    slice_marker_size : size of points in the 2D slice panel
    trajectory_marker_size : size of small background trajectory markers in the 3D overview
    alpha_bg / alpha_bg_marker / alpha_highlight / alpha_2d : deprecated opacity aliases
    marker_size_bg / marker_size_highlight / marker_size_2d : deprecated size aliases

    Returns
    -------
    go.Figure — caller can further customise or call .write_html() themselves.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("plotly is required for time_slice_html")

    positions = np.asarray(positions, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    time_values = np.asarray(time_values, dtype=float)
    N_e, T, _ = positions.shape
    if alpha_bg is not None:
        trajectory_trace_alpha = float(alpha_bg)
    if alpha_bg_marker is not None:
        trajectory_marker_alpha = float(alpha_bg_marker)
    if alpha_highlight is not None:
        current_time_marker_alpha = float(alpha_highlight)
    if alpha_2d is not None:
        slice_marker_alpha = float(alpha_2d)
    if marker_size_bg is not None:
        trajectory_marker_size = int(marker_size_bg)
    if marker_size_highlight is not None:
        current_time_marker_size = int(marker_size_highlight)
    if marker_size_2d is not None:
        slice_marker_size = int(marker_size_2d)

    color_map = _resolve_color_map(labels, color_map)
    unique_labels = sorted(color_map.keys()) if labels is not None else [None]

    # Fixed axis limits.
    obs_xy = positions[mask]
    xy_pad = 0.05
    x_range = obs_xy[:, 0].max() - obs_xy[:, 0].min() or 1.0
    y_range = obs_xy[:, 1].max() - obs_xy[:, 1].min() or 1.0
    xlim = [obs_xy[:, 0].min() - xy_pad * x_range, obs_xy[:, 0].max() + xy_pad * x_range]
    ylim = [obs_xy[:, 1].min() - xy_pad * y_range, obs_xy[:, 1].max() + xy_pad * y_range]
    zlim = [float(time_values[0]), float(time_values[-1])]

    # -----------------------------------------------------------------------
    # Subplot layout — 1 or 2 columns depending on add_slice_panel.
    # -----------------------------------------------------------------------
    if add_slice_panel:
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "scene"}, {"type": "xy"}]],
            subplot_titles=("3D overview", "Current time slice"),
            horizontal_spacing=0.06,
            column_widths=[0.55, 0.45],
        )
    else:
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"type": "scene"}]],
            subplot_titles=("3D overview",),
        )

    for ann in fig.layout.annotations:
        ann.font.size = 18

    # -----------------------------------------------------------------------
    # Legend proxy traces — keep clickable phenotype entries fully opaque even
    # though the background trajectory cloud is intentionally dimmed.
    # -----------------------------------------------------------------------
    for lbl in unique_labels:
        color = color_map.get(lbl, "#4C78A8") if lbl is not None else "#4C78A8"
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode="lines+markers",
            name=str(lbl) if lbl is not None else "unknown",
            legendgroup=str(lbl),
            showlegend=True,
            visible=True,
            line=dict(color=color, width=3),
            marker=dict(color=color, size=6, opacity=1.0),
            hoverinfo="skip",
        ), row=1, col=1)

    # -----------------------------------------------------------------------
    # Static background traces — all trajectories, dimmed.
    # -----------------------------------------------------------------------
    for lbl in unique_labels:
        if lbl is not None:
            emb_idx = np.where(labels == lbl)[0]
            color = color_map.get(lbl, "#4C78A8")
        else:
            emb_idx = np.arange(N_e)
            color = "#4C78A8"

        xs, ys, zs, ids_hover = [], [], [], []
        for i in emb_idx:
            obs_t = np.where(mask[i])[0]
            if len(obs_t) < 2:
                continue
            xs.extend(positions[i, obs_t, 0].tolist() + [None])
            ys.extend(positions[i, obs_t, 1].tolist() + [None])
            zs.extend(time_values[obs_t].tolist() + [None])
            eid = str(embryo_ids[i]) if embryo_ids is not None else str(i)
            ids_hover.extend([eid] * len(obs_t) + [""])

        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines+markers",
            name=str(lbl) if lbl is not None else "unknown",
            legendgroup=str(lbl),
            showlegend=False,
            line=dict(color=color, width=1.5),
            marker=dict(
                color=color,
                size=trajectory_marker_size,
                opacity=min(trajectory_marker_alpha / max(trajectory_trace_alpha, 1e-6), 1.0),
            ),
            opacity=trajectory_trace_alpha,
            text=ids_hover,
            hovertemplate="<b>%{text}</b><br>dim1=%{x:.3f}<br>dim2=%{y:.3f}<br>time=%{z:.1f} hpf<extra></extra>",
        ), row=1, col=1)

    n_bg = len(fig.data)

    # -----------------------------------------------------------------------
    # Per-frame animated traces.
    # -----------------------------------------------------------------------
    def _make_highlight_trace(lbl, xs, ys, zs, ids_h) -> "go.Scatter3d":
        color = color_map.get(lbl, "#4C78A8") if lbl is not None else "#4C78A8"
        return go.Scatter3d(
            x=xs, y=ys, z=zs, mode="markers",
            name=str(lbl) if lbl is not None else "unknown",
            legendgroup=str(lbl), showlegend=False,
            marker=dict(color=color, size=current_time_marker_size, opacity=current_time_marker_alpha,
                        line=dict(color="white", width=0.5)),
            text=ids_h,
            hovertemplate="<b>%{text}</b><br>dim1=%{x:.3f}<br>dim2=%{y:.3f}<br>time=%{z:.1f} hpf<extra></extra>",
        )

    def _make_scatter2d_trace(lbl, xs, ys, ids_h) -> "go.Scatter":
        color = color_map.get(lbl, "#4C78A8") if lbl is not None else "#4C78A8"
        return go.Scatter(
            x=xs, y=ys, mode="markers",
            name=str(lbl) if lbl is not None else "unknown",
            legendgroup=str(lbl), showlegend=False,
            marker=dict(color=color, size=slice_marker_size, opacity=slice_marker_alpha, line=dict(width=0)),
            text=ids_h,
            hovertemplate="<b>%{text}</b><br>dim1=%{x:.3f}<br>dim2=%{y:.3f}<extra></extra>",
        )

    def _frame_traces(t_idx: int) -> list:
        obs = np.where(mask[:, t_idx])[0]
        z = float(time_values[t_idx])
        hl, sc = [], []
        for lbl in unique_labels:
            idx = obs[labels[obs] == lbl] if lbl is not None else obs
            xs = positions[idx, t_idx, 0].tolist()
            ys = positions[idx, t_idx, 1].tolist()
            ids_h = [str(embryo_ids[i]) if embryo_ids is not None else str(i) for i in idx]
            zs = [z] * len(idx)
            hl.append(_make_highlight_trace(lbl, xs, ys, zs, ids_h))
            if add_slice_panel:
                sc.append(_make_scatter2d_trace(lbl, xs, ys, ids_h))
        return hl + sc

    # Seed first-frame placeholder traces.
    first_traces = _frame_traces(0)
    n_hl = len(unique_labels)
    for i, tr in enumerate(first_traces):
        col = 2 if (add_slice_panel and i >= n_hl) else 1
        fig.add_trace(tr, row=1, col=col)

    anim_trace_indices = list(range(n_bg, len(fig.data)))

    # Build frames + slider steps.
    frames, slider_steps = [], []
    for t_idx in range(T):
        z = float(time_values[t_idx])
        frames.append(go.Frame(
            data=_frame_traces(t_idx),
            traces=anim_trace_indices,
            name=str(t_idx),
        ))
        slider_steps.append(dict(
            args=[[str(t_idx)], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
            label=f"{z:.0f}",
            method="animate",
        ))

    fig.frames = frames

    # -----------------------------------------------------------------------
    # Layout
    # -----------------------------------------------------------------------
    scene_common = dict(
        xaxis=dict(title="dim 1", range=xlim),
        yaxis=dict(title="dim 2", range=ylim),
        zaxis=dict(title="time (hpf)", range=zlim),
        camera=dict(eye=dict(x=1.4, y=1.4, z=1.0)),
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=1.2),
    )

    layout_kwargs: dict = dict(
        title=title,
        template="plotly_white",
        width=width,
        height=height,
        legend=dict(
            itemsizing="constant",
            tracegroupgap=2,
            groupclick="togglegroup",
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="rgba(40,40,40,0.45)",
            borderwidth=1,
        ),
        scene=scene_common,
        margin=dict(t=160, b=120),
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix="time: ", suffix=" hpf", font=dict(size=13), visible=True),
            pad=dict(t=50, b=10, l=120),
            steps=slider_steps,
            x=0.08, len=0.92,
        )],
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=0.0, x=0.0, xanchor="left", yanchor="bottom",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=400, redraw=True),
                                      fromcurrent=True, mode="immediate")]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
            ],
        )],
    )

    if add_slice_panel:
        layout_kwargs["xaxis"] = dict(title="dim 1", range=xlim, showgrid=True)
        layout_kwargs["yaxis"] = dict(title="dim 2", range=ylim, showgrid=True)

    fig.update_layout(**layout_kwargs)

    # -----------------------------------------------------------------------
    # Annotations: raise subplot titles above the plot; add dynamic hpf label
    # below the right title (add_slice_panel only).
    # -----------------------------------------------------------------------
    title_y = 1.10
    for ann in fig.layout.annotations:
        ann.y = title_y
        ann.yanchor = "bottom"

    if add_slice_panel:
        right_title_x = fig.layout.annotations[1].x
        hpf_y = 1.03

        def _hpf_annotation(z: float) -> dict:
            return dict(
                text=f"Time  <b>{z:.0f} hpf</b>",
                x=right_title_x, y=hpf_y,
                xref="paper", yref="paper",
                xanchor="center", yanchor="bottom",
                showarrow=False, font=dict(size=16),
            )

        fig.add_annotation(**_hpf_annotation(float(time_values[0])))

        ann0 = dict(**{k: v for k, v in fig.layout.annotations[0].to_plotly_json().items()})
        ann1 = dict(**{k: v for k, v in fig.layout.annotations[1].to_plotly_json().items()})
        for frame in fig.frames:
            z_frame = float(time_values[int(frame.name)])
            frame.layout = go.Layout(annotations=[ann0, ann1, _hpf_annotation(z_frame)])

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path), include_plotlyjs="cdn")
        print(f"Saved time-slice HTML: {output_path}")

    return fig


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _resolve_color_map(
    labels: np.ndarray | None,
    color_map: dict[str, str] | None,
) -> dict[str, str]:
    if labels is None:
        return {}
    if color_map is not None:
        return {str(k): v for k, v in color_map.items()}
    try:
        import plotly.express as px
        palette = px.colors.qualitative.Plotly
    except ImportError:
        palette = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B",
                   "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC"]
    unique = sorted(np.unique(labels).tolist())
    return {lbl: palette[i % len(palette)] for i, lbl in enumerate(unique)}

