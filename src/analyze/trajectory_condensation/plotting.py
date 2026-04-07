"""
plotting.py
-----------
Visualization of ordered 2D coordinate slices across time.

Works with any source of (N_e, T, 2) coordinates — AlignedUMAP, condensation
output, PCA, or anything else. No knowledge of CosmologyData or any upstream
model lives here.

Four rendering modes:
  panels       — one 2D scatter per time bin (faceted)
  trajectories — embryo paths through (x, y) connected across time
  color_by_time — single 2D scatter, points colored by hpf
  stacked_3d   — (x, y, time) with time as z; explicitly labeled as metadata

Public API:
  plot(positions, mask, time_values, ...)          — dispatch by mode
  plot_panels(positions, mask, time_values, ...)
  plot_trajectories(positions, mask, time_values, ...)
  plot_color_by_time(positions, mask, time_values, ...)
  plot_stacked_3d(positions, mask, time_values, ...)
"""
from __future__ import annotations

from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def plot(
    positions: np.ndarray,
    mask: np.ndarray,
    time_values: np.ndarray,
    labels: np.ndarray | None = None,
    mode: Literal["panels", "trajectories", "color_by_time", "stacked_3d"] = "trajectories",
    color_map: dict[str, str] | None = None,
    **kwargs,
):
    """Dispatch to the requested rendering mode.

    Parameters
    ----------
    positions : (N_e, T, 2)
    mask : (N_e, T) bool
    time_values : (T,) float — hpf or any ordered time axis
    labels : (N_e,) str, optional — used for color grouping
    mode : one of panels | trajectories | color_by_time | stacked_3d
    color_map : label → color string. Auto-generated if None.
    **kwargs : passed through to the mode function
    """
    _validate_inputs(positions, mask, time_values)
    dispatch = {
        "panels": plot_panels,
        "trajectories": plot_trajectories,
        "color_by_time": plot_color_by_time,
        "stacked_3d": plot_stacked_3d,
    }
    if mode not in dispatch:
        raise ValueError(f"Unknown mode {mode!r}. Choose from: {list(dispatch)}")
    return dispatch[mode](
        positions=positions,
        mask=mask,
        time_values=time_values,
        labels=labels,
        color_map=color_map,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Mode 1: Faceted panels
# ---------------------------------------------------------------------------

def plot_panels(
    positions: np.ndarray,
    mask: np.ndarray,
    time_values: np.ndarray,
    labels: np.ndarray | None = None,
    color_map: dict[str, str] | None = None,
    snapshot_times: list[float] | None = None,
    ncols: int = 4,
    panel_size: float = 3.0,
    point_size: float = 12.0,
    alpha: float = 0.7,
    title: str = "",
):
    """One 2D scatter per time bin.

    Parameters
    ----------
    snapshot_times : subset of time_values to show. Shows all if None.
    ncols : number of columns in the panel grid.

    Returns
    -------
    (fig, axes)
    """
    import matplotlib.pyplot as plt

    times = _select_times(time_values, snapshot_times)
    color_map = resolve_color_map(labels, color_map)
    nrows = int(np.ceil(len(times) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * panel_size, nrows * panel_size),
        squeeze=False,
    )

    for ax_idx, t_val in enumerate(times):
        t = np.searchsorted(time_values, t_val)
        ax = axes[ax_idx // ncols][ax_idx % ncols]
        obs = np.where(mask[:, t])[0]

        if labels is not None:
            for label in np.unique(labels[obs]):
                idx = obs[labels[obs] == label]
                ax.scatter(
                    positions[idx, t, 0], positions[idx, t, 1],
                    c=color_map[label], s=point_size, alpha=alpha,
                    label=label, linewidths=0,
                )
        else:
            ax.scatter(
                positions[obs, t, 0], positions[obs, t, 1],
                c="steelblue", s=point_size, alpha=alpha, linewidths=0,
            )

        ax.set_title(f"{t_val:.0f} hpf", fontsize=9)
        ax.set_xlabel("dim 1", fontsize=7)
        ax.set_ylabel("dim 2", fontsize=7)
        ax.tick_params(labelsize=6)

    # hide unused panels
    for ax_idx in range(len(times), nrows * ncols):
        axes[ax_idx // ncols][ax_idx % ncols].set_visible(False)

    if labels is not None:
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=color_map[l], markersize=6, label=l)
            for l in sorted(color_map)
        ]
        fig.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.8)

    if title:
        fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Mode 2: Trajectory lines
# ---------------------------------------------------------------------------

def plot_trajectories(
    positions: np.ndarray,
    mask: np.ndarray,
    time_values: np.ndarray,
    labels: np.ndarray | None = None,
    color_map: dict[str, str] | None = None,
    min_obs: int = 2,
    alpha_line: float = 0.25,
    alpha_point: float = 0.6,
    linewidth: float = 0.8,
    point_size: float = 8.0,
    figsize: tuple[float, float] = (8, 7),
    title: str = "",
):
    """Per-embryo trajectory lines through (x, y) connected across time.

    Parameters
    ----------
    min_obs : minimum number of observed time bins to draw a line for an embryo.

    Returns
    -------
    (fig, ax)
    """
    import matplotlib.pyplot as plt

    color_map = resolve_color_map(labels, color_map)
    fig, ax = plt.subplots(figsize=figsize)
    N_e = positions.shape[0]

    groups = np.unique(labels) if labels is not None else [None]

    for group in groups:
        if group is not None:
            embryo_idx = np.where(labels == group)[0]
            color = color_map[group]
        else:
            embryo_idx = np.arange(N_e)
            color = "steelblue"

        first = True
        for i in embryo_idx:
            obs_t = np.where(mask[i, :])[0]
            if len(obs_t) < min_obs:
                continue
            xs = positions[i, obs_t, 0]
            ys = positions[i, obs_t, 1]
            ax.plot(
                xs, ys,
                color=color, alpha=alpha_line, linewidth=linewidth,
                label=group if first else "_nolegend_",
            )
            ax.scatter(xs, ys, color=color, alpha=alpha_point, s=point_size, linewidths=0)
            first = False

    if labels is not None:
        ax.legend(loc="best", fontsize=8, framealpha=0.7)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_title(title or "Embryo trajectories")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Mode 3: Color by time
# ---------------------------------------------------------------------------

def plot_color_by_time(
    positions: np.ndarray,
    mask: np.ndarray,
    time_values: np.ndarray,
    labels: np.ndarray | None = None,
    color_map: dict[str, str] | None = None,
    cmap_name: str = "viridis",
    point_size: float = 12.0,
    alpha: float = 0.7,
    figsize: tuple[float, float] = (8, 7),
    title: str = "",
):
    """Single 2D scatter with points colored by time bin.

    `labels` and `color_map` are ignored here — time is the color axis.
    Pass labels=... only if you want per-group marker shapes (not yet implemented).

    Returns
    -------
    (fig, ax)
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(figsize=figsize)
    T = len(time_values)
    norm = mcolors.Normalize(vmin=time_values.min(), vmax=time_values.max())
    cmap = cm.get_cmap(cmap_name)

    for t, hpf in enumerate(time_values):
        obs = np.where(mask[:, t])[0]
        if len(obs) == 0:
            continue
        ax.scatter(
            positions[obs, t, 0], positions[obs, t, 1],
            c=[cmap(norm(hpf))] * len(obs),
            s=point_size, alpha=alpha, linewidths=0,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="hpf")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_title(title or "Embedding colored by time")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Mode 4: Stacked 3D (x, y, time)
# ---------------------------------------------------------------------------

def plot_stacked_3d(
    positions: np.ndarray,
    mask: np.ndarray,
    time_values: np.ndarray,
    labels: np.ndarray | None = None,
    color_map: dict[str, str] | None = None,
    connect_trajectories: bool = True,
    min_obs: int = 2,
    alpha_line: float = 0.2,
    alpha_point: float = 0.6,
    linewidth: float = 0.6,
    point_size: float = 8.0,
    figsize: tuple[float, float] = (9, 8),
    title: str = "",
    elev: float = 25.0,
    azim: float = -60.0,
):
    """Stacked 3D view: x = dim1, y = dim2, z = time.

    z is time metadata stacked for visualization — NOT a learned third
    embedding dimension. AlignedUMAP (or any 2D method) produced only
    dim1 and dim2; time is added here for visual continuity across slices.

    Parameters
    ----------
    connect_trajectories : draw per-embryo lines connecting successive time bins.
    elev, azim : initial viewing angle.

    Returns
    -------
    (fig, ax)
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    color_map = resolve_color_map(labels, color_map)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    N_e = positions.shape[0]

    groups = np.unique(labels) if labels is not None else [None]

    for group in groups:
        if group is not None:
            embryo_idx = np.where(labels == group)[0]
            color = color_map[group]
        else:
            embryo_idx = np.arange(N_e)
            color = "steelblue"

        first = True
        for i in embryo_idx:
            obs_t = np.where(mask[i, :])[0]
            if len(obs_t) < min_obs:
                continue
            xs = positions[i, obs_t, 0]
            ys = positions[i, obs_t, 1]
            zs = time_values[obs_t]

            if connect_trajectories:
                ax.plot(
                    xs, ys, zs,
                    color=color, alpha=alpha_line, linewidth=linewidth,
                    label=group if first else "_nolegend_",
                )
            ax.scatter(xs, ys, zs, color=color, alpha=alpha_point, s=point_size)
            first = False

    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_zlabel("time (hpf)")
    ax.view_init(elev=elev, azim=azim)

    subtitle = "z = time (hpf) — stacked for visualization, not a learned 3D manifold"
    ax.set_title((title + "\n" + subtitle) if title else subtitle, fontsize=9)

    if labels is not None:
        ax.legend(loc="upper left", fontsize=8, framealpha=0.7)

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_inputs(
    positions: np.ndarray,
    mask: np.ndarray,
    time_values: np.ndarray,
) -> None:
    assert positions.ndim == 3 and positions.shape[2] == 2, (
        f"positions must be (N_e, T, 2), got {positions.shape}"
    )
    assert mask.shape == positions.shape[:2], (
        f"mask shape {mask.shape} != positions (N_e, T) = {positions.shape[:2]}"
    )
    assert len(time_values) == positions.shape[1], (
        f"time_values length {len(time_values)} != T={positions.shape[1]}"
    )


def resolve_color_map(
    labels: np.ndarray | None,
    color_map: dict[str, str] | None,
) -> dict[str, str]:
    if labels is None:
        return {}
    if color_map is not None:
        return color_map
    import matplotlib.pyplot as plt
    unique = sorted(np.unique(labels))
    cmap = plt.get_cmap("tab10")
    return {l: cmap(i % 10) for i, l in enumerate(unique)}


def _select_times(
    time_values: np.ndarray,
    snapshot_times: list[float] | None,
) -> list[float]:
    if snapshot_times is None:
        return list(time_values)
    valid = []
    for t in snapshot_times:
        nearest = time_values[np.argmin(np.abs(time_values - t))]
        if nearest not in valid:
            valid.append(nearest)
    return valid
