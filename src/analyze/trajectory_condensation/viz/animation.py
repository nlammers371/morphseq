"""
animation.py
------------
Video generation for ordered 2D coordinate slices across time.

Two distinct videos, each answering a different question:

  Video A — final structure, slow rotation
    animate_rotation(positions, mask, time_values, ...)
    "What is the geometry of the final structure?"
    Use for: presentation, branch inspection, geometry sanity check.

  Video B — optimization progress, fixed viewpoint
    animate_iterations(position_history, mask, time_values, ...)
    "How did the structure form? Is it converging?"
    Use for: debugging, hyperparameter tuning, convergence assessment.

Both accept the same minimal input contract as plotting.py:
  positions : (N_e, T, 2)
  mask      : (N_e, T) bool
  time_values : (T,) float

position_history for Video B is (n_saved_iters, N_e, T, 2) — save every
N iterations during dynamics, not every step.

Important: z = time (hpf) in all 3D views. This is stacked visualization
metadata, not a learned embedding coordinate. All figure titles say so.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Video A: final structure, slow rotation
# ---------------------------------------------------------------------------

def animate_rotation(
    positions: np.ndarray,
    mask: np.ndarray,
    time_values: np.ndarray,
    labels: np.ndarray | None = None,
    color_map: dict[str, str] | None = None,
    output_path: str | Path = "rotation.mp4",
    n_frames: int = 120,
    elev: float = 25.0,
    azim_start: float = -60.0,
    azim_end: float = 300.0,
    fps: int = 24,
    dpi: int = 120,
    figsize: tuple[float, float] = (8, 7),
    point_size: float = 8.0,
    alpha_point: float = 0.6,
    alpha_line: float = 0.2,
    linewidth: float = 0.6,
    min_obs: int = 2,
    title: str = "",
):
    """Slow camera orbit around the final stacked 3D structure.

    x = dim1, y = dim2, z = time (hpf).
    Camera rotates from azim_start to azim_end over n_frames.
    Elevation is fixed (or can be given as a float or (n_frames,) array).

    Parameters
    ----------
    positions : (N_e, T, 2) — final positions
    output_path : .mp4 or .gif
    n_frames : total number of frames
    azim_start, azim_end : azimuth range in degrees
    fps : frames per second
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    _validate_inputs(positions, mask, time_values)
    color_map = _resolve_color_map(labels, color_map)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    _draw_stacked_3d(
        ax, positions, mask, time_values, labels, color_map,
        point_size=point_size, alpha_point=alpha_point,
        alpha_line=alpha_line, linewidth=linewidth, min_obs=min_obs,
    )
    subtitle = "z = time (hpf) — visualization metadata, not a learned coordinate"
    ax.set_title((title + "\n" + subtitle) if title else subtitle, fontsize=8)
    ax.view_init(elev=elev, azim=azim_start)

    azimuths = np.linspace(azim_start, azim_end, n_frames)

    def update(frame):
        ax.view_init(elev=elev, azim=azimuths[frame])
        return []

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 / fps, blit=False
    )
    _save_animation(anim, output_path, fps, dpi)
    plt.close(fig)
    print(f"Saved rotation video: {output_path}")


# ---------------------------------------------------------------------------
# Video B: optimization progress, fixed viewpoint
# ---------------------------------------------------------------------------

def animate_iterations(
    position_history: np.ndarray,
    mask: np.ndarray,
    time_values: np.ndarray,
    snapshot_iters: list[int] | None = None,
    labels: np.ndarray | None = None,
    color_map: dict[str, str] | None = None,
    output_path: str | Path = "iterations.mp4",
    elev: float = 25.0,
    azim: float = -60.0,
    azim_end: float = 300.0,
    rotation: bool = False,
    fps: int = 12,
    dpi: int = 120,
    figsize: tuple[float, float] = (8, 7),
    point_size: float = 8.0,
    alpha_point: float = 0.6,
    alpha_line: float = 0.2,
    linewidth: float = 0.6,
    min_obs: int = 2,
    title: str = "",
):
    """Fixed-viewpoint animation of condensation optimization progress.

    Each frame shows the stacked 3D structure at one saved iteration.
    By default the camera is fixed; set rotation=True to orbit the camera
    from azim to azim_end across all frames.

    Parameters
    ----------
    position_history : (n_saved_iters, N_e, T, 2)
        Positions saved during optimization at regular intervals.
        Produce by passing save_every=N to run_condensation (future flag).
    snapshot_iters : list of iteration numbers, length n_saved_iters.
        Used for frame titles. E.g. [0, 10, 20, ...].
    azim_end : float
        Final azimuth angle (degrees) when rotation=True. Ignored otherwise.
    rotation : bool
        If True, linearly sweep azimuth from azim to azim_end across frames.
        Default False preserves the original fixed-camera behavior.
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    n_iters = position_history.shape[0]
    _validate_inputs(position_history[0], mask, time_values)
    color_map = _resolve_color_map(labels, color_map)

    if snapshot_iters is None:
        snapshot_iters = list(range(n_iters))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)

    def update(frame):
        ax.cla()
        positions = position_history[frame]
        _draw_stacked_3d(
            ax, positions, mask, time_values, labels, color_map,
            point_size=point_size, alpha_point=alpha_point,
            alpha_line=alpha_line, linewidth=linewidth, min_obs=min_obs,
        )
        subtitle = "z = time (hpf) — visualization metadata, not a learned coordinate"
        iter_str = f"iter {snapshot_iters[frame]}"
        ax.set_title(
            ((title + " | " + iter_str) if title else iter_str) + "\n" + subtitle,
            fontsize=8,
        )
        if rotation:
            current_azim = azim + (azim_end - azim) * frame / max(n_iters - 1, 1)
        else:
            current_azim = azim
        ax.view_init(elev=elev, azim=current_azim)
        return []

    anim = animation.FuncAnimation(
        fig, update, frames=n_iters, interval=1000 / fps, blit=False
    )
    _save_animation(anim, output_path, fps, dpi)
    plt.close(fig)
    print(f"Saved iteration video: {output_path}")


def animate_init_final_rotation(
    x0: np.ndarray,
    positions: np.ndarray,
    mask: np.ndarray,
    time_values: np.ndarray,
    labels: np.ndarray | None = None,
    color_map: dict[str, str] | None = None,
    output_path: str | Path = "init_final_rotation.gif",
    n_frames: int = 120,
    elev: float = 25.0,
    azim_start: float = -60.0,
    azim_end: float = 300.0,
    fps: int = 24,
    dpi: int = 120,
    figsize: tuple[float, float] = (14, 7),
    point_size: float = 8.0,
    alpha_point: float = 0.6,
    alpha_line: float = 0.2,
    linewidth: float = 0.6,
    min_obs: int = 2,
    title: str = "",
):
    """Slow camera orbit comparing init and final stacked 3D structures side by side."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    _validate_inputs(x0, mask, time_values)
    _validate_inputs(positions, mask, time_values)
    color_map = _resolve_color_map(labels, color_map)

    fig = plt.figure(figsize=figsize)
    ax_init = fig.add_subplot(121, projection="3d")
    ax_final = fig.add_subplot(122, projection="3d")

    _draw_stacked_3d(
        ax_init, x0, mask, time_values, labels, color_map,
        point_size=point_size, alpha_point=alpha_point,
        alpha_line=alpha_line, linewidth=linewidth, min_obs=min_obs,
    )
    _draw_stacked_3d(
        ax_final, positions, mask, time_values, labels, color_map,
        point_size=point_size, alpha_point=alpha_point,
        alpha_line=alpha_line, linewidth=linewidth, min_obs=min_obs,
    )
    subtitle = "z = time (hpf) — visualization metadata, not a learned coordinate"
    ax_init.set_title("Initialized structure\n" + subtitle, fontsize=8)
    ax_final.set_title("Condensed structure\n" + subtitle, fontsize=8)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()

    azimuths = np.linspace(azim_start, azim_end, n_frames)

    def update(frame):
        azim = azimuths[frame]
        ax_init.view_init(elev=elev, azim=azim)
        ax_final.view_init(elev=elev, azim=azim)
        return []

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 / fps, blit=False
    )
    _save_animation(anim, output_path, fps, dpi)
    plt.close(fig)
    print(f"Saved init/final rotation video: {output_path}")


# ---------------------------------------------------------------------------
# Shared 3D drawing helper
# ---------------------------------------------------------------------------

def _draw_stacked_3d(
    ax,
    positions: np.ndarray,
    mask: np.ndarray,
    time_values: np.ndarray,
    labels: np.ndarray | None,
    color_map: dict[str, str],
    point_size: float,
    alpha_point: float,
    alpha_line: float,
    linewidth: float,
    min_obs: int,
) -> None:
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
            ax.plot(
                xs, ys, zs,
                color=color, alpha=alpha_line, linewidth=linewidth,
                label=group if first else "_nolegend_",
            )
            ax.scatter(xs, ys, zs, color=color, alpha=alpha_point, s=point_size)
            first = False

    ax.set_xlabel("dim 1", fontsize=7)
    ax.set_ylabel("dim 2", fontsize=7)
    ax.set_zlabel("time (hpf)", fontsize=7)
    if labels is not None and color_map:
        ax.legend(loc="upper left", fontsize=7, framealpha=0.6)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save_animation(anim, output_path: str | Path, fps: int, dpi: int) -> None:
    output_path = Path(output_path)
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        writer = "pillow"
    elif suffix == ".mp4":
        writer = "ffmpeg"
    else:
        raise ValueError(f"Unsupported output format {suffix!r}. Use .mp4 or .gif")
    anim.save(str(output_path), writer=writer, fps=fps, dpi=dpi)


# ---------------------------------------------------------------------------
# Input helpers (shared with plotting.py contract)
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


def _resolve_color_map(
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
