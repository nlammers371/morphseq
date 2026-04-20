"""
animation.py
------------
Video generation for ordered 2D coordinate slices across time.

Three distinct videos, each answering a different question:

  Video A — final structure, slow rotation
    animate_rotation(positions, mask, time_values, ...)
    "What is the geometry of the final structure?"
    Use for: presentation, branch inspection, geometry sanity check.

  Video B — optimization progress, fixed viewpoint
    animate_iterations(position_history, mask, time_values, ...)
    "How did the structure form? Is it converging?"
    Use for: debugging, hyperparameter tuning, convergence assessment.

  Video C — time-slice scrubber
    animate_time_slice(positions, mask, time_values, ...)
    "What does the UMAP look like at each time bin?"
    Left panel: stacked 3D overview with a highlighted z-plane sweeping through
    time. Right panel: 2D scatter of just the current time bin's points.
    Use for: presentations, inspecting how the cross-section evolves over time.

All accept the same minimal input contract as plotting.py:
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

from analyze.viz.styling.color_utils import apply_label_map


# ---------------------------------------------------------------------------
# Video A: final structure, slow rotation
# ---------------------------------------------------------------------------

def animate_rotation(
    positions: np.ndarray,
    mask: np.ndarray,
    time_values: np.ndarray,
    labels: np.ndarray | None = None,
    label_map: dict[str, str] | None = None,
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
    labels = apply_label_map(labels, label_map)
    color_map = _resolve_color_map(labels, color_map, label_map=label_map)

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
    label_map: dict[str, str] | None = None,
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
    labels = apply_label_map(labels, label_map)
    color_map = _resolve_color_map(labels, color_map, label_map=label_map)

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


def animate_time_slice(
    positions: np.ndarray,
    mask: np.ndarray,
    time_values: np.ndarray,
    labels: np.ndarray | None = None,
    label_map: dict[str, str] | None = None,
    color_map: dict[str, str] | None = None,
    output_path: str | Path = "time_slice.gif",
    elev: float = 25.0,
    azim: float = -60.0,
    fps: int = 8,
    n_interp: int = 3,
    hold_frames: int = 6,
    dpi: int = 120,
    figsize: tuple[float, float] = (14, 6),
    point_size: float = 18.0,
    alpha_point: float = 0.75,
    alpha_line: float = 0.15,
    alpha_slice_plane: float = 0.18,
    linewidth: float = 0.6,
    min_obs: int = 2,
    title: str = "",
):
    """Time-slice scrubber: stacked 3D overview (left) + 2D cross-section (right).

    The animation sweeps through time bins with smooth interpolation between
    consecutive bins. Between each pair of adjacent bins, ``n_interp`` sub-frames
    are rendered where:
      - The z-plane in the left panel glides continuously between the two hpf values.
      - On the right, embryos present in *both* bins have their 2D positions
        linearly interpolated. Embryos entering the next bin fade in; embryos
        leaving the current bin fade out.

    Parameters
    ----------
    positions : (N_e, T, 2)
    mask : (N_e, T) bool
    time_values : (T,) float — hpf values, used for z-axis and slice label
    labels : (N_e,) str, optional
    color_map : label → color string, auto-generated if None
    output_path : .gif or .mp4
    elev, azim : fixed 3D camera angles (degrees)
    fps : frames per second (default 8)
    n_interp : sub-frames rendered between each pair of real time bins.
    hold_frames : extra frames to hold on each keyframe before transitioning.
        Total frames = (T - 1) * (n_interp + hold_frames) + 1 + hold_frames.
    alpha_slice_plane : opacity of the highlighted z-plane in the left panel
    min_obs : minimum observations required to draw a trajectory line
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    _validate_inputs(positions, mask, time_values)
    labels = apply_label_map(labels, label_map)
    color_map = _resolve_color_map(labels, color_map, label_map=label_map)

    T = positions.shape[1]

    # Fixed axis limits computed once from all observed positions.
    obs_xy = positions[mask]
    xy_pad = 0.05
    x_all, y_all = obs_xy[:, 0], obs_xy[:, 1]
    x_range = x_all.max() - x_all.min() or 1.0
    y_range = y_all.max() - y_all.min() or 1.0
    xlim = (x_all.min() - xy_pad * x_range, x_all.max() + xy_pad * x_range)
    ylim = (y_all.min() - xy_pad * y_range, y_all.max() + xy_pad * y_range)
    zlim = (float(time_values[0]), float(time_values[-1]))

    # Build the flat list of (t_lo, t_hi, alpha) for every rendered frame.
    # alpha=0.0 → fully at bin t_lo; alpha=1.0 → fully at bin t_hi.
    # Each keyframe is held for hold_frames before the transition begins.
    frames_spec: list[tuple[int, int, float]] = []
    for _ in range(hold_frames + 1):
        frames_spec.append((0, 0, 0.0))
    for t in range(T - 1):
        for k in range(1, n_interp + 1):
            frames_spec.append((t, t + 1, k / n_interp))
        for _ in range(hold_frames):
            frames_spec.append((t + 1, t + 1, 0.0))

    n_frames_total = len(frames_spec)

    fig = plt.figure(figsize=figsize)
    ax3d = fig.add_subplot(121, projection="3d")
    ax2d = fig.add_subplot(122)

    # Static 3D background — all trajectories, dimmed.
    _draw_stacked_3d(
        ax3d, positions, mask, time_values, labels, color_map,
        point_size=point_size * 0.4, alpha_point=alpha_point * 0.35,
        alpha_line=alpha_line, linewidth=linewidth, min_obs=min_obs,
    )
    ax3d.set_xlim3d(*xlim)
    ax3d.set_ylim3d(*ylim)
    ax3d.set_zlim3d(*zlim)
    ax3d.view_init(elev=elev, azim=azim)
    ax3d.set_title("z = time (hpf)", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()

    slice_plane_artists: list = []
    highlight_artists: list = []

    def _make_slice_plane(z: float):
        x0, x1 = xlim
        y0, y1 = ylim
        verts = [[(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]]
        return Poly3DCollection(
            verts, alpha=alpha_slice_plane,
            facecolor="gold", edgecolor="goldenrod", linewidth=0.8,
        )

    def update(frame_idx: int):
        t_lo, t_hi, alpha = frames_spec[frame_idx]
        z = float(time_values[t_lo]) * (1 - alpha) + float(time_values[t_hi]) * alpha

        # --- Left: update slice plane ---
        for art in slice_plane_artists:
            art.remove()
        slice_plane_artists.clear()
        for art in highlight_artists:
            art.remove()
        highlight_artists.clear()

        poly = _make_slice_plane(z)
        ax3d.add_collection3d(poly)
        slice_plane_artists.append(poly)

        # Highlight points at the current z (only at keyframes for clarity).
        if alpha == 0.0 or alpha == 1.0:
            t_key = t_hi if alpha == 1.0 else t_lo
            obs = np.where(mask[:, t_key])[0]
            if len(obs):
                groups = np.unique(labels[obs]) if labels is not None else [None]
                for group in groups:
                    idx = obs[labels[obs] == group] if group is not None else obs
                    color = color_map.get(group, "steelblue") if group is not None else "steelblue"
                    sc = ax3d.scatter(
                        positions[idx, t_key, 0], positions[idx, t_key, 1],
                        np.full(len(idx), z),
                        color=color, s=point_size * 2.0, alpha=1.0,
                        zorder=5, edgecolors="white", linewidths=0.4,
                    )
                    highlight_artists.append(sc)

        # --- Right: interpolated 2D scatter ---
        ax2d.cla()
        ax2d.set_xlim(*xlim)
        ax2d.set_ylim(*ylim)
        ax2d.set_xlabel("dim 1", fontsize=8)
        ax2d.set_ylabel("dim 2", fontsize=8)
        ax2d.tick_params(labelsize=7)

        obs_lo = set(np.where(mask[:, t_lo])[0].tolist())
        obs_hi = set(np.where(mask[:, t_hi])[0].tolist())
        shared = np.array(sorted(obs_lo & obs_hi), dtype=int)
        only_lo = np.array(sorted(obs_lo - obs_hi), dtype=int)
        only_hi = np.array(sorted(obs_hi - obs_lo), dtype=int)

        def _scatter_group(idx: np.ndarray, t_src: int, point_alpha: float):
            if len(idx) == 0:
                return
            groups = np.unique(labels[idx]) if labels is not None else [None]
            for group in groups:
                g_idx = idx[labels[idx] == group] if group is not None else idx
                color = color_map.get(group, "steelblue") if group is not None else "steelblue"
                ax2d.scatter(
                    positions[g_idx, t_src, 0], positions[g_idx, t_src, 1],
                    color=color, s=point_size, alpha=point_alpha,
                    label=group, linewidths=0,
                )

        if len(shared):
            # Interpolate positions for shared embryos.
            xy = (1 - alpha) * positions[shared, t_lo, :] + alpha * positions[shared, t_hi, :]
            groups = np.unique(labels[shared]) if labels is not None else [None]
            for group in groups:
                g_mask = labels[shared] == group if group is not None else np.ones(len(shared), bool)
                color = color_map.get(group, "steelblue") if group is not None else "steelblue"
                ax2d.scatter(
                    xy[g_mask, 0], xy[g_mask, 1],
                    color=color, s=point_size, alpha=alpha_point,
                    label=group, linewidths=0,
                )

        # Fade out embryos leaving (only in t_lo): full alpha → 0 as alpha→1
        _scatter_group(only_lo, t_lo, alpha_point * (1 - alpha))
        # Fade in embryos entering (only in t_hi): 0 → full alpha as alpha→1
        _scatter_group(only_hi, t_hi, alpha_point * alpha)

        if labels is not None:
            handles, lab_names = ax2d.get_legend_handles_labels()
            seen: dict[str, object] = {}
            for h, l in zip(handles, lab_names):
                seen.setdefault(l, h)
            ax2d.legend(seen.values(), seen.keys(), loc="best", fontsize=7, framealpha=0.7)

        n_obs = len(obs_lo) if alpha < 0.5 else len(obs_hi)
        ax2d.set_title(f"{z:.1f} hpf  (n={n_obs})", fontsize=9)

        return []

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames_total, interval=1000 / fps, blit=False,
    )
    _save_animation(anim, output_path, fps, dpi)
    plt.close(fig)
    print(f"Saved time-slice video: {output_path}")


def animate_init_final_rotation(
    x0: np.ndarray,
    positions: np.ndarray,
    mask: np.ndarray,
    time_values: np.ndarray,
    labels: np.ndarray | None = None,
    label_map: dict[str, str] | None = None,
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
    labels = apply_label_map(labels, label_map)
    color_map = _resolve_color_map(labels, color_map, label_map=label_map)

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
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    label_map: dict[str, str] | None = None,
) -> dict[str, str]:
    if labels is None:
        return {}
    if color_map is not None:
        if label_map:
            return {label_map.get(str(k), str(k)): v for k, v in color_map.items()}
        return color_map
    import matplotlib.pyplot as plt
    unique = sorted(np.unique(labels))
    cmap = plt.get_cmap("tab10")
    return {l: cmap(i % 10) for i, l in enumerate(unique)}

