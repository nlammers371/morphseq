"""
viz.py
------
Public convenience API for trajectory condensation visualization.

Wraps plotting.py and animation.py without changing their signatures.

Public API:
  VizConfig       — shared style defaults (7 knobs)
  RunDescriptor   — typed container for one condensation run's data
  load_run(path)  — load condensed_positions.npz → RunDescriptor
  render_run(...) — write full output bundle (plots + GIFs) to a directory
  compare_runs(runs, mode) — side-by-side figure for N≥2 runs
  compare_run_grid(run_grid, mode) — row/column comparison grid for grouped sweeps

Embryo alignment note
---------------------
compare_runs compares distributions / overall shape. It does NOT align
embryos by embryo_id across runs — index order within each run is used
as-is. embryo_ids is stored in RunDescriptor so future callers can
implement alignment without an NPZ format change.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Animation defaults (internal constants — add to VizConfig when needed)
# ---------------------------------------------------------------------------
_FPS_ROTATION = 24
_FPS_ITERATIONS = 4
_N_FRAMES = 120
_ELEV = 25.0
_AZIM_START = -60.0
_AZIM_END = 300.0


# ---------------------------------------------------------------------------
# VizConfig
# ---------------------------------------------------------------------------

@dataclass
class VizConfig:
    """Shared style configuration for trajectory visualizations.

    Animation-specific parameters (fps, n_frames, elev, azim) are kept as
    internal constants and will be added here when callers need to override them.
    """
    dpi: int = 150
    figsize: tuple[float, float] = (8, 7)
    alpha_line: float = 0.25
    alpha_point: float = 0.6
    point_size: float = 8.0
    linewidth: float = 0.8
    panel_size: float = 3.0


# ---------------------------------------------------------------------------
# RunDescriptor
# ---------------------------------------------------------------------------

@dataclass
class RunDescriptor:
    """Typed container for one condensation run's data.

    Shapes
    ------
    positions : (N_e, T, 2)  — NaN where mask=False (normalized in __post_init__)
    mask      : (N_e, T) bool — True = observed
    time_values : (T,) float  — sorted ascending
    labels    : (N_e,) str   — genotype or other categorical label
    embryo_ids : (N_e,) str  — stored for future cross-run alignment; not used now
    x0        : (N_e, T, 2)  — initial positions (AlignedUMAP); used by render_run
    position_history : (n_iters, N_e, T, 2) — saved optimization snapshots
    snapshot_iters   : list[int]            — iteration numbers for position_history frames
    """
    positions: np.ndarray
    mask: np.ndarray
    time_values: np.ndarray
    labels: np.ndarray | None = None
    embryo_ids: np.ndarray | None = None
    color_map: dict[str, str] | None = None
    title: str = ""
    x0: np.ndarray | None = None
    position_history: np.ndarray | None = None
    snapshot_iters: list[int] | None = None

    def __post_init__(self) -> None:
        N_e, T = self.positions.shape[:2]
        if self.mask.shape != (N_e, T):
            raise ValueError(
                f"mask shape {self.mask.shape} does not match positions "
                f"(N_e={N_e}, T={T})"
            )
        if len(self.time_values) != T:
            raise ValueError(
                f"time_values length {len(self.time_values)} != T={T}"
            )
        if self.labels is not None and len(self.labels) != N_e:
            raise ValueError(
                f"labels length {len(self.labels)} != N_e={N_e}"
            )
        # normalize mask dtype
        if not np.issubdtype(self.mask.dtype, np.bool_):
            self.mask = self.mask.astype(bool)
        # normalize: positions[~mask] = NaN for consistent downstream rendering
        self.positions = self.positions.copy().astype(float)
        self.positions[~self.mask] = np.nan


# ---------------------------------------------------------------------------
# load_run
# ---------------------------------------------------------------------------

def load_run(
    path: str | Path,
    title: str = "",
    color_map: dict[str, str] | None = None,
) -> RunDescriptor:
    """Load a condensed_positions.npz into a RunDescriptor.

    Required NPZ keys
    -----------------
    positions, mask, time_values

    Optional NPZ keys
    -----------------
    labels, embryo_ids, x0, position_history, snapshot_iters

    All other keys are silently ignored.

    Parameters
    ----------
    path : path to condensed_positions.npz
    title : display title for this run (shown in compare_runs subplots)
    color_map : optional label→color dict; auto-generated from labels if None
    """
    path = Path(path)
    d = np.load(path, allow_pickle=True)

    for key in ("positions", "mask", "time_values"):
        if key not in d:
            raise KeyError(
                f"Required key {key!r} missing from {path}. "
                f"Available keys: {list(d.files)}"
            )

    labels = d["labels"] if "labels" in d.files else None
    embryo_ids = d["embryo_ids"] if "embryo_ids" in d.files else None
    x0 = d["x0"] if "x0" in d.files else None
    position_history = d["position_history"] if "position_history" in d.files else None
    snapshot_iters = (
        [int(x) for x in d["snapshot_iters"]]
        if "snapshot_iters" in d.files
        else None
    )

    return RunDescriptor(
        positions=d["positions"],
        mask=d["mask"],
        time_values=d["time_values"],
        labels=labels,
        embryo_ids=embryo_ids,
        color_map=color_map,
        title=title,
        x0=x0,
        position_history=position_history,
        snapshot_iters=snapshot_iters,
    )


# ---------------------------------------------------------------------------
# render_run
# ---------------------------------------------------------------------------

def render_run(
    run: RunDescriptor,
    output_dir: str | Path,
    *,
    config: VizConfig | None = None,
    title_prefix: str = "",
    snapshot_idx: list[int] | None = None,
    skip_animations: bool = False,
) -> dict[str, Path]:
    """Write the standard output bundle for one condensation run.

    All figures are closed after saving. Returns {filename: Path} for
    every file written — callers can verify outputs without filesystem stat.

    Static plots (always written)
    ------------------------------
    plot_trajectories.png
    plot_trajectories_init.png  (only if run.x0 is not None)
    plot_panels.png
    plot_stacked_3d.png

    Animations (skipped when skip_animations=True)
    -----------------------------------------------
    rotation.gif
    init_vs_final_rotation.gif  (only if run.x0 is not None)
    iterations.gif or iterations.png — smart frame-count handling:
        position_history missing → omitted entirely
        1 frame → PNG + UserWarning
        ≥2 frames → GIF with rotation=True (slow camera orbit)

    Parameters
    ----------
    snapshot_idx : indices into time_values for the panels plot.
        Defaults to np.linspace(0, T-1, min(6, T), dtype=int).
    title_prefix : prepended to all plot titles with " | " separator.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import animation as tc_animation
    from .plotting import (
        plot_panels,
        plot_stacked_3d,
        plot_trajectories,
    )

    cfg = config or VizConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _title(base: str) -> str:
        return f"{title_prefix} | {base}" if title_prefix else base

    T = len(run.time_values)
    if snapshot_idx is None:
        snapshot_idx = list(np.linspace(0, T - 1, min(6, T), dtype=int))
    snapshot_times = [float(run.time_values[i]) for i in snapshot_idx]

    written: dict[str, Path] = {}

    # ---- static: final trajectories ----
    fig, _ = plot_trajectories(
        run.positions, run.mask, run.time_values,
        labels=run.labels, color_map=run.color_map,
        alpha_line=cfg.alpha_line, alpha_point=cfg.alpha_point,
        linewidth=cfg.linewidth, point_size=cfg.point_size,
        figsize=cfg.figsize,
        title=_title("trajectories"),
    )
    p = output_dir / "plot_trajectories.png"
    fig.savefig(p, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)
    written["plot_trajectories.png"] = p

    # ---- static: init trajectories ----
    if run.x0 is not None:
        fig, _ = plot_trajectories(
            run.x0, run.mask, run.time_values,
            labels=run.labels, color_map=run.color_map,
            alpha_line=cfg.alpha_line, alpha_point=cfg.alpha_point,
            linewidth=cfg.linewidth, point_size=cfg.point_size,
            figsize=cfg.figsize,
            title=_title("trajectories (init)"),
        )
        p = output_dir / "plot_trajectories_init.png"
        fig.savefig(p, dpi=cfg.dpi, bbox_inches="tight")
        plt.close(fig)
        written["plot_trajectories_init.png"] = p

    # ---- static: panels ----
    fig, _ = plot_panels(
        run.positions, run.mask, run.time_values,
        labels=run.labels, color_map=run.color_map,
        snapshot_times=snapshot_times,
        panel_size=cfg.panel_size,
        title=_title("panels"),
    )
    p = output_dir / "plot_panels.png"
    fig.savefig(p, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)
    written["plot_panels.png"] = p

    # ---- static: stacked 3D ----
    fig, _ = plot_stacked_3d(
        run.positions, run.mask, run.time_values,
        labels=run.labels, color_map=run.color_map,
        alpha_line=cfg.alpha_line, alpha_point=cfg.alpha_point,
        linewidth=cfg.linewidth, point_size=cfg.point_size,
        figsize=cfg.figsize,
        elev=_ELEV, azim=_AZIM_START,
        title=_title("stacked 3D"),
    )
    p = output_dir / "plot_stacked_3d.png"
    fig.savefig(p, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)
    written["plot_stacked_3d.png"] = p

    if skip_animations:
        return written

    # ---- animation: rotation of final structure ----
    p = output_dir / "rotation.gif"
    tc_animation.animate_rotation(
        run.positions, run.mask, run.time_values,
        labels=run.labels, color_map=run.color_map,
        output_path=p,
        n_frames=_N_FRAMES,
        elev=_ELEV, azim_start=_AZIM_START, azim_end=_AZIM_END,
        fps=_FPS_ROTATION, dpi=cfg.dpi, figsize=cfg.figsize,
        point_size=cfg.point_size,
        alpha_point=cfg.alpha_point, alpha_line=cfg.alpha_line,
        linewidth=cfg.linewidth,
        title=_title("rotation"),
    )
    written["rotation.gif"] = p

    # ---- animation: init vs final rotation ----
    if run.x0 is not None:
        p = output_dir / "init_vs_final_rotation.gif"
        tc_animation.animate_init_final_rotation(
            run.x0, run.positions, run.mask, run.time_values,
            labels=run.labels, color_map=run.color_map,
            output_path=p,
            n_frames=_N_FRAMES,
            elev=_ELEV, azim_start=_AZIM_START, azim_end=_AZIM_END,
            fps=_FPS_ROTATION, dpi=cfg.dpi,
            point_size=cfg.point_size,
            alpha_point=cfg.alpha_point, alpha_line=cfg.alpha_line,
            linewidth=cfg.linewidth,
            title=_title("init vs final"),
        )
        written["init_vs_final_rotation.gif"] = p

    # ---- animation: iteration progress (smart frame count) ----
    if run.position_history is not None:
        n_frames_hist = run.position_history.shape[0]
        iter_labels = run.snapshot_iters if run.snapshot_iters is not None else list(range(n_frames_hist))

        if n_frames_hist == 1:
            warnings.warn(
                "position_history has 1 frame; saving PNG instead of GIF. "
                "Pass save_every > 1 iteration to get an animated GIF.",
                UserWarning,
                stacklevel=2,
            )
            fig = plt.figure(figsize=cfg.figsize)
            ax = fig.add_subplot(111, projection="3d")
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            from .animation import _draw_stacked_3d, _resolve_color_map
            color_map_resolved = _resolve_color_map(run.labels, run.color_map)
            _draw_stacked_3d(
                ax, run.position_history[0], run.mask, run.time_values,
                run.labels, color_map_resolved,
                point_size=cfg.point_size, alpha_point=cfg.alpha_point,
                alpha_line=cfg.alpha_line, linewidth=cfg.linewidth, min_obs=2,
            )
            subtitle = "z = time (hpf) — visualization metadata, not a learned coordinate"
            iter_str = f"iter {iter_labels[0]}"
            ax.set_title(
                (_title(iter_str) if title_prefix else iter_str) + "\n" + subtitle,
                fontsize=8,
            )
            ax.view_init(elev=_ELEV, azim=_AZIM_START)
            p = output_dir / "iterations.png"
            fig.savefig(p, dpi=cfg.dpi, bbox_inches="tight")
            plt.close(fig)
            written["iterations.png"] = p
        else:
            p = output_dir / "iterations.gif"
            tc_animation.animate_iterations(
                run.position_history, run.mask, run.time_values,
                iter_labels=iter_labels,
                labels=run.labels, color_map=run.color_map,
                output_path=p,
                elev=_ELEV, azim=_AZIM_START, azim_end=_AZIM_END,
                rotation=True,
                fps=_FPS_ITERATIONS, dpi=cfg.dpi, figsize=cfg.figsize,
                point_size=cfg.point_size,
                alpha_point=cfg.alpha_point, alpha_line=cfg.alpha_line,
                linewidth=cfg.linewidth,
                title=_title("iterations"),
            )
            written["iterations.gif"] = p

    return written


# ---------------------------------------------------------------------------
# compare_runs
# ---------------------------------------------------------------------------

def compare_runs(
    runs: list[RunDescriptor],
    mode: Literal["trajectories", "stacked_3d"] = "trajectories",
    *,
    config: VizConfig | None = None,
    align_axes: bool = True,
    figsize: tuple[float, float] | None = None,
    output_path: str | Path | None = None,
) -> tuple | Path:
    """Side-by-side comparison figure for N≥2 condensation runs.

    Parameters
    ----------
    runs : list of RunDescriptor, must have length ≥ 2
    mode : "trajectories" or "stacked_3d"
    config : shared style config; uses VizConfig defaults if None
    align_axes : if True, unify axis limits across all runs after drawing,
        so spatial scales are directly comparable
    figsize : override auto-computed size
        (auto = config.figsize[0] * N_runs, config.figsize[1])
    output_path : if given, saves PNG/PDF and closes the figure;
        returns Path(output_path) instead of (fig, axes)

    Returns
    -------
    (fig, axes) when output_path is None — caller owns the open figure
    Path        when output_path is given — figure saved and closed

    axes shape
    ----------
    trajectories → list[Axes]
    stacked_3d   → list[Axes3D]

    Notes
    -----
    Embryo alignment: comparisons are not aligned by embryo_id — index
    order within each run is used as-is. A warning is emitted when runs
    have differing embryo_ids.

    GIF output is not supported in v1; use output_path with a .png/.pdf
    extension. Animated comparison via compare_runs_rotation_gif() is
    planned for v2.
    """
    if len(runs) < 1:
        raise ValueError("compare_runs requires at least 1 run")
    if mode not in ("trajectories", "stacked_3d"):
        raise ValueError(f"Unknown mode {mode!r}. v1 supports: 'trajectories', 'stacked_3d'")

    cfg = config or VizConfig()
    N_runs = len(runs)

    if len(runs) > 1:
        _warn_embryo_mismatch(runs)

    auto_figsize = (cfg.figsize[0] * N_runs, cfg.figsize[1])
    fig_size = figsize or auto_figsize

    if mode == "trajectories":
        fig, axes = _compare_trajectories(runs, cfg, fig_size)
    else:
        fig, axes = _compare_stacked_3d(runs, cfg, fig_size)

    if align_axes:
        _align_axes(axes, mode)

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        fig.savefig(output_path, dpi=cfg.dpi, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        return output_path

    return fig, axes


def compare_run_grid(
    run_grid: list[list[RunDescriptor]],
    mode: Literal["trajectories", "stacked_3d"] = "trajectories",
    *,
    config: VizConfig | None = None,
    align_axes: bool = True,
    figsize: tuple[float, float] | None = None,
    output_path: str | Path | None = None,
) -> tuple | Path:
    """Render a row/column grid of runs for method-by-strength style sweeps."""
    if not run_grid or not run_grid[0]:
        raise ValueError("compare_run_grid requires a non-empty rectangular grid.")
    n_rows = len(run_grid)
    n_cols = len(run_grid[0])
    for row in run_grid:
        if len(row) != n_cols:
            raise ValueError("compare_run_grid requires all rows to have the same length.")
    cfg = config or VizConfig()
    fig_size = figsize or (cfg.figsize[0] * n_cols, cfg.figsize[1] * n_rows)

    _warn_embryo_mismatch([run for row in run_grid for run in row])

    if mode == "trajectories":
        fig, axes = _compare_grid_trajectories(run_grid, cfg, fig_size)
    elif mode == "stacked_3d":
        fig, axes = _compare_grid_stacked_3d(run_grid, cfg, fig_size)
    else:
        raise ValueError(f"Unknown mode {mode!r}. v1 supports: 'trajectories', 'stacked_3d'")

    if align_axes:
        _align_axes([ax for row in axes for ax in row], mode)

    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        fig.savefig(output_path, dpi=cfg.dpi, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        return output_path
    return fig, axes


# ---------------------------------------------------------------------------
# Private: embryo id warning
# ---------------------------------------------------------------------------

def _warn_embryo_mismatch(runs: list[RunDescriptor]) -> None:
    ids = [r.embryo_ids for r in runs]
    if all(x is not None for x in ids):
        sets = [set(x.tolist()) for x in ids]
        if len(set.union(*sets)) != len(set.intersection(*sets)):
            warnings.warn(
                "Runs have different embryo_ids; compare_runs compares "
                "distributions/shape, not aligned embryos. "
                "Index order within each run is used as-is.",
                UserWarning,
                stacklevel=3,
            )


# ---------------------------------------------------------------------------
# Private: mode-specific figure builders
# ---------------------------------------------------------------------------

def _compare_trajectories(runs, cfg: VizConfig, fig_size):
    import matplotlib.pyplot as plt

    fig, axes_arr = plt.subplots(1, len(runs), figsize=fig_size, squeeze=False)
    axes = [axes_arr[0][r] for r in range(len(runs))]
    for ax, run in zip(axes, runs):
        _draw_trajectories(ax, run, cfg)
        ax.set_title(run.title or f"Run {runs.index(run) + 1}", fontsize=9)
    return fig, axes


def _compare_stacked_3d(runs, cfg: VizConfig, fig_size):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=fig_size)
    axes = []
    for r, run in enumerate(runs):
        ax = fig.add_subplot(1, len(runs), r + 1, projection="3d")
        ax.view_init(elev=_ELEV, azim=_AZIM_START)
        _draw_stacked_3d_viz(ax, run, cfg)
        ax.set_title(run.title or f"Run {r + 1}", fontsize=9)
        axes.append(ax)
    return fig, axes


def _compare_grid_trajectories(run_grid, cfg: VizConfig, fig_size):
    import matplotlib.pyplot as plt

    n_rows = len(run_grid)
    n_cols = len(run_grid[0])
    fig, axes_arr = plt.subplots(n_rows, n_cols, figsize=fig_size, squeeze=False)
    axes = [[axes_arr[r][c] for c in range(n_cols)] for r in range(n_rows)]
    for r in range(n_rows):
        for c in range(n_cols):
            run = run_grid[r][c]
            ax = axes[r][c]
            _draw_trajectories(ax, run, cfg)
            ax.set_title(run.title or f"Run {r + 1},{c + 1}", fontsize=9)
    return fig, axes


def _compare_grid_stacked_3d(run_grid, cfg: VizConfig, fig_size):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    n_rows = len(run_grid)
    n_cols = len(run_grid[0])
    fig = plt.figure(figsize=fig_size)
    axes: list[list] = []
    for r in range(n_rows):
        row_axes = []
        for c in range(n_cols):
            ax = fig.add_subplot(n_rows, n_cols, r * n_cols + c + 1, projection="3d")
            ax.view_init(elev=_ELEV, azim=_AZIM_START)
            run = run_grid[r][c]
            _draw_stacked_3d_viz(ax, run, cfg)
            ax.set_title(run.title or f"Run {r + 1},{c + 1}", fontsize=9)
            row_axes.append(ax)
        axes.append(row_axes)
    return fig, axes


# ---------------------------------------------------------------------------
# Private: axis alignment
# ---------------------------------------------------------------------------

def _align_axes(axes, mode: str) -> None:
    if mode == "trajectories":
        xlims = [ax.get_xlim() for ax in axes]
        ylims = [ax.get_ylim() for ax in axes]
        xmin = min(lo for lo, _ in xlims)
        xmax = max(hi for _, hi in xlims)
        ymin = min(lo for lo, _ in ylims)
        ymax = max(hi for _, hi in ylims)
        for ax in axes:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
    elif mode == "stacked_3d":
        xlims = [ax.get_xlim3d() for ax in axes]
        ylims = [ax.get_ylim3d() for ax in axes]
        zlims = [ax.get_zlim3d() for ax in axes]
        xmin = min(lo for lo, _ in xlims)
        xmax = max(hi for _, hi in xlims)
        ymin = min(lo for lo, _ in ylims)
        ymax = max(hi for _, hi in ylims)
        zmin = min(lo for lo, _ in zlims)
        zmax = max(hi for _, hi in zlims)
        for ax in axes:
            ax.set_xlim3d(xmin, xmax)
            ax.set_ylim3d(ymin, ymax)
            ax.set_zlim3d(zmin, zmax)


# ---------------------------------------------------------------------------
# Private: drawing helpers
# Extracted from plotting.plot_trajectories / plot_stacked_3d.
# MAINTENANCE: if plotting.py rendering logic changes, update these to match.
# ---------------------------------------------------------------------------

def _draw_trajectories(ax, run: RunDescriptor, cfg: VizConfig) -> None:
    """Render trajectory lines + scatter into an existing 2D axis."""
    from .plotting import resolve_color_map

    color_map = resolve_color_map(run.labels, run.color_map)
    N_e = run.positions.shape[0]
    groups = np.unique(run.labels) if run.labels is not None else [None]

    for group in groups:
        if group is not None:
            embryo_idx = np.where(run.labels == group)[0]
            color = color_map[group]
        else:
            embryo_idx = np.arange(N_e)
            color = "steelblue"

        first = True
        for i in embryo_idx:
            obs_t = np.where(run.mask[i, :])[0]
            if len(obs_t) < 2:
                continue
            xs = run.positions[i, obs_t, 0]
            ys = run.positions[i, obs_t, 1]
            ax.plot(
                xs, ys,
                color=color, alpha=cfg.alpha_line, linewidth=cfg.linewidth,
                label=group if first else "_nolegend_",
            )
            ax.scatter(xs, ys, color=color, alpha=cfg.alpha_point,
                       s=cfg.point_size, linewidths=0)
            first = False

    if run.labels is not None:
        ax.legend(loc="best", fontsize=7, framealpha=0.7)
    ax.set_xlabel("dim 1", fontsize=8)
    ax.set_ylabel("dim 2", fontsize=8)


def _draw_stacked_3d_viz(ax, run: RunDescriptor, cfg: VizConfig) -> None:
    """Render stacked 3D (x, y, time) into an existing Axes3D."""
    from .plotting import resolve_color_map

    color_map = resolve_color_map(run.labels, run.color_map)
    N_e = run.positions.shape[0]
    groups = np.unique(run.labels) if run.labels is not None else [None]

    for group in groups:
        if group is not None:
            embryo_idx = np.where(run.labels == group)[0]
            color = color_map[group]
        else:
            embryo_idx = np.arange(N_e)
            color = "steelblue"

        first = True
        for i in embryo_idx:
            obs_t = np.where(run.mask[i, :])[0]
            if len(obs_t) < 2:
                continue
            xs = run.positions[i, obs_t, 0]
            ys = run.positions[i, obs_t, 1]
            zs = run.time_values[obs_t]
            ax.plot(
                xs, ys, zs,
                color=color, alpha=cfg.alpha_line, linewidth=cfg.linewidth,
                label=group if first else "_nolegend_",
            )
            ax.scatter(xs, ys, zs, color=color, alpha=cfg.alpha_point,
                       s=cfg.point_size)
            first = False

    ax.set_xlabel("dim 1", fontsize=7)
    ax.set_ylabel("dim 2", fontsize=7)
    ax.set_zlabel("time (hpf)", fontsize=7)
    subtitle = "z = time (hpf) — stacked for visualization, not a learned 3D manifold"
    ax.set_title(subtitle, fontsize=7)
    if run.labels is not None and color_map:
        ax.legend(loc="upper left", fontsize=7, framealpha=0.6)
