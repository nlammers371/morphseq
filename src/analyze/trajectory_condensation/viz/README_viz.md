# `trajectory_condensation/viz` — Visualization README

Static comparison figures, stacked 3D plots, GIF rendering, and the condensed time-slice HTML viewer for condensation runs.

For force/config API details, see the root module:
- [state.py](../condensation/state.py)
- [run.py](../condensation/engine/run.py)

## Quick Start

```python
import analyze.trajectory_condensation as tc

run = tc.load_run("results/.../condensed_positions.npz", title="shrunk 4class")
tc.render_run(run, "output/my_run/")
```

This writes the standard visualization bundle for one run.

## Data Contract

All visualization functions expect:

| Array | Shape | Notes |
|-------|-------|-------|
| `positions` | `(N, T, 2)` | NaN where `mask=False` |
| `mask` | `(N, T)` bool | Observation mask |
| `time_values` | `(T,)` float | Sorted ascending |
| `labels` | `(N,)` str | Optional |
| `color_map` | `dict[str, str]` | Optional |

`load_run()` reads the standard NPZ output written by the condensation runner.

## Main API (`api.py`)

### `load_run(path, title="", color_map=None)`

```python
run = tc.load_run("condensed_positions.npz", title="my experiment")
```

Required NPZ keys: `positions`, `mask`, `time_values`

Optional NPZ keys: `labels`, `embryo_ids`, `x0`, `position_history`, `snapshot_iters`

### `VizConfig`

Shared style configuration for figures.

```python
cfg = tc.VizConfig(
    dpi=150,
    figsize=(8, 7),
    alpha_line=0.25,
    alpha_point=0.6,
    point_size=8.0,
    linewidth=0.8,
    panel_size=3.0,
)
```

For the common presentation bundle, `VizConfig` is the lightest way to control
the repeated size and alpha defaults without touching lower-level helpers.

### `render_run(run, output_dir, *, config=None, title_prefix="", snapshot_idx=None, skip_animations=False)`

Writes the standard figure/GIF bundle for a single run.

Standard outputs:
- `plot_trajectories.png`
- `plot_trajectories_init.png` if `x0` is present
- `plot_panels.png`
- `plot_stacked_3d.png`
- `rotation.gif` unless `skip_animations=True`
- `init_vs_final_rotation.gif` if `x0` is present
- `iterations.gif` if saved iteration history is available

The interactive browser is not a generic Plotly utility. Use `tc.time_slice_html(...)`,
implemented in [condensed_time_slice_viewer.py](condensed_time_slice_viewer.py),
when you want the condensed trajectory viewer with the 3D trajectory context,
highlighted current-time points, and the optional 2D current-slice panel.

### Example API: time slice viewer

```python
tc.time_slice_html(
    run.positions,
    run.mask,
    run.time_values,
    labels=run.labels,
    label_map={
        "ab": "wildtype",
        "wik_ab": "wildtype",
        "inj_ctrl": "inj. ctrl",
        "pbx1b_crispant": "pbx1b",
        "pbx4_crispant": "pbx4",
    },
    color_map=run.color_map,
    embryo_ids=run.embryo_ids,
    title="<b>3D</b> Trajectories",
    subplot_titles=("<b>3D</b> Trajectories", "Current time slice"),
    output_path="time_slice.html",
)
```

### `compare_runs(runs, mode="trajectories", *, config=None, align_axes=True, figsize=None, output_path=None)`

Side-by-side comparison for two or more runs.

Supported modes: `"trajectories"`, `"stacked_3d"`

```python
r1 = tc.load_run("results/.../run_A/condensed_positions.npz", title="run A")
r2 = tc.load_run("results/.../run_B/condensed_positions.npz", title="run B")

tc.compare_runs([r1, r2], mode="trajectories", output_path="compare.png")
tc.compare_runs([r1, r2], mode="stacked_3d", output_path="compare_3d.png")
```

### `compare_run_grid(...)`

Grid comparison helper for sweep outputs (rows = method, columns = strength/condition).

See [api.py](api.py) and [03_force_diagnostics.py](../../../../results/mcolon/20260407_pbx_analysis_cont/03_force_diagnostics.py).

## Condensed Time-Slice Viewer

### `condensed_time_slice_viewer.py`

Owner file for `tc.time_slice_html(...)`. It writes a self-contained HTML viewer
for condensed coordinates. It shows all trajectories as dimmed 3D context,
highlights the selected time bin, and optionally shows the current time slice in
a linked 2D panel. Keep this behavior here rather than in generic Plotly helpers
or solver modules.

```python
tc.time_slice_html(
    run.positions,
    run.mask,
    run.time_values,
    labels=run.labels,
    color_map=run.color_map,
    embryo_ids=run.embryo_ids,
    output_path="time_slice.html",
)
```

## Lower-Level API

### `plotting.py`

Lower-level figure builders:
- `plot_trajectories(...)`
- `plot_panels(...)`
- `plot_stacked_3d(...)`
- `plot_color_by_time(...)`

### `animation.py`

Lower-level GIF/animation helpers:
- `animate_rotation(...)`
- `animate_iterations(...)`
- `animate_init_final_rotation(...)`

The common presentation path now also accepts `label_map` so callers can
normalize labels at the API boundary instead of rewriting labels before and
after rendering.

```python
from analyze.trajectory_condensation.viz import animation

animation.animate_rotation(
    positions,
    mask,
    time_values,
    label_map={
        "ab": "wildtype",
        "wik_ab": "wildtype",
        "inj_ctrl": "inj. ctrl",
    },
    output_path="rotation.gif",
    n_frames=120,
    fps=24,
    elev=25.0,
    azim_start=-60.0,
    azim_end=300.0,
)
```

## Typical Pipeline

```python
run = tc.load_run("results/.../condensed_positions.npz", title="my run")
tc.render_run(run, "output/run_bundle/")

runs = [
    tc.load_run("results/.../run_A/condensed_positions.npz", title="A"),
    tc.load_run("results/.../run_B/condensed_positions.npz", title="B"),
]
tc.compare_runs(runs, mode="stacked_3d", output_path="compare_3d.png")
```

## PBX Sweep Examples

- [02_pairwise_trajectory_condensation.py](../../../../results/mcolon/20260407_pbx_analysis_cont/02_pairwise_trajectory_condensation.py)
- [03_force_diagnostics.py](../../../../results/mcolon/20260407_pbx_analysis_cont/03_force_diagnostics.py)
- [06_render_sweep_init_rotation_gif.py](../../../../results/mcolon/20260407_pbx_analysis_cont/06_render_sweep_init_rotation_gif.py)
