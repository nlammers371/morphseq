# analyze-trajectory-condensation - Viz Reference

Use this for visualization under `src/analyze/trajectory_condensation/viz/`.

The full user-facing documentation is
`src/analyze/trajectory_condensation/viz/README_viz.md`. The solver/method
context lives in `src/analyze/trajectory_condensation/ALGORITHM.md` and
`src/analyze/trajectory_condensation/DESIGN.md`.

## Current State

- Visualization consumes solved outputs; it should not define solver policy.
- Normal callers should import through the root package:

```python
import analyze.trajectory_condensation as tc
```

- `tc.render_run(...)` writes the standard one-run bundle.
- `tc.compare_runs(...)` and `tc.compare_run_grid(...)` are for run comparisons.
- `tc.time_slice_html(...)` is the public entry point for the condensed time-slice HTML viewer in `src/analyze/trajectory_condensation/viz/condensed_time_slice_viewer.py`.
- Iteration-choice plots are available from
  `analyze.trajectory_condensation.viz.iteration_choice_plots`.

## Data Contract

Visualization expects:

| Array | Shape | Notes |
|---|---|---|
| `positions` | `(N_e, T, 2)` | NaN where `mask=False` |
| `mask` | `(N_e, T)` bool | True where observed |
| `time_values` | `(T,)` float | sorted hpf values |
| `labels` | `(N_e,)` str | optional category labels |
| `embryo_ids` | `(N_e,)` str | optional, stored but not used for cross-run alignment |
| `x0` | `(N_e, T, 2)` | optional initialization for init-vs-final views |
| `position_history` | `(n_snaps, N_e, T, 2)` | optional saved solver snapshots |
| `snapshot_iters` | `list[int]` | optional iteration numbers for snapshots |

`tc.load_run(...)` reads the standard `condensed_positions.npz` format and
returns a `tc.RunDescriptor`.

## Standard One-Run Bundle

```python
import analyze.trajectory_condensation as tc

run = tc.load_run("results/.../condensed_positions.npz", title="shrunk 4class")
paths = tc.render_run(run, "figures/shrunk_4class/")
```

Standard outputs currently include:

- `plot_trajectories.png`
- `plot_trajectories_init.png` when `x0` is present
- `plot_panels.png`
- `plot_stacked_3d.png`
- `rotation.gif` unless animations are skipped
- `init_vs_final_rotation.gif` when `x0` is present
- `iterations.gif` when `position_history` and `snapshot_iters` are present

Use `skip_animations=True` when debugging static output or running on a slow
filesystem.

## Run Comparison

```python
runs = [
    tc.load_run("results/.../run_A/condensed_positions.npz", title="run A"),
    tc.load_run("results/.../run_B/condensed_positions.npz", title="run B"),
]

tc.compare_runs(runs, mode="trajectories", output_path="compare_trajectories.png")
tc.compare_runs(runs, mode="stacked_3d", output_path="compare_3d.png")
```

Important state: `compare_runs(...)` compares run distributions and shape. It
does not align embryos by `embryo_id` across runs; it uses each run's index order.

## Sweep/Grid Comparison

Use `tc.compare_run_grid(...)` for sweeps where rows and columns have meaning,
for example method by strength or condition by initialization. See
`src/analyze/trajectory_condensation/viz/README_viz.md` and
`results/mcolon/20260407_pbx_analysis_cont/03_force_diagnostics.py` for the
current example pattern.

## Condensed Time-Slice Viewer

```python
tc.time_slice_html(
    run.positions,
    run.mask,
    run.time_values,
    labels=run.labels,
    color_map=run.color_map,
    embryo_ids=run.embryo_ids,
    trajectory_trace_alpha=0.16,
    current_time_marker_alpha=1.0,
    slice_marker_alpha=0.75,
    trajectory_marker_alpha=0.15,
    current_time_marker_size=6,
    slice_marker_size=6,
    trajectory_marker_size=2,
    output_path="time_slice.html",
)
```

This is not a generic Plotly helper. It renders the condensed trajectory viewer:
a dimmed 3D trajectory context, highlighted current-time points, and an optional
2D current-slice panel driven by the time slider. Keep this behavior in
`src/analyze/trajectory_condensation/viz/condensed_time_slice_viewer.py`
instead of adding Plotly logic to solver files.

Important current behavior:

- `color_map` is the public phenotype color hook. Pass a label-to-color dict to
  override the default palette.
- `trajectory_trace_alpha` controls the connecting 3D trajectory lines.
- `current_time_marker_*` controls the highlighted current-time points.
- `slice_marker_*` controls the right-panel 2D slice points.
- `trajectory_marker_*` controls the small dimmed background trajectory markers.
- The clickable phenotype legend uses fully opaque proxy traces so it remains
  readable even when the background trajectories are dimmed.
- The legend should toggle phenotype groups on the first click; do not use
  `visible="legendonly"` for the proxy traces.
- Legacy args like `alpha_bg`, `alpha_bg_marker`, `alpha_highlight`, `alpha_2d`,
  `marker_size_bg`, `marker_size_highlight`, and `marker_size_2d` are backward-
  compatible aliases, but new callers should prefer the user-facing names above.

For quick visual inspection during iteration, it is fine to add a tiny local
results-side preview script that loads a real `condensed_positions.npz` and
writes a single `time_slice.html` next to the run figures.

## Iteration Choice

When the final iteration is not the most interpretable geometry, use the
iteration-ranking workflow:

- `tc.iteration_ranking` for scoring/selecting saved iterations
- `tc.viz.iteration_choice_plots.plot_iteration_scores(...)`
- `tc.viz.iteration_choice_plots.render_selected_iteration_bundle(...)`

This depends on saved snapshots, so run condensation with `save_every` set.
