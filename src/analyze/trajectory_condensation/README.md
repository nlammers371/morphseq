# `trajectory_condensation`

Embeds embryo morphology trajectories into a 2D space and optimizes their layout via a physics-inspired condensation step. Produces static plots and GIF animations for inspection and comparison.

---

## Quick start

```python
import analyze.trajectory_condensation as tc

# Load a run from disk
run = tc.load_run("results/.../condensed_positions.npz", title="shrunk 4class")

# Write all standard outputs (plots + GIFs) to a directory
tc.render_run(run, "output/my_run/")

# Compare two runs side by side
r1 = tc.load_run("results/.../run_A/condensed_positions.npz", title="run A")
r2 = tc.load_run("results/.../run_B/condensed_positions.npz", title="run B")

tc.compare_runs([r1, r2], mode="trajectories", output_path="compare.png")
tc.compare_runs([r1, r2], mode="stacked_3d",  output_path="compare_3d.png")
```

---

## Data contract

All visualization functions share the same input shapes:

| Array | Shape | Notes |
|-------|-------|-------|
| `positions` | `(N_e, T, 2)` | NaN where `mask=False` |
| `mask` | `(N_e, T)` bool | `True` = embryo observed at that time bin |
| `time_values` | `(T,)` float | Sorted ascending (hpf) |
| `labels` | `(N_e,)` str | Optional; genotype or other category |
| `color_map` | `dict[str, str]` | Optional; auto-generated from `tab10` if absent |

`RunDescriptor.__post_init__` validates these shapes and normalizes `positions[~mask] = NaN`.

The standard NPZ format (written by condensation runner scripts) contains all of the above plus `embryo_ids`, `x0`, `position_history`, and `snapshot_iters`. `load_run()` reads all of them.

---

## Public API

### `VizConfig`

Shared style defaults. Pass to `render_run` or `compare_runs` to override.

```python
cfg = tc.VizConfig(
    dpi=150,
    figsize=(8, 7),
    alpha_line=0.25,
    alpha_point=0.6,
    point_size=8.0,
    linewidth=0.8,
    panel_size=3.0,   # size of each panel in plot_panels
)
```

Animation-specific knobs (fps, n_frames, elev, azim) are internal constants — open an issue or edit `viz.py::_FPS_*` / `_ELEV` / `_AZIM_*` if you need to override them.

---

### `load_run(path, title="", color_map=None)`

```python
run = tc.load_run("condensed_positions.npz", title="my experiment")
```

Required NPZ keys: `positions`, `mask`, `time_values`.
Optional: `labels`, `embryo_ids`, `x0`, `position_history`, `snapshot_iters`.

---

### `render_run(run, output_dir, *, config=None, title_prefix="", snapshot_idx=None, skip_animations=False)`

Writes the full output bundle and returns `{filename: Path}`.

| File | When |
|------|------|
| `plot_trajectories.png` | always |
| `plot_trajectories_init.png` | if `run.x0` present |
| `plot_panels.png` | always |
| `plot_stacked_3d.png` | always |
| `rotation.gif` | unless `skip_animations` |
| `init_vs_final_rotation.gif` | unless `skip_animations`, if `run.x0` present |
| `iterations.gif` | unless `skip_animations`, if `position_history` has ≥2 frames |
| `iterations.png` | if `position_history` has exactly 1 frame (+ warning) |

`snapshot_idx`: list of integer indices into `time_values` for the panels plot. Defaults to 6 evenly spaced frames.

The iterations GIF uses a slow camera orbit (`rotation=True`) by default.

---

### `compare_runs(runs, mode="trajectories", *, config=None, align_axes=True, figsize=None, output_path=None)`

Side-by-side figure for N≥2 runs.

```python
# Interactive (returns open figure)
fig, axes = tc.compare_runs([r1, r2, r3], mode="trajectories")

# Save and close (returns Path)
tc.compare_runs([r1, r2], mode="stacked_3d",
                align_axes=True,
                output_path="compare_3d.png")
```

**Modes (v1):**
- `"trajectories"` — 2D per-embryo lines, one panel per run
- `"stacked_3d"` — 3D stacked view (x, y, z=time), one panel per run

`align_axes=True` (default): unifies axis limits across all panels for direct visual comparison.

**Embryo alignment:** `compare_runs` compares distributions / overall shape. It does **not** align embryos by ID across runs — index order within each run is used as-is. A `UserWarning` is emitted when runs have differing `embryo_ids`.

---

## Lower-level API

Use these directly when you need fine-grained control or want to compose your own figure layout.

### `plotting.py`

```python
from analyze.trajectory_condensation import plotting

fig, ax   = plotting.plot_trajectories(positions, mask, time_values, labels=..., color_map=..., title=...)
fig, axes = plotting.plot_panels(positions, mask, time_values, snapshot_times=[24., 28., ...], title=...)
fig, ax   = plotting.plot_stacked_3d(positions, mask, time_values, elev=25., azim=-60., title=...)
fig, ax   = plotting.plot_color_by_time(positions, mask, time_values, cmap_name="viridis")
```

`resolve_color_map(labels, color_map)` is also public — useful for building consistent color maps across multiple figures in the same script.

### `animation.py`

```python
from analyze.trajectory_condensation import animation

animation.animate_rotation(positions, mask, time_values, output_path="rotation.gif",
                           n_frames=120, fps=24, elev=25., azim_start=-60., azim_end=300.)

animation.animate_iterations(position_history, mask, time_values, output_path="iters.gif",
                             rotation=True,   # slow camera orbit across frames (default: False)
                             azim_end=300.,   # target azimuth when rotation=True
                             fps=4)

animation.animate_init_final_rotation(x0, positions, mask, time_values,
                                      output_path="init_vs_final.gif")
```

**`animate_iterations` rotation knob:**
- `rotation=False` (default) — fixed camera; only geometry changes frame to frame
- `rotation=True` — camera orbits from `azim` to `azim_end` across all frames

---

## Iteration ranking

After condensation, use `iteration_ranking` to score and render the best candidate iterations:

```python
from analyze.trajectory_condensation.iteration_ranking import (
    score_saved_iterations,
    plot_iteration_scores,
    render_selected_iteration_bundle,
    save_ranking_outputs,
)

scores = score_saved_iterations(
    position_history, snapshot_iters, mask, labels, time_values,
    metrics_df=metrics_df,
    objective="balanced",
)
plot_iteration_scores(scores, "ranking/plot_scores.png")

for _, row in scores.head(3).iterrows():
    render_selected_iteration_bundle(
        positions=..., mask=mask, time_values=time_values,
        labels=labels, color_map=color_map,
        output_dir=f"ranking/iter_{int(row.snapshot_iter):04d}_rank_{int(row.rank]):02d}/",
        title_prefix="my experiment",
        snapshot_iter=int(row.snapshot_iter),
        metadata={...},
    )
```

---

## Schema / data loading

```python
from analyze.trajectory_condensation import schema

# From a pairwise signed-margin CSV (columns like "ctrl_vs_crispant")
data = schema.from_pairwise_margin_csv("pairwise_shrunk_vectors.csv", label_col="genotype")

# From a multiclass probability CSV (columns like "p_ctrl", "p_crispant")
data = schema.from_multiclass_csv("classifier_probs.csv", label_col="genotype")

# data.features  (N_e, T, K)
# data.mask      (N_e, T)
# data.labels    (N_e,)
# data.time_values (T,)
```

---

## Typical pipeline

```
schema.from_*_csv()
  → init_embedding.aligned_umap_init()  [or nan_aware_aligned_umap_init]
  → run_condensation(x0, mask, config, stopping, save_every=25)
  → tc.render_run(run, output_dir)          # all standard plots + GIFs
  → score_saved_iterations(...)             # rank snapshots
  → render_selected_iteration_bundle(...)   # render top-K candidates
```

See `results/mcolon/20260407_pbx_analysis_cont/02_pairwise_trajectory_condensation.py` for the reference implementation.
