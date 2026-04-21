# `trajectory_condensation`

Trajectory condensation: force-based optimization that maps embryo feature vectors
into a shared 2D coordinate space over developmental time.

## Current State

This package is the active implementation of trajectory condensation. It is
organized as a small public API over a split internal design:

- data contract and input conversion in `schema.py`
- initialization in `init_embedding.py`
- solver/config/forces under `condensation/`
- post hoc geometry diagnostics in `iteration_ranking.py`,
  `space_density_metrics.py`, and `force_diagnostics.py`
- visualization in `viz/`
- downstream principal-tree interpretation in `principal_tree/`

For method semantics, read [ALGORITHM.md](ALGORITHM.md). For file ownership and
extension points, read [DESIGN.md](DESIGN.md). The older
`results/.../trajectory_cosmology/` copies should be treated as experiment-local
history; new reusable work should target this package.
For the original conceptual motivation behind trajectory cosmology / trajectory
condensation, see [INITIAL_INSPIRATION.md](INITIAL_INSPIRATION.md).

For agent-facing usage notes, see
[`ai/skills/analyze-trajectory-condensation`](../../../ai/skills/analyze-trajectory-condensation).

## Public Entry Points

Prefer importing from the root package:

```python
import analyze.trajectory_condensation as tc
```

Stable public entry points currently include:

| Task | Entry point |
|------|-------------|
| Validate package-native tensors | `tc.CondensationData`, `tc.validate(...)` |
| Build input from classifier directions | `tc.from_classifier_directions(df, vd, time_col=...)` |
| Build initialization | `tc.init_embedding.aligned_umap_init(...)` |
| Run solver | `tc.run_condensation(...)` with `tc.CondensationConfig` and optional `tc.StoppingConfig` |
| Load a saved NPZ run | `tc.load_run(...)` |
| Render the standard viz bundle | `tc.render_run(...)` |
| Compare saved runs | `tc.compare_runs(...)`, `tc.compare_run_grid(...)` |
| Render condensed time-slice HTML viewer | `tc.time_slice_html(...)` in `viz/condensed_time_slice_viewer.py` |
| Diagnose force balance | `tc.force_snapshot(...)`, `tc.force_target_table(...)` |
| Rank saved iterations | `tc.iteration_ranking` helpers |
| Fit downstream principal tree | `tc.fit_principal_tree(...)` and related `tc.principal_tree` helpers |

Use lower-level module imports only when extending internals or debugging a
specific owner file from [DESIGN.md](DESIGN.md).

## Submodules

| Module | Purpose |
|--------|---------|
| `condensation/` | Core solver: forces, config, engine, stopping |
| `viz/` | Visualization surface: static plots, GIFs, run comparison, condensed time-slice viewer |
| `viz/condensed_time_slice_viewer.py` | Interactive condensed trajectory viewer: 3D trajectory context plus per-time 2D slice slider |
| `schema.py` | `CondensationData` - canonical input tensors |
| `init_embedding.py` | NaN-aware UMAP-based initialization |
| `iteration_ranking.py` | Geometry scoring and iteration selection |
| `space_density_metrics.py` | Per-iteration geometry metrics |
| `force_diagnostics.py` | Force decomposition diagnostics |

## Naming Conventions

### Data types
- `CondensationData` - canonical input tensors (features, mask, time_values, labels)
- `CondensationConfig` - hyperparameters for the solver
- `CondensationResult` - output of a completed run (positions, snapshot_iters, metrics)
- `RunDescriptor` - lightweight container for loading and visualizing a saved run

### Config field prefixes
| Prefix | Domain |
|--------|--------|
| `attract_*` | Attraction force (e.g. `attract_k`, with `attract_weight` as the legacy field name for the strength) |
| `temporal_cohere_*` | Temporal coherence (e.g. `temporal_cohere_window`) |
| `elastic_*` | Elasticity force (e.g. `elastic_strength`, `elastic_mix`) |
| `outlier_*` | Slice outlier correction (e.g. `outlier_strength`, `outlier_cutoff_mode`) |
| `void_*` | Void/occupancy repulsion (e.g. `void_strength`, `void_bandwidth`) |
| `solver_*` | Optimizer (e.g. `solver_lr`, `solver_momentum`, `solver_max_iter`) |
| `fidelity_*` | Anchor fidelity (e.g. `fidelity_init_strength`, `fidelity_half_life`) |
| `local_scale_*` | Local neighborhood scale (e.g. `local_scale_strength`) |

Internal/calibration-only fields (`sigma`, `epsilon_r`, `lambda_stretch`, `lambda_bend`) are
set by geometry calibration and are not user-facing.

### Array shapes
| Name | Shape | Notes |
|------|-------|-------|
| `positions` | `(N_e, T, 2)` | NaN where `mask=False` |
| `mask` | `(N_e, T)` bool | Observation mask |
| `time_values` | `(T,)` float | Sorted ascending, in hpf |
| `labels` | `(N_e,)` str | Genotype or other categorical |
| `position_history` | `(n_snaps, N_e, T, 2)` | Saved during optimization |
| `snapshot_iters` | `list[int]` | Iteration numbers for `position_history` frames |

### Function verb conventions
- `render_*` - high-level orchestration: saves files, returns `dict[str, Path]`
- `plot_*` - creates and returns `(fig, ax)` objects
- `animate_*` - writes GIF/MP4 files
- `score_*` / `select_*` - pure math, no I/O
- `estimate_*` - geometry calibration
- `compute_*` - numerical computation

## Quick Start

```python
import analyze.trajectory_condensation as tc
from analyze.morphology_geometry.io import load_classifier_directions

# 1. Load validated classifier directions
vd = load_classifier_directions("results/.../classifier_directions/", feature_set="vae")

# 2. Build canonical tensors (raw dot-product projection, one dim per vector_id)
data = tc.from_classifier_directions(df, vd, time_col="stage_hpf")

# 3. Initialize and run solver
x0 = tc.init_embedding.aligned_umap_init(data.features, data.mask)
config = tc.CondensationConfig(solver_max_iter=500)
result = tc.run_condensation(x0=x0, mask=data.mask, config=config, save_every=10)

# 4. Visualize
run = tc.load_run("results/.../condensed_positions.npz", title="my run")
tc.render_run(run, "output/my_run/")
```

`aligned_umap_init(...)` is the standard public initialization entry point and
routes through the NaN-aware UMAP path by default. Do not use `pca_init(...)` —
it has not produced acceptable trajectory layouts in this repo.

## Legacy input path (deprecated)

`from_pairwise_margin_csv()` and `from_multiclass_csv()` accepted CSVs from the
old contrast-coordinate classification path. Those values are clipped by the SVM
margin or softmax saturation and give a distorted projection onto the classifier
axis. They are retained in `schema.py` for historical script compatibility
(`results/mcolon/20260329_pbx_crispant_analysis_cont/`) but are **not re-exported
from the package** and emit `DeprecationWarning`. Use `from_classifier_directions()`
for all new work.

## Visualization

See [viz/README_viz.md](viz/README_viz.md) for the full visualization API.

The standard bundle is written by `tc.render_run(...)` and currently includes
static trajectory plots, panel plots, stacked 3D plots, rotation GIFs, and
iteration GIFs when saved `position_history` is available. The interactive
condensed trajectory browser is `tc.time_slice_html(...)`, implemented in
`viz/condensed_time_slice_viewer.py`; it is intentionally separate from generic
Plotly utilities because it owns the two-panel time-slice viewer for condensed
coordinates.

## Force / Config API

See [condensation/state.py](condensation/state.py) for `CondensationConfig` and
[condensation/engine/run.py](condensation/engine/run.py) for `run_condensation`.
For the force breakdown and scale-determination details, see [FORCES.md](FORCES.md).
