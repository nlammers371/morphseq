# `trajectory_condensation`

Trajectory condensation: force-based optimization that maps embryo feature vectors
into a shared 2D coordinate space over developmental time.

## Submodules

| Module | Purpose |
|--------|---------|
| `condensation/` | Core solver: forces, config, engine, stopping |
| `viz/` | All visualization: static plots, GIFs, run comparison |
| `schema.py` | `CosmologyData` — canonical input tensors |
| `init_embedding.py` | UMAP / PCA initialization |
| `iteration_ranking.py` | Geometry scoring and iteration selection |
| `space_density_metrics.py` | Per-iteration geometry metrics |
| `force_diagnostics.py` | Force decomposition diagnostics |

## Naming Conventions

### Data types
- `CondensationData` — canonical input tensors (features, mask, time_values, labels)
- `CondensationConfig` — hyperparameters for the solver
- `CondensationResult` — output of a completed run (positions, snapshot_iters, metrics)
- `RunDescriptor` — lightweight container for loading and visualizing a saved run

### Config field prefixes
| Prefix | Domain |
|--------|--------|
| `attract_*` | Attraction force (e.g. `attract_k`, `attract_weight`) |
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
- `render_*` — high-level orchestration: saves files, returns `dict[str, Path]`
- `plot_*` — creates and returns `(fig, ax)` objects
- `animate_*` — writes GIF/MP4 files
- `score_*` / `select_*` — pure math, no I/O
- `estimate_*` — geometry calibration
- `compute_*` — numerical computation

## Quick Start

```python
import analyze.trajectory_condensation as tc

# Run condensation
result = tc.run_condensation(data, config)

# Visualize
run = tc.load_run("condensed_positions.npz", title="my run")
tc.render_run(run, "output/")
```

## Visualization

See [viz/README_viz.md](viz/README_viz.md) for the full visualization API.

## Force / Config API

See [condensation/state.py](condensation/state.py) for `CondensationConfig` and
[condensation/engine/run.py](condensation/engine/run.py) for `run_condensation`.
