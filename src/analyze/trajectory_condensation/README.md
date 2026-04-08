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
