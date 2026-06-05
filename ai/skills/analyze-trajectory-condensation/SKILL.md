# analyze-trajectory-condensation - Solver Reference

Use this skill for the active trajectory condensation package:
`src/analyze/trajectory_condensation/`.

This is distinct from `src/analyze/trajectory_analysis/`, which owns older DTW
clustering and bootstrap projection workflows.

## Current State

- The package-level docs are the source of truth:
  - `src/analyze/trajectory_condensation/README.md` - quick use and public API
  - `src/analyze/trajectory_condensation/ALGORITHM.md` - method semantics
  - `src/analyze/trajectory_condensation/DESIGN.md` - file ownership and extension points
  - `src/analyze/trajectory_condensation/viz/README_viz.md` - visualization API
- Import from the root package for normal use:

```python
import analyze.trajectory_condensation as tc
```

- Treat `results/.../trajectory_cosmology/` copies as experiment-local history.
  New reusable work should target `src/analyze/trajectory_condensation/`.
- Visualization is separate from solving.
- Principal-tree fitting is downstream interpretation of condensed coordinates,
  not part of the condensation solver.

## Public Entry Points

| Task | Entry point |
|---|---|
| Data contract | `tc.CondensationData`, `tc.validate(...)` |
| Build input from directions | `tc.from_classifier_directions(df, vd, time_col=...)` |
| Initialization | `tc.init_embedding.aligned_umap_init(...)` |
| Solver config | `tc.CondensationConfig`, `tc.StoppingConfig` |
| Run solver | `tc.run_condensation(...)` |
| Force diagnostics | `tc.force_snapshot(...)`, `tc.force_target_table(...)`, `tc.describe_force_balance(...)` |
| Iteration scoring | `tc.iteration_ranking` |
| Saved-run viz | `tc.load_run(...)`, `tc.render_run(...)`, `tc.compare_runs(...)`, `tc.compare_run_grid(...)` |
| Condensed time-slice viewer | `tc.time_slice_html(...)` in `src/analyze/trajectory_condensation/viz/condensed_time_slice_viewer.py` |
| Principal tree | `tc.fit_principal_tree(...)`, `tc.project_observations_to_tree(...)`, `tc.run_all_branch_tests(...)` |

## Minimal Solver Pattern

```python
import analyze.trajectory_condensation as tc
from analyze.morphology_geometry.io import load_classifier_directions

vd = load_classifier_directions("results/.../classifier_directions/", feature_set="vae")
data = tc.from_classifier_directions(df, vd, time_col="stage_hpf")
x0 = tc.init_embedding.aligned_umap_init(data.features, data.mask)

config = tc.CondensationConfig(
    solver_max_iter=500,
    solver_lr=0.01,
    attract_k=15,
    attract_weight=1.0,
    temporal_cohere_window=3,
    temporal_cohere_weight=1.0,
)

result = tc.run_condensation(
    x0=x0,
    mask=data.mask,
    config=config,
    save_every=10,
    verbose=True,
)
```

Do not use `from_pairwise_margin_csv()` or `from_multiclass_csv()` for new work —
those functions read contrast-coordinate CSVs where values are clipped by the SVM
margin, giving a distorted projection. They emit `DeprecationWarning` and are not
re-exported from the package. Historical scripts in
`results/mcolon/20260329_pbx_crispant_analysis_cont/` still use them.

Do not recommend `pca_init(...)`. It has not worked well enough to be supported.

## Data Contract

Core input arrays:

| Name | Shape | Notes |
|---|---|---|
| `features` | `(N_e, T, F)` | embryo-by-time feature tensor |
| `mask` | `(N_e, T)` bool | True where an embryo has an observation |
| `time_values` | `(T,)` float | sorted hpf values |
| `labels` | `(N_e,)` str | genotype or other category |
| `embryo_ids` | `(N_e,)` str | stable embryo identifiers when available |

Core solver/viz arrays:

| Name | Shape | Notes |
|---|---|---|
| `x0` | `(N_e, T, 2)` | initialization |
| `positions` | `(N_e, T, 2)` | final or selected condensed coordinates |
| `position_history` | `(n_snaps, N_e, T, 2)` | saved snapshots when `save_every` is set |
| `snapshot_iters` | `list[int]` | iteration numbers for snapshots |

Missing observations must stay represented by `mask`; downstream positions are
usually NaN where `mask=False`.

## Ownership Map

- `schema.py` owns shapes, masks, labels, and CSV conversion.
- `init_embedding.py` owns the NaN-aware UMAP initialization path.
- `condensation/state.py` owns `CondensationConfig`, `CondensationResult`, and state dataclasses.
- `condensation/geometry_refs.py` owns geometry-dependent calibration scales.
- `condensation/forces/*` owns individual force families.
- `condensation/forces/total.py` owns force aggregation.
- `condensation/engine/run.py` owns the runtime loop.
- `condensation/engine/stopping.py` owns stopping policy.
- `iteration_ranking.py` owns post hoc iteration scoring.
- `viz/` owns rendering only.
- `principal_tree/` owns downstream branch/tree interpretation.

When changing internals, read `DESIGN.md` first and edit the owner file instead
of adding cross-cutting logic to the engine.
