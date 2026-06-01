# Trajectory Condensation Design

## TL;DR
- `ALGORITHM.md` explains what the solver does.
- This document explains which file owns which part of the package.
- The package is intentionally split into four layers: data contract, solver, visualization, and downstream tree analysis.
- Force terms are modular by file, aggregated in one place, and executed by a single engine loop.
- Visualization is separate from solving.
- Principal-tree analysis is separate from both.

## Design Principles
- Keep method logic separate from rendering.
- Keep data-contract code separate from optimization code.
- Keep individual force definitions isolated so each term has a visible owner.
- Keep the public API small enough to be usable, but broad enough that common workflows do not require internal imports.
- Treat principal-tree fitting as downstream interpretation, not as part of the solver core.

## Public API Surface
Stable imports are meant to come from [__init__.py](__init__.py).

Main public concepts:
- `CondensationData` and data-loading helpers from [schema.py](schema.py)
- `CondensationConfig`, `CondensationResult`, and related dataclasses from [condensation/state.py](condensation/state.py)
- `run_condensation`, `resolve_force_balance`, and related runtime helpers from the condensation package
- viz entrypoints such as `load_run`, `render_run`, `compare_runs`, and `time_slice_html`
- principal-tree entrypoints for downstream branch analysis

Internal mechanics are allowed to be more fluid inside:
- `condensation/coherence/*`
- `condensation/forces/*`
- `condensation/engine/*`

That split is deliberate. Users should not need to know which low-level file computes a single force term in order to run the package.

## Package Ownership Map

### Root package
- [__init__.py](__init__.py)
  - public export surface only
- [README.md](README.md)
  - package overview and naming conventions
- [ALGORITHM.md](ALGORITHM.md)
  - method semantics
- [INITIAL_INSPIRATION.md](INITIAL_INSPIRATION.md)
  - original conceptual motivation
- [DESIGN.md](DESIGN.md)
  - architecture and ownership

### Data contract
- [schema.py](schema.py)
  - owns canonical `CondensationData`
  - validates shapes, masks, labels, and time axes
  - converts external CSV outputs into package-native tensors

If something is wrong with shapes, masks, or array semantics, this is the file to inspect first.

### Initialization
- [init_embedding.py](init_embedding.py)
  - owns aligned UMAP, NaN-aware aligned UMAP, and PCA init
  - produces `x0`

This file is responsible for starting geometry, not for optimizing it.

### Solver surface
- [condensation/__init__.py](condensation/__init__.py)
  - solver export wrapper
- [condensation/api.py](condensation/api.py)
  - public-facing solver API layer
- [condensation/state.py](condensation/state.py)
  - dataclasses and config/state/result contracts
- [condensation/geometry_refs.py](condensation/geometry_refs.py)
  - geometry-dependent calibration references

This layer defines what the solver expects and returns.

### Coherence internals
- [condensation/coherence/compute.py](condensation/coherence/compute.py)
  - coherence computation
- [condensation/coherence/kernels.py](condensation/coherence/kernels.py)
  - coherence kernels
- [condensation/coherence/neighborhoods.py](condensation/coherence/neighborhoods.py)
  - time/local neighborhood construction

This subpackage owns temporal coherence math and should not leak plotting or solver-loop concerns.

### Force internals
- [condensation/forces/attraction.py](condensation/forces/attraction.py)
- [condensation/forces/elasticity.py](condensation/forces/elasticity.py)
- [condensation/forces/fidelity.py](condensation/forces/fidelity.py)
- [condensation/forces/local_scale.py](condensation/forces/local_scale.py)
- [condensation/forces/repulsion.py](condensation/forces/repulsion.py)
- [condensation/forces/slice_outlier.py](condensation/forces/slice_outlier.py)
- [condensation/forces/void.py](condensation/forces/void.py)
- [condensation/forces/anisotropy.py](condensation/forces/anisotropy.py)
- [condensation/forces/total.py](condensation/forces/total.py)

Ownership rule:
- each file owns one force family
- [total.py](condensation/forces/total.py) owns aggregation

This prevents force logic from collapsing into a single opaque engine file.

### Engine internals
- [condensation/engine/run.py](condensation/engine/run.py)
  - the main optimizer loop
  - resolves force balance
  - computes metrics and snapshots
- [condensation/engine/stopping.py](condensation/engine/stopping.py)
  - stopping policy and monitor

This layer owns runtime control flow.
It should orchestrate force evaluation, not define the mathematics of every force.

### Diagnostics and post hoc selection
- [force_diagnostics.py](force_diagnostics.py)
  - force decomposition and debugging views
- [space_density_metrics.py](space_density_metrics.py)
  - geometry metrics for saved states
- [iteration_ranking.py](iteration_ranking.py)
  - iteration scoring and representative-iteration selection

These files exist because the best interpretation point is not always the literal final iterate.

### Visualization
- [viz/__init__.py](viz/__init__.py)
  - viz export surface
- [viz/api.py](viz/api.py)
  - high-level rendering orchestration
- [viz/plotting.py](viz/plotting.py)
  - static figure builders
- [viz/animation.py](viz/animation.py)
  - GIF/animation generation
- [viz/condensed_time_slice_viewer.py](viz/condensed_time_slice_viewer.py)
  - condensed time-slice HTML viewer: 3D trajectory context plus per-time 2D slice slider
- [viz/iteration_choice_plots.py](viz/iteration_choice_plots.py)
  - plots specific to iteration-ranking workflows
- [viz/README_viz.md](viz/README_viz.md)
  - viz contract and bundle documentation

Visualization should consume solved outputs.
It should not contain solver policy.

### Principal tree
- [principal_tree/__init__.py](principal_tree/__init__.py)
  - tree-analysis export surface
- [principal_tree/core.py](principal_tree/core.py)
  - tree fitting, projection, branch assignment, branch testing
- [principal_tree/viz.py](principal_tree/viz.py)
  - tree-specific plots and animations

This subpackage is intentionally downstream of condensation. It interprets geometry after the solver has run.

## Data Flow
1. External tables are converted into `CondensationData` in [schema.py](schema.py).
2. An initialization `x0` is built in [init_embedding.py](init_embedding.py).
3. Geometry references are estimated in [condensation/geometry_refs.py](condensation/geometry_refs.py).
4. Coherence and force terms are assembled by the condensation internals.
5. [condensation/engine/run.py](condensation/engine/run.py) produces a `CondensationResult`.
6. Viz functions consume the saved run or result object.
7. Principal-tree analysis optionally consumes the condensed geometry downstream.

## Why The Force Split Exists
The force split is not aesthetic. It is there so that:
- each term has one clear owner
- changing one term does not require editing the optimizer loop
- diagnostics can reason about forces individually
- future force additions do not turn `run.py` into a monolith

The aggregation point in [condensation/forces/total.py](condensation/forces/total.py) is the contract boundary between force definition and solver execution.

## Why Visualization Is Separate
The package would become much harder to trust if solver files also owned rendering.

Keeping `viz/` separate means:
- solver changes do not silently change plotting behavior
- rendering code can evolve without mutating force logic
- batch figure writing, Plotly HTML, and GIF generation stay out of the numerical core

## Why Principal Tree Is Separate
The condensed layout is one product.
The fitted principal tree is an interpretation of that product.

Keeping `principal_tree/` separate prevents a false story where the optimizer is secretly solving a tree-fitting problem. It is not.

## Developer Reading Order
For a new developer, the clean reading order is:
1. [ALGORITHM.md](ALGORITHM.md)
2. [INITIAL_INSPIRATION.md](INITIAL_INSPIRATION.md)
3. [README.md](README.md)
4. [schema.py](schema.py)
5. [init_embedding.py](init_embedding.py)
6. [condensation/state.py](condensation/state.py)
7. [condensation/geometry_refs.py](condensation/geometry_refs.py)
8. [condensation/forces/total.py](condensation/forces/total.py)
9. [condensation/engine/run.py](condensation/engine/run.py)
10. [iteration_ranking.py](iteration_ranking.py)
11. [viz/api.py](viz/api.py)
12. [principal_tree/core.py](principal_tree/core.py)

## Future Improvements
- add a short force registry table with one row per force term and one owner file
- add a small developer note for how to introduce a new force without breaking the total-force contract
- tighten README references so they point to the current engine files instead of older paths mentioned historically
- consider a dedicated `docs/` directory only if the package-level docs grow beyond a few focused markdown files
