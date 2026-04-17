# Trajectory Condensation Algorithm

## Problem Statement
The package solves the following problem:
- Given embryo feature vectors observed across developmental time,
- with missing observations represented by a mask,
- find a shared 2D coordinate system in which embryos form interpretable developmental trajectories.

The intended output is not just a scatterplot. It is a time-aware arrangement of embryo paths that preserves local structure while remaining smooth and readable over time.

## TL;DR
- Input is a masked embryo-by-time feature tensor plus time values and labels.
- Build an initial 2D trajectory layout with the NaN-aware aligned UMAP default.
- Resolve geometry-dependent scales for each force from the current layout.
- Optimize embryo trajectories with a coherence-gated attraction term plus elasticity, fidelity, repulsion, local-scale, outlier, and void corrections.
- Save intermediate snapshots and metrics during optimization.
- Select a representative iteration for inspection when the literal final iterate is not the most interpretable.
- Treat principal-tree fitting as downstream analysis on the condensed coordinates, not part of the solver itself.

## Dynamic Parameters

The user-facing knobs are strength-based, not raw weights.

### API Nomenclature

- `attract_*` is the attraction family; the current code still exposes `attract_weight` as the legacy field name for the attraction strength
- `temporal_cohere_*` is the temporal coherence family; the current code still exposes `temporal_cohere_weight` as the legacy field name for the coherence strength
- `elastic_*` controls stretch/bend regularization
- `fidelity_*` controls the decaying anchor to `x0`
- `local_scale_*` controls local neighborhood scale preservation
- `outlier_*` controls slice-relative outlier suppression
- `void_*` controls broad occupancy / density repulsion
- `solver_*` controls the optimizer

Internal resolved quantities are geometry-calibrated and are not meant to be set directly:

- `sigma` / `sigma_coh`
- `epsilon_r`
- `lambda_stretch` / `lambda_bend`
- `mu`

### Local Geometry

The solver estimates the geometry scales once from the initialization `x0` and reuses them throughout the run. That means users do not need to specify a separate scale for each force every time.

The main geometry references are:

- `s_local`: local neighbor spacing
- `s_step`: per-step trajectory displacement
- `s_bend`: trajectory curvature
- `s_global`: slice-wide / inter-bundle spread

How they are used:

- attraction uses `s_global` for its baseline bandwidth, with `attract_bandwidth_mult` as an optional override
- temporal coherence uses `s_local` for its bandwidth, with `temporal_cohere_bandwidth_mult` as an optional override
- repulsion is normalized from the local spacing scale via `epsilon_r`
- elasticity uses `s_step` for stretch and `s_bend` for bend
- fidelity is anchored to the local spacing scale and decays over iterations
- local-scale preservation uses the fixed neighbor graph and radii estimated from `x0`
- slice outlier handling uses per-slice centers and cutoffs measured from `x0`
- void repulsion uses `s_global` as the broad occupancy scale

### Strength vs Mult

This is the key distinction:

- `mult` chooses the geometric distance or bandwidth the force operates on
- `strength` chooses how much energy or gradient that force contributes once the scale is fixed

Example:

- suppose `s_local = 2.0`
- if `attract_bandwidth_mult = 1.5`, then `sigma_att = 3.0`
- if `void_strength = 0.5`, then the void term is just scaled by `0.5`

So `mult` changes the question “what geometric distance does this force reach?”, while `strength` changes “how hard does it push or pull at that bandwidth?”

### Equation

At a high level, the objective is assembled as

```text
E_total = E_attr + E_rep + E_void + E_elastic + E_fidelity + E_scale + E_outlier
```

where attraction is gated by temporal coherence:

```text
E_attr = - sum_t sum_{i<j} C_ijt * K_ijt * G_ijt
```

- `C_ijt` is the temporal coherence field
- `K_ijt` is the spatial Gaussian kernel
- `G_ijt` is the optional kNN gate

The rest of the terms stabilize the solve, preserve local geometry, and prevent collapse.

### Forces

For the full force breakdown and the scale-determination details, see [FORCES.md](FORCES.md).

At a glance:

- attraction pulls together pairs that the coherence field says belong together
- elasticity keeps each embryo trajectory smooth over time
- fidelity keeps the early solve near the initialization and then decays
- local-scale preservation stops dense and sparse neighborhoods from being treated identically
- repulsion prevents local collapse
- void repulsion creates broader occupancy pressure
- slice outlier handling suppresses detached time-slice strays
- anisotropy is currently a stub for future directional shaping

### Solver

The runtime loop applies SGD with momentum to the resolved total gradient. In practice it:

- recomputes or refreshes coherence as needed
- resolves force scales from geometry calibration
- evaluates the total energy and gradient
- updates positions with the configured learning rate and momentum
- records metrics and optional snapshots
- stops based on the configured convergence monitor or iteration cap

## Core Intuition
Trajectory condensation turns high-dimensional embryo features into a shared 2D developmental movie.
Each embryo is a path through time, not an independent cloud of points.
The solver tries to satisfy two demands at once: preserve biologically meaningful neighborhood structure, and keep each embryo trajectory temporally coherent.
That means the objective is not a plain embedding loss. It is a balance of forces that pull similar observations together, keep paths smooth, resist collapse, and preserve useful structure from the initialization.
The result is a condensed spacetime layout that can be inspected directly or passed downstream to principal-tree analysis.

For the original concept note that motivated this package, see
[INITIAL_INSPIRATION.md](INITIAL_INSPIRATION.md).


## Data Model
The canonical input object is `CondensationData` from [schema.py](schema.py).

Core arrays:
- `features`: per-embryo, per-time feature vectors
- `mask`: boolean observation mask
- `time_values`: sorted developmental times in hpf
- `labels`: categorical per-embryo labels such as genotype
- `embryo_ids`: optional stable embryo identifiers

Core geometry arrays downstream of solving:
- `x0` / initialization: `(N_e, T, 2)`
- `positions`: `(N_e, T, 2)`
- `position_history`: `(n_snaps, N_e, T, 2)` when snapshots are saved

The mask is part of the method, not bookkeeping. Missing observations are expected and must be carried through every stage of initialization, force evaluation, and visualization.

## Stage 1: Initialization
Initialization lives in [init_embedding.py](init_embedding.py).

Recommended initializers:
- `aligned_umap_init(...)`

Role in the algorithm:
- provide a first low-dimensional layout `x0`
- preserve coarse neighborhood geometry before force optimization begins
- give the fidelity term something meaningful to anchor to early in the solve

Why this matters:
- condensation is not initialization-free
- bad `x0` can distort the final geometry even if the force system is reasonable
- the NaN-aware path exists because missing features are common in this data regime

Repository note:
- `aligned_umap_init(...)` is the standard public entry point and now routes
  through the NaN-aware UMAP path by default.
- `pca_init(...)` still exists as a debug helper, but it should not be treated
  as part of the working pipeline here. It has not produced acceptable
  trajectory layouts in practice.

## Stage 2: Geometry Calibration
Geometry calibration lives in [condensation/geometry_refs.py](condensation/geometry_refs.py).

This stage estimates reference quantities from the current layout, including:
- local geometric scales
- slice outlier reference scales
- calibration-only parameters such as `epsilon_r`, `lambda_stretch`, and `lambda_bend`

These are not user-facing biological parameters. They normalize the force system so that a given force strength means something comparable across datasets and initializations.

## Stage 3: Force System
The force terms live in [FORCES.md](FORCES.md), with implementation in [condensation/forces](condensation/forces).

The main force classes are:

- attraction in [condensation/forces/attraction.py](condensation/forces/attraction.py)
- elasticity in [condensation/forces/elasticity.py](condensation/forces/elasticity.py)
- fidelity in [condensation/forces/fidelity.py](condensation/forces/fidelity.py)
- local scale in [condensation/forces/local_scale.py](condensation/forces/local_scale.py)
- repulsion in [condensation/forces/repulsion.py](condensation/forces/repulsion.py)
- slice outlier handling in [condensation/forces/slice_outlier.py](condensation/forces/slice_outlier.py)
- void repulsion in [condensation/forces/void.py](condensation/forces/void.py)
- anisotropy in [condensation/forces/anisotropy.py](condensation/forces/anisotropy.py)
- total assembly in [condensation/forces/total.py](condensation/forces/total.py)

Temporal coherence is not a force module in the same directory, but it is the gate that makes attraction selective instead of unconditional:

- [condensation/coherence/compute.py](condensation/coherence/compute.py)
- [condensation/coherence/kernels.py](condensation/coherence/kernels.py)
- [condensation/coherence/neighborhoods.py](condensation/coherence/neighborhoods.py)

The practical takeaway is:

- coherence decides which pairs are allowed to matter
- attraction turns that gating field into a spatial pull
- the remaining terms keep the solver from breaking the intended trajectory geometry

## Stage 4: Optimization Loop
The main runtime lives in [condensation/engine/run.py](condensation/engine/run.py).

Core responsibilities:
- resolve force balance from config plus geometry refs
- compute coherence structure
- evaluate total energy and gradient
- apply optimizer updates
- save metric rows and optional trajectory snapshots

Main solver knobs:
- `solver_lr`
- `solver_momentum`
- `solver_max_iter`
- `solver_tol`

The optimization is not just “run gradient descent until done.” It is a managed loop that also tracks support diagnostics, force-balance quantities, and snapshot history.

## Stage 5: Stopping
Stopping logic lives in [condensation/engine/stopping.py](condensation/engine/stopping.py).

Core objects:
- `StoppingConfig`
- `StoppingMonitor`

Purpose:
- decide when continued optimization is no longer buying useful structure
- stop based on monitored improvement criteria instead of blindly running to the iteration cap

Stopping is part of the algorithmic contract because the visually best iteration is not guaranteed to be the last one.

## Selecting a Representative Iteration
Post hoc iteration scoring lives in [iteration_ranking.py](iteration_ranking.py).

Purpose:
- score saved iterations by geometry metrics
- select a representative or best iteration for review
- support a workflow where the final iterate is not automatically trusted as the best visual summary

Related metrics live in [space_density_metrics.py](space_density_metrics.py).

This stage matters because condensation is an optimization process, not a closed-form solution. The run can briefly pass through more interpretable states than the literal final one.

## Force Diagnostics
Force diagnostics live in [force_diagnostics.py](force_diagnostics.py).

Purpose:
- decompose the solver state into per-force contributions
- explain why a run collapsed, over-stretched, over-smoothed, or failed to separate classes

This is diagnostic machinery, not part of the objective itself, but it is essential for tuning and failure analysis.

## Outputs
The main solver outputs are represented by `CondensationResult` in [condensation/state.py](condensation/state.py).

Typical saved content includes:
- final `positions`
- `x0` if persisted
- optional `position_history`
- `snapshot_iters`
- per-iteration metrics history

These outputs are what the viz layer and principal-tree layer consume.

## Principal Tree Is Downstream, Not the Solver
Principal-tree logic lives in:
- [principal_tree/core.py](principal_tree/core.py)
- [principal_tree/viz.py](principal_tree/viz.py)

Purpose:
- fit an interpretable tree to the condensed spacetime geometry
- assign observations to branches or arms
- test branch enrichment patterns

This is downstream analysis.
It is not part of the condensation objective and should not be described as if the solver itself is fitting a tree.

## Known Failure Modes
Common failure modes include:
- poor initialization geometry
- force imbalance that over-regularizes or under-regularizes
- branch collapse due to overly strong coherence or repulsion settings
- noisy tails driven by weak support in late or sparse time bins
- over-anchoring to initialization via the fidelity term
- choosing the literal final iteration when an earlier snapshot is more representative

## File Pointers
If you want to read the method in code order:
1. [schema.py](schema.py)
2. [init_embedding.py](init_embedding.py)
3. [condensation/geometry_refs.py](condensation/geometry_refs.py)
4. [condensation/coherence](condensation/coherence)
5. [condensation/forces](condensation/forces)
6. [condensation/engine/run.py](condensation/engine/run.py)
7. [condensation/engine/stopping.py](condensation/engine/stopping.py)
8. [iteration_ranking.py](iteration_ranking.py)
9. [viz/api.py](viz/api.py)
10. [principal_tree/core.py](principal_tree/core.py)
