# Trajectory Condensation Algorithm

## TL;DR
- Input is a masked embryo-by-time feature tensor plus time values and labels.
- Build an initial 2D trajectory layout with aligned UMAP or PCA.
- Calibrate geometry-dependent reference scales from the current layout.
- Optimize embryo trajectories in 2D using a weighted sum of attraction, coherence, elasticity, fidelity, and repulsion-style forces.
- Save intermediate snapshots and metrics during optimization.
- Select a representative iteration for inspection when the literal final iterate is not the most interpretable.
- Treat principal-tree fitting as downstream analysis on the condensed coordinates, not part of the solver itself.

## Core Intuition
Trajectory condensation turns high-dimensional embryo features into a shared 2D developmental movie.
Each embryo is a path through time, not an independent cloud of points.
The solver tries to satisfy two demands at once: preserve biologically meaningful neighborhood structure, and keep each embryo trajectory temporally coherent.
That means the objective is not a plain embedding loss. It is a balance of forces that pull similar observations together, keep paths smooth, resist collapse, and preserve useful structure from the initialization.
The result is a condensed spacetime layout that can be inspected directly or passed downstream to principal-tree analysis.

## Problem Statement
The package solves the following problem:
- Given embryo feature vectors observed across developmental time,
- with missing observations represented by a mask,
- find a shared 2D coordinate system in which embryos form interpretable developmental trajectories.

The intended output is not just a scatterplot. It is a time-aware arrangement of embryo paths that preserves local structure while remaining smooth and readable over time.

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

Available initializers:
- `aligned_umap_init(...)`
- `nan_aware_aligned_umap_init(...)`
- `pca_init(...)`

Role in the algorithm:
- provide a first low-dimensional layout `x0`
- preserve coarse neighborhood geometry before force optimization begins
- give the fidelity term something meaningful to anchor to early in the solve

Why this matters:
- condensation is not initialization-free
- bad `x0` can distort the final geometry even if the force system is reasonable
- the NaN-aware path exists because missing features are common in this data regime

## Stage 2: Geometry Calibration
Geometry calibration lives in [condensation/geometry_refs.py](condensation/geometry_refs.py).

This stage estimates reference quantities from the current layout, including:
- local geometric scales
- slice outlier reference scales
- calibration-only parameters such as `epsilon_r`, `lambda_stretch`, and `lambda_bend`

These are not user-facing biological parameters. They normalize the force system so that a given force strength means something comparable across datasets and initializations.

## Stage 3: Force System
The force terms live in [condensation/forces](condensation/forces).

### Attraction
File: [condensation/forces/attraction.py](condensation/forces/attraction.py)

Purpose:
- pull observations with strong support or similarity toward each other
- preserve local neighborhood structure from the input features

Main config knobs:
- `attract_k`
- `attract_weight`
- `attract_bandwidth_mult`

### Temporal Coherence
Files:
- [condensation/coherence/compute.py](condensation/coherence/compute.py)
- [condensation/coherence/kernels.py](condensation/coherence/kernels.py)
- [condensation/coherence/neighborhoods.py](condensation/coherence/neighborhoods.py)

Purpose:
- make each embryo trajectory temporally coherent
- define how nearby time bins influence one another
- prevent jagged, discontinuous path behavior that is inconsistent with developmental time

Main config knobs:
- `temporal_cohere_window`
- `temporal_cohere_mode`
- `temporal_cohere_weight`
- `temporal_cohere_bandwidth_mult`

### Elasticity
File: [condensation/forces/elasticity.py](condensation/forces/elasticity.py)

Purpose:
- regularize path shape over time
- penalize excessive stretching and bending
- stabilize trajectories when attraction alone is too noisy

Main config knobs:
- `elastic_strength`
- `elastic_mix`
- `elastic_kernel`

### Fidelity to Initialization
File: [condensation/forces/fidelity.py](condensation/forces/fidelity.py)

Purpose:
- keep early optimization near the initialization
- reduce the chance that the solver immediately destroys meaningful coarse geometry
- allow that anchor to decay over time

Main config knobs:
- `fidelity_init_strength`
- `fidelity_half_life`

### Local Scale
File: [condensation/forces/local_scale.py](condensation/forces/local_scale.py)

Purpose:
- adapt force behavior to local neighborhood scale
- reduce pathologies where dense and sparse regions are treated identically

Main config knob:
- `local_scale_strength`

### Slice Outlier Handling
File: [condensation/forces/slice_outlier.py](condensation/forces/slice_outlier.py)

Purpose:
- correct or suppress pathological per-slice outliers
- keep a few extreme observations from dominating local geometry

Main config knobs:
- `outlier_strength`
- `outlier_cutoff_mode`
- `outlier_cutoff_value`

### Repulsion and Void Control
Files:
- [condensation/forces/repulsion.py](condensation/forces/repulsion.py)
- [condensation/forces/void.py](condensation/forces/void.py)

Purpose:
- prevent collapse into degenerate overlap
- control occupancy of space
- keep paths from unrealistically piling into the same region

Main config knobs:
- `void_strength`
- `void_bandwidth`

### Anisotropy
File: [condensation/forces/anisotropy.py](condensation/forces/anisotropy.py)

Purpose:
- apply directional regularization where needed
- discourage geometries that violate the intended directional structure of the layout

This is a more specialized term than attraction or coherence and should be treated as a shaping term, not the main organizing signal.

### Total Force / Objective Assembly
File: [condensation/forces/total.py](condensation/forces/total.py)

Purpose:
- combine all active terms into one energy and gradient
- serve as the single aggregation point for the solver loop

This file is the mathematical join point of the method.

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
