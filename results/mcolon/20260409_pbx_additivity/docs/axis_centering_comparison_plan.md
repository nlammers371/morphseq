# Plan: Raw-Direction Centering Comparison Through Standard Trajectory Condensation

## Goal

Pick the centering scheme that gives the best-looking condensed developmental
trajectories while staying faithful to the original trajectory-cosmology idea:

- build **relative coordinates** from raw classifier directions
- generate a UMAP-based initialization from those coordinates
- run the standard trajectory-condensation solver
- judge the result using the standard run bundle and GIFs

This is not a generic feature-table exercise. The real decision surface is the
final condensed trajectory geometry.

For the original motivation, keep
`src/analyze/trajectory_condensation/INITIAL_INSPIRATION.md` in view while
implementing this. The point is persistent trajectory structure, not just
per-bin score prettiness.

## Principle

The representation choice and the initialization choice are separate.

1. Representation:
   centered raw-direction coordinates built from morphology geometry
2. Initialization:
   aligned UMAP on the resulting feature cube
3. Condensation:
   standard force-based solver
4. Evaluation:
   standard trajectory-condensation output bundle

The previous draft got lost by mixing these layers together and by introducing a
custom condensation path instead of using the existing pipeline shape.

## What We Are Comparing

### Representation variants

For each `(comparison_id, time_bin_center)` direction vector from the full
`all_pairs` genotype set, project embryos onto the raw unit direction and then
center that score in one of these ways:

| Key | Formula | Meaning |
|---|---|---|
| `intercept_centered` | `raw_score + intercept / coef_norm` | zero = classifier boundary |
| `neg_centroid_centered` | `raw_score - neg_binned_mean` | zero = reference-class mean |
| `midpoint_centered` | `raw_score - 0.5 * (pos_binned_mean + neg_binned_mean)` | zero = midpoint between class clouds |
| `raw_projection` | `raw_score` | origin-sensitive technical control |

### Reference baseline

Keep the existing `pairwise_raw_vectors.csv` run as a contextual reference, but
do not treat it as a fully peer method because it comes from a different
representation path and different binning conventions.

If we want a fair boundary-centered baseline, generate it in the same
raw-direction pipeline at the same bin width.

## Hard Invariants

### Representation invariant

`raw_score = unit_coef dot x_binned`

All centering variants must be computed from the same binned projected
representation for a given vector and time bin.

### Binning invariant

Projection inputs must match the classifier-direction binning contract.

Use:
- `_build_binary_labels(...)`
- `_bin_and_aggregate(...)`

from `src/analyze/classification/engine/data_prep.py`, in the same comparison
loop shape used by `extract_classifier_directions(...)`.

Do not use an ad hoc manual `groupby(...).mean()` on the full dataframe.

### Initialization invariant

Use UMAP-based initialization only:
- `aligned_umap_init(...)`

Do not route this workflow through `pca_init(...)`.

## Files

### Create

```text
results/mcolon/20260410_axis_init_comparison/
└── 01_axis_centering_comparison.py
```

### Read

- `results/mcolon/20260407_pbx_analysis_cont/common.py`
- `src/analyze/classification/directions/extract.py`
- `src/analyze/classification/engine/data_prep.py`
- `src/analyze/morphology_geometry/`
- `src/analyze/trajectory_condensation/schema.py`
- `src/analyze/trajectory_condensation/init_embedding.py`
- `src/analyze/trajectory_condensation/condensation/api.py`
- `src/analyze/trajectory_condensation/viz/api.py`
- `results/mcolon/20260407_pbx_analysis_cont/02_pairwise_trajectory_condensation.py`
- `results/mcolon/20260407_pbx_analysis_cont/09_run_subset_init_condensation_matrix.py`

## Implementation Shape

Use the same two-stage pattern that already exists in the PBX condensation
scripts:

1. Build and save candidate `x0` bundles
2. Run condensation from saved `x0`
3. Render the standard output bundle
4. Compare runs with the package viz API

That pattern is already closer to the intended workflow than a bespoke
single-script condensation implementation.

## Stage A: Build Centered Coordinate Cubes

### A1. Extract classifier directions

Use `extract_classifier_directions(...)` with `comparisons="all_pairs"` and the
desired feature set.

### A2. Reproduce the classification binning contract

For each resolved comparison:

1. filter with `_build_binary_labels(...)`
2. bin with `_bin_and_aggregate(...)`
3. compute raw projections against the unit direction vector for every genotype
   pair and time bin
4. compute all centering variants from that shared projected table

### A3. Save representation artifacts per variant

For each centering variant, save:

```text
results/<variant>/
├── wide_scores.parquet
├── axis_metadata.parquet
├── summary.csv
└── data_manifest.json
```

`wide_scores.parquet` is the actual trajectory-condensation input table for that
variant. For the five-genotype PBX sensitivity set, the intended structure is
`5 choose 2 = 10` pairwise comparisons across the supported time bins, so the
feature count should be approximately `10 x n_time_bins`.

### A4. Build `CondensationData`

Convert each variant table into a canonical feature cube:

- `features`: `(N_e, T, K_variant)`
- `mask`
- `embryo_ids`
- `time_values`
- `labels`

Call `schema.validate(..., allow_feature_nans=True)` if the representation is
sparse.

## Stage B: Build UMAP Initializations

For each centering variant:

1. load its `CondensationData`
2. call `aligned_umap_init(...)`
3. rely on the default NaN-aware path
4. save `x0_init.npz`
5. render init-only preview plots

Outputs:

```text
results/<variant>/init/
├── x0_init.npz
├── plot_trajectories_init.png
└── plot_stacked_3d_init.png
```

This stage answers:
- does the representation already produce coherent developmental geometry before
  solving?

## Stage C: Run Standard Trajectory Condensation

For each variant:

1. load the saved `x0_init.npz`
2. run `run_condensation(...)`
3. save `condensed_positions.npz` with:
   - `positions`
   - `x0`
   - `mask`
   - `time_values`
   - `embryo_ids`
   - `labels`
   - `position_history`
   - `snapshot_iters`
4. save `metrics.csv`
5. call `tc.render_run(...)`

Outputs:

```text
results/<variant>/run/
├── condensed_positions.npz
├── metrics.csv
├── plot_trajectories.png
├── plot_trajectories_init.png
├── plot_panels.png
├── plot_stacked_3d.png
├── time_slice.gif
├── rotation.gif
├── init_vs_final_rotation.gif
└── iterations.gif
```

These are the primary artifacts for deciding what "looks best."

## Stage D: Compare Runs

Generate package-native comparison outputs:

- `compare_run_grid(..., mode="trajectories")`
- `compare_run_grid(..., mode="stacked_3d")`

Suggested layout:
- rows = centering variant
- columns = init / final / selected iteration view if needed

Also keep a small set of pre-condensation diagnostics:

- center-values-over-time plot
- score distribution plots for representative bins

But these are sanity checks, not the main deliverable.

## Decision Criteria

Choose the centering variant that best satisfies all of the following:

1. Init geometry is already plausible and not obviously folded or noisy.
2. Final condensed trajectories show smooth, interpretable developmental flow.
3. Branching and divergence structure are visible without obvious collapse.
4. The run is stable across saved iterations, not only at one lucky frame.
5. The result still feels like relative trajectory structure, not a decorative
   2D embedding.

## API Weaknesses This Plan Exposes

This exercise highlights a few places where the API is weaker than it should be.

### 1. No package-native builder from long/wide score tables to `CondensationData`

We repeatedly hand-build the cube. That should become a supported helper.

Potential standardization:
- `tc.from_wide_scores(...)`
- `tc.from_long_scores(...)`

### 2. No standard "save init bundle" helper

The repo already uses `x0_init.npz` plus preview plots, but this is not exposed
as a reusable package function.

Potential standardization:
- `tc.save_init_bundle(...)`
- `tc.render_init_bundle(...)`

### 3. No standard runner that takes `(data, x0, output_dir)`

Current experiment scripts reimplement the same save/render flow.

Potential standardization:
- `tc.run_and_save(...)`

### 4. Representation sweeps and condensation sweeps are not clearly separated

The package is strong at solving and rendering runs, but weak at standardizing
representation experiments upstream of solving.

Potential standardization:
- a small `workflows/` layer or helper module for representation-to-run sweeps

### 5. The historical vision was easier to lose than the API

That is now partially fixed by promoting the original inspiration doc into the
package docs, but it is still a sign that the conceptual framing needs a stable
home.

## Working Boundary For This Project

For this PBX additivity work, keep active method development inside the local
`phenotype_direction` package unless there is a clear reason not to.

Working rule:
- prefer adding and changing code in `results/mcolon/20260409_pbx_additivity/phenotype_direction/`
- do not grow `analyze.morphology_geometry` with speculative primitives that may
  not survive this project
- if a `morphology_geometry` helper is needed locally, it is acceptable to copy
  it into `phenotype_direction` and evolve it there
- only export code back to `morphology_geometry` after the method is stable and
  demonstrably useful outside this project

This keeps the results-side package as the incubator and avoids polluting the
shared package with primitives we never end up using.

## Final Cleanup Step

At the end of this project, do a trim pass:

1. identify which local `phenotype_direction` methods were actually used
2. delete dead local helpers and compatibility layers
3. decide which surviving pieces are genuinely generic
4. export only those generic survivors into `analyze.morphology_geometry`
5. leave project-specific workflow code in the results folder

The goal is to trim all the fat after the representation and condensation
workflow has settled, not during the exploratory phase.

## Verification

### Smoke

```bash
conda run -n segmentation_grounded_sam --no-capture-output \
  python results/mcolon/20260410_axis_init_comparison/01_axis_centering_comparison.py --smoke
```

Smoke checks:
- each variant writes `wide_scores.parquet`
- each variant writes `x0_init.npz`
- each variant writes `condensed_positions.npz`
- `rotation.gif` and `init_vs_final_rotation.gif` exist
- `iterations.gif` exists when snapshots are saved

### Full

```bash
conda run -n segmentation_grounded_sam --no-capture-output \
  python results/mcolon/20260410_axis_init_comparison/01_axis_centering_comparison.py
```

Full checks:
- compare-grid figures exist
- standard bundles exist for all variants
- one variant is clearly preferred on trajectory geometry grounds
