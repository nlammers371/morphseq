# Data Containers (`BinObject`) — Audit Guide

This directory contains the new typed data-container runtime for MorphSeq bin-based and cross-bin analyses.

- Primary goal: make per-bin and cross-bin feature construction deterministic and auditable.
- Primary default path: VAE embedding columns (`z_mu_b*`) are auto-materialized as per-bin mean features.
- Reference spec: `bin_object_specs.md`.

## Directory Map

- `bin_object.py`
  - `BinObject`: front-door API
  - `LevelCollection`: typed levels storage + `inspect()`
  - Implements `from_raw`, `validate_reducer`, `cross_bin_reduce`, `cross_bin_reduce_batch`
- `specs.py`
  - `InputRef`, `FeatureSpec`, `ReducerSpec`, `LevelName`
- `reducers/`
  - `registry.py`: reducer registry
  - `builtins.py`: built-in reducers (`max`, `top2`, `auc`, etc.)
  - `factories.py`: reducer generators (centering/group-difference)
  - `__init__.py`: package exports + built-in registration
- `reports.py`
  - `SupportReport` audit artifact
- `tests/test_bin_object.py`
  - Focused runtime tests for first-pass behavior
- `bin_object_specs.md`
  - Design/contract specification used for implementation

## Data Levels

`BinObject.levels` contains:

1. `raw` (grain: embryo_id × bin_id after bin labels are attached)
2. `binned` (grain: embryo_id × bin_id; per-bin features)
3. `bin_meta` (grain: embryo_id × bin_id; e.g., `n_frames`, `bin_center_time`)
4. `embryo_meta` (grain: embryo_id)
5. `cross_bin` (grain: embryo_id; outputs from cross-bin reductions)

Use:

```python
bo.levels.inspect()
```

to print grain, row counts, and keys per level.

## API Surface

### 1) Construct from raw table

```python
from analyze.utils.data_containers import BinObject

bo = BinObject.from_raw(
    raw_df,
    embryo_id_col="embryo_id",
    time_col="predicted_stage_hpf",
    bin_width=2.0,
)
```

Behavior:

- Adds bin metadata (`bin_id`, `bin_start`, `bin_end`, `bin_center_time`, `bin_width_seconds`).
- Auto-detects numeric features.
- Auto-detects VAE columns containing `z_mu_b` and materializes deterministic binned keys:
  - `bin__vae__{suffix}__mean` (default and primary path)
- Non-VAE numeric columns are also materialized with `mean` by default.

### 2) Validate reducer feasibility

```python
bo.validate_reducer(
    reducer="max",
    feature_key="bin__vae__0__mean",
    time_window=(30.0, 70.0),
)
```

Checks:

- feature key exists
- bins in scope are computable
- required reducer inputs are present by level
- required bin floor is feasible for selected bins

### 2b) Add calculated features (minimal API)

```python
bo.add_feature(
    level="binned",          # or "cross_bin"
    values=series_or_df,      # Series keyed by grain, or single-column DataFrame
    key="bin__vae__0__mean__minus_ref_wt",
    overwrite=False,
)
```

Rules:

- `values` must match target level grain exactly:
  - `binned`/`raw`/`bin_meta`: (`embryo_id`, `bin_id`)
  - `cross_bin`/`embryo_meta`: (`embryo_id`)
- duplicate grain rows are rejected
- overwrite contract is enforced (see below)

### 3) Cross-bin reduction (single feature)

```python
meta_df, report = bo.cross_bin_reduce(
    features="bin__vae__0__mean",
    reducer="max",
    time_window=(30.0, 70.0),
    bin_fract=0.8,
    min_bins=None,
)
```

Returns:

- `meta_df`: embryo-level table (one row per retained embryo)
- `report`: `SupportReport` for audit

Output key naming:

- `xbin__{feature_without_bin_prefix}__{reducer}__{tmin}_{tmax}`
- Example: `xbin__vae__0__mean__max__30_70`

### 4) Cross-bin reduction (batch/cohort-consistent)

```python
meta_df, report = bo.cross_bin_reduce_batch(
    features=["bin__vae__0__mean", "bin__vae__1__mean"],
    reducer="mean_equal_bin",
    time_window=(30.0, 70.0),
    bin_fract=0.8,
)
```

Batch behavior:

- Computes in-scope bins once
- Computes required-bins once
- Filters embryos once using all requested feature columns
- Emits one shared `SupportReport`

## Deterministic Rules (Audit-Critical)

### Feature overwrite contract

All feature writes follow one explicit contract:

- `overwrite=False` (default): writing an existing key is an error.
- `overwrite=True`: key replacement is allowed only within the same target level.
- Cross-level overwrite is never allowed (a key owned by another level is always an error).

For current APIs, this contract is applied when writing `xbin__...` outputs into `cross_bin`.
When overwrite is allowed, replacement is in-place at the same level grain and does not
silently transfer key ownership across levels.

### Bin selection

In-scope bins are selected by bin-center inclusion in `[t_min, t_max]`.

### Required bins

For a given reducer and window:

`required_bins = max(math_min_bins, ceil(bin_fract * bins_in_scope), min_bins or 0)`

Embryo is retained iff:

`bins_present >= required_bins`

If `required_bins > bins_in_scope`, an explicit error is raised.

### Input routing

Reducers declare consumed inputs via `InputRef(level, key)`.

- `InputRef(level="binned", key="target")` binds to the selected `features` key
- `bin_meta` and `embryo_meta` inputs are level-checked
- `raw` inputs are rejected in current cross-bin reducer execution path

## Built-in Reducers

Defined in `reducers/builtins.py`:

- `mean_equal_bin` (`math_min_bins=1`)
- `max` (`math_min_bins=1`)
- `top2` (`math_min_bins=2`)
- `auc` (`math_min_bins=2`, consumes `bin_meta:bin_center_time`)
- `mean_frame_weighted` (consumes `bin_meta:n_frames`)
- `mean_time_weighted` (consumes `bin_meta:bin_width_seconds`)

## Custom / Auto-Generated Reducers

Reducer factories are provided in `reducers/factories.py` and exported at package level.

### 1) Fixed-baseline centering

```python
from analyze.utils.data_containers import make_centered_reducer

make_centered_reducer(
  name="centered_mean",
  base_reducer="mean_equal_bin",
  baseline_value=0.42,
)
```

### 2) Group-specific centering (e.g., per genotype)

```python
from analyze.utils.data_containers import make_group_centered_reducer

make_group_centered_reducer(
  name="centered_by_genotype",
  group_key="genotype",
  baseline_by_group={"wt": 0.35, "mut": 0.51},
  base_reducer="mean_equal_bin",
)
```

### 3) Auto-generate from current cohort (recommended)

`BinObject` now includes convenience methods to derive baselines from the same
window/dials you are reducing over:

```python
reducer = bo.build_centered_reducer_from_group(
  features="bin__vae__0__mean",
  time_window=(30.0, 70.0),
  group_key="genotype",
  reference_group="wt",
  base_reducer="mean_equal_bin",
)

meta_df, report = bo.cross_bin_reduce(
  features="bin__vae__0__mean",
  reducer=reducer.name,
  time_window=(30.0, 70.0),
)
```

Additional helper methods:

- `bo.build_group_centered_reducer(...)` for per-group centering
- `bo.build_group_difference_reducer(..., reference_group="wt")` for value - WT style differences
- `bo.summarize_cross_bin_by_group(...)` to inspect group baselines before registration

This provides a clean path for “remove WT baseline” / “center against WT” without
hard-coding values by hand.

## `SupportReport` Fields

`reports.py` defines the required audit payload:

- time window and selected bins/centers
- `bins_in_scope`, `required_bins`
- dial values (`bin_fract`, `min_bins`, reducer `math_min_bins`)
- kept/dropped embryo ids and drop reasons
- reducer identity + consumed inputs
- provenance dict
- optional class-imbalance warning (`confounding_warning`)

## Example End-to-End

```python
from analyze.utils.data_containers import BinObject

bo = BinObject.from_raw(raw_df, bin_width=2.0)
bo.levels.inspect()

meta_df, report = bo.cross_bin_reduce(
    features="bin__vae__0__mean",
    reducer="max",
    time_window=(30.0, 70.0),
    bin_fract=0.8,
)

print(meta_df.head())
print(report.as_dict())
```

## Validation / Reproducibility

Run focused tests with the project environment rule:

```bash
conda run -n segmentation_grounded_sam --no-capture-output python -m pytest src/analyze/utils/data_containers/tests/test_bin_object.py -q
```

## Current Status vs Full Spec

Implemented in this first pass:

- typed levels and BinObject front door
- deterministic bin selection and required-bin filtering
- reducer registry + declared `InputRef` consumption
- cross-bin output naming with `xbin__` prefix
- mandatory `SupportReport` return path
- VAE mean-first default materialization path

Not fully expanded yet (future hardening work):

- richer feature-family registration ergonomics beyond current defaults
- full persistence/indexing strategy for multiple `cross_bin` runs in one object
- broader integration into classification pipeline entry points
- expanded package-level test matrix beyond focused unit tests
