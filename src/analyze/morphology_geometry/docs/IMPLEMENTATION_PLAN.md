# Morphology Geometry + Classification `directions/` Refactor — Implementation Plan

## Context

The morphseq classification pipeline (`src/analyze/classification/`) answers **separability**
questions — "can condition A be distinguished from B at time bin t?" — with AUROC and
permutation p-values. A new line of analysis is needed to answer **geometric** questions in
the shared morphology feature space:

- Are phenotype directions *aligned* across conditions? (cosine alignment over time)
- Is a double perturbation *more of the same* along one axis? (projection additivity)
- Where do embryos sit along a reference phenotype axis, and how does that evolve in time?

A working prototype already exists at
`results/mcolon/20260409_pbx_additivity/phenotype_direction/` (`io.py`, `vectors.py`,
`projection.py`) plus the scripts `01_pbx_direction_smoke.py`,
`02_projection_feature_over_time.py`, `03_direction_cosine_heatmaps.py`. Those scripts
validate the math and the artifact contract end-to-end.

Rather than just promoting the prototype, this plan does two things **together**:

1. **Refactor classification so "direction vectors" is a dedicated, discoverable
   subsystem** — `src/analyze/classification/directions/` — with one source of truth for
   fitting, assembling, and persisting `ClassifierDirections` artifacts. Today those
   concerns are tangled inside a 1100-line `engine/loop.py` alongside CV, permutation,
   AUROC, confusion, and contrast coordinates. After the refactor, `loop.py` *calls* the
   directions subsystem, it does not define it.

2. **Build `morphology_geometry/` as a pure numpy + pandas subsystem** that consumes a
   single, loudly validated `ClassifierDirections` artifact and produces geometry
   products (aligned axes, cosine matrices, projection scores, per-embryo summaries).

The core principle for the refactor: **scope is visible from the path**. If a file name
does not communicate what it contains, split it. Direction logic lives under
`classification/directions/` and nowhere else; geometry math lives under
`morphology_geometry/` and imports only the artifact contract.

Outcomes:

- Classification owns direction production + artifact persistence.
- Geometry owns downstream math on already-fit vectors.
- Downstream subsystems (morphology_geometry, trajectory_condensation, future interactive
  tools) rely on one stable contract (`ClassifierDirections`) without importing
  classification internals.
- If direction math changes, it changes in one place; any drift is caught by a single
  validator at artifact load time.

## Refactor A — `classification/directions/`

### Goals

- One source of truth for direction production: fit, payload assembly, persistence, ids,
  and the lightweight extractor entry point all live under
  `src/analyze/classification/directions/`.
- `engine/loop.py` shrinks: its direction-specific helpers move to `directions/` and
  `loop.py` imports them.
- A lightweight `extract_classifier_directions(...)` entry point produces the same
  on-disk artifact as `run_classification(..., save_classifier_directions=True)` without
  the AUROC / permutation machinery. Users who only want geometry can generate directions
  fast.

### File layout (exactly 6 files — do not split further in the first pass)

```
src/analyze/classification/directions/
├── __init__.py          # re-exports: ClassifierDirections, fit_classifier_direction,
│                        # build_classifier_directions_payload,
│                        # extract_classifier_directions, vector_id helpers
├── artifact.py          # ClassifierDirections dataclass + save/load (MOVED from
│                        # engine/analysis.py; engine/analysis.py keeps a thin
│                        # deprecated re-export so existing imports continue to work)
├── fit.py               # fit_classifier_direction(X, y, feature_cols, ...) + its
│                        # private helpers: _make_logistic_classifier,
│                        # _estimator_config, _preprocess_fingerprint, _json_safe.
│                        # ONE place where sklearn is called for direction production.
├── build_payload.py     # build_classifier_directions_payload(fits, *, feature_sets)
│                        # Pure assembly of a list of per-(feature_set, comparison,
│                        # time_bin) fit dicts into a ClassifierDirections dataclass.
│                        # No sklearn, no fitting.
├── extract.py           # extract_classifier_directions(df, *, class_col, id_col,
│                        # time_col, comparisons, features, bin_width, ...) — the
│                        # lightweight runner. Composes classification-general helpers
│                        # (_resolve_feature_columns, _build_binary_labels,
│                        # _bin_and_aggregate, resolve_comparisons) with
│                        # fit_classifier_direction + build_classifier_directions_payload.
└── ids.py               # vector_id format, construction, parsing. Even if small, it
                         # lives alone so geometry code can inspect it without digging
                         # into build_payload.py.
```

**Not created (deliberate — avoids bloat):**

- No `schema.py` separate from `artifact.py`. The dataclass and its save/load are the
  schema. If/when a stricter schema description is needed, add it as a module docstring
  in `artifact.py` first; only split if it actually grows.
- No `io.py` separate from `artifact.py`. `save`/`load` are methods on the dataclass.
- No `utils.py`. Ever.
- No `validation.py` on the classification side. Producer-side sanity (no NaN vectors,
  unit-norm before save, fingerprint present) lives as assertions inside `build_payload.py`.
  Consumer-side contract enforcement is geometry's job (see Refactor B).

### What moves, from where, to where

Current location → new location (all file:line references from earlier exploration):

- `src/analyze/classification/engine/analysis.py:17` `ClassifierDirections` dataclass →
  **`classification/directions/artifact.py`**. `engine/analysis.py` keeps a
  one-line re-export: `from analyze.classification.directions.artifact import
  ClassifierDirections  # re-exported; new code should import from directions.artifact`.
  A `DeprecationWarning` at module import time is overkill here — a comment is enough,
  because the symbol is still reachable from the old path.

- `src/analyze/classification/engine/loop.py:99` `_fit_classifier_direction` →
  **`classification/directions/fit.py::fit_classifier_direction`** (public name). The
  body is unchanged — same sklearn call, same sign-correction logic, same unit
  normalization, same preprocess fingerprint, same return-dict shape. `loop.py` calls
  `fit_classifier_direction` directly.

- `src/analyze/classification/engine/loop.py:37` `_make_logistic_classifier` and the
  surrounding helpers `_estimator_config`, `_preprocess_fingerprint`, `_json_safe`
  (currently around loop.py:80-96) → **`classification/directions/fit.py`** as private
  module-level helpers. They are used nowhere else in the classification package.
  (Quick check before the move: `grep -n _make_logistic_classifier
  src/analyze/classification/`. If anything outside `loop.py` imports it, that call site
  switches to `fit_classifier_direction` or gets its own factory. The earlier
  exploration found only the direction fit path using it.)

- `src/analyze/classification/engine/loop.py` `_collect_classifier_directions` (around
  loop.py:834) → body extracted into
  **`classification/directions/build_payload.py::build_classifier_directions_payload`**.
  The `_collect_classifier_directions` name stays in `loop.py` as a thin wrapper that
  calls the new public helper, so existing call sites in the loop are untouched. (If
  nothing else in `loop.py` calls `_collect_classifier_directions` by name, the wrapper
  can be inlined at the single call site; confirm during the refactor.)

- vector_id construction logic (whatever format is currently used inside
  `_collect_classifier_directions`) → **`classification/directions/ids.py`** with a
  public constructor and parser, e.g.:
  ```python
  def make_vector_id(*, feature_set: str, comparison_id: str, time_bin_center: float) -> str: ...
  def parse_vector_id(vector_id: str) -> dict: ...
  ```
  `build_classifier_directions_payload` calls `make_vector_id`. Geometry's validator can
  call `parse_vector_id` if it ever needs to cross-check ids against metadata rows.

**Classification-general helpers that stay put:**

- `_resolve_feature_columns` (loop.py:156) — not direction-specific; stays in `loop.py`.
  Both `run_classification` and `extract_classifier_directions` import it from its
  current location.
- `_build_binary_labels` (loop.py:194) — same: classification-general, stays.
- `_bin_and_aggregate` (loop.py:222) — same: stays.
- `comparison_resolution.resolve_comparisons` — stays in
  `engine/comparison_resolution.py`; `extract.py` imports it.

The refactor deliberately does **not** move these. They belong to classification's
general data-assembly path, not to "directions" specifically. Moving them would entangle
the directions package with responsibilities that don't belong to it.

### Before vs. after for `loop.py`

Before: `_fit_classifier_direction`, `_make_logistic_classifier`, `_estimator_config`,
`_preprocess_fingerprint`, `_json_safe`, and `_collect_classifier_directions` are all
defined inside `loop.py`, contributing to its ~1100-line size.

After: those definitions are gone. `loop.py` has two new imports at the top:

```python
from analyze.classification.directions.fit import fit_classifier_direction
from analyze.classification.directions.build_payload import build_classifier_directions_payload
```

and its direction-related call sites become one-liners. `loop.py` is smaller and its
remaining responsibilities (CV, permutation nulls, AUROC, confusion, contrast
coordinates) are more clearly what it owns.

### Lightweight runner: `extract_classifier_directions`

Signature (lives in `classification/directions/extract.py`):

```python
def extract_classifier_directions(
    df: pd.DataFrame,
    *,
    class_col: str,
    id_col: str,
    time_col: str,
    comparisons: ComparisonScheme,
    features: dict[str, str | list[str]],
    bin_width: float = 4.0,
    min_samples_per_group: int = 3,
    min_samples_per_member: int = 2,
    random_state: int = 42,
    class_weight: Any | None = "balanced",
    verbose: bool = False,
    save_dir: str | Path | None = None,
    overwrite: bool = False,
) -> ClassifierDirections:
    """Fit per-(feature_set, comparison, time_bin) binary logistic regressions and return
    the unit direction vectors — no CV, no permutation test, no AUROC, no confusion.

    Produces an artifact with the same schema as
    run_classification(..., save_classifier_directions=True), except auroc_obs is NaN.
    """
```

Body (pseudocode):

```python
resolved = resolve_comparisons(comparisons, class_col=class_col, df=df)
feature_sets = _resolve_feature_columns(df, features)   # dict[str, list[str]]
fits = []
for feature_set, feature_cols in feature_sets.items():
    for comparison in resolved:
        labeled = _build_binary_labels(df, class_col, comparison)
        if labeled[class_col].nunique() < 2:  # group size check via min_samples_per_group
            continue
        binned = _bin_and_aggregate(labeled, id_col, time_col, feature_cols, bin_width)
        for t, sub in binned.groupby("time_bin_center"):
            X = sub[feature_cols].to_numpy(dtype=float)
            y = sub["_y"].to_numpy(dtype=int)
            if (y == 1).sum() < min_samples_per_member or (y == 0).sum() < min_samples_per_member:
                continue
            fit = fit_classifier_direction(
                X=X, y_binary=y, feature_cols=feature_cols,
                random_state=random_state, class_weight=class_weight,
            )
            if fit is None:
                continue
            fit.update({
                "feature_set": feature_set,
                "comparison_id": comparison.comparison_id,
                "positive_label": comparison.positive_label,
                "negative_label": comparison.negative_label,
                "time_bin_center": float(t),
                "n_pos": int((y == 1).sum()),
                "n_neg": int((y == 0).sum()),
            })
            fits.append(fit)

directions = build_classifier_directions_payload(fits, feature_sets=feature_sets)
if save_dir is not None:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=overwrite)
    directions.metadata.to_parquet(save_path / "classifier_directions.parquet", index=False)
    directions.save(save_path / "classifier_directions_vectors.npz")
return directions
```

`run_classification(..., save_classifier_directions=True)` and
`extract_classifier_directions(...)` must produce byte-identical unit vectors on the same
data (same `random_state`, same estimator config). This is guaranteed by both paths
routing through the same `fit_classifier_direction` and the same
`build_classifier_directions_payload`.

### Naming check (applied to this layout)

- Can a new lab member guess what a file contains from its name? **Yes.** `artifact.py`,
  `fit.py`, `build_payload.py`, `extract.py`, `ids.py` — each name says "what kind of
  thing is here."
- Does each file contain one "kind" of thing? **Yes.** artifact (nouns: persistable
  dataclass), fit (verb: compute one direction), build (verb: assemble payload),
  extract (entry point), ids (naming convention).
- Is direction logic duplicated anywhere else? **After the refactor, no.** `loop.py`
  imports from `directions/`; the prototype's `phenotype_direction/` re-exports from
  `morphology_geometry/`; `morphology_geometry/` imports `ClassifierDirections` only.
- Does `engine/loop.py` only orchestrate? **Yes** — after the move, direction math is not
  defined there.

## Refactor B — `morphology_geometry/`

### Goals

- Pure numpy + pandas. No sklearn. No scipy beyond tiny helpers.
- One and only one classification import, confined to `io.py`:
  `from analyze.classification.directions.artifact import ClassifierDirections`.
- A single, loudly-failing validator sits at the front door. Every geometry function
  takes `ValidatedDirections` (the validator's return type), not raw
  `ClassifierDirections` — so it is impossible to run geometry math against an unchecked
  artifact.

### File layout

```
src/analyze/morphology_geometry/
├── __init__.py                   # public API re-exports
├── io.py                         # load_classifier_directions(path, ...) -> ValidatedDirections
│                                 # Only file in the package that imports anything from
│                                 # analyze.classification.
├── validation.py                 # validate_classifier_directions, ValidatedDirections,
│                                 # ClassifierDirectionContractError
├── vectors.py                    # cosine_alignment, axis_alignment, direction_matrix,
│                                 # weighted_axis, cosine_alignment_matrix
├── projection.py                 # project_binned_features
├── normalize.py                  # zscore_scores_within_bin, embryo_mean_zscore
├── results.py                    # GeometryAnalysis dataclass + save/load
├── run.py                        # run_morphology_geometry()
├── viz/
│   ├── __init__.py
│   ├── cosine_over_time.py
│   ├── cosine_heatmap.py
│   └── projection_traces.py
└── tests/
    ├── __init__.py
    ├── conftest.py               # synthetic ClassifierDirections fixtures
    ├── test_validation.py
    ├── test_vectors.py
    ├── test_projection.py
    ├── test_normalize.py
    ├── test_results.py
    ├── test_run.py
    └── test_no_classification_internal_imports.py
```

### The validator (central piece)

`morphology_geometry/validation.py`:

```python
class ClassifierDirectionContractError(ValueError):
    """Raised when a ClassifierDirections artifact does not meet the geometry contract.
    Message always names the offending field and expected vs observed value.
    """

@dataclass(frozen=True)
class ValidatedDirections:
    metadata: pd.DataFrame           # filtered to feature_set, sorted by
                                     # (comparison_id, time_bin_center)
    vectors: np.ndarray              # (n_rows, n_features), rows aligned with metadata
    feature_names: list[str]         # authoritative column order
    feature_set: str
    inferred_bin_width: float
    bin_centers: np.ndarray          # sorted unique time_bin_center
    preprocess_fingerprint: str      # constant across rows (one per feature_set)
    has_auroc: bool                  # whether auroc_obs is populated (False for extract)

def validate_classifier_directions(
    directions: ClassifierDirections,
    *,
    feature_set: str,
    required_comparison_ids: Sequence[str] | None = None,
    expected_bin_width: float | None = None,
    unit_norm_tol: float = 1e-6,
    check_preprocess_fingerprint: bool = True,
) -> ValidatedDirections: ...
```

### What the validator checks (exhaustive, all loud)

Required metadata columns (parquet):

- `vector_id` (str, unique)
- `feature_set` (str)
- `comparison_id` (str)
- `positive_label`, `negative_label` (str)
- `time_bin_center` (float)
- `n_pos`, `n_neg` (int)
- `coef_norm` (float)
- `intercept` (float)
- `sign_flipped` (bool)
- `centroid_dot` (float)
- `direction_space` (str; must equal `"raw"` for the current contract)
- `preprocess_fingerprint` (str)
- `auroc_obs` (float, **optional**; present when produced by `run_classification`, absent
  when produced by `extract_classifier_directions`)

Required NPZ keys:

- One key per `vector_id` → `np.ndarray` of shape `(n_features,)`, float64
- `feature_names__<feature_set>` → str array of length `n_features`

Failure modes (all raise `ClassifierDirectionContractError` with a specific message):

- Missing metadata column, missing npz key, `vector_id` in metadata not in npz (or vice
  versa)
- `feature_names[feature_set]` missing or empty
- Vector length != `len(feature_names[feature_set])`
- Vector contains NaN / Inf
- `|‖v‖ - 1| > unit_norm_tol`. Zero-vectors are allowed **only** if the corresponding
  metadata row has `coef_norm == 0`, in which case a warning is logged (not an error)
  and that row is dropped from the returned `vectors` slice
- `required_comparison_ids` not all present in metadata (after filtering by feature_set)
- Bin widths: compute `diff(sorted(unique(time_bin_center)))`; if not uniform to
  floating-point tolerance, raise. If `expected_bin_width` is supplied and disagrees with
  the inferred width, raise with both values in the message
- `preprocess_fingerprint` drift: if `check_preprocess_fingerprint=True`, assert the
  fingerprint is constant across all rows in the filtered metadata

### Public API of `morphology_geometry`

```python
from analyze.morphology_geometry import (
    # IO + validation
    load_classifier_directions,      # (path, *, feature_set, ...) -> ValidatedDirections
    validate_classifier_directions,  # (ClassifierDirections, ...) -> ValidatedDirections
    ValidatedDirections,
    ClassifierDirectionContractError,
    # vectors
    cosine_alignment, axis_alignment, direction_matrix, weighted_axis,
    cosine_alignment_matrix,
    # projection
    project_binned_features,
    # normalization / summaries
    zscore_scores_within_bin, embryo_mean_zscore,
    # result object
    GeometryAnalysis,
    # top-level entry point
    run_morphology_geometry,
)
```

The lightweight extractor is imported from its own namespace, **not** re-exported from
geometry:

```python
from analyze.classification.directions import extract_classifier_directions
```

### `run_morphology_geometry()` signature

```python
def run_morphology_geometry(
    df: pd.DataFrame,
    *,
    classification_save_dir: str | Path,
    feature_set: str,
    class_col: str,
    id_col: str,
    time_col: str,
    bin_width: float,                      # validated against the artifact bin centers
    axis_comparison_id: str,
    projection_comparison_ids: Sequence[str] | None = None,
    reference_group: str,
    weight_mode: str = "auroc_minus_half", # falls back to "uniform" if has_auroc=False
    save_dir: str | Path | None = None,
    overwrite: bool = False,
) -> GeometryAnalysis: ...
```

First two lines of the body are always load + validate:

```python
raw = ClassifierDirections.load(  # imported only in io.py; run.py calls load_classifier_directions
    Path(classification_save_dir) / "classifier_directions.parquet",
    Path(classification_save_dir) / "classifier_directions_vectors.npz",
)
vd = validate_classifier_directions(
    raw,
    feature_set=feature_set,
    required_comparison_ids=[axis_comparison_id, *(projection_comparison_ids or [])],
    expected_bin_width=bin_width,
)
```

All downstream code in `run.py` operates on `vd` only.

### `GeometryAnalysis` result object

```python
@dataclass
class GeometryAnalysis:
    reference_axis: np.ndarray              # (n_features,) unit vector
    reference_axis_meta: pd.DataFrame       # per-bin directions + axis_weight used
    cosine_matrix: pd.DataFrame | None      # (comp_a, comp_b, time_bin_center, cosine)
    projections: pd.DataFrame               # id, time_bin_center, class, score, zscore
    embryo_summary: pd.DataFrame            # id, class, mean_zscore, n_bins
    uns: dict                               # schema_version, created_at, git_commit,
                                            # classification_save_dir, feature_set,
                                            # axis_comparison_id, weight_mode, bin_width,
                                            # reference_group, preprocess_fingerprint,
                                            # inferred_bin_width, has_auroc

    def save(self, path, overwrite=False) -> Path: ...
    @classmethod
    def load(cls, path) -> "GeometryAnalysis": ...
```

Persistence: `reference_axis.npz`, `reference_axis_meta.parquet`, `cosine_matrix.parquet`,
`projections.parquet`, `embryo_summary.parquet`, `metadata.json`. Same house style as
`ClassificationAnalysis.save()`.

## Critical files to read / touch

**Read (do not modify):**

- `src/analyze/classification/engine/analysis.py:17` — `ClassifierDirections` dataclass
  (the definition that moves to `directions/artifact.py`).
- `src/analyze/classification/engine/loop.py:37,80-96,99,222,834` — helpers to move or
  reference during the refactor.
- `src/analyze/trajectory_condensation/schema.py:25` — `CondensationData`, future handoff
  target. No edits in this plan.

**Modify (surgical):**

- `src/analyze/classification/engine/analysis.py` — remove the body of
  `ClassifierDirections`; replace with a one-line re-export from
  `analyze.classification.directions.artifact`.
- `src/analyze/classification/engine/loop.py` — delete the direction-specific helpers
  (`_fit_classifier_direction`, `_make_logistic_classifier`, `_estimator_config`,
  `_preprocess_fingerprint`, `_json_safe`, body of `_collect_classifier_directions`).
  Replace with imports from `directions.fit` and `directions.build_payload`. Behavior
  must be preserved — existing classification tests pass without modification.

**Create under `src/analyze/classification/directions/`:**

- `__init__.py`, `artifact.py`, `fit.py`, `build_payload.py`, `extract.py`, `ids.py`.

**Create under `src/analyze/morphology_geometry/`:**

- Full module layout above.

**Promote from the prototype:**

- `results/mcolon/20260409_pbx_additivity/phenotype_direction/vectors.py`
- `results/mcolon/20260409_pbx_additivity/phenotype_direction/projection.py`
- `results/mcolon/20260409_pbx_additivity/phenotype_direction/io.py`

into `morphology_geometry/vectors.py`, `projection.py`, `io.py` — with the key change
that functions now accept `ValidatedDirections`, not raw `ClassifierDirections`.

**Thin shim to leave in place:**

- `results/mcolon/20260409_pbx_additivity/phenotype_direction/__init__.py` — re-exports
  from `analyze.morphology_geometry` with a `DeprecationWarning`. Existing scripts
  (`01_*`, `02_*`, `03_*`) continue to run unchanged.

## Implementation order

**Step 1 — Refactor A, part 1: create `classification/directions/` and move direction
code into it, preserving behavior.** ✅ DONE
- Created `classification/directions/` with 6 files: `__init__.py`, `artifact.py`,
  `fit.py`, `build_payload.py`, `ids.py`, `extract.py`.
- `ClassifierDirections` moved to `directions/artifact.py`; `engine/analysis.py`
  re-exports it with a comment.
- `_fit_classifier_direction` + private helpers (`_make_logistic_classifier`,
  `_estimator_config`, `_preprocess_fingerprint`, `_json_safe`) moved to `fit.py`
  and renamed to `fit_classifier_direction` (public).
- `build_classifier_directions_payload` created in `build_payload.py`.
- `make_vector_id` / `parse_vector_id` in `ids.py`.
- Shared data-prep helpers (`_resolve_feature_columns`, `_build_binary_labels`,
  `_bin_and_aggregate`) extracted from `loop.py` into `engine/data_prep.py` — both
  `loop.py` and `directions/extract.py` import from there; neither imports the other.
- `loop.py` imports direction symbols from `directions/`; its definitions are gone.
- **All 166 classification tests pass.**

**Step 2 — Refactor A, part 2: `extract_classifier_directions`.** ✅ DONE
- Implemented in `directions/extract.py`.
- Verified byte-identical vectors vs `run_classification(..., n_permutations=0,
  save_classifier_directions=True)` on a synthetic fixture (same `vector_id` set,
  same `unit_coef`, same `preprocess_fingerprint`, same `feature_names`).

**Step 3 — Refactor B, part 1: `morphology_geometry/validation.py`.** ✅ DONE
- `validation.py` written: `ClassifierDirectionContractError`, `ValidatedDirections`
  (frozen dataclass), `validate_classifier_directions()` with all 10 checks:
  feature_set present, required metadata columns, filter + sort, vector_id integrity,
  vector stacking with shape/finite/unit-norm checks, direction_space, preprocess
  fingerprint constancy, required_comparison_ids, bin uniformity + expected_bin_width,
  has_auroc detection.
- `tests/test_validation.py` written: 44 tests covering all 10 checks, all error paths,
  happy-path field values, and zero-norm warn-and-drop behavior. All 44 pass.

**Step 4 — Refactor B, part 2: promote vectors + projection + io.** ✅ DONE
- `io.py`: `load_classifier_directions(path, *, feature_set, ...)` — only file importing
  from `analyze.classification`; loads raw artifact, runs validator, returns
  `ValidatedDirections`.
- `vectors.py`: `cosine_alignment`, `axis_alignment`, `direction_matrix`,
  `weighted_axis` — all accept `ValidatedDirections`. `weighted_axis` falls back to
  "uniform" when `has_auroc=False`.
- `projection.py`: `project_binned_features` — takes `vd: ValidatedDirections` for
  authoritative feature column order; `feature_set` parameter removed.
- `__init__.py`: re-exports full public API.
- `tests/conftest.py`: shared `minimal_directions` + `validated` fixtures.
- `tests/test_io.py`: 7 tests (round-trip, missing files, error forwarding).
- `tests/test_vectors.py`: 21 tests (cosine, axis, direction_matrix, weighted_axis).
- `tests/test_projection.py`: 9 tests including column-order invariance, bin-center
  arithmetic, projection-math exact check.
- `tests/test_no_classification_internal_imports.py`: AST guard — only io.py and
  validation.py may import from analyze.classification, only the artifact type.
- `phenotype_direction/__init__.py` replaced with shim: re-exports pure math helpers
  from `analyze.morphology_geometry`; `load_classifier_directions` returns raw
  `ClassifierDirections` with DeprecationWarning (backward compat for scripts 01/02/03).
- **All 254 tests pass (88 morphology_geometry + 166 classification).**

**Step 5 — `cosine_alignment_matrix()` in `vectors.py`.**
Takes a `ValidatedDirections` + list of comparison IDs, returns a long-format df with
one row per `(comp_a, comp_b, time_bin_center)`. Tests: symmetry, diagonal == 1, trivial
aligned / anti-aligned / orthogonal synthetic cases.

**Step 6 — `normalize.py`.**
Two pure-pandas helpers: `zscore_scores_within_bin(scores_df, *, reference_group, ...)`
and `embryo_mean_zscore(zscored_df, *, id_col, ...)`. Unit tests against
hand-constructed DataFrames.

**Step 7 — `GeometryAnalysis` in `results.py`.**
Dataclass + `save`/`load`. Round-trip test.

**Step 8 — `run_morphology_geometry()` in `run.py`.**
Pure composition of steps 3–7. End-to-end test:
- Build a tiny in-memory df (≥2 classes, ≥2 embryos per class, ≥2 time bins, ≥4 features)
- Path A: `extract_classifier_directions(df, ..., save_dir=tmpA)` →
  `run_morphology_geometry(df, classification_save_dir=tmpA, ...)`
- Path B: `run_classification(df, ..., save_classifier_directions=True,
  n_permutations=0, save_dir=tmpB)` → `run_morphology_geometry(df,
  classification_save_dir=tmpB, ...)`
- Assert both paths produce identical `GeometryAnalysis` (`reference_axis`,
  `projections`, `embryo_summary`). Both entry points must be interchangeable from
  geometry's perspective.

**Step 9 — `viz/` plotters.**
Port the three plotting patterns from the PBX scripts (`plot_cosine_over_time`,
`plot_cosine_heatmap`, `plot_projection_traces`). Use `analyze.viz.contract` wrappers to
stay consistent with the project-wide matplotlib contract.

**Step 10 — Update the 20260409 PBX scripts.**
Scripts 01/02/03 import from `analyze.morphology_geometry` directly (or continue
through the shim — both work). Diff produced figures against the committed versions
under `results/mcolon/20260409_pbx_additivity/figures/`. Any change is a regression.

**Deferred (explicitly not in the first pass):**

- Trajectory-condensation handoff. A later `to_condensation_data(geometry, ...)` adapter
  lives in its own file so `morphology_geometry/` has no dependency on
  `trajectory_condensation/schema.py` in the first pass.
- Multiclass classifier direction extraction.
- Any geometry operation that requires something the current `ClassifierDirections`
  schema does not persist. The fix in that case is: extend the artifact schema in
  `directions/artifact.py`, update the producer in `directions/build_payload.py`, update
  the validator in `morphology_geometry/validation.py`. Never import classification
  internals from geometry.

## Architectural risks & mitigations

1. **Bin-width / time-coordinate mismatch.** *Mitigation:* the validator infers bin width
   from `diff(sorted(unique(time_bin_center)))`, requires uniform spacing, and — if the
   caller supplies `expected_bin_width` — fails loudly on mismatch.
   `run_morphology_geometry` always supplies it.

2. **Feature-column ordering drift.** *Mitigation:* `ValidatedDirections.feature_names`
   is authoritative; `project_binned_features` indexes `df` exclusively by that list.
   Test: shuffle `df` column order and confirm projection is unchanged.

3. **Preprocess fingerprint drift within a feature_set.** Combining two comparisons that
   used different estimator configs or different feature orders is meaningless.
   *Mitigation:* validator enforces a single `preprocess_fingerprint` per feature_set and
   propagates it into `GeometryAnalysis.uns`.

4. **Classification refactor drift.** *Mitigation:* the validator is the single canary.
   `test_validation.py` asserts the required columns / keys explicitly. The Step 8
   end-to-end test runs both entry points against the same fixture so their contracts
   stay in sync numerically.

5. **Sign-correction consistency between the two entry points.** *Mitigation:* both
   `run_classification` and `extract_classifier_directions` route through the same
   `fit_classifier_direction` in `directions/fit.py` and the same
   `build_classifier_directions_payload`. Step 2's byte-identity test enforces this.

6. **Duplicated direction logic creeping back in.** *Mitigation:* an AST-based
   `test_no_classification_internal_imports` test greps every file under
   `src/analyze/morphology_geometry/` and asserts the only `analyze.classification` import
   anywhere is `from analyze.classification.directions.artifact import ClassifierDirections`
   (and only in `io.py`). A second test ensures direction-math symbol names
   (`fit_classifier_direction`, `build_classifier_directions_payload`) are defined
   exactly once across `src/analyze/`.

7. **`directions/` becoming a dumping ground.** *Mitigation:* the 6-file layout is a
   ceiling, not a starting point. Adding a 7th file requires naming it by kind (no
   `utils.py`, no `helpers.py`) and articulating which of the existing 6 files is failing
   the "one kind per file" test. If that articulation is hard, the new code probably
   belongs in one of the existing files.

8. **`loop.py` post-refactor gaining new direction code.** *Mitigation:* the Step 1 test
   run must show the only direction-related symbols remaining in `loop.py` are imports
   from `directions.*`. A lightweight grep test in
   `src/analyze/classification/tests/` asserts `_fit_classifier_direction` no longer
   exists as a definition anywhere in `engine/` after the refactor.

9. **Premature trajectory-condensation coupling.** *Mitigation:* zero imports of
   `trajectory_condensation/schema.py` from `morphology_geometry/` in this plan.

## Verification

**Commands (per CLAUDE.md):**

```bash
conda run -n segmentation_grounded_sam --no-capture-output \
  python -m pytest src/analyze/classification/tests/ -v
conda run -n segmentation_grounded_sam --no-capture-output \
  python -m pytest src/analyze/morphology_geometry/tests/ -v

conda run -n segmentation_grounded_sam --no-capture-output \
  python results/mcolon/20260409_pbx_additivity/01_pbx_direction_smoke.py
conda run -n segmentation_grounded_sam --no-capture-output \
  python results/mcolon/20260409_pbx_additivity/02_projection_feature_over_time.py
conda run -n segmentation_grounded_sam --no-capture-output \
  python results/mcolon/20260409_pbx_additivity/03_direction_cosine_heatmaps.py
```

**Test matrix:**

- `classification/tests/` — existing tests pass unchanged after Step 1 (behavior
  preservation). New `test_extract_directions.py` asserts byte-identity between the two
  entry points (Step 2).
- `morphology_geometry/tests/test_validation.py` — one failure case per validator rule.
- `morphology_geometry/tests/test_vectors.py` — cosine / axis / weighted_axis /
  cosine_alignment_matrix on synthetic fixtures.
- `morphology_geometry/tests/test_projection.py` — known features × axis → known scores;
  column-order invariance.
- `morphology_geometry/tests/test_normalize.py` — within-bin z-score + embryo mean.
- `morphology_geometry/tests/test_results.py` — `GeometryAnalysis` save/load round-trip.
- `morphology_geometry/tests/test_run.py` — end-to-end via both entry points; both must
  produce identical `GeometryAnalysis`.
- `morphology_geometry/tests/test_no_classification_internal_imports.py` — static check
  that no file imports from `analyze.classification.*` except `io.py`, which may only
  import `ClassifierDirections` from `analyze.classification.directions.artifact`.

**PBX regression:** re-run `01_*`, `02_*`, `03_*` and diff produced figures against the
committed versions under `results/mcolon/20260409_pbx_additivity/figures/`.

## Summary of answers to the deliverables

1. **Separate subsystem?** Yes — and symmetrically. Classification gets a new
   `directions/` subsystem that owns direction production (fit, assemble, persist, ids,
   lightweight runner). Geometry gets `morphology_geometry/` — pure numpy + pandas —
   that owns downstream math. `loop.py` orchestrates; it no longer defines direction
   math.

2. **Reusable from classification into geometry:** exactly one symbol —
   `ClassifierDirections` — imported only in `morphology_geometry/io.py`.
   Classification-general helpers (`_resolve_feature_columns`, `_build_binary_labels`,
   `_bin_and_aggregate`, `resolve_comparisons`) are reused *inside* classification (by
   `directions/extract.py`), never by geometry.

3. **Reimplement / wrap:** central `validate_classifier_directions()` on the geometry
   side that returns `ValidatedDirections`; every geometry function operates on that
   wrapper, never on raw `ClassifierDirections`. Prototype vectors/projection/io are
   promoted and rewired to the validated type.

4. **Module layout:** `classification/directions/` (6 files) + `morphology_geometry/`
   (full layout above).

5. **Public API:** geometry re-exports validation + vectors + projection + normalize +
   results + `run_morphology_geometry`. `extract_classifier_directions` is imported
   separately from `analyze.classification.directions`.

6. **Implementation order:** 10 steps. The classification refactor (Steps 1–2) comes
   first because the validator in Step 3 is easier to write against the clean
   `directions/artifact.py` contract. Geometry sits on top.

7. **Risks:** bin width mismatch, feature order drift, preprocess fingerprint drift,
   classification refactor drift, sign-correction consistency, duplicated direction
   logic creeping back in, `directions/` bloat, `loop.py` regrowing direction code,
   premature condensation coupling — each with a concrete, testable mitigation.
