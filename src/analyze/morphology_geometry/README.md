# morphology_geometry

`morphology_geometry` is the geometry layer for classifier direction artifacts.
It consumes a validated `ClassifierDirections` artifact and exposes pure numpy +
pandas helpers for alignment and projection math.

## Read first

1. [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) — current design and
   implementation notes.
2. [validation.py](validation.py) — the contract gate.
3. [vectors.py](vectors.py) — cosine / axis alignment and weighted axes.
4. [projection.py](projection.py) — projection of binned features onto an axis.
5. [tests/](tests/) — synthetic fixtures and contract coverage.

## Subpackage map

| Path | Purpose | Status |
|---|---|---|
| [io.py](io.py) | Load and validate saved classifier direction artifacts | Shipped |
| [validation.py](validation.py) | Geometry contract gate and validated artifact view | Shipped |
| [vectors.py](vectors.py) | Axis similarity, matrix extraction, weighted axes | Shipped |
| [projection.py](projection.py) | Project binned features onto a reference axis | Shipped |
| [viz/](viz/) | Reserved for future plotting helpers | Empty for now |

## Current public API

Import from the package root for normal use:

```python
from analyze.morphology_geometry import (
    load_classifier_directions,
    validate_classifier_directions,
    ValidatedDirections,
    ClassifierDirectionContractError,
    cosine_alignment,
    axis_alignment,
    direction_matrix,
    weighted_axis,
    project_binned_features,
)
```

### IO and validation

- `load_classifier_directions(path, *, feature_set, ...)`
- `validate_classifier_directions(directions, *, feature_set, ...)`
- `ValidatedDirections`
- `ClassifierDirectionContractError`

### Vector helpers

- `cosine_alignment(u, v, *, allow_sign_flip=False)`
- `axis_alignment(u, v)`
- `direction_matrix(vd, *, comparison_id=None)`
- `weighted_axis(vd, *, comparison_id, weight_mode="auroc_minus_half")`

### Projection helpers

- `project_binned_features(df, *, vd, axis, id_col, time_col, bin_width, ...)`

## Contract

- The only classification import in this package is in [io.py](io.py).
- Geometry functions accept `ValidatedDirections`, not raw `ClassifierDirections`.
- Feature order comes from `ValidatedDirections.feature_names`.
- `project_binned_features(...)` bins first, then projects.
- `weighted_axis(...)` falls back to uniform weighting when AUROC is unavailable.

## Typical workflow

```python
import analyze.morphology_geometry as mg

vd = mg.load_classifier_directions(
    "results/.../classifier_directions/",
    feature_set="vae",
)

# TODO: flesh out this example with weighting modes and expected outputs.
axis, axis_meta = mg.weighted_axis(vd, comparison_id="pbx1b_vs_wt")
proj = mg.project_binned_features(
    df,
    vd=vd,
    axis=axis,
    id_col="embryo_id",
    time_col="stage_hpf",
    bin_width=4.0,
)
```

## Notes

- Use `load_classifier_directions(...)` when reading saved geometry artifacts from disk.
- Use `validate_classifier_directions(...)` directly only when you already have a
  `ClassifierDirections` object in memory.
- The package is intentionally small: validation, vector math, and projection live
  here; higher-level run orchestration and plotting can be added later without
  changing the core contract.
- `viz/` is currently empty; plot the returned DataFrames with repo-standard
  matplotlib or plotly code for now.
