---
name: analyze-morphology-geometry
description: morphseq morphology geometry — classifier direction validation, axis alignment, weighted axes, and projection onto validated phenotype directions
---

You are a morphseq morphology geometry expert. When the user asks about classifier direction artifacts, validated geometry contracts, phenotype-axis alignment, or projections onto classifier directions, use `src/analyze/morphology_geometry/`.

## Canonical references

Read these first:

- `src/analyze/morphology_geometry/README.md` — user-facing overview and current API.
- `src/analyze/morphology_geometry/docs/IMPLEMENTATION_PLAN.md` — design and implementation notes.
- `src/analyze/morphology_geometry/validation.py` — the contract gate.
- `src/analyze/morphology_geometry/vectors.py` — cosine / axis alignment and weighted axes.
- `src/analyze/morphology_geometry/projection.py` — projection onto a validated axis.
- `src/analyze/morphology_geometry/tests/` — synthetic fixtures and contract coverage.

## When to use this module

- Validating a saved `ClassifierDirections` artifact before geometry math
- Comparing phenotype directions with cosine or sign-free axis alignment
- Building a weighted reference axis for a comparison
- Projecting binned embryo features onto a validated direction vector
- Checking feature-order stability and bin-width consistency

## Guardrails

- Treat `ValidatedDirections` as the only supported input for downstream geometry.
- Do not import classification internals anywhere except `io.py`.
- Use `load_classifier_directions(...)` for on-disk artifacts and
  `validate_classifier_directions(...)` only when a raw artifact is already in memory.
- Use `axis_alignment(...)` when sign should not matter.
- Use `weighted_axis(...)` for a reference phenotype axis; it falls back to uniform
  weights when AUROC is unavailable.
- Use `project_binned_features(...)` when you want per-embryo projections over time.

## Setup

```python
import matplotlib
matplotlib.use("Agg")

import analyze.morphology_geometry as mg
```

## Common pattern

```python
vd = mg.load_classifier_directions(
    "results/.../classifier_directions/",
    feature_set="vae",
)

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

## What this package is not

- It is not the classifier fitting layer.
- It is not the trajectory condensation solver.
- It is not a visualization package.

