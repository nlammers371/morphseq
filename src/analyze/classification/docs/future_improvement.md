
# Future improvements 
There might sometimes be cases where we don't want to, for example in label transfer, where we want to train on all images. I don't know, I need to think about this: train on all images, but do the image-level group-by images, but then do the k-fold group-level classification. Also, I have to make my bootstrapping be embryo ID-aware to get statistical significance. 





# Future improvements — classification (AUROC over time)

> **Status: historical snapshot (2026-03-05).** Kept for roadmap context.
> Much of the P0 work has since landed in `run_classification` / `ClassificationAnalysis`.
> The "Current state" section below describes the API *as of March 2026* and is now out of date
> — see [`README.md`](../README.md) for the current canonical reference.

## Active: contrast-coordinates shrinkage rewrite

The contrast-coordinates layer group (`save_contrast_coordinates=True` → 7
layers: `raw_contrast_scores_long`, `contrast_support_long`,
`contrast_specificity_by_timebin`, `raw_coordinates`, `shrunk_coordinates`,
`residual_coordinates`, `probe_index`) is **frozen** pending a rewrite of the
shrinkage formula. The current probe-weight shrinkage
(`clip((auroc_obs - null_mean) / 0.5, 0, 1)`) is not numerically well-justified
and the residual decomposition leans on that weighting.

**Recommendation for users:** use `save_classifier_directions=True` instead.
Classifier directions give you the same underlying geometry (one coefficient
vector per pairwise classifier) without the shrinkage step, and are the stable
surface that downstream consumers (`morphology_geometry/`,
`trajectory_condensation/`) depend on.

The code path stays in the repo so existing saved runs remain loadable, but do
not enable the flag for new runs.

---

This document captures planned improvements to the `analyze.classification` API surface.
It prioritizes changes that reduce user friction when making talk figures.

## Current state (as of 2026-03-05)

- Compute:
  - `run_classification_test(...) -> MulticlassOVRResults` (canonical long-form `comparisons` table)
- Plot:
  - `analyze.classification.viz.plot_multiple_aurocs(...)` (matplotlib overlay helper)
  - `analyze.classification.viz.plot_aurocs_over_time(...)` (faceting-engine based; plotly/mpl)
- Accumulate:
  - `ClassificationResults` stacks multiple runs into one table and supports save/load with a `tag` column.

## Design goals

1. DataFrame-centric UX: users operate on tables/objects, not directories.
2. One plotting “top API”: a single AUROC-over-time entrypoint scales from single → overlay → faceted.
3. Deterministic, talk-safe figures: consistent legends, baselines, and significance markers.
4. Future-proof multi-run support: allow multiple runs per metric via a boring `tag` column.

## Contrast coordinates (shrinkage rewrite)

The contrast-coordinate layer is frozen while the shrinkage math is rewritten.
Keep the code for compatibility, but do not treat it as the recommended user
surface. For new work, prefer classifier directions as the stable geometry API.

---

## P0 — Make plotting feel inevitable (highest priority)

### P0.1: Make `plot_aurocs_over_time` the top-level plotting primitive

Ensure a single call covers:
- one comparison (one curve)
- multiple comparisons (overlay)
- multiple metrics (facet by `metric`)
- optional significance markers and optional chance baseline

Keep the bare function explicit (no “smart” auto-faceting surprises).

**Acceptance**
- A user can pass a long-form comparisons DataFrame and get correct overlays/facets without pre-munging.

### P0.2: Provide a container method with smart defaults (no tag leakage)

`ClassificationResults.plot_aurocs_over_time(tag="default", ...)`:
- filters to `tag="default"` unless `tag="all"`
- if multiple metrics and no facet specified, auto `facet_col="metric"` (container only)

**Acceptance**
- Users do not need to know what `tag` is to make normal figures.

### P0.3: Standardize plot semantics for talks

Legend entries:
- dashed line = random chance (AUROC=0.5) with a clear label (e.g. “Random chance”)
- open circle = significance (e.g. `p ≤ 0.1`)

Avoid explanatory text embedded in the plot body; put semantics in the legend.
Provide an ergonomic “legend outside” mode that does not shrink the main panel.

**Acceptance**
- Slide-ready output with consistent legend + baseline labeling.

---

## P1 — Reduce “folder leakage” and persistence friction

### P1.1: Standardize persistence for multi-metric runs

Prefer one directory with:
- `comparisons.parquet`
- optional `null_summary.parquet`
- `metadata.json` (created_at, git_commit, python_version, schema_version, columns list)

Avoid forcing users into “load-from-folder → concat tables” workflows.

### P1.2: Migration bridge from legacy dir bundles

Support `ClassificationResults.from_dirs(...)` as a bridge (explicitly marked legacy-only).

---

## P2 — Null distributions / uncertainty (optional but likely)

### P2.1: Carry null summaries in the accumulator consistently

Always allow stacking null summaries with `metric` + `tag`.
Plotting should optionally show:
- mean±std band (current)
- future: quantile bands (e.g. 95% envelope) when available

### P2.2: Multiple-comparison / q-values (future)

Consider storing `qval` or per-window correction outputs if we start reporting many time bins.

---

## P3 — Binning UX

### P3.1: Canonical time column contract + alias coercion on ingest

Canonical: `time_bin_center`

Accepted aliases: `pred_hpf_bin_center`, `bin_center`, etc.

If needed: derive center from left/right or start/end edges.

### P3.2: Post-hoc rebinning (investigate)

Explore whether we can rebin AUROC-over-time outputs for plotting without re-running models.

Caveat: p-values / permutation-derived stats do not aggregate trivially.

---

## P4 — Small ergonomic helpers

- `ClassificationResults.subset(...)` for interactive slicing (metric/positive/time range).
- Optional plot-config saving:
  - store a small JSON next to figures capturing plot params for reproducibility.

---

## Notes / non-goals

- Do not auto-increment tags on collisions; that creates confusing doubled plots.
- Keep bare plotting functions explicit; only container methods get convenience behavior.

