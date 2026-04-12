# Future improvements — classification (AUROC over time)

Last updated: 2026-03-05

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

