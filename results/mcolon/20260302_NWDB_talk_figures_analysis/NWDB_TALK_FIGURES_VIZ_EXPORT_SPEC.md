# NWDB Talk Figures And Videos Viz Export Spec

## Summary

This document captures what was actually produced in `results/mcolon/20260302_NWDB_talk_figures_analysis`, which outputs should be treated as canonical, and how those outputs should be exported into reusable `viz` surfaces instead of remaining bespoke talk scripts.

This is a documentation-first pass. The goal is to preserve the current NWDB talk assets, identify the reusable figure and video families, and define implementation contracts for moving them into reusable plotting and rendering APIs.

## Current Analysis Folder Shape

The folder currently contains three kinds of assets:

1. Classification result bundles under `plot_dir/`, especially:
   - `classification/`
   - `classification_bin4/`
   - `classification_bin2_10_48/`
   - `classification_bin4_10_48/`

2. Talk-facing exported assets under `figures/`, especially:
   - `classification/`
   - `curvature_het_vs_wt/`
   - `phenotype_distribution/`
   - `phenotype_overlays/`
   - `phenotype_overlays_homozygous/`
   - `phenotype_transition_homozygous/`
   - `spawn_transition_penetrance_story/`
   - `genotype_overlay_video/`
   - `plate_timelapse/`

3. Bespoke authoring scripts that mostly fall into one of two categories:
   - thin wrappers around existing reusable plotting APIs,
   - talk-specific curation scripts with fixed embryos, fixed windows, and fixed presentation presets.

## Canonical Export Families

The following families should be treated as canonical reusable exports.

### 1. Classification AUROC Trend Exports

Current source scripts:
- `01_run_reference_genotype_classification_curvature.py`
- `02_plot_het_vs_wt_no_null.py`
- `03_plot_het_and_homo_vs_wt_no_null.py`
- `06_plot_selected_bin4_no_null_sig_legend_outside.py`

Current upstream reusable surface:
- `src/analyze/classification/viz/classification.py`

Current data contract:
- input bundle directory contains `comparisons.parquet`, `metadata.json`, and usually `summary.csv`
- bundle metadata defines `groups`, `reference`, `features`, `bin_width`, and permutation settings

Canonical outputs to preserve:
- single-comparison AUROC curves
- overlaid AUROC curves
- significance markers
- optional vertical reference line
- static `png` and `pdf`

Canonical NWDB presets:
- `no null band` presentation variant is primary
- `legend outside` is canonical for slide-ready exports
- `bin4` selected overlays for curvature, length, and embedding are canonical presentation outputs

Implementation target:
- extend `src/analyze/classification/viz` with a small export-oriented layer that accepts a classification bundle directory and export options rather than requiring results-folder-specific wrapper scripts

Required parameters:
- bundle dir
- comparison set
- metric label
- time window
- show or hide null band
- significance threshold and marker toggle
- legend placement
- optional reference lines
- output stem and formats

Acceptance criteria:
- same AUROC traces as current exports
- same x-axis restriction for NWDB talk presets
- same significance behavior
- same file outputs in `png` and `pdf`

### Classification Mode Clarification

The package default is a multiclass problem reported one-vs-rest per class. That is the right mental model for a default run with no explicit comparison arguments.

The NWDB genotype talk analysis is not using that default. It uses explicit binary comparisons:
- `cep290_heterozygous vs cep290_wildtype`
- `cep290_homozygous vs cep290_wildtype`

That distinction matters when exporting or reusing these figures:
- explicit `positive` and `negative` pairs should be described as binary comparison problems
- one-vs-rest should be described as multiclass reporting, not a separate problem type
- pooled negatives like `positive="pbx4_crispant", negative=("wik_ab", "inj_ctrl")` stay binary and should export as one resolved comparison against a pooled control

### 2. Static Genotype Curvature Overlay Exports

Current source script:
- `10_plot_curvature_het_vs_wt.py`

Current upstream reusable surface:
- `src/analyze/viz/plotting/feature_over_time.py`

Current data contract:
- `results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv`
- normalized curvature derived from `baseline_deviation_normalized`

Canonical outputs to preserve:
- genotype-specific curvature-over-time plots
- overlays for heterozygous vs wildtype and heterozygous vs wildtype vs homozygous
- static `png` and `pdf`

Canonical NWDB presets:
- HPF window `24-120`
- trend-line variant and individual-only variant
- outside legend
- talk palette from genotype color mapping config

Implementation target:
- keep rendering on top of `plot_feature_over_time`
- add an export helper in `src/analyze/viz` or a nearby domain-specific module that owns common layout, labels, and save behavior

Required parameters:
- feature column or derived feature preset
- genotype groups
- time window
- y-limits
- bin width
- trend smoothing
- individual or trend display toggles
- output formats

Acceptance criteria:
- same included embryos and time filters
- same trend behavior
- same legend and axis labels
- same output formats

### 3. Phenotype Distribution And Summary Exports

Current source scripts:
- `07_plot_phenotype_distribution_by_pair.py`
- `make_nwdb_phenotype_summary_individual_se_plots.py`
- parts of `make_nwdb_phenotype_transition_static_plots.py`

Current upstream reusable surfaces:
- `src/analyze/viz/plotting/feature_over_time.py`
- `src/analyze/viz/plotting/proportions.py`

Current data contract:
- `embryo_data_with_labels.csv`
- `embryo_cluster_labels.csv`

Canonical outputs to preserve:
- phenotype distribution by breeding pair
- phenotype proportions by pair
- phenotype summary overlays with individual traces and summary curves
- static transition stills that bridge genotype and phenotype narratives

Canonical NWDB presets:
- `Intermediate` is collapsed into `Low_to_High`
- pair-level distribution figures are homozygous-only and filtered to the selected experiment set
- summary exports preserve the current phenotype color mapping and slide-ready layout

Implementation target:
- reuse the generic plotting layer for the data drawing
- add a thin domain-specific preset layer for phenotype label normalization, color mappings, and named NWDB presets

Required parameters:
- phenotype mapping rules
- genotype and pair filters
- time window
- summary statistic and error band configuration
- legend placement
- output subfamily

Acceptance criteria:
- same phenotype grouping and relabeling
- same pair-level counts and proportions
- same summary trends and still composition

### 4. Spawn Transition Penetrance Story Exports

Current source script:
- `08_plot_spawn_transition_penetrance_story.py`

Current upstream reusable dependencies:
- validated penetrance implementation in `results/mcolon/20260308_penetrance_quantile_envelope`
- `src/analyze` utilities for time binning

Current data contract:
- labeled embryo-frame table from `embryo_data_with_labels.csv`
- spawn-specific pair filtering

Canonical outputs to preserve:
- staged story figures showing `Low_to_High`, `High_to_Low`, and pooled transition penetrance trends
- slide-ready `png` and `pdf`

Canonical NWDB presets:
- spawn-only subset
- story time window `24-110`
- pooled transition curve shown as a distinct combined group
- talk-specific smoothing and line styling

Implementation target:
- promote the story plotting logic into a reusable penetrance-story export surface instead of leaving it as a results-local composition script

Required parameters:
- frame table
- pair filter
- genotype filter
- phenotype groups
- time window
- smoothing mode
- point, band, and legend toggles

Acceptance criteria:
- same penetrance summaries as the validated penetrance code
- same staged curve variants
- same slide-ready presentation behavior

### 5. Synchronized Feature-Trace And Embryo Video Exports

Current source script:
- `make_nwdb_curvature_animation.py`

Current data contract:
- `embryo_data_with_labels.csv`
- `embryo_cluster_labels.csv`
- embryo snips under `morphseq_playground/training_data/bf_embryo_snips/{experiment_date}/`

Canonical outputs to preserve:
- one trace MP4 per selected panel item
- one embryo MP4 per selected panel item
- background still PNGs from the same preset
- optional trace-only stills

Why this is canonical:
- the matched feature plot plus synchronized embryo imagery is one of the highest-value outputs from this work
- it directly connects abstract feature trends to embryo appearance
- it should remain a first-class reusable export, not a talk-only artifact

Canonical NWDB presets:
- long x-axis context from `24-120` HPF
- shorter animated cursor window for talk pacing
- fixed featured embryos when using named NWDB presets
- `panel_by=genotype` and `panel_by=cluster_categories` are both valid canonical modes

Implementation target:
- create a reusable video export interface under `viz` or a closely related domain-specific module
- preserve the current paired-output model instead of collapsing it into a single merged movie

Required parameters:
- labeled frame table
- snip root
- feature column
- panel mode
- selected groups
- featured embryo ids or a selection strategy
- HPF window
- cursor window
- fps
- output frame count
- still export toggles
- output stem

Acceptance criteria:
- paired MP4s remain synchronized
- still PNGs come from the same preset as the videos
- same embryo selection when a named preset is reused
- same plot window and cursor behavior

### 6. Phenotype/Genotype Transition Video Exports

Current source script:
- `make_nwdb_phenotype_genotype_transition.py`

Current data contract:
- `embryo_data_with_labels.csv`
- `embryo_cluster_labels.csv`

Canonical outputs to preserve:
- ordered transition video sequences on a homozygous background
- corresponding background still PNGs
- matching static presentation plots in `phenotype_transition_homozygous/static_plots`

Canonical NWDB presets:
- fixed overlay embryos:
  - `20251106_H04_e01` for `Low_to_High`
  - `20251113_A02_e01` for `High_to_Low`
- ordered story variants with one or both overlays
- both `15s` and `20s` durations currently exist

Implementation target:
- expose a reusable transition-video preset system where background group, overlay sequence, colors, timing, and output stems are explicit inputs

Required parameters:
- background preset or explicit background group
- overlay specs
- time window
- fps
- frame count or duration
- smoothing
- styling preset
- output stem

Acceptance criteria:
- same video ordering and overlay composition
- same corresponding stills
- same duration-specific behavior when reusing a named preset

### 7. 96-Well Plate Timelapse Exports

Current source script:
- `make_96well_plate_timelapse.py`

Current data contract:
- embryo metadata CSV from `morphseq_playground/metadata/embryo_metadata_files/{experiment}_embryo_metadata.csv`
- snip JPEG root under `morphseq_playground/training_data/bf_embryo_snips`
- YX1 coordinate CSV at `morphseq_playground/metadata/YX1_nd2_ref_plate_xy_coordinates.csv`

Current reusable value:
- this is not just a talk movie
- it is reusable infrastructure for rendering a whole-plate developmental movie from metadata and snips

Canonical outputs to preserve:
- talk-ready full-plate MP4s
- dark and light themes
- 1080p and 4k variants
- shorter `15s` and `20s` presentation variants

Current important behavior to preserve:
- one selected embryo per well
- preference for `e01` when present
- fallback ranking by survival proxy and coverage
- optional verification that first and last snips exist
- death handling policy, currently `freeze_at_death`

Implementation target:
- create a reusable plate-render export surface rather than leaving the renderer only in this results folder

Required parameters:
- experiment id or explicit metadata CSV
- snip root
- coordinates CSV
- layout mode
- theme
- resolution
- fps
- duration or frame count
- HPF window
- missing-frame policy
- death policy
- well subset
- output path

Acceptance criteria:
- same well ordering and placement
- same single-embryo-per-well selection logic
- same theme and labeling behavior
- same presentation outputs for named NWDB presets

## Non-Canonical Or Archived Families

These should be documented but not promoted as primary export targets:

- null-band backup classification figures from `04_plot_het_vs_wt_with_null.py` and `05_plot_het_and_homo_vs_wt_with_null.py`
- palette-variant experiments from `09_plot_homo_vs_wt_multimetric_bin4_palette_variants.py`
- candidate-track triage exports from `make_not_penetrant_candidate_track_images.py`
- smoke, debug, verify, and layout-check outputs under `plate_timelapse_smoke/` and `plate_timelapse/`
- notebook copies and exploratory notebooks

These assets are useful references but should not drive the public `viz` interface.

## Proposed Export Surfaces

The end state should not be a single giant module. The reusable surfaces should stay close to the current architecture.

### Classification

Target area:
- `src/analyze/classification/viz/`

Responsibility:
- consume classification result bundles
- export slide-ready AUROC figures with minimal wrapper code

### Static Time-Series And Proportion Plots

Target area:
- `src/analyze/viz/plotting/`
- small domain-specific preset helpers adjacent to it

Responsibility:
- keep generic data drawing in reusable plotting APIs
- move phenotype/genotype preset logic into thin reusable wrappers

### Motion Exports

Target area:
- a new video-oriented `viz` surface under `src/analyze/viz/` or a nearby domain-specific package

Responsibility:
- synchronized trace plus embryo videos
- transition videos
- plate timelapse videos

Design rule:
- preserve named presets for NWDB talk assets, but keep those presets separate from the generic renderer contracts

## Named NWDB Presets To Preserve

These presets should be represented explicitly in the eventual implementation so users can recreate the current talk assets without reintroducing hardcoded one-off scripts.

### Classification Presentation Presets

- `classification_no_null`
- `classification_bin4_selected`
- `classification_legend_outside`

### Synced Video Presets

- genotype panels over `24-120` HPF
- cursor window over the narrower talk range
- fixed featured embryos for the current NWDB talk exports

### Transition Video Presets

- homozygous background with ordered `Low_to_High` and `High_to_Low` overlays
- `15s` and `20s` presentation variants

### Plate Timelapse Presets

- experiment `20260206`
- `dark` and `light`
- `15s` and `20s`
- `1080p` and `4k`
- `freeze_at_death`

## Implementation Notes

- Prefer reusing `src/analyze/classification/viz/classification.py` for AUROC drawing instead of copying plot logic again.
- Prefer reusing `plot_feature_over_time` and `plot_proportions` for static summary figures, with domain-specific wrappers only where phenotype relabeling and preset behavior are needed.
- Do not bury featured-embryo ids, story order, or experiment presets in renderer internals. Those belong in named preset/config objects.
- Keep static and motion exports aligned so a single preset can emit both canonical stills and matching videos where appropriate.

## Core Script Specs And Design Principles

This section is intentionally script-centric. It records the key spec, the core principle behind each script, how it currently works, and what must change to make it modular and maintainable.

### `01_run_reference_genotype_classification_curvature.py`

Key spec:
- build reusable classification bundles for CEP290 reference data across curvature, length, and embedding feature sets
- cache the filtered reference dataframe
- write bundle directories under `plot_dir/<classification_subdir>/`

Core principle:
- separate expensive statistical computation from downstream plotting
- store outputs in a stable bundle shape so multiple figure variants can reuse the same results

How it works:
- loads the CEP290 reference dataset
- keeps only embryos with valid labels
- filters to the requested HPF window
- drops `cep290_unknown`
- derives normalized `curvature`
- runs permutation-based classification for Het vs WT and Homo vs WT
- writes `comparisons.parquet`, `metadata.json`, and summary outputs per feature family

What needs to change for modularity:
- move the bundle-building contract out of the results folder into a stable library entrypoint
- define a typed bundle schema instead of relying on ad hoc directory conventions
- separate data loading, preprocessing, and classification execution into importable functions
- treat cache naming as a parameterized policy rather than embedding it in this script

### `02_plot_het_vs_wt_no_null.py`, `03_plot_het_and_homo_vs_wt_no_null.py`, `04_plot_het_vs_wt_with_null.py`, `05_plot_het_and_homo_vs_wt_with_null.py`, `06_plot_selected_bin4_no_null_sig_legend_outside.py`

Key spec:
- render talk-facing AUROC figures from classification bundles
- vary only a small set of presentational choices:
  - one curve vs two curves
  - null band on or off
  - significance on or off
  - legend inside or outside
  - selected `bin4` presentation preset

Core principle:
- plotting should be a thin preset layer on top of reusable classification visualization code

How they work:
- load a bundle with `MulticlassOVRResults.from_dir`
- optionally drop null-band columns or p-values
- call `plot_multiple_aurocs`
- apply the NWDB axis restriction and legend placement helper
- save slide-ready `png` and `pdf`

What needs to change for modularity:
- replace these near-duplicate scripts with one configurable export API plus a preset registry
- centralize filename conventions and figure-save behavior
- move NWDB-specific presentation presets into config objects rather than separate files
- make “with null”, “no null”, “selected bin4”, and “outside legend” named presets instead of bespoke scripts

### `07_plot_phenotype_distribution_by_pair.py`

Key spec:
- produce pair-level phenotype distribution figures and phenotype proportion plots for the selected homozygous subset

Core principle:
- use generic plotting functions for the drawing, and keep domain-specific logic limited to data filtering and color/preset selection

How it works:
- loads labeled embryo-frame data
- normalizes curvature
- filters to homozygous embryos, selected breeding pairs, and selected experiments
- remaps `Intermediate` to `Low_to_High`
- calls `plot_feature_over_time` for trajectory panels
- calls `plot_proportions` for stacked bar composition

What needs to change for modularity:
- move phenotype relabeling and pair/experiment filters into reusable dataset-preparation helpers
- expose the pair-summary export as a stable function instead of a script
- stop hardcoding the selected experiment dates and pair regex in the renderer layer

### `08_plot_spawn_transition_penetrance_story.py`

Key spec:
- produce a staged transition-penetrance story from validated penetrance calculations

Core principle:
- reuse validated analytical logic, then add only a thin story-composition layer for presentation

How it works:
- loads the labeled frame table
- restricts to the spawn-related subset and story time window
- computes embryo-bin penetrance summaries by phenotype group
- constructs a pooled transition group
- renders staged curve variants with the shared NWDB transition style

What needs to change for modularity:
- define a supported library interface for penetrance summary generation rather than importing from a neighboring results folder
- move story-group definitions and smoothing/display presets into named config
- isolate data summarization from presentation sequencing

### `10_plot_curvature_het_vs_wt.py`

Key spec:
- render genotype-specific and genotype-overlay curvature plots in a slide-ready format

Core principle:
- generic time-series plotting should do the heavy lifting, with only a small export wrapper handling labels, presets, and saving

How it works:
- loads labeled embryo-frame data
- derives normalized `curvature`
- filters to the HPF window and target genotypes
- loops across plot families and trend/no-trend variants
- calls `plot_feature_over_time`
- rewrites labels and saves canonical exports

What needs to change for modularity:
- move the variant matrix into a preset table instead of hardcoded loops
- replace custom label rewriting with a reusable display-label mapping layer
- share save/layout logic with the phenotype-summary exporters

### `make_nwdb_phenotype_transition_static_plots.py`

Key spec:
- generate the canonical static “bridge” figures between genotype and phenotype narratives
- generate both overlay stills and summary stills in one place

Core principle:
- a single preset-driven renderer can produce multiple narrative stills if overlays, backgrounds, and summary variants are expressed as data rather than inline logic

How it works:
- loads embryo frames and phenotype labels
- defines typed overlay, background, combo, and summary specs
- builds unfaded background plots for specific backgrounds
- draws selected overlay embryos on top
- also renders phenotype summary variants with configurable error bands and smoothing

What needs to change for modularity:
- split background rendering, overlay rendering, and summary rendering into separate reusable functions
- move the embedded overlay embryo ids into preset config
- stop combining too many export families in one script
- treat typed spec objects as the seed of a real reusable configuration layer

### `make_nwdb_phenotype_summary_individual_se_plots.py`

Key spec:
- generate phenotype summary plots using the generic API-native error band behavior rather than fully bespoke summary rendering

Core principle:
- validate that the generic plotting layer can cover the phenotype-summary use case if given the right preset parameters

How it works:
- loads the same labeled embryo-frame data
- prepares phenotype subsets
- loops across summary families, error-band toggles, and legend toggles
- calls `plot_feature_over_time` directly

What needs to change for modularity:
- absorb these variants into the future phenotype-summary preset registry
- avoid maintaining both this script and `make_nwdb_phenotype_transition_static_plots.py` as overlapping export pathways

### `make_nwdb_curvature_animation.py`

Key spec:
- generate synchronized feature-trace and embryo-snip videos, plus matching stills

Core principle:
- the rendering unit is a preset-driven synchronized pair of outputs, not just a single MP4
- the abstract feature trace and the concrete embryo imagery must stay aligned

How it works:
- loads labeled frame data and phenotype labels
- chooses panel groups by genotype or phenotype category
- selects or accepts featured embryo ids
- renders a background trace plot over the full context window
- advances a cursor through a possibly narrower talk window
- emits one trace MP4 and one embryo MP4 per selected item
- also writes still PNGs from the same preset

What needs to change for modularity:
- extract embryo selection, trace rendering, still rendering, and movie encoding into separate reusable components
- define a stable preset object for panel mode, selected groups, featured embryos, and time windows
- replace file/folder naming conventions embedded in the script with a reusable exporter interface

### `make_nwdb_phenotype_genotype_transition.py`

Key spec:
- generate ordered transition videos that bridge from genotype-background context to phenotype-overlay stories

Core principle:
- story sequencing should be declarative
- overlay order, background, and duration are presentation presets, not renderer internals

How it works:
- defines typed overlay specs and ordered video specs
- loads the labeled frame data
- renders each transition sequence on a fixed background with fixed overlay embryos
- writes MP4s and corresponding background stills

What needs to change for modularity:
- move ordered transition definitions into reusable preset config
- extract a general transition renderer from the current NWDB-specific story
- share styling, save behavior, and data preparation with the static transition export path

### `make_96well_plate_timelapse.py`

Key spec:
- render a full-plate MP4 from embryo metadata, snip images, and plate coordinates

Core principle:
- plate timelapse rendering is reusable infrastructure, not just a one-off presentation artifact
- the renderer should make one high-quality choice per well and then compose a stable plate movie

How it works:
- loads embryo metadata for one experiment
- ranks embryos within each well, preferring `e01` and better coverage/survival
- optionally verifies snip availability
- maps wells into the requested layout
- composites circular well tiles into a 96-well plate movie
- applies theme, labels, and death-handling policy
- writes one MP4 per requested preset

What needs to change for modularity:
- separate well-level embryo selection from image compositing
- define a typed plate-render config object
- promote plate layout, theme, and death-policy logic into reusable utilities
- keep debug and verification outputs outside the canonical export API

### `_plot_nwdb_genotype_classification_utils.py`

Key spec:
- shared helper for axis restrictions, legend placement, and figure saving in classification exports

Core principle:
- small presentation helpers should be shared instead of copied across near-identical scripts

How it works:
- enforces the NWDB x-axis range
- standardizes outside-legend placement
- centralizes `png` and `pdf` saving

What needs to change for modularity:
- move these helpers into a stable export utility module instead of keeping them results-local
- generalize the helpers so they can support non-NWDB presets too

### `_nwdb_transition_plot_style.py`

Key spec:
- shared style constants for the transition-story family

Core principle:
- transition assets need a consistent presentation style across static and motion exports

How it works:
- defines figure size, DPI, label sizing, and subplot margins
- applies them through a small styling helper

What needs to change for modularity:
- migrate the style preset into a reusable style registry
- keep the preset name stable, but avoid tying the style helper to this one results folder

## Test And Parity Checklist

For each canonical family, future implementation should verify:

- input contract matches current results-folder scripts
- output files are produced in the expected formats
- NWDB named preset reproduces the current talk-facing asset family
- grouping, filtering, and relabeling match current behavior
- time windows and cursor windows match current behavior
- still and video outputs derived from the same preset stay visually aligned

Specific parity checks:

- Classification:
  - identical time-bin filtering
  - identical comparison selection
  - identical significance marker placement

- Static summaries:
  - identical phenotype relabeling
  - identical pair filtering
  - identical trend and error-band configuration

- Synced videos:
  - identical featured embryo selection for named presets
  - identical trace and embryo-video synchronization

- Transition videos:
  - identical story order
  - identical overlay composition

- Plate timelapse:
  - identical well ordering
  - identical best-embryo-per-well selection behavior
  - identical death handling and fallback behavior

## Defaults Chosen In This Spec

- Static figures and motion assets are both first-class export targets.
- The synchronized feature-trace plus embryo-video pattern remains canonical and should not be reduced to stills only.
- The 96-well plate timelapse renderer is reusable infrastructure and should be exported as such.
- Canonical documentation is preferred over exhaustive artifact cataloging.
- Backup, debug, and exploratory outputs remain documented but non-canonical.
