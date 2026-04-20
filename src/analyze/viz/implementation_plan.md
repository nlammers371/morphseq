# Visualization Implementation Plan

This document turns the recent review of `results/mcolon/20260413_research_reports_presentation`
and `results/mcolon/20260302_NWDB_talk_figures_analysis` into a concrete
implementation plan for `src/analyze/viz`.

The organizing principle is minimal-first:

- change defaults when the current default is broadly wrong
- add small helpers when the default is acceptable but awkward to reuse
- add API hooks when scripts are forced to mutate figures after render

## Classification Rules

Use these labels consistently when making changes:

- `default change`
  Change the library's built-in behavior because repeated overrides show the
  current default is broadly wrong.
- `new helper`
  Add a small reusable helper or preset because the current behavior is fine,
  but the way to invoke it is too repetitive or too local to results scripts.
- `API hook`
  Add an explicit argument or structure because users currently have to patch
  figure objects after render or rewrite legends/titles manually.

## Evidence Summary

The same problems recur across both result folders:

- repeated output-directory creation before saving
- repeated label remapping and ordered color-map construction
- repeated legend placement and legend text rewriting
- repeated alpha, linewidth, figure-size, and talk-style overrides
- repeated post-render title and annotation cleanup

That maps cleanly onto three buckets:

- `bad defaults`
  dense plots are too cluttered by default, generic plotting still leaks
  domain-aware color behavior, and style values are not always the source of
  truth
- `hard to use defaults`
  local talk helpers such as NWDB transition styling are useful, but they live
  in `results/` rather than in reusable `src/` presets
- `API failure`
  scripts rewrite legends, annotations, and figure titles after render because
  the library does not expose those controls directly

## Phase 1: Minimal Default Fixes

These should be done first. They remove repeated script boilerplate without
changing the plotting model.

### `src/analyze/viz/plotting/faceting_engine/__init__.py`

- `default change`
  Make `render()` create the parent directory for `output_path` before writing
  HTML, PNG, or PDF output.
- `default change`
  Keep save behavior consistent across `plotly`, `matplotlib`, and `both`
  modes.

Why:
- repeated `mkdir(parents=True, exist_ok=True)` in result scripts is pure
  boilerplate and does not carry analysis intent

Validation targets:
- `results/mcolon/20260413_research_reports_presentation/01_render_raw_projection_time_slice.py`
- `results/mcolon/20260302_NWDB_talk_figures_analysis/_plot_nwdb_genotype_classification_utils.py`

### `src/analyze/viz/plotting/faceting_engine/style/defaults.py`

- `default change`
  Make `StyleSpec` the real source of truth for individual alpha, individual
  width, trend alpha, trend width, band alpha, legend font size, and legend
  location.
- `default change`
  Revisit dense-facet defaults. Keep repeated axis titles and repeated tick
  labels opt-in rather than the default for crowded layouts.
- `new helper`
  Add named presets rather than more ad hoc constants.
  Suggested first set:
  `default_style()`, `paper_style()`, `presentation_style()`, and
  `dense_facet_style()`.

Why:
- current style fields exist, but several plot builders still hardcode values
- repeated talk-style overrides show users want named presets more than raw
  per-script numbers

Validation targets:
- `results/mcolon/20260413_research_reports_presentation/02_render_presentation_animations.py`
- `results/mcolon/20260302_NWDB_talk_figures_analysis/_nwdb_transition_plot_style.py`

### `src/analyze/viz/plotting/feature_over_time.py`

- `default change`
  Route individual trace alpha/width and trend alpha/width through `StyleSpec`
  instead of hardcoding them in the subplot builder.
- `default change`
  Keep the generic path palette-first by default. Do not treat genotype-aware
  color resolution as the generic default behavior.
- `API hook`
  Add an optional figure-size/style hook that lets callers request the common
  presentation sizing directly rather than resizing the returned figure.

Why:
- hardcoded styling values in the plot builder undermine `StyleSpec`
- repeated `fig.set_size_inches(...)` calls show the sizing controls are not
  easy to access through the current API

Validation targets:
- `results/mcolon/20260302_NWDB_talk_figures_analysis/10_plot_curvature_het_vs_wt.py`
- `results/mcolon/20260302_NWDB_talk_figures_analysis/07_plot_phenotype_distribution_by_pair.py`

### `src/analyze/viz/plotting/faceting_engine/renderers/matplotlib.py`

- `default change`
  Keep legend assembly deterministic and renderer-owned.
- `default change`
  Use style-controlled legend placement consistently, especially for the common
  "outside right" case.
- `default change`
  Make axis-label and tick-label visibility honor style defaults without
  requiring script-level cleanup.

Why:
- many NWDB and talk scripts use `legend_loc="outside"` or post-hoc figure
  adjustments because this behavior is not yet a first-class default

Validation targets:
- `results/mcolon/20260302_NWDB_talk_figures_analysis/06_plot_selected_bin4_no_null_sig_legend_outside.py`
- `results/mcolon/20260302_NWDB_talk_figures_analysis/07_plot_phenotype_distribution_by_pair.py`

### `src/analyze/viz/plotting/faceting_engine/renderers/plotly.py`

- `default change`
  Align legend placement and facet-label behavior with the Matplotlib renderer
  as closely as practical.
- `default change`
  Keep Plotly title and annotation defaults less invasive so users are not
  forced to patch `fig.layout.annotations` after render.

Why:
- the raw projection talk script strips and rewrites annotations after render,
  which is a sign that the renderer is forcing too much structure into the
  figure by default

Validation targets:
- `results/mcolon/20260413_research_reports_presentation/01_render_raw_projection_time_slice.py`

## Phase 2: Shared Reuse Helpers

These changes are still small, but they move recurring talk/NWDB patterns out
of result folders and into `src/`.

### `src/analyze/viz/styling/color_utils.py`

- `new helper`
  Add a helper to build an ordered category-to-color mapping from:
  values present, a preferred display order, and a preferred palette or lookup.
- `new helper`
  Add a helper to remap raw labels into display labels before ordered color
  assignment.

Suggested shapes:

```python
apply_label_map(values, label_map)
ordered_present_values(values, preferred_order=None, label_map=None)
build_ordered_color_lookup(values, preferred_order=None, color_lookup=None, palette=None)
```

Why:
- result scripts repeatedly define `DISPLAY_NAME_MAP`, `DISPLAY_ORDER`, and a
  presentation palette
- this behavior belongs in reusable styling helpers, not per-script ad hoc code

Validation targets:
- `results/mcolon/20260413_research_reports_presentation/01_render_raw_projection_time_slice.py`
- `results/mcolon/20260413_research_reports_presentation/02c_render_rotation_gifs.py`

### `src/analyze/viz/styling/__init__.py`

- `new helper`
  Export the new label and ordered-color helpers cleanly.
- `new helper`
  Export any named presentation presets that are intended to be public.

Why:
- callers should not need deep imports into utility modules for common plotting
  setup

### `src/analyze/viz/plotting/plotting_3d.py`

- `default change`
  Keep the generic path generic. Use palette-first color assignment unless the
  caller explicitly asks for genotype-aware resolution.
- `new helper`
  Add a small style object or preset helper for common 3D presentation choices:
  camera, point size, alpha, linewidth, FPS/DPI where applicable.
- `API hook`
  Add a style/preset argument so scripts do not have to pass the same camera and
  visibility parameters repeatedly.

Suggested shape:

```python
@dataclass
class Plot3DStyle:
    point_size: int = 4
    point_opacity: float = 0.65
    line_width: float = 1.5
    line_opacity: float = 0.3
    camera_eye: dict[str, float] | None = None

def talk_3d_style() -> Plot3DStyle:
    ...
```

Why:
- the presentation scripts keep repeating the same 3D visual defaults
- this is a strong sign that users want one named preset, not more low-level
  knobs

Validation targets:
- `results/mcolon/20260413_research_reports_presentation/02_render_presentation_animations.py`

## Phase 3: API Hooks For Manual Override Pain

These are the changes that directly address cases where scripts currently patch
the rendered figure object.

### `src/analyze/viz/plotting/faceting_engine/ir.py`

- `API hook`
  Expand figure-level metadata so common post-render edits become declarative.

Suggested additions:
- `subtitle`
- `legend_order`
- `legend_label_map`
- `suppress_facet_labels`

Why:
- legend and title behavior should be specified before rendering, not rewritten
  on the returned figure object

### `src/analyze/viz/plotting/feature_over_time.py`

- `API hook`
  Add a `label_map` or `display_name_map` argument that applies to legend text
  and any derived labels before rendering.
- `API hook`
  Add a `figure_size` or `style`-driven size override path that avoids manual
  `set_size_inches`.

Why:
- NWDB curvature scripts currently rewrite legend labels after render
- repeated sizing changes are another sign the current API does not expose the
  right control point

Validation targets:
- `results/mcolon/20260302_NWDB_talk_figures_analysis/10_plot_curvature_het_vs_wt.py`

### `src/analyze/viz/plotting/faceting_engine/renderers/plotly.py`

- `API hook`
  Respect `subtitle`, `legend_label_map`, `legend_order`, and
  `suppress_facet_labels` if added to `FigureData`.

Why:
- the raw projection talk script currently strips subplot labels and rewrites
  the title after render

Validation targets:
- `results/mcolon/20260413_research_reports_presentation/01_render_raw_projection_time_slice.py`

## Phase 4: Documentation And Public Surface Cleanup

This phase should happen after the first two phases are in place.

### `src/analyze/viz/README.md`

- `default change`
  Fix the broken migration example where the before/after import story is not
  actually different.
- `default change`
  Make the documentation honest about what is generic vs what is domain-aware.
- `new helper`
  Add examples showing the preferred use of `style=` and any public presets.

Why:
- the current README overstates how domain-agnostic the current defaults are
- new helpers and hooks should be demonstrated immediately or they will not be
  adopted

## Out Of Scope For This Pass

These are useful, but should not block the minimal implementation:

- broad redesign of the faceting engine
- moving all NWDB- or talk-specific logic into `src/`
- creating a new domain-specific visualization package
- collapsing all results-local helper scripts into one API immediately

The first pass should focus on removing repeated boilerplate and post-render
patching while keeping the current plotting model intact.

## Trajectory Condensation Viz Addendum

The presentation review also surfaced a parallel set of issues in
`src/analyze/trajectory_condensation/viz`. These should be tracked in the same
plan because the talk-facing scripts are leaning heavily on that API, not just
on `src/analyze/viz`.

The same three labels apply here:

- `default change`
  move repeated talk-style constants and save behavior into the reusable
  condensation viz layer
- `new helper`
  add small presets for display-label remapping, color ordering, and
  presentation rendering
- `API hook`
  expose title, annotation, legend, camera, and output controls before render
  so scripts stop mutating figures after the fact

### Why This Needs Its Own Section

The 2026-04-13 presentation scripts repeatedly override behavior from:

- `src/analyze/trajectory_condensation/viz/animation.py`
- `src/analyze/trajectory_condensation/viz/api.py`
- `src/analyze/trajectory_condensation/viz/condensed_time_slice_viewer.py`

Recurring symptoms:

- repeated label normalization and ordered color-map construction
- repeated camera, DPI, alpha, linewidth, and figure-size overrides
- repeated post-render title and annotation cleanup in the Plotly time-slice
  viewer
- repeated local runtime/cache setup in result scripts because the rendering
  path is brittle in headless environments

### `src/analyze/trajectory_condensation/viz/api.py`

- `default change`
  Promote the animation defaults that are currently private module constants
  into a user-facing style or config object.
- `new helper`
  Add a named presentation preset for the standard orbit / iteration / time
  slice outputs used in talks.
- `API hook`
  Allow `render_run()` and related helpers to accept a rendering preset or
  expanded config so callers do not need to pass low-level display values into
  each lower-level animation function separately.

Why:
- `_FPS_ROTATION`, `_FPS_ITERATIONS`, `_N_FRAMES`, `_ELEV`, `_AZIM_START`, and
  `_AZIM_END` are internal constants today, but the presentation scripts are
  repeatedly overriding the same values externally
- `VizConfig` currently covers only part of the visual surface

Validation targets:
- `results/mcolon/20260413_research_reports_presentation/02_render_presentation_animations.py`
- `results/mcolon/20260413_research_reports_presentation/02b_render_iterations_mp4.py`

### `src/analyze/trajectory_condensation/viz/animation.py`

- `default change`
  Centralize shared 3D animation styling defaults instead of repeating them in
  every public animation function signature.
- `new helper`
  Add a reusable 3D animation style object or named preset for talk/paper/debug
  output variants.
- `API hook`
  Expose title policy and legend policy explicitly so callers do not have to
  post-process saved outputs or reimplement local drawing wrappers just to hide
  subtitles, restyle legends, or change title structure.

Suggested shape:

```python
@dataclass
class Animation3DStyle:
    figsize: tuple[float, float] = (8, 7)
    dpi: int = 120
    point_size: float = 8.0
    alpha_point: float = 0.6
    alpha_line: float = 0.2
    linewidth: float = 0.6
    elev: float = 25.0
    azim_start: float = -60.0
    azim_end: float = 300.0

def presentation_animation_style() -> Animation3DStyle:
    ...
```

Why:
- the presentation scripts repeatedly pass `elev`, `azim`, `fps`, `dpi`,
  `figsize`, `point_size`, `alpha_point`, `alpha_line`, and `linewidth`
- that is a strong sign of missing preset support rather than a need for more
  one-off wrappers

Validation targets:
- `results/mcolon/20260413_research_reports_presentation/02_render_presentation_animations.py`
- `results/mcolon/20260413_research_reports_presentation/02c_render_rotation_gifs.py`

### `src/analyze/trajectory_condensation/viz/condensed_time_slice_viewer.py`

- `default change`
  Keep the time-slice viewer's default subplot titles and annotation behavior
  less invasive for presentation use.
- `new helper`
  Add a label-display preset path so genotype-like raw labels can be normalized
  consistently before legend and hover generation.
- `API hook`
  Add explicit controls for:
  `subplot_titles`,
  `show_default_subplot_titles`,
  `label_map`,
  `ordered_labels`,
  and `main_title`.

Why:
- the talk script currently edits `fig.layout.annotations`, frame annotations,
  and the top-level title after render
- that is a direct API failure signal, not just a style preference

Validation targets:
- `results/mcolon/20260413_research_reports_presentation/01_render_raw_projection_time_slice.py`

### `src/analyze/trajectory_condensation/viz/README_viz.md`

- `default change`
  Document the distinction between the standard reusable bundle output and the
  talk-facing presentation preset output.
- `new helper`
  Add examples that show how to use the planned presets rather than repeating
  raw camera and alpha arguments.

Why:
- the current README exposes the low-level animation API, but not a reusable
  preset story for the most common presentation outputs

### Headless Runtime Setup

This likely belongs outside `viz` proper, but it should still be tracked.

- `new helper`
  Add a tiny shared helper for headless Matplotlib / cache-dir setup that
  result scripts can call instead of repeating local `_configure_runtime_env()`.

Possible homes:

- `src/analyze/viz/runtime.py`
- `src/analyze/utils/runtime.py`
- `src/analyze/trajectory_condensation/viz/runtime.py`

Why:
- the same cache/bootstrap code is repeated in multiple result scripts
- this is not a plotting default issue; it is a reuse/helper issue

## Recommended Execution Order

1. `src/analyze/viz/plotting/faceting_engine/__init__.py`
2. `src/analyze/viz/plotting/faceting_engine/style/defaults.py`
3. `src/analyze/viz/plotting/feature_over_time.py`
4. `src/analyze/viz/plotting/faceting_engine/renderers/matplotlib.py`
5. `src/analyze/viz/plotting/faceting_engine/renderers/plotly.py`
6. `src/analyze/viz/styling/color_utils.py`
7. `src/analyze/viz/styling/__init__.py`
8. `src/analyze/viz/plotting/plotting_3d.py`
9. `src/analyze/viz/plotting/faceting_engine/ir.py`
10. `src/analyze/viz/README.md`
11. `src/analyze/trajectory_condensation/viz/api.py`
12. `src/analyze/trajectory_condensation/viz/animation.py`
13. `src/analyze/trajectory_condensation/viz/condensed_time_slice_viewer.py`
14. `src/analyze/trajectory_condensation/viz/README_viz.md`

## Success Criteria

The first pass is successful when the following become true:

- result scripts no longer need to create output directories manually before
  calling `render()`
- common trace alpha and linewidth changes can be made by editing `StyleSpec`
  rather than patching plot builders
- legend placement does not require repeated local helper code for common cases
- scripts do not need to rewrite legend text or strip figure annotations after
  render for the common presentation outputs
- generic plotting defaults are easier to explain and less domain-assumptive
- trajectory-condensation presentation outputs can be produced from reusable
  presets rather than repeated per-script camera/title/alpha overrides
