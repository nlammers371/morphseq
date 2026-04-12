# WT Quantile Envelope Penetrance Pipeline

This directory is a results-local precursor to a future penetrance package migration.
Another agent should treat this folder as the authoritative local implementation for the
current penetrance workflow.

## High-level workflow
The pipeline is organized as a sequence of stages:

1. `data_loading.py`
   - loads `embryo_data_with_labels.csv`
   - adds `time_bin` using the configured bin width
2. `envelope.py`
   - aggregates frame rows into embryo-bin summaries for the main presentation path
   - computes raw WT quantiles by time bin
   - fits smoothed WT envelopes using LOESS and robust bin exclusion
3. `calls.py`
   - applies raw / smoothed / hybrid thresholds to mark rows as penetrant
4. `summaries.py`
   - computes penetrance-by-time tables, embryo consistency summaries, and calibration tables
5. `penetrance_plots.py`
   - renders all presentation and diagnostic figures from the saved summary tables / in-memory outputs
6. `run_pipeline.py`
   - orchestrates the full run and writes outputs
7. `01_compute_envelope_and_penetrance.py`
   - compatibility wrapper that simply calls `run_pipeline.main()`

## Module roles
- `config.py`: paths, constants, plot defaults, colors, threshold settings, and smoothing settings
- `data_loading.py`: load labeled embryo data and add time bins
- `smoothing.py`: LOESS implementation and curve-selection logic
- `envelope.py`: embryo-bin aggregation and WT envelope construction
- `calls.py`: threshold application and penetrance calls
- `summaries.py`: penetrance, consistency, and calibration summaries
- `penetrance_plots.py`: plotting surface for diagnostics and presentation figures
- `run_pipeline.py`: orchestration entrypoint for the full results-local pipeline
- `01_compute_envelope_and_penetrance.py`: stable wrapper entrypoint

## Plotting infrastructure
`penetrance_plots.py` is the plotting API for this folder.

Design rules:
- plotting functions create their own figure/axes when none are passed
- plotting functions return `(fig, ax)` or `(fig, axes)` and do not save files directly
- orchestration code in `run_pipeline.py` owns disk I/O and naming
- common styling is centralized through `_style_ax(...)` and `_resolve_colors(...)`

Important plotting functions:
- `plot_wt_envelope_diagnostic(...)`
  - WT scatter, raw quantiles, smoothed envelope, unsupported-bin shading
- `plot_penetrance_curves(...)`
  - category-level penetrance over time
  - supports `curve_mode="raw"` and `curve_mode="smoothed"`
  - supports `band_mode="se"` and `band_mode="iqr"`
  - supports explicit toggles for `show_band`, `show_line`, and `show_points`
- `plot_scatter_and_penetrance(...)`
  - genotype-specific scatter on top, penetrance panel on bottom
- `plot_penetrance_heatmap(...)` and `plot_embryo_consistency(...)`
  - supporting summary views

The plotting surface is intentionally separated from the statistics. If another agent wants
to change colors, legends, titles, line widths, alpha, or which visual elements appear,
that should usually happen in `penetrance_plots.py` or `config.py`, not in the inference code.

## Colors
Color definitions live in `config.py`.

Current category colors:
- `Low_to_High` → red `#E74C3C`
- `High_to_Low` → blue `#3498DB`
- `Intermediate` → purple `#9B59B6`
- `Not Penetrant` → green `#2ECC71`

Current genotype colors:
- `cep290_wildtype` → blue `#1f77b4`
- `cep290_heterozygous` → orange `#ff7f0e`
- `cep290_homozygous` → red `#d62728`

Color handling rules:
- explicit maps in `config.py` are the first source of truth
- `_resolve_colors(...)` in `penetrance_plots.py` fills any missing groups from a fallback palette
- if a new category or genotype is introduced, add it to `config.py` first so downstream figures stay stable

## Smoothing and presentation policy
The presentation-facing figures should emphasize the **smoothed** penetrance plots for trend communication.
That is the current recommendation and should remain the default unless there is a deliberate reason to change it.

Current display policy:
- `PRESENTATION_CURVE_MODE = "smoothed"`
- `PRESENTATION_CURVE_FRAC = 0.20`
- presentation penetrance variants currently use **SE ribbons**
- top-axis headroom is intentionally added so points/ribbons are not visually clipped at `1.0` / `100%`

Why smoothed plots are preferred:
- raw penetrance points are useful as evidence of sampling density and local variability
- smoothed curves are better for presentation because they communicate the developmental trend cleanly
- the audience should read the smoothed line as the trend and the points/ribbon as support around it

Practical rule for future edits:
- if the figure is for a talk, poster, or overview slide, start from the smoothed outputs
- if the figure is for diagnostics or debugging, raw points and raw-envelope views are appropriate

## Statistical method summary
The current method is:
- bin time using `TIME_BIN_WIDTH`
- aggregate the main presentation path at the embryo-bin level using `EMBRYO_BIN_AGG`
- estimate WT lower/upper quantiles per time bin
- robustly exclude outlier bins from smoothing when selecting the LOESS fit
- build raw and smoothed envelope bounds
- call penetrance using `EMBRYO_CALL_MODE` (`hybrid` by default)
- summarize penetrance by category or genotype over time
- render smoothed presentation figures from those summaries

Important distinction:
- the WT envelope smoothing is part of threshold construction
- the penetrance curve smoothing is a **display** choice for visualization

## Canonical run command
```bash
conda run -n segmentation_grounded_sam --no-capture-output python \
  results/mcolon/20260308_penetrance_quantile_envelope/01_compute_envelope_and_penetrance.py
```

## Outputs
- Tables: `results/mcolon/20260308_penetrance_quantile_envelope/outputs/tables/`
- Presentation figures: `results/mcolon/20260308_penetrance_quantile_envelope/outputs/figures/presentation/`
- Diagnostic figures: `results/mcolon/20260308_penetrance_quantile_envelope/outputs/figures/diagnostics/`

### Spawn-only transition story outputs
There is now an additional presentation branch for the talk narrative:
- filter to `24–120 hpf`
- filter to CEP290 spawn rows using `pair == "cep290_spawn"` with future-proof fallback for missing / `"none"`
- recompute the embryo-level WT envelope and penetrance calls on that filtered subset
- restrict the story figures to `cep290_homozygous`
- focus on `Low_to_High`, `High_to_Low`, and a pooled `Transition_Combined` group

Story figure directory:
- `results/mcolon/20260308_penetrance_quantile_envelope/outputs/figures/presentation/spawn_24_120_transition_story/`

Story outputs are split into:
- main sequential overlays on a single penetrance axis (`01` → `03`)
- `supplemental/` for extra singles, legend variants, and the X-mark scatter view

Story tables:
- `wt_threshold_summary_embryo_spawn_24_120.csv`
- `embryo_bin_classification_spawn_24_120.csv`
- `category_penetrance_by_time_embryo_spawn_24_120_homo_transition_story_pct.csv`

Key presentation outputs:
- `penetrance_curves_by_category_embryo_smoothed.png`
- `penetrance_curves_by_category_embryo__band_only.png`
- `penetrance_curves_by_category_embryo__line_only.png`
- `penetrance_curves_by_category_embryo__band_line_dots.png`
- `scatter_penetrance_by_genotype_embryo_smoothed.png`
- `scatter_penetrance_by_genotype_embryo__band_line_dots.png`

## Guidance for the next agent
- Do not change the entrypoint contract unless necessary; keep `01_compute_envelope_and_penetrance.py` runnable
- Prefer changing visual behavior in `penetrance_plots.py` and `config.py`
- Prefer changing threshold or summary behavior in `envelope.py`, `calls.py`, or `summaries.py`
- Preserve the current smoothed presentation figures unless the change is explicitly intended to alter the narrative
- If aesthetics are changed, rerun the full pipeline and visually inspect the `presentation/` outputs, especially the smoothed category curve and the combined scatter+penetrance figure
