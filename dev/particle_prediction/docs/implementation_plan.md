# MorphSeq Beta Predictor: Implementation Plan

## 1. Purpose

This document converts the beta model spec into an implementation sequence that a human or Codex agent can execute without drifting into unnecessary complexity.

The core principles are:

- keep the predictive core morphology-first,
- make visualization central,
- implement all new code in `dev/particle_prediction/`,
- reuse legacy infrastructure from `dev/dynamo/` only where it still fits,
- and replace the old time-centric dynamical core rather than extending it.

## Current status snapshot

As of the current `dev/particle_prediction/` implementation:

- implemented: loading, SG smoothing, arc-length resampling, transition-window extraction, transition-bank construction, query/task helpers, ordered-history matching, fast-summary matching, tangent-aligned one-step kernel sampling, and the corresponding smoothing / bank / matching / one-step visualization helpers
- partially implemented: notebook walkthroughs, with `01` and `02` materially populated and `03` / `04` still mostly scaffolded
- not yet implemented: rollout prediction, `eval/` modules, rollout/evaluation visualizations, baseline suite, and notebook-backed evaluation workflow

This plan below remains the source of truth, but the immediate work queue should prioritize the missing pieces from the current state rather than repeating already-landed milestones.

---

## Immediate priority roadmap

## Priority 1 — rollout predictor and result container

### Why first

The current code has a good one-step local kernel but stops before the core beta user-facing task: multi-step forecast rollouts.

### Required deliverables

- extend `models/local_transition_pf.py` from one-step prediction to `n_steps` particle rollout
- add canonical rollout result containers in `eval/predictions.py` or an equivalent documented location
- update particle histories after each step
- surface per-step support diagnostics, not just one-step summaries

### Done when

- snapshot and history queries can both roll forward for configurable horizons
- outputs include per-horizon means, covariance summaries, and particle clouds
- weak-support behavior is visible rather than silently collapsed

## Priority 2 — rollout visualization

### Why second

The docs make visual inspection a gating requirement, and rollout behavior cannot be trusted without it.

### Required deliverables

- add `plot_prediction_fan(...)`
- add `plot_rollout_against_truth(...)`
- add `plot_support_diagnostics_along_rollout(...)`
- keep these in notebook-friendly figure-returning helpers

### Done when

- one can inspect both successful and failing rollouts directly from notebook `03`

## Priority 3 — evaluation package and baselines

### Why third

Once rollouts exist, the next bottleneck is comparative evaluation. The docs explicitly require interpretable geometry-first reporting and early baselines.

### Required deliverables

- create `eval/metrics.py`
- create `eval/evaluate.py`
- implement persistence and linear extrapolation baselines
- implement no-history, ordered-history, and fast-summary local-model variants as comparable evaluators
- report endpoint error by horizon, ADE, and optional truth-in-cloud distance

### Done when

- a single evaluation runner can compare all planned baselines on held-out tasks

## Priority 4 — evaluation visualization

### Why fourth

Numbers alone are not enough for this project, especially for multimodal forecast behavior.

### Required deliverables

- add `viz/evaluation.py`
- implement `plot_error_vs_horizon(...)`
- implement `plot_model_comparison_table(...)`
- implement `plot_error_vs_support(...)`
- implement `plot_failure_gallery(...)`

### Done when

- evaluation outputs can be read both numerically and visually without custom ad hoc plotting

## Priority 5 — notebook completion

### Why fifth

The notebook sequence is meant to be the primary onboarding and debugging path, and it is currently incomplete in the highest-value stages.

### Required deliverables

- convert notebook `03_matching_and_prediction.ipynb` from scaffold to executable walkthrough
- convert notebook `04_evaluation.ipynb` from scaffold to executable walkthrough
- ensure the notebook sequence reflects the actual package API rather than legacy placeholders

### Done when

- a contributor can run notebooks `01` through `04` in order and inspect the full pipeline

## Priority 6 — hardening and contract-alignment cleanup

### Why sixth

Gap handling is now critical and should be made more explicit in tests, diagnostics, and docs-driven checks.

### Required deliverables

- expand tests around hard gaps, interpolatable gaps, and segment-boundary filtering
- add contract checks that window lineage and gap flags remain consistent through bank construction
- make sure package exports and docstrings reflect the canonical gap-aware objects

### Done when

- gap-aware behavior is treated as first-class, not incidental

---

## 2. Proposed repo layout

Recommended package layout inside `morphseq/dev/particle_prediction/` (new implementation target):

```text
data/
  loading.py              # existing; lightly extended
  smoothing.py            # new
  resampling.py           # new
  transition_bank.py      # new
  dataset.py              # refactor for query/task datasets

models/
  local_transition_pf.py  # new predictor core
  matching.py             # new matching utilities
  kernels.py              # optional shared kernel functions

eval/
  predictions.py          # keep interface, extend if needed
  metrics.py              # simplify around geometric metrics
  evaluate.py             # refactor for rollout evaluation

viz/
  smoothing.py            # new
  transition_bank.py      # new
  matching.py             # new
  prediction.py           # new
  evaluation.py           # new

notebooks/
  01_loading_and_smoothing.ipynb
  02_resampling_and_transition_bank.ipynb
  03_matching_and_prediction.ipynb
  04_evaluation.ipynb
```

---

## 3. Recycle map from legacy code

## 3.1 Keep with light refactoring

### `data/loading.py`
Keep as the basis for trajectory loading and metadata management.

Additions:
- optional metadata passthrough dict
- utility hooks for smoothing and resampling outputs
- clearer split between raw trajectory objects and derived trajectory objects

### `eval/predictions.py`
Keep the predictor interface and result container idea.

Refactor:
- do not assume Gaussian NLL is the central output
- keep `predicted_mean`, `predicted_cov_diag`, and `forward_samples`
- allow support diagnostics to be attached cleanly

## 3.2 Keep structure, change semantics

### `data/dataset.py`
The current fragment sampler is time-horizon-centric and teacher-forcing-centric.

Refactor into:
- a query dataset for snapshot/history-conditioned prediction tasks
- optional rollout tasks on resampled arc-length trajectories

Do not keep the current horizon sampling semantics.

### `eval/evaluate.py`
Keep the general orchestration pattern.

Refactor to:
- evaluate one-step and rollout geometry in arc-length steps
- report simple interpretable metrics
- support visualization hooks

### `eval/metrics.py`
Keep the modular metric style.

Refactor away from Gaussian NLL as the default.
Beta metrics should focus on distance-based rollout accuracy.

## 3.3 Mine utilities only

### `models/particle_filter.py`
Do not use as the architectural base for the new model.

Mine only:
- reference-bank construction ideas
- particle result summarization
- helper patterns for forward samples / mean / covariance

The current file is too tied to developmental-progress-in-time assumptions.

### `viz/panels.py`
Useful later, but not central for the beta build.
Reuse ideas only where convenient.

---

## 4. Build order

## Milestone 0 — repo scaffolding and docs

### Deliverables
- create new module skeletons
- land the doc set in the repo
- add minimal tests/import checks for new modules

### Done when
- every planned module imports
- notebooks folder exists
- docs clearly point to one source of truth

---

## Milestone 1 — smoothing

### Files
- `data/smoothing.py`
- `viz/smoothing.py`
- notebook `01_loading_and_smoothing.ipynb`

### Core work
- implement SG smoothing in time units
- convert `window_seconds` to an odd frame count using experiment-level `delta_t`
- smooth each latent dimension independently
- expose residual diagnostics

### Required visualizations
- raw vs smoothed latent coordinates over time
- latent trajectory overlays before/after smoothing
- SG parameter sweep for a few embryos

### Tests
- smoothing preserves shape and length
- smoothed trajectories remain finite
- shorter trajectories degrade gracefully

### Done when
- you can look at several embryos and trust the smoothed geometry

---

## Milestone 2 — arc-length resampling

### Files
- `data/resampling.py`
- `viz/transition_bank.py`
- notebook `02_resampling_and_transition_bank.ipynb`

### Core work
- compute cumulative arc length on smoothed trajectories
- resample onto fixed `delta_s`
- preserve mapping back to original times for inspection only

### Required visualizations
- arc length vs original time
- smoothed trajectory with resampled points overlaid
- before/after local segment distributions

### Tests
- cumulative arc length is monotone
- resampled trajectories have expected spacing
- interpolation behaves correctly at endpoints

### Done when
- a single `delta_s` produces visually sensible, comparable morphology steps across experiments

---

## Milestone 3 — transition-bank construction

### Files
- `data/transition_bank.py`
- `data/dataset.py` (refactor)
- `viz/transition_bank.py`

### Core work
- construct transition windows from resampled trajectories
- store:
  - current state
  - next increment
  - canonical ordered history segments
  - optional fast summary features
  - source metadata

### Required visualizations
- examples of extracted windows from a few embryos
- distribution of increment norms
- effect of chosen history length `K`

### Tests
- every window has valid source lineage
- history segments line up with increments correctly
- no off-by-one errors near boundaries

### Done when
- the transition bank is inspectable and obviously correct

---

## Milestone 4 — matching engine

### Files
- `models/matching.py`
- `viz/matching.py`
- notebook `03_matching_and_prediction.ipynb`

### Core work
Implement both matching modes:

#### Default
- local candidate discovery by position
- history-aware reranking with small offset tolerance

#### Optional fast mode
- mean recent position and direction summary matching

### Required visualizations
- query point plus candidate cloud
- ranking before and after history reranking
- heatmap of history mismatch over offset band
- comparison of default vs fast method

### Tests
- local candidate retrieval returns stable results
- reranking changes candidate ordering when histories differ
- offset tolerance behaves as intended

### Done when
- nearest references look biologically sensible in plots

---

## Milestone 5 — local transition kernel

### Files
- `models/kernels.py` or within `local_transition_pf.py`
- `viz/prediction.py`

### Core work
- compute final candidate weights from position/history/class terms
- sample empirical increments
- add tangent-aligned jitter
- expose isotropic fallback mode

### Required visualizations
- local increment cloud at a query point
- sampled next-step predictions
- effect of jitter parameters on spread

### Tests
- weights normalize
- covariance construction is PSD
- jitter scale is numerically stable even for tiny increments

### Done when
- one-step transition clouds look smooth but still data-anchored

---

## Milestone 6 — rollout predictor

### Files
- `models/local_transition_pf.py`
- `eval/predictions.py`
- `viz/prediction.py`

### Core work
- initialize particles from snapshot or fragment
- roll forward for `n_steps`
- update particle history
- return mean, covariance, and forward samples
- expose support diagnostics

### Required visualizations
- prediction fans
- branch-preserving clouds when multimodality exists
- failure cases in sparse regions

### Tests
- rollout length matches request
- particle history updates correctly
- no silent collapse when support is weak

### Done when
- rollout fans are interpretable and stable on handpicked examples

---

## Milestone 7 — evaluation

### Files
- `eval/metrics.py`
- `eval/evaluate.py`
- `viz/evaluation.py`
- notebook `04_evaluation.ipynb`

### Core work
- run one-step and multi-step evaluations
- compare snapshot vs history-conditioned queries
- compare default history matching vs fast summary matching
- compare against simple baselines

### Required visualizations
- error vs horizon
- qualitative pass/fail galleries
- candidate-support diagnostics vs error

### Tests
- metrics are reproducible
- baseline ordering is sensible on sanity checks
- plots match numeric summaries

### Done when
- you can quickly tell whether a code change helped or hurt

---

## Milestone 8 — polish and notebook tutorialization

### Files
- notebooks
- `viz/*`
- doc touch-ups

### Core work
- make the notebook walkthrough coherent
- ensure every modeling step has a visual inspection entry point
- leave hooks for later stage-field and app work

### Done when
- a new contributor can run the notebooks in order and understand the system

---

## 5. Baselines to implement early

Implement these in order:

1. persistence baseline
2. linear extrapolation baseline
3. local transition model without history reranking
4. full local transition model with history reranking
5. optional fast summary-matching ablation

These are cheap sanity anchors and will prevent overinterpreting the full model.

---

## 6. Configuration surface

Keep the initial config small.

Required beta knobs:

- `sg_window_seconds`
- `sg_poly_order`
- `delta_s`
- `history_length`
- `history_offset_tolerance`
- `candidate_k` or `candidate_radius`
- `sigma_pos`
- `sigma_hist`
- `jitter_mode`
- `sigma_parallel`
- `sigma_perp`
- `n_particles`
- optional class-prior settings

Do not add stage-related config yet.

---

## 7. Practical coding rules

- Preserve source lineage on every derived object.
- Keep raw, smoothed, and resampled objects distinct.
- Prefer small composable functions over monolithic classes.
- Every new algorithmic step must have at least one matching visualization helper.
- Default behavior should favor interpretability over maximal cleverness.

---

## 8. Deferred next steps

Do not build these in the first pass:

- stage prediction model
- formal trajectory likelihood
- learned transition network
- interactive app

Leave clean interfaces so they can be added later.
