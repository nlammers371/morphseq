# MorphSeq Beta Predictor: Visualization Specification

## 1. Purpose

Visualization is a core dependency of this project.

The model should be built so that every major algorithmic choice can be inspected visually before it is trusted numerically.

This document defines the required visualization module and the minimum set of plots / inspectors that must exist.

---

## 2. Module layout

Recommended `viz/` structure:

```text
viz/
  smoothing.py
  transition_bank.py
  matching.py
  prediction.py
  evaluation.py
```

These can later be consolidated or reorganized, but the conceptual separation should remain.

---

## 3. Design principles

- Every plot should answer one concrete debugging question.
- Functions should return figure objects, not immediately display by side effect.
- A notebook should be able to call any visualization function directly.
- Default plots should be lightweight and legible.
- Use the same data objects defined in `data_contract.md`.

---

## 4. Required visualizations by modeling stage

## 4.1 Smoothing

### Must-have functions

- `plot_raw_vs_smoothed_timeseries(...)`
- `plot_latent_trajectory_before_after_smoothing(...)`
- `plot_sg_parameter_sweep(...)`

### Questions these should answer

- Is the smoother removing obvious noise without distorting geometry?
- Are the endpoints behaving sensibly?
- Is the time-scaled SG window consistent across experiments?

---

## 4.2 Arc-length resampling

### Must-have functions

- `plot_arc_length_vs_time(...)`
- `plot_resampled_points_on_trajectory(...)`
- `plot_increment_norm_distribution(...)`

### Questions these should answer

- Is `delta_s` too small or too large?
- Are resampled points visually uniform along the smoothed path?
- Are increment sizes stable enough for a fixed-step predictor?

---

## 4.3 Transition-bank inspection

### Must-have functions

- `plot_transition_windows_for_embryo(...)`
- `plot_history_segments_example(...)`
- `plot_bank_state_density(...)`

### Questions these should answer

- Are transition windows constructed correctly?
- Is the canonical history representation intuitive?
- Where is the bank dense or sparse?

---

## 4.4 Matching diagnostics

### Must-have functions

- `plot_query_and_candidate_neighbors(...)`
- `plot_history_reranking(...)`
- `plot_history_offset_heatmap(...)`
- `compare_default_vs_fast_matching(...)`

### Questions these should answer

- Are the initial candidate neighbors plausible?
- Does history reranking help in the expected situations?
- Is the offset tolerance actually rescuing near-misaligned matches?
- How much accuracy is traded away by the fast matching mode?

---

## 4.5 Local transition-kernel inspection

### Must-have functions

- `plot_local_increment_cloud(...)`
- `plot_sampled_next_steps(...)`
- `plot_jitter_ellipse_or_covariance(...)`

### Questions these should answer

- Are the sampled next-step increments coherent?
- Is the jitter too small or too large?
- Is the local forecast multimodal where expected?

---

## 4.6 Rollout prediction

### Must-have functions

- `plot_prediction_fan(...)`
- `plot_rollout_against_truth(...)`
- `plot_support_diagnostics_along_rollout(...)`

### Questions these should answer

- Does the rollout track the observed future?
- Does the forecast cloud remain on-manifold?
- Does uncertainty widen sensibly when support gets weak?

---

## 4.7 Evaluation plots

### Must-have functions

- `plot_error_vs_horizon(...)`
- `plot_model_comparison_table(...)`
- `plot_error_vs_support(...)`
- `plot_failure_gallery(...)`

### Questions these should answer

- Which model variant is actually better?
- At what horizon does performance break down?
- Are failures caused by sparse support, bad matching, or bad smoothing?

---

## 5. Notebook plan

The visualization layer must be tightly coupled to the notebook walkthrough.

The initial notebook sequence should be:

1. `01_loading_and_smoothing.ipynb`
2. `02_resampling_and_transition_bank.ipynb`
3. `03_matching_and_prediction.ipynb`
4. `04_evaluation.ipynb`

Each notebook must produce both:
- a small number of clean summary figures
- and a few richer inspection figures for debugging

---

## 6. Future interactive app

A later app can wrap the visualization primitives defined here.

That future app should be able to:
- load a dataset,
- browse embryos,
- inspect smoothing,
- inspect local matches,
- launch forecasts,
- and explore rollout support diagnostics.

But the app should be built **after** the notebook-first visualization primitives already exist.

---

## 7. Minimum shipping bar

The beta model should not be considered ready until a user can visually inspect:

- one smoothed embryo,
- one resampled embryo,
- one transition-bank window,
- one query’s candidate neighbors,
- one next-step transition cloud,
- and one rollout fan.

If those plots are not trustworthy, the model is not trustworthy.
