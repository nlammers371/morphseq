# Notebook Tutorial Plan

These notebooks are intended to be the fastest path to intuition for the beta predictor.

They should be lightweight, visual, and runnable in order.

These notebooks should live under `dev/particle_prediction/notebooks/`.

---

## 01_loading_and_smoothing.ipynb

### Purpose
Inspect the raw data and confirm that SG smoothing behaves sensibly.

### Minimum contents
- load a small subset of trajectories
- plot raw vs smoothed trajectories
- show SG window selection in seconds and corresponding frame counts per experiment
- inspect a few trajectories in latent space before/after smoothing

### Success criterion
You trust the smoothed geometry.

---

## 02_resampling_and_transition_bank.ipynb

### Purpose
Confirm that arc-length resampling and transition-window extraction are correct.

### Minimum contents
- compute cumulative arc length
- overlay resampled points on smoothed trajectories
- inspect several transition windows
- visualize stored history segments and next-step increments

### Success criterion
You trust the transition bank.

---

## 03_matching_and_prediction.ipynb

### Purpose
Understand local neighbor retrieval, history reranking, and rollout behavior.

### Minimum contents
- show position-only candidate retrieval
- show history-aware reranking with offset tolerance
- compare default and fast matching modes
- inspect local increment clouds
- run short rollouts and visualize prediction fans

### Success criterion
You trust why the model chooses the references it chooses.

---

## 04_evaluation.ipynb

### Purpose
Summarize whether the model helps relative to trivial baselines.

### Minimum contents
- one-step error tables
- rollout error vs horizon plots
- example successes
- example failures
- error vs support diagnostic plots

### Success criterion
You can tell whether a change actually improved the predictor.

---

## Future notebooks

These are explicitly later work:

- stage / hpf annotation notebook
- interactive explorer prototype notebook
- broader perturbation-generalization benchmarking notebook
