# MorphSeq Beta Predictor: Codex Agent Guide

## 1. Purpose

This document tells Codex agents how to work on the beta predictor without reintroducing discarded ideas or overengineering the system.

The source of truth is:

1. `model_spec.md`
2. `data_contract.md`
3. `implementation_plan.md`
4. `visualization_spec.md`
5. `evaluation_plan.md`

If a code change conflicts with those docs, the docs win unless a human updates them.

## Implementation target

- All new implementation code lives in `dev/particle_prediction/`.
- Legacy code in `dev/dynamo/` is read-only reference material unless a human explicitly authorizes edits there.
- Reuse by selective porting, not by extending the old architecture in place.

---

## 2. Project rules

## 2.1 Do not reintroduce time into the predictive kernel

The forecasting core operates in fixed arc-length steps. Time metadata may be preserved for visualization and future stage work, but it is not the main forecasting variable in v1.

## 2.2 Do not add a stage model yet

A stage / hpf field is a future extension. Do not make it a hidden dependency of smoothing, matching, or rollout code.

## 2.3 Visualization is not optional

Any new algorithmic module should come with at least one corresponding visualization helper or notebook checkpoint.

## 2.4 Prefer simple interpretable defaults

When choosing between:
- a simple inspectable method and
- a more sophisticated opaque one,

choose the simple one for v1.

## 2.5 Preserve lineage and metadata

Every derived object must retain source embryo / experiment / class information.

---

## 3. What already exists and how to use it

### Reuse directly or lightly refactor
- `data/loading.py`
- `eval/predictions.py`

### Reuse structure, but rewrite semantics
- `data/dataset.py`
- `eval/evaluate.py`
- `eval/metrics.py`

### Mine utilities only
- `models/particle_filter.py`
- `viz/panels.py`

Do not anchor new architecture to the legacy particle filter.

---

## 4. Recommended parallel work packets

## Agent A — smoothing and resampling

### Scope
- `data/smoothing.py`
- `data/resampling.py`
- `viz/smoothing.py`

### Deliverables
- SG smoothing in time units
- arc-length resampling
- visual checks

### Do not do
- stage inference
- Kalman smoothing
- matching logic

---

## Agent B — transition bank and data contracts

### Scope
- `data/transition_bank.py`
- `data/dataset.py`
- contract compliance checks

### Deliverables
- transition-window extraction
- canonical history storage
- optional fast summary features

### Do not do
- rollout predictor
- new evaluation logic

---

## Agent C — matching engine

### Scope
- `models/matching.py`
- `viz/matching.py`

### Deliverables
- local candidate retrieval
- default history reranking with offset tolerance
- optional fast summary matching

### Do not do
- learned embedding search
- stage-aware matching

---

## Agent D — predictor core

### Scope
- `models/local_transition_pf.py`
- maybe shared kernel helpers

### Deliverables
- empirical transition kernel
- tangent-aligned jitter
- particle rollout
- support diagnostics

### Do not do
- formal likelihood modeling
- neural transition models

---

## Agent E — evaluation and viz

### Scope
- `eval/metrics.py`
- `eval/evaluate.py`
- `viz/prediction.py`
- `viz/evaluation.py`
- notebooks

### Deliverables
- distance-based evaluation
- rollout plots
- notebook walkthrough

### Do not do
- complex probabilistic scoring
- app development

---

## 5. Acceptance criteria for any PR

A change is not done unless:

- it matches the current docs,
- it preserves data lineage,
- it has at least one sanity check,
- and it is inspectable in a notebook or visualization function.

If a PR adds logic but no way to inspect it, it is incomplete.

---

## 6. Things agents must ask before changing

If an agent wants to add any of the following, it should stop and ask:

- stage model
- absolute time prediction
- full covariance modeling
- learned transition network
- global trajectory prototypes
- formal class likelihood scoring

Those are all out of scope for the beta.

---

## 7. Good defaults to assume

Unless the docs are updated:

- smoothing = Savitzky–Golay
- forecast step = fixed arc-length increment
- history = ordered recent segments
- default history matcher = offset-tolerant segment comparison
- optional fast matcher = mean position + direction summary
- kernel = empirical increments + tangent-aligned Gaussian jitter
- evaluation = geometric rollout accuracy + visualization
