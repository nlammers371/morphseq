# Recycling Assessment: Legacy Spec and Code

## 1. Bottom line

The legacy effort is useful as:

- infrastructure,
- file layout precedent,
- interface ideas,
- and historical context.

It is **not** useful as the dynamical core for the new beta predictor.

The right strategy is:

- keep the plumbing,
- replace the theory.

---

## 2. Legacy `model_spec.md`

## What transfers well

- explicit document structure
- separation of problem setting, implementation, evaluation, and visualization
- staged build mindset
- strong attention to baselines
- emphasis on interpretability

## What does not transfer

- global SDE-centric theory
- potential / mode / antisymmetric-matrix formalism
- closed-form embryo-level solves
- rate and temperature as core forecast variables
- teacher-forced time-step likelihood as the main training/evaluation frame

### Verdict
Use as a **template for doc organization**, not a model blueprint.

---

## 3. Legacy code

## 3.1 `data/loading.py`

### Reuse level
High.

### Why
- already contains trajectory loading
- already preserves key metadata
- already handles PCA projection and timestamp repair

### Action
Keep and lightly extend.

---

## 3.2 `data/dataset.py`

### Reuse level
Medium.

### Why
- batching ideas and metadata handling are still useful
- current semantics are tied to teacher-forced time-horizon targets

### Action
Refactor rather than discard, but do not keep current task semantics.

---

## 3.3 `eval/predictions.py`

### Reuse level
High.

### Why
- shared predictor interface is still the right idea

### Action
Keep and extend for support diagnostics.

---

## 3.4 `eval/evaluate.py`

### Reuse level
Medium.

### Why
- orchestration pattern still works
- metrics and tasks need simplification and redefinition

### Action
Refactor around geometric rollout evaluation.

---

## 3.5 `eval/metrics.py`

### Reuse level
Medium.

### Why
- modular metric structure is good
- current primary metrics are too tied to Gaussian-likelihood framing

### Action
Keep style, replace center of gravity.

---

## 3.6 `models/particle_filter.py`

### Reuse level
Low to medium.

### Why
- contains useful helper ideas
- architectural assumptions are now misaligned:
  - time-stepped developmental-progress matching
  - speed ratios
  - recruitment tied to time-centric progression logic

### Action
Mine utilities only. Do not make this the base class for the beta predictor.

---

## 3.7 `viz/panels.py`

### Reuse level
Low to medium.

### Why
- useful plotting patterns
- too tied to the old phi0/mode-loading worldview

### Action
Reuse ideas later if convenient, but build new beta visualization primitives first.

---

## 4. Practical repo recommendation

Treat the legacy code as four bins:

### Keep
- `loading.py`
- `predictions.py`

### Refactor
- `dataset.py`
- `evaluate.py`
- `metrics.py`

### Mine utilities only
- `particle_filter.py`
- `panels.py`

### Archive mentally
- old dynamical theory in `model_spec.md`

---

## 5. Risk to avoid

The biggest practical risk is half-reusing the old particle filter and accidentally dragging back in:
- time-centric stepping,
- speed-ratio logic,
- and hidden assumptions about developmental progress.

That would create a confused hybrid system.

Do not do that. Build the new predictor cleanly around:
- smoothing,
- arc-length resampling,
- local transition windows,
- history-aware matching,
- and empirical multimodal rollouts.


## Repo decision

- The new implementation target is `dev/particle_prediction/`.
- Legacy code under `dev/dynamo/` should be treated as donor/reference code only.
- Reusable pieces should be ported selectively rather than extended in place.
