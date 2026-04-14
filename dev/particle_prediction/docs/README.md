# MorphSeq Dynamo Docs

This folder is the starter documentation set for the beta predictive latent-space model.

## Local working rules

- For work scoped to `dev/particle_prediction/`, treat this doc set as the local source of truth.
- Operate only inside `dev/particle_prediction/` unless a task explicitly says otherwise.
- The correct conda environment for commands in this subtree is `morphseq-env`.
- When running Python or pytest for this subtree, use `conda run -n morphseq-env --no-capture-output ...`.
- Do not inherit environment assumptions from unrelated repo areas when working in `dev/particle_prediction/`.

## Documents

- `model_spec.md` — source of truth for the beta model.
- `implementation_plan.md` — concrete build order, file layout, and recycle map from the legacy code.
- `data_contract.md` — canonical object schemas and invariants.
- `evaluation_plan.md` — simple, interpretable forecast evaluation for the beta.
- `visualization_spec.md` — required visual diagnostics and module layout.
- `codex_agent_guide.md` — guardrails and work packets for Codex agents.
- `recycling_assessment.md` — what to keep, mine, refactor, or ignore from the older effort.
- `notebooks/README.md` — tutorial notebook plan.

## Current design decisions

- Predictive core is **morphology-first**, not time-first.
- Forecasting is performed in **fixed arc-length steps** after smoothing and resampling.
- Absolute stage / hpf is **out of scope for v1** and should be handled later as a post hoc annotation layer.
- Reference matching uses:
  - current position,
  - optional class priors,
  - history-aware reranking with a small offset tolerance,
  - and an optional fast summary method based on recent mean position and direction.
- Transition sampling is empirical and multimodal, with small tangent-aligned Gaussian jitter.
- Visualization is a first-class deliverable at every stage.

## Repo target

- All new implementation code should live in `dev/particle_prediction/`.
- Legacy code in `dev/dynamo/` is reference material only and should not be modified unless a task explicitly says so.
