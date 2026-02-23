# UOT Phase 2: Implementation Tracking

CPU MVP pipeline complete. Pluggable backend architecture in place with working `POTBackend`. JAX (0.6.2) and ott-jax already installed. Goal: add OTT GPU backend with strict concordance, thread backend selection through the timeseries path, improve viz, and build analysis layers. Structured for parallel agent execution with atomic commits.

---

## Commit Plan (atomic, in order)

1. `feat(ot): add ott backend + optional import wiring`
2. `test(ot): add pot-vs-ott concordance suite`
3. `feat(uot): pass backend through timeseries + loader plumbing`
4. `feat(viz): enforce support-mask/NaN contract`
5. `feat(ref): add reference field + deviation metrics`

Each commit only after tests pass for that scope.

**Pre-commit 1 spike:** Run a tiny POT-vs-OTT parity test (identity + translation) to validate ott-jax API behavior with installed versions (jax==0.6.2, jaxlib==0.6.2) before broad implementation.

**Agent protocol:** Each workstream agent gives status updates and makes local git commits. Review via `git diff` to track each implementation.

## Workstream Directories

- `stream_a_ott_backend/` — OTT backend + concordance tests
- `stream_b_backend_plumbing/` — Backend parameter threading through timeseries
- `stream_c_viz_contract/` — Visualization NaN contract enforcement
- `stream_d_reference_embryo/` — Reference embryo + deviation metrics
- `stream_e_future/` — Future workstreams (feature-over-time, difference testing)
- `stream_f_feature_development/` — OT output contract + feature compaction spikes

Each directory has a `DECISIONS.md` tracking rationale, links to plots/data, issues encountered, and parameter values chosen with justification.

---

## Workstream A: OTT Backend + Concordance (Commits 1-2)

**Status:** COMPLETE (committed as 97a28b06)

### Files Created
- `src/analyze/utils/optimal_transport/backends/ott_backend.py`
- `src/analyze/utils/optimal_transport/backends/tests/__init__.py`
- `src/analyze/utils/optimal_transport/backends/tests/test_ott_backend.py`

### Files Modified
- `src/analyze/utils/optimal_transport/backends/__init__.py` — conditional `OTTBackend` import
- `src/analyze/utils/optimal_transport/__init__.py` — conditional re-export for discoverability

### Key Design Decisions

1. **No cost-matrix scaling** — POTBackend normalizes weights (a,b), not cost matrix. OTTBackend matches this exactly.
2. **No hard GPU assert at import** — `OTTBackend` works on CPU too. Device selection is runtime, not import-time.
3. **`m_src`/`m_tgt` diagnostics required** — both backends return these as parity keys in `BackendResult.diagnostics`.
4. **`solve_batch` uses sequential loop, not full vmap** — safer memory profile.

### Parameter Mapping
- `config.epsilon` → ott-jax `epsilon`
- `config.marginal_relaxation` (reg_m) → `tau_a = tau_b = reg_m / (reg_m + epsilon)`
- `config.metric = "sqeuclidean"` → `ott.geometry.costs.SqEuclidean()`

### Concordance Testing

| Metric | Tolerance | Note |
|--------|-----------|------|
| Total transport cost | rtol=5% + atol=1e-3 | atol=1e-3 avoids false failures on identity-like tests |
| Coupling marginals | rtol=10% + atol=1e-3 | Sinkhorn convergence differs |
| Velocity direction | cosine similarity > 0.9 | Flow agreement |
| Mass created % | 5% absolute | Same creation/destruction |
| Mass destroyed % | 5% absolute | Same |
| `m_src`/`m_tgt` diagnostics | present in both | Parity keys |

### Spike Results (eps sweep on canonical grid)

See `stream_a_ott_backend/DECISIONS.md` and `stream_a_ott_backend/spike_test_results/`.

Key findings:
- **eps=1e-4**: POT and OTT agree within 0.21% on cost, nearly identical velocity distributions — **concordance sweet spot**
- **eps=1e-5**: POT has Gibbs kernel underflow on canonical grid (99.96% mass creation/destruction, near-zero cost). OTT handles it correctly (0.59% creation, meaningful cost=34.08)
- POT is unstable at very small epsilon on canonical grid; OTT is more robust in this regime

### 2026-02-08 control rerun (batch-context sanity)
- Added `stream_a_ott_backend/spike_test_results/01_run_control_concordance.py`
- Confirms:
  - `eps=1e-4`: strong POT/OTT concordance on known cross-embryo pair
  - `eps=1e-5`: POT instability reappears; OTT remains stable
  - identity control remains near-zero transport for both backends

---

## Workstream B: Backend Plumbing Through Timeseries (Commit 3, P0)

**Status:** COMPLETE (committed as 97a28b06)

- `run_timeseries.py` — `backend: Optional[UOTBackend]` parameter added, passes through to `run_uot_pair()`
- `frame_mask_io.py` — `data_root` plumbing added to all `load_mask_*` functions with `MORPHSEQ_DATA_ROOT` env var fallback

---

## Workstream C: Visualization Contract Enforcement (Commit 4)

**Status:** COMPLETE (committed as 97a28b06)

### Files Modified
- `src/analyze/optimal_transport_morphometrics/uot_masks/viz.py`

### Files Created
- `src/analyze/optimal_transport_morphometrics/uot_masks/tests/test_viz.py`

### Key Functions
- `apply_nan_mask(field, support_mask)` — NaN for non-support pixels
- `plot_uot_summary(uot_result, ...)` — 4-panel summary with NaN masking + numeric annotations
- `plot_velocity_histogram(uot_result, ...)`
- `write_diagnostics_json(uot_result, output_path)`

---

## Workstream D: Reference Embryo + Deviation (Commit 5)

**Status:** COMPLETE (committed as 97a28b06)

### Files Created
- `src/analyze/optimal_transport_morphometrics/uot_masks/reference_embryo.py`
- `src/analyze/optimal_transport_morphometrics/uot_masks/tests/test_reference_embryo.py`

### Functions
- `ReferenceField` / `ReferenceTimeseries` dataclasses
- `build_reference_field(uot_results, frame_pair, method="mean")`
- `compute_deviation_from_reference(uot_result, reference)` → RMSE, cosine similarity, residuals
- `deviation_timeseries(embryo_results, reference)` → tidy DataFrame

### 2026-02-08 execution refresh (actual artifacts now present)
- Built cohort manifests with explicit selection policy:
  - `3` WT references + `3` held-out WT + `20` mutants
  - 2-hour bins from `24` to `48` hpf
- Completed batch OT export:
  - run_id `phase2_24_48_ott_v1`
  - `313/313` transitions successful, `0` failed
- Built per-bin WT reference fields (`12` bins), deviation tables/plots, raw-field PCA outputs,
  and difference/classification/clustering summaries under
  `stream_d_reference_embryo/`.

---

## Yolk Alignment Fix — COMPLETE

**Problem:** Silent yolk fallback — when `align_mode="yolk"` but yolk masks aren't loaded, the aligner silently falls back to embryo centroid, producing misaligned embryos.

**Why yolk alignment matters:** Embryos must be aligned by yolk centroid (not whole-embryo centroid) so that morphological differences in non-yolk regions (tail, head) are captured as real transport rather than alignment artifact. Without yolk alignment, two embryos with identical shape but different yolk positions would show spurious velocity fields.

**Fixes applied:**
1. `uot_grid.py` — `CanonicalAligner.align()` now raises `ValueError` when `use_yolk=True` but yolk is None/empty (no silent fallback)
2. `run_transport.py` — `run_from_csv()` accepts `data_root` and passes to `load_mask_pair_from_csv()`; `__main__` block accepts `--data-root` CLI arg
3. `calibrate_marginal_relaxation.py` — `calibrate_on_identity()` accepts `data_root` and passes to `load_mask_from_csv()`
4. `benchmark_resolution.py` — `benchmark_downsample()` accepts `data_root` and passes to `load_mask_pair_from_csv()`
5. `viz_field_comparison.py` (spike test) — passes `data_root=morphseq_root / "morphseq_playground"` to both `load_mask_from_csv()` calls

**Verified:** Spike test `viz_field_comparison.py` runs end-to-end with `canonical_grid_align_mode="yolk"`. Yolk masks loaded for both test embryos (A05: 35058 px, E04: 23589 px). All 16 existing tests pass.

**Yolk mask location:** `{data_root}/segmentation/yolk_v1_0050_predictions/{date}/{well}_t{time}*.jpg`

---

## Future Workstreams (after commits 1-5)

### Feature-over-Time + Embedding Pipeline
- `timeseries_features.py` — extract scalar features from UOT results
- `field_embedding.py` — PCA on flattened velocity fields, clustering
- `feature_compaction/` — DCT-based compaction experiments and fidelity sweeps
- Compatible with existing `feature_over_time.py` and `spline_fit_wrapper()`

### Difference Testing
- Permutation tests on transport field distances
- Temporal divergence analysis
- Reuse `src/analyze/difference_detection/` framework

---

## Key Existing Files

| Purpose | Path |
|---------|------|
| Backend ABC | `src/analyze/utils/optimal_transport/backends/base.py` |
| POT backend | `src/analyze/utils/optimal_transport/backends/pot_backend.py` |
| OTT backend | `src/analyze/utils/optimal_transport/backends/ott_backend.py` |
| Config/dataclasses | `src/analyze/utils/optimal_transport/config.py` |
| Transport maps | `src/analyze/utils/optimal_transport/transport_maps.py` |
| Metrics | `src/analyze/utils/optimal_transport/metrics.py` |
| Pipeline entry | `src/analyze/optimal_transport_morphometrics/uot_masks/run_transport.py` |
| Timeseries runner | `src/analyze/optimal_transport_morphometrics/uot_masks/run_timeseries.py` |
| Preprocessing | `src/analyze/optimal_transport_morphometrics/uot_masks/preprocess.py` |
| Canonical grid | `src/analyze/optimal_transport_morphometrics/uot_masks/uot_grid.py` |
| Current viz | `src/analyze/optimal_transport_morphometrics/uot_masks/viz.py` |
| Frame I/O | `src/analyze/optimal_transport_morphometrics/uot_masks/frame_mask_io.py` |
| Plotting contract | `results/mcolon/20260121_uot-mvp/PLOTTING_CONTRACT.md` |
| Proven plotting code | `results/mcolon/20260121_uot-mvp/debug_uot_params.py` |
| CEP290 cluster labels | `results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_cluster_labels.csv` |
| Backends __init__ | `src/analyze/utils/optimal_transport/backends/__init__.py` |
| Top-level OT __init__ | `src/analyze/utils/optimal_transport/__init__.py` |
