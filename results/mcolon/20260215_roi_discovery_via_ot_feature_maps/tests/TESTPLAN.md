# ROI Discovery Test Plan — Phases 0 through 2.5

**Date:** 2026-02-16
**Status:** Draft — execute sequentially; each task has its own test file.

## How to use this plan

Each task below maps to a test file in this `tests/` directory. Run them in order.
Files are structured as standalone pytest modules with synthetic data fixtures.
No real data or GPU required — all tests use tiny grids (16x16 or 32x32) and NumPy fallbacks where possible.

**Key biological prior (cep290):** The cep290 mutant phenotype produces signal
predominantly in the **tail region** of the embryo. Planted-ROI tests embed signal
in the bottom rows of the grid to simulate this. Sanity checks verify that
discovered ROIs concentrate there, not at the head/top.

---

## Phase 0: S-bin Localization + QC Foundation

| # | Task | Test File | What it checks |
|---|------|-----------|----------------|
| 0.1 | OT map generation | `test_p0_01_ot_maps.py` | WT-reference map generation count, provenance metadata, failure handling |
| 0.2 | Outlier detection + QC | `test_p0_02_outlier_qc.py` | IQR thresholds, dropped sample table, QC artifact outputs |
| 0.3 | S coordinate map | `test_p0_03_s_coordinate.py` | S coverage, orientation, tangent/normal validity |
| 0.4 | S-bin feature aggregation | `test_p0_04_sbin_features.py` | Bin completeness, V0/V1 schema, occupancy diagnostics |
| 0.5 | Bin-wise localization | `test_p0_05_bin_localization.py` | Grouped-CV AUROC profile + CI |
| 0.6 | Interval selection | `test_p0_06_interval_search.py` | Deterministic interval search + sanity checks |
| 0.7 | Nulls + stability | `test_p0_07_nulls_stability.py` | Selection-aware permutations + bootstrap endpoint variability |
| 0.8 | Phase 0 handoff | `test_p0_08_handoff_summary.py` | Run artifacts complete for Phase 1 go/no-go review |

## Phase 1: Core Pipeline

| # | Task | Test File | What it checks |
|---|------|-----------|----------------|
| 1.1 | Config construction + validation | `test_p1_01_config.py` | Enums, presets, frozen dataclasses, resolve_lambda/mu |
| 1.2 | FeatureDataset build + validate | `test_p1_02_feature_dataset.py` | Zarr/Parquet round-trip, manifest schema, QC outlier flagging |
| 1.3 | Loader + CV splits | `test_p1_03_loader.py` | Group-aware splits, no embryo leakage, class weights fold-local |
| 1.4 | TV edge construction + computation | `test_p1_04_tv.py` | Edge list correctness, mask-aware boundaries, boundary fraction |
| 1.5 | Trainer (logistic + L1 + TV) | `test_p1_05_trainer.py` | Convergence on planted signal, weight map localization, objective logging |
| 1.6 | ROI extraction | `test_p1_06_roi_extraction.py` | Quantile thresholding, area_fraction, n_components, tail concentration |
| 1.7 | Lambda/mu sweep + selection | `test_p1_07_sweep.py` | Pareto knee, epsilon-best, sweep table completeness, determinism |
| 1.8 | Permutation null (NULL 1) | `test_p1_08_null_permutation.py` | Selection-aware p-value, null AUROC distribution, embryo-level shuffle |
| 1.9 | Bootstrap stability (NULL 3) | `test_p1_09_null_bootstrap.py` | IoU distribution, fixed-(lam,mu), group-aware resampling |
| 1.10 | API integration (fit/plot/report) | `test_p1_10_api.py` | End-to-end smoke test through roi_api.fit() |

## Phase 2.0: Occlusion Validation

| # | Task | Test File | What it checks |
|---|------|-----------|----------------|
| 2.1 | Perturbation + baseline | `test_p2_01_perturbation.py` | Spatial baseline from WT only, preserve/delete operator shapes, fold safety |
| 2.2 | Occlusion evaluation | `test_p2_02_occlusion_eval.py` | Logit gaps, AUROC deltas, single-class NaN handling |
| 2.3 | Bootstrap occlusion (OOB) | `test_p2_03_occlusion_bootstrap.py` | OOB-only evaluation, degenerate OOB handling, threshold sensitivity |
| 2.4 | Resampling helpers | `test_p2_04_resampling.py` | iter_bootstrap_groups, stratification, OOB empty/single-class flags |
| 2.5 | Integration fixes (ADDENDUM A) | `test_p2_05_integration_fixes.py` | compute_logits shared utility, channel_names in TrainResult, config import |

## Phase 2.5: Learned Mask (Fixed Model)

| # | Task | Test File | What it checks |
|---|------|-----------|----------------|
| 2.5.1 | Mask parameterization | `test_p25_01_mask_param.py` | sigmoid conversion, upsample, TV loss, jitter shift |
| 2.5.2 | Mask objective (dual) | `test_p25_02_mask_objective.py` | Preserve > delete on planted ROI, score monotonicity |
| 2.5.3 | Mask trainer (fixed model) | `test_p25_03_mask_trainer.py` | Mask recovers planted ROI, objective decreases, jitter stability |

---

## Running

```bash
# From the roi_discovery directory:
cd results/mcolon/20260215_roi_discovery_via_ot_feature_maps

# Run all tests:
pytest tests/ -v

# Run a single phase:
pytest tests/test_p1_*.py -v

# Run a single task:
pytest tests/test_p1_04_tv.py -v
```

---

## CRITICAL TESTING PRINCIPLE: Magnitude vs. Signed Discriminative Power

### The Problem

Logistic regression weights can have **high magnitude but low discriminative power** when
opposing positive and negative weights cancel out. This happens with weak regularization.

### Example (Empirical Data from test_p1_05_trainer)

**Ground truth** (actual logit contributions):
- Tail discrimination: 63.7 (signal planted here)
- Head discrimination: 1.4 (just noise)
- **True winner**: TAIL (46× stronger)

**Wrong metric** (`mean(|w|)` — average magnitude):
- Tail: 0.041
- Head: 0.206 (5× higher!)
- **Predicted winner**: HEAD ❌

**Correct metric** (`|mean(w)|` — absolute net effect):
- Tail: 0.041
- Head: 0.020
- **Predicted winner**: TAIL ✓

### Why This Happens

The logit computation is:
```python
logit = sum(X_pixel * w_pixel) + bias
```

So **net signed weight** (mean) determines contribution, not **average size** (magnitude).

**Head weights** (oscillating, high magnitude, low net):
```
[+0.8, -0.7, +0.9, -0.8, ...] → mean(|w|) = 0.8, mean(w) = 0.02
```

**Tail weights** (consistent, moderate magnitude, high net):
```
[+0.3, +0.3, +0.3, +0.3, ...] → mean(|w|) = 0.3, mean(w) = 0.30
```

### Testing Standard

For all weight-based localization tests:
- ✅ Use `abs(mean(weights))` to measure discriminative focus
- ❌ Do NOT use `mean(abs(weights))` — measures activity, not discrimination

### Implications for ROI Visualization

`extract_roi()` uses magnitude thresholding, which is useful for showing "where the model
is active" but can include canceling weights. For biological interpretation:

1. **Magnitude map** (`sqrt(sum(w**2))`) — shows model activity
2. **Signed weight map** (raw `w`) — shows direction of effect
3. **Strong TV regularization** forces smooth, same-sign weights → magnitude ≈ signed effect

See `test_p1_05_trainer.py` and `test_p1_06_roi_extraction.py` for implementation.

---

## Execution files (new)

For day-to-day execution, use the structured task files in:

- `tests/execution/TASK_LIST.md` (quick run checklist)
- `tests/execution/phase0/*.md`
- `tests/execution/phase1/*.md`
- `tests/execution/phase2_0/*.md`
- `tests/execution/phase2_5/*.md`

Each task file includes explicit checks, minimal pseudo-logic, and expected artifacts,
with a dedicated tail-localization inspection step for cep290-relevant ROI outputs.
