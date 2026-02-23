# Implementation Plan (MorphSeq) — ROI Discovery via WT-Referenced OT Feature Maps

**(Updated: Phase 0 → Phase 1 → Phase 2.0 → Phase 2.5 ordering with success criteria)**

## Purpose

- Learn contiguous, interpretable regions of importance (ROIs) on a 512×512 canonical embryo grid that explain genotype discriminability (WT vs cep290) using OT-derived template-space feature maps.
- Build a scalable backend for large datasets and a simple, biologist-friendly front end.
- Ensure statistical validity via selection-aware null distributions and stability resampling.
- Explicitly separate:
  - **(A)** computationally stable defaults (robust + fast to run)
  - **(B)** biology-dependent tuning (signal-to-noise, phenotype type, dataset size)

## Inspiration (principles, not full reproduction)

- Fong & Vedaldi 2017 "Meaningful Perturbations":
  https://arxiv.org/pdf/1704.03296
- Fong et al. 2019 "Extremal Perturbations":
  https://arxiv.org/abs/1910.08485

---

## Phase Ordering

| Phase | Name | Goal | Modules |
|-------|------|------|---------|
| **0** | 1D S-bin localization | Fast sanity check: do OT features separate WT/cep290 along the body axis? | `p0_*.py`, `run_phase0.py` |
| **1** | 2D weight-map ROI | Full spatial ROI via L1+TV regularized logistic regression | `roi_trainer.py`, `roi_sweep.py`, `roi_nulls.py`, `roi_api.py` |
| **2.0** | Occlusion validation | Causal check: does masking/preserving the ROI change classification? | `roi_occlusion.py`, `roi_perturbation.py`, `roi_resampling.py` |
| **2.5** | Mask learning | Learn soft ROI mask directly from perturbation objective | `roi_mask_param.py`, `roi_mask_trainer.py`, `roi_mask_objective.py` |

---

## PHASE 0: 1D S-bin Localization (WT-referenced OT)

### Purpose

Before investing in 2D weight-map optimization, answer:
> "Do OT-derived features (cost, displacement, mass) discriminate WT from cep290,
> and if so, WHERE along the rostral-caudal axis?"

### Pipeline Steps

| Step | Module | Gate |
|------|--------|------|
| 0.1 OT map generation | `p0_ot_maps.py` | Maps generated for all samples |
| 0.2 QC + filtering | `p0_qc.py` | Visual check: mean maps not dominated by alignment failures |
| 0.3 S coordinate | `p0_s_coordinate.py` | S_map covers >95% of mask pixels |
| 0.4 S-bin features | `p0_sbin_features.py` | features_sbins.parquet written |
| 0.5 AUROC localization | `p0_classification.py` | At least one bin has AUROC > 0.65 |
| 0.6 Dynamics (V1) | `p0_sbin_features.py` + `p0_classification.py` | Compare cost-only vs dynamics models |
| 0.7 Interval search | `p0_interval_search.py` | Selected interval passes sanity checks |
| 0.8 Nulls + stability | `p0_nulls.py` | Permutation p < 0.05, bootstrap interval endpoints stable |

### Data Contract (Phase 0 Extension)

Phase 0 FeatureDataset adds to the base contract:
- `features.zarr/optional/S_map_ref` — (512,512) float32, S ∈ [0,1]
- `features.zarr/optional/tangent_ref` — (512,512,2) float32
- `features.zarr/optional/normal_ref` — (512,512,2) float32
- `manifest.json` adds: `phase`, `stage_window`, `OT_params_hash`, `S_orientation`

### Channel Sets

| Set | Channels | C |
|-----|----------|---|
| `V0_COST` | cost_density | 1 |
| `V1_DYNAMICS` | cost_density, disp_u, disp_v, disp_mag, delta_mass | 5 |

### S-Bin Features (per sample per bin)

| Feature | V0 | V1 | Definition |
|---------|----|----|------------|
| cost_mean | x | x | mean cost_density in bin |
| disp_mag_mean | | x | mean displacement magnitude |
| disp_par_mean | | x | mean displacement · e_parallel |
| disp_perp_mean | | x | mean displacement · e_perp |

### Interval Selection Rules

- **Parsimony (default):** smallest interval within ε of best AUROC
- **Penalized:** maximize AUROC − γ·(len/K)

### Null Tests (selection-aware)

1. **Permutation null (max AUROC_k):** embryo-level label permutation, record max AUROC across bins
2. **Permutation null (best interval):** full interval search under permuted labels
3. **Bootstrap stability:** embryo-level within-class resampling → AUROC CI bands + interval endpoint distributions

### Visualization Deliverables

| Figure | What it shows |
|--------|---------------|
| A1–A3 | Mean cost density: WT, cep290, difference |
| A4+ | Smoothed contour versions at σ=1,2,4 |
| B1–B3 | Mean displacement quiver fields (V1 only) |
| C1 | S coordinate map on reference |
| D1 | AUROC vs S-bin (per feature) with bootstrap CI |
| D2 | Logistic coefficient profile |
| E1 | Interval search results |
| F1–F2 | Permutation null histograms |
| F3 | Bootstrap interval stability |

### Success Criteria (Phase 0)

- [ ] OT maps generated for all samples without errors
- [ ] QC gate passed: mean maps are biologically plausible, not alignment artifacts
- [ ] At least one S-bin achieves AUROC > 0.65 (cost channel)
- [ ] Selected interval passes only-interval / drop-interval sanity checks
- [ ] Permutation p-value < 0.05 for max AUROC_k
- [ ] Bootstrap interval endpoints have std < 2 bins
- [ ] All visualization deliverables generated

---

## PHASE 1: 2D Weight-Map ROI (sections A–G from original plan)

### A) DATA CONTRACT (finalize before model code)

#### FeatureDataset = standardized on-disk source of truth

**Format:** Zarr (arrays) + Parquet (metadata) + JSON manifest (provenance + rules)

#### Directory

    results/<date>/roi_feature_dataset_<tag>/
      manifest.json
      metadata.parquet
      features.zarr/
        X/            (N, 512, 512, C) float32
        y/            (N,) int {0,1}
        mask_ref/     (512,512) bool/uint8
        qc/
          total_cost_C/    (N,) float32
          outlier_flag/    (N,) bool

#### Manifest (must include; hard validation)

- canonical_grid: 512×512
- channel_schema: names + definitions + units
- QC rules: IQR outlier filter on total_cost_C (1.5×IQR), logged, never deleted
- split_policy: group_key = embryo_id (MANDATORY; prevents leakage)
- class_balance_strategy (MANDATORY): sklearn balanced class_weight, TRAIN fold only
- chunking/compression spec for X

### B) DATA INGESTION (streamable + deterministic)

- Chunk X along N; keep 512×512 tiles contiguous: chunk (8,512,512,C) or (16,512,512,C)
- Loader supports full-batch (small N) and minibatch (large N).
- Filtering is deterministic via qc/outlier_flag.
- CV splits grouped by embryo_id; fold-local class weights.

### C) TV DEFINITION (explicit boundary behavior)

- TV defined over edges; include edge (p,q) only if both inside mask_ref.
- No zero-padding outside embryo.
- Boundary pixels have fewer valid neighbors ("reduced-degree boundary").
- Diagnostics: boundary_fraction of ROI.

### D) MODEL TRAINING (JAX backend)

- JAX + Optax; jit train_step
- params: w_low (learn_res×learn_res×C), b
- w_full = bilinear_upsample(w_low → 512×512×C)
- logits: <X, w_full> + b
- loss: class-weighted logistic + λL1(w_low) + μTV(w_low)
- Objective logs: logistic_loss_raw, l1_raw, tv_raw, l1_weighted, tv_weighted, total_objective

### E) λ/μ SWEEP + DETERMINISTIC SELECTION

- Coarse λ×μ grid
- Selection: Pareto knee (recommended) or ε-best
- Complexity metrics: area_fraction, n_components, boundary_fraction
- Store full sweep table + selection metadata

### F) NULLS + STABILITY

- **NULL 1:** embryo-level label permutation (selection-aware)
- **NULL 3:** bootstrap stability at FIXED (λ,μ)
- Bootstrap: resample embryos, fit at fixed params, measure IoU

### G) BIOLOGIST-FACING FRONT END

```python
morphseq.roi.fit(genotype="cep290", features="cost", ...)
morphseq.roi.plot(run_id, style="filled_contours", overlays=["outline"])
morphseq.roi.report(run_id)
```

### Success Criteria (Phase 1)

- [ ] FeatureDataset validates (schema, shapes, QC arrays, split policy)
- [ ] JAX trainer converges (objective decreasing, loss < initial)
- [ ] Sweep produces non-trivial ROI (area_fraction > 0.01 and < 0.5)
- [ ] Permutation p < 0.05
- [ ] Bootstrap IoU > 0.5 (median)
- [ ] boundary_fraction < 0.3 (no edge artifacts)
- [ ] `roi.fit()`, `roi.plot()`, `roi.report()` run end-to-end

---

## PHASE 2.0: Occlusion Validation

### Purpose

Causal check: does masking/preserving the ROI change classification?

### Protocol

- **Delete perturbation:** X_perturbed = (1-mask)*X + mask*baseline
- **Preserve perturbation:** X_perturbed = mask*X + (1-mask)*baseline
- Baseline = WT spatial mean (computed from in-bag training samples only)
- Bootstrap OOB evaluation: fit on in-bag, evaluate on OOB

### Metrics

- delete_gap = AUROC(original) − AUROC(delete_ROI)
- preserve_gap = AUROC(preserve_ROI) − AUROC(random_preserve)

### Success Criteria (Phase 2.0)

- [ ] delete_gap > 0.05 (removing ROI hurts classification)
- [ ] preserve_gap > 0.05 (ROI alone is sufficient)
- [ ] Bootstrap CI for delete_gap does not include 0

---

## PHASE 2.5: Mask Learning

### Purpose

Learn a soft ROI mask directly by optimizing a perturbation objective:
> "Find the smallest mask m such that preserving m·X keeps classification intact."

### Model

- Parameterize m via sigmoid of learnable low-res field
- Dual objective: score(preserve_m) − score(delete_m)
- Penalties: λ_area * ||m||_1 + μ_TV * TV(m) + entropy term
- Anti-cheating: circular jittering during training

### Success Criteria (Phase 2.5)

- [ ] Learned mask has IoU > 0.3 with Phase 1 weight-map ROI
- [ ] Learned mask is more compact (smaller area_fraction) than Phase 1 ROI
- [ ] Delete gap with learned mask >= delete gap with Phase 1 ROI

---

## File Inventory

### Phase 0 modules (NEW)

| File | Purpose |
|------|---------|
| `roi_config.py` | Extended: Phase0RunConfig, Phase0FeatureSet, S-bin/interval/null configs |
| `roi_feature_dataset.py` | Extended: Phase0FeatureDatasetBuilder with S_map_ref, basis vectors |
| `p0_ot_maps.py` | OT map generation (fixed WT ref → all targets) |
| `p0_qc.py` | QC suite: IQR filter, histograms, worst-sample montage, gate check |
| `p0_s_coordinate.py` | S coordinate from centerline + local tangent/normal basis |
| `p0_sbin_features.py` | S-bin feature table builder (v0: cost, v1: dynamics) |
| `p0_classification.py` | AUROC per bin + grouped-CV logistic regression |
| `p0_interval_search.py` | 1D contiguous interval search + sanity checks |
| `p0_nulls.py` | Selection-aware permutation null + bootstrap stability |
| `p0_viz.py` | Full Phase 0 visualization suite |
| `run_phase0.py` | Phase 0 orchestrator |

### Phase 1 modules (existing, complete)

| File | Purpose |
|------|---------|
| `roi_trainer.py` | JAX trainer: L1+TV regularized logistic regression |
| `roi_tv.py` | Total variation with mask-aware boundaries |
| `roi_sweep.py` | λ/μ sweep + Pareto/ε selection |
| `roi_nulls.py` | Label permutation + bootstrap stability |
| `roi_loader.py` | Streaming loader with grouped CV splits |
| `roi_api.py` | Biologist-facing API (fit/plot/report) |

### Phase 2.0 modules (existing, complete)

| File | Purpose |
|------|---------|
| `roi_occlusion.py` | Bootstrap occlusion validation |
| `roi_perturbation.py` | Perturbation primitives + spatial baseline |
| `roi_resampling.py` | Group-aware bootstrap resampling |

### Phase 2.5 modules (existing, partial)

| File | Purpose |
|------|---------|
| `roi_mask_param.py` | Mask parameterization (sigmoid, upsample, TV, jitter) |
| `roi_mask_trainer.py` | Fixed-model mask learning trainer |
| `roi_mask_objective.py` | Perturbation objectives (stub) |

### Shared

| File | Purpose |
|------|---------|
| `roi_config.py` | All configuration dataclasses |
| `roi_feature_dataset.py` | FeatureDataset builder + validator |
| `roi_viz.py` | Phase 1 visualization (weight maps, Pareto, nulls) |
| `run_roi_discovery.py` | Example driver script |
