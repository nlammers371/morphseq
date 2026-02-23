# PHASE 0 SPEC (MorphSeq) — WT-Referenced OT + Rostral–Caudal (S) Bins

Pilot genotype: cep290 (curvature phenotype)
Primary goal: prove (1) OT mapping + QC is trustworthy and (2) discriminative signal localizes along S.
Secondary goal: show directionality (displacement dynamics) can improve AUROC beyond scalar cost.

## Core principles (locked)

- One fixed source: pre-selected WT reference mask (template on canonical grid).
- Many targets: WT and cep290 embryos from ONE 2 hpf window (single window only for Phase 0).
- Stability first: filtering + visual QC must pass before interpreting any classifier.
- Statistics use UNSMOOTHED features. Smoothing is visualization-only.
- CV and resampling are embryo-level (group by embryo_id).
- Class imbalance handled using existing MorphSeq "balance method" used by prior classifiers (do not re-invent).

## Data Requirements (Phase 0 Implementation)

**Canonical grid:**
- Shape: 256×576 (H×W) at 10.0 µm/px
- Matches `CanonicalGridConfig` from [uot_grid.py](../../src/analyze/optimal_transport_morphometrics/uot_masks/uot_grid.py)
- UOT pipeline handles alignment automatically via `use_canonical_grid=True`

**Stage window:**
- 47–49 hpf developmental window
- One frame per embryo (frame closest to 48.0 hpf selected)

**Sample composition:**
- 10 WT (`genotype == "cep290_wildtype"`)
- 10 mutant (`genotype == "cep290_homozygous"`)
- Excludes `cep290_unknown` and heterozygous

**Reference mask selection:**
- Single WT mask from 47–49 hpf window
- Selected by highest mean IoU among WT cohort (most "typical")
- Visualized and approved via `scripts/s01_select_reference_mask.py`

**Data source:**
- CSV: `results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv`
- Masks: RLE-encoded at raw resolution (2189×1152 typical, varying µm/px)
- Metadata: `embryo_id`, `frame_index`, `genotype`, `predicted_stage_hpf`, physical dimensions

**Yolk masks (required):**
- Canonical alignment uses yolk-based orientation (`use_yolk=True`).
- Yolk masks are loaded from Build02 segmentation outputs via `data_root`.
- Reference implementation: [results/mcolon/20260213_stream_d_reference_embryo](results/mcolon/20260213_stream_d_reference_embryo)

## 0) ARTIFACTS + DATA CONTRACT (before any modeling)

### 0.1 FeatureDataset (single source of truth; streamable on disk)

Location:
```
results/<date>/roi_feature_dataset_phase0_<tag>/
```

Files:
```
manifest.json
metadata.parquet
features.zarr/
  X/             (N, 256, 576, C) float32   # per-sample OT-derived feature maps on canonical grid
  y/             (N,) int {0,1}
  mask_ref/      (256, 576) bool/uint8
  qc/
    total_cost_C/    (N,) float32
    outlier_flag/    (N,) bool
  optional/
    S_map_ref/       (256, 576) float32 in [0,1]   # rostral→caudal coordinate map in template space
    tangent_ref/     (256, 576, 2) float32 (optional) # local unit tangent e_parallel at each pixel
    normal_ref/      (256, 576, 2) float32 (optional) # local unit normal e_perp at each pixel
```

### manifest.json must declare (hard validation)

- canonical_grid: 512×512
- stage_window: one 2 hpf bin (explicit start/end)
- channel_schema (C channels) with exact definitions
- OT_params_hash + OT solver version
- reference_mask_id
- QC rules: IQR multiplier=1.5 on total_cost_C; logging requirements
- split_policy: group_key=embryo_id
- class_balance_strategy: "morphseq_balance_method" (existing utility; fold-local weights preferred)
- smoothing_policy: "visualization-only; stats computed on unsmoothed features"
- S orientation: S=0 head, S=1 tail

### metadata.parquet required columns

- sample_id, embryo_id, snip_id
- label_int (0 WT, 1 cep290), label_str
- stage/timepoint fields (explicit)
- total_cost_C, qc_outlier_flag
- provenance tags (batch/experiment/session)

## 1) BUILD OT MAPS (source fixed, targets many)

### 1.1 Inputs

- mask_ref (template WT mask)
- per-sample aligned mask_target on canonical grid (WT + cep290)

### 1.2 OT mapping exports per sample (template space)

For each sample i, run unbalanced OT (fixed params) ref → target, export:
- cost_density c_i(x,y)        (scalar map)
- displacement field d_i(x,y)  = (u,v) (vector map)
- delta_mass Δm_i(x,y)         (scalar map; unbalanced OT)
- total_cost_C_i               (scalar)

Store at least these channels in X (Phase 0 base):
```
Channel set v0 (minimum):
  C=1: cost_density

Channel set v1 (add dynamics; after v0 stable):
  + disp_u, disp_v
  + disp_mag = sqrt(u^2+v^2)
  + delta_mass

Optional derived (later):
  + divergence (finite diff of d in template grid)
```

## 2) QC + OUTLIER FILTERING (must run before classification)

### 2.1 Primary filter (locked)

- Compute total_cost_C_i for all samples.
- IQR filter (1.5×IQR): flag outliers.
- Do NOT delete; store qc_outlier_flag and a dropped_samples table.

### 2.2 QC deliverables (required figures; these are "gates")

- QC-1: histogram/violin of total_cost_C (before/after filtering)
- QC-2: montage (or grid) of top-N highest-cost samples with their cost maps
- QC-3: summary table of dropped samples (embryo_id, snip_id, C, reason)

Gate to proceed:
- Post-filter mean maps are not dominated by obvious alignment failures.

## 3) VISUALIZATIONS (Phase 0's main debugging instrument)

All visual smoothing is for display only.

### 3.1 Cost density maps (template space)

- Fig A1: mean cost density (WT)
- Fig A2: mean cost density (cep290)
- Fig A3: difference (cep290 − WT)
- Fig A4/A5/A6: filled-contour versions with Gaussian smoothing (sigma grid: e.g. 1, 2, 4)

Preferred clean style (for all scalar maps):
- embryo outline (mask_ref boundary)
- filled contours (smoothed scalar field)
- thin contour lines on top (same levels)
- consistent color scale across WT/mutant/diff

### 3.2 Displacement dynamics maps (template space)

- Fig B1: mean displacement vector field (WT) (quiver downsampled)
- Fig B2: mean displacement vector field (cep290)
- Fig B3: difference vector field (cep290 − WT)

Optional:
- scalar map of mean displacement magnitude (|d|) with same contour style

## 4) DEFINE S COORDINATE + BIN INTO K SEGMENTS

### 4.1 Compute S_map_ref once (template coordinate map)

- Fit reference spline/centerline on the WT template.
- Produce S_map_ref(x,y) in [0,1] for pixels in mask_ref.
- Orientation convention fixed: S=0 head, S=1 tail.
- Store S_map_ref in features.zarr/optional.

### 4.2 Precompute local basis in template (enables direction features)

- e_parallel(x,y): unit tangent direction at S(x,y)
- e_perp(x,y): unit normal
- Store as tangent_ref/normal_ref (optional but recommended once directionality is enabled).

### 4.3 Bin definition

- Choose K=10 initially (also run K=20 as robustness check).
- Bin k: S ∈ [k/K, (k+1)/K)

## 5) BUILD THE S-BIN FEATURE TABLE (models consume this, not images)

Output table (tiny; fast; reusable):
```
results/<date>/roi_feature_dataset_phase0_<tag>/features_sbins.parquet
```

Rows: N×K (sample-by-bin)

Columns (minimum v0):
- sample_id, embryo_id, snip_id, label_int, stage_window, qc_outlier_flag
- k_bin, S_lo, S_hi
- cost_mean

Add dynamics v1 (after v0 stable):
- disp_mag_mean
- disp_par_mean = mean(d · e_parallel)
- disp_perp_mean = mean(d · e_perp)

Optional later:
- delta_mass_mean
- divergence_mean

Important:
- These per-bin summaries are computed on UNSMOOTHED maps.
- Smoothing may be applied to plotted curves only.

## 6) CLASSIFICATION + AUROC LOCALIZATION (the "signal exists here" proof)

All analyses run:
- (A) with all samples
- (B) with qc_outlier_flag excluded

Report both to demonstrate filtering impact.

### 6.1 Univariate AUROC per bin (fast, interpretable; required)

For each bin k:
- AUROC_k(cost_mean)
- AUROC_k(disp_mag_mean) (once v1 enabled)
- AUROC_k(disp_par_mean), AUROC_k(disp_perp_mean) (once v1 enabled)

Deliverables:
- Plot: AUROC vs S-bin (one curve per feature)
- Optional display smoothing of AUROC curve (visualization only)

### 6.2 Multivariate logistic regression across bins (still simple; required)

Model:
- Input vector for each sample: X_i = [feature(i,0), ..., feature(i,K-1)]
- CV: grouped by embryo_id
- Class imbalance: use MorphSeq balance method (existing utility)

Outputs:
- overall CV AUROC
- coefficient profile over bins (interpretability)
- compare models:
  - cost-only
  - dynamics-only (disp features)
  - cost + dynamics (expected best)

### 6.3 Directionality-as-2D feature per bin (optional but valuable once v1 works)

Per bin k, define 2D vector feature:
- v_i,k = (disp_par_mean, disp_perp_mean)
- Train tiny logistic regression for each bin k separately with embryo-level CV, report AUROC_k(dir).

## 7) 1D PATCH (INTERVAL) SEARCH ON S (automated localization)

Goal: find contiguous region I=[a..b] that best discriminates.

### Procedure (required)

For each interval I:
- define an interval feature:
  - simplest: mean over bins in I (per feature)
  - or concatenate bins in I (still small)
- score via grouped-CV AUROC (embryo_id)

Deterministic selection rule (locked and logged):
- Option A (recommended): smallest interval within ε of best AUROC
- Option B: add length penalty gamma and maximize AUROC - gamma×(len/K)

Sanity checks (required):
- Only-interval: keep bins in I → AUROC should stay high
- Drop-interval: zero bins in I → AUROC should drop

Deliverables:
- Plot: best AUROC vs interval length
- Plot: selected interval highlighted on S axis
- Table: top-10 intervals + scores

## 8) NULLS + STABILITY (selection-aware; required)

### 8.1 Permutation null (embryo-level; selection-aware)

- Permute labels at embryo_id unit.
- Repeat the SAME selection procedure:
  - if reporting max AUROC_k: record max under permutation
  - if reporting best interval: record best-interval AUROC under permutation
- p-value computed against selected statistic (no leakage / no "fixed lambda" trap).

### 8.2 Bootstrap stability

- Bootstrap embryos within class.
- Recompute:
  - AUROC_k curves (confidence bands)
  - selected interval endpoints distribution
  - interval overlap stability metric

## 9) PHASE 0 "DONE" CRITERIA (gates to Phase 1 2D ROI)

Phase 0 is done when:
- Visual QC plots show coherent WT vs mutant differences post-filtering.
- AUROC_k has stable peaks (not flat noise) and is robust under bootstrap.
- Best interval is stable (endpoints don't wander wildly) and sanity checks behave as expected.
- Adding dynamics (disp-based features) provides equal or improved AUROC vs cost-only, and the directionality plots look biologically sensible (not boundary artifacts).

### Note on future extensibility (mention only, not implement now)

- Regionizer abstraction:
  - today: uniform S bins (K segments)
  - future: anatomical parcels (head/trunk/tail/yolk/etc.) as drop-in replacements
- The same downstream pipeline applies: per-region summaries → AUROC → patch search → nulls.

## Implementation order (Phase 0)

0. Build FeatureDataset contract + validator (manifest + zarr + parquet)
1. Generate OT maps (cost + displacement) and qc scalars; write FeatureDataset
2. Run QC plots + IQR filtering report (must look sane)
3. Compute S_map_ref (+ tangent/normal if enabling directionality)
4. Build features_sbins.parquet (cost-only first)
5. Run AUROC_k (cost) + grouped-CV logistic (cost)
6. Enable dynamics features (disp_mag, disp_par, disp_perp) and rerun models
7. Run interval search + sanity checks
8. Run permutation null + bootstrap stability (selection-aware)
