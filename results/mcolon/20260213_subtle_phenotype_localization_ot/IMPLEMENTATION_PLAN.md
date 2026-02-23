# Implementation Plan: Subtle-Phenotype Localization via WT-Referenced OT

**Reference Documents**:
- `PLAN.md` - Full specification
- **`PLOTTING_CONVENTION.md`** - ⭐ **CRITICAL: READ FIRST** - Defines WT reference grid as coordinate system for ALL visualizations
- **`BATCH_PROCESSING_GUIDE.md`** - Production-ready batch OT export for scaling to AUROC analysis

**Date**: 2026-02-13
**Status**: Ready to implement

---

## ⚠️ CRITICAL CONVENTION ⚠️

**ALL features are plotted on the WT reference grid**, regardless of OT direction.

- **OT direction**: WT → mutant (source=WT, target=mutant)
- **Visualization grid**: WT reference (template space)
- **Rationale**: Standardizes spatial reference frame despite variable mutant morphology

See `PLOTTING_CONVENTION.md` for complete details. **Read this before implementing any visualization code.**

---

## Executive Summary

This implementation plan translates the `PLAN.md` specification into concrete scripts and workflows. We will leverage:
- **Existing OT infrastructure**: UOT grid, canonical alignment, feature extraction
- **Reference embryo selection**: Pre-existing cohort from Stream D (24-48 hpf bins)
- **Plotting utilities**: `viz.py` functions for heatmaps, quivers, cost fields
- **Bootstrap utilities**: `spline_fitting/bootstrap.py` for uncertainty estimation
- **Spline fitting**: MorphSeq's existing spline centerline code

---

## Critical Data Requirements

### WT Controls vs WT Reference

**You need BOTH:**

1. **WT reference embryo (n=1)**:
   - Single WT mask serving as the OT target/template
   - Defines the spatial coordinate system (S bins, canonical grid)
   - **NOT part of the statistical comparison**

2. **WT control embryos (n≥10)**:
   - Additional WT embryos mapped to the reference
   - Provide the WT distribution for comparison against mutants
   - **Required for permutation testing**

3. **Mutant embryos (n≥20)**:
   - cep290 mutants mapped to reference
   - Compared against WT controls

**Why WT controls are mandatory:**
- Permutation tests require shuffling genotype labels across embryos
- Without WT controls, there is no "WT distribution" to compare against
- The reference embryo alone cannot provide a distribution

**Example from UOT MVP:** Stream D uses:
- `reference_wt` (n=3): Template embryos (excluded from binary tests)
- `heldout_wt` (n=10+): WT controls for permutation testing
- `mutant` (n=20+): Test embryos

See: `results/mcolon/20260213_stream_d_reference_embryo/pipeline/06_difference_classification_clustering.py` line 149:
```python
# Primary binary cohort: mutants vs heldout WT (exclude reference WT from this binary test).
binary_df = emb[emb["set_type"].isin(["mutant", "heldout_wt"])].copy()
```

---

## Key Infrastructure Already Available

### 1. Reference Embryo Selection (DONE - MIGRATED)

**Current Location** (as of 2026-02-13):
```
results/mcolon/20260213_stream_d_reference_embryo/
├── pipeline/
│   └── 01_build_cohort_manifest.py  ← Cohort selection script
└── output/
    └── cohort_selection/
        ├── cohort_selected_embryos.csv  ← Reference + heldout WT + mutants
        ├── cohort_bin_frame_manifest.csv  ← Per-bin frame assignments
        ├── cohort_qc_table.csv  ← All candidates with QC metrics
        └── cohort_qc_scatter.png  ← Selection visualization
```

**NOTE**: If files are not found at the above location, search for `01_build_cohort_manifest.py` in `results/mcolon/2026*` directories.

**Already Selected**:
- **3 reference WT embryos** (`set_type=reference_wt`): Ranked by coverage, then curvature
- **3 heldout WT embryos** (`set_type=heldout_wt`): WT controls for statistical comparison  
- **20 mutant embryos** (`set_type=mutant`): cep290 homozygous mutants
- **Selection criteria**: 
  1. Maximize 24-48 hpf bin coverage (2-hour bins)
  2. Among tied coverage, minimize curvature (straighter embryos)
- **HPF bins covered**: 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48 hpf (13 bins total)

**Selection Logic** (from `01_build_cohort_manifest.py`):
```python
# 1. Build QC table for all embryos in window
# 2. Sort by: (n_bins_covered DESC, curvature_median ASC, n_frames DESC, embryo_id ASC)
# 3. Pick top embryos:
#    - reference_wt: n_ref_wt (default 3) - best coverage, lowest curvature
#    - heldout_wt: next n_holdout_wt (default 3) - used in statistical tests
#    - mutant: n_mutants (default 20) - cep290_homozygous
```

**For this pilot**:
- Use **48 hpf** as reference timepoint (single 2 hpf window, matching tolerance `±1.25 hpf`)
- Select **ONE** reference WT embryo from `reference_wt` cohort at 48 hpf (highest rank)
- Select **≥10** heldout WT controls from `heldout_wt` cohort at 48 hpf  
- Select **≥20** mutants from `mutant` cohort at 48 hpf
- OT parameters already tuned: `epsilon=1e-4`, `marginal_relaxation=10.0`

### 2. UOT Grid Configuration (DONE)

**Canonical Grid Settings**:
- **Grid shape**: 256 × 576 pixels (H × W)
- **Pixel size**: 10.0 μm/pixel
- **Alignment mode**: "yolk" (yolk-based anchoring)
- **Coordinate system**: Image coordinates (origin top-left, +y down)

**Source**: `/home/user/morphseq/src/analyze/optimal_transport_morphometrics/uot_masks/uot_grid.py`

### 3. Existing Plotting Functions (READY TO USE)

**Location**: `/home/user/morphseq/src/analyze/optimal_transport_morphometrics/uot_masks/viz.py`

**Available Functions**:
- `plot_uot_cost_field()` - Cost density heatmaps (Section 1.A)
- `plot_uot_quiver()` - Displacement vector fields (Section 1.B)
- `plot_uot_creation_destruction()` - Mass creation/destruction heatmaps (Section 1)
- `plot_uot_summary()` - 4-panel diagnostic summary
- `apply_nan_mask()` - Enforce NaN masking outside support

**Note**: These functions already support NaN masking and consistent vmin/vmax for cross-run comparisons.

### 4. Bootstrap Utilities (READY TO USE)

**Location**: `/home/user/morphseq/src/analyze/spline_fitting/bootstrap.py`

**Available Functions**:
- `spline_fit_wrapper()` - Bootstrap spline fitting with uncertainty
  - Parameters: `n_bootstrap=10` (default), `bootstrap_size=2500`
  - Returns: Mean spline + standard error columns (`{col}_se`)
  - Supports group-by mode for fitting multiple phenotypes

**Usage for AUROC bootstrap**: Adapt the bootstrap sampling logic for resampling embryos when computing AUROC confidence intervals.

### 5. Centerline Extraction via Skeletonization (READY TO USE)

**Location**: `/home/user/morphseq/segmentation_sandbox/scripts/body_axis_analysis/centerline_extraction.py`

**`extract_centerline()` function**:
- **Method**: Geodesic skeletonization (default) - robust to highly curved embryos
- **Process**:
  1. Gaussian blur preprocessing (sigma=15.0, threshold=0.7) - also provides density measure
  2. Skeletonization via `skimage.morphology`
  3. Geodesic distance along skeleton to extract centerline
  4. B-spline smoothing (s=5.0) for curvature measurement
- **Returns**: `spline_x, spline_y, curvature, arc_length`
- **Orientation convention**: We will enforce S=0 (head/rostral) to S=1 (tail/caudal) via `orient_spline_head_to_tail()`

**Key advantage**: This is the same method used in the data pipeline for curvature calculation, ensuring consistency.

**Note**: For template-space centerline, we run `extract_centerline()` on the WT reference mask once, then assign each pixel an S value based on nearest point on the extracted spline.

### 6. Batch OT Export for AUROC Scaling (PRODUCTION READY)

**Location**: `results/mcolon/20260213_stream_d_reference_embryo/pipeline/02_run_batch_ot_export.py`

**Status**: Validated on 313 transitions (0 failures, ~1.76s/pair on GPU)

**What it does**:
- Sequential processing of many WT→mutant OT problems with smart frame caching
- Uses OTT backend by default (GPU-enabled, ~2-5× faster than POT)
- Resume-safe via (run_id, pair_id) keys
- Exports to structured parquet files + raw field artifacts

**Why sequential, not vmap**:
- **Variable embryo shapes** → Different bbox sizes after cropping
- **Shape bucketing not implemented** → vmap requires identical tensor shapes
- **Recompilation avoided** → Sequential prevents JAX recompiling for each unique shape
- **Frame caching** → Huge I/O savings (load each frame once, reuse across pairs)

**Key insight**: JAX speedup comes from JIT compilation + GPU execution **per solve**. Sequential processing with cached frames is optimal for variable-shape OT problems.

**See**: `BATCH_PROCESSING_GUIDE.md` for complete usage guide, parameters, and pilot study checklist.

---

## Implementation Roadmap: Scripts by Section

### SECTION 0: Data Preparation & Reference Selection

**Script**: `00_select_reference_and_load_data.py`

**Purpose**:
- Load pre-existing cohort manifest from Stream D migration
- Extract embryos for target HPF bin (48 hpf)
- Select **one WT reference embryo** from `reference_wt` cohort (rank 1)
- Select **WT controls** from `heldout_wt` cohort (all available at 48 hpf)
- Select **mutants** from `mutant` cohort (all available at 48 hpf)
- Load masks for selected embryos
- **Critical**: Must load WT controls (n≥10) in addition to reference for permutation testing
- Output: Manifest with embryo_id, frame_index, genotype, role (reference/control/mutant)

**Inputs**:
- **Cohort manifest** (pre-existing): 
  - Location: `results/mcolon/20260213_stream_d_reference_embryo/output/cohort_selection/cohort_selected_embryos.csv`
  - If not found: Search for `cohort_selected_embryos.csv` in `results/mcolon/2026*` directories
  - Fallback: Re-run `results/mcolon/20260213_stream_d_reference_embryo/pipeline/01_build_cohort_manifest.py`
- **Bin-frame manifest**:
  - Location: `results/mcolon/20260213_stream_d_reference_embryo/output/cohort_selection/cohort_bin_frame_manifest.csv`
  - Maps embryo_id → frame_index for each 2h bin
- **Source data CSV**: `results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv`
  - Contains mask RLE, predicted_stage_hpf, curvature, etc.

**Outputs** (written to `data/`):
- `selected_embryos_48hpf.csv` (columns: embryo_id, frame_index, genotype, role, set_type, matched_stage_hpf)
  - role: 'reference' | 'control' | 'mutant'
  - set_type: 'reference_wt' | 'heldout_wt' | 'mutant'
- `reference_embryo_info.json` (single reference embryo metadata)
- `embryo_summary.txt` (counts by role)
- `cohort_contract_48hpf.json` (locked contract with selection params + exact embryo_id/frame_index lists)

**Key Functions to Write**:
- `load_cohort_manifest(manifest_path) -> pd.DataFrame`
  - Load pre-selected cohort from Stream D
  - Handle file-not-found gracefully with search logic
- `load_bin_frame_manifest(manifest_path) -> pd.DataFrame`  
  - Load frame assignments per embryo per bin
- `extract_embryos_for_bin(bin_manifest, cohort_df, target_hpf, tolerance=1.25) -> pd.DataFrame`
  - Filter to target HPF bin (e.g., 48 hpf ± 1.25)
  - Join with cohort metadata
  - Return: embryo_id, frame_index, genotype, set_type, matched_stage_hpf
- `assign_roles(selected_df) -> pd.DataFrame`
  - Map set_type to role:
    - reference_wt → 'reference' (take rank 1 only for pilot)
    - heldout_wt → 'control'
    - mutant → 'mutant'
- `summarize_selection(selected_df) -> dict`
  - Count by role, report stage range, coverage

**Example workflow**:
```python
# Load pre-existing manifests
cohort_path = Path("results/mcolon/20260213_stream_d_reference_embryo/output/cohort_selection/cohort_selected_embryos.csv")
bin_path = cohort_path.parent / "cohort_bin_frame_manifest.csv"

if not cohort_path.exists():
    # Search for migrated location
    cohort_path = search_for_file("cohort_selected_embryos.csv", base_dir="results/mcolon/2026*")
    
cohort_df = pd.read_csv(cohort_path)
bin_manifest = pd.read_csv(bin_path)

# Extract 48 hpf embryos
selected = extract_embryos_for_bin(bin_manifest, cohort_df, target_hpf=48.0, tolerance=1.25)

# Filter by set_type and assign roles
reference = selected[selected['set_type'] == 'reference_wt'].iloc[:1]  # Take rank 1 only
controls = selected[selected['set_type'] == 'heldout_wt']
mutants = selected[selected['set_type'] == 'mutant']

assert len(controls) >= 10, f"Need ≥10 WT controls, got {len(controls)}"
assert len(mutants) >= 20, f"Need ≥20 mutants, got {len(mutants)}"

reference['role'] = 'reference'
controls['role'] = 'control'
mutants['role'] = 'mutant'

final = pd.concat([reference, controls, mutants], ignore_index=True)
final.to_csv(output_root / "data" / "selected_embryos_48hpf.csv", index=False)
```

**Optional robustness window (`46-50 hpf`)**:
- If using an expanded 4 hpf window, collapse to one row per embryo before any statistical testing.
- Collapse rule: per-embryo median of each feature across included frames.
- Do not treat multiple frames from one embryo as independent samples.

**Dependencies**:
- `pandas` (for CSV loading and filtering)
- Existing cohort manifests from Stream D migration

---

### SECTION 1: OT Mapping + Outlier Filtering + Visualization

**Script**: `01_run_ot_mapping_and_filtering.py`

**Purpose**:
- Load WT reference mask
- Run unbalanced OT for all embryos → reference (using canonical grid)
- Export: `c(x)`, `d(x)`, `Δm(x)`, `C` (total cost)
- Apply IQR-based outlier filtering on total cost `C`
- Log filtered embryos to `outliers_removed.csv`
- Compute mean cost/displacement fields for WT vs mutants
- Generate Section 1 figures (A, A', B, QC panel)

**Inputs**:
- `reference_embryo_manifest.json`
- `embryo_mask_paths.csv`

**Outputs**:
- `ot_results/` directory:
  - `{embryo_id}_{snip_id}_cost_density.npy` (H×W array)
  - `{embryo_id}_{snip_id}_displacement_yx.npy` (H×W×2 array)
  - `{embryo_id}_{snip_id}_mass_delta.npy` (H×W array)
  - `{embryo_id}_{snip_id}_total_cost.json` (scalar)
- `ot_metrics.csv` (columns: embryo_id, snip_id, genotype, total_cost, is_outlier, removal_reason)
- `outliers_removed.csv`
- Figures:
  - `fig_1a_cost_heatmap_wt.png`
  - `fig_1a_cost_heatmap_mutant.png`
  - `fig_1a_cost_heatmap_diff.png`
  - `fig_1a_prime_cost_heatmap_smoothed.png` (Gaussian kernel smoothed)
  - `fig_1b_displacement_field_wt.png`
  - `fig_1b_displacement_field_mutant.png`
  - `fig_1b_displacement_field_diff.png`
  - `fig_qc_total_cost_distribution.png` (violin plot before/after filtering)

**Key Functions to Write**:
- `run_ot_to_reference(mask, reference_mask, config) -> dict`
  - Returns: `cost_density`, `displacement_yx`, `mass_delta`, `total_cost`
  - Wraps `run_transport()` from existing OT infrastructure
- `filter_outliers_iqr(df, cost_col='total_cost') -> (df_filtered, df_outliers)`
- `apply_gaussian_kernel_smoothing(field_2d, sigma=2.0) -> field_2d_smoothed`
  - Use Gaussian filter from `scipy.ndimage`
- `plot_cost_heatmap(cost_field, title, outpath)`
- `plot_displacement_quiver(disp_field_yx, mask, title, outpath)`

**Dependencies**:
- `src/analyze/optimal_transport_morphometrics/uot_masks/run_transport.py`
- `src/analyze/optimal_transport_morphometrics/uot_masks/uot_grid.py`
- `src/analyze/optimal_transport_morphometrics/uot_masks/viz.py`
- `scipy.ndimage.gaussian_filter` (for Gaussian kernel smoothing)

---

### SECTION 2: Spline Centerline + S Assignment + Along-S Profiles

**Script**: `02_compute_spline_and_s_profiles.py`

**Purpose**:
- Fit spline to WT reference mask centerline
- Define S ∈ [0, 1] along spline (S=0 head/rostral, S=1 tail/caudal)
- Assign each pixel in template grid an S value (via nearest point on spline)
- Define K bins along S (K=10 initially, K=20 for higher resolution)
- For each embryo, compute along-S profiles:
  - Mean cost density per S bin: `c̄_k`
  - Mean displacement magnitude: `|d̄|_k`
  - Mean axial displacement: `d̄∥_k`
  - Mean perpendicular displacement: `d̄⊥_k`
  - Mean divergence: `div̄_k`
  - Mean mass delta: `Δm̄_k`
- Generate Section 2 plots (1, 1', 2, 3, 4)

**Inputs**:
- WT reference mask
- `ot_results/` (cost, displacement, mass_delta arrays)
- `ot_metrics.csv` (filtered embryos only)

**Outputs**:
- `reference_spline.csv` (columns: spline_point_index, x, y, S, tangent_x, tangent_y, normal_x, normal_y)
- `s_assignment_map.npy` (H×W array, S value per pixel)
- `s_bin_masks/` directory:
  - `s_bin_{k:02d}.npy` (H×W boolean mask for bin k)
- `s_profiles.csv` (columns: embryo_id, snip_id, genotype, s_bin, c_bar, d_mag_bar, d_parallel_bar, d_perp_bar, div_bar, mass_delta_bar)
- Figures:
  - `fig_2_1_cost_along_s.png` (WT vs mutant, with confidence bands)
  - `fig_2_1_prime_cost_along_s_smoothed.png` (Gaussian kernel smoothed)
  - `fig_2_2_displacement_mag_along_s.png`
  - `fig_2_3_displacement_axial_vs_perp_along_s.png`
  - `fig_2_4_divergence_mass_delta_along_s.png`

**Key Functions to Write**:
- `extract_centerline_with_orientation(mask, um_per_pixel=10.0) -> spline_df`
  - Use `extract_centerline()` from `body_axis_analysis.centerline_extraction`
  - Geodesic skeletonization + B-spline smoothing (built-in)
  - Returns: spline_x, spline_y, curvature, arc_length
  - Enforces head-to-tail orientation automatically
- `assign_s_to_pixels(spline_df, grid_shape_hw) -> s_map`
  - For each pixel, find nearest spline point and assign its S value
- `compute_tangent_normal_at_s(spline_df) -> (tangent_yx, normal_yx)`
  - Compute tangent vectors (derivative along spline)
  - Compute normal vectors (perpendicular to tangent)
- `compute_divergence(disp_field_yx) -> div_field`
  - Finite difference divergence: ∂u/∂x + ∂v/∂y
- `compute_s_profile(embryo_id, s_bin_masks, cost_field, disp_field, mass_delta_field, spline_df) -> dict`
  - For each S bin k, compute mean of fields within bin
  - Decompose displacement into parallel/perpendicular components
- `plot_s_profile(df, feature_col, title, outpath)`
  - Line plot with WT vs mutant, confidence bands (mean ± SE)
- `apply_gaussian_kernel_smoothing_1d(profile_1d, sigma=1.0) -> profile_1d_smoothed`
  - Gaussian smoothing along S axis

**Dependencies**:
- `segmentation_sandbox.scripts.body_axis_analysis.centerline_extraction` (`extract_centerline()`)
- `scipy.ndimage` (for divergence via finite differences, Gaussian smoothing)
- `scipy.ndimage.gaussian_filter` (for 2D Gaussian kernel smoothing - also provides density measure)
- `scipy.ndimage.gaussian_filter1d` (for 1D Gaussian kernel smoothing)

---

### SECTION 3: Feature Table Construction

**Script**: `03_build_feature_table.py`

**Purpose**:
- Consolidate S-bin features into a standardized feature table
- Organize features into sets A, B, C (cost, vector, deformation+mass)
- Save feature table for downstream AUROC analysis

**Inputs**:
- `s_profiles.csv`

**Outputs**:
- `feature_table.csv` (columns: embryo_id, snip_id, stage_bin, label, k_bin, c_bar, d_mag_bar, d_parallel_bar, d_perp_bar, div_bar, mass_delta_bar)
- `feature_sets.json` (metadata: which features belong to sets A, B, C)

**Key Functions to Write**:
- `build_feature_table(s_profiles_df) -> feature_df`
  - Pivot/reshape into wide format if needed
  - Add `label` column: 1 for mutant, 0 for WT
  - Add `stage_bin` column (for future multi-timepoint extension)
- `validate_feature_table(feature_df)`
  - Check for NaNs, missing embryos, bin coverage

**Dependencies**:
- `pandas`

---

### SECTION 4: AUROC-by-Region Analysis

**Script**: `04_compute_auroc_by_s_bin.py`

**Purpose**:
- For each S bin k independently, compute AUROC for each feature
- Scalar features (c̄_k, |d̄|_k, div̄_k, Δm̄_k): univariate AUROC via Mann-Whitney U
- Directional feature (d̄∥_k, d̄⊥_k): 2D logistic regression AUROC with CV-by-embryo
- **Embryo-level label permutation test** for significance (following UOT MVP pattern)
- **FDR correction** (Benjamini-Hochberg) across S bins for multiple testing
- Bootstrap confidence intervals (embryo-level resampling)
- Gaussian kernel smoothed AUROC profiles (visualization only)
- Bootstrap resample to compute confidence intervals on AUROC
- Generate Section 4 plots (AUROC along S, smoothed AUROC)

**Inputs**:
- `feature_table.csv`

**Outputs**:
- `auroc_by_s_bin.csv` (columns: s_bin, feature_name, auroc, pvalue_raw, pvalue_fdr, significant_fdr, auroc_ci_low, auroc_ci_high, n_wt, n_mutant)
- `permutation_nulls/` directory:
  - `null_dist_s{k:02d}_{feature}.npy` (null distributions per bin/feature)
- Figures:
  - `fig_4_auroc_along_s_cost.png` (AUROC profile for cost features, mark significant bins)
  - `fig_4_auroc_along_s_displacement.png` (AUROC for displacement features)
  - `fig_4_auroc_along_s_divergence_mass.png` (AUROC for divergence/mass delta)
  - `fig_4_auroc_along_s_smoothed.png` (Gaussian kernel smoothed AUROC profiles)
  - `fig_4_permutation_null_s{k:02d}_{feature}.png` (histograms of null distributions)

**Key Functions to Write**:
- `compute_auroc_univariate(y_true, y_score) -> float`
  - Use `sklearn.metrics.roc_auc_score`
  - Equivalent to Mann-Whitney U for binary classification
- `compute_auroc_2d_directional(df, s_bin, features=['d_parallel_bar', 'd_perp_bar']) -> float`
  - Fit logistic regression with CV-by-embryo (GroupKFold)
  - Return cross-validated AUROC
- `embryo_label_shuffle_auroc_per_bin(df, feature_cols_per_bin, n_permutations=999) -> results_dict`
  - **Pattern from UOT MVP** (`06_difference_classification_clustering.py::_embryo_label_shuffle_pvalue`)
  - For each S bin k:
    - Shuffle genotype labels at embryo level (not snip level)
    - Recompute AUROC with shuffled labels
    - Build null distribution
    - Compute p-value: (1 + sum(null >= observed)) / (n_perm + 1)
  - Return dict with observed_auroc, pvalue, null_distribution per bin
- `apply_fdr_correction(pvalues, alpha=0.05) -> (reject, pvals_corrected)`
  - Use `statsmodels.stats.multitest.multipletests` with method='fdr_bh'
- `bootstrap_auroc_ci(df, s_bin, feature, embryo_id_col, n_bootstrap=1000) -> (ci_low, ci_high)`
  - Resample embryos with replacement (not snips)
  - Recompute AUROC on each resample
  - Return 95% CI via percentile method
- `plot_auroc_along_s(auroc_df, feature_subset, title, outpath)`
  - Line plot with error bars (CI)
  - One curve per feature
- `apply_gaussian_kernel_smoothing_1d(auroc_profile, sigma=1.0) -> smoothed_auroc`

**Dependencies**:
- `sklearn.metrics.roc_auc_score`
- `sklearn.linear_model.LogisticRegression`
- `sklearn.model_selection.GroupKFold`
- `scipy.ndimage.gaussian_filter1d`

---

### SECTION 5: Automated ROI Discovery (1D Patch Search)

**Script**: `05_patch_search_on_s.py`

**Purpose**:
- Search over contiguous S intervals I = [a, b]
- For each interval, train classifier using only features from bins in I
- Score by cross-validated AUROC (GroupKFold by embryo_id)
- Select interval(s) maximizing AUROC (with optional length penalty)
- **Embryo-level label permutation test** for selected interval significance
- Bootstrap stability analysis (interval overlap via Jaccard similarity)
- Perform patch ablation: remove features in interval I, measure AUROC drop
- Generate Section 5 plots (AUROC vs interval length, selected intervals, null distribution)

**Inputs**:
- `feature_table.csv`

**Outputs**:
- `patch_search_results.csv` (columns: interval_start, interval_end, interval_length, auroc, auroc_std)
- `best_intervals.json` (selected intervals with AUROC, p-value, Jaccard stability)
- `best_interval_permutation_null.npy` (null distribution from label shuffling)
- Figures:
  - `fig_5_auroc_vs_interval_length.png` (tradeoff curve)
  - `fig_5_selected_intervals_on_s.png` (highlight selected intervals on S axis)
  - `fig_5_patch_ablation_importance.png` (AUROC drop per interval)
  - `fig_5_permutation_null_best_interval.png` (histogram of null distribution vs observed)

**Key Functions to Write**:
- `search_contiguous_intervals(df, min_length=2, max_length=10) -> results_df`
  - Enumerate all contiguous intervals [a, b] with length constraints
  - For each interval, select features in S bins [a, b]
  - Train logistic regression with GroupKFold CV-by-embryo
  - Return AUROC and std
- `best_interval_permutation_test(df, best_interval, feature_cols, n_permutations=999) -> pvalue_dict`
  - **Pattern from UOT MVP** (embryo-level label shuffling)
  - For each permutation:
    - Shuffle genotype labels at embryo level
    - Re-run patch search to find best interval on shuffled data
    - Record AUROC of best interval in null
  - P-value: fraction of null AUROCs >= observed best-interval AUROC
  - Returns: observed_auroc, pvalue, null_distribution
- `bootstrap_interval_stability(df, min_length, max_length, n_bootstrap=100) -> jaccard_scores`
  - Resample embryos with replacement
  - Re-run interval search on each bootstrap sample
  - Measure Jaccard overlap between selected intervals across resamples
  - Return mean and std of Jaccard similarity
- `ablate_interval(df, interval, full_auroc) -> auroc_drop`
  - Train classifier on ALL S bins
  - Train classifier with interval features removed/zeroed
  - Return `full_auroc - ablated_auroc`
- `bootstrap_interval_stability(df, interval, n_bootstrap=100) -> overlap_scores`
  - Resample embryos, re-run interval search
  - Measure overlap of selected intervals across resamples
- `plot_interval_search(results_df, outpath)`
  - Scatter plot: x=interval_length, y=AUROC
  - Highlight Pareto front
- `plot_selected_intervals(intervals, s_axis, outpath)`
  - Show S axis with highlighted regions

**Dependencies**:
- `sklearn.linear_model.LogisticRegression`
- `sklearn.model_selection.GroupKFold`
- `itertools` (for interval enumeration)

---

### SECTION 6: 2D Sparse Mask Learning (OPTIONAL)

**Script**: `06_sparse_mask_learning.py` (OPTIONAL - defer until Sections 1-5 validated)

**Purpose**:
- Learn mask m(x) in template space via L1 + TV regularization
- Sweep (λ, μ) via Pareto optimization
- Validate with patch ablation and bootstrap stability
- Generate Section 6 plots (learned mask overlays, Pareto front)

**Inputs**:
- `feature_table.csv`
- `ot_results/` (for pixel-level features)

**Outputs**:
- `learned_masks/` directory (one mask per (λ, μ) setting)
- `pareto_front.csv` (columns: lambda, mu, auroc, mask_area_frac, n_components)
- Figures:
  - `fig_6_learned_mask_overlay.png`
  - `fig_6_pareto_front.png`
  - `fig_6_bootstrap_stability.png` (mask overlap heatmaps)

**Key Functions to Write**:
- `learn_sparse_mask(features_2d, labels, lambda_l1, mu_tv) -> mask_2d`
  - Optimize: classification loss + λ×L1(m) + μ×TV(m)
  - Use gradient descent or PyTorch
- `sweep_pareto_front(features_2d, labels, lambda_range, mu_range) -> pareto_df`
  - Grid search over (λ, μ)
  - Record AUROC, area, n_components for each setting
- `compute_mask_overlap(mask_list) -> overlap_matrix`
  - Jaccard similarity between pairs of masks
- `plot_learned_mask(mask, reference_mask, outpath)`

**Dependencies**:
- `torch` or `jax` (for differentiable optimization)
- `skimage.measure.label` (for connected components)
- `scipy.ndimage.total_variation` (TV computation)

**Note**: This section is **optional** and should be deferred until Sections 1-5 are fully validated.

---

## Utility Modules to Create

### `utils/ot_features.py`

**Functions**:
- `extract_cost_density(uot_result) -> np.ndarray`
- `extract_displacement_field(uot_result) -> np.ndarray`
- `extract_mass_delta(uot_result) -> np.ndarray`
- `compute_divergence(disp_field_yx) -> np.ndarray`

### `utils/s_bin_utils.py`

**Functions**:
- `create_s_bins(s_map, n_bins=10) -> List[np.ndarray]`
- `bin_feature_mean(feature_map, bin_masks) -> np.ndarray`
- `decompose_displacement_parallel_perp(disp_yx, tangent_yx, normal_yx) -> (d_parallel, d_perp)`

### `utils/gaussian_kernel.py`

**Functions**:
- `gaussian_kernel_smooth_2d(field, sigma=2.0) -> field_smoothed`
- `gaussian_kernel_smooth_1d(profile, sigma=1.0) -> profile_smoothed`

### `utils/statistical_testing.py`

**Functions** (using `src/analyze/utils/resampling` framework):

```python
from analyze.utils.resampling import resample
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
import numpy as np

def auroc_permutation_test_per_bin(
    features_per_bin: np.ndarray,  # shape: (n_embryos, n_bins)
    labels: np.ndarray,            # shape: (n_embryos,), 0=WT, 1=mutant
    n_iters: int = 999,
    seed: int = 42,
    apply_fdr: bool = True,
    alpha: float = 0.05
) -> dict:
    """Test AUROC > 0.5 for each S bin via embryo-level label permutation.
    
    Returns dict with keys:
        - auroc_observed: (n_bins,) array
        - pvalues_raw: (n_bins,) array
        - pvalues_fdr: (n_bins,) array (if apply_fdr=True)
        - reject_fdr: (n_bins,) bool array (if apply_fdr=True)
    """
    n_bins = features_per_bin.shape[1]
    auroc_obs = np.zeros(n_bins)
    pvals = np.zeros(n_bins)
    
    for k in range(n_bins):
        features_k = features_per_bin[:, k]
        
        # Observed AUROC
        auroc_obs[k] = roc_auc_score(labels, features_k)
        
        # Permutation test
        spec = resample.permute_labels()
        stat = resample.statistic(
            f"auroc_bin_{k}",
            lambda data, rng: roc_auc_score(data["labels"], data["features_k"])
        )
        out = resample.run(
            data={"features_k": features_k, "labels": labels},
            spec=spec,
            stat=stat,
            n_iters=n_iters,
            seed=seed + k,  # Different seed per bin
            alternative="two-sided"
        )
        summary = resample.aggregate(out)
        pvals[k] = summary.pvalue
    
    result = {
        "auroc_observed": auroc_obs,
        "pvalues_raw": pvals,
    }
    
    if apply_fdr:
        reject, pvals_fdr, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
        result["pvalues_fdr"] = pvals_fdr
        result["reject_fdr"] = reject
    
    return result


def auroc_bootstrap_ci(
    features: np.ndarray,
    labels: np.ndarray,
    embryo_ids: np.ndarray,
    n_iters: int = 1000,
    seed: int = 42,
    alpha: float = 0.05
) -> dict:
    """Bootstrap CI for AUROC by resampling embryos.
    
    Returns dict with keys:
        - auroc_observed
        - ci_low
        - ci_high
        - ci_method
    """
    spec = resample.bootstrap(unit="embryo_id")
    stat = resample.statistic(
        "auroc",
        lambda data, rng: roc_auc_score(
            data["labels"][data["indices"]],
            data["features"][data["indices"]]
        )
    )
    out = resample.run(
        data={
            "features": features,
            "labels": labels,
            "embryo_id": embryo_ids,
            "n": len(labels)
        },
        spec=spec,
        stat=stat,
        n_iters=n_iters,
        seed=seed
    )
    summary = resample.aggregate(out, alpha=alpha)
    
    return {
        "auroc_observed": summary.observed,
        "ci_low": summary.ci_low,
        "ci_high": summary.ci_high,
        "ci_method": summary.ci_method,
    }
```

**Example usage pattern from UOT MVP** (`06_difference_classification_clustering.py`):

```python
# Binary comparison: mutants vs WT controls (exclude reference WT)
binary_df = df[df["set_type"].isin(["mutant", "heldout_wt"])].copy()

# Embryo-level label shuffling for permutation test
def _embryo_label_shuffle_pvalue(df, feature_cols, n_perm, random_state):
    rng = np.random.default_rng(random_state)
    embryo_map = df.groupby("embryo_id")["set_type"].agg(lambda s: s.mode().iloc[0]).to_dict()
    embryo_ids = np.array(sorted(embryo_map.keys()))
    labels = np.array([embryo_map[e] for e in embryo_ids], dtype=object)
    
    observed = _grouped_logreg_cv(df, feature_cols, label_col="set_type", group_col="embryo_id")["auroc"]
    null = []
    for _ in range(n_perm):
        perm = labels.copy()
        rng.shuffle(perm)  # Shuffle embryo-level labels
        perm_map = {e: p for e, p in zip(embryo_ids, perm)}
        dperm = df.copy()
        dperm["set_type"] = dperm["embryo_id"].map(perm_map)
        try:
            auc = _grouped_logreg_cv(dperm, feature_cols, label_col="set_type", group_col="embryo_id")["auroc"]
            null.append(float(auc))
        except Exception:
            continue
    null_arr = np.asarray(null, dtype=np.float64)
    pval = float((1.0 + np.sum(null_arr >= observed)) / (len(null_arr) + 1.0))
    return {"observed_auroc": float(observed), "pvalue": pval, "null_distribution": null_arr}
```

---

## Testing Strategy

### Unit Tests

**Location**: `tests/`

**Test Files**:
- `test_s_bin_utils.py` - Test S binning and feature aggregation
- `test_ot_features.py` - Test OT feature extraction
- `test_gaussian_kernel.py` - Test smoothing functions
- `test_auroc_computation.py` - Test AUROC calculation and bootstrap CI

### Integration Tests

**Test Scripts**:
- `test_end_to_end_single_embryo.py` - Run full pipeline on 1 WT + 1 mutant
- `test_outlier_filtering.py` - Verify IQR filtering with synthetic outliers
- `test_s_profile_reproducibility.py` - Check S profiles are consistent across runs

### Validation

**Validation Checks**:
- **Section 1**: Cost distributions are stable after outlier removal (IQR test)
- **Section 2**: S profiles are smooth and robust to K (10 vs 20 bins)
- **Section 4**: AUROC localizes to expected regions (tail for cep290 curvature)
- **Section 5**: Selected intervals are stable across bootstrap resamples

---

## Execution Order (Immediate Next Steps)

1. ✅ **DONE**: Copy PLAN.md to dated results folder
2. ✅ **DONE**: Create IMPLEMENTATION_PLAN.md (this document)
3. **NEXT**: `00_select_reference_and_load_data.py`
   - Select HPF bin (48 hpf recommended)
   - Select reference WT embryo from cohort
   - Load masks for WT + cep290
4. **NEXT**: `01_run_ot_mapping_and_filtering.py`
   - Run UOT for all embryos → reference
   - Filter outliers via IQR
   - Generate Section 1 figures
5. **NEXT**: `02_compute_spline_and_s_profiles.py`
   - Fit spline to reference mask
   - Assign S to pixels
   - Compute along-S profiles
   - Generate Section 2 figures
6. **NEXT**: `03_build_feature_table.py`
   - Consolidate features into table
7. **NEXT**: `04_compute_auroc_by_s_bin.py`
   - Compute AUROC per S bin
   - Bootstrap confidence intervals
   - Generate Section 4 figures
8. **NEXT**: `05_patch_search_on_s.py`
   - Search for best S intervals
   - Patch ablation
   - Generate Section 5 figures
9. **(OPTIONAL)**: `06_sparse_mask_learning.py`
   - Only after validating Sections 1-5

---

## Configuration File

**File**: `config.yaml`

```yaml
# Subtle-Phenotype Localization Config
# 2026-02-13

# Data selection
hpf_bin_start: 47.0
hpf_bin_end: 49.0  # Use 48 hpf as reference timepoint
genotype_wt: "cep290_wildtype"
genotype_mutant: "cep290_homozygous"
n_reference_wt: 1  # WT reference (OT target/template, defines coordinate system)
n_control_wt: 10  # WT controls (mapped to reference, used in statistical tests)
n_mutants: 20  # Mutants (mapped to reference, compared to WT controls)
# NOTE: Reference WT is NOT included in statistical comparisons

# OT parameters (already tuned)
ot_epsilon: 1.0e-4
ot_marginal_relaxation: 10.0
ot_downsample_factor: 1
ot_canonical_grid_um_per_pixel: 10.0
ot_canonical_grid_shape_hw: [256, 576]
ot_align_mode: "yolk"

# Outlier filtering
outlier_method: "iqr"
outlier_iqr_multiplier: 1.5

# S binning
n_s_bins: 10  # Start with 10, try 20 for higher resolution
s_orientation: "head_to_tail"  # S=0 head, S=1 tail

# Gaussian kernel smoothing (visualization only)
gaussian_kernel_sigma_2d: 2.0  # For 2D cost density maps
gaussian_kernel_sigma_1d: 1.0  # For 1D along-S profiles

# AUROC computation
auroc_n_bootstrap: 1000  # Bootstrap iterations for CI
auroc_cv_folds: 5  # GroupKFold folds for CV-by-embryo
auroc_stratify: true  # Stratify resampling by genotype

# Patch search
patch_min_length: 2  # Minimum S bins per interval
patch_max_length: 10  # Maximum S bins per interval
patch_length_penalty: 0.01  # Optional penalty on interval length

# Random seed
random_seed: 42

# Output
save_intermediate_maps: true  # Save c(x), d(x), etc. as .npy files
save_diagnostic_plots: true
```

---

## Key Design Decisions

### 1. Single HPF Bin for Pilot

**Decision**: Use **48 hpf bin** (or 48 hpf) as the "single 2 hpf window" for initial implementation.

**Rationale**:
- Stream D cohort has good coverage at 48 hpf
- 48 hpf represents mature phenotype stage (better differentiation between WT and mutants)
- Provides well-formed embryo morphology for robust centerline extraction
- Once pipeline is validated, extending to multiple bins is trivial (just loop over bins)

### 2. Single WT Reference Embryo

**Decision**: Use **one WT reference embryo** from the `reference_wt` cohort.

**Rationale**:
- Simpler for pilot (avoid averaging multiple references)
- Consistent with "pre-selected WT reference mask already available" in PLAN.md
- Can extend to averaged reference later if needed

### 3. Unbalanced OT Parameters (Fixed)

**Decision**: Use `epsilon=1e-4`, `marginal_relaxation=10.0` (already tuned in Stream D).

**Rationale**:
- Parameters already validated on cep290 data
- Avoids re-tuning for pilot
- Treat as fixed per PLAN.md

### 4. K=10 S Bins Initially

**Decision**: Start with K=10 S bins, then try K=20 for resolution.

**Rationale**:
- K=10 provides coarse spatial localization (10% of embryo length per bin)
- K=20 provides finer resolution (5% per bin)
- Robustness check: results should be stable across K

### 5. Heat Kernel Smoothing (Visualization Only)

**Decision**: All statistical tests (AUROC, filtering) on **unsmoothed** features; smoothing is **visualization only**.

**Rationale**:
- Avoids artificially inflating spatial coherence in stats
- Provides clean, interpretable plots
- Consistent with PLAN.md smoothing policy

### 6. Bootstrap Resampling by Embryo (Not Snip)

**Decision**: Resample entire embryos (embryo_id), not individual snips.

**Rationale**:
- Avoids leakage across snips from same embryo
- GroupKFold CV-by-embryo ensures independent test sets
- Consistent with Global QC hygiene in PLAN.md

---

## Dependencies Summary

**Python Packages**:
- `numpy`, `pandas` (data manipulation)
- `scipy` (gaussian filtering, divergence, stats)
- `scikit-image` (skeletonize, morphology)
- `scikit-learn` (AUROC, LogisticRegression, GroupKFold)
- `matplotlib`, `seaborn` (plotting)
- `tqdm` (progress bars)
- `pyyaml` (config loading)

**MorphSeq Modules**:
- `src.analyze.optimal_transport_morphometrics.uot_masks.*`
- `src.analyze.spline_fitting.*`
- `src.analyze.utils.optimal_transport.*`

**Data Sources**:
- MorphSeq mask zarr stores
- Cohort manifest CSV (from Stream D or regenerated)

---

## Success Metrics (Per Section)

### Section 1 (OT Mapping)
- ✅ Outlier removal reduces cost variance
- ✅ Mean vector fields show coherent structure (not alignment failures)
- ✅ Gaussian-kernel-smoothed maps highlight spatial patterns

### Section 2 (S Profiles)
- ✅ Along-S profiles are smooth and sensible
- ✅ Profiles robust to K (10 vs 20 bins)
- ✅ Curvature phenotype shows expected head/tail emphasis

### Section 4 (AUROC)
- ✅ AUROC localizes to interpretable S regions
- ✅ Gaussian-kernel-smoothed AUROC shows clear peaks
- ✅ Patterns stable under bootstrap resampling

### Section 5 (Patch Search)
- ✅ Selected interval aligns with known phenotype
- ✅ Interval stable across bootstrap resamples
- ✅ Ablation confirms interval is most important

---

## File Structure (Output)

```
results/mcolon/20260213_subtle_phenotype_localization_ot/
├── PLAN.md  (specification document)
├── IMPLEMENTATION_PLAN.md  (this document)
├── config.yaml  (configuration)
├── scripts/
│   ├── 00_select_reference_and_load_data.py
│   ├── 01_run_ot_mapping_and_filtering.py
│   ├── 02_compute_spline_and_s_profiles.py
│   ├── 03_build_feature_table.py
│   ├── 04_compute_auroc_by_s_bin.py
│   ├── 05_patch_search_on_s.py
│   └── 06_sparse_mask_learning.py  (optional)
├── utils/
│   ├── ot_features.py
│   ├── s_bin_utils.py
│   ├── gaussian_kernel.py
│   └── bootstrap_auroc.py
├── tests/
│   ├── test_s_bin_utils.py
│   ├── test_ot_features.py
│   ├── test_gaussian_kernel.py
│   └── test_auroc_computation.py
├── data/
│   ├── reference_embryo_manifest.json
│   ├── embryo_mask_paths.csv
│   ├── cohort_manifest.csv  (if regenerated)
├── ot_results/
│   ├── {embryo_id}_{snip_id}_cost_density.npy
│   ├── {embryo_id}_{snip_id}_displacement_yx.npy
│   ├── {embryo_id}_{snip_id}_mass_delta.npy
│   └── ...
├── s_bin_masks/
│   ├── s_bin_00.npy
│   ├── s_bin_01.npy
│   └── ...
├── outputs/
│   ├── ot_metrics.csv
│   ├── outliers_removed.csv
│   ├── reference_spline.csv
│   ├── s_assignment_map.npy
│   ├── s_profiles.csv
│   ├── feature_table.csv
│   ├── feature_sets.json
│   ├── auroc_by_s_bin.csv
│   ├── patch_search_results.csv
│   ├── best_intervals.json
├── figures/
│   ├── section_1/
│   │   ├── fig_1a_cost_heatmap_*.png
│   │   ├── fig_1b_displacement_field_*.png
│   │   └── fig_qc_total_cost_distribution.png
│   ├── section_2/
│   │   ├── fig_2_1_cost_along_s.png
│   │   ├── fig_2_1_prime_cost_along_s_smoothed.png
│   │   └── ...
│   ├── section_4/
│   │   ├── fig_4_auroc_along_s_*.png
│   │   └── ...
│   ├── section_5/
│   │   ├── fig_5_auroc_vs_interval_length.png
│   │   └── ...
│   └── section_6/  (optional)
└── logs/
    ├── run_log_section_1.txt
    ├── run_log_section_2.txt
    └── ...
```

---

## Next Steps

1. **Review this implementation plan** with the user
2. **Create utility modules** (`utils/*.py`)
3. **Implement Section 0** (data loading + reference selection)
4. **Implement Section 1** (OT mapping + filtering + viz)
5. **Validate Section 1** before proceeding to Section 2
6. **Iterate through Sections 2-5** sequentially
7. **(Optional)** Implement Section 6 after validating Sections 1-5

---

## Notes

- All AUROC and statistical tests computed on **unsmoothed** per-bin features
- Gaussian kernel smoothing is **visualization only** (applied to heatmaps, 1D profiles, AUROC profiles)
- Bootstrap resampling always by `embryo_id` (never by snip)
- Cross-validation always uses `GroupKFold` by `embryo_id`
- Reference embryo and OT parameters are **fixed** for pilot (no re-tuning)
- HPF bin is **single 2 hpf window** (48 hpf recommended)
- Extension to multiple timepoints is straightforward once pipeline validated

---

**Status**: Ready to begin implementation. Proceed to Section 0 data loading.
