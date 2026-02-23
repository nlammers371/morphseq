# Subtle-Phenotype Localization via WT-Referenced OT

**Project Start Date**: 2026-02-13
**Pilot Dataset**: cep290 (simple curvature phenotype)
**HPF Bin**: 48 hpf (`tolerance_hpf=1.25` for initial implementation)

---

## Overview

This project builds a time-consistent, interpretable spatial localization pipeline for subtle phenotypes using:
- **Unbalanced optimal transport** (UOT) to map mutant masks onto WT reference
- **WT reference** as spatial coordinate system (template space)
- **WT controls + mutants** mapped to reference for statistical comparison
- **Rostral-caudal (S) coordinates** for embryo-intrinsic spatial localization
- **AUROC-by-region analysis** to identify discriminative spatial regions
- **Embryo-level label permutation** for significance testing with FDR correction
- **Automated ROI discovery** via patch search on S axis

**Goal**: Answer "where along the embryo does the phenotype manifest?" with stable, interpretable spatial signals.

**Key distinction**: 
- **Reference WT (n=1)**: Defines coordinate system (template space), NOT used in statistics
- **WT controls (n≥10)**: Mapped to reference, compared to mutants in statistical tests
- **Mutants (n≥20)**: Mapped to reference, compared to WT controls

---

## Key Documents

- **`PLAN.md`**: Full specification document
- **`IMPLEMENTATION_PLAN.md`**: Detailed implementation roadmap with scripts, functions, and dependencies
- **`config.yaml`**: Configuration parameters for the analysis
- **`data/cohort_contract_48hpf.json`**: Locked cohort contract (selection params + exact embryo/frame assignments)

---

## Infrastructure Leveraged

### From Existing OT Work

1. **Reference Embryo Selection** (Stream D - MIGRATED):
   - **Migrated location** (as of 2026-02-13): `results/mcolon/20260213_stream_d_reference_embryo/`
   - **Cohort selection script**: `pipeline/01_build_cohort_manifest.py`
   - **Pre-selected embryos**: `output/cohort_selection/cohort_selected_embryos.csv`
   - 1 reference WT embryo for pilot anchoring
   - 10 heldout WT controls for statistical comparison
   - 20 mutants (cep290_homozygous)
   - Selection criteria: maximize coverage, minimize curvature
   - **NOTE**: If files moved again, search for `cohort_selected_embryos.csv` in `results/mcolon/2026*`

2. **UOT Grid Configuration**:
   - Canonical grid: 256×576 pixels at 10 μm/pixel
   - Yolk-based alignment
   - Parameters: `epsilon=1e-4`, `marginal_relaxation=10.0` (already tuned)

3. **Plotting Utilities** (`viz.py`):
   - Cost heatmaps, quiver plots, mass creation/destruction
   - NaN masking enforced
   - Consistent vmin/vmax for cross-run comparisons

4. **Bootstrap Utilities** (`spline_fitting/bootstrap.py`):
   - Bootstrap spline fitting with uncertainty estimation
   - Stratified resampling by embryo_id for AUROC confidence intervals

5. **Centerline Extraction** (`body_axis_analysis/centerline_extraction.py`):
   - Geodesic skeletonization method (same as data pipeline)
   - Gaussian blur preprocessing (sigma=15.0) - also provides density measure
   - B-spline smoothing for curvature measurement
   - Basis for S coordinate assignment

---

## Directory Structure

```
.
├── PLAN.md                    # Specification (DO NOT MODIFY)
├── IMPLEMENTATION_PLAN.md     # Implementation roadmap
├── README.md                  # This file
├── config.yaml                # Configuration parameters
├── scripts/                   # Analysis scripts (Sections 0-6)
├── utils/                     # Utility modules
├── tests/                     # Unit and integration tests
├── data/                      # Input data manifests
├── ot_results/                # OT outputs (c(x), d(x), Δm(x))
├── s_bin_masks/               # S bin masks
├── outputs/                   # CSV/JSON results by section
├── figures/                   # Plots by section
└── logs/                      # Execution logs
```

---

## Implementation Sections

### Section 0: Data Preparation ✨ NEXT
- Select reference WT embryo
- Load masks for 48 hpf bin (WT + cep290; tolerance 1.25)
- **Script**: `scripts/00_select_reference_and_load_data.py`

### Section 1: OT Mapping + Outlier Filtering
- Run UOT for all embryos → reference
- IQR-based outlier filtering
- Cost heatmaps, displacement fields
- **Script**: `scripts/01_run_ot_mapping_and_filtering.py`

### Section 2: Spline Centerline + S Profiles
- Fit spline to reference mask
- Assign S ∈ [0,1] to pixels (head→tail)
- Compute along-S profiles (cost, displacement, divergence)
- **Script**: `scripts/02_compute_spline_and_s_profiles.py`

### Section 3: Feature Table Construction
- Consolidate S-bin features
- Organize into feature sets A, B, C
- **Script**: `scripts/03_build_feature_table.py`

### Section 4: AUROC-by-Region Analysis
- Compute AUROC per S bin (univariate + 2D directional)
- **Embryo-level label permutation** for p-values (shuffle WT control vs mutant labels)
- **FDR correction** across S bins (Benjamini-Hochberg)
- Bootstrap confidence intervals (resample embryos)
- Gaussian kernel smoothed AUROC profiles (visualization only)
- **Script**: `scripts/04_compute_auroc_by_s_bin.py`

### Section 5: Automated ROI Discovery (1D Patch Search)
- Search over contiguous S intervals
- **Permutation test** for interval significance
- Patch ablation for importance
- Bootstrap stability testing
- **Script**: `scripts/05_patch_search_on_s.py`

### Section 6: 2D Sparse Mask Learning (OPTIONAL)
- L1 + TV regularization
- Pareto optimization over (λ, μ)
- **Script**: `scripts/06_sparse_mask_learning.py`

---

## Key Design Decisions

1. **Single HPF bin (48 hpf, tolerance 1.25)** for pilot → extends to multiple bins trivially
2. **Single WT reference embryo** from Stream D cohort at 48 hpf
3. **Fixed OT parameters** (epsilon, marginal_relaxation) → already tuned
4. **K=10 S bins initially** → validate, then try K=20
5. **Gaussian kernel smoothing for visualization only** → stats on unsmoothed features; also provides density measure
6. **Bootstrap by embryo_id** (not snip) → avoid leakage
7. **Geodesic skeletonization** (same method as data pipeline) → consistent with existing curvature calculations

---

## Success Criteria

### Section 1
- ✅ Outlier removal stabilizes cost distributions
- ✅ Mean vector fields show coherent structure
- ✅ Gaussian kernel smoothing highlights spatial patterns

### Section 2
- ✅ S profiles are smooth and sensible
- ✅ Robust to K (10 vs 20 bins)
- ✅ Curvature phenotype shows expected head/tail emphasis

### Section 4
- ✅ AUROC localizes to interpretable S regions
- ✅ **Significant bins pass FDR correction** (p < 0.05)
- ✅ Smoothed AUROC profiles show clear peaks
- ✅ Stable under bootstrap resampling

### Section 5
- ✅ Selected interval aligns with known phenotype
- ✅ **Interval is statistically significant** (permutation p < 0.05)
- ✅ Stable across bootstrap resamples (Jaccard > 0.7)
- ✅ Ablation confirms interval importance

---

## Statistical Testing Protocol

### Null Distributions: Embryo-Level Label Permutation

**Setup:**
- WT controls (n≥10) + mutants (n≥20) all mapped to reference
- Each embryo has features per S bin (e.g., c̄_k, |d̄|_k)

**Null hypothesis (per S bin k):**
- WT controls and mutants have the same feature distribution

**Null distribution:**
1. Shuffle genotype labels **at embryo level** (not snip/frame level)
2. Each embryo keeps its features, but "WT" vs "mutant" labels are randomly reassigned
3. Recompute AUROC with shuffled labels
4. Repeat 999 times → null distribution

**P-value:**
- `(1 + count(null_AUROC >= observed_AUROC)) / (n_permutations + 1)`

**Multiple testing correction:**
- Apply FDR (Benjamini-Hochberg) across S bins

**Pattern from UOT MVP:**
- See `results/mcolon/20260213_stream_d_reference_embryo/pipeline/06_difference_classification_clustering.py`
- Function: `_embryo_label_shuffle_pvalue()`
- Shuffles labels at embryo level, uses GroupKFold CV

**Bootstrap for confidence intervals:**
- Resample embryos with replacement (not snips)
- Recompute AUROC on each bootstrap sample
- Report 95% CI via percentile method

---

## Next Steps

1. ✅ **DONE**: Copy PLAN.md to dated folder
2. ✅ **DONE**: Create IMPLEMENTATION_PLAN.md
3. ✅ **DONE**: Create directory structure
4. ✅ **DONE**: Create config.yaml
5. **TODO**: Implement utility modules (`utils/*.py`)
6. **TODO**: Implement Section 0 (data loading)
7. **TODO**: Implement Section 1 (OT mapping)
8. **TODO**: Validate Section 1 before proceeding

---

## Dependencies

**Python Packages**:
- numpy, pandas, scipy, scikit-image, scikit-learn
- matplotlib, seaborn, tqdm, pyyaml

**MorphSeq Modules**:
- `src.analyze.optimal_transport_morphometrics.uot_masks.*`
- `src.analyze.spline_fitting.*`
- `src.analyze.utils.optimal_transport.*`

---

## Notes

- All statistical tests on **unsmoothed** features
- Gaussian kernel smoothing is **visualization only**
- CV always by `embryo_id` (GroupKFold)
- **Permutation tests**: Shuffle genotype labels at embryo level (not snip)
- **Bootstrap resampling**: Resample embryos with replacement (not snips)
- **FDR correction**: Apply Benjamini-Hochberg across S bins (multiple testing)
- Reference embryo and OT params **fixed** for pilot
- **WT controls required**: Without them, no null distribution for discriminability
- Optional 4 hpf robustness window (`[46, 50]`) is allowed only with per-embryo feature collapse (median across frames) before WT-vs-mutant stats

---

**Status**: Ready for implementation. Begin with Section 0 data loading.
