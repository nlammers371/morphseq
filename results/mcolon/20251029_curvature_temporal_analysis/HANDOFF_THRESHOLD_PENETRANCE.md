# Handoff: Threshold Optimization & Penetrance Analysis

**Date:** October 30, 2025
**Status:** Core implementations complete, diagnostic refinement in progress

---

## What We've Accomplished

### ✅ Part 1: Threshold Optimization Analysis

**Script:** `05_threshold_optimization.py`

**Implementation:**
- Variance-minimization threshold optimization
- Per-genotype analysis (WT, Het, Homo - **no mixing**)
- Horizon plot visualization (upper-right corner orientation, early→late time)
- Bootstrap validation: 50 iterations with 20% embryo holdout
- Parallel processing using multiprocessing (-1 CPU cores)
- Fixed nested multiprocessing bug (serial optimization within bootstrap)

**Key Features:**
- Finds optimal threshold τ* that minimizes within-group variance at future timepoints
- Classification at t46 hpf → predicts t100 hpf
- Threshold split trajectory visualization showing how τ separates embryos
- Stability metrics (SD across bootstraps)

**Outputs:**
- Horizon plots (3 per genotype): optimal thresholds, stability, variance reduction
- Threshold comparison across genotypes
- Trajectory split plots showing high/low groups
- CSV matrices with all τ* values and bootstrap statistics

**Issues Fixed:**
- Horizon plot orientation (now uses `plot_single_horizon()` from module with `origin='upper'`)
- Prediction target specified: t46→t100 hpf
- Parallel bootstrap working correctly

---

### ✅ Part 2: Penetrance Analysis

**Script:** `06_penetrance_analysis.py`

**Implementation:**
- WT reference-based penetrance (2.5-97.5% percentile envelope)
- Per-genotype analysis (WT, Het, Homo analyzed separately)
- Sample-level AND embryo-level penetrance metrics
- Onset time analysis (when do embryos first deviate?)
- Wilson confidence intervals (robust for small n)
- Bootstrap validation: 50 iterations with 20% embryo holdout
- Parallel processing for bootstrap

**Key Features:**
- WT defines "normal" range, deviations = penetrant
- Tracks penetrance over developmental time
- Quantifies dose-response: Homo > Het > WT

**Outputs:**
- WT reference envelope plot
- Penetrance curves (sample + embryo level) per genotype
- Onset time distributions (Het vs Homo)
- Comparison plots showing WT/Het/Homo together
- CSV tables with penetrance by time bin

**Issues Fixed:**
- **Percentage display bug:** Penetrance values now multiplied by 100 for plotting
- **Added WT to analysis:** WT baseline penetrance now shown for comparison
- Output filename updated: `penetrance_comparison_WT_Het_Homo.png`

---

### ✅ Part 3: Penetrance Threshold Calibration

**Script:** `06b_penetrance_threshold_calibration.py`

**Purpose:** Find WT threshold calibration that achieves biologically meaningful baseline (~1-5% WT penetrant).

**Calibrations Tested:**
1. **Percentile-based:** 90%, 95%, 99%, 99.9% bands
2. **IQR-based:** k = 1.5, 2.0, 3.0

**Key Finding:**
- **Percentile methods FAIL:** Lock in 5-10% WT penetrance by construction
- **IQR methods WORK:** Adapt to WT variance structure
  - IQR ±3.0σ achieves **2.5% WT baseline** (best)
  - IQR ±2.0σ achieves **~8% WT baseline** (acceptable)
  - IQR ±1.5σ too lenient (~15%)

**Interpretation:**
- Percentile failure suggests WT has outliers/heavy tails
- Small sample size per bin (18 WT embryos ÷ 46 bins = sparse)
- Time-varying variance handled well by IQR

**Outputs:**
- Calibration sweep: 3 genotypes × 7 calibrations grid
- WT baseline bar chart (color-coded by stringency)
- Dose-response curves for selected calibrations
- Diagnostic plots: threshold bounds + sample counts per time bin

**Current Issue:**
- Even with IQR ±3.0σ (2.5% WT baseline), seeing "spikiness" in penetrance curves
- Need to diagnose: genuine embryo noise vs. threshold artifacts

---

### ✅ Part 4: WT Noise Diagnosis

**Script:** `06c_wt_noise_diagnosis.py`

**Purpose:** Visualize individual WT embryo trajectories to assess noise vs threshold artifacts.

**Approach:**
- Uses IQR 2.0 calibration
- Counts outlier instances per embryo
- Identifies top 3 "spiky" (most outliers) and top 3 "smooth" (least outliers)

**Outputs:**
- Plot 1: Top 3 spiky embryos with outlier points highlighted
- Plot 2: Top 3 smooth embryos with outlier points highlighted
- Plot 3: All WT embryos overlaid with spiky/smooth highlighted

**Purpose:**
- If spiky embryos have many outlier circles → genuine biological noise
- If spiky embryos look clean → threshold too strict (false positives)
- Smooth embryos show what "normal" WT looks like

---

## Current Decision Point

### 🔍 Next Immediate Step: Mask & Spline Visualization

**Goal:** Determine if outlier curvature measurements are real morphology or segmentation/spline artifacts.

**Proposed Script:** `06d_visualize_troublesome_masks.py`

**What We Need:**
1. Load **raw unbinned data** (not time-binned aggregates)
2. Access mask + spline for specific `snip_id` values
3. Visualize side-by-side:
   - **Outlier timepoints:** Where embryo flagged as penetrant
   - **Clean timepoints:** Same embryo, non-flagged time

**Comparison:**
- Does "high curvature" outlier show genuinely bent embryo?
- Or is spline fitting poorly (artifacts, occlusions, bad segmentation)?
- Are smooth embryos consistently well-segmented?

**Data Requirements:**
- Location of RLE masks (check `01_` and `02_` scripts for hints)
- Location of spline data (keypoints, fitted curve)
- `snip_id` → mask/spline linkage

**Implementation Plan:**
1. Explore `01_individual_trajectories.py` and `02_horizon_plots.py` for mask loading examples
2. Identify data structure: where are masks stored? (RLE in CSV? Separate files?)
3. Create visualization showing:
   - Binary mask (decoded)
   - Overlaid spline curve
   - Annotated with curvature value, outlier status
4. Generate 4-panel comparisons per spiky embryo

---

## Key Files & Locations

### Scripts
- `05_threshold_optimization.py` - Threshold optimization via variance minimization
- `06_penetrance_analysis.py` - WT reference-based penetrance
- `06b_penetrance_threshold_calibration.py` - Calibration sweep (7 methods)
- `06c_wt_noise_diagnosis.py` - Individual WT trajectory visualization
- `06d_visualize_troublesome_masks.py` - **[NEXT TO CREATE]** Mask/spline diagnostic

### Data
- **Input:** `load_data.py` → loads combined curvature + embeddings
  - Curvature: `curvature_metrics_summary_20251017_combined.csv`
  - Embeddings: `df03_final_output_with_latents_20251017_combined.csv`
- **Genotypes:**
  - WT: 18 embryos, 502 timepoints
  - Het: 25 embryos, 501 timepoints
  - Homo: 35 embryos, 1,763 timepoints
- **Time bins:** 2 hpf windows, 46 bins covering 24-130 hpf

### Outputs
```
outputs/
├── 05_threshold_optimization/
│   ├── figures/ (10 horizon plots + trajectory splits)
│   └── tables/  (6 CSV matrices)
├── 06_penetrance_analysis/
│   ├── figures/ (8 penetrance plots)
│   └── tables/  (6 CSV tables)
├── 06b_penetrance_calibration/
│   ├── figures/ (calibration sweep + diagnostics)
│   └── tables/  (calibration summary)
├── 06c_wt_noise_diagnosis/
│   └── (3 trajectory comparison plots)
└── 06d_mask_spline_diagnosis/  [TO BE CREATED]
    ├── embryo_[id]_comparison.png
    └── summary_table.csv
```

---

## Technical Details

### Design Principles
- ✅ Per-genotype analysis (no mixing across genotypes)
- ✅ 20% embryo holdout for bootstrap (not fixed count)
- ✅ 2 hpf time bins (consistent with existing analyses)
- ✅ `normalized_baseline_deviation` as primary metric
- ✅ Parallel processing (-1 CPU cores)
- ✅ Uses standard `plot_single_horizon()` from module

### Key Metrics
- **Threshold optimization:** Variance minimization at future time
- **Penetrance:** Embryo-level (% embryos with ≥1 penetrant frame)
- **WT baseline:** Target <5%, achieved 2.5% with IQR ±3.0σ
- **Bootstrap:** 50 iterations, Wilson CIs

### Sample Sizes per Genotype
- WT: 18 embryos → 20% holdout ≈ 4 embryos
- Het: 25 embryos → 20% holdout ≈ 5 embryos
- Homo: 35 embryos → 20% holdout ≈ 7 embryos

---

## Open Questions & Future Work

### Immediate
1. **Mask/spline visualization:** Diagnose whether outliers are real or artifacts
2. **Sparse bin handling:** Should we smooth WT envelope across time?
3. **Persistent outlier definition:** Require N consecutive penetrant frames?

### Medium-term
1. **Temporal smoothing:** Apply spline to WT envelope to reduce bin-to-bin noise
2. **Pooled IQR:** Use global IQR across all time bins (less sensitive to sparse bins)
3. **Embryo-level filtering:** Flag only if persistent deviation (≥3 consecutive timepoints)

### Long-term
1. **Integration with ML models:** Compare threshold-based vs ML-based classifications
2. **Predictive power:** Does early threshold crossing predict late phenotype?
3. **Genotype stratification:** Use thresholds to refine Het/Homo subgroups

---

## How to Resume

### Option 1: Continue Mask Visualization
```bash
# 1. Explore existing scripts for mask loading
grep -r "mask" 01_individual_trajectories.py 02_horizon_plots.py

# 2. Check data structure
python -c "from load_data import get_analysis_dataframe; df, _ = get_analysis_dataframe(); print(df.columns)"

# 3. Create 06d script following plan above
```

### Option 2: Refine Calibration
```bash
# Run calibration with smoothed WT envelope
# Or test pooled IQR approach
# Modify 06b_penetrance_threshold_calibration.py
```

### Option 3: Run Existing Analyses
```bash
cd results/mcolon/20251029_curvature_temporal_analysis/

# Full pipeline
python 05_threshold_optimization.py    # ~30-60 min
python 06_penetrance_analysis.py       # ~20-40 min
python 06b_penetrance_threshold_calibration.py  # ~10-20 min
python 06c_wt_noise_diagnosis.py      # ~2-5 min
```

---

## Context for New Person

**Problem Statement:**
- Need to define optimal curvature thresholds that separate normal from abnormal embryos
- Need penetrance metric that uses WT as baseline (not classifier-based)

**Approach:**
- Threshold optimization: Find τ that minimizes future variance
- Penetrance: Mark as penetrant if outside WT reference band

**Challenge:**
- Percentile-based WT bands don't work (lock in X% penetrance)
- IQR-based bands work but show spikiness (need to diagnose)

**Current Status:**
- Core implementations done and validated
- Calibration analysis shows IQR ±3.0σ achieves 2.5% WT baseline
- Individual embryo trajectories reveal some genuinely noisy WT embryos
- **Next:** Check if noise is biological (real) or technical (segmentation artifacts)

---

## Contact & References

**Related Documentation:**
- `THRESHOLD_PENETRANCE_ANALYSIS_PLAN.md` - Original detailed plan
- `IMPLEMENTATION_NOTES_THRESHOLD_PENETRANCE.md` - Technical implementation guide
- `HANDOFF.md` (main analysis directory) - Original temporal analysis handoff

**Key Decisions Made:**
- ✅ Use IQR-based calibration (not percentile)
- ✅ Target 1-5% WT baseline penetrance
- ✅ 20% embryo holdout for bootstrap
- ✅ Embryo-level penetrance (not sample-level) for biology
- ✅ t46→t100 hpf for threshold split visualization

**Code Quality:**
- Parallel processing implemented and tested
- Functions reusable across scripts
- Consistent output structure
- Comprehensive documentation
