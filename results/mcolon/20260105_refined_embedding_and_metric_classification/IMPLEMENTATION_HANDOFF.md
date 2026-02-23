# Implementation Handoff: Plotting Enhancement Integration

**Date:** 2026-01-05
**Status:** Steps 1-3 Complete, Steps 4-5 Pending
**Next Agent:** Continue with validation script and testing

---

## What Has Been Completed

### ✅ Step 1: Enhanced Panel A in `create_comparison_figure()` (COMPLETE)
**File:** `utils/plotting.py` lines 97-207

**Changes Made:**
1. Added `auroc_color` parameter (default: `'#2ca02c'`)
2. Added `baseline_auroc_df` parameter for optional baseline comparison
3. Replaced filled stars with **open circle** significance markers (`facecolors='none'`)
4. Changed null bands to **color-matched** with alpha=0.10 (more subtle)
5. Added baseline plotting support (dashed gray line)
6. Added **color-matched stars** for highly significant (p < 0.01)
7. Added spine removal for cleaner aesthetic

**Key Visual Improvements:**
- Open circles (s=200) stand out more than filled markers
- Color-matched null bands make each comparison easier to track
- More subtle alpha (0.10 vs 0.20) reduces visual clutter
- Baseline comparison provides context (e.g., Het vs WT should be ~0.5)

**Backward Compatibility:** ✅ All new parameters are optional with sensible defaults

---

### ✅ Step 2: Added `plot_temporal_emergence()` Function (COMPLETE)
**File:** `utils/plotting.py` lines 432-575 (appended to end)

**Source:** Direct copy from `results/mcolon/20260102_labmeeting_plots/cep290_statistical_analysis_1hr.py:396-447`

**Minimal Adaptations Made:**
- Changed `COLORS` global → `colors` parameter
- Changed `TIME_BIN_WIDTH` → `time_bin_width` parameter
- Changed hardcoded cluster iteration → `results_dict.items()`

**Function Signature:**
```python
def plot_temporal_emergence(
    results_dict: Dict[str, Dict],  # {label: {'classification': df, 'summary': dict}}
    colors: Dict[str, str],         # {label: hex_color}
    time_bin_width: float = 4.0,
    title_prefix: str = "",
    figsize_per_panel: float = 5.0,
    save_path: Optional[Path] = None,
) -> plt.Figure:
```

**Features:**
- Bar plots showing AUROC per time bin
- Significant bins (p < 0.05) get full opacity + black border
- Highly significant (p < 0.01) marked with stars
- Green vertical line marks earliest significant timepoint
- Handles single or multiple comparisons (subplots)

---

### ✅ Step 3: Updated `cep290_analysis.py` (COMPLETE)
**File:** `cep290_analysis.py` lines 208-229

**Changes Made:**
1. Added import: `from utils.plotting import plot_temporal_emergence`
2. Added Plot 3 section after existing plots
3. Created results_dict and colors mapping
4. Generated temporal emergence bar plot
5. Updated output summary to mention new plot

**Code Added:**
```python
# Plot 3: Temporal Emergence Bar Plot
from utils.plotting import plot_temporal_emergence

results_dict = {
    'Penetrant_vs_Control': metric_results,
}
colors = {'Penetrant_vs_Control': '#D32F2F'}

fig3 = plot_temporal_emergence(
    results_dict,
    colors=colors,
    time_bin_width=4.0,
    title_prefix='CEP290: ',
    save_path=OUTPUT_DIR / 'cep290_temporal_emergence.png',
)
plt.close(fig3)
```

**New Output:** `cep290_temporal_emergence.png`

---

## What Remains To Be Done

### ⏳ Step 4: Create `cep290_validation_analysis.py` (PENDING)

**Purpose:**
1. **Validate** enhanced plotting produces same results as working implementation
2. **Investigate** pre-20 hpf discrepancy between implementations

**Critical Investigation Points:**

**User Report:**
> "For some reason the current implementation gave me different results for pre-20 hpf. However, there's only one experiment from that time. We found homozygous supposedly penetrant in curvature compared to what the reference is. Previous implementation didn't find that, not sure why the newer one did."

**Hypotheses to Test:**
1. **Bin Width:** Working uses `bin_width=2.0`, Refined uses `bin_width=4.0`
   - 2hr bins: 10-12, 12-14, 14-16, 16-18, 18-20
   - 4hr bins: 12-16, 16-20
   - Different binning may aggregate different samples

2. **Data Sources:**
   - Working: `results/mcolon/20251229_cep290_phenotype_extraction/data/clustering_data__early_homo.pkl`
   - Refined: `results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv`
   - Check if different embryo IDs are included/excluded

3. **Sample Thresholds:**
   - Working: `MIN_SAMPLES_PER_BIN=3`
   - Refined: Check if different threshold used

**Validation Script Structure:**
```python
# 1. Load both data sources
with open(WORKING_DATA_PATH, 'rb') as f:
    working_data = pickle.load(f)
refined_data = pd.read_csv(REFINED_DATA_PATH)

# 2. Compare embryo IDs pre-20 hpf
working_ids_pre20 = ...
refined_ids_pre20 = ...
print(f"IDs only in working: {set(working_ids_pre20) - set(refined_ids_pre20)}")
print(f"IDs only in refined: {set(refined_ids_pre20) - set(working_ids_pre20)}")

# 3. Run both bin configurations
results_2hr = compare_groups(..., bin_width=2.0, ...)
results_4hr = compare_groups(..., bin_width=4.0, ...)

# 4. Compare AUROC values pre-20 hpf
comparison_table = pd.DataFrame([
    {'time_bin': ..., 'config': '2hr', 'auroc': ..., 'pval': ..., 'n_samples': ...},
    {'time_bin': ..., 'config': '4hr', 'auroc': ..., 'pval': ..., 'n_samples': ...},
])

# 5. Generate plots with enhanced styling
fig = create_comparison_figure(
    ...,
    auroc_color='#D32F2F',  # Use new parameter
    baseline_auroc_df=baseline_results['classification'],  # Test baseline
)
```

**Expected Outputs:**
- `pre20_hpf_investigation.csv` - Comparison table showing AUROC differences
- `embryo_id_comparison.txt` - Which embryos differ between datasets
- `validation_plots/` - Plots using enhanced styling for visual comparison

**Add one more critical confound check (recommended):**
- Within-bin time distribution imbalance can create apparent signal in coarse bins.
  Example: with `bin_width=4.0`, `time_bin=12` mixes 12–16 hpf. If Penetrant embryos
  skew older (or have more rows) within that bin, embeddings/metrics correlated with time
  can yield AUROC > 0.5 even if mean trajectories look similar.
- The validation script should compute whether `predicted_stage_hpf` (within the bin)
  predicts label, e.g. AUROC(mean_hpf → label). If this is far from 0.5, treat early AUROC
  as potentially time-confounded and retry with smaller bins or within-bin matching.

#### ✅ Added scripts to do this now

Two scripts now exist under `results/mcolon/20260105_refined_embedding_and_metric_classification/`:

1) `cep290_validation_analysis.py` (pandas/sklearn required)
- Compares bin widths (2h vs 4h) using `compare_groups()`
- Compares embryo-id inclusion between the "working" `.pkl` source and refined `.csv` source
- Audits within-bin time-distribution imbalance per embryo

2) `cep290_minimal_confound_check.py` (NO pandas/sklearn required)
- Streams the refined CSV and aggregates per embryo within a chosen bin
- Reports whether **mean predicted_stage_hpf within bin** predicts label
- Reports whether **n_rows within bin** predicts label (sampling/missingness confound)

**Observed with the refined CSV (pre <20 hpf):**
- `bin_width=4`, `time_bin=12` (12–16 hpf): separability present in the coarse bin.
- `bin_width=2`, `time_bin=12` (12–14 hpf): little/no separability.
- `bin_width=2`, `time_bin=14` (14–16 hpf): separability present.

This strongly suggests the “12 hpf” signal in 4h bins is driven by the later half (14–16 hpf),
and that the apparent Panel A vs Panel C contradiction can be a binning/labeling artifact
(interpreting `time_bin=12` as “12 hpf” rather than “12–16 hpf”).

---

### ⏳ Step 5: Testing and Final Validation (PENDING)

**Test Checklist:**

**Visual Tests:**
- [ ] Null distribution bands visible and color-matched
- [ ] Open circles render correctly (not filled)
- [ ] Stars appear for p < 0.01 points
- [ ] Temporal emergence bar plots highlight significant bins
- [ ] Green vertical line marks earliest significant timepoint
- [ ] Baseline comparison (if provided) shows as dashed gray

**Quantitative Tests:**
- [ ] AUROC values match working implementation (when using same bin_width)
- [ ] P-values match exactly
- [ ] Run cep290_analysis.py and check all 3 plots generate
- [ ] Verify backward compatibility (existing calls still work)

**Integration Tests:**
- [ ] Test with baseline_auroc_df parameter
- [ ] Test with different auroc_color values
- [ ] Test plot_temporal_emergence() with single comparison
- [ ] Test plot_temporal_emergence() with multiple comparisons

---

## Key Files Modified

### Primary Changes:
1. **`utils/plotting.py`** (~146 lines added/modified)
   - Lines 10-27: Added new parameters to `create_comparison_figure()`
   - Lines 70-73: Updated docstring
   - Lines 97-207: Enhanced Panel A implementation
   - Lines 432-575: New `plot_temporal_emergence()` function

2. **`cep290_analysis.py`** (~23 lines added)
   - Lines 208-229: Added temporal emergence plot generation
   - Line 271: Updated output summary

### Reference Files (DO NOT MODIFY):
- **Preferred Style Source:** `results/mcolon/20260102_labmeeting_plots/cep290_statistical_analysis_1hr.py`
  - Lines 248-322: `plot_cluster_vs_wt_auroc()` (source for Panel A style)
  - Lines 396-447: `plot_temporal_emergence()` (source for new function)

---

## Design Decisions Made

### 1. Color-Matched Null Bands (Alpha=0.10)
**Rationale:** Each comparison gets its own colored band, making it easier to visually track null distribution context. More subtle than previous alpha=0.20.

### 2. Open Circles vs Filled Stars
**Rationale:** Open circles (s=200, facecolors='none') stand out more and don't obscure data points. User feedback indicated preference for this style.

### 3. Optional Baseline Support
**Rationale:** Allows showing Het vs WT as dashed gray baseline for context (Het should be ~0.5 if similar to WT).

### 4. Backward Compatibility
**Rationale:** All new parameters are optional, so existing code continues to work without modification.

---

## Known Issues / Pending Questions

### Issue 1: Pre-20 hpf Discrepancy
**Status:** NEEDS INVESTIGATION
**Impact:** CRITICAL - affects interpretation of early phenotype detection
**Next Step:** Create validation script to compare bin widths and data sources

### Issue 2: Script Execution Test
**Status:** PENDING
**Note:** `cep290_analysis.py` was updated but execution was interrupted (exit code 137 - likely memory/timeout)
**Next Step:** Run script with longer timeout or in background to verify plots generate correctly

---

## Next Agent Instructions

### Immediate Next Steps:

1. **Create `cep290_validation_analysis.py`** (Priority 1)
   - Follow structure outlined in "Step 4" above
   - Focus on pre-20 hpf investigation
   - Compare 2hr vs 4hr bins
   - Document findings in `output/cep290/pre20_hpf_investigation.csv`

2. **Test Enhanced Plotting** (Priority 2)
   - Run `python cep290_analysis.py` (may need longer timeout)
   - Verify 3 plots generate:
     - `cep290_metric_auroc_only.png` (with enhanced Panel A)
     - `cep290_embedding_vs_metric.png` (with enhanced Panel A)
     - `cep290_temporal_emergence.png` (NEW)
   - Visual inspection against working implementation plots

3. **Document Investigation Findings** (Priority 3)
   - Update this handoff with pre-20 hpf explanation
   - Add recommendation for future analyses (2hr vs 4hr bins)
   - Update HANDOFF.md with new plotting capabilities

### Optional Extensions:

4. **Apply to B9D2 Analyses** (if time permits)
   - Add temporal emergence plots to `b9d2_hta_analysis.py`
   - Add temporal emergence plots to `b9d2_ce_analysis.py`
   - Use consistent color scheme

5. **Enhanced AUROC Overlay** (if requested)
   - Could enhance `create_auroc_comparison_figure()` with same style
   - Currently simple implementation, could add null bands + open circles

---

## Testing Commands

```bash
# Change to refined framework directory
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20260105_refined_embedding_and_metric_classification

# Test enhanced plotting (may need longer timeout)
python cep290_analysis.py

# Visual inspection of outputs
ls -lh output/cep290/*.png

# Check plot file sizes (should be >50KB if generated correctly)
du -h output/cep290/*.png
```

---

## Success Criteria

### Minimum Viable:
- [ ] All 3 plots generate from cep290_analysis.py
- [ ] Enhanced Panel A shows open circles and color-matched null bands
- [ ] Temporal emergence bar plot shows significance highlighting
- [ ] Pre-20 hpf discrepancy explained (bin width or data difference)

### Full Success:
- [ ] Validation script confirms AUROC values match (with same bin_width)
- [ ] Visual comparison shows plots match preferred style
- [ ] Documentation updated with investigation findings
- [ ] Recommendation provided for bin width choice

---

## Contact / Questions

**Plan File:** `/net/trapnell/vol1/home/mdcolon/.claude/plans/twinkling-wishing-sedgewick.md`

**Key Context:**
- User prefers style from `cep290_statistical_analysis_1hr.py` (open circles, color-matched bands)
- Bar plots are complementary analysis (not replacing Panel A)
- Pre-20 hpf discrepancy is critical to resolve (only one experiment in that range)

**If Stuck:**
- Check plan file for detailed implementation notes
- Compare against source files in `results/mcolon/20260102_labmeeting_plots/`
- User is investigating difference between implementations

---

## Change Summary

**Files Modified:** 2
**Lines Added:** ~169
**Lines Modified:** ~23
**New Functions:** 1 (`plot_temporal_emergence`)
**Enhanced Functions:** 1 (`create_comparison_figure` Panel A)

**Backward Compatible:** ✅ Yes (all new parameters optional)
**Tests Passing:** ⏳ Pending (script execution interrupted)
**Ready for Validation:** ✅ Yes (code complete, needs testing)
