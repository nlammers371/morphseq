# Pipeline Benchmarking - Agent Handoff Document

**Date**: October 27, 2025
**Project**: MorphSeq Pipeline Performance Analysis
**Status**: Ready to execute benchmarking script

---

## 📋 Current State

### What Was Accomplished

1. **Created Standalone Mask Cleaning Module**
   - Location: `segmentation_sandbox/scripts/utils/mask_cleaning.py`
   - Function: `clean_embryo_mask(mask, verbose=False)`
   - Returns: `(cleaned_mask, cleaning_stats)`
   - Reusable across entire pipeline

2. **Integrated Mask Cleaning into Test Scripts**
   - `results/mcolon/20251024/test_head_tail_identification.py` - Updated to import from module
   - `results/mcolon/20251024/diagnose_mask_issues.py` - Updated to import from module
   - `results/mcolon/20251024/debug_b05_cleaning.py` - Kept inline for step-by-step debugging

3. **Created Comprehensive Benchmarking Script**
   - Location: `results/mcolon/20251027/benchmark_pipeline_performance.py`
   - **Ready to run** - all code is complete and tested
   - Will profile mask cleaning + geodesic vs PCA analysis

---

## 🎯 Next Steps (For Next Agent)

### Step 1: Run the Benchmark
```bash
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq
python results/mcolon/20251027/benchmark_pipeline_performance.py
```

**What it will do:**
- Load 10 random embryos from `df03_final_output_with_latents_20251017_part2.csv`
- Time each cleaning step
- Time geodesic centerline extraction + B-spline fitting
- Time PCA centerline extraction + B-spline fitting
- Generate visualizations and performance report

**Expected runtime**: ~1-2 minutes total (10 embryos)

**Outputs generated:**
- `benchmark_results_10embryos.csv` - Detailed timing data
- `benchmark_timing_breakdown.png` - Bar charts of each step
- `benchmark_scaling_analysis.png` - Scaling estimates and throughput
- `benchmark_summary.txt` - Text report with recommendations

### Step 2: Review Results

Check the summary report for:
- **Bottleneck identification**: Which steps are slowest?
- **Geodesic vs PCA comparison**: Which method is faster?
- **Throughput estimates**: Embryos/hour per method
- **Scaling projections**: Can we process 10K, 100K embryos?
- **Recommendation**: Which method to use for production?

### Step 3: Decision Making

Based on the benchmark results:

**If PCA is significantly faster (>2x speedup):**
- ✅ Use PCA method for full pipeline deployment
- Update production code to use PCA by default

**If Geodesic is more accurate but slower:**
- ⚖️ Trade-off decision needed
- Consider: Quality vs. Speed requirements
- Option: Use PCA for bulk processing, Geodesic for validation

**If both methods are too slow (>5s per embryo):**
- 🔧 Optimization needed
- Consider parallelization (multiprocessing)
- Consider GPU acceleration
- Profile to find specific bottleneck operations

### Step 4: Optimization (If Needed)

**If mask cleaning is the bottleneck:**
- Cache morphological structuring elements (disk(5), disk(10), etc.)
- Reduce closing iterations (cap at 3 instead of 5)
- Skip skeleton computation in cleaning (only compute when needed)

**If geodesic graph building is the bottleneck:**
- Use spatial indexing (KD-tree) instead of brute-force neighbor search
- Parallelize graph construction

**If PCA slicing is the bottleneck:**
- Reduce number of slices (100 → 50)
- Downsample mask before PCA

---

## 📁 Key Files & Locations

### Mask Cleaning Module
```
segmentation_sandbox/scripts/utils/mask_cleaning.py
```
- Main function: `clean_embryo_mask(mask, verbose=False)`
- 5-step pipeline: debris removal, closing, fill holes, opening, keep largest
- Returns cleaned mask + detailed stats

### Benchmarking Script
```
results/mcolon/20251027/benchmark_pipeline_performance.py
```
- Ready to execute
- Tests 10 random embryos
- Compares Geodesic vs PCA methods

### Test Scripts (Updated)
```
results/mcolon/20251024/test_head_tail_identification.py
results/mcolon/20251024/diagnose_mask_issues.py
results/mcolon/20251024/debug_b05_cleaning.py
```
- First two import from mask_cleaning module
- Last one has inline implementation for debugging

### Analysis Methods
```
results/mcolon/20251022/geodesic_bspline_smoothing.py
results/mcolon/20251022/test_pca_smoothing.py
```
- Geodesic: `GeodesicBSplineAnalyzer` class
- PCA: `PCACurvatureAnalyzer` class

---

## 🧪 Data Source

**CSV File:**
```
/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/metadata/build06_output/df03_final_output_with_latents_20251017_part2.csv
```
- Total embryos: 1,129
- Experiment: 20251017_part2
- Columns include: `snip_id`, `mask_rle`, `mask_height_px`, `mask_width_px`, etc.

---

## 📊 Morphology Analysis Findings (October 27, 2025)

### Solidity Threshold for Opening Operation

**Analysis**: Computed morphology metrics on 500 random embryo samples
**Key Finding**: Opening operation should only be applied to masks with **solidity < 0.6**

**Results**:
- Analyzed distribution of solidity, extent, eccentricity, and other metrics
- Found that masks with solidity >= 0.6 are already solid enough
- Applying opening to high-solidity masks wastes computation time
- Low-solidity masks (< 0.6) benefit from opening to remove spindly protrusions

**Impact**:
- **Performance improvement**: Skips expensive opening operation for ~40% of masks
- **Quality preservation**: Only applies smoothing when actually needed
- **Data source**: `when_to_compute_opening_cleaning/morphology_metrics_500samples.csv`

---

## 🔧 Mask Cleaning Pipeline Details

### 5-Step Process:

1. **Remove Small Debris** (<10% of total area)
   - Filters out tiny artifacts
   - Keeps significant components

2. **Iterative Adaptive Closing** (Connect Components)
   - Starts with radius = `max(5, perimeter/100)`
   - Increases by 5px each iteration
   - Max 5 iterations, cap at 50px
   - Stops when 1 component achieved

3. **Fill Holes**
   - Binary fill for internal gaps
   - Uses `scipy.ndimage.binary_fill_holes`

4. **Conditional Adaptive Opening** (Smooth & Remove Spindly Parts)
   - **Only applied if solidity < 0.6** ← **NEW: Based on morphology analysis**
   - Skipped for already-solid masks (saves computation time)
   - Radius = `max(3, perimeter/150)` ← **Gentler than original /100**
   - Removes thin protrusions while preserving thin tails
   - Analysis showed opening only needed for low-solidity masks

5. **Final Safety Check**
   - Keeps largest component after opening
   - Ensures single connected output

### Key Parameters:
- **Opening radius divisor**: Changed from `/100` to `/150` for gentler smoothing
- **Closing max iterations**: 5
- **Closing max radius**: 50px
- **Debris threshold**: 10% of total area

---

## 📊 What the Benchmark Measures

### Cleaning Steps:
- `clean_debris` - Remove small components
- `clean_closing` - Iterative closing to connect
- `clean_holes` - Fill internal gaps
- `clean_opening` - Smooth protrusions
- `clean_largest` - Keep largest component
- `clean_total` - Total cleaning time

### Geodesic Steps:
- `geo_skeleton` - Skeletonize mask
- `geo_graph` - Build 8-connected graph
- `geo_dijkstra` - Find geodesic path
- `geo_bspline` - Fit B-spline (s=5.0)
- `geo_total` - Total geodesic time

### PCA Steps:
- `pca_centerline` - Extract centerline via PCA slicing
- `pca_bspline` - Fit B-spline (s=5.0)
- `pca_total` - Total PCA time

### Pipeline Totals:
- `pipeline_geodesic_total` - Cleaning + Geodesic
- `pipeline_pca_total` - Cleaning + PCA
- `speedup_factor` - Geodesic time / PCA time

---

## 🐛 Known Issues & Decisions

### Issue 1: Opening Operation Cuts Tail
- **Problem**: Original opening radius (perimeter/100) was too aggressive
- **Solution**: Reduced to perimeter/150 for gentler smoothing
- **Result**: Preserves thin tails while removing spindly artifacts

### Issue 2: Disconnected Components
- **Problem**: Some masks have 2+ large components (e.g., B05: 75%/25% split)
- **Solution**: Iterative closing with increasing radius
- **Max radius**: 50px to prevent over-expansion

### Issue 3: Taper Direction Method
- **Problem**: Midpoint-based measurement made performance worse
- **Solution**: Reverted to simple full-centerline gradient method
- **File**: `test_head_tail_identification.py`

### Decision: Speed vs Quality
- **Optimization postponed**: User said "let's worry about this another day"
- **Current focus**: Measure performance first, optimize later
- **Next step**: Benchmark will reveal if optimization is needed

---

## 💡 Expected Outcomes

### If Everything Works:
- Benchmark completes in ~1-2 minutes
- Clear timing breakdown for each step
- Bottleneck identification (which step is slowest)
- Recommendation on Geodesic vs PCA
- Scaling projections for 1K, 10K, 100K embryos

### Possible Results:

**Scenario A: PCA is Much Faster (>2x)**
- Recommendation: Use PCA for production
- Action: Update full pipeline to use PCA by default

**Scenario B: Similar Performance**
- Recommendation: Choose based on accuracy, not speed
- Action: Run accuracy comparison next

**Scenario C: Both Too Slow (>5s/embryo)**
- Recommendation: Optimization needed
- Action: Profile individual operations, parallelize, or use GPU

---

## 📝 Notes for Next Agent

### Important Context:
1. The mask cleaning module is **production-ready** and can be used anywhere
2. The benchmark script is **complete** - just run it
3. User wants to know if pipeline can scale to **full dataset** (thousands/millions)
4. **PCA method** found in `results/mcolon/20251022/test_pca_smoothing.py`
5. **Geodesic method** found in `results/mcolon/20251022/geodesic_bspline_smoothing.py`

### If Errors Occur:
- Check imports are correct (mask_cleaning module, etc.)
- Verify CSV path exists
- Ensure all dependencies installed (scipy, scikit-learn, skimage)
- Check Python environment is activated

### After Benchmarking:
- Review `benchmark_summary.txt` for recommendations
- Share visualizations with user
- Discuss trade-offs (speed vs accuracy)
- Decide on next steps based on results

---

## ✅ Checklist for Next Agent

- [ ] Run benchmark script
- [ ] Review generated outputs (CSV, PNGs, TXT)
- [ ] Analyze bottleneck steps
- [ ] Compare Geodesic vs PCA performance
- [ ] Check scaling projections (can we handle 10K+ embryos?)
- [ ] Report findings to user
- [ ] Recommend: Geodesic or PCA for production
- [ ] If too slow: Propose optimization strategy

---

## 📞 Contact Points

**Previous Agent**: Completed mask cleaning module integration and created benchmark script
**User Requirement**: "I need to know if this can scale up to a whole pipeline level"
**Target**: Determine if pipeline is fast enough for production deployment

---

## 🔬 PCA vs Geodesic Comparison Analysis (October 27, 2025 - COMPLETED)

### What Was Accomplished

**1. Comprehensive Comparison Pipeline**
- Analyzed 1000 embryos from both part1 and part2 datasets
- Parallelized execution (191 workers) for performance
- Both methods extract splines from **cleaned masks** (5-step pipeline with conditional opening)
- Spline coordinates stored in results CSV for visualization

**2. Spline Alignment Before Comparison**
- Created `bodyaxis_spline_utils.py` module
- Implements head/tail identification based on width tapering
- Automatically aligns PCA and Geodesic splines to same orientation before comparison
- Prevents false disagreements from reversed spline directions

**3. Bimodal Distribution Analysis**
- Fitted Gaussian Mixture Model to Hausdorff distances
- Found natural threshold: **114.78 pixels**
- Classification: 97.5% agree, 2.5% disagree
- Two clear populations: "methods agree" vs "methods disagree"

**4. Comprehensive Visualizations Generated**
- `bimodal_fit_hausdorff.png` - GMM analysis showing two populations
- `distribution_analysis.png` - 3-panel histograms (mean, max, Hausdorff distances)
- `hausdorff_histogram_with_bins.png` - Full distribution with 15px interval markers
- `hausdorff_interval_examples.png` - Spline overlays binned by distance (0-15px, 15-30px, etc.)
- `threshold_comparison_examples.png` - Examples above/below threshold
- All show **cleaned masks** (as used for spline extraction) with overlaid splines:
  - **Blue** = PCA spline
  - **Red** = Geodesic spline

**5. Decision Rules & Metrics**
- Analyzed correlations between mask metrics and spline disagreement
- Found key predictors: solidity, extent, eccentricity, circularity
- Generated threshold recommendations for when to use Geodesic vs PCA

---

## ⚠️ Known Issues Requiring Fixes (For Next Agent)

### Issue 1: Geodesic Through-Fin Problem
**Symptom**: Geodesic centerline sometimes goes through the tail fin instead of along the body axis

**Example**: `20251017_part2_H07_e01_t0008`
- Hausdorff distance: 55.88 px (should be "agree" but shows disagreement)
- Geodesic path extends slightly further into tail fin
- PCA correctly identifies body axis termination

**Root Cause**:
- Skeleton-based geodesic path includes fin structures
- Graph endpoint detection finds furthest skeleton point (in fin)
- Need to distinguish body axis from appendages

**Proposed Solutions**:
1. **Fin detection**: Identify fins based on width-to-length ratio or branching
2. **Skeleton pruning**: Remove skeleton branches with high branching factor
3. **Medial axis focus**: Weight graph toward thicker body regions
4. **Endpoint refinement**: Select endpoints based on body width (head/tail have max width before taper)

**Files to modify**:
- `results/mcolon/20251022/geodesic_bspline_smoothing.py` (GeodesicBSplineAnalyzer class)
- Consider creating `identify_fins()` function in `bodyaxis_spline_utils.py`

### Issue 2: PCA Fails on Highly Curved Embryos
**Symptom**: PCA centerline extraction fails when embryo curvature is extreme

**Example**: `20251017_part2_E07_e01_t0038`
- Hausdorff distance: 141.60 px (large disagreement)
- Very curved/coiled embryo configuration
- PCA slicing method fails to capture tight curvature

**Root Cause**:
- PCA assumes approximately linear primary axis
- Perpendicular slicing fails when curvature is too high
- Slices may intersect body multiple times or miss regions

**Proposed Solutions**:
1. **Detect high curvature**: Pre-compute rough centerline curvature
2. **Adaptive slicing**: Use smaller slice spacing for curved regions
3. **Iterative refinement**: Start with coarse PCA, then refine locally
4. **Fallback to Geodesic**: If curvature exceeds threshold, automatically use Geodesic
5. **Hybrid approach**: Use Geodesic for initial path, then PCA for local smoothing

**Detection Criteria**:
- Eccentricity > 0.98 (very elongated)
- Extent < 0.35 (occupies small portion of bounding box = highly curved)
- Solidity < 0.6 (non-convex = likely curved)

**Files to modify**:
- `results/mcolon/20251022/test_pca_smoothing.py` (PCACurvatureAnalyzer class)
- Add curvature detection to `compare_pca_vs_geodesic.py`
- Create decision logic: if high curvature detected → use Geodesic only

### Issue 3: Conditional Opening Threshold (RESOLVED ✓)
**Status**: Already implemented
- Opening only applied when solidity < 0.6
- Saves computation time (~40% of masks skip opening)
- Documented in mask_cleaning.py

---

## 📊 Current Performance Metrics

**From 1000 embryo analysis:**
- PCA success rate: 100% (1000/1000)
- Geodesic success rate: 100% (1000/1000)
- Both methods succeed: 100% (1000/1000)
- Agreement rate (Hausdorff < 114.78px): 97.5%
- Disagreement rate: 2.5%

**Distance Statistics (for agreeing cases):**
- Mean distance: [see distribution_analysis.png]
- Max distance: [see distribution_analysis.png]
- Hausdorff distance: mean ~45px, median ~38px (see bimodal fit)

**Speed Comparison** (from earlier benchmarking):
- PCA: ~5.2s per embryo (696 embryos/hour)
- Geodesic: ~14.4s per embryo (250 embryos/hour)
- **PCA is 2.78x faster**

---

## 🎯 Recommendations for Production Pipeline

### Method Selection Strategy

**Option 1: PCA-First with Geodesic Fallback**
```python
# Detect problematic cases before extraction
if eccentricity > 0.98 or extent < 0.35 or solidity < 0.6:
    use_geodesic = True  # High curvature likely
else:
    use_pca = True  # Fast path

# After extraction, validate
if pca_failed or curvature_metric_high:
    use_geodesic = True  # Fallback
```

**Option 2: Always Use PCA (Fastest)**
- Acceptable if 97.5% agreement is sufficient
- 2.5% of embryos may have suboptimal centerlines
- Fastest processing: 696 embryos/hour

**Option 3: Hybrid Method** (Recommended for accuracy)
1. Run PCA first (fast)
2. Detect high curvature regions in PCA result
3. Use Geodesic only for high-curvature embryos
4. Best balance of speed and accuracy

### Next Steps Priority

1. **Fix Geodesic fin issue** (high priority)
   - Impacts accuracy even on "easy" cases
   - Example: 20251017_part2_H07_e01_t0008

2. **Fix PCA curvature issue** (high priority)
   - Impacts 2.5% of embryos
   - Example: 20251017_part2_E07_e01_t0038

3. **Implement hybrid selection logic** (medium priority)
   - Use mask metrics to predict which method will work better
   - Thresholds already identified in decision rules

4. **Validate on full dataset** (low priority)
   - Run on all embryos in part1 and part2
   - Generate quality control reports

---

Good luck! The comparison framework is complete - now focus on fixing the edge cases.

---

## 🔧 Skeleton Pruning Implementation (October 27, 2025 - NEW)

### What Was Implemented

**1. Width-Based Skeleton Pruning Utility**
- Location: `segmentation_sandbox/scripts/utils/bodyaxis_spline_utils.py`
- Function: `prune_skeleton_for_geodesic(skeleton, mask, width_percentile_threshold=25.0)`
- Strategy: Remove skeleton pixels in narrow regions (fins/protrusions)
- Uses distance transform to identify thin regions
- Configurable threshold (10th, 25th, 50th percentile)

**2. Comprehensive Analysis Scripts**
- `test_pruning_single_embryo.py` - Single embryo visualization and testing
- `test_pruned_geodesic.py` - Full pipeline comparison (200 embryos)
  - Compares original vs pruned geodesic methods
  - Tests multiple pruning thresholds
  - Generates scatter plots: mask metrics vs Hausdorff distance
  - Outputs improvement statistics

**3. Key Visualizations Generated**
- `test_pruning_single_embryo.png` - Visual proof of pruning concept
  - Shows original skeleton vs pruned at 10th, 25th, 50th percentiles
  - Distance transform overlay showing width-based logic
  - Color-coded visualization of kept vs removed skeleton pixels
- `metrics_vs_hausdorff_scatter.png` - 6-panel scatter plot grid
  - Solidity, eccentricity, extent, area, perimeter, circularity
  - Each vs Hausdorff distance (PCA-Geodesic disagreement)
  - Pearson and Spearman correlations displayed
  - Agreement/disagreement regions color-coded
- `correlation_heatmap.png` - Full correlation matrix
  - Shows which metrics best predict method disagreement
- `pruning_improvement_comparison.png` - Before/after analysis
  - Original geodesic vs pruned geodesic Hausdorff distances
  - 3 panels for 10th, 25th, 50th percentile pruning
  - Shows improvement/worsening counts

### Pruning Algorithm Details

**Method**: Width-based percentile pruning
```python
# Compute distance transform (local radius)
distance_map = distance_transform_edt(mask)

# Get widths at skeleton points (diameter = 2 * radius)
skeleton_widths = 2 * distance_map[skeleton]

# Find threshold (e.g., 25th percentile)
width_threshold = np.percentile(skeleton_widths, 25.0)

# Remove skeleton in regions thinner than threshold
pruned_skeleton = skeleton.copy()
pruned_skeleton[distance_map * 2 < width_threshold] = 0
```

**Example Results** (embryo `20251017_part2_H07_e01_t0008`):
- Original skeleton: 1068 pixels
- 10th percentile: removes 104 px (9.7%), threshold=26px
- 25th percentile: removes 267 px (25.0%), threshold=38px
- 50th percentile: removes 534 px (50.0%), threshold=81px

### Analysis Findings

**Correlation Analysis** (from `metrics_vs_hausdorff_scatter.png`):
- **Best predictors of disagreement:**
  1. Extent (low extent = curved embryo)
  2. Solidity (low solidity = non-convex shape)
  3. Eccentricity (high eccentricity = elongated)
- **Weak predictors:**
  - Area, perimeter (size doesn't predict curvature)

**Decision Tree Implications:**
```python
# Use existing mask metrics for fast triage
if extent < 0.35 or solidity < 0.6 or eccentricity > 0.98:
    use_geodesic = True  # Highly curved, PCA will fail
else:
    use_pca = True  # Fast path, works well
```

### Next Steps

**1. Run Full Pruned Analysis**
```bash
python results/mcolon/20251027/test_pruned_geodesic.py
```
- Processes 200 embryos (150 agree, 50 disagree)
- Tests all three pruning thresholds (10th, 25th, 50th)
- Generates improvement statistics

**2. Validate Fin Problem Fix**
- Check if pruning fixes `20251017_part2_H07_e01_t0008` (known fin case)
- Compare original Hausdorff (55.88 px) vs pruned
- Visualize before/after splines

**3. Determine Optimal Threshold**
- Balance: too aggressive (removes body) vs too gentle (keeps fins)
- Recommendation: start with 25th percentile
- Adaptive threshold based on embryo characteristics

**4. Implement Curvature Metrics**
- Straightness metric: perpendicular distance variance
- Tortuosity: geodesic/Euclidean distance ratio
- These could replace/augment existing mask metrics

**5. Create Decision Logic**
- Hybrid PCA/Geodesic selector
- Use pruned Geodesic for curved embryos
- Integrate into production pipeline

### Files Created/Modified

**New Files:**
- `segmentation_sandbox/scripts/utils/bodyaxis_spline_utils.py` (added pruning function)
- `results/mcolon/20251027/test_pruning_single_embryo.py`
- `results/mcolon/20251027/test_pruned_geodesic.py`

**Visualizations Generated:**
- `test_pruning_single_embryo.png`
- `metrics_vs_hausdorff_scatter.png`
- `correlation_heatmap.png`
- `pruning_improvement_comparison.png` (will be generated after full run)

---

## 📊 New Outputs

When you run the script, it will generate:
1. `pca_vs_geodesic_comparison_1000embryos.csv` - Full results
2. `bimodal_fit_hausdorff.png` - GMM analysis with threshold
3. `distribution_analysis.png` - 3-panel distance histograms
4. `hausdorff_interval_examples.png` - Spline overlays binned by 15px Hausdorff intervals
5. `threshold_comparison_examples.png` - Examples above/below the computed threshold
6. `hausdorff_histogram_with_bins.png` - Distribution showing interval bins and threshold
7. Console output with best/worst case IDs

**Visualization Details:**
- **Interval Analysis**: Shows 2 examples per 15px Hausdorff bin (0-15, 15-30, 30-45, etc.)
- **Threshold Comparison**: Shows 3 examples below threshold (agreement) and 3 above (disagreement)
- **Spline Overlays**: PCA (blue) vs Geodesic (red) centerlines on actual embryo masks
- **Histogram**: Shows full Hausdorff distribution with interval markers and computed threshold
