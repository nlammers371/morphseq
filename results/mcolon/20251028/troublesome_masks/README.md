# Troublesome Masks Investigation

## Purpose
This directory contains scripts and results for investigating 4-5 embryos that failed during previous batch processing. The goal is to:
1. Reproduce the current behavior with the latest pipeline
2. Determine if the failures persist or have been resolved
3. If failures persist, test alternative preprocessing methods to fix them

## Problematic Embryos

### Embryo List (4 confirmed)
1. **20251017_combined_A02_e01_t0064**
   - Experiment: 20251017_combined
   - Well: A02
   - Timepoint: t0064

2. **20250512_E06_e01_t0086**
   - Experiment: 20250512
   - Well: E06
   - Timepoint: t0086

3. **20251017_combined_C04_e01_t0114**
   - Experiment: 20251017_combined
   - Well: C04
   - Timepoint: t0114

4. **20251017_combined_F11_e01_t0065**
   - Experiment: 20251017_combined
   - Well: F11
   - Timepoint: t0065

*Note: User mentioned "these five embryos" but listed 4. May need to confirm if there's a 5th.*

## Investigation Plan

### Phase 1: Reproduce Current Behavior
- **Script**: `process_troublesome_embryos.py`
- **Pipeline**: Current default (geodesic + Gaussian blur preprocessing)
- **Outputs**:
  - Individual visualizations showing processing stages for each embryo
  - Summary grid showing all embryos
  - CSV with success/failure tracking and metrics

### Phase 2: Analysis & Remediation (TBD based on Phase 1 results)
- **If all succeed**: Investigate what changed in the pipeline since previous failures
- **If some/all fail**: Test alternative preprocessing methods:
  - Different Gaussian blur sigma values (5, 10, 15, 20, 25)
  - Alpha shape preprocessing with various alpha values (30, 50, 70, 90, 110)
  - Other mask refinement approaches

## Current Pipeline Configuration
- **Mask cleaning**: 5-step pipeline
  1. Remove small debris (<10% area)
  2. Iterative adaptive closing (radius = perimeter/100, max 5 iterations)
  3. Fill holes
  4. Conditional opening (only if solidity < 0.6, radius = perimeter/150)
  5. Keep largest component

- **Preprocessing**: Gaussian blur
  - **sigma = 15.0** (OPTIMIZED - see Phase 3 results below)
  - threshold = 0.7

- **Centerline extraction**: Geodesic method
  - B-spline smoothing = 5.0
  - Fast mode (O(N) graph building)
  - Connected component filtering (keeps only largest skeleton component)
  - Exhaustive endpoint detection (no sampling bias)
  - Random seed = 42

## Files

### Scripts
- `process_troublesome_embryos.py` - Main processing script
- `diagnose_failures.py` - Diagnostic script to trace failure points

### Outputs
- `results.csv` - Detailed results for each embryo
- `{embryo_id}_visualization.png` - Individual 4-panel visualizations
- `summary_grid.png` - All embryos in one view
- `{embryo_id}_diagnostic.png` - Step-by-step diagnostic visualizations (for failed embryos)
- `failure_analysis.csv` - Detailed metrics at each processing stage

## PROGRESS

### Phase 1: Reproduce Current Behavior ✓ COMPLETE
**Date:** 2025-10-28

**Results:** 2/4 successful, 2/4 failed

**Successful Embryos:**
- ✓ `20251017_combined_A02_e01_t0064` - Length: 3322.34 μm, Mean κ: 0.0010 1/μm
- ✓ `20250512_E06_e01_t0086` - Length: 2175.53 μm, Mean κ: 0.0022 1/μm

**Failed Embryos:**
- ✗ `20251017_combined_C04_e01_t0114` - Error: Empty centerline returned
- ✗ `20251017_combined_F11_e01_t0065` - Error: Empty centerline returned

**Key Observations:**
- Both failures occur during Stage 3 (Centerline extraction)
- Preprocessing completes successfully (Stage 1 & 2 work fine)
- Failed embryos show 90%+ area retention after preprocessing
- No obvious preprocessing issues visible

**Next Step:** Diagnose WHERE in the geodesic pipeline the failure occurs

### Phase 2: Detailed Failure Point Analysis ✓ COMPLETE
**Date:** 2025-10-28

**Root Cause Found:**
1. **Sampling-based endpoint detection** (50 random skeleton points) was unreliable for fragmented skeletons
2. **Missing disconnected component cleanup** - graph contained isolated skeleton fragments

**Solution Implemented:**
- ✓ Replaced sampling with exhaustive endpoint search
- ✓ Added connected component filtering to keep only largest skeleton component
- ✓ Vectorized computation for robustness

**Result:** Both previously failing embryos now pass:
- ✓ `20251017_combined_C04_e01_t0114` - 3420.44 μm
- ✓ `20251017_combined_F11_e01_t0065` - 2961.97 μm

### Phase 3: Preprocessing Optimization ✓ COMPLETE
**Date:** 2025-10-28

**Goal:** Find optimal Gaussian blur sigma value to prevent fin/artifact extension

**Test Setup:**
- Tested sigma values: 2, 5, 10, 15, 20, 25, 30
- Tested embryos: 
  - `20251017_combined_A02_e01_t0064`
  - `20251017_combined_D12_e01_t0057`
  - `20251017_combined_D10_e01_t0108`

**Findings:**
- **sigma=10** (old default): Variable results, occasional fin extension
- **sigma=15** (optimized): Consistent results across all test embryos, prevents fin artifacts
- **sigma=20+**: Over-smoothing, may lose fine embryo details

**Recommendation:** Update pipeline to use **sigma=15.0** for Gaussian blur preprocessing

**Output:** `{embryo_id}_gaussian_sweep.png` shows side-by-side comparison of all sigma values

### Phase 4: Sigma-Threshold Parameter Sweep ✓ COMPLETE
**Date:** 2025-10-28

**Goal:** Systematically evaluate sigma (10-50) and threshold (0.4-0.9) combinations

**Findings:**
- **Optimal parameters vary by embryo** - No single sigma/threshold works best for all
- **Trade-off observed:** Higher sigma smooths fins but may lose fine structure
- **Longest path criterion:** Different params produce different centerline lengths
  - Sigma=20, θ=0.7 performs well as default
  - Other combinations: sigma=30, θ=0.1 / sigma=25, θ=0.6 also promising

**Key Insight:**
For maximum accuracy, the algorithm could:
1. Try multiple sigma/threshold combinations
2. Extract centerline for each
3. Select the **longest valid centerline** (most likely to be true body axis)
4. Trade-off: Computational cost increases ~5-10x for exhaustive search

**Current Decision:** Use sigma=20, θ=0.7 as production default
- Good balance between accuracy and speed
- Stable across embryo morphologies
- See `sigma_threshold_sweep/` subfolder for detailed parameter exploration

## Notes
- All embryos will be loaded from their respective CSV files in `morphseq_playground/metadata/build06_output/`
- Visualizations will show: original mask, cleaned mask, preprocessed mask, and final centerline overlay
- Diagnostic visualizations will show each internal processing stage with metrics

## Recommendations for Future Work

### 1. Multi-Parameter Optimization (Future Enhancement)
Instead of single sigma/threshold, try grid search and select longest path:
```python
best_centerline = None
best_length = 0

for sigma in [20, 25, 30]:
    for threshold in [0.5, 0.6, 0.7]:
        centerline = extract_centerline(mask, sigma=sigma, threshold=threshold)
        if centerline.length > best_length:
            best_length = centerline.length
            best_centerline = centerline
```
**Cost:** ~5-10x slower but guarantees longest valid path
**Benefit:** More accurate length measurements, fewer fin artifacts

### 2. Computational Optimizations
Several opportunities exist to speed up the geodesic method:

**a) Caching Connected Components**
- Current: Recomputes connected components every run
- Optimization: Cache after first computation if mask unchanged
- Speedup: ~5-10%

**b) Early Termination in Path Finding**
- Current: Always computes full Dijkstra
- Optimization: Stop when path found (if endpoints clearly separated)
- Speedup: 10-20% for well-connected skeletons

**c) Skeleton Simplification**
- Current: Uses all skeleton points for graph building
- Optimization: Thin skeleton to single-pixel width before graph building
- Speedup: 15-30% depending on skeleton thickness

**d) Parallel Multi-Parameter Evaluation**
- Current: Sequential sigma/threshold trials
- Optimization: Process multiple combinations in parallel
- Speedup: ~4-8x with 4-8 cores

**e) GPU Acceleration**
- Current: CPU-based Dijkstra via scipy.sparse.csgraph
- Optimization: GPU-accelerated Dijkstra for large skeletons
- Speedup: 5-50x for skeletons with >5000 points (rare for embryos)

### 3. Robustness Improvements
- Adaptive sigma based on embryo size/morphology
- Automatic threshold detection based on mask statistics
- Multi-scale skeleton analysis for complex topologies
