# Geodesic Pathfinding Optimization Plan

## Executive Summary

Current bottleneck: **Endpoint detection = 81% of runtime** (~11.8s out of 14.6s per embryo)

**Phase 1 Target**: 14.6s → 6-8s per embryo (**45-55% speedup**, 1-2 hours implementation)

---

## Part 1: All 10 Optimization Methods - Complete Ranking

### Tier 1: Quick Wins (Easy + High Impact)

| # | Method | Feasibility | Impact | Total Speedup | Risk | Status |
|---|--------|-------------|--------|---------------|------|--------|
| 1 | Vectorized Graph Construction | Easy | High | 2-3x graph building (~6% total) | Low | **PHASE 1** |
| 2 | Convolution Endpoint Detection | Medium | High | 40-80% endpoint detection (~32-65% total) | Low-Medium | **PHASE 1** |
| 3 | Skeleton Thinning | Easy | Medium | 15-30% total | Low | **PHASE 1** |
| 4 | CSR Matrix Slicing | Easy | Low | 2-5% total | Low | Phase 2 |
| 5 | Spur Trimming | Medium | Medium | 10-20% total | Medium | Phase 2 |
| 6 | KD-Tree Graph Building | Medium | Medium | 20-40% graph building | Medium | Phase 2 |
| 7 | Parallel Dijkstra | Medium | Very High | 3-7x (multi-core) | High | Phase 3 |
| 8 | Adaptive Parameter Search | Medium | High (accuracy) | 4-8x compute | Medium | Phase 3 |
| 9 | Downsample Initial Pass | Medium | High (large only) | 2-5x for rare cases | High | Phase 3 |
| 10 | Early Exit Validation | Easy | Low | Variable | Low | Phase 3 |

---

## Part 2: Phase 1 Implementation Details

### Optimization #1: Vectorized Graph Construction

**File**: `segmentation_sandbox/scripts/body_axis_analysis/geodesic_method.py`
**Lines**: 56-103 (replace entire `_build_graph_fast` method)

**Current approach**: Dictionary lookup for neighbor finding
```python
point_to_index = {tuple(pt): idx for idx, pt in enumerate(skel_points)}
# Then for each point, look up neighbors in dictionary
jdx = point_to_index.get(neighbour)
```

**Optimized approach**: Grid-based indexing with array slicing
- Create dense index grid mapping (x,y) → node index
- Use NumPy slicing to find neighbor pairs
- Build sparse matrix directly from results

**Expected improvement**: 2-3x faster graph building (0.5s → 0.2s)

**Implementation**: See geodesic_method_optimized.py

---

### Optimization #2: Convolution-Based Endpoint Detection

**File**: `segmentation_sandbox/scripts/body_axis_analysis/geodesic_method.py`
**Lines**: Insert before line 212 (before "Step 3: Extract centerline")

**Current approach**: Exhaustive sampling from 100 points, runs Dijkstra 100 times

**Optimized approach**:
1. Use 3×3 convolution to identify true endpoints (degree-1 pixels)
2. Pre-filter candidates: only run Dijkstra from actual topological endpoints
3. Typically reduces from 100 → 2-10 Dijkstra calls

**Expected improvement**: 40-80% faster endpoint detection (11.8s → 2-3s)

**Algorithm**:
- Convolve skeleton with ones kernel: endpoint has exactly 2 (self + 1 neighbor)
- Map back to skeleton point indices
- Run Dijkstra only from candidate endpoints

**Implementation**: See geodesic_method_optimized.py

---

### Optimization #3: Skeleton Thinning

**File**: `segmentation_sandbox/scripts/body_axis_analysis/geodesic_method.py`
**Lines**: ~166 (after skeletonization)

**Current approach**: Direct skeletonization result used as-is

**Optimized approach**: Apply additional thinning pass
```python
skeleton = morphology.skeletonize(self.mask)
skeleton = morphology.thin(skeleton)  # Add this one line
```

**Expected improvement**: 15-30% reduction in skeleton points, cascading speedup

**Why it works**:
- Skeletonization can leave redundant pixels
- Additional thin() pass ensures minimal thickness
- Fewer points = faster all downstream operations

---

## Part 3: Implementation Workflow

### Step 1: Create optimized version
```bash
cp geodesic_method.py geodesic_method_optimized.py
```
- Rename class: `GeodesicCenterlineExtractor` → `GeodesicCenterlineExtractorOptimized`
- Original remains unchanged (safety net)

### Step 2: Implement optimizations in optimized version
1. Add skeleton thinning (trivial)
2. Replace `_build_graph_fast()` (20 lines)
3. Add convolution endpoint detection (30 lines)

### Step 3: Test on sample embryos
- Verify centerline length ±2%
- Verify curvature profile ±5%
- Benchmark timing improvements

### Step 4: Git commits
```bash
git add geodesic_method_optimized.py
git commit -m "Add Phase 1 optimization: skeleton thinning + vectorized graph construction"

git add geodesic_method_optimized.py
git commit -m "Add Phase 1 optimization: convolution-based endpoint detection"
```

---

## Part 4: Phase 2 Optimizations (Optional, 20-30% additional speedup)

### Optimization #4: CSR Matrix Slicing
- Replace manual adjacency matrix rebuilding (lines 195-210)
- Use SciPy's optimized CSR slicing: `adj_matrix[indices][:, indices]`
- Expected: 0.1s → 0.02s component filtering

### Optimization #5: Spur Trimming
- Iteratively remove short branches before endpoint detection
- Simplifies graph topology
- Expected: 10-20% speedup

### Optimization #6: KD-Tree Graph Building
- Alternative to vectorized approach
- Better for non-uniform skeleton distributions
- Expected: Similar performance to vectorized, different characteristics

---

## Part 5: Phase 3 Optimizations (Advanced)

### Optimization #7: Parallel Dijkstra
- Multi-core endpoint candidate evaluation
- Expected: 3-7x with 4-8 cores
- **Only worthwhile if processing >1000 embryos in batch**

### Optimization #8: Adaptive Parameter Selection
- Try 4-8 sigma/threshold combinations, pick longest
- Expected: Much better accuracy, 4-8x slower per embryo
- **Justification**: Documentation shows optimal parameters vary by embryo

### Optimization #9: Downsample Initial Pass
- For very large embryos (>2000×2000)
- Expected: 2-5x for rare large cases

### Optimization #10: Early Exit Validation
- Quick quality checks before expensive operations
- Skip processing invalid masks

---

## Part 6: Testing Protocol

### Baseline Metrics (before optimization)
Run on 10-20 representative embryos:
- Mean centerline length
- Standard deviation of length
- Mean curvature profile
- Processing time per embryo

### Validation After Each Optimization
1. **Correctness checks**:
   - Centerline length matches within ±2%
   - Curvature profile matches within ±5%
   - No new failures on previously successful cases

2. **Edge case testing**:
   - Highly curved embryos
   - Embryos with fins
   - Small embryos (<1000 skeleton points)
   - Large embryos (>5000 skeleton points)
   - Fragmented masks

3. **Performance benchmarking**:
   - Measure each optimization independently
   - Profile with cProfile to verify bottleneck reduction
   - Memory usage check

---

## Part 7: Risk Assessment

### Low Risk (Safe to implement immediately)
- ✅ Skeleton thinning
- ✅ Path length vectorization
- ✅ Sparse matrix cleanup
- ✅ Early exit validation

### Medium Risk (Test thoroughly before deploying)
- ⚠️ Convolution endpoint detection (may miss endpoints in unusual topologies)
- ⚠️ Spur trimming (may remove valid short branches)
- ⚠️ KD-Tree approach (edge cases with non-uniform distributions)

### High Risk (Requires extensive validation)
- 🔴 Parallel Dijkstra (race conditions, overhead)
- 🔴 Downsample initial pass (may miss fine details)
- 🔴 GPU acceleration (platform dependency)

---

## Part 8: Current Performance Profile

### Timing Breakdown (per embryo, ~14.6s total)

```
Mask cleaning:           0.1s   (0.7%)
Gaussian blur:          0.05s   (0.3%)
Skeletonization:         0.2s   (1.4%)
Component filtering:     0.1s   (0.7%)
Graph building:          0.5s   (3.4%)
Endpoint detection:     11.8s   (81%)    ← PRIMARY BOTTLENECK
Dijkstra path trace:     0.8s   (5.5%)
B-spline fitting:        0.4s   (2.7%)
```

### After Phase 1 Optimizations (Expected)

```
Mask cleaning:          0.1s   (1-2%)
Gaussian blur:         0.05s   (1%)
Skeletonization:        0.1s   (1-2%)    [Reduced by skeleton thinning]
Component filtering:    0.1s   (1-2%)
Graph building:         0.2s   (3-5%)    [2-3x faster: vectorization]
Endpoint detection:     2-3s   (30-50%)  [3-4x faster: convolution]
Dijkstra path trace:    0.5s   (5-10%)   [Slightly faster: fewer points]
B-spline fitting:       0.3s   (3-5%)    [Slightly faster: fewer points]
TOTAL:                 6-8s    (45-55% speedup)
```

---

## Part 9: Files and Implementation References

### File Structure
```
segmentation_sandbox/scripts/body_axis_analysis/
  geodesic_method.py              (original - DO NOT MODIFY)
  geodesic_method_optimized.py    (optimization work)

results/mcolon/20251028/
  geodesic_pathfinding_optimization/
    OPTIMIZATION_PLAN.md          (this file)
    potential_improvements.txt    (original documentation)
    test_comparison.py            (validation script)
```

### Key Code Locations (Original)
- Line 56-103: `_build_graph_fast()` - target for vectorization
- Line 166: Skeletonization - add thinning here
- Line 182-210: Component filtering - can optimize with CSR slicing
- Line 212-245: Endpoint detection - target for convolution pre-filtering

---

## Part 10: Success Criteria

### Phase 1 Success = All three are true:
1. ✅ Processing time: 14.6s → 6-8s per embryo (45-55% speedup achieved)
2. ✅ Correctness: Centerline length ±2%, curvature profile ±5%
3. ✅ Safety: All previous test cases still pass

### Phase 1 Completion:
- [ ] geodesic_method_optimized.py created with all 3 optimizations
- [ ] Test script validates optimized vs original on 10+ embryos
- [ ] Git commits pushed with incremental progress
- [ ] Documentation updated with results

---

## Implementation Timeline

| Phase | Methods | Time | Expected Speedup | Status |
|-------|---------|------|------------------|--------|
| Phase 1 | Skeleton thinning, vectorized graph, convolution endpoints | 1-2 hrs | 45-55% | 🚀 ACTIVE |
| Phase 2 | CSR slicing, spur trimming, KD-tree | 4-6 hrs | +20-30% | 📋 Ready |
| Phase 3 | Parallel, adaptive search, downsample, validation | 2+ days | +15-30% | 📋 Ready |

---

**Document created**: 2025-10-28
**Analysis based on**:
- `potential_improvements.txt` analysis
- `geodesic_method.py` code review
- Current performance profiling
