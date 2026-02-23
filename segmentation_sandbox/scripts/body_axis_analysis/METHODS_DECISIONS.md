# Body Axis Analysis: Methods & Decisions

Quick reference for methods chosen during spline analysis optimization and the reasoning behind each decision.

---

## Method Selection: Geodesic vs PCA

### Decision: Use Geodesic Method as Primary (Default)

**Why Geodesic is Default?**
- Handles highly curved embryos where head is near tail (PCA fails catastrophically)
- Topology-aware: uses actual skeleton instead of linear projections
- More robust for complex body shapes (non-convex, irregular)
- Only ~2.8x slower than PCA (14.6s vs 5.3s per embryo)
- Safe choice when embryo morphology is unknown

**When to use PCA?**
- Speed is critical AND embryos are known to be normally shaped
- Extent > 0.35 AND Solidity > 0.6 AND Eccentricity < 0.98
- For bulk processing where preprocessing validates normal shape
- Trade-off: faster but may fail on edge cases

**Evidence:**
- 1000-embryo comparison analysis (Oct 27, 2024)
- 97.5% agreement between methods (Hausdorff < 114.78px)
- 2.5% disagreement caused by extreme curvature in PCA cases
- PCA method fails when principal axis assumption breaks down

### No Auto-Selection (Removed)

Auto-selection was removed to keep API simple and explicit. Instead:
- **Default to Geodesic** (robust, safe)
- **Explicitly choose PCA** if needed for speed
- **Use `compare_methods()`** to validate on specific data

**Future Enhancement (Stub):**
If performance becomes critical, auto-selection logic can be re-implemented based on mask morphology metrics, but currently explicit method choice is preferred.

---

## Mask Preprocessing: Gaussian Blur (Default)

### Decision: Apply Gaussian Blur Before Spline Fitting

**Why Gaussian Blur?**
- **Cheap**: <0.1s per mask
- **Fast**: Applied to all masks regardless of method
- **Effective**: Removes small protrusions and smooths boundaries
- **Non-destructive**: Preserves overall embryo shape

**Parameters:**
- sigma = 10.0 (default blur amount)
- threshold = 0.7 (re-threshold value)

**What it does:**
1. Blur mask with Gaussian filter (σ=10)
2. Re-threshold at high value (0.7)
3. Only pixels with strong "mask presence" in neighborhood survive
4. Result: Smoother boundaries, fewer spiny protrusions

**Experimental Alternative: Alpha Shape**
- More geometric approach
- Uses concave hull via Delaunay triangulation
- Slower (~1s vs <0.1s) but more principled
- **Currently not implemented** (reserved for future testing with alternative geodesic methods)
- Will be implemented if different centerline extraction methods are tested

**Decision Rationale:**
- Speed-critical (< 0.1s vs ~1s for alpha shape)
- Works well in practice (tested on 500+ embryos)
- Standard in spline fitting pipelines
- Keep alpha shape as future extension when new methods are explored

---

## B-Spline Smoothing: s=5.0

### Decision: Fixed Smoothing Parameter s=5.0

**Why 5.0?**
- Tested across smoothing levels: 0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0
- 5.0 provides optimal balance:
  - Smooth enough to reduce skeleton noise
  - Rough enough to preserve biological curvature features
  - Consistent across different embryo shapes

**What s=5.0 means:**
- Internally: `s = 5.0 * len(centerline)` for scipy's splprep
- Adaptive: Longer centerlines get more smoothing (good!)
- Not a free parameter: Use geodesic/PCA directly if different s needed

**Curvature Computation:**
- κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
- Computed from analytical B-spline derivatives
- More accurate than numerical differentiation

---

## Skeleton Pruning: Removed (Didn't Work)

### Decision: DO NOT Use Skeleton Pruning

**Why removed?**
- Tested adaptive fin removal on 200+ embryos
- Parameter tuning was fragile (no universal thresholds)
- Unpredictable behavior on edge cases
- Geodesic method already robust without pruning

**What was tried:**
- Length-based thresholds (10-50% of embryo length)
- Width-based thresholds (median skeleton width percentiles)
- Angle-based removal (sharp angles from main trunk)

**What actually works:**
- Good mask cleaning (solid > 0.6 threshold)
- Mask preprocessing (Gaussian blur)
- Geodesic method is naturally robust to skeleton spines

**When needed in future:**
- If skeleton clearly extends through fins: improve mask cleaning
- If spline follows skeleton noise: increase B-spline smoothing parameter

---

## Mask Cleaning: Solidity < 0.6 Threshold

### Decision: Apply Conditional Opening Only When Solidity < 0.6

**5-Step Pipeline (mask_cleaning.py):**
1. Remove small debris (<10% of mask area)
2. Iterative adaptive closing (max 5 iterations, radius capped)
3. Fill holes (binary fill)
4. **Conditional opening** (ONLY if solidity < 0.6) ← KEY DECISION
5. Keep largest component (safety)

**Solidity Definition:**
- solidity = (mask area) / (convex hull area)
- solidity=1.0: Perfect convex shape (circle, rectangle)
- solidity<0.6: Highly concave (concave, spiny, irregular)

**Why this threshold?**
- Based on 500-embryo morphology analysis
- ~40% of embryos have solidity < 0.6
- Opening removes spiny protrusions effectively
- But expensive operation: skip when not needed

**Opening Parameters:**
- Radius = max(embryo_length / 150, 1) pixel
- Chosen to be gentle (was /100, changed to /150)
- Preserves thin tails while removing spindly structures

---

## Head/Tail Orientation: Width Tapering Method

### Decision: Use Width Tapering to Identify Head

**Method:**
1. Sample 20 points along centerline
2. Measure local width at each point
3. Fit linear regression to width vs position
4. Head = end where width is largest

**Why width tapering?**
- Biological: zebrafish head is wider than tail
- Universal: works for all body orientations
- Robust: doesn't depend on skeleton structure
- Fast: O(n) computation

**Result:**
- Spline oriented: head → tail
- Consistent direction for all analyses
- Required for temporal/trajectory analyses

---

## Summary: What Changed from Original Analysis

| Aspect | Original | Final | Reason |
|--------|----------|-------|--------|
| Centerline method | Both tested | Geodesic primary | Handles curved embryos |
| PCA method | Experimental | Fast fallback | Keep for speed when safe |
| Skeleton pruning | Experimental | Removed | Didn't work reliably |
| B-spline smoothing | Variable | Fixed s=5.0 | Optimal trade-off |
| Mask preprocessing | None | Gaussian blur | Fast, effective |
| Opening threshold | Always applied | solidity < 0.6 | Conditional saves time |

---

## Usage Examples

### Simple Usage (Default: Geodesic)
```python
from body_axis_analysis import extract_centerline

# Uses Geodesic method by default (robust)
spline_x, spline_y, curvature, arc_length = extract_centerline(mask)
```

### Explicit Method Selection
```python
# Use Geodesic (robust, handles curved embryos)
x, y, curv, arc = extract_centerline(mask, method='geodesic')

# Use PCA (faster, for known-good data)
x, y, curv, arc = extract_centerline(mask, method='pca')
```

### Tune B-Spline Smoothing
```python
# Custom smoothing (default s=5.0)
x, y, curv, arc = extract_centerline(mask, bspline_smoothing=3.0)

# More aggressive smoothing
x, y, curv, arc = extract_centerline(mask, bspline_smoothing=10.0)
```

### Get Full Results
```python
results = extract_centerline(mask, return_intermediate=True)
print(f"Length: {results['stats']['total_length']:.1f}")
print(f"Mean curvature: {results['stats']['mean_curvature']:.6f}")
print(f"Method used: {results['stats']['method']}")
```

### Compare Methods
```python
from body_axis_analysis import compare_methods

comparison = compare_methods(mask)
print(f"Hausdorff distance: {comparison['hausdorff_distance']:.1f}")
print(f"Aligned distance: {comparison['mean_aligned_distance']:.1f}")
# If aligned_distance < 20: methods agree
# If aligned_distance > 40: methods disagree (investigate mask morphology)
```

### Reproducible Results
```python
# Same seed ensures same endpoint detection in Geodesic method
x1, y1, _, _ = extract_centerline(mask, random_seed=42)
x2, y2, _, _ = extract_centerline(mask, random_seed=42)
# x1 ≈ x2 (identical results)
```

---

## Key Files

- `centerline_extraction.py`: Main API
- `geodesic_method.py`: Geodesic implementation (primary)
- `pca_method.py`: PCA implementation (fallback)
- `mask_preprocessing.py`: Gaussian blur + alpha shape
- `spline_utils.py`: Head/tail identification, spline alignment
- `../utils/mask_cleaning.py`: 5-step mask cleaning pipeline

---

## References

- 1000-embryo PCA vs Geodesic comparison: `results/mcolon/20251027/compare_pca_vs_geodesic.py`
- 200-embryo skeleton pruning analysis: `results/mcolon/20251027/test_pruned_geodesic.py`
- Mask morphology analysis: `results/mcolon/20251020/penetrance/` (heatmaps)
- B-spline smoothing parameter sweep: `results/mcolon/20251022/test_pca_smoothing.py`

---

## Questions?

If behavior differs from these decisions:
1. Check method used (geodesic vs PCA)
2. Verify mask preprocessing applied
3. Check morphology metrics (extent, solidity, eccentricity)
4. Use `compare_methods()` to see if methods disagree
5. Refer to comparison analysis scripts in results/ folder
