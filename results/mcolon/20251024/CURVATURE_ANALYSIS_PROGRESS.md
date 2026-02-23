# Embryo Curvature Analysis - Progress & Next Steps

**Date**: 2025-10-24
**Author**: mcolon
**Goal**: Develop robust curvature analysis pipeline for embryo morphology tracking

---

## What We've Achieved

### 1. Centerline Extraction Methods Tested

#### ✅ **Geodesic Skeleton Method (WINNER)**
- **Location**: `results/mcolon/20251022/geodesic_curvature_analysis.py`
- **Approach**:
  - Skeletonize binary mask
  - Build graph of 8-connected skeleton pixels
  - Use Dijkstra's algorithm to find maximum geodesic distance endpoints
  - Trace path between endpoints
- **Advantages**:
  - Handles highly curved embryos (head near tail)
  - Finds true endpoints via geodesic distance
  - Robust to complex shapes

#### ✅ **PCA Slicing Method**
- **Location**: `results/mcolon/20251022/extract_embryo_D06_t0022_pca_smoothed.py`
- **Approach**:
  - Find principal axis via PCA
  - Create perpendicular slices along axis
  - Compute centroids of each slice
- **Issue**: Less robust for highly curved embryos where head/tail are close

### 2. Smoothing Methods Compared

#### ✅ **B-spline Smoothing (RECOMMENDED)**
- **Location**: `results/mcolon/20251024/geodesic_bspline_smoothing.py`
- **Approach**:
  - Extract RAW centerline from geodesic skeleton
  - Fit cubic B-spline with smoothing parameter `s`
  - Compute curvature from analytical derivatives
- **Parameter**: `s = 5.0` (provides good balance)
- **Advantages**:
  - Global optimization for smooth curve
  - Analytical derivatives (more accurate)
  - Very responsive to smoothing parameter
  - Comparable to PCA method

#### ✅ **Gaussian Smoothing (TESTED - LESS EFFECTIVE)**
- **Issue**: Skeleton is already smooth, so Gaussian filtering has minimal effect
- **Why it failed**:
  - Only local averaging of coordinates
  - Can't create dramatic smoothing of already-smooth skeleton
  - Cumulative arc length acts as additional low-pass filter

#### ✅ **Other Methods Tested**
- **Location**: `results/mcolon/20251022/compare_smoothing_methods.py`
- Savitzky-Golay filter
- Moving average
- LOWESS (if statsmodels available)

### 3. Current Pipeline

```
Binary Mask
    ↓
Skeletonize
    ↓
Build Graph (8-connected)
    ↓
Find Endpoints (max geodesic distance)
    ↓
Trace Geodesic Path
    ↓
Fit B-spline (s=5.0)
    ↓
Compute Curvature (analytical derivatives)
    ↓
Curvature Profile
```

---

## What We Need to Do Next

### Priority 1: Curvature Metrics

Define and implement key curvature measurements:

#### 1.1 **Maximum Curvature**
```python
max_curvature = np.max(curvature)
max_curvature_location = arc_length[np.argmax(curvature)]
```
- **Biological meaning**: Where embryo bends most strongly
- **Use case**: Identify most curved region (often mid-trunk or tail)

#### 1.2 **Mean Curvature**
```python
mean_curvature = np.mean(curvature)
```
- **Biological meaning**: Overall body-axis deviation from straight line
- **Use case**: Global shape descriptor (straighter vs more curved embryos)

#### 1.3 **Curvature Profile (Normalized)**
```python
# Normalize arc length to [0, 1]
normalized_arc_length = arc_length / arc_length[-1]
# 0 = head, 1 = tail (after head/tail identification)
```
- **Biological meaning**: Regional curvature variation along A-P axis
- **Use case**: Compare curvature patterns across embryos/stages

#### 1.4 **Additional Metrics to Consider**
- **Total curvature** (integral of curvature): `np.trapz(curvature, arc_length)`
- **Curvature variance**: `np.var(curvature)`
- **Peak curvature count**: Number of local maxima
- **Curvature asymmetry**: Compare anterior vs posterior halves

### Priority 2: Head vs Tail Identification

**Problem**: Currently, endpoints are arbitrary (whichever geodesic finds first)

**Proposed Solutions**:

#### Option 2.1: **Local Mask Area (RECOMMENDED)**
```python
def identify_head_tail_by_area(mask, endpoint1, endpoint2, radius=50):
    """
    Head typically has larger cross-sectional area (rounder).
    Tail is thinner/more elongated.
    """
    # Extract local patches around each endpoint
    # Measure area within radius
    # Endpoint with larger area = head
    pass
```

#### Option 2.2: **Local Shape Circularity**
```python
def identify_head_tail_by_circularity(mask, endpoint1, endpoint2, radius=50):
    """
    Head is more circular/spherical.
    Tail is more elongated.
    """
    # Extract local patches
    # Compute circularity = 4π*area / perimeter²
    # Higher circularity = head
    pass
```

#### Option 2.3: **Curvature-based**
```python
def identify_head_tail_by_curvature(centerline, endpoint1, endpoint2):
    """
    Head often has smoother approach.
    Tail may have higher local curvature.
    """
    # Compare curvature near endpoints
    # Lower curvature = head
    pass
```

#### Option 2.4: **Width Profile**
```python
def identify_head_tail_by_width(mask, centerline):
    """
    Measure width perpendicular to centerline at regular intervals.
    Head end typically wider on average.
    """
    # Compute width profile along centerline
    # End with higher mean width = head
    pass
```

**Implementation Priority**: Try Option 2.1 (local area) first - simplest and most robust.

### Priority 3: Normalized Curvature Profile

Once head/tail is identified:

```python
def compute_normalized_curvature_profile(arc_length, curvature, head_idx):
    """
    Create curvature profile with consistent orientation.

    Args:
        arc_length: Arc length array
        curvature: Curvature array
        head_idx: Index of head endpoint (0 or -1)

    Returns:
        normalized_s: Arc length normalized to [0, 1], 0=head
        curvature_oriented: Curvature from head to tail
    """
    if head_idx != 0:
        # Reverse arrays so head is at index 0
        arc_length = arc_length[::-1]
        curvature = curvature[::-1]

    # Normalize arc length
    normalized_s = arc_length / arc_length[-1]

    return normalized_s, curvature
```

### Priority 4: Validation & Testing

**Test Cases** (already loaded in demo scripts):
1. `20251017_part2_D06_e01_t0022` - Good baseline
2. `20250512_E06_e01_t0086` - Extreme curvature
3. `20250512_E06_e01_t0181` - Self-overlap (challenging)

**Validation Steps**:
1. Visual inspection: Does head/tail identification make biological sense?
2. Consistency: Same embryo at different smoothing levels should have similar head/tail
3. Curvature profiles: Do they capture expected biological variation?

### Priority 5: Integration into Pipeline

**Target Script**: `src/build/build03A_process_images.py`

Add curvature analysis as optional feature:
```python
if compute_curvature:
    from curvature_analysis import GeodesicBSplineAnalyzer

    analyzer = GeodesicBSplineAnalyzer(
        mask,
        um_per_pixel=metadata['um_per_pixel'],
        bspline_smoothing=5.0
    )
    results = analyzer.analyze()

    # Store curvature metrics
    embryo_data['curvature_metrics'] = {
        'max_curvature': results['stats']['max_curvature'],
        'mean_curvature': results['stats']['mean_curvature'],
        'total_curvature': np.trapz(results['curvature'], results['arc_length']),
        'curvature_profile': results['curvature'].tolist(),
        'normalized_arc_length': (results['arc_length'] / results['arc_length'][-1]).tolist()
    }
```

---

## Open Questions

1. **What smoothing parameter for B-spline?**
   - Current: `s = 5.0`
   - Should this vary by embryo size?
   - Formula: `s = 5.0 * (total_length / 500)` to scale with embryo size?

2. **How to handle failed cases?**
   - Disconnected skeletons
   - Multiple branches
   - Very small embryos

3. **Physical units?**
   - Currently using pixels
   - Need to convert to microns for comparisons
   - Already have `um_per_pixel` in metadata

4. **Temporal analysis?**
   - Track curvature changes over time
   - Need trajectory linking first

5. **Regional curvature?**
   - Divide embryo into regions (anterior, trunk, posterior)
   - Compute curvature per region

---

## File Organization

```
results/mcolon/
├── 20251022/
│   ├── geodesic_curvature_analysis.py          # Geodesic + Gaussian smoothing
│   ├── extract_embryo_D06_t0022_pca_smoothed.py # PCA method
│   ├── compare_smoothing_methods.py             # Smoothing comparison
│   └── pca_smoothing_comparison.png             # Visual results
│
└── 20251024/
    ├── geodesic_bspline_smoothing.py            # CURRENT: Geodesic + B-spline
    ├── CURVATURE_ANALYSIS_PROGRESS.md           # This file
    └── [outputs will be generated here]
```

---

## Next Immediate Actions

1. **Implement head/tail identification** (Option 2.1: local area method)
2. **Add curvature metrics** to analyzer class
3. **Test on all 3 test embryos** and validate head/tail assignments
4. **Create normalized curvature profile visualization**
5. **Compare curvature profiles** across test cases

---

## Notes

- B-spline smoothing with `s=5.0` is now the recommended approach
- Geodesic skeleton method handles curved embryos well
- Need to standardize head/tail orientation for comparisons
- Curvature profiles will enable temporal and cross-embryo analysis
