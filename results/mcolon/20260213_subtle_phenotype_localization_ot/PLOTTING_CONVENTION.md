# Plotting Convention: WT Reference Grid as Canonical Coordinate System

**Date**: 2026-02-13
**Status**: CRITICAL - Read this before implementing any visualization code

---

## Core Principle

**ALL visualizations are plotted on the WT reference grid**, regardless of OT direction.

This establishes a **standardized spatial reference frame** that remains constant across all mutant comparisons, despite variable mutant morphology.

---

## OT Direction vs. Visualization Grid

### OT Computation Direction

**Source → Target**: **WT → mutant**

- **Source (μ)**: WT reference mask
- **Target (ν)**: Mutant mask
- **Transport plan P**: Maps WT mass to mutant mass
- **Interpretation**: "Where does the WT have to move/deform to become the mutant?"

### Visualization Grid

**All features plotted on WT reference grid** (template space):

- Cost density c(x): Where x ∈ WT grid
- Displacement field d(x): Where x ∈ WT grid
- Mass delta Δm(x): Where x ∈ WT grid
- S-coordinate bins: Defined on WT centerline

---

## Why This Convention?

### Problem: Variable Mutant Morphology

Mutants have:
- Different shapes (curvature, length, aspect ratio)
- Variable mask coverage on canonical grid
- Inconsistent spatial topology across individuals

**If we plotted on mutant grids**, we would:
- ❌ Lose spatial consistency across embryos
- ❌ Complicate averaging/aggregation
- ❌ Break S-coordinate correspondence
- ❌ Make AUROC-by-region analysis incoherent

### Solution: WT Reference as Template

By plotting on WT grid:
- ✅ **Standardized anatomy**: All features in same spatial reference
- ✅ **Stable S-coordinates**: Head-to-tail axis consistent across comparisons
- ✅ **Aggregation-friendly**: Can average c(x), d(x) across embryos
- ✅ **AUROC interpretability**: "Region at S=0.3" means same anatomy for all embryos

---

## Implementation Details

### UOT Setup (WT → mutant)

```python
from src.analyze.optimal_transport_morphometrics.uot_masks import run_uot_pair

# Load masks
wt_mask = load_mask_from_csv(csv_path, wt_embryo_id, wt_frame_idx)
mut_mask = load_mask_from_csv(csv_path, mut_embryo_id, mut_frame_idx)

# Create pair: WT is SOURCE, mutant is TARGET
pair = UOTFramePair(
    src=UOTFrame(embryo_mask=wt_mask),  # WT reference
    tgt=UOTFrame(embryo_mask=mut_mask),  # Mutant
)

# Run OT (WT → mutant)
result = run_uot_pair(pair, config)
```

### Feature Extraction (Always WT Grid)

```python
from utils.ot_features import extract_ot_features

# Extract features (all on WT grid by default)
features = extract_ot_features(result, mask_ref=wt_mask)

# Features are already on WT reference grid:
# - features.cost_density: (H, W) on WT grid
# - features.displacement_u/v: (H, W) on WT grid
# - features.mass_delta: (H, W) on WT grid
```

**IMPORTANT**:
- `result.cost_src_px` = cost on **source** (WT) grid ✅ Use this
- `result.cost_tgt_px` = cost on **target** (mutant) grid ❌ Don't use for visualization

### S-Coordinate Assignment (WT Centerline)

```python
from segmentation_sandbox.scripts.body_axis_analysis.centerline_extraction import extract_centerline

# Fit centerline on WT reference mask
centerline_yx = extract_centerline(wt_mask)

# Parameterize S ∈ [0, 1] (head → tail)
# S=0: rostral (head)
# S=1: caudal (tail)

# Assign S to every pixel in WT grid
s_coords = assign_s_coordinates(wt_mask, centerline_yx)

# Bin features by S (on WT grid)
s_bins = bin_features_by_s(features, s_coords, K=10)
```

### Visualization (WT Grid Overlay)

```python
from utils.canonical_grid_viz import plot_feature_on_canonical_grid

# Plot cost density on WT reference grid
plot_feature_on_canonical_grid(
    feature=features.cost_density,
    mask=wt_mask,
    title="Cost Density (WT→mutant, plotted on WT grid)",
    output_path="cost_density_wt_grid.png",
)

# Plot displacement vectors on WT grid
plot_feature_on_canonical_grid(
    feature=features.displacement_mag,
    vectors=(features.displacement_u, features.displacement_v),
    mask=wt_mask,
    title="Displacement Field (WT→mutant, plotted on WT grid)",
    output_path="displacement_wt_grid.png",
)
```

---

## Aggregation Across Multiple Mutants

When averaging features across multiple mutant embryos:

```python
# Run OT for each mutant → WT
cost_densities = []
for mut_embryo in mutant_list:
    # WT → mutant OT
    pair = UOTFramePair(
        src=UOTFrame(embryo_mask=wt_ref_mask),
        tgt=UOTFrame(embryo_mask=mut_embryo.mask),
    )
    result = run_uot_pair(pair, config)
    features = extract_ot_features(result, mask_ref=wt_ref_mask)
    cost_densities.append(features.cost_density)

# Aggregate on WT grid (all arrays are same shape!)
mean_cost = np.nanmean(cost_densities, axis=0)  # Still on WT grid
std_cost = np.nanstd(cost_densities, axis=0)

# Smooth AFTER aggregation (not before)
from utils.ot_features import smooth_feature_for_viz
mean_cost_smooth = smooth_feature_for_viz(
    mean_cost, wt_ref_mask, sigma_um=20.0, um_per_pixel=10.0
)
```

**CRITICAL**: All `cost_densities` are on the same WT grid → can aggregate directly.

---

## S-Bin Profiles (Along WT Centerline)

All S-bin profiles are computed along the **WT centerline**:

```python
# For each S bin k ∈ {1, ..., K}:
# 1. Define region R_k on WT grid
# 2. Extract features inside R_k
# 3. Aggregate (mean, median, etc.)

s_profile = []
for k in range(K):
    mask_k = (s_coords >= k/K) & (s_coords < (k+1)/K) & wt_ref_mask
    c_k = np.nanmean(features.cost_density[mask_k])
    s_profile.append(c_k)

# Plot cost along S (WT anatomy)
plt.plot(s_profile, label="Cost along S (WT centerline)")
plt.xlabel("S (0=head, 1=tail, WT reference anatomy)")
plt.ylabel("Mean cost density (μm²)")
```

**Interpretation**: S=0.3 on WT anatomy = same anatomical region for all mutants.

---

## AUROC-by-Region (WT Regions)

AUROC analysis partitions the **WT reference grid** into regions:

```python
# For each S bin k (defined on WT grid):
# Extract feature for all WT vs mutant embryos
wt_features_k = [extract_in_region_k(wt_i) for wt_i in wt_list]
mut_features_k = [extract_in_region_k(mut_i) for mut_i in mut_list]

# Compute AUROC for region k
auroc_k = compute_auroc(wt_features_k, mut_features_k)

# Plot AUROC along S (WT anatomy)
plt.plot(auroc_values, label="AUROC by S-bin (WT regions)")
plt.xlabel("S bin (WT anatomy)")
plt.ylabel("AUROC (discriminability)")
```

**Interpretation**: "Region S ∈ [0.2, 0.3] on WT anatomy shows high discriminability."

---

## Summary Table

| Aspect | Convention | Rationale |
|--------|-----------|-----------|
| **OT direction** | WT → mutant | "How does WT deform to match mutant?" |
| **Visualization grid** | WT reference grid | Standardized spatial reference |
| **Cost density c(x)** | x ∈ WT grid | Use `result.cost_src_px` |
| **Displacement d(x)** | x ∈ WT grid | Use `result.velocity_px_per_frame_yx` |
| **S-coordinates** | WT centerline | S=0 (head) to S=1 (tail) on WT |
| **Aggregation** | Average on WT grid | All arrays same shape |
| **AUROC regions** | WT grid regions | Consistent anatomy across embryos |

---

## Checklist for Every Visualization

Before plotting ANY feature, verify:

- [ ] OT ran as WT → mutant (source=WT, target=mutant)
- [ ] Features extracted on WT grid (use `cost_src_px`, not `cost_tgt_px`)
- [ ] Mask overlay shows WT reference anatomy
- [ ] S-coordinates defined on WT centerline
- [ ] Aggregation performed on WT grid (same shape across embryos)
- [ ] Plot title/labels clarify "plotted on WT reference grid"

---

## Code Review Red Flags

**❌ WRONG:**
```python
# Don't do this!
result = run_uot_pair(mutant_src, wt_tgt)  # Backwards!
cost = result.cost_tgt_px  # Wrong grid!
```

**✅ CORRECT:**
```python
# Do this!
result = run_uot_pair(wt_src, mutant_tgt)  # WT → mutant
cost = result.cost_src_px  # WT grid
```

---

**TLDR**: WT reference = spatial coordinate system. All features, all plots, all aggregations happen on WT grid. Never plot on mutant grid.
