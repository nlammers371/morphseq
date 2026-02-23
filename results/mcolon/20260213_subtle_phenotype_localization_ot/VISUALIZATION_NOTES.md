# Visualization Notes: Canonical Grid Overlay Stack

**Date**: 2026-02-13

## Core Principle: Clean Layered Overlays

All visualizations use a canonical-grid-first approach where everything lives on the same H×W template grid:
- Cost density c(x,y)
- Displacement field d(x,y) = (u, v)
- Mass delta Δm(x,y)
- Spline coordinate S(x,y) ∈ [0,1]
- Reference mask

This makes layering trivial—each overlay is just another matplotlib layer.

---

## The "Clean Contours" Recipe (Not Pixel Confetti)

### Layer Order (bottom to top):
1. **Filled contours** (quantile-based zones)
2. **Thin contour lines** (crisp boundaries)
3. **Embryo outline** (white border)
4. **Subsampled vector field** (aggressive stride, e.g., every 12 pixels)
5. **S-bin isolines** (cyan dashed lines like "latitude lines")

### Key Steps:
1. **Mask to embryo region**: Set values outside `mask_ref` to NaN
2. **Boundary-safe Gaussian smoothing**:
   ```python
   f_smooth = gaussian(f * mask) / gaussian(mask)
   ```
   This prevents bleeding across the embryo boundary
3. **Quantile levels for comparisons**:
   - Use percentiles (50th, 70th, 85th, 92nd, 97th) instead of fixed thresholds
   - Ensures consistent "top X%" bands across WT vs mutant
4. **Subsample vectors aggressively**:
   - Never plot at full resolution (becomes visual noise)
   - Start with stride=12-20, adjust until just-barely-sparse
5. **S-bin isolines**:
   - Draw contours of S(x,y) at bin edges (e.g., [0.1, 0.2, ..., 0.9] for K=10)
   - These show rostral→caudal sections clearly

---

## **CRITICAL: Aggregation Strategy (Mean THEN Smooth)**

When computing mean fields across embryos (Section 1: WT mean vs mutant mean):

### ✅ CORRECT: Mean-then-smooth
```python
# Aggregate across embryos FIRST
mean_cost_wt = np.mean(cost_fields_wt, axis=0)

# THEN smooth for visualization
mean_cost_wt_smooth = smooth_inside_mask(mean_cost_wt, mask_ref, sigma=2.0)
```

### ❌ WRONG: Smooth-then-mean
```python
# DO NOT smooth individual embryos then average
# This over-blurs and loses spatial structure
smooth_costs_wt = [smooth_inside_mask(c, mask_ref, sigma=2.0) for c in cost_fields_wt]
mean_cost_wt = np.mean(smooth_costs_wt, axis=0)  # TOO SMOOTH
```

### Why?
- **Gaussian smoothing is for visualization only**, not for the data itself
- Smoothing individual embryos compounds the blur when averaged
- The raw mean preserves spatial detail; smoothing at the end provides just enough visual clarity

### Exception: Per-embryo visualizations
When plotting a single embryo's cost map, smooth that embryo's field directly (no aggregation involved).

---

## Difference Maps (Mutant - WT)

For mutant-minus-WT difference maps:
1. Compute mean fields: `mean_mutant` and `mean_wt`
2. Compute difference: `diff = mean_mutant - mean_wt`
3. Smooth the difference: `diff_smooth = smooth_inside_mask(diff, mask_ref, sigma=2.0)`
4. Use **symmetric levels around 0**:
   ```python
   max_abs = np.max(np.abs(diff_smooth[mask_ref]))
   levels = np.linspace(-max_abs, max_max, 7)  # Odd number for symmetry
   ```
5. Use diverging colormap: `RdBu_r` (red = higher in mutant, blue = higher in WT)

This ensures visual fairness—positive and negative deviations are equally visible.

---

## ⭐ CRITICAL: Real Physical Units (Micrometers)

**Key advantage**: Because OT analysis is on the canonical grid with **10 μm/pixel**, all metrics are in **real physical units**, not arbitrary pixel values.

### What this means:
- **Cost density c(x)**: Units of **μm²** (squared transport distance)
- **Displacement d(x)**: Units of **μm/frame** (physical distance moved)
- **Mass delta Δm(x)**: Units of **μm²** (area created/destroyed)
- **Contour levels**: Can be defined in **μm²** (e.g., "cost > 50 μm²")
- **Gaussian sigma**: Can be specified in **μm** (e.g., σ=20 μm)

### Why this matters:
1. **Interpretability**: Thresholds have physical meaning ("20 μm transport distance")
2. **Comparability**: Results are comparable across experiments/datasets
3. **Biological relevance**: Can relate to known anatomical scales (e.g., "cell diameter ~10 μm")
4. **Publication ready**: Units make sense to readers without conversion

### Practical impact:
```python
# Cost density in μm² (physical units)
cost_density = compute_cost(mutant_mask, reference_mask)  # returns μm² values

# Contour levels in μm²
levels = [10, 25, 50, 100, 200]  # μm² thresholds (interpretable!)

# Gaussian sigma in μm
sigma_um = 20.0  # 20 microns smoothing (anatomically meaningful)
sigma_pixels = sigma_um / um_per_pixel  # Convert to pixels for scipy
```

---

## Choosing Sigma (Gaussian Kernel Width)

**Goal**: Sigma should be tied to an anatomical scale, not "whatever looks nice per figure."

### Recommended approach:
- **Work in microns**: sigma=20.0-30.0 μm is typical (matches 2-3 pixels at 10 μm/px)
- **Convert to pixels for scipy**: `sigma_px = sigma_um / um_per_pixel`
- **Keep sigma FIXED across all conditions** (WT, mutant, timepoints) for fair comparisons

### Empirical tuning:
1. Start with sigma=20 μm (2 pixels at 10 μm/px)
2. If contours look too noisy: increase to 30-40 μm
3. If contours over-smooth (lose fine structure): decrease to 15-20 μm
4. Once chosen, **document in config.yaml** in μm units and use consistently

### Example (real units):
```yaml
# config.yaml
gaussian_kernel_sigma_2d: 20.0  # μm (not pixels!)
um_per_pixel: 10.0  # Canonical grid resolution

# In code:
sigma_px = config['gaussian_kernel_sigma_2d'] / config['um_per_pixel']
smooth_field = gaussian_filter(field, sigma=sigma_px)
```

---

## Contour Level Strategies

### A) Quantile levels (default for WT vs mutant comparisons)
```python
levels = np.quantile(field[mask_ref], [0.5, 0.7, 0.85, 0.92, 0.97])
```
- **Pros**: Consistent "top X%" bands across conditions
- **Use for**: Comparing WT mean vs mutant mean, time series

### B) Fixed absolute levels
```python
levels = np.linspace(0, max_cost, 7)
```
- **Pros**: Shows absolute scale changes
- **Use for**: When units are stable and meaningful (e.g., microns, cost in μm²)

### C) Symmetric levels (for difference maps)
```python
max_abs = np.max(np.abs(diff[mask_ref]))
levels = np.linspace(-max_abs, max_abs, 7)
```
- **Pros**: Visual fairness for positive/negative deviations
- **Use for**: Mutant-minus-WT, condition comparisons

---

## S-Bin Isolines ("Latitude Lines")

S-bin boundaries instantly communicate "which region along the body axis."

### Implementation:
```python
K = 10  # Number of S bins
s_bin_edges = np.linspace(0, 1, K+1)[1:-1]  # [0.1, 0.2, ..., 0.9]

ax.contour(S, levels=s_bin_edges, colors='cyan',
           linewidths=0.8, linestyles='dashed', alpha=0.6)
```

### Visual tips:
- Use **cyan dashed lines** so they don't compete with cost contours
- Keep linewidth thin (0.7-0.9)
- Set alpha=0.6 so they don't dominate

---

## Vector Field Display

Vectors get messy fast. The clean approach:

### Subsampling:
```python
stride = 12  # Every 12 pixels
Y = np.arange(H)[::stride]
X = np.arange(W)[::stride]
XX, YY = np.meshgrid(X, Y)

# Only plot inside mask
inside = mask_ref[YY, XX]

ax.quiver(XX[inside], YY[inside], u[YY, XX][inside], v[YY, XX][inside],
          angles='xy', scale_units='xy', scale=1.0,
          width=0.003, color='white', alpha=0.7)
```

### Optional: magnitude filtering
```python
mag = np.sqrt(u**2 + v**2)
valid = inside & (mag > min_threshold)
```
Only show vectors above a minimum magnitude to reduce clutter.

---

## Titles That Say Exactly What It Is

Good title structure:
```
"{Condition} mean {field_name}"
"(Gaussian σ={sigma} px; filled contours = quantile bands; vectors = displacement)"
```

Examples:
- "WT mean cost density (σ=2.0 px; quantiles [50, 70, 85, 92, 97]%)"
- "Mutant-WT difference (σ=2.0 px; symmetric levels; red=higher in mutant)"

---

## Implementation in Scripts

### Section 1 (OT mapping + visualization):
```python
from utils.canonical_grid_viz import plot_mean_field_comparison

# Aggregate cost fields (MEAN THEN SMOOTH)
cost_fields_wt = [...]  # List of (H, W) arrays
cost_fields_mutant = [...]

# 3-panel figure: WT mean, mutant mean, difference
fig = plot_mean_field_comparison(
    cost_fields_wt,
    cost_fields_mutant,
    mask_ref=reference_mask,
    sigma=2.0,
    field_name="cost density",
    s_bin_edges=np.linspace(0, 1, 11)[1:-1],  # K=10 bins
    S=s_coordinate_map,
    um_per_pixel=10.0,
)
fig.savefig("outputs/section_1/fig_1a_cost_comparison.png", dpi=300)
```

### Section 2 (along-S profiles):
- Compute mean profiles (already aggregated along S bins)
- Apply 1D Gaussian smoothing for visualization:
  ```python
  from scipy.ndimage import gaussian_filter1d
  profile_smooth = gaussian_filter1d(profile, sigma=1.0)
  ```

---

## Common Pitfalls to Avoid

1. **Smoothing individual embryos before averaging** → Over-blurs
2. **Using pixel indices when grid has physical units** → Sigma becomes meaningless
3. **Plotting vectors at full resolution** → Visual noise
4. **Using fixed levels for quantile comparisons** → Misleading when distributions shift
5. **Asymmetric levels for difference maps** → Visual bias toward one direction

---

## Summary Checklist for Every Figure

- [ ] Mask to embryo region (NaN outside)
- [ ] Aggregate THEN smooth (for mean fields)
- [ ] Use quantile levels (for comparisons) or symmetric levels (for differences)
- [ ] Subsample vectors (stride ≥ 12 pixels)
- [ ] Add S-bin isolines (cyan dashed)
- [ ] Add embryo outline (white border)
- [ ] Title says exactly what it is (σ, levels, conditions)
- [ ] Sigma consistent across all figures in analysis

---

## File: `utils/canonical_grid_viz.py`

See `utils/canonical_grid_viz.py` for full implementation with:
- `smooth_inside_mask()` - Boundary-safe Gaussian smoothing
- `quantile_levels()` - Compute contour levels from quantiles
- `symmetric_levels()` - Symmetric levels for difference maps
- `plot_canonical_overlay()` - Main plotting function
- `plot_difference_map()` - Difference map with symmetric levels
- `plot_mean_field_comparison()` - 3-panel WT/mutant/difference

---

**Status**: Ready to use in Section 1 (OT mapping) and Section 2 (S profiles).
