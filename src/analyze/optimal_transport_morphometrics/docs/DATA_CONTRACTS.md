# UOT Optimal Transport Data Contracts

**Document Purpose**: Define the precise semantics of data flowing through the Unbalanced Optimal Transport (UOT) pipeline. This prevents silent bugs where similar-looking data means different things.

**Last Updated**: January 24, 2026  
**Criticality**: ESSENTIAL - Violations of these contracts cause morphological misinterpretation

---

## Table of Contents

1. [Overview: Goals and Scope](#overview-goals-and-scope)
2. [Core Concept: src_support Sampling](#core-concept-src_support-sampling)
3. [Mass Metrics Contract](#mass-metrics-contract)
4. [Velocity Field Contract](#velocity-field-contract)
5. [Canonical Alignment Contract](#canonical-alignment-contract)
6. [Yolk Mask Contract](#yolk-mask-contract)
7. [Grid Hierarchy Contract](#grid-hierarchy-contract)
8. [Visualization Contract](#visualization-contract)
9. [Directory Overview](#directory-overview)
10. [Project Status / Next Steps](#project-status--next-steps)
11. [Data Flow Diagram](#data-flow-diagram)

---

## Overview: Goals and Scope

**Primary goal**: produce biologically meaningful OT maps by standardizing each embryo mask into a **canonical grid** with consistent orientation, scale, and anchoring **before** transport is computed. This ensures transport costs reflect **morphological dynamics**, not camera framing or rigid translation.

**In scope**:
- Canonical alignment contract (PCA, flip search, yolk anchoring)
- Sampling/metrics contracts for UOT outputs
- Visualization contract to avoid fabricated data

**Out of scope (next steps)**:
- Visualization extraction, dynamics clustering, GPU acceleration (see below)

---

## Core Concept: src_support Sampling

### The Fundamental Design

The UOT solver does NOT operate on the full canonical grid. Instead:

```
Canonical Grid (256 × 576 px)
         ↓ downsample
Work Grid (smaller, faster)
         ↓ sparse sampling
Support Points (~5000 points)
         ↓ UOT solver
Coupling Matrix (~5000 × ~5000)
         ↓ upsample back
Canonical Results
```

**Key Numbers**:
- Canonical grid: 256 × 576 = 147,456 pixels
- Canonical target resolution: 10.0 μm/px (configurable via `canonical_grid_um_per_pixel`)
- Source support points (src_support): ~5000 (via `max_support_points` parameter)
- Target support points (tgt_support): ~5000 (via `max_support_points` parameter)
- Coverage: 5000 / 147,456 ≈ 3.4% (but src and tgt have different spatial locations!)

### Why This Design?

1. **Computational efficiency**: Solver is O(n²) in support points, not pixels
2. **Biological accuracy**: Actual cell motion is sparse and localized
3. **Sampling theory**: ~5000 points sufficient for ~150k pixel domain
4. **Separate src/tgt sampling**: Source and target have different shapes, so they get different support point sets

### Critical Consequence

**ALL metrics derived from the coupling matrix are implicitly SAMPLING-BASED.**

If you see:
- `destroyed_mass_pct = 0.02%`
- `mean_velocity_px = 15.0`
- `support_coverage = 3.4%`

These are NOT "the full source mass" or "all pixels". They are **calculations at the 5000 support points**, extrapolated to the full domain.

---

## Mass Metrics Contract

### Definition

**created_mass_pct** and **destroyed_mass_pct** are:

```python
created_mass_pct = (sum_of_mass_created_at_tgt_support_points / total_target_mass) × 100
destroyed_mass_pct = (sum_of_mass_destroyed_at_src_support_points / total_source_mass) × 100
```

Where:
- `sum_of_mass_created_at_tgt_support_points` = sum at ~5000 **target** support points only
- `sum_of_mass_destroyed_at_src_support_points` = sum at ~5000 **source** support points only
- `total_target_mass` = sum of entire target mask (all pixels)
- `total_source_mass` = sum of entire source mask (all pixels)

**CRITICAL**: The solver samples support points from BOTH source and target separately:
- **src_support**: ~5000 points sampled from source mask
- **tgt_support**: ~5000 points sampled from target mask

These are **different sets of points**, so creation and destruction have different coverage percentages!

### Semantics

| Metric | Means | Does NOT Mean |
|--------|-------|---------------|
| `destroyed_mass_pct = 0.02%` | Of the total source mass, 0.02% was destroyed (as measured at support points) | Only 0.02% of pixels have motion |
| `destroyed_mass_pct = 2.5%` | Of the total source mass, 2.5% was destroyed (as measured at support points) | All source pixels lost 2.5% mass |
| `destroyed_mass_pct = 100.0%` | Complete source annihilation (all source mass became creation) | Impossible with current regularization |

### Design Rationale

This metric answers: **"From the perspective of the solver's sampled points, how much total mass moved?"**

- Identity test (src == tgt): expect ~0% (some rounding error)
- Translation test (src → tgt, different location): expect ~0% creation/destruction
- Shape change test (circle → oval): expect >0% at boundaries

### Common Mistakes

❌ **Mistake**: "destroyed_mass_pct is the fraction of source pixels that moved"  
✅ **Correct**: "destroyed_mass_pct is the fractional amount of total source mass lost, calculated from ~5000 sampled support points"

❌ **Mistake**: "If destroyed_mass_pct = 2%, then 98% of the source persists"  
✅ **Correct**: "If destroyed_mass_pct = 2%, then at the sampled support points, 2% of total source mass was lost; the other 98% was transported"

---

## Velocity Field Contract

### Definition

```python
velocity_field: (H, W, 2) array, units = μm/frame
```

Where:
- Non-zero at ~5000 support points
- Zeros elsewhere (from `np.kron` replication)
- (H, W) matches canonical grid dimensions

### Semantics

| Pixel | Velocity | Meaning | NOT Meaning |
|-------|----------|---------|------------|
| Support point | 5.2 μm/frame | This support point moved 5.2 μm/frame | All nearby pixels moved 5.2 μm/frame |
| Non-support pixel | 0.0 | Not a support point; velocity is undefined | Pixel has zero velocity / no motion |
| After NaN masking | NaN | Not a support point; velocity is undefined | This mask is better; use it |

### The Problem with Zeros

When the velocity field is broadcast back to canonical grid via `np.kron`, non-support pixels are filled with **0.0**.

This is semantically wrong because:
- 0.0 means "zero velocity" (physics: not moving)
- But these pixels are "undefined velocity" (sampling: not measured)

### The Solution: NaN Masking

After rasterization, the plotting contract requires:
```python
velocity_mag_masked = velocity_mag.copy()
support_mask = velocity_mag > 0
velocity_mag_masked[~support_mask] = np.nan
```

Now:
- Support points: meaningful velocity values
- Non-support: NaN (explicitly undefined)
- Visualization: only shows support points (no fabricated data)

### Velocity Statistics

**mean_velocity_px**: Includes all pixels (mostly zeros)
```
mean_velocity_px = mean(velocity_field)  # ≈ 0.03 μm/frame (mostly zeros!)
```

**mean_nonzero_velocity_px**: Only support points
```
mean_nonzero_velocity_px = mean(velocity_field[velocity_field > 0])  # ≈ 8.5 μm/frame
```

Use `mean_nonzero_velocity_px` for biological interpretation!

---

## Canonical Alignment Contract

**Contract**: Every frame is independently canonicalized into a **biological reference frame** before UOT. This is not pairwise alignment; it is alignment to a shared stereotype.

**Coordinate system**: **Image coordinates** (origin top-left, +y down). All alignment logic uses this convention.

### Alignment Steps (CanonicalAligner)

1. **Scale**  
   `scale = source_um_per_px / target_um_per_px`  
   Canonical target is **10.0 μm/px** unless overridden by config.

2. **PCA Long-Axis Alignment**  
   Rotate so major axis matches grid long axis (x for landscape, y for portrait).  
   PCA rotation is forced when `use_pca=True`.

3. **Flip Search (Chirality)**  
   Evaluate candidate transforms (0/180 degrees + optional horizontal flip).  
   Score favors:
   - **Head/yolk near origin** (minimize x+y)
   - **Back far from origin** (maximize x+y)

4. **Yolk Anchoring**  
   If yolk exists and `anchor_mode=yolk_anchor`, translate so yolk COM lands at anchor.  
   Default anchor = center `(0.5, 0.5)` of grid (configurable).

5. **Clipping Handling**  
   If mask exceeds grid, **warn by default** (configurable to error).

### Back (Dorsal) Definition

The “back” is computed from a **yolk-centered ring** (donut) to avoid tail bias:
- Inner radius = `1.2 × yolk_radius`
- Outer radius = `3.0 × yolk_radius`
- Fallback to far-quantile if ring is sparse

This focuses alignment on the dorsal region near the yolk rather than distal tail mass.

---

## Yolk Mask Contract

**Source**: Build02 predictions under:

```
<data_root>/segmentation/yolk_v1_0050_predictions/{experiment_date}/
```

**Loading**:
- `load_mask_from_csv(..., data_root=...)` loads embryo mask + yolk mask (if present)
- If `data_root` not passed, environment variable `MORPHSEQ_DATA_ROOT` is used
- Yolk mask is attached to `UOTFrame.meta["yolk_mask"]`

**Missing yolk**:
If yolk is missing or empty, alignment falls back to embryo COM and/or back-quantile.

---

## Transport Cost Attribution Contract

### Definition

Per-support transport cost is attributed as:

```text
cost_src[i] = sum_j (P_ij * C_ij)
cost_tgt[j] = sum_i (P_ij * C_ij)
```

Where:
- `P` is the (rescaled) coupling in pixel-mass units
- `C` is the pairwise cost matrix in solver coordinate units
- `cost_src` and `cost_tgt` are 1D arrays aligned to src/tgt support points

These are stored on the `UOTResult`:
- `cost_src_support`, `cost_tgt_support`: per-support vectors (length n_src/n_tgt)
- `cost_src_px`, `cost_tgt_px`: rasterized maps in canonical/work grid for plotting

### Semantics

| Field | Means | Does NOT Mean |
|-------|-------|---------------|
| `cost_src_support` | Cost attributable to each src support point | Cost per canonical pixel everywhere |
| `cost_tgt_support` | Cost attributable to each tgt support point | Cost per canonical pixel everywhere |
| `cost_src_px` | Rasterized src-support cost map | Physical cost at non-support pixels |
| `cost_tgt_px` | Rasterized tgt-support cost map | Physical cost at non-support pixels |

### Plotting Rule

Plots should use `cost_src_px` / `cost_tgt_px` directly (with NaN masking),
and should **not** recompute cost from `P` and `C` inside the plotting layer.

---

## Grid Hierarchy Contract

### Grid Levels

```
┌─────────────────────────────────────────┐
│  Canonical Grid                         │
│  (256 × 576 px, 10.0 μm/px)            │
│  Physical size: 2.56 mm × 5.76 mm      │
│  Used for: input/output, visualization │
└──────────────┬──────────────────────────┘
               │
               ├─ Canonical alignment (PCA + flip + yolk anchor)
               │  (independent per frame)
               ├─ Pair crop (bbox)
               │  (variable size, cropped region)
               │
               ├─ Padding (crop_pad_hw)
               │  (boundary pixels for derivative stability)
               │
               └─ Work Grid (downsampled)
                  (Hw × Ww, with padding)
                  ├─ Coverage valid mask
                  │  (1 = real, 0 = padded)
                  │
                  └─ Support Points (~5000)
                     (sampled from work grid)
```

### Rasterization Rules

**From canonical → work grid:**
- Downsample via max-pooling (preserve spatial locality)

**From work grid → canonical:**
- Upsample via `np.kron` (replicate blocks)
- Apply coverage normalization for **mass** (divide by # of real pixels in block)
- NO coverage normalization for **velocity** (velocity is not conserved)

**Critical**: Padding pixels (outside real crop) must be zeroed after rasterization.

---

## Visualization Contract

**Principle**: Never fabricate data. Show what was sampled, clearly labeled.

**Axis convention**: Plotting defaults to **image coordinates** (y down).  
If a Cartesian view is desired, only the display transforms change (never the data).

### The Three Panels (Enforced in flow_field.png)

#### Panel 1: Support Mask
```
axes[0].imshow(support_mask, cmap="gray")
title: "Support Coverage\n{support_pct:.2f}% defined"
```

Shows: Where the solver actually computed velocity (3.4% of pixels)

#### Panel 2: Velocity Magnitude (NaN-Masked)
```
velocity_mag_masked = velocity_mag.copy()
velocity_mag_masked[~support_mask] = np.nan
axes[1].imshow(velocity_mag_masked, cmap="viridis")
title: "Velocity Magnitude (support only)\np50/p90/p99: {p50}/{p90}/{p99}"
```

Shows: Velocity only where defined (support points)  
Statistics: Calculated on support only (no zeros from padding)

#### Panel 3: Velocity Histogram
```
axes[2].hist(support_velocities, bins=...)
title: "Velocity Distribution (% of all pixels)"
```

Shows: Distribution of velocity magnitudes  
X-axis label: Shows as % of total pixels (emphasizes sparsity)

### The Four Panels (Enforced in creation_destruction.png)

#### Panel (0,0): Creation Support Mask
#### Panel (0,1): Destruction Support Mask
- Shows where mass was created/destroyed (non-zero locations)
- Title includes: "{pct_defined:.2f}% defined" for this metric

#### Panel (1,0): Creation Mass Heatmap (NaN-Masked)
```
created_masked = created_mass.copy()
created_masked[~created_mask] = np.nan
axes[1,0].imshow(created_masked, cmap="Reds")
title: "Mass Created (tgt_support only)\n{created_mass_pct:.2f}% of total target (from tgt_support sampling ONLY!)"
```

Title breakdown:
- `tgt_support only` - calculated at TARGET support points (~5000 points sampled from target)
- `{created_mass_pct:.2f}%` - percentage of total (denominator: all target pixels)
- `from tgt_support sampling ONLY!` - reminder that this is partial sampling from target

#### Panel (1,1): Destruction Mass Heatmap (NaN-Masked)
```
destroyed_masked = destroyed_mass.copy()
destroyed_masked[~destroyed_mask] = np.nan
axes[1,1].imshow(destroyed_masked, cmap="Blues")
title: "Mass Destroyed (src_support only)\n{destroyed_mass_pct:.2f}% of total source (from src_support sampling ONLY!)"
```

Title breakdown:
- `src_support only` - calculated at SOURCE support points (~5000 points sampled from source)
- `{destroyed_mass_pct:.2f}%` - percentage of total (denominator: all source pixels)
- `from src_support sampling ONLY!` - reminder that this is partial sampling from source

**Why Different Coverage Percentages?**

If you see:
- Creation Support (tgt_support): 2.99% defined
- Destruction Support (src_support): 3.41% defined

This is **normal and expected**! Source and target have different shapes/sizes, so they get sampled at different spatial densities. The solver samples ~5000 points from each independently.

### Quiver Plot (creation_destruction_quiver.png - stride=6)

```
Only plot arrows at support points:
axes.quiver(xx[support_mask], yy[support_mask], u[support_mask], v[support_mask])
title: "Velocity Field (Quiver, stride={stride})\nSupport: {support_pct:.2f}%, Arrows: {n_arrows}"
```

Shows: Arrows only at sampled points (no interpolation between support)

### Forbidden Practices

❌ Don't smooth velocity field to "fill" zero regions  
❌ Don't show zeros as colored pixels (use NaN instead)  
❌ Don't plot velocity statistics across all pixels (use support-only)  
❌ Don't title: "Mass Created: 0.5%" without saying "from support points"  
❌ Don't use grayscale for NaN regions (make them visibly blank)

---

## Data Flow Diagram

```
                        INPUT
                          │
                  ┌───────┴────────┐
                  ▼                ▼
            Raw Src Mask     Raw Tgt Mask
            (+ optional yolk) (+ optional yolk)
                  │                │
                  ├── Canonical alignment (PCA + flip + yolk anchor)
                  │   (independent per frame)
                  ▼                ▼
            Source Mask     Target Mask
            (canonical)     (canonical)
                  │                │
                  ├─ Pair crop box (via pair_frame)
                  │
                  ▼
            Work Grid (padded)
                  │
        ┌─────────┼─────────┐
        │         │         │
        ▼         ▼         ▼
    Coverage  Source pts  Target pts (~5000 each)
    Valid Mask    │          │
        │         └──┬───────┘
        │            │
        ▼            ▼
    Weights   Cost Matrix (sparse)
    (mu, nu)        │
        │           ▼
        │    Gibbs Kernel K = exp(-C/ε)
        │           │
        │           ▼
        │    Sinkhorn Coupling
        │    (sparse solution)
        │           │
    ┌───┴───────────┤
    │               │
    ▼               ▼
Marginals    Coupling Matrix P
(mu_hat,  (~5000 × ~5000)
 nu_hat)
    │               │
    ├───┬───────────┤
    │   │           │
    ▼   ▼           ▼
Mass Transport Target Position
Created Velocity (via barycentric
Destroyed   Field   projection T)
    │       │           │
    └─┬─────┴─┬─────────┘
      │       │
      ▼       ▼
   Rasterize to Work Grid
   (coverage-aware for mass,
    velocity direct)
      │
      ▼
   Upscale to Canonical
   (via kron, apply padding mask)
      │
      ├─ Apply NaN masking
      │  (non-support → NaN)
      │
      └─ OUTPUT
         ├─ mass_created_canonical (canonical shape, NaN outside)
         ├─ mass_destroyed_canonical (canonical shape, NaN outside)
         ├─ velocity_canonical (canonical shape, zeros→NaN)
         ├─ created_mass_pct (support-only metric)
         ├─ destroyed_mass_pct (support-only metric)
         └─ support_mask (for visualization)
```

---

## Directory Overview

**Scope**: UOT morphometrics lives under `src/analyze/optimal_transport_morphometrics/`.

```
src/analyze/optimal_transport_morphometrics/
├─ docs/
│  └─ DATA_CONTRACTS.md          # this file (ground truth semantics)
└─ uot_masks/
   ├─ frame_mask_io.py           # load masks/yolk from CSV + data_root
   ├─ uot_grid.py                # CanonicalAligner (PCA, flip, anchor)
   ├─ preprocess.py              # canonicalize + crop + rasterize
   ├─ run_transport.py           # main UOT execution path
   ├─ run_timeseries.py          # time series driver
   ├─ viz.py                     # plotting helpers
   ├─ benchmark_resolution.py    # resolution experiments
   └─ calibrate_marginal_relaxation.py
```

**Related**:
- `src/analyze/utils/optimal_transport/config.py` — canonical grid config (10.0 μm/px default)
- `results/mcolon/20260121_uot-mvp/` — debug scripts and outputs used for UOT MVP

---

## Project Status / Next Steps

**Current status (Jan 24, 2026)**:
- Canonical alignment is working with PCA + flip search + yolk anchoring.
- Yolk masks can be loaded from Build02 predictions (`MORPHSEQ_DATA_ROOT`).
- Debug overlays (masks + quiver + cost) are in place for rapid inspection.

**Next steps** (planned):
1. Extract standardized visualizations (batch exports for QA and review).
2. Cluster dynamics across embryos/timepoints using UOT-derived features.
3. GPU acceleration for UOT (speedup from current CPU-bound runs).

---

## Implementation Checklist

When adding new features to this pipeline:

- [ ] **If using mass values**: Document whether calculated from support or full grid
- [ ] **If computing statistics**: Explicitly state "on support only" or "across all pixels"
- [ ] **If visualizing**: Apply NaN masking to non-support regions
- [ ] **If labeling plots**: Include "(from src_support sampling ONLY!)" where applicable
- [ ] **If writing metrics CSV**: Add columns explaining the denominator (total vs support)
- [ ] **If threshold filtering**: Use IQR-based or percentile (not max) for robustness
- [ ] **If documenting elsewhere**: Link back to this file

---

## Examples of Correct Language

✅ **Correct**: "destroyed_mass_pct = 0.02%, indicating minimal mass loss at the sampled support points"

✅ **Correct**: "p50 velocity at support points: 5.2 μm/frame; this is the typical speed among sampled regions"

✅ **Correct**: "Support coverage is 3.4%, so mass percentages represent ~5000 sampled locations, not the full source/target"

✅ **Correct**: "The velocity field uses NaN for non-support pixels to avoid fabricating data"

---

## Examples of WRONG Language

❌ **Wrong**: "destroyed_mass_pct = 0.02%, so only 0.02% of the source lost mass"

❌ **Wrong**: "mean_velocity_px = 15.0 μm/frame, so all pixels are moving 15 μm/frame"

❌ **Wrong**: "3.4% of pixels have defined velocity, so 96.6% are stationary"

❌ **Wrong**: "Zero velocity means the pixel is not moving" (in the context of non-support pixels)

---

## Questions & Answers

### Q: Why not just compute on the full grid?
A: Computational cost is O(n²) in grid size. Full grid would be ~20 billion operations; support points ~25 million.

### Q: Why 5000 support points specifically?
A: Sampling density of ~0.03 points/μm² is sufficient for the ~150k pixel canonical grid based on biological motion scales.

### Q: Can I interpolate velocity at non-support pixels?
A: No. That fabricates data. If you need full-grid velocity, modify the sampling strategy.

### Q: What if I need to know "total source behavior"?
A: You have it: created_mass_pct and destroyed_mass_pct are calculated relative to total source/target. But remember they're sampling-based estimates.

### Q: Should I use mean_velocity_px or mean_nonzero_velocity_px?
A: For biology: use mean_nonzero_velocity_px (only support points). For diagnostics: use mean_velocity_px (includes sampling sparsity).

---

## Related Documents

- [Debug Parameters README](../../../results/mcolon/20260121_uot-mvp/DEBUG_PARAMS_README.md) - practical usage guide
- [Transport Maps](../utils/optimal_transport/transport_maps.py) - implementation details
- [UOT Config](../utils/optimal_transport/config.py) - configuration options
