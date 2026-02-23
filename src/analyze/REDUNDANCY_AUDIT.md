# Plotting System Redundancy Audit

**Date:** January 22, 2026  
**Auditor:** AI Assistant  
**Status:** ✅ **CLEAN - No Major Redundancy Found**

---

## Summary

Checked for redundant plotting functionality between:
- `src/analyze/utils/`
- `src/analyze/viz/`  
- `src/analyze/trajectory_analysis/viz/`

**Result:** The systems are properly separated with clear responsibilities. No significant redundancy detected.

---

## System Architecture (Hierarchical)

```
src/analyze/
├── utils/                          # Level 0: Generic utilities
│   ├── timeseries/                 # DTW, DBA, interpolation (domain-agnostic)
│   ├── optimal_transport/          # OT algorithms
│   ├── binning.py                  # Time binning
│   └── (No plotting!)              ✅ Clean separation
│
├── viz/                            # Level 1: Generic plotting
│   └── plotting/
│       ├── time_series.py          # plot_feature_over_time()
│       └── faceted/
│           └── time_series.py      # plot_feature_over_time (row_by/col_by)
│
└── trajectory_analysis/            # Level 2: Domain-specific
    └── viz/
        ├── plotting/
        │   ├── core.py             # Trajectory-specific 2D plots
        │   ├── plotting_3d.py      # 3D scatter (Plotly)
        │   └── faceted/            # Trajectory faceted plots
        └── dendrogram.py           # Cluster dendrograms
```

---

## Responsibilities by Module

### 1. **utils/** - Pure Algorithms ✅
**Purpose:** Domain-agnostic algorithmic utilities  
**Contains:**
- `timeseries/dba.py` - DTW Barycenter Averaging
- `timeseries/dtw.py` - DTW distance computation  
- `timeseries/interpolation.py` - Trajectory interpolation
- `binning.py` - Time bin utilities
- `optimal_transport/` - OT algorithms

**Key Point:** No plotting code! This is clean.

---

### 2. **viz/plotting/** - Generic Plotting ✅  
**Purpose:** Domain-agnostic time series visualization  
**Uses:** `utils/timeseries/` for algorithms  
**Contains:**
- `time_series.py` - `plot_feature_over_time()`
- `faceted/time_series.py` - removed (use `plot_feature_over_time` with `row_by` / `col_by`)

**API:**
- Parameters: `feature=`, `time_col=`, `id_col=`, `color_by=`
- No domain logic (no genotypes, pairs, etc.)
- Works with any time series data

**Key Point:** This is the bioinformatics-standard API we refactored to.

---

### 3. **trajectory_analysis/viz/** - Domain-Specific ✅
**Purpose:** Trajectory-specific visualizations with domain knowledge  
**Uses:** Both `viz/plotting/` and adds domain logic  
**Contains:**

#### `plotting/core.py`
- Trajectory-specific 2D plots
- Uses trajectory-specific styling (genotypes, phenotypes)

#### `plotting/plotting_3d.py`  
- **Unique functionality:** 3D interactive Plotly scatter plots
- For embedding spaces, PCA, latent representations
- **Not redundant** - this is specialized 3D visualization

#### `plotting/faceted/`
- Trajectory-specific faceted plots
- `plot_feature_over_time()` for single-feature faceting
- Genotype-aware coloring, pair analysis logic

#### `dendrogram.py`
- Cluster dendrograms
- Uses `scipy.cluster.hierarchy`

**Key Point:** These add domain knowledge on top of generic tools.

---

## No Redundancy - Here's Why

| Feature | utils/ | viz/ | trajectory_analysis/viz/ |
|---------|--------|------|--------------------------|
| DTW algorithm | ✅ | Uses it | Uses it |
| DBA algorithm | ✅ | Uses it | Uses it |
| Generic time series plots | ❌ | ✅ | Uses it |
| Genotype styling | ❌ | ❌ | ✅ |
| 3D scatter plots | ❌ | ❌ | ✅ (Unique!) |
| Dendrograms | ❌ | ❌ | ✅ (Unique!) |
| Pair analysis | ❌ | ❌ | ✅ (Unique!) |

---

## Potential Confusion Points (Resolved)

### 1. **Two Time Series Plotting Systems**
**Question:** Why `viz/plotting/time_series.py` AND `trajectory_analysis/viz/plotting/core.py`?

**Answer:** Different purposes:
- `viz/` = Generic (any time series, any feature)
- `trajectory_analysis/viz/` = Domain-specific (trajectories with genotypes, pairs)

**No redundancy** - they build on each other hierarchically.

### 2. **plotting_3d.py Location**
**Question:** Should 3D plotting be in `viz/`?

**Answer:** No - it's trajectory-specific:
- Uses trajectory-specific color palettes (`PHENOTYPE_COLORS`)
- Has domain parameters (`line_by='embryo_id'`)
- Designed for embedding/PCA spaces from trajectory analysis

**Correct location:** `trajectory_analysis/viz/plotting/plotting_3d.py` ✅

### 3. **Faceted Plotting Systems**
**Question:** Why faceted plots in both locations?

**Answer:**
- `viz/plotting/faceted/` = Generic (feature-first API)
- `trajectory_analysis/viz/plotting/faceted/` = Trajectory-specific (genotype coloring, pair logic)

**No redundancy** - different use cases.

---

## Shims to Remove (From Previous Plan)

These are the only cleanup items:

1. ✅ Delete `trajectory_analysis/viz/plotting/faceted.py` (deprecated shim)
2. ✅ Delete `trajectory_analysis/viz/plotting/faceted.py.bak` (backup)

---

## Recommended Actions

### ✅ Keep Current Structure
The hierarchical organization is correct:
```
utils/          → Pure algorithms (no plotting)
viz/            → Generic plotting (domain-agnostic)
trajectory_*/   → Domain-specific (adds domain logic)
```

### ✅ No Code to Move or Consolidate
Everything is in the right place.

### ✅ Just Remove Deprecated Shims
Execute the plan in `CLEANUP_PLOTTING_SHIMS_PLAN.md`

### ✅ Add Documentation
Create `viz/plotting/README.md` to clarify the two systems (see cleanup plan).

---

## Anti-Patterns NOT Present ✅

- ❌ **Duplicate implementations** - Not found
- ❌ **Copy-pasted code** - Not found  
- ❌ **Plotting in utils/** - Not present (correct!)
- ❌ **3D plotting in multiple places** - Only one location
- ❌ **Generic plotting in trajectory_analysis/** - Not present

---

## Conclusion

**The codebase is well-organized.** There is NO significant redundancy to remove. The apparent duplication is actually proper **hierarchical composition**:

1. **utils/** provides algorithms
2. **viz/** provides generic plotting using those algorithms
3. **trajectory_analysis/viz/** provides domain-specific plotting using both

The only cleanup needed is removing the deprecated shim files from the previous refactor.

---

## Next Steps

1. Execute `CLEANUP_PLOTTING_SHIMS_PLAN.md` to remove deprecated files
2. Add `viz/plotting/README.md` to document the two systems
3. Consider this audit complete ✅
