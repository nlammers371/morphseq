# MD-DTW Analysis Plan for b9d2 Phenotype Divergence

**Date:** December 18, 2025  
**Location:** `results/mcolon/20251218_MD-DTW-morphseq_analysis/`

---

## Inspiration & Reference Documents

- [MD-DTW_approahc.md](MD-DTW_approahc.md) - Original strategy document outlining the MD-DTW approach
- [Initial_notes-on-b9d2-phenotype-pair-trends.md](Initial_notes-on-b9d2-phenotype-pair-trends.md) - Notes on b9d2 phenotype observations
- `src/analyze/trajectory_analysis/PLOTTING_README.md` - Existing two-level plotting architecture (Level 1: generic faceting, Level 2: pair-specific)

---

## Objective

Distinguish **HTA (Head-Trunk Angle)** vs **CE (Convergent Extension)** phenotypes in `b9d2` mutants using **Multivariate Dynamic Time Warping (MD-DTW)** on joint `[Z_curvature, Z_length]` trajectories.

---

## Background

### The Problem: Coupled Phenotypic Divergence

The `b9d2` mutants exhibit two distinct phenotypic endpoints that share a common early developmental trajectory but diverge later in time:

- **HTA (Head-Trunk Angle):** Body axis defect characterized by curvature deviations *without* severe shortening
- **CE (Convergent Extension):** Early curvature deviations that evolve into severe shortening after ~32hpf

### Why MD-DTW?

1. **Shared Features:** Both phenotypes share high curvature initially - univariate analysis groups them incorrectly
2. **Coupled Signal:** The distinguishing factor is the *relationship* between Length and Curvature over time
3. **Heterochrony:** Onset timing varies between embryos - DTW handles temporal shifts elastically

### Phenotype Assignments by Pair

| Pair | Phenotype | Notes |
|------|-----------|-------|
| `b9d2_pair_5` | **HTA** | Head-Trunk Angle defect (curvy, normal length) |
| `b9d2_pair_6` | **CE** | Convergent Extension defect (curvy → short) |
| `b9d2_pair_7` | **CE** | Convergent Extension defect |
| `b9d2_pair_8` | **CE** | Convergent Extension defect |
| `b9d2_pair_1` | **Non-penetrant** | No phenotype in progeny |
| `b9d2_pair_2` | **Non-penetrant** | No phenotype in progeny |

---

## Phase 1: Prototype Development

**Location:** This directory (`results/mcolon/20251218_MD-DTW-morphseq_analysis/`)

### Files to Create

| File | Purpose |
|------|---------|
| `md_dtw_prototype.py` | Core MD-DTW distance matrix computation & clustering |
| `multimetric_plotting.py` | New multi-metric visualization functions |
| `run_analysis.py` | Main analysis script tying everything together |

---

## Phase 2: Implementation Details

### 2.1 Data Loading & Preparation

```python
from src.analyze.trajectory_analysis.data_loading import load_experiment_dataframe

# Load experiments with tracked b9d2 pairs
experiment_ids = ['20251104', '20251119', '20251121']
df = load_experiment_dataframe(exp_id, format_version='qc_staged')
```

**Preprocessing Steps:**
1. Filter for `b9d2_pair_*` genotypes
2. Extract features: `baseline_deviation_normalized`, `total_length_um`
3. Z-score normalize each feature (equal DTW weight via `StandardScaler`)
4. Interpolate to common time grid
5. Output: 3D array `(N_embryos, T_timesteps, 2_features)`

### 2.2 MD-DTW Distance Matrix

**Tool:** `tslearn.metrics.cdist_dtw`

```python
from tslearn.metrics import cdist_dtw
from sklearn.preprocessing import StandardScaler

# X shape: (n_embryos, n_timepoints, n_features)
distance_matrix = cdist_dtw(X, X, sakoe_chiba_radius=3)
```

**Key Considerations:**
- Sakoe-Chiba band constraint for efficiency
- Handle variable-length trajectories via interpolation
- Symmetric distance matrix output

### 2.3 Clustering Pipeline

**Use existing infrastructure:**

```python
from src.analyze.trajectory_analysis import (
    run_bootstrap_hierarchical,
    compute_posteriors,
    classify_membership,
)

# Bootstrap clustering on MD-DTW distance matrix
results = run_bootstrap_hierarchical(distance_matrix, k=3, embryo_ids=embryo_ids)

# Posterior probabilities
posteriors = compute_posteriors(results)

# Core/uncertain/outlier classification
classifications = classify_membership(posteriors)
```

**Manual K-Selection:** Visually inspect dendrogram to identify stable clusters that separate HTA from CE.

### 2.4 New Multi-Metric Visualization Functions

**Problem:** Current `faceted_plotting.py` only supports single-metric y-axis.

**Solution:** Create `multimetric_plotting.py` with new functions that **leverage the existing faceted plotting infrastructure**.

---

## New Tools to Create

### Summary Table

| Tool | File | Purpose | Priority | Leverages Existing |
|------|------|---------|----------|-------------------|
| `compute_md_dtw_distance_matrix()` | `md_dtw_prototype.py` | Compute multivariate DTW distance matrix | Required | None (new capability) |
| `prepare_multivariate_array()` | `md_dtw_prototype.py` | Convert DataFrame → 3D array for tslearn | Required | `trajectory_utils.py` patterns |
| `plot_dendrogram()` | `md_dtw_prototype.py` | Interactive/static dendrogram for K-selection | **PRIMARY** | `scipy.cluster.hierarchy` |
| `plot_multimetric_trajectories()` | `multimetric_plotting.py` | **PRIMARY:** Rows=metrics, Cols=clusters grid | **PRIMARY** | `faceted_plotting.py` |
| `plot_cluster_summary()` | `multimetric_plotting.py` | Mean ± SD ribbons per cluster | Secondary | `compute_binned_mean()` |
| `plot_phase_plane()` | `multimetric_plotting.py` | 2D phase space trajectory paths | Optional | `_plot_faceted_plotly()` patterns |

---

### Tool 1: `compute_md_dtw_distance_matrix()`

**Location:** `md_dtw_prototype.py`

Computes pairwise multivariate DTW distances using `tslearn`.

```python
def compute_md_dtw_distance_matrix(
    X: np.ndarray,  # Shape: (n_embryos, n_timepoints, n_features)
    sakoe_chiba_radius: int = 3,
    n_jobs: int = -1,
    verbose: bool = False,
) -> np.ndarray:
    """
    Compute MD-DTW distance matrix for multivariate time series.
    
    Uses tslearn.metrics.cdist_dtw which handles multivariate data natively.
    
    Returns:
        distance_matrix: (n_embryos, n_embryos) symmetric distance matrix
    """
```

---

### Tool 2: `prepare_multivariate_array()`

**Location:** `md_dtw_prototype.py`

Converts long-format DataFrame to 3D array for tslearn. Leverages existing `trajectory_utils.py` patterns.

```python
def prepare_multivariate_array(
    df: pd.DataFrame,
    metrics: List[str],  # ['baseline_deviation_normalized', 'total_length_um']
    time_col: str = 'predicted_stage_hpf',
    embryo_id_col: str = 'embryo_id',
    time_grid: Optional[np.ndarray] = None,  # Interpolation grid
    normalize: bool = True,  # Z-score each feature
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Convert long-format DataFrame to tslearn-compatible 3D array.
    
    Returns:
        X: (n_embryos, n_timepoints, n_features) array
        embryo_ids: List of embryo identifiers
        time_grid: The time grid used
    """
```

---

### Tool 2.5: `plot_dendrogram()` **[PRIMARY - K-SELECTION]**

**Location:** `md_dtw_prototype.py`

**Status:** **PRIMARY** - Required for manual K-selection (choosing number of clusters).

**Leverages:** Bootstrap clustering results from `run_bootstrap_hierarchical()` and `scipy.cluster.hierarchy` for dendrogram visualization.

```python
def plot_dendrogram(
    bootstrap_results: Dict[str, Any],  # Output from run_bootstrap_hierarchical()
    distance_matrix: np.ndarray,  # Original distance matrix
    k_values: Optional[List[int]] = None,  # Try multiple K values (e.g., [2, 3, 4, 5])
    method: str = 'average',  # 'average', 'ward', 'complete', 'single'
    color_threshold: Optional[float] = None,
    show_labels: bool = True,
    show_posterior_colors: bool = True,  # Color leaves by cluster confidence
    backend: str = 'plotly',  # 'plotly', 'matplotlib', or 'both'
    output_path: Optional[Path] = None,
    title: str = 'MD-DTW Hierarchical Clustering Dendrogram',
) -> Any:
    """
    Plot dendrogram from bootstrap clustering results for K-selection.
    
    Takes the output from run_bootstrap_hierarchical() and visualizes the
    hierarchical clustering structure. Optionally overlays cluster assignments
    for multiple K values to help choose the optimal number of clusters.
    
    Key Features:
    - Shows bootstrap consensus clustering structure
    - Can overlay multiple K-value cuts for comparison
    - Colors leaves by cluster membership confidence (posterior probability)
    - Interactive (Plotly) or static (matplotlib) output
    
    Example:
        >>> # First run bootstrap clustering
        >>> bootstrap_results = run_bootstrap_hierarchical(
        ...     distance_matrix, k=3, embryo_ids=embryo_ids, n_bootstrap=100
        ... )
        
        >>> # Then plot dendrogram to validate K choice
        >>> plot_dendrogram(
        ...     bootstrap_results,
        ...     distance_matrix,
        ...     k_values=[2, 3, 4],  # Compare different K values
        ...     backend='both',
        ...     output_path='figures/dendrogram'
        ... )
        # Inspect dendrogram to confirm K=3 is appropriate
    """
```

---

### Tool 3: `plot_multimetric_trajectories()` **[PRIMARY VISUALIZATION]**

**Location:** `multimetric_plotting.py`

**Status:** **PRIMARY** - This is the main visualization for multi-metric trajectory analysis.

**Leverages:** Existing `faceted_plotting.py` architecture - specifically the `_prepare_facet_grid_data()` and backend pattern (`_plot_faceted_plotly()`, `_plot_faceted_matplotlib()`).

**Design: "Double-Decker" Stacked Time-Series Layout**

**Grid Structure:**
- **Rows:** Metrics (Row 1 = Curvature, Row 2 = Length)
- **Columns:** Clusters (Col 1 = Cluster 0, Col 2 = Cluster 1, Col 3 = Cluster 2, etc.)
- **X-axis:** Time (hpf), **shared and aligned** across all panels
- **Y-axis:** Metric value (can be normalized or raw)

```
         Cluster 0      Cluster 1      Cluster 2
Row 1:  [Curvature]    [Curvature]    [Curvature]  <- All share x-axis (time)
Row 2:  [Length]       [Length]       [Length]     <- All share x-axis (time)
```

```python
def plot_multimetric_trajectories(
    df: pd.DataFrame,
    metrics: List[str],  # ['baseline_deviation_normalized', 'total_length_um']
    cluster_col: str = 'cluster',  # Column defining clusters
    x_col: str = 'predicted_stage_hpf',
    line_by: str = 'embryo_id',
    color_by: str = 'cluster',  # 'cluster', 'genotype', 'pair'
    metric_labels: Optional[Dict[str, str]] = None,  # Display names
    share_x: bool = True,  # Share x-axis across all panels (recommended)
    share_y: str = 'row',  # Share y-axis: 'row', 'all', or 'none'
    backend: str = 'plotly',
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    **kwargs,
) -> Any:
    """
    Create multi-metric time-series plots with metrics in rows, clusters in columns.
    
    **PRIMARY VISUALIZATION for MD-DTW analysis.**
    
    Grid Layout:
    - Rows: One row per metric (Curvature, Length)
    - Columns: One column per cluster
    - Within each panel: Individual embryo trajectories
    
    Why This Wins:
    - **Cluster Comparison**: See all clusters side-by-side for each metric
    - **Synchrony**: Aligned time axes let you see when divergence happens
    - **Flexibility**: Color by cluster (identity), genotype (validation), or pair (biology)
    
    Example Usage:
        >>> # Version 1: Color by cluster (see cluster identity)
        >>> plot_multimetric_trajectories(
        ...     df,
        ...     metrics=['baseline_deviation_normalized', 'total_length_um'],
        ...     cluster_col='cluster',
        ...     color_by='cluster',
        ...     output_path='figures/by_cluster.html'
        ... )
        
        >>> # Version 2: Color by genotype (validate clusters match genotypes)
        >>> plot_multimetric_trajectories(
        ...     df,
        ...     metrics=['baseline_deviation_normalized', 'total_length_um'],
        ...     cluster_col='cluster',
        ...     color_by='genotype',  # See if clusters separate genotypes
        ...     output_path='figures/by_genotype.html'
        ... )
        
        >>> # Version 3: Color by pair (validate clusters match pairs)
        >>> plot_multimetric_trajectories(
        ...     df,
        ...     metrics=['baseline_deviation_normalized', 'total_length_um'],
        ...     cluster_col='cluster',
        ...     color_by='pair',  # See if clusters separate pairs
        ...     output_path='figures/by_pair.html'
        ... )
    """
```

---

### Tool 4: `plot_phase_plane()` **[OPTIONAL/EXPERIMENTAL]**

**Location:** `multimetric_plotting.py`

**Status:** **OPTIONAL** - Experimental visualization that may be harder to interpret than stacked time-series.

**Leverages:** Existing `_plot_faceted_plotly()` patterns for trace styling, hover, legend groups.

**Limitations:**
- Loses explicit time information (time becomes implicit in line progression)
- Harder to see *when* divergence occurs
- Less intuitive for biologists than aligned time-series

**When to Use:**
- After establishing clusters with stacked plots
- For theoretical/exploratory phase space analysis
- When presenting to audiences familiar with dynamical systems

```python
def plot_phase_plane(
    df: pd.DataFrame,
    x_metric: str,  # 'total_length_um' (x-axis)
    y_metric: str,  # 'baseline_deviation_normalized' (y-axis)
    time_col: str = 'predicted_stage_hpf',
    line_by: str = 'embryo_id',
    color_by: str = 'cluster',
    row_by: Optional[str] = None,  # Facet rows (e.g., 'pair')
    col_by: Optional[str] = None,  # Facet columns
    show_time_markers: bool = True,  # Dots at key timepoints (24, 32, 48 hpf)
    show_arrows: bool = False,  # Direction arrows on trajectories
    backend: str = 'plotly',
    output_path: Optional[Path] = None,
) -> Any:
    """
    Plot trajectory paths in 2D phase space (x_metric vs y_metric).
    
    **OPTIONAL/EXPERIMENTAL:** Consider using plot_multimetric_trajectories first.
    
    Each embryo is a line showing its path through phase space over time.
    Time is encoded as line progression (start → end) and optionally markers.
    
    Note: Loses explicit temporal synchrony - harder to answer "when does
    divergence occur?" compared to stacked time-series plots.
    """
```

---

### Tool 5: `plot_cluster_summary()`

**Location:** `multimetric_plotting.py`

**Leverages:** Existing `compute_binned_mean()` from `pair_analysis/data_utils.py`.

```python
def plot_cluster_summary(
    df: pd.DataFrame,
    metrics: List[str],
    x_col: str = 'predicted_stage_hpf',
    cluster_col: str = 'cluster',
    show_individual: bool = False,  # Faint individual traces
    show_ribbon: bool = True,  # Mean ± SD ribbon
    ribbon_alpha: float = 0.2,
    backend: str = 'plotly',
    output_path: Optional[Path] = None,
) -> Any:
    """
    Plot cluster mean trajectories with confidence ribbons.
    
    Creates one column per metric with mean ± SD for each cluster.
    Makes it easy to see exactly when/where clusters diverge.
    """
```

---

## How We Leverage Existing Faceted Plotting

The existing `faceted_plotting.py` provides a robust two-level architecture (see `src/analyze/trajectory_analysis/PLOTTING_README.md`):

```
┌─────────────────────────────────────────────────────────────┐
│  Level 2: pair_analysis/plotting.py (Pair-Specific)         │
│  - plot_pairs_overview(), plot_genotypes_by_pair(), etc.    │
└───────────────────────┬─────────────────────────────────────┘
                        │ calls
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Level 1: faceted_plotting.py (Generic)                     │
│  - plot_trajectories_faceted(row_by, col_by, overlay, ...)  │
└─────────────────────────────────────────────────────────────┘
```

**Our extension adds a new Level 2 for multi-metric analysis:**

```
┌─────────────────────────────────────────────────────────────┐
│  Level 2: multimetric_plotting.py (Multi-Metric) [NEW]      │
│  - plot_multimetric_trajectories()                          │
│  - plot_phase_plane()                                       │
│  - plot_cluster_summary()                                   │
└───────────────────────┬─────────────────────────────────────┘
                        │ uses patterns from
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Level 1: faceted_plotting.py (Generic)                     │
│  - _prepare_facet_grid_data()  → data preparation           │
│  - _plot_faceted_plotly()      → Plotly trace patterns      │
│  - _plot_faceted_matplotlib()  → Matplotlib patterns        │
│  - compute_binned_mean()       → Mean trajectory calc       │
│  - get_color_for_genotype()    → Color assignment           │
└─────────────────────────────────────────────────────────────┘
```

**Specific reuse:**
1. **Plotly trace patterns:** Hover templates, legend groups, subplot management
2. **Matplotlib patterns:** Axis styling, grid layout, tight_layout
3. **Color system:** `get_color_for_genotype()` from `genotype_styling.py`
4. **Data binning:** `compute_binned_mean()` for mean trajectories
5. **Config:** Constants from `plot_config.py` (alpha, linewidth, sizes)

---

### 2.5 Validation & Difference Detection

**Use existing infrastructure:**

```python
from src.analyze.difference_detection import distribution_test
```

**Validation Steps:**
1. Cross-tabulate cluster assignments vs known pair phenotypes
2. Distribution tests: HTA vs WT, CE vs WT, HTA vs CE
3. Find earliest `hpf` where clusters become distinguishable
4. Compare MD-DTW clustering to univariate clustering

---

## Phase 3: Export to `src/` (If Successful)

After prototype validation, refactor to production code:

| Destination | New Content |
|-------------|-------------|
| `src/analyze/trajectory_analysis/multivariate_dtw.py` | `compute_md_dtw_distance_matrix()`, `prepare_multivariate_array()` |
| `src/analyze/trajectory_analysis/multimetric_plotting.py` | `plot_multimetric_trajectories()`, `plot_phase_plane()`, `plot_cluster_summary()` |
| `src/analyze/trajectory_analysis/__init__.py` | Export new functions |
| `src/analyze/trajectory_analysis/config.py` | Add `DEFAULT_FEATURE_LIST` parameter |

---

## Expected Outputs

```
results/mcolon/20251218_MD-DTW-morphseq_analysis/
├── ANALYSIS_PLAN.md              # This file
├── md_dtw_prototype.py           # MD-DTW implementation
├── multimetric_plotting.py       # New visualization functions
├── run_analysis.py               # Main analysis script (optional - can run interactively)
├── output/
│   ├── md_dtw_distance_matrix.npy
│   ├── cluster_labels.csv
│   ├── figures/
│   │   ├── dendrogram.html                    # Interactive dendrogram for K-selection
│   │   ├── dendrogram.png                     # Static dendrogram
│   │   ├── multimetric_colored_by_cluster.html   # PRIMARY: Rows=metrics, Cols=clusters, colored by cluster ID
│   │   ├── multimetric_colored_by_genotype.html  # VALIDATION: Same grid, colored by genotype
│   │   ├── multimetric_colored_by_pair.html      # VALIDATION: Same grid, colored by pair
│   │   ├── cluster_summary_ribbons.html       # Mean ± SD per cluster
│   │   ├── phase_plane.html                   # (Optional) Phase space plot
│   │   └── cluster_validation.png             # Crosstab heatmap
│   └── tables/
│       ├── cluster_vs_pair_crosstab.csv
│       └── cluster_vs_genotype_crosstab.csv
```

---

## Execution Order

| Step | Action | Success Criteria |
|------|--------|------------------|
| 1 | Create `md_dtw_prototype.py` | tslearn works, distance matrix is symmetric and reasonable |
| 2 | Create `multimetric_plotting.py` | **Stacked time-series plots show clear divergence at t~32hpf** |
| 3 | Create `run_analysis.py` | Full pipeline runs end-to-end |
| 4 | Run on b9d2 data | Clusters separate HTA (pair_5) from CE (pairs 6,7,8) |
| 5 | Validate with known labels | Cluster assignments match expected phenotypes |
| 6 | (Optional) Test phase plane plots | Confirm they match stacked plot insights |
| 7 | Export to `src/` | Production-ready, tested code |

---

## Dependencies

**Required packages:**
- `tslearn` - Multivariate DTW computation
- `scipy.cluster.hierarchy` - Hierarchical clustering, dendrogram
- `scikit-learn` - StandardScaler, clustering utilities
- `plotly` - Interactive visualization
- `matplotlib` - Static visualization

**Existing infrastructure:**
- `src/analyze/trajectory_analysis/data_loading.py` - Data loading
- `src/analyze/trajectory_analysis/bootstrap_clustering.py` - Robust clustering
- `src/analyze/trajectory_analysis/cluster_posteriors.py` - Posterior computation
- `src/analyze/trajectory_analysis/faceted_plotting.py` - Base plotting (to extend)

---

## Notes

- The key insight is that HTA and CE share early curvature but diverge in length after ~32hpf
- MD-DTW should capture this coupled signal better than analyzing each metric independently
- **Visualization strategy: Stacked time-series plots ("double-decker") are PRIMARY**
  - Grid layout: Rows = metrics (curvature, length), Columns = clusters
  - Time-aligned x-axes allow visual synchrony (draw vertical line at t=32hpf)
  - See both metrics diverge simultaneously in time
  - No information loss on magnitude, rate, or timing
- **Generate THREE versions with different coloring:**
  1. **Color by cluster** - see cluster identity and consistency
  2. **Color by genotype** - validate clusters separate genotypes
  3. **Color by pair** - validate clusters separate pairs (HTA vs CE)
- **Dendrogram is essential** for choosing K (number of clusters) before plotting
- Phase plane plots are OPTIONAL - lose explicit temporal information
- Priority: Interpretability for biological audiences over abstract phase space representations
