# TMEM67 Cluster Projection onto CEP290 Reference Clusters

## Goal
Project TMEM67 embryos onto CEP290 cluster space and compare cluster trajectory frequencies. Primary comparison: CEP290 spawn vs TMEM67 spawn. Secondary: analyze TMEM67 pair_1 data as well. Hypothesis: TMEM67 has higher percentage of High_to_Low trajectories than CEP290.

## Approach
**Use existing CEP290 cluster assignments as reference** (NO re-clustering):
1. **Nearest Neighbor projection** (primary): Assign TMEM67 embryos to cluster of closest CEP290 embryo in DTW space
2. **K-NN with posteriors** (secondary): Compute probability distribution over clusters for comparison

The CEP290 clusters (0-5 mapping to Low_to_High, High_to_Low, Intermediate, Not Penetrant) are already finalized. We load the pre-computed labels and use DTW distances to assign TMEM67 embryos to these existing clusters.

---

## Implementation Steps

### Step 1: Create Project Structure
**Location**: `results/mcolon/20260104_tmem67_cluster_projection_to_cep290/`

Create:
- `run_projection.py` - Main analysis script (easy to run/iterate)
- `projection_utils.py` - Helper functions (cross-DTW, assignment methods)
- `output/` - For saved results

### Step 2: Load CEP290 Reference Data (PRE-COMPUTED CLUSTERS)
**Source**: `results/mcolon/20251229_cep290_phenotype_extraction/final_data/`

**Important**: We are NOT re-clustering. We load the already-finalized cluster assignments and use them as reference.

```python
CEP290_DIR = Path("results/mcolon/20251229_cep290_phenotype_extraction/final_data")

# Load CEP290 trajectory data (for DTW distance computation)
df_cep290 = pd.read_csv(CEP290_DIR / "embryo_data_with_labels.csv")

# Load PRE-COMPUTED cluster labels (these are fixed - do not modify)
df_cep290_labels = pd.read_csv(CEP290_DIR / "embryo_cluster_labels.csv")

# Filter for spawn only (both data and labels)
df_cep290_spawn = df_cep290[df_cep290['pair'] == 'cep290_spawn']
labels_spawn = df_cep290_labels[
    df_cep290_labels['embryo_id'].isin(df_cep290_spawn['embryo_id'].unique())
].drop_duplicates(subset='embryo_id')

# Build embryo_id -> cluster mapping from PRE-COMPUTED labels
cep290_cluster_map = dict(zip(labels_spawn['embryo_id'], labels_spawn['clusters']))
cep290_category_map = dict(zip(labels_spawn['embryo_id'], labels_spawn['cluster_categories']))

print(f"CEP290 spawn reference: {len(cep290_cluster_map)} embryos with cluster labels")
print(f"Cluster categories: {labels_spawn['cluster_categories'].unique()}")
```

Key columns: `embryo_id`, `pair`, `clusters` (0-5), `cluster_categories` (Low_to_High, High_to_Low, Intermediate, Not Penetrant)

### Step 3: Load TMEM67 Data
**Experiments**: 20250711, 20251205

```python
from src.analyze.trajectory_analysis.data_loading import load_experiment_dataframe

# Load TMEM67 experiments
experiment_ids = ['20250711', '20251205']
dfs = []
for exp_id in experiment_ids:
    df_exp = load_experiment_dataframe(exp_id, format_version='df03')
    df_exp['experiment_id'] = exp_id
    dfs.append(df_exp)
    print(f"Loaded {exp_id}: {len(df_exp)} rows, {df_exp['embryo_id'].nunique()} embryos")

df_tmem67 = pd.concat(dfs, ignore_index=True)

# Filter to TMEM67 genotypes only
df_tmem67 = df_tmem67[df_tmem67["genotype"].str.contains("tmem67", case=False, na=False)]

# Mark spawn (missing pair = spawn)
df_tmem67.loc[df_tmem67["pair"].isna(), "pair"] = "tmem67_spawn"

# Keep ALL pairs (spawn AND pair_1, etc.) for full analysis
print(f"TMEM67 total: {df_tmem67['embryo_id'].nunique()} embryos")
print(f"TMEM67 pairs: {df_tmem67['pair'].value_counts()}")
```

**Note**: We keep all TMEM67 data (spawn + pair_1, etc.) and can filter by pair later for specific comparisons.

### Step 4: Prepare Multivariate Arrays with Shared Time Grid

```python
METRICS = ['baseline_deviation_normalized', 'total_length_um']

# CEP290 reference (get time_grid from here)
X_cep290, cep290_ids, time_grid = prepare_multivariate_array(
    df_cep290_spawn, metrics=METRICS, normalize=True
)

# TMEM67 target (use SAME time_grid)
X_tmem67, tmem67_ids, _ = prepare_multivariate_array(
    df_tmem67_spawn, metrics=METRICS, time_grid=time_grid, normalize=True
)
```

### Step 5: Implement Cross-Dataset DTW Distance

**New function** (will export to `cluster_projection.py` later):

```python
def compute_cross_dtw_distance_matrix(
    X_source: np.ndarray,  # (N_source, T, M) - TMEM67
    X_target: np.ndarray,  # (N_target, T, M) - CEP290
    sakoe_chiba_radius: int = 3,
    n_jobs: int = -1
) -> np.ndarray:
    """Returns D_cross of shape (N_source, N_target)."""
    # Uses _dtw_multivariate_pair from dtw_distance.py
    # Parallelized with joblib
```

### Step 6: Implement Projection Methods

```python
def assign_clusters_nearest_neighbor(
    D_cross, source_ids, target_ids, target_cluster_map
) -> pd.DataFrame:
    """Assign via nearest neighbor."""
    # For each source: find argmin(D_cross[i,:]), get cluster of nearest target

def assign_clusters_knn_posterior(
    D_cross, source_ids, target_ids, target_cluster_map, k=5
) -> pd.DataFrame:
    """Assign via K-NN voting with posterior probabilities."""
    # Returns p(cluster) for each source embryo
```

### Step 7: Compare Cluster Frequencies

```python
# Compute frequencies
cep290_freqs = labels_spawn['cluster_categories'].value_counts(normalize=True)
tmem67_freqs = df_nn['cluster_category'].value_counts(normalize=True)

# Statistical tests
# - Chi-square test for overall distribution
# - Fisher's exact for High_to_Low vs Other (2x2)
```

### Step 8: Visualizations

1. **Stacked bar chart**: Distribution comparison
2. **Side-by-side bar chart**: Per-category comparison
3. **Trajectory overlays**: TMEM67 on CEP290 clusters (validation)
4. **Distance histogram**: Nearest-neighbor distances (quality check)

### Step 9: Export to Module (after validation)

Move functions to `src/analyze/trajectory_analysis/cluster_projection.py`:
- `compute_cross_dtw_distance_matrix()`
- `assign_clusters_nearest_neighbor()`
- `assign_clusters_knn_posterior()`
- `project_to_reference_clusters()` (high-level wrapper)

---

## Critical Files

| File | Purpose |
|------|---------|
| `results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv` | CEP290 reference data |
| `results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_cluster_labels.csv` | CEP290 cluster assignments |
| `src/analyze/trajectory_analysis/dtw_distance.py` | `_dtw_multivariate_pair()`, `prepare_multivariate_array()` |
| `src/analyze/trajectory_analysis/cluster_extraction.py` | Pattern for cluster utilities |

---

## Data Flow

```
CEP290 spawn data                    TMEM67 spawn data
       |                                    |
       v                                    v
prepare_multivariate_array()    prepare_multivariate_array(time_grid=...)
       |                                    |
       v                                    v
X_cep290 (N_cep, T, M)             X_tmem67 (N_tmem, T, M)
       |                                    |
       +-----------> compute_cross_dtw <----+
                            |
                            v
                    D_cross (N_tmem, N_cep)
                            |
              +-------------+-------------+
              |                           |
              v                           v
    assign_nearest_neighbor()    assign_knn_posterior()
              |                           |
              v                           v
    cluster assignments           posterior probabilities
              |                           |
              +-------------+-------------+
                            |
                            v
                  compare_frequencies()
                            |
                            v
              statistical tests + visualizations
```

---

## Separate Task: trajectory_analysis Reorganization

Create instruction file at `src/analyze/trajectory_analysis/REORGANIZATION.md` with recommendations:

1. **Quality Control Consolidation**: Merge `outliers.py`, `distance_filtering.py` into `quality_control.py`
2. **Plotting Restructure**: Create `plotting/` subpackage with `core.py`, `cluster_plots.py`, `pair_plots.py`
3. **Utilities Subfolder**: Move `pca_embedding.py`, `phenotype_io.py`, `correlation_analysis.py` to `utilities/`
4. **Documentation**: Add `WORKFLOW.md` showing typical analysis pipeline

(This is a separate task from the projection work)

---

## Lessons Learned: Good to Know for Similar Analysis

### Critical Design Decisions

1. **Shared Time Grid is Essential**
   - When computing cross-dataset DTW, both datasets MUST use the same time grid
   - Use `prepare_multivariate_array(time_grid=...)` to force alignment
   - Mismatch in time grids = invalid distance comparisons

2. **Pre-computed Clusters vs Re-clustering**
   - Make it crystal clear upfront whether you're using existing clusters or creating new ones
   - This analysis uses CEP290 clusters as REFERENCE (no re-clustering)
   - Document this prominently to avoid confusion

3. **Spawn vs Pair Filtering**
   - Clarify early which experimental conditions to include
   - For cross-genotype comparisons, consider whether to match pairing conditions
   - Document filtering decisions in the analysis plan

4. **Distance Matrix Orientation**
   - Cross-dataset: D_cross[i, j] = distance from source[i] to target[j]
   - Shape: (N_source, N_target) not (N_target, N_source)
   - Document this clearly in function docstrings

### Technical Gotchas

1. **DTW Normalization**
   - `prepare_multivariate_array()` does global Z-score normalization by default
   - This is GOOD for cross-dataset comparison (ensures fair metric weighting)
   - But be aware: metrics normalized across ALL data before DTW

2. **Cluster Label Mapping**
   - CEP290 has numeric clusters (0-5) AND semantic categories (High_to_Low, etc.)
   - Always maintain both mappings: `embryo_id -> cluster_num` and `cluster_num -> category`
   - Use categories for interpretation, numbers for computation

3. **Statistical Testing Caveats**
   - Chi-square requires sufficient sample size (n ≥ 5 per cell)
   - Fisher's exact is better for small samples but only works for 2x2
   - Consider multiple testing correction if doing many comparisons

4. **Parallelization**
   - Cross-DTW is O(N_source × N_target), can be slow
   - Use `n_jobs=-1` for full parallelization
   - Monitor memory usage with large datasets

### Workflow Recommendations

1. **Start with Small Subsets**
   - Test the full pipeline on 10-20 embryos first
   - Validate distance computation, assignment logic, visualizations
   - Then scale up to full dataset

2. **Sanity Checks**
   - Plot distance distribution: should be reasonable range (not all zeros or all inf)
   - Check nearest neighbor distances: outliers indicate problems
   - Validate cluster assignments make biological sense

3. **Incremental Implementation**
   - Get nearest neighbor working first (simplest)
   - Then add K-NN posteriors (more complex)
   - Don't implement everything at once

4. **Save Intermediate Results**
   - Save D_cross matrix (can reuse for different assignment methods)
   - Save cluster assignments (for reproducibility)
   - Document random seeds if using any stochastic methods

### What to Document for Future You

1. **Data Provenance**
   - Which experiments were included
   - Which filtering was applied (genotype, pair, time window)
   - Any embryos excluded and why

2. **Parameter Choices**
   - DTW window size (sakoe_chiba_radius)
   - Metrics used (baseline_deviation_normalized, total_length_um)
   - K value for K-NN (if used)

3. **Validation Results**
   - How many embryos were projected
   - Distance distribution statistics
   - Any quality flags (e.g., very high distances indicating poor matches)

4. **Biological Interpretation**
   - What do the cluster frequency differences mean?
   - Are results consistent with hypothesis?
   - What follow-up experiments are suggested?
