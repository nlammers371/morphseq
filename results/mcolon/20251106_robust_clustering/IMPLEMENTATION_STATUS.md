# Implementation Status: Hierarchical Clustering with Posterior Probabilities

**Date:** 2025-11-07
**Status:** Partial Implementation

---

## ✅ COMPLETED

### Phase 1: K-Medoids Posterior Analysis (FULLY WORKING)

**Files Created:**
1. `bootstrap_posteriors.py` - Label alignment + posterior computation
2. `adaptive_classification.py` - 2D gating classifier
3. `compare_methods_v2.py` - Comparison pipeline
4. `plot_quality_comparison.py` - Visualization suite
5. `plot_posterior_heatmaps.py` - Heatmap plots
6. `plot_cluster_trajectories.py` - Trajectory integration
7. `README.md` - Usage guide
8. Updated `cluster_assignment_quality.md` - Full documentation

**Validation Results (K-Medoids on Small Dataset):**
```
k=2: 62.9% core (+450% improvement over co-association)
k=3: 80.0% core (+86.7% improvement)
k=4: 51.4% core (+800% improvement)
k=5: 48.6% core (+1600% improvement)
```

**Key Achievement:** Successfully implemented and validated bootstrap assignment posteriors with 2D gating for k-medoids clustering.

---

## 🔄 IN PROGRESS

### Phase 2: Hierarchical Clustering Adaptation

**What's Done:**
1. ✅ Copied hierarchical clustering script → `run_hierarchical_posterior_clustering.py`
2. ✅ Modified `run_bootstrap_hierarchical()` function to store labels (NOT co-association)
3. ✅ Updated docstrings and headers

**What's Not Done:**
1. ❌ Replace consensus clustering step with posterior analysis integration
2. ❌ Remove old co-association matrix plots
3. ❌ Add posterior analysis function calls
4. ❌ Update plotting functions for posterior-weighted visualizations
5. ❌ Add import statements for `bootstrap_posteriors` and `adaptive_classification`

---

## 📋 TODO: Complete Hierarchical Implementation

### Step 1: Integration with Posterior Analysis (~30 min)

**File:** `run_hierarchical_posterior_clustering.py` line ~428

**Current code (to replace):**
```python
# Bootstrap to get co-occurrence matrix
boot_result = run_bootstrap_hierarchical(D, k, ...)
C = boot_result['coassoc']

# Consensus clustering
labels, _ = cluster_hierarchical_cooccurrence(C, k)

# Membership analysis
membership = analyze_membership(D, labels, C, core_thresh=CORE_THRESHOLD)
```

**New code (to add):**
```python
from bootstrap_posteriors import analyze_bootstrap_results
from adaptive_classification import classify_embryos_2d, get_classification_summary

# Bootstrap to get labels
boot_result = run_bootstrap_hierarchical(D, k, ...)
labels = boot_result['reference_labels']  # Direct clustering on full data

# Posterior analysis
posterior_analysis = analyze_bootstrap_results(boot_result)

# 2D gating classification
classification = classify_embryos_2d(
    max_p=posterior_analysis['max_p'],
    log_odds_gap=posterior_analysis['log_odds_gap'],
    modal_cluster=posterior_analysis['modal_cluster'],
    threshold_max_p=0.8,
    threshold_log_odds=0.7
)

summary = get_classification_summary(classification)
```

**Result structure (to update):**
```python
all_results[k] = {
    'labels': labels,
    'posterior_analysis': posterior_analysis,
    'classification': classification,
    'summary': summary,
    'silhouette': sil,
    'n_core': summary['n_core'],
    'n_uncertain': summary['n_uncertain'],
    'n_outlier': summary['n_outlier']
}
```

### Step 2: Update Plotting Functions (~1 hour)

**Files to modify:**
- `run_hierarchical_posterior_clustering.py` (plotting section starting line ~494)

**Changes needed:**

1. **Remove co-association matrix plots** (or make optional)
2. **Add posterior heatmap plots:**
   ```python
   from plot_posterior_heatmaps import plot_posterior_heatmap

   plot_posterior_heatmap(
       k=k,
       data_dir='output/data',
       output_dir=f'output/figures/hierarchical/{genotype}/posterior_heatmaps'
   )
   ```

3. **Add 2D scatter plots:**
   ```python
   from plot_quality_comparison import plot_2d_scatter

   plot_2d_scatter(
       k=k,
       all_results=all_results,
       output_dir=f'output/figures/hierarchical/{genotype}/posterior_scatters'
   )
   ```

4. **Replace temporal_trends plots with TWO versions:**

   **Version A: Continuous alpha (posterior-weighted)**
   ```python
   # Prepare dataframe with posterior probabilities
   df_with_posteriors = prepare_trajectory_dataframe(
       df_long, posterior_analysis, classification
   )

   # Plot with continuous alpha
   from src.analyze.utils.plotting import plot_embryos_metric_over_time

   fig, ax = plot_embryos_metric_over_time(
       df=df_with_posteriors,
       embryo_id_col='embryo_id',
       time_col='predicted_stage_hpf',
       metric_col='normalized_baseline_deviation',
       color_by='cluster',
       alpha_col='posterior_prob',  # KEY: alpha varies by posterior
       show_individual=True,
       show_mean=True,
       show_sd_band=True,
       title=f'{genotype} - k={k} (Posterior-Weighted)'
   )

   save_path = output_dir / f'temporal_trends_posterior/temporal_trends_k{k}.png'
   fig.savefig(save_path, dpi=300)
   ```

   **Version B: Category colors (core/uncertain/outlier)**
   ```python
   # Same dataframe, different color mapping
   fig, ax = plot_embryos_metric_over_time(
       df=df_with_posteriors,
       embryo_id_col='embryo_id',
       time_col='predicted_stage_hpf',
       metric_col='normalized_baseline_deviation',
       color_by='category',  # KEY: color by core/uncertain/outlier
       show_individual=True,
       show_mean=True,
       show_sd_band=True,
       title=f'{genotype} - k={k} (Assignment Quality)'
   )

   save_path = output_dir / f'temporal_trends_category/temporal_trends_k{k}.png'
   fig.savefig(save_path, dpi=300)
   ```

### Step 3: Add Helper Function for Data Preparation (~20 min)

**Add this function to `run_hierarchical_posterior_clustering.py`:**

```python
def prepare_trajectory_dataframe(df_long, posterior_analysis, classification, embryo_ids_ordered):
    """
    Merge trajectory data with posterior probabilities and classifications.

    Returns dataframe with added columns:
    - 'cluster': Cluster assignment
    - 'category': core/uncertain/outlier
    - 'posterior_prob': Posterior probability of assigned cluster
    - 'max_p': Maximum posterior probability
    - 'entropy': Assignment entropy
    """
    df = df_long.copy()

    # Create mapping from embryo_id to index
    embryo_id_to_idx = {eid: idx for idx, eid in enumerate(embryo_ids_ordered)}

    # Map to index
    df['embryo_idx'] = df['embryo_id'].map(embryo_id_to_idx)

    # Add cluster assignments
    df['cluster'] = df['embryo_idx'].map(
        lambda idx: posterior_analysis['modal_cluster'][idx]
    )

    # Add classification category
    df['category'] = df['embryo_idx'].map(
        lambda idx: classification['category'][idx]
    )

    # Add posterior probability of assigned cluster
    def get_posterior_prob(row):
        idx = row['embryo_idx']
        cluster = row['cluster']
        return posterior_analysis['p_matrix'][idx, cluster]

    df['posterior_prob'] = df.apply(get_posterior_prob, axis=1)

    # Add quality metrics
    df['max_p'] = df['embryo_idx'].map(
        lambda idx: posterior_analysis['max_p'][idx]
    )
    df['entropy'] = df['embryo_idx'].map(
        lambda idx: posterior_analysis['entropy'][idx]
    )

    # Scale posterior_prob to alpha range (0.2 - 1.0)
    df['posterior_alpha'] = 0.2 + 0.8 * df['posterior_prob']

    return df
```

### Step 4: Create Genotype Comparison Plots (~30 min)

**Add new function:**

```python
def plot_genotype_overlay(experiment_id, k, genotypes, verbose=True):
    """
    Create overlay plot of all genotypes for a given k.
    Color by genotype, alpha by posterior probability.
    """
    from src.analyze.utils.plotting import plot_embryos_metric_over_time

    # Load and combine data from all genotypes
    df_all_genotypes = []

    for genotype in genotypes:
        results_file = output_dir / f'data/posteriors_{genotype}_k{k}.pkl'
        if not results_file.exists():
            continue

        with open(results_file, 'rb') as f:
            results = pickle.load(f)

        # Load trajectory data
        df_traj = load_trajectories_for_genotype(experiment_id, genotype)

        # Merge with posterior data
        df_merged = prepare_trajectory_dataframe(
            df_traj,
            results['posterior_analysis'],
            results['classification'],
            results['embryo_ids']
        )

        # Add genotype column
        df_merged['genotype'] = genotype

        df_all_genotypes.append(df_merged)

    # Combine all genotypes
    df_combined = pd.concat(df_all_genotypes, ignore_index=True)

    # Plot with genotype coloring, posterior alpha
    fig, ax = plot_embryos_metric_over_time(
        df=df_combined,
        embryo_id_col='embryo_id',
        time_col='predicted_stage_hpf',
        metric_col='normalized_baseline_deviation',
        color_by='genotype',
        alpha_col='posterior_alpha',
        show_individual=True,
        show_mean=True,
        show_sd_band=True,
        facet_by='cluster',  # Separate subplot per cluster
        title=f'Genotype Comparison - k={k} ({experiment_id})',
        figsize=(14, 5*k)
    )

    save_path = output_dir / f'figures/comparison/genotype_overlay_k{k}.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if verbose:
        print(f"  Saved {save_path}")
```

---

## 📊 Next Steps (Priority Order)

1. **Complete hierarchical integration** (Steps 1-3 above) - ~1.5 hours
2. **Test on single genotype/k combination** - ~15 min
3. **Debug and fix any issues** - ~30 min
4. **Run full pipeline on 20251017_combined dataset** - ~1 hour (includes compute time)
5. **Generate genotype comparison plots** (Step 4) - ~30 min
6. **Compare hierarchical vs k-medoids results** - ~1 hour
7. **Document findings in comparison report** - ~30 min

**Total estimated time to completion: ~5-6 hours**

---

## 🎯 Expected Final Outputs

```
output/20251017_combined/
├── data/
│   ├── hierarchical/
│   │   ├── cep290_wildtype_k{2-7}.pkl
│   │   ├── cep290_heterozygous_k{2-7}.pkl
│   │   ├── cep290_homozygous_k{2-7}.pkl
│   │   └── cep290_unknown_k{2-7}.pkl
│   └── kmedoids/
│       └── [existing k=2-5 results]
│
├── figures/
│   ├── hierarchical/
│   │   ├── [genotype]/
│   │   │   ├── posterior_heatmaps/posterior_heatmap_k{2-7}.png
│   │   │   ├── posterior_scatters/posterior_scatter_k{2-7}.png
│   │   │   ├── temporal_trends_posterior/temporal_trends_k{2-7}.png
│   │   │   └── temporal_trends_category/temporal_trends_k{2-7}.png
│   │
│   └── comparison/
│       ├── genotype_overlay_k{3,4,5}.png
│       ├── core_fraction_hierarchical_vs_kmedoids.png
│       ├── entropy_hierarchical_vs_kmedoids.png
│       └── outlier_analysis_k6-8.png
│
└── comparison_report.md
```

---

## 🚀 How to Resume

**Option A: Manual completion (recommended if modifying code)**
1. Open `run_hierarchical_posterior_clustering.py`
2. Follow Steps 1-4 above
3. Test incrementally

**Option B: Let Claude complete it**
1. Share this status document
2. Request: "Complete Steps 1-4 in IMPLEMENTATION_STATUS.md"
3. Claude will make the remaining edits

**Option C: Simplified version (faster testing)**
1. Skip hierarchical for now
2. Focus on generating better plots for existing k-medoids results
3. Add genotype comparison plots using existing data
