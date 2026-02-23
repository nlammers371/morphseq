# Trajectory Analysis Package Test (v0.2.0)

## Overview

This folder tests the new v0.2.0 DataFrame-centric API of the `src/analyze/trajectory_analysis/` package by replicating the analysis from `results/mcolon/20251106_robust_clustering/cluster_assignment_quality.md`.

**Goal:** Verify that the refactored package functions work correctly end-to-end with improved workflow and time-axis alignment.

## What's Being Tested

### 1. Data Processing (DataFrame-centric)
- `extract_trajectories_df()` - Extract trajectories while keeping time column (hpf)
- `interpolate_to_common_grid_df()` - Interpolate all trajectories to common grid
- `df_to_trajectories()` - One-line conversion to arrays for DTW (eliminates manual grid tracking)

### 2. Clustering & Analysis
- `compute_dtw_distance_matrix()` - DTW distance computation
- `run_bootstrap_hierarchical()` - Bootstrap resampling with label storage
- `analyze_bootstrap_results()` - Posterior probability with Hungarian label alignment
- `classify_membership_2d()` - 2D gating classification (max_p × log_odds_gap)

### 3. Visualization (New Functions)
- `plot_posterior_heatmap()` - Posterior probability heatmap
- `plot_2d_scatter()` - 2D classification scatter
- `plot_cluster_trajectories_df()` - Cluster trajectories (DataFrame version - preserves time alignment)
- `plot_membership_trajectories_df()` - Membership trajectories (DataFrame version)
- `plot_membership_vs_k()` - **NEW** Membership quality trends across k values

## General Data Loading Utility

This test framework uses a **general, reusable data loading utility** implemented in `src/analyze/trajectory_analysis/data_loading.py`:

### How It Works

The `load_experiment_dataframe(experiment_id)` function:
1. Takes just an experiment ID (e.g., `'20251017_combined'`)
2. Searches standard locations for curvature and metadata files
3. Automatically detects file names (handles variations)
4. Merges curvature metrics with metadata on `snip_id` or `embryo_id`
5. Returns complete merged DataFrame

### File Search Strategy

The loader searches for:
- **Curvature files:** `curvature_metrics_{experiment_id}.csv` or `{experiment_id}_curvature.csv`
- **Metadata files:** `df03_final_output_with_latents_{experiment_id}.csv` or `{experiment_id}_metadata.csv`

In standard locations:
- `morphseq_playground/metadata/body_axis/summary/`
- `morphseq_playground/metadata/build06_output/`

### Reusability

This means you can now test ANY experiment by just changing the experiment ID:

```python
from src.analyze.trajectory_analysis.data_loading import load_experiment_dataframe

# Load any experiment
df = load_experiment_dataframe('20251017_combined')
df = load_experiment_dataframe('20250815_exp2')
df = load_experiment_dataframe('your_future_experiment')
```

Or in the test script:
```bash
python test_trajectory_analysis.py -e 20250815_exp2 -g wildtype
python test_trajectory_analysis.py -e 20250815_exp2 -g mutant
```

## Key Improvements Over Original Analysis

### Original Approach (20251106_robust_clustering)
```python
# Old: Manual time grid tracking
trajectories, embryo_ids, orig_lens = extract_trajectories(df, genotype='wildtype')
trajs_interp, _, grid = interpolate_to_common_grid(trajectories)
D = compute_dtw_distance_matrix(trajs_interp, window=5)
# ... bootstrap & posterior analysis ...
fig = plot_cluster_trajectories(trajs_interp, grid, labels)  # BUG: wrong times!
```

### New Approach (v0.2.0 DataFrame-centric)
```python
# New: Time column stays with data
df_filtered = extract_trajectories_df(df, genotype_filter='wildtype')
df_interpolated = interpolate_to_common_grid_df(df_filtered)
trajectories, embryo_ids, grid = df_to_trajectories(df_interpolated)
D = compute_dtw_distance_matrix(trajectories, window=5)
# ... bootstrap & posterior analysis ...
fig = plot_cluster_trajectories_df(df_interpolated, labels)  # CORRECT!
```

**Benefits:**
- No time-axis alignment bugs (DataFrame handles it)
- Simpler code (fewer helper functions)
- Time column (hpf) always explicit and preserved
- Can visualize trajectories directly from DataFrame

## File Structure

```
20251109_testing_trajectory_analysis_package/
├── README.md                          # This file
├── test_trajectory_analysis.py        # Main test script
└── output/
    ├── data/                          # Pickled analysis results
    │   └── {genotype}_results.pkl
    └── figures/                       # Generated plots
        └── {genotype}/
            ├── posterior_heatmaps/    # p(cluster | embryo) matrices
            ├── posterior_scatters/    # max_p vs log_odds_gap plots
            ├── trajectories/          # Trajectory plots by cluster/membership
            └── membership_trends/     # Membership percentages vs k
```

## Usage

### Basic run (default: 20251017_combined, cep290_homozygous)
```bash
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251109_testing_trajectory_analysis_package
python test_trajectory_analysis.py
```

### Specify genotype for default experiment
```bash
python test_trajectory_analysis.py --genotype cep290_wildtype
```

### Specify different experiment and genotype
```bash
python test_trajectory_analysis.py --experiment 20251017_combined --genotype cep290_heterozygous
```

### Adjust k range
```bash
python test_trajectory_analysis.py --genotype cep290_homozygous --k_min 2 --k_max 5
```

### Short flags
```bash
python test_trajectory_analysis.py -e 20251017_combined -g cep290_wildtype
```

### Skip plot generation (faster, just save data)
```bash
python test_trajectory_analysis.py --skip_plots
```

### Run with debug output
```bash
python test_trajectory_analysis.py -e 20251017_combined -g cep290_homozygous 2>&1 | tee analysis.log
```

## Expected Output

### Analysis Summary
```
======================================================================
Analysis for cep290_homozygous
======================================================================

1. Loading data...
  Filtered to 1234 measurements for cep290_homozygous
  Embryos: 47
  Time range: 2.5 - 8.5 hpf

2. Extracting trajectories...
  Extracted 47 embryos

3. Interpolating to common time grid...
  Grid points: 15

4. Converting to arrays for DTW computation...
  Trajectories: 47
  Grid: 15 points

5. Computing DTW distance matrix...
  Distance matrix shape: (47, 47)

6.2. Bootstrap clustering for k=2...
  Results: Core=38/47 (80.9%) Uncertain=7/47 (14.9%) Outlier=2/47 (4.3%)

6.3. Bootstrap clustering for k=3...
  Results: Core=35/47 (74.5%) Uncertain=9/47 (19.1%) Outlier=3/47 (6.4%)

[... k=4 through k=7 ...]

✓ Saved results to output/data/cep290_homozygous_results.pkl
```

### Generated Plots (per genotype)
- **`membership_vs_k.png`** - Core/uncertain/outlier percentages across k (new plot!)
- **`posterior_heatmaps/heatmap_k{k}.png`** - Assignment posterior probabilities
- **`posterior_scatters/scatter_k{k}.png`** - 2D classification gating
- **`trajectories/clusters_k{k}.png`** - Trajectories colored by cluster (with time axis fixed!)
- **`trajectories/membership_k{k}.png`** - Trajectories colored by membership quality

## Comparison with Original Analysis

### Membership Percentages Should Match
If the API refactor was successful, the membership percentages across k values should be **identical or very similar** to those from `20251106_robust_clustering/output/data/comparison_summary.csv`.

Example:
```
Original (20251106):
k=2: Core=82%, Uncertain=15%, Outlier=3%
k=3: Core=75%, Uncertain=19%, Outlier=6%
k=4: Core=68%, Uncertain=23%, Outlier=9%

New (20251109 test):
k=2: Core=82%, Uncertain=15%, Outlier=3%  ✓
k=3: Core=75%, Uncertain=19%, Outlier=6%  ✓
k=4: Core=68%, Uncertain=23%, Outlier=9%  ✓
```

If they differ significantly, there may be a bug in the refactored code.

### Plot Quality Improvements
The new plots should have:
- **Better time alignment** in trajectory plots (late-starting embryos now correct)
- **Cleaner trajectory plots** (DataFrame handles time mapping automatically)
- **New membership_vs_k plot** showing quality trends (helpful for choosing optimal k)

## Dependencies

All dependencies are in the main `trajectory_analysis` package:
- `numpy`, `scipy`, `pandas` - Core data
- `sklearn` - Clustering algorithms
- `matplotlib`, `seaborn` - Plotting
- `scipy.optimize.linear_sum_assignment` - Hungarian algorithm for label alignment

## Testing Checklist

- [ ] Script runs without errors
- [ ] All k values (2-7) complete successfully
- [ ] Output pickle files created
- [ ] All plots generated (membership_vs_k + heatmaps + scatters + trajectories)
- [ ] Membership percentages match original analysis
- [ ] Trajectory plots show correct time alignment (no late-start embryo shifts)
- [ ] Classification distributions reasonable (mostly core members for k=2-3)
- [ ] Can load and inspect results from pickle files

## Debugging

### If data loading fails
Check `test_trajectory_analysis.py` lines 104-114 for the data path logic. The script tries:
1. `morphseq_playground/metadata/body_axis/summary/curvature_metrics_20251017_combined.csv`
2. `results/data/20251017_combined.csv`
3. `data/20251017_combined.csv`

If none exist, you may need to point to the correct location manually.

### If plots fail to generate
Common issues:
- Missing matplotlib/seaborn (check imports)
- Output directory not writable
- Incompatible array shapes (check that all k values produce results)

### If membership percentages don't match
- Check that `N_BOOTSTRAP`, `BOOTSTRAP_FRAC`, `RANDOM_SEED` match original (100, 0.8, 42)
- Verify thresholds are correct: max_p=0.8, log_odds=0.7, outlier_max_p=0.5
- Check that label alignment algorithm is working (Hungarian via scipy)

## References

- **Original implementation:** `results/mcolon/20251106_robust_clustering/cluster_assignment_quality.md`
- **Package documentation:** `src/analyze/trajectory_analysis/README.md`
- **Package source:** `src/analyze/trajectory_analysis/`
- **New plot function:** `src/analyze/trajectory_analysis/plotting.py:plot_membership_vs_k()`

## Contact

For issues or questions:
1. Check trajectory_analysis README for API details
2. Review cluster_assignment_quality.md for methodology
3. Inspect pickle files to debug intermediate results

---

**Test Date:** 2025-11-09
**Trajectory Analysis Version:** 0.2.0 (DataFrame-centric)
**Test Framework:** Replication of 20251106 robust clustering analysis
