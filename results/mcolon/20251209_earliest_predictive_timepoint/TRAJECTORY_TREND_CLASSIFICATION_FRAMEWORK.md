# Trajectory Trend Classification Framework

## Overview

A reusable framework to test if **binned embeddings can predict trajectory fate** (increasing vs decreasing curvature) using **binary classification** instead of continuous regression.

## Rationale

### Problem
- **Continuous curvature regression is underpowered** with small embryo samples
- Predicting exact curvature values requires high signal-to-noise ratio
- Many predictions fail (negative R²) even when biological pattern exists

### Solution: Binary Classification
- **Rephrase the question**: "Which fate group does this embryo belong to?" (not "what exact curvature?")
- **Binary classification is more robust** with limited samples
- **F1-score is a better metric** for this biological question than R²
- Early timepoints may distinguish fate even if exact values are noisy

## Validated Framework

### Step 1: DTW Clustering to Define Trajectory Groups

**Objective**: Partition embryos into trajectory fate groups based on curvature shape

**Method**:
1. Load experiment data (curvature + embeddings)
2. Extract curvature trajectories for each embryo
3. Interpolate to common time grid (step=0.5 hpf)
4. Compute pairwise DTW distances (Sakoe-Chiba window=3)
5. Run bootstrap hierarchical clustering (k=3, n_bootstrap=100)
6. Extract modal cluster assignment for each embryo

**Cluster Labeling** (fit linear trend to mean trajectory):
- **Increasing** (slope > 0): curvature gets worse over time → more severe phenotype
- **Decreasing** (slope < 0): curvature rescues over time → recovery/compensation
- **Wildtype-like** (mean < 0.05): exclude from prediction task

**Genotype Verification**:
- Create cluster-genotype cross-tabulation
- Visualize trajectories colored by genotype
- Confirm biological plausibility of cluster assignments

### Step 2: Bin Embeddings by Time

**Objective**: Prepare features for time-specific classification

**Method**:
1. Create time bins: `floor(time / bin_width) * bin_width` (default bin_width=2.0 hours)
2. Group by (embryo_id, time_bin)
3. Average all embedding dimensions: z_mu_n_*, z_mu_b_*
4. Result: one row per (embryo_id, time_bin) pair

**Output**:
- Columns: embryo_id, time_bin, z_mu_n_00_binned, z_mu_n_01_binned, ..., genotype
- Data structure ready for time-bin-specific classification

### Step 3: Classification at Each Time Bin

**Objective**: Determine earliest timepoint where fate can be predicted from embeddings

**For each time bin** (e.g., 12, 14, 16, ..., 72 hpf):

1. **Prepare data**:
   - X = binned embeddings at this time bin (all embryos with data)
   - y = binary trajectory class (0=decreasing, 1=increasing)
   - Exclude wildtype-like embryos (mean curvature < 0.05)
   - Skip if either class has <2 samples

2. **Cross-validation**: GroupKFold by embryo_id (5 folds)
   - Prevents embryo leakage between train/test
   - Ensures true generalization across individuals

3. **Classification pipeline**:
   ```
   Imputation (median) → StandardScaler → LogisticRegression(L2, max_iter=1000)
   ```

4. **Metrics per fold**:
   - F1-score (primary metric)
   - Precision
   - Recall

5. **Aggregate across folds**:
   - F1_mean, F1_std
   - Precision_mean, Recall_mean
   - Sample counts (n_increasing, n_decreasing)

### Step 4: Output & Visualization

**CSV Results** (`classification_results.csv`):
```
time_bin, f1_mean, f1_std, precision_mean, recall_mean, n_samples, n_increasing, n_decreasing
12, 0.523, 0.087, 0.612, 0.453, 25, 12, 13
14, 0.587, 0.095, 0.651, 0.521, 28, 15, 13
...
```

**Plot**: F1-score vs time bin
- X-axis: Time bin (hpf)
- Y-axis: F1-score (0-1)
- Error bars: ±1 std across CV folds
- Horizontal lines: chance level (0.5), target threshold (0.7)
- Annotation: earliest timepoint where F1 ≥ 0.7

**Interpretation**:
- **F1 > 0.7**: Strong predictive signal
- **F1 > 0.5**: Better than chance
- **F1 < 0.5**: Fate not yet distinguishable at this timepoint
- **Earliest high-F1 timepoint**: when phenotypic trajectory becomes predictable from embeddings

## cep290 (20250512) - Validation Results

### Clustering Outcomes
- **Cluster 0** (increasing, n=29): Mixed mutants (het=9, hom=8, unknown=10, wt=2)
  - Mean curvature: 0.121 | Slope: +0.000158
- **Cluster 1** (decreasing, n=48): Rescue group (het=17, wt=19, unknown=8, hom=4)
  - Mean curvature: 0.051 | Slope: -0.001438
- **Cluster 2** (decreasing, n=1): Single homozygous outlier

### Key Finding
**cep290 lacks clean bimodal mutant pattern**:
- NOT: "increasing mutants vs decreasing mutants"
- INSTEAD: "rescue trajectory (WT+het) vs progression trajectory (mixed mutants)"
- **10 "unknown" embryos** in increasing cluster complicate interpretation
- **Conclusion**: Framework validates but cep290 not ideal for trend classification

### Biological Interpretation
The clustering reveals that **WT + het embryos cluster together (decreasing)** while **penetrant homozygous mutants cluster separately (increasing)**. This makes sense:
- WT/het: relatively normal development, possibly slight compensation
- Homozygous: progressive curvature that worsens over time

## Framework Application to b9d2

### Prerequisites for b9d2 (or any new experiment)
1. Have curvature metrics (baseline_deviation_normalized)
2. Have VAE embeddings (z_mu_n_* or z_mu_b_* columns)
3. Have embryo metadata (embryo_id, time measurements)
4. **CRITICAL**: Have reliable genotype labels (cep290 has 10 "unknown")

### Expected Differences from cep290
- **b9d2 likely has bimodal mutant phenotype**: both increasing AND decreasing mutants
- Different penetrance patterns may reveal different trajectory groups
- Cleaner genotype labels expected

### Execution Steps for b9d2
1. Modify script: change `EXPERIMENT_ID = '20250512'` → `EXPERIMENT_ID = 'b9d2'`
2. Run Phase A: DTW clustering with genotype visualization
   - Verify 2+ distinct mutant trends exist
   - Confirm genotype segregation in clusters
3. If bimodal pattern confirmed: Run Phase B classification
4. Compare F1-score curves: cep290 vs b9d2
5. Identify earliest predictive timepoint for b9d2

## Script Location & Usage

**File**: `results/mcolon/20251209_earliest_predictive_timepoint/trajectory_trend_classifier.py`

**Commands**:
```bash
# Phase A only: clustering and visualization
python trajectory_trend_classifier.py --phase clustering

# Phase B only: classification (requires Phase A completed)
python trajectory_trend_classifier.py --phase classification

# Both phases (recommended first run)
python trajectory_trend_classifier.py
```

**Output directory**: `results/mcolon/20251209_earliest_predictive_timepoint/output/`

## Key Parameters (Tunable)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `K_CLUSTERS` | 3 | Start with 3; adjust if different number of trajectory groups |
| `BIN_WIDTH` | 2.0 hours | Balance between temporal resolution and sample size per bin |
| `N_FOLDS` | 5 | Standard CV; reduce if <10 unique embryos per class |
| `WILDTYPE_THRESHOLD` | 0.05 | Embryos with mean curvature < this are excluded as "WT-like" |
| `DTW_WINDOW` | 3 | Sakoe-Chiba window for DTW; increase if trajectories very different |
| `N_BOOTSTRAP` | 100 | Clustering robustness; increase to 500 for publication |

## References & Dependencies

**Data loading**: `src/analyze/trajectory_analysis/data_loading.py`
- `load_experiment_dataframe(experiment_id, format_version='df03')`

**Clustering**: `src/analyze/trajectory_analysis/`
- `compute_dtw_distance_matrix(trajectories, window)`
- `run_bootstrap_hierarchical(D, k, embryo_ids, n_bootstrap, frac)`
- `analyze_bootstrap_results(bootstrap_results)`

**Utilities**: `horizon_prediction.py` (same directory)
- Binning functions
- Ridge regression patterns

## Known Limitations

1. **Requires distinct trajectory groups**: Won't work if single continuous phenotype
2. **Genotype quality matters**: Unknown genotypes (like cep290) reduce interpretability
3. **Sample size**: Need ≥5 embryos per class per time bin for robust CV
4. **Early timepoints sparse**: May have <2 embryos per class at very early times

## Future Improvements

1. **Multi-class extension**: Support 3+ trajectory groups (increasing, stable, decreasing)
2. **Experiment comparison**: Cross-experiment validation (train on cep290, test on b9d2)
3. **Feature importance**: Which embeddings drive fate prediction?
4. **Temporal dynamics**: How stable is cluster assignment over time?
5. **Genotype-specific models**: Separate classifiers for WT vs het vs hom

---

**Framework validated**: 2025-12-09
**Tested on**: cep290 (20250512) - bimodal rescue vs progression pattern
**Next application**: b9d2 (expected bimodal mutant pattern)
**Status**: Ready for deployment to new experiments
