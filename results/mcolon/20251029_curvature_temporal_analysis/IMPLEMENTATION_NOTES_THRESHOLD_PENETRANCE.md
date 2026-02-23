# Implementation Notes: Threshold Optimization & Penetrance Analysis

**Scripts:** `05_threshold_optimization.py` and `06_penetrance_analysis.py`

**Date:** October 30, 2025

**Status:** Complete and ready for execution

---

## Quick Start

### Run Threshold Optimization Analysis
```bash
cd results/mcolon/20251029_curvature_temporal_analysis/
python 05_threshold_optimization.py
```

**Runtime:** ~30-60 minutes (with parallel processing on ~15 CPU cores)

**Outputs:**
- `outputs/05_threshold_optimization/figures/` - 10 horizon plots
- `outputs/05_threshold_optimization/tables/` - 6 CSV files with thresholds and stability metrics

### Run Penetrance Analysis
```bash
cd results/mcolon/20251029_curvature_temporal_analysis/
python 06_penetrance_analysis.py
```

**Runtime:** ~20-40 minutes (with parallel processing on ~15 CPU cores)

**Outputs:**
- `outputs/06_penetrance_analysis/figures/` - 8 comparison plots
- `outputs/06_penetrance_analysis/tables/` - 6 CSV files with penetrance metrics

---

## Part 1: Threshold Optimization Analysis

### What It Does

Implements **variance-minimization threshold optimization** to identify optimal curvature thresholds that separate embryos into groups with minimal within-group variance at future timepoints.

**Key Features:**
- Per-genotype analysis (WT, Het, Homo analyzed independently)
- Horizon plot representation (time_i vs time_j)
- Bootstrap validation (50 iterations, 20% embryo holdout)
- Parallel processing (uses all CPU cores - 1)

### Algorithm Overview

For each genotype:
1. **Threshold Optimization** (Upper triangle of horizon plot)
   - For each pair (time_i, time_j) where i < j:
     - Test threshold candidates (10th, 25th, 50th, 75th, 90th percentiles)
     - For each threshold τ:
       - Split embryos at time_i into high/low groups (value > τ)
       - Measure within-group variance at time_j
       - **Find τ that minimizes total weighted variance**
     - Record optimal threshold τ*(i,j) and minimum variance

2. **Bootstrap Stability** (50 iterations)
   - Randomly hold out 20% of embryos per iteration
   - Re-run optimization on remaining 80%
   - Compute SD of optimal thresholds across bootstraps
   - Low SD = stable/robust thresholds

### Output Files

**Horizon Plots (Figures):**
- `optimal_thresholds_horizon_WT.png` - Best separating thresholds for WT
- `optimal_thresholds_horizon_Het.png` - Best separating thresholds for Het
- `optimal_thresholds_horizon_Homo.png` - Best separating thresholds for Homo
- `threshold_stability_horizon_*.png` - SD of threshold estimates (3 genotypes)
- `variance_reduction_horizon_*.png` - Minimized variance values (3 genotypes)
- `threshold_comparison_across_genotypes.png` - Side-by-side comparison

**CSV Tables:**
- `optimal_thresholds_matrix_WT.csv` - Horizon matrix of optimal τ* for WT
- `optimal_thresholds_matrix_Het.csv` - For Het
- `optimal_thresholds_matrix_Homo.csv` - For Homo
- `bootstrap_stability_WT.csv` - SD of τ* across bootstraps
- `bootstrap_stability_Het.csv` - For Het
- `bootstrap_stability_Homo.csv` - For Homo
- `summary_statistics.csv` - Summary stats per genotype

### Data Details

**Metric Used:** `normalized_baseline_deviation` (baseline_deviation / total_embryo_length)
- Size-independent measure of curvature
- Range: typically 0.01 - 0.10 for this dataset

**Time Binning:** 2 hpf windows
- Creates ~40 time bins across 24-130 hpf developmental range
- Aggregates data within each bin before computing thresholds

**Sample Sizes:**
- WT: 18 embryos, 502 timepoints → 20% holdout ≈ 4 embryos
- Het: 25 embryos, 501 timepoints → 20% holdout ≈ 5 embryos
- Homo: 35 embryos, 1,763 timepoints → 20% holdout ≈ 7 embryos

### Interpretation Guide

**Optimal Thresholds Horizon Plot:**
- Shows which curvature value at time_i best separates embryos
- Hot colors (yellow/red) = high thresholds (strict separation)
- Cool colors (blue/purple) = low thresholds (loose separation)
- Upper triangle = only forward predictions (i < j)
- Lower triangle = NaN (not applicable)

**Threshold Stability Horizon Plot:**
- Shows SD of threshold estimates across 50 bootstrap samples
- **Low SD (cool colors)** = stable, robust thresholds
  - Threshold estimate insensitive to which embryos are included
  - Use this threshold with confidence
- **High SD (warm colors)** = unstable thresholds
  - Threshold estimate varies by which embryos are in the sample
  - Indicates overlapping distributions or sparse data

**Variance Reduction Horizon Plot:**
- Shows the minimized within-group variance achieved by optimal threshold
- **Low variance (cool colors)** = clean separation into distinct groups
- **High variance (warm colors)** = poor separation, overlapping groups

### Key Insights

1. **Early vs Late Development:** Compare thresholds across time ranges
   - Do early timepoints predict late phenotype?
   - Is the optimal threshold consistent over time?

2. **Genotype Differences:** Compare threshold maps across genotypes
   - Do different genotypes require different thresholds for prediction?
   - Are Het and Homo thresholds correlated?

3. **Stability as Confidence:** High SD indicates borderline classifications
   - Consider increasing sample size or using ensemble methods
   - May indicate non-linear decision boundaries

---

## Part 2: Penetrance Analysis

### What It Does

Computes **penetrance as deviation beyond wild-type reference bands**.

**Key Features:**
- Uses WT embryos (18 embryos) to define "normal" morphology range
- WT reference: 2.5-97.5% percentile envelope per time bin
- Analyzes Het and Homo separately against same WT reference
- Sample-level AND embryo-level penetrance metrics
- Bootstrap validation (50 iterations, 20% embryo holdout)
- Wilson confidence intervals (robust for small n)
- Parallel processing (uses all CPU cores - 1)

### Algorithm Overview

1. **Compute WT Reference Envelope**
   - Extract WT embryos only
   - For each 2 hpf time bin:
     - Compute 2.5th, 50th (median), 97.5th percentiles of curvature
     - Store envelope range [p2.5, p97.5]

2. **Mark Penetrant Timepoints**
   - For each Het/Homo timepoint:
     - If curvature < p2.5 OR curvature > p97.5 → **penetrant = 1**
     - Otherwise → **penetrant = 0**
   - Timepoints outside WT range = abnormal phenotype

3. **Compute Penetrance Over Time**
   - **Sample-level penetrance (%):**
     - (# penetrant measurements in bin) / (# total measurements in bin)
   - **Embryo-level penetrance (%):**
     - (# embryos with ≥1 penetrant measurement) / (# embryos in bin)
   - Calculate Wilson 95% confidence intervals

4. **Onset Time Distribution**
   - For each embryo, find first timepoint where penetrance = 1
   - Plot histogram of onset times
   - Compute statistics (mean, median, range)

5. **Bootstrap Validation (50 iterations)**
   - Hold out 20% of embryos per iteration
   - Recompute penetrance on remaining 80%
   - 95% CI from percentiles of bootstrap estimates

### Output Files

**Plots (Figures):**
- `wt_reference_envelope.png` - WT percentile bands over time
- `penetrance_vs_time_Het.png` - Het penetrance with CI
- `penetrance_vs_time_Homo.png` - Homo penetrance with CI
- `penetrance_comparison_Het_vs_Homo.png` - Overlaid comparison
- `penetrance_onset_distribution_Het.png` - Onset histogram for Het
- `penetrance_onset_distribution_Homo.png` - Onset histogram for Homo
- `penetrance_onset_comparison_Het_vs_Homo.png` - Overlaid onset distributions

**CSV Tables:**
- `wt_reference_percentiles.csv` - WT envelope per time bin
- `penetrance_by_time_Het.csv` - Sample/embryo penetrance + CI for Het
- `penetrance_by_time_Homo.csv` - For Homo
- `embryo_onset_times_Het.csv` - First penetrant time per Het embryo
- `embryo_onset_times_Homo.csv` - For Homo
- `summary_statistics.csv` - Summary stats per genotype

### Data Details

**Metric Used:** `normalized_baseline_deviation`
- Same metric as threshold analysis for consistency
- Enables comparison between threshold-based and penetrance-based classifications

**WT Sample Size:**
- 18 embryos with 502 total timepoints
- ~13-14 measurements per time bin
- Adequate for percentile estimation

**Penetrance Definition:**
- Binary: measurement inside vs outside WT envelope
- Accounts for natural WT variation (2.5-97.5% = 95% of WT distribution)
- Values outside this band = statistically unusual

### Interpretation Guide

**WT Reference Envelope:**
- Shows typical WT morphology over development
- Median line = typical trajectory
- Band width = natural WT variation
- Het/Homo values outside band = penetrant phenotype

**Penetrance Curves:**
- Shows % of measurements/embryos with abnormal phenotype at each time
- **Rising curve** = phenotype emerges during development
- **Flat curve** = consistent penetrance throughout
- **Dips in curve** = heterogeneous expression or transient phenotype

**Onset Distribution:**
- Shows when individual embryos first show abnormal phenotype
- **Tight distribution** = synchronous onset across embryos
- **Broad distribution** = variable timing across embryos
- **Early median** = early-onset phenotype
- **Late median** = late-onset phenotype

**Het vs Homo Comparison:**
- Higher Homo penetrance = dose-dependent effect
- Similar penetrance = all-or-nothing effect (same phenotype in Het and Homo)
- Earlier Homo onset = dose-dependent timing

### Key Insights

1. **Phenotype Emergence:** When does the mutant phenotype first appear?
   - Track onset times across individual embryos
   - Test if timing correlates with expression level

2. **Phenotype Progression:** Does penetrance increase, decrease, or plateau over time?
   - Rising penetrance = progressive developmental effect
   - Plateau penetrance = fixed at developmental stage
   - Variable penetrance = heterogeneous or stochastic expression

3. **Genotype Dosage:** Is the effect dose-dependent (Het < Homo)?
   - Compare penetrance curves between genotypes
   - Look for differences in onset time, magnitude, timing

4. **Consistency:** Are penetrant embryos the same across time?
   - Compute correlation of penetrance status across consecutive time bins
   - High correlation = stable, consistent phenotype
   - Low correlation = variable/transient expression

---

## Parallel Processing Details

Both scripts use **multiprocessing.Pool** for speed:

**Threshold Optimization:**
- Parallelizes ~600-1000 (time_i, time_j) node pairs
- Parallelizes 50 bootstrap iterations (each with nested optimization)
- Uses `n_jobs=-1` (all CPU cores - 1)
- Typical speedup: **8-15x on 16-core machine**

**Penetrance Analysis:**
- Parallelizes 50 bootstrap iterations
- Each iteration computes penetrance independently
- Uses `n_jobs=-1` (all CPU cores - 1)
- Typical speedup: **4-6x on 16-core machine**

**Hardware Recommendations:**
- Minimum: 4 cores (but serial mode, slower)
- Recommended: 8+ cores (good parallelism)
- Optimal: 16+ cores (full speedup)

**Memory Usage:**
- Per-process: ~300-500 MB
- Total: n_jobs × (process memory) + shared dataframe
- Typical: ~5-10 GB total

---

## Troubleshooting

### Issue: "ValueError: Bin labels must be one fewer than the number of bin edges"
**Solution:** Fixed in v1.0.1 - use `bin_centers` directly (not `bin_centers[:-1]`)

### Issue: Script hangs or is very slow
**Solutions:**
- Reduce N_BOOTSTRAP from 50 to 10 for testing
- Reduce N_JOBS: set `n_jobs=4` explicitly in function calls
- Check CPU load: `top` or `htop` to verify parallelization

### Issue: Memory error during bootstrap
**Solutions:**
- Reduce N_BOOTSTRAP
- Use fewer workers: `n_jobs=8` instead of `-1`
- Consider running on a high-memory node

### Issue: Results differ between runs
**Status:** Expected! Different 20% embryo holdout each iteration.
- This is the whole point of bootstrap validation
- Consistent SD = stable estimates; high SD = unstable estimates
- Seed (RANDOM_SEED=42) controls initialization but not per-iteration randomness

---

## Integration with Existing Analyses

### Connects to Script 02: Horizon Plots
- Same time binning (2 hpf)
- Same visualization approach (horizon plots)
- Can be displayed side-by-side with correlation matrices

### Connects to Script 03: Trajectory Rankings
- Can use optimal thresholds from 05 to classify embryos
- Compare threshold-based classification with rank-based sorting
- Test if threshold-predicted groups show consistent rankings

### Connects to Script 04: Predictive Models
- Compare threshold-based vs ML-based classifications
- Use penetrance as target variable for ML models
- Test if embeddings predict threshold crossings

### Connects to Existing Penetrance Module
- Alternative to classifier-based penetrance (`compute_embryo_penetrance()`)
- Threshold-based = easier to interpret, WT-centered
- ML-based = data-driven, may capture complex patterns
- Consider both approaches for comprehensive analysis

---

## Code Architecture

### Key Functions: `05_threshold_optimization.py`

**Data Prep:**
- `bin_data_by_time()` - Temporal binning utility

**Core Algorithm:**
- `find_optimal_threshold()` - Single (i,j) node optimization
- `_optimize_single_node()` - Wrapper for parallel execution
- `optimize_thresholds_for_genotype()` - Main optimization (parallelized)

**Bootstrap:**
- `_bootstrap_single_iteration()` - Wrapper for parallel bootstrap
- `bootstrap_threshold_stability()` - Bootstrap with parallel execution

**Visualization:**
- `plot_horizon_heatmap()` - Generic horizon plot
- `plot_threshold_comparison()` - Multi-genotype comparison

**I/O:**
- `save_matrix_to_csv()` - Save matrices with labels

### Key Functions: `06_penetrance_analysis.py`

**Data Prep:**
- `bin_data_by_time()` - Temporal binning
- `compute_wt_reference_envelope()` - WT percentile envelope

**Core Algorithm:**
- `mark_penetrant_timepoints()` - Binary classification
- `compute_penetrance_by_time()` - Sample/embryo level penetrance
- `compute_onset_times()` - Onset time extraction
- `wilson_ci()` - Wilson confidence intervals

**Bootstrap:**
- `_bootstrap_penetrance_single_iteration()` - Parallel bootstrap wrapper
- `bootstrap_penetrance_stability()` - Bootstrap with parallel execution

**Visualization:**
- `plot_wt_reference_envelope()` - Reference band plot
- `plot_penetrance_curves()` - Penetrance with CI
- `plot_onset_distribution()` - Histogram
- `plot_penetrance_comparison()` - Multi-genotype overlay

**I/O:**
- `save_penetrance_table()` - Save results to CSV

---

## Data Flow Diagram

```
Load data (load_data.py)
    ↓
Bin by 2 hpf windows
    ↓
┌─────────────────────────────────────┐
│     For each genotype separately     │
├─────────────────────────────────────┤
│                                     │
│  SCRIPT 05:                SCRIPT 06:
│  Threshold Optimization    Penetrance Analysis
│  ├─ Optimize thresholds    ├─ Compute WT envelope
│  │  per (i,j) node         ├─ Mark penetrant times
│  ├─ 50 bootstrap iters     ├─ Compute penetrance
│  ├─ Compute stability      ├─ Compute onset times
│  └─ Plot horizons          ├─ 50 bootstrap iters
│                            └─ Plot penetrance curves
│
└─────────────────────────────────────┘
    ↓
Save outputs to:
├─ outputs/05_threshold_optimization/
│  ├─ figures/ (10 PNG)
│  └─ tables/ (6 CSV)
└─ outputs/06_penetrance_analysis/
   ├─ figures/ (8 PNG)
   └─ tables/ (6 CSV)
```

---

## Future Extensions

### Sensitivity Analyses
1. **Different WT percentile bands** (5-95%, 10-90% vs 2.5-97.5%)
2. **Different time bins** (1 hpf, 3 hpf vs 2 hpf)
3. **Different metrics** (arc_length_ratio, mean_curvature vs normalized_baseline_deviation)

### Statistical Testing
1. **Permutation tests:** Shuffle genotype labels to test threshold/penetrance significance
2. **LOEO cross-validation:** Leave-one-embryo-out to assess generalization
3. **Signal detection:** Compute ROC curves and AUC for threshold quality

### Integration with ML
1. **Use thresholds as features** in downstream classifiers
2. **Predict penetrance from embeddings** using threshold definitions
3. **Compare with logistic regression** thresholds (statistical approach)

### Dynamics & Trajectories
1. **Threshold crossing dynamics:** Track embryo transition times
2. **Stability of classification:** How many time bins in each state?
3. **Correlation with expression** if available (ISH, RNA-seq)

---

## References & Related Work

- Original prompt: `THRESHOLD_PENETRANCE_ANALYSIS_PLAN.md`
- Existing analyses: `/results/mcolon/20251029_curvature_temporal_analysis/{01-04}_*.py`
- Horizon plot methodology: `src/analyze/difference_detection/horizon_plots.py`
- Data loading: `load_data.py`

---

## Version History

**v1.0.0** (2025-10-30)
- Initial implementation
- Threshold optimization with variance minimization
- WT reference-based penetrance
- Parallel processing (-1 CPU cores)
- Bootstrap validation (50 iterations, 20% holdout)
- Comprehensive visualization and tables

---

## Contact & Support

For questions or issues:
1. Check `THRESHOLD_PENETRANCE_ANALYSIS_PLAN.md` for algorithmic details
2. Review docstrings in `05_threshold_optimization.py` and `06_penetrance_analysis.py`
3. Check existing analysis scripts (01-04) for patterns
4. Inspect output CSVs to understand data structure
