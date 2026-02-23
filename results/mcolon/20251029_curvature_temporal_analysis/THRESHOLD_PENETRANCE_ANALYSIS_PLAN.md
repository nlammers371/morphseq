# Threshold Optimization + Penetrance Analysis Implementation Plan

## Overview

This document outlines the implementation plan for two new analyses:
1. **Part 1:** Threshold optimization via horizon plots (variance minimization approach)
2. **Part 2:** WT reference-based penetrance analysis

Both analyses follow per-genotype design with **no mixing across genotypes** and use **20% embryo holdout** for bootstrap validation.

---

## Part 1: Threshold Optimization via Horizon Plot

**Script:** `05_threshold_optimization.py`

### Algorithm

#### 1. Data Preparation
- Load combined dataset using `load_data.py`
- Bin data into 2 hpf windows (matching existing pattern in `02_horizon_plots.py`)
- Focus on `normalized_baseline_deviation` metric
- **Analyze each genotype separately** (WT, Het, Homo)

#### 2. Per-Genotype Threshold Optimization Loop
For each genotype independently:
- For each (time_i, horizon_j) pair where i < i+j:
  - Generate threshold candidates (percentiles: 10th, 25th, 50th, 75th, 90th of the distribution at time_i)
  - For each threshold τ:
    - Split embryos at time_i into high/low groups based on whether their curvature > τ
    - Compute within-group variance of curvature at time_(i+j) for both groups
    - Record total weighted variance = (n_low × var_low + n_high × var_high) / n_total
  - Select τ* that minimizes the total weighted variance
  - Record both τ* and min_variance

#### 3. Bootstrap Stability Analysis
- **50 iterations** per genotype
- **Hold out 20% of embryos randomly** per iteration:
  - WT: 18 embryos → ~4 embryos held out
  - Het: 25 embryos → ~5 embryos held out
  - Homo: 35 embryos → ~7 embryos held out
- Re-run optimization on remaining 80% of embryos
- Compute mean and SD of τ* across bootstraps for each (i, j) node
- **Per-genotype bootstrap** (no mixing across genotypes)

### Outputs

#### Figures (`outputs/figures/05_threshold_optimization/`)
- `optimal_thresholds_horizon_WT.png` - Horizon plot of τ*(i,j) for WT only
- `optimal_thresholds_horizon_Het.png` - Horizon plot for Het only
- `optimal_thresholds_horizon_Homo.png` - Horizon plot for Homo only
- `threshold_stability_horizon_WT.png` - SD(τ*) across bootstraps, WT
- `threshold_stability_horizon_Het.png` - SD(τ*) for Het
- `threshold_stability_horizon_Homo.png` - SD(τ*) for Homo
- `variance_reduction_horizon_WT.png` - Minimized variance values, WT
- `variance_reduction_horizon_Het.png` - Minimized variance values, Het
- `variance_reduction_horizon_Homo.png` - Minimized variance values, Homo
- `threshold_comparison_across_genotypes.png` - Compare optimal thresholds across genotypes

#### Tables (`outputs/tables/05_threshold_optimization/`)
- `optimal_thresholds_matrix_WT.csv` - τ* for all (i,j) pairs, WT only
- `optimal_thresholds_matrix_Het.csv` - Het only
- `optimal_thresholds_matrix_Homo.csv` - Homo only
- `bootstrap_stability_WT.csv` - Mean/SD of τ* across bootstraps, WT
- `bootstrap_stability_Het.csv` - Het stability
- `bootstrap_stability_Homo.csv` - Homo stability

### Key Design Choices
- **Variance minimization:** Find threshold that minimizes within-group variance at future timepoint
- **Per-genotype analysis:** Each genotype analyzed independently
- **Bootstrap validation:** 50 iterations with 20% holdout to assess threshold stability
- **Horizon plot visualization:** Upper triangle shows forward-time predictions (i < i+j)

---

## Part 2: WT Reference-Based Penetrance Analysis

**Script:** `06_penetrance_analysis.py`

### Algorithm

#### 1. Compute WT Reference Envelope
- Extract **WT embryos only** (18 embryos, 502 timepoints)
- For each time bin, compute percentile envelope:
  - Default: 2.5th and 97.5th percentiles (95% reference band)
  - Optional sensitivity analysis: 5-95%, 10-90% bands
- Optional: fit smoothing spline with ± k·σ(t) tolerance bands for smoother envelope

#### 2. Mark Penetrant Timepoints (Het and Homo Separately)
**For Het embryos:**
- Compare each Het timepoint to WT envelope
- Mark timepoint as penetrant (1) if curvature falls outside WT reference band
- Binary flag: penetrant = 1, non-penetrant = 0

**For Homo embryos:**
- Compare each Homo timepoint to same WT envelope
- Mark timepoint as penetrant if outside WT reference
- **Keep Het and Homo analyses completely separate**

#### 3. Compute Penetrance Over Time (Per Genotype)
**For Het (independent analysis):**
- **Sample-level penetrance:** % of Het measurements penetrant in each time bin
  - Calculation: (# penetrant Het timepoints in bin) / (# total Het timepoints in bin)
- **Embryo-level penetrance:** % of Het embryos with ≥1 penetrant frame in each time bin
  - Calculation: (# Het embryos with any penetrant frame in bin) / (# total Het embryos in bin)
- Calculate Wilson 95% confidence intervals for both metrics

**For Homo (independent analysis):**
- Same metrics computed separately
- **No mixing with Het data**
- Independent confidence intervals

#### 4. Onset Time Analysis (Per Genotype)
- **For each Het embryo:** Find first time bin where penetrance flag = 1
- **For each Homo embryo:** Find first time bin where penetrance flag = 1
- Plot histograms side-by-side (Het vs Homo)
- Compute summary statistics: median onset, quartiles, range

#### 5. Bootstrap Validation (50 Iterations, 20% Holdout)
**For Het:**
- Hold out 20% of Het embryos (~5 embryos)
- Recompute penetrance metrics on remaining 80%
- Compute 95% CI of penetrance estimates across bootstraps

**For Homo:**
- Hold out 20% of Homo embryos (~7 embryos)
- Recompute penetrance metrics on remaining 80%
- **No mixing with Het data**
- Independent bootstrap CI

### Outputs

#### Figures (`outputs/figures/06_penetrance_analysis/`)
- `wt_reference_envelope.png` - WT percentile bands over time (2.5-97.5%, with median)
- `penetrance_vs_time_Het.png` - Het penetrance % vs time with CI (sample + embryo level)
- `penetrance_vs_time_Homo.png` - Homo penetrance % vs time with CI (sample + embryo level)
- `penetrance_comparison_Het_vs_Homo.png` - Side-by-side comparison of penetrance curves
- `penetrance_onset_distribution_Het.png` - Onset time histogram for Het
- `penetrance_onset_distribution_Homo.png` - Onset time histogram for Homo
- `penetrance_onset_comparison.png` - Overlay of Het vs Homo onset distributions
- `penetrance_sensitivity.png` - Penetrance vs WT band width (2.5-97.5%, 5-95%, 10-90%)

#### Tables (`outputs/tables/06_penetrance_analysis/`)
- `wt_reference_percentiles.csv` - WT envelope values per time bin (2.5th, 50th, 97.5th percentiles)
- `penetrance_by_time_Het.csv` - Het sample/embryo penetrance per bin with CI
- `penetrance_by_time_Homo.csv` - Homo sample/embryo penetrance per bin with CI
- `embryo_onset_times_Het.csv` - First penetrant time per Het embryo
- `embryo_onset_times_Homo.csv` - First penetrant time per Homo embryo
- `onset_summary_statistics.csv` - Median, quartiles, range of onset times by genotype

### Key Design Choices
- **WT reference band:** Use WT 2.5-97.5% percentile envelope as "normal" range
- **Threshold-based penetrance:** Deviation beyond WT band = penetrant
- **Per-genotype analysis:** Het and Homo analyzed separately against same WT reference
- **Bootstrap validation:** 50 iterations with 20% holdout for robust CI
- **Onset quantification:** First time bin where embryo shows penetrant phenotype

---

## Implementation Notes

### Code Structure
- Follow existing pattern: scripts in `results/mcolon/20251029_curvature_temporal_analysis/`
- Leverage `load_data.py` for data loading (single source of truth)
- Use existing utilities from `src/analyze/difference_detection/` where applicable:
  - `horizon_plots.py` for visualization
  - `time_matrix.py` for time binning
- Consistent output structure: `outputs/figures/XX_*/` and `outputs/tables/XX_*/`

### Key Requirements
- **Per-genotype analysis with NO mixing** across genotypes
- **20% embryo holdout for bootstrapping** (not fixed count)
- 2 hpf time bins for consistency with existing analyses
- `normalized_baseline_deviation` as primary metric
- Wilson confidence intervals for penetrance (handles small sample sizes)

### Sample Sizes Per Genotype
- **WT:** 18 embryos, 502 timepoints → 20% holdout = ~4 embryos
- **Het:** 25 embryos, 501 timepoints → 20% holdout = ~5 embryos
- **Homo:** 35 embryos, 1,763 timepoints → 20% holdout = ~7 embryos

### Validation Approach
- Bootstrap with 50 iterations, 20% embryos held out per iteration
- Wilson confidence intervals for penetrance estimates (robust for small n)
- Cross-reference with existing trajectory rankings (script 03)
- Per-genotype stability checks to ensure robust thresholds
- Sensitivity analysis for WT band width (2.5-97.5%, 5-95%, 10-90%)

### Integration with Existing Codebase
- **Data loading:** Use `load_data.py` → `get_analysis_dataframe()`
- **Genotype utilities:** Use `get_genotype_short_name()`, `get_genotype_color()`
- **Time binning:** Follow pattern from `02_horizon_plots.py` (2 hpf bins)
- **Visualization:** Use `plot_horizon_grid()` from `analyze.difference_detection.horizon_plots`
- **Statistical testing:** Use scipy.stats, statsmodels for CI

---

## Expected Deliverables

### Scripts
- `05_threshold_optimization.py` (~500-600 lines)
- `06_penetrance_analysis.py` (~500-600 lines)

### Figures
- **Part 1:** 10 figures (3 genotypes × 3 metrics + 1 comparison)
- **Part 2:** 8 figures (WT envelope + 3 Het + 3 Homo + 1 comparison + 1 sensitivity)
- **Total:** 18 publication-quality figures

### Tables
- **Part 1:** 6 CSV files (3 genotypes × 2 metrics)
- **Part 2:** 6 CSV files (WT reference + 2 genotypes × 2 metrics + 1 summary)
- **Total:** 12-15 summary CSV tables

### Documentation
- Integration with existing horizon plot infrastructure
- README or HANDOFF document describing usage
- Comments explaining variance minimization algorithm
- Notes on interpreting penetrance metrics

---

## Pseudocode Sketches

### Part 1: Threshold Optimization
```python
# Load data
df, metadata = get_analysis_dataframe()

# Per genotype
for genotype in ['cep290_wildtype', 'cep290_heterozygous', 'cep290_homozygous']:
    genotype_df = df[df['genotype'] == genotype]

    # Bin into 2 hpf windows
    time_bins = bin_data_by_time(genotype_df, bin_width=2.0)

    # Initialize results matrix
    optimal_thresholds = np.zeros((n_times, n_times))
    variance_reduction = np.zeros((n_times, n_times))

    # Loop over horizon plot nodes
    for i, time_i in enumerate(time_bins):
        for j, time_j in enumerate(time_bins):
            if time_i >= time_j:
                continue  # Only upper triangle

            # Get embryos at time_i
            embryos_at_i = get_embryos_at_time(genotype_df, time_i)

            # Generate threshold candidates (percentiles)
            thresholds = np.percentile(embryos_at_i['normalized_baseline_deviation'],
                                       [10, 25, 50, 75, 90])

            best_threshold = None
            min_variance = np.inf

            # Test each threshold
            for tau in thresholds:
                # Split into high/low groups
                high_group = embryos_at_i[embryos_at_i['normalized_baseline_deviation'] > tau]['embryo_id']
                low_group = embryos_at_i[embryos_at_i['normalized_baseline_deviation'] <= tau]['embryo_id']

                # Get curvature at time_j for both groups
                high_at_j = get_embryos_at_time(genotype_df, time_j, embryo_ids=high_group)
                low_at_j = get_embryos_at_time(genotype_df, time_j, embryo_ids=low_group)

                # Compute within-group variances
                var_high = np.var(high_at_j['normalized_baseline_deviation'])
                var_low = np.var(low_at_j['normalized_baseline_deviation'])

                # Weighted total variance
                n_high = len(high_at_j)
                n_low = len(low_at_j)
                total_var = (n_high * var_high + n_low * var_low) / (n_high + n_low)

                # Track best
                if total_var < min_variance:
                    min_variance = total_var
                    best_threshold = tau

            optimal_thresholds[i, j] = best_threshold
            variance_reduction[i, j] = min_variance

    # Bootstrap for stability
    bootstrap_thresholds = []
    for iter in range(50):
        # Hold out 20% of embryos
        n_holdout = int(0.2 * len(embryo_ids))
        holdout_embryos = random.sample(embryo_ids, n_holdout)
        train_df = genotype_df[~genotype_df['embryo_id'].isin(holdout_embryos)]

        # Re-run optimization
        optimal_thresholds_iter = run_optimization(train_df, time_bins)
        bootstrap_thresholds.append(optimal_thresholds_iter)

    # Compute stability (SD across bootstraps)
    threshold_stability = np.std(bootstrap_thresholds, axis=0)

    # Plot horizon plots
    plot_horizon_grid({genotype: optimal_thresholds}, ...)
    plot_horizon_grid({genotype: threshold_stability}, ...)
```

### Part 2: Penetrance Analysis
```python
# Load data
df, metadata = get_analysis_dataframe()

# Compute WT reference envelope
wt_df = df[df['genotype'] == 'cep290_wildtype']
time_bins = bin_data_by_time(wt_df, bin_width=2.0)

wt_reference = {}
for time_bin in time_bins:
    wt_at_time = wt_df[wt_df['time_bin'] == time_bin]
    wt_reference[time_bin] = {
        'p2.5': np.percentile(wt_at_time['normalized_baseline_deviation'], 2.5),
        'p50': np.percentile(wt_at_time['normalized_baseline_deviation'], 50),
        'p97.5': np.percentile(wt_at_time['normalized_baseline_deviation'], 97.5)
    }

# Per genotype (Het, Homo)
for genotype in ['cep290_heterozygous', 'cep290_homozygous']:
    genotype_df = df[df['genotype'] == genotype]

    # Mark penetrant timepoints
    genotype_df['penetrant'] = 0
    for idx, row in genotype_df.iterrows():
        time_bin = row['time_bin']
        curvature = row['normalized_baseline_deviation']

        if (curvature < wt_reference[time_bin]['p2.5'] or
            curvature > wt_reference[time_bin]['p97.5']):
            genotype_df.loc[idx, 'penetrant'] = 1

    # Compute penetrance over time
    penetrance_by_time = {}
    for time_bin in time_bins:
        bin_df = genotype_df[genotype_df['time_bin'] == time_bin]

        # Sample-level penetrance
        sample_penetrance = bin_df['penetrant'].mean()

        # Embryo-level penetrance
        embryos_in_bin = bin_df['embryo_id'].unique()
        embryos_penetrant = bin_df[bin_df['penetrant'] == 1]['embryo_id'].unique()
        embryo_penetrance = len(embryos_penetrant) / len(embryos_in_bin)

        # Wilson CI
        sample_ci = wilson_ci(sample_penetrance, len(bin_df))
        embryo_ci = wilson_ci(embryo_penetrance, len(embryos_in_bin))

        penetrance_by_time[time_bin] = {
            'sample_penetrance': sample_penetrance,
            'sample_ci': sample_ci,
            'embryo_penetrance': embryo_penetrance,
            'embryo_ci': embryo_ci
        }

    # Onset time analysis
    onset_times = {}
    for embryo_id in genotype_df['embryo_id'].unique():
        embryo_df = genotype_df[genotype_df['embryo_id'] == embryo_id]
        first_penetrant = embryo_df[embryo_df['penetrant'] == 1].sort_values('predicted_stage_hpf')

        if len(first_penetrant) > 0:
            onset_times[embryo_id] = first_penetrant.iloc[0]['predicted_stage_hpf']

    # Bootstrap validation
    bootstrap_penetrance = []
    for iter in range(50):
        # Hold out 20% of embryos
        embryo_ids = genotype_df['embryo_id'].unique()
        n_holdout = int(0.2 * len(embryo_ids))
        holdout_embryos = random.sample(list(embryo_ids), n_holdout)
        train_df = genotype_df[~genotype_df['embryo_id'].isin(holdout_embryos)]

        # Recompute penetrance
        penetrance_iter = compute_penetrance(train_df, wt_reference, time_bins)
        bootstrap_penetrance.append(penetrance_iter)

    # Compute bootstrap CI
    penetrance_mean = np.mean(bootstrap_penetrance, axis=0)
    penetrance_ci = np.percentile(bootstrap_penetrance, [2.5, 97.5], axis=0)

    # Plot penetrance vs time
    plot_penetrance_curve(penetrance_by_time, penetrance_ci, genotype)
    plot_onset_histogram(onset_times, genotype)
```

---

## Interpretation Guide

### Threshold Optimization Results
- **Optimal threshold horizon plot:** Shows which curvature threshold at time i best predicts future variance at time i+j
  - Hot colors = high thresholds (only very curved embryos separate from normal)
  - Cool colors = low thresholds (even mildly curved embryos separate)
- **Threshold stability:** Shows confidence in threshold estimates
  - Low SD = robust threshold across bootstrap samples
  - High SD = threshold sensitive to sample composition
- **Variance reduction:** Shows how well the optimal threshold separates embryos
  - Low variance = clean separation into distinct groups
  - High variance = overlapping distributions, poor separation

### Penetrance Results
- **Penetrance curves:** Show % of embryos/measurements outside WT range over time
  - Rising penetrance = phenotype emerges during development
  - Flat penetrance = consistent phenotype throughout
- **Onset distribution:** Shows when individual embryos first deviate from WT
  - Tight distribution = synchronous onset
  - Broad distribution = variable onset timing
- **Het vs Homo comparison:** Tests if penetrance differs by genotype
  - Higher Homo penetrance = dose-dependent effect
  - Similar penetrance = all-or-nothing effect

---

## Next Steps After Implementation

1. **Cross-validation:** Compare threshold-based classifications with existing trajectory rankings (script 03)
2. **Genotype comparison:** Test if optimal thresholds differ significantly across genotypes
3. **Temporal dynamics:** Analyze how thresholds evolve over developmental time
4. **Penetrance-curvature correlation:** Test if high penetrance embryos have more severe curvature
5. **Integration with ML models:** Compare threshold-based vs ML-based penetrance definitions (script 04)
