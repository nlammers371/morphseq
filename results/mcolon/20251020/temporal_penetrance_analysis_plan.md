# Temporal Analysis Plan: Time-Dependent Penetrance

**Date**: 2025-01-20
**Goal**: Investigate how the distance-probability relationship changes over developmental time
**Hypothesis**: Early timepoints (<30 hpf) may have weak phenotypes, causing breakdown of linear relationship

---

## Motivation

### Key Questions:
1. **When does penetrance become detectable?**
   - At what developmental stage do mutant phenotypes diverge from WT?
   - Is there an "onset time" for penetrance?

2. **Does the distance-probability relationship strengthen over time?**
   - Is R² higher at later timepoints?
   - Does the slope (β₁) increase as phenotype becomes more pronounced?

3. **Should cutoffs be time-dependent?**
   - Do we need different d* thresholds for early vs late development?
   - Or is a single aggregate cutoff sufficient?

4. **Why does CEP290 have lower R² than TMEM67?**
   - Is it due to early timepoints with weak phenotype diluting the signal?
   - Would restricting to later timepoints improve CEP290's model fit?

---

## Analysis Strategy

### Approach 1: Per-Time-Bin Regression
**Analyze each time bin independently**

For each time bin with sufficient data:
1. Compute correlation (Pearson, Spearman) within that bin
2. Fit OLS regression: `prob = β₀(t) + β₁(t) × distance`
3. Extract time-dependent metrics:
   - R²(t): Variance explained over time
   - β₁(t): Slope over time (phenotype strength)
   - n(t): Sample size per bin
   - d*(t): Time-dependent cutoff at prob = 0.5

**Minimum samples per bin**: 10-15 embryos (to get stable estimates)

**Outputs**:
- CSV: `{genotype}_temporal_regression.csv` with columns:
  - time_bin, n_embryos, pearson_r, spearman_rho, r_squared, beta0, beta1, beta0_se, beta1_se, d_star
- Plots (see visualization section below)

---

### Approach 2: Sliding Window Analysis
**Capture continuous trends**

Use overlapping time windows to smooth temporal trends:
- Window size: 10-20 hpf
- Slide step: 2-5 hpf
- Compute regression metrics for each window
- Identify critical transition periods

**Advantage**: Less noisy than per-bin analysis
**Disadvantage**: Reduced independence between windows

---

### Approach 3: Interaction Model
**Test if time modulates distance effect**

Fit multivariate regression:
```
prob = β₀ + β₁ × distance + β₂ × time + β₃ × (distance × time) + ε
```

**Interpretation**:
- β₁: Baseline distance effect
- β₂: Time effect on probability
- β₃: **Interaction term** - does distance effect change with time?

If β₃ > 0: Distance effect strengthens over time
If β₃ ≈ 0: Constant relationship across development
If β₃ < 0: Distance effect weakens (unlikely)

**Statistical test**: Is β₃ significantly different from zero?

---

## Implementation Plan

### Module Structure

Create new module: `penetrance_analysis/temporal_analysis.py`

**Functions**:
1. `compute_per_bin_regression(df_distances, df_predictions, genotype, min_samples=10)`
   - Returns: DataFrame with regression metrics per time bin

2. `compute_sliding_window_regression(df_distances, df_predictions, genotype, window_size=15, step=5)`
   - Returns: DataFrame with windowed regression metrics

3. `fit_interaction_model(df_distances, df_predictions, genotype)`
   - Returns: Statsmodels results with time × distance interaction

4. `identify_penetrance_onset(temporal_results, r_squared_threshold=0.3, slope_threshold=0.05)`
   - Returns: Estimated time when penetrance becomes detectable

5. `compute_temporal_cutoffs(temporal_results, prob_threshold=0.5)`
   - Returns: Time-dependent d* values

---

### Visualization Plan

Create new subfolder: `plots/penetrance/temporal/`

**Plot 1: Correlation Strength Over Time**
- X-axis: Time bin (hpf)
- Y-axis: Correlation coefficient (Pearson r, Spearman ρ)
- Multiple lines for CEP290 vs TMEM67
- Shaded region for "weak phenotype" zone (<0.3)
- Identify onset time

**Plot 2: R² Evolution**
- X-axis: Time bin
- Y-axis: R² (variance explained)
- Line plot + confidence bands
- Highlight when R² crosses meaningful thresholds (0.3, 0.5)
- Compare genotypes

**Plot 3: Slope (β₁) Evolution**
- X-axis: Time bin
- Y-axis: Slope coefficient (effect size)
- Error bars: 95% CI
- Interpretation: How fast does distance → probability mapping strengthen?

**Plot 4: Time-Dependent Cutoffs**
- X-axis: Time bin
- Y-axis: d* (penetrance threshold)
- Shows how much distance is needed for penetrance at each stage
- Useful for identifying "critical windows"

**Plot 5: Sample Scatter by Time Bin** (multi-panel)
- Grid of scatterplots: Distance vs Probability
- One panel per time bin (or grouped: early/mid/late)
- Regression lines overlaid
- Visual comparison of relationship strength

**Plot 6: Heatmap: Distance × Time → Probability**
- X-axis: Time bin
- Y-axis: Distance bins
- Color: Mean predicted probability
- Contour lines for penetrance threshold (50%)
- Visualize phenotype emergence

**Plot 7: Interaction Model Results**
- Scatter: Distance vs Prob, colored by time
- Multiple regression lines (one per time quartile)
- Shows if slopes differ across development

**Plot 8: Penetrance Onset Detection**
- Multi-panel showing:
  - R²(t) with onset threshold
  - β₁(t) with significance markers
  - Sample size n(t)
  - Estimated onset time with uncertainty

---

## Folder Organization

```
plots/penetrance/
├── step1_correlation/          # Move Step 1 plots here
│   ├── cep290_homozygous_scatter.png
│   ├── tmem67_homozygous_scatter.png
│   └── correlation_comparison.png
├── step2_regression/           # Move Step 2 plots here
│   ├── cep290_homozygous_ols_fit.png
│   ├── cep290_homozygous_ols_diagnostics.png
│   ├── cep290_homozygous_logit_fit.png
│   ├── cep290_homozygous_logit_diagnostics.png
│   ├── tmem67_homozygous_ols_fit.png
│   ├── tmem67_homozygous_ols_diagnostics.png
│   ├── tmem67_homozygous_logit_fit.png
│   ├── tmem67_homozygous_logit_diagnostics.png
│   └── regression_comparison.png
└── temporal/                   # NEW: Temporal analysis plots
    ├── correlation_over_time.png
    ├── r_squared_evolution.png
    ├── slope_evolution.png
    ├── cutoff_evolution.png
    ├── scatter_by_timebin_cep290.png
    ├── scatter_by_timebin_tmem67.png
    ├── heatmap_distance_time_cep290.png
    ├── heatmap_distance_time_tmem67.png
    ├── interaction_model_cep290.png
    ├── interaction_model_tmem67.png
    ├── penetrance_onset_cep290.png
    ├── penetrance_onset_tmem67.png
    └── temporal_comparison.png

data/penetrance/
└── temporal/                   # NEW: Temporal analysis data
    ├── cep290_homozygous_temporal_regression.csv
    ├── tmem67_homozygous_temporal_regression.csv
    ├── cep290_homozygous_interaction_model.csv
    ├── tmem67_homozygous_interaction_model.csv
    └── temporal_summary.csv
```

---

## Expected Outcomes

### Scenario 1: Early Phenotype Weakness (Your Hypothesis)
**Predictions**:
- R²(t) starts low (<0.2) at early bins (<30 hpf)
- R²(t) increases significantly at later bins
- β₁(t) near zero early, then increases
- CEP290's aggregate R² = 0.32 is dragged down by early timepoints
- Restricting to >30 hpf may improve CEP290 to R² ~ 0.5+

**Implication**: Use time-dependent cutoffs or restrict analysis to post-onset

---

### Scenario 2: Constant Relationship
**Predictions**:
- R²(t) relatively flat across development
- β₁(t) constant
- Interaction term β₃ ≈ 0 (not significant)

**Implication**: Single aggregate cutoff is appropriate

---

### Scenario 3: Genotype-Specific Dynamics
**Predictions**:
- TMEM67: Early onset, stable high R²(t)
- CEP290: Late onset, increasing R²(t)

**Implication**: Different analysis strategies per genotype

---

## Statistical Considerations

### Sample Size Issues
- Each time bin has fewer embryos than aggregate
- Some bins may have <10 samples → unstable estimates
- Solution:
  - Report sample sizes clearly
  - Use sliding windows to smooth
  - Bootstrap CIs for each bin

### Multiple Testing Correction
- Testing ~20-30 time bins → multiple comparisons
- Consider Bonferroni or FDR correction for significance tests
- Or treat as exploratory (follow-up with validation)

### Confounding Factors
- Time and distance may be correlated
  - Embryos develop phenotypes over time
  - Distance may naturally increase with time
- Interaction model helps disentangle these

---

## Implementation Steps

### Step 1: Create `temporal_analysis.py` module
- Per-bin regression functions
- Sliding window functions
- Interaction model fitting
- Onset detection algorithm

### Step 2: Create temporal visualization functions
Add to `visualization.py`:
- `plot_correlation_over_time()`
- `plot_r_squared_evolution()`
- `plot_slope_evolution()`
- `plot_temporal_cutoffs()`
- `plot_scatter_by_timebin()`
- `plot_distance_time_heatmap()`
- `plot_interaction_model()`
- `plot_penetrance_onset()`

### Step 3: Create `run_penetrance_temporal.py` script
Main script that:
1. Loads Step 1 outputs (distances + predictions)
2. Runs per-bin regression
3. Runs interaction model
4. Detects penetrance onset
5. Generates all temporal plots
6. Saves temporal CSV outputs

### Step 4: Organize folder structure
- Create new subfolders
- You'll move existing plots manually

### Step 5: Run analysis and interpret
- Identify key timepoints
- Determine if time-dependent cutoffs are needed
- Update Step 3 plan accordingly

---

## Success Metrics

1. **Clear temporal trends identified**
   - Can visualize when penetrance emerges
   - Quantify onset time with uncertainty

2. **Understand CEP290 vs TMEM67 difference**
   - Is CEP290's lower R² due to temporal mixing?
   - Or fundamental biological difference?

3. **Inform Step 3 cutoff strategy**
   - Use time-dependent cutoffs if warranted
   - Or justify single aggregate cutoff

4. **Publication-ready figures**
   - Clear visualization of phenotype emergence
   - Temporal dynamics of penetrance

---

## Next Steps After This Analysis

Depending on findings:

**If time-dependent cutoffs needed**:
- Step 3 modified to classify penetrance per time bin
- Define penetrant as: "embryo crosses d*(t) threshold at time t"

**If single cutoff sufficient**:
- Proceed with aggregate regression cutoffs
- Restrict analysis to post-onset timepoints (e.g., >30 hpf)

**If early data is noisy**:
- Re-run Steps 1-2 with filtered data (>30 hpf only)
- Compare aggregate vs filtered results

---

## Estimated Implementation Time

- **Module creation**: ~2-3 hours
- **Visualization functions**: ~2-3 hours
- **Main script**: ~1 hour
- **Testing and debugging**: ~1 hour
- **Analysis and interpretation**: ~2-3 hours

**Total**: ~8-12 hours of development + analysis

---

**Ready to proceed with implementation?**
