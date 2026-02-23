# Comprehensive Plan: `07_threshold_penetrance_prediction.py`

## **Goal**
Find optimal thresholds τ* at early timepoints that best predict penetrance at future timepoints across three complementary approaches.

---

## **Part 1: Penetrance Separation Maximization**

### Method
For each (time_i, time_j) pair where i < j:
1. At time_i, test candidate thresholds τ (from percentiles of distribution)
2. Split embryos into high (>τ) and low (≤τ) groups
3. At time_j, compute penetrance for each group using WT envelope
4. Best τ* = maximizes: `Δ_penetrance = penetrance_high - penetrance_low`

### Horizon Plots (3 plots per genotype)

**Plot 1: Optimal Threshold Matrix**
- **X-axis:** Prediction time (time_i, 24-110 hpf)
- **Y-axis:** Target time (time_j, 24-130 hpf, reversed so early at top)
- **Color:** Optimal threshold τ* value
- **Constraint:** Only upper-right triangle (j > i)
- **Interpretation:** How threshold changes with timing

**Plot 2: Penetrance Separation Matrix**
- **X-axis:** Prediction time (time_i)
- **Y-axis:** Target time (time_j, reversed)
- **Color:** Δ_penetrance = max separation achieved by τ*
- **Colormap:** viridis (dark=low, bright=high separation)
- **Annotation:** Highlight best (i,j) pairs
- **Interpretation:** Which time pairs give best predictive power

**Plot 3: Prediction Stability Matrix**
- **X-axis:** Prediction time (time_i)
- **Y-axis:** Target time (time_j, reversed)
- **Color:** Standard deviation of Δ_penetrance across bootstrap iterations
- **Colormap:** coolwarm (blue=stable, red=noisy)
- **Interpretation:** Which predictions are robust

### Additional Plots

**Plot 4: Trajectory Split Visualization**
- Focus on best (t_early=46 hpf → t_late=100 hpf)
- Left panel: Distribution at t=46 with optimal τ* marked
- Right panel: Two trajectories (high vs low groups) over time
- Shows how early threshold separates future trajectories

**Plot 5: Time-to-Separation Curve**
- X-axis: Prediction time t_i (early development)
- Y-axis: Minimum time lag Δt needed for good separation (Δ_penetrance > 20%)
- Shows: "If I threshold at 40 hpf, when can I reliably predict outcome?"

---

## **Part 2: ROC/AUC Analysis**

### Method
1. Define ground truth: "penetrant at time_j" (binary, using WT envelope)
2. At time_i, for each embryo, get metric value
3. Use metric as classifier score, compute ROC curve
4. Optimal τ* = point on ROC curve (Youden's J statistic or closest to top-left)

### Plots (per genotype)

**Plot 6: Multi-timepoint ROC Grid**
- 3×3 grid of ROC curves
- Rows: Prediction times (30, 46, 60 hpf)
- Columns: Target times (80, 100, 120 hpf)
- Each subplot: ROC curve with AUC in legend
- Mark optimal τ* on curve with sensitivity/specificity values

**Plot 7: AUC Horizon Plot**
- X-axis: Prediction time (time_i)
- Y-axis: Target time (time_j, reversed)
- Color: AUC value (0.5=random, 1.0=perfect)
- Overlay contour lines at AUC = 0.7, 0.8, 0.9

**Plot 8: Optimal Threshold Comparison**
- X-axis: Time lag (t_j - t_i)
- Y-axis: Optimal threshold τ*
- Color by prediction time t_i
- Shows: Does optimal τ* depend on how far ahead you're predicting?

**Plot 9: Classification Performance Summary**
- Bar chart for key (t_i, t_j) pairs
- Grouped bars: Sensitivity, Specificity, PPV, NPV
- Error bars from bootstrap
- Compare across genotypes

---

## **Part 3: Temporal Consistency Analysis**

### Method
1. Fix prediction time t_i (e.g., 46 hpf)
2. For each threshold τ, compute prediction accuracy at ALL future times
3. Aggregate metric: Mean AUC across all t_j > t_i
4. Best τ* = maximizes temporal consistency score

### Plots

**Plot 10: Consistency Score vs Threshold**
- X-axis: Threshold τ (range of metric values)
- Y-axis: Mean AUC across all future timepoints
- Shaded region: ±1 SD across timepoints
- Mark optimal τ* with vertical line
- One curve per prediction time (30, 46, 60 hpf)

**Plot 11: Temporal Prediction Heatmap**
- X-axis: Threshold τ
- Y-axis: Target time t_j (all future times)
- Color: AUC at each (τ, t_j) combination
- Vertical line at optimal τ* (maximizes column mean)
- Shows: Which τ works best across development

**Plot 12: Robust vs Specific Thresholds**
- Scatter plot
- X-axis: Peak AUC (best performance at single timepoint)
- Y-axis: Mean AUC (performance across all timepoints)
- Each point = one threshold candidate
- Color by threshold value
- Upper-right = robust AND accurate

---

## **Part 4: Information Gain / Mutual Information**

### Method
1. Define discrete states: "penetrant at time_j" (binary outcome)
2. At time_i, for each threshold τ, create binary feature: "crosses threshold"
3. Compute mutual information: `I(threshold_crossing; penetrance_at_j)`
4. Best τ* = maximizes I(X;Y), measures information shared between early threshold and future penetrance

**Mutual Information Formula:**
```
I(X;Y) = H(Y) - H(Y|X)
where:
  H(Y) = entropy of penetrance outcome
  H(Y|X) = conditional entropy given threshold crossing
```

### Plots

**Plot 19: Mutual Information Horizon Plot**
- **X-axis:** Prediction time (time_i)
- **Y-axis:** Target time (time_j, reversed)
- **Color:** Mutual information (bits)
- **Colormap:** plasma (0=no info, high=strong prediction)
- **Interpretation:** Which (i,j) pairs have strongest information transfer

**Plot 20: MI vs Threshold Sweep**
- **X-axis:** Threshold τ (scan across metric range)
- **Y-axis:** Mutual information (bits)
- **Multiple curves:** Different prediction times (30, 46, 60 hpf)
- **Fixed target:** t_j = 100 hpf
- **Mark:** Optimal τ* for each curve
- **Shows:** Does MI have clear peak or plateau?

**Plot 21: Information Cascade Plot**
- **X-axis:** Time lag (t_j - t_i)
- **Y-axis:** Maximum mutual information (across all τ)
- **Color by:** Prediction time t_i
- **Exponential fit:** I_max ~ exp(-λ * Δt) (information decay)
- **Shows:** How quickly does predictive information degrade?

**Plot 22: Conditional Entropy Decomposition**
- Stacked bar chart for key (t_i, t_j) pairs
- **Total bar height:** H(Y) = outcome entropy
- **Bottom segment:** H(Y|X) = residual uncertainty after threshold
- **Top segment:** I(X;Y) = information gained
- **Shows:** How much uncertainty remains vs resolved

**Plot 23: Joint Distribution Heatmap**
- 2×2 grid of heatmaps (one per genotype: Het, Homo)
- **X-axis:** Threshold crossing at t_i (binary: yes/no)
- **Y-axis:** Penetrant at t_j (binary: yes/no)
- **Color:** Count or probability
- **Overlay:** Marginal distributions
- **Shows:** Joint probability P(threshold_cross, penetrant)

---

## **Part 5: DTW Trajectory Clustering (Step 2)**

### **Hypothesis**
**Two trajectory groups exist with anti-correlated early/late curvature:**
- **Group A:** High curvature early (44-50 hpf) → Low curvature late (80-100+ hpf)
- **Group B:** Low curvature early (44-50 hpf) → High curvature late (80-100+ hpf)

This flip-flop pattern suggests different progression dynamics or compensatory mechanisms.

### **Part 5a: Missing Data Validation (Sequential)**

**Strategy:** Test 4 imputation methods on synthetic holdout data, validate, pick best performer.

**Methods:**
1. Linear interpolation (baseline)
2. Spline interpolation (smooth biological curves)
3. Forward fill / Last Observation Carried Forward (conservative)
4. Model-based (IterativeImputer using other embryos)

### Plots

**Plot 19: Imputation Method Validation**
- **X-axis:** Imputation method
- **Y-axis:** RMSE on held-out known values
- **Boxplot:** Distribution across embryos
- **Inset:** Example trajectory (true vs imputed for each method)
- **Shows:** Which method recovers missing values best?

**Plot 20: Clustering Sensitivity to Imputation**
- **X-axis:** Imputation method
- **Y-axis:** Adjusted Rand Index (clustering agreement vs reference)
- **Boxplot:** Distribution across bootstrap samples
- **Reference line:** y=1 (perfect agreement)
- **Shows:** Does imputation method choice affect final clusters?

---

### **Part 5b: DTW Clustering & Anti-Correlation Test**

**Method:**
- Compute pairwise DTW distances (Sakoe-Chiba band, window=3)
- K-means clustering with precomputed distances
- Test k=2,3,4 (focus on k=2 for flip-flop hypothesis)
- Test anti-correlation: early vs late curvature per cluster

### Plots

**Plot 21: Cluster Selection (2×2 Grid)**
- **Panel A:** Elbow curve (inertia vs k)
- **Panel B:** Silhouette scores vs k
- **Panel C:** Gap statistic vs k
- **Panel D:** Penetrance correlation vs k
- **Vertical lines:** Mark optimal k candidates
- **Shows:** Statistical evidence for number of groups

**Plot 22: Anti-Correlation Test (Key Plot!)**
- **X-axis:** Mean curvature early window (44-50 hpf)
- **Y-axis:** Mean curvature late window (80-100 hpf)
- **Points:** Individual embryos, colored by DTW cluster
- **Regression lines:** One per cluster with 95% CI
- **Annotations:** Pearson r, p-value, permutation p-value per cluster
- **Shows:** Evidence for flip-flop (anticorrelated) pattern

**Plot 23: Cluster Trajectories (Focus on k=2)**
- **2 panels:** One per cluster
- **X-axis:** Time (hpf)
- **Y-axis:** Curvature metric (normalized_baseline_deviation)
- **Light lines:** Individual embryo trajectories in cluster
- **Bold line:** Cluster centroid
- **Shaded regions:** Mark early (44-50 hpf) and late (80-100 hpf) windows
- **Annotations:** n embryos, genotype breakdown (% WT/Het/Homo)
- **Shows:** Visual confirmation of flip-flop pattern

**Plot 24: Temporal Profile Comparison (Centroids)**
- **X-axis:** Time (hpf, 24-130)
- **Y-axis:** Curvature metric
- **Two bold lines:** Cluster 1 vs Cluster 2 centroids
- **Confidence bands:** ±1 SD per cluster
- **Vertical dashed lines:** Mark critical windows (44-50, 80-100)
- **Shows:** When do trajectories diverge and converge?

**Plot 25: Genotype Distribution Across Clusters**
- **Stacked bar chart:** One bar per cluster
- **Colors:** WT (blue), Het (orange), Homo (red)
- **Numbers:** Count overlaid on each color segment
- **Shows:** Does flip-flop pattern relate to genotype dosage?

**Plot 26: Multi-Metric Cluster Profiles (Heatmap)**
- **Rows:** Cluster (1, 2, 3)
- **Columns:** Metrics (arc_length, baseline_dev, mean_curvature, max_curvature)
- **Color:** Standardized mean value per cluster
- **Separate heatmaps:** Early window (44-50 hpf) vs Late window (80-100+ hpf)
- **Shows:** Multivariate signature of each trajectory type

**Plot 27: Onset Time Distribution by Cluster**
- **X-axis:** Time of first penetrance event (hpf)
- **Y-axis:** Density
- **Two curves:** One per cluster
- **Vertical lines:** Median onset per cluster
- **Shows:** Do clusters differ in when penetrance begins?

**Plot 28: Dose-Response Within Clusters**
- **2 panels:** One per cluster
- **X-axis:** Genotype (WT/Het/Homo)
- **Y-axis:** Penetrance at t=100 hpf (%)
- **Bar chart:** Mean ± 95% CI
- **Shows:** Does dose-response (WT < Het < Homo) hold within each trajectory type?

**Plot 29: DTW Distance Matrix (Block Structure)**
- **Heatmap:** Embryo × embryo DTW distances
- **Rows/columns:** Sorted by cluster assignment
- **Color:** Distance value (dark=similar, light=different)
- **Shows:** Within-cluster similarity vs between-cluster separation

**Plot 30: Trajectory Clustering PCA Embedding**
- **X/Y axes:** First two principal components of DTW distance matrix
- **Points:** Individual embryos
- **Color:** DTW cluster assignment
- **Shape:** Genotype (triangle=WT, circle=Het, square=Homo)
- **Ellipses:** 95% confidence region per cluster
- **Shows:** 2D visualization of trajectory similarity space

**Plot 31: General Temporal Trends by Cluster** *(NEW)*
- **Layout:** One panel per cluster
- **Light gray lines:** Individual embryo trajectories in cluster
- **Bold black line:** Mean trajectory with markers
- **Black shaded band:** IQR (25th-75th percentile)
- **Blue shaded band:** ±1 SD confidence interval
- **Colored regions:** Early window (cyan) and late window (red)
- **Title annotations:** Cluster size, genotype breakdown, correlation coefficient & interpretation
- **Shows:** Within-cluster variability + central trend + progression dynamics

**Plot 32: Cluster Trajectories Overlay** *(NEW)*
- **X-axis:** Time (hpf, 24-130)
- **Y-axis:** Curvature metric
- **Multiple lines:** One per cluster with unique color
- **Confidence bands:** ±1 SD per cluster
- **Vertical shaded regions:** Early and late windows
- **Shows:** Direct visual comparison of how different trajectory groups progress through development

### Summary Tables

**Table 4: Cluster Characteristics**
Columns: Cluster | n_embryos | %_WT | %_Het | %_Homo | Early_curvature_mean | Late_curvature_mean | Early_late_correlation | Penetrance_mean_t100

**Table 5: Anti-Correlation Evidence**
Columns: Cluster | Early_mean | Late_mean | Pearson_r | P_value | Permutation_p_value | Interpretation

**Table 6: Imputation Method Comparison**
Columns: Method | RMSE_validation | Runtime_sec | Final_k_optimal | ARI_vs_spline | Recommendation

**Table 7: Embryo-Cluster Assignments**
Columns: embryo_id | genotype | cluster_assignment | distance_to_center | early_curvature | late_curvature | penetrant_at_t100

---

## **Part 6: Bayesian Threshold Selection (Step 3, DTW-Informed)**

### Method

**Prior Specification:**
```
τ ~ Normal(median_WT, σ²)  [informative, biology-based]
τ ~ Uniform(min, max)       [non-informative]
τ(t) ~ τ(t-1) + ε          [time-smooth prior]
```

**Likelihood Model:**
- For threshold τ at (t_i, t_j), split embryos into high/low groups
- Each group has penetrance: k_high ~ Binomial(n_high, p_high)
- Likelihood based on separation: Δ = p_high - p_low
- Or variance-based: L(τ) ∝ exp(-weighted_variance)

**Posterior:**
```
P(τ | data) ∝ P(data | τ) × P(τ)
```
- Compute via grid approximation or MCMC
- Summary: MAP (mode), posterior mean/median, credible intervals

**Hierarchical Extension:**
```
τ_global ~ Prior
τ_genotype ~ Normal(τ_global, σ_genotype²)
τ_embryo ~ Normal(τ_genotype, σ_embryo²)
```

### Plots

**Plot 19: Prior vs Posterior Distributions**
- **Multiple panels:** Key (t_i, t_j) pairs
- **X-axis:** Threshold τ
- **Y-axis:** Density
- **Two curves:** Prior (dashed), Posterior (solid)
- **Shaded:** 95% credible interval
- **Shows:** How much data updates prior belief

**Plot 20: Posterior Width Heatmap**
- **X-axis:** Prediction time t_i
- **Y-axis:** Target time t_j (reversed)
- **Color:** Width of 95% credible interval
- **Colormap:** viridis (narrow=identifiable, wide=uncertain)
- **Shows:** Which (i,j) pairs allow tight threshold identification

**Plot 21: MAP vs Posterior Mean**
- **X-axis:** Prediction time (30-70 hpf)
- **Y-axis:** Optimal threshold τ*
- **Two lines:** MAP estimate, Posterior mean
- **Shaded bands:** 50% and 95% credible intervals
- **Shows:** Stability and identifiability across development

**Plot 22: Hierarchical Shrinkage**
- **X-axis:** Embryo index (sorted by raw estimate)
- **Y-axis:** Threshold estimate
- **Three marker types:**
  - Raw estimate (no pooling)
  - Partial pooling (hierarchical)
  - Complete pooling (genotype mean)
- **Shows:** How much shrinkage toward genotype mean

**Plot 23: Time-Smoothed Threshold Trajectory**
- **X-axis:** Prediction time t_i
- **Y-axis:** Optimal threshold τ*(t_i) for fixed t_j=100 hpf
- **Smooth curve:** Time-smooth prior τ(t) ~ τ(t-1) + ε
- **Comparison:** Jump-free vs time-binned estimates
- **Shows:** Does threshold change smoothly during development?

**Plot 24: Posterior Predictive Check**
- **X-axis:** Observed penetrance separation
- **Y-axis:** Density
- **Histogram:** Observed data
- **Curve:** Posterior predictive distribution
- **Shows:** Model fit quality

---

## **Part 6: Logistic Regression with Threshold**
*(Note: This will be implemented in a separate script)*

### Method

**Single-feature model:**
```
logit(P(penetrant at t_j)) = β₀ + β₁ × I(metric > τ at t_i)
```
- Grid search over τ to maximize log-likelihood
- Best τ* = threshold giving highest classification accuracy

**Multi-feature extension:**
```
logit(P(penetrant at t_j)) = β₀ + Σ βₖ × I(metric > τₖ at t_k)
```
- Multiple timepoints contribute features
- L1/L2 regularization for feature selection

### Plots

**Plot 25: Log-Likelihood Surface**
- **X-axis:** Threshold τ
- **Y-axis:** Prediction time t_i
- **Color:** Log-likelihood of model
- **Contour lines:** 90%, 95%, 99% of max likelihood
- **Mark:** Global optimum (τ*, t_i*)
- **Shows:** Is optimum unique or flat?

**Plot 26: Logistic Regression Curves**
- 2×3 grid (2 genotypes × 3 prediction times)
- **X-axis:** Metric value at t_i
- **Y-axis:** P(penetrant at t_j=100 hpf)
- **Sigmoid curve:** Fitted logistic function
- **Points:** Observed data (jittered for visibility)
- **Vertical line:** Optimal threshold τ* (P=0.5 crossing)
- **Shows:** Steepness of transition (sharp vs gradual)

**Plot 27: Model Calibration Curve**
- **X-axis:** Predicted probability (from logistic model)
- **Y-axis:** Observed frequency (binned)
- **Diagonal line:** Perfect calibration
- **Curve:** Actual calibration
- **Error bars:** 95% CI per bin
- **Shows:** Is model well-calibrated?

**Plot 28: Multi-timepoint Feature Importance**
- **X-axis:** Feature index (threshold at different times)
- **Y-axis:** |β coefficient| (absolute value)
- **Color:** Sign of coefficient (red=positive, blue=negative)
- **Error bars:** Bootstrap 95% CI
- **Shows:** Which timepoints matter most for prediction?

**Plot 29: Sequential Feature Selection**
- **X-axis:** Number of timepoint features included
- **Y-axis:** Model AUC (cross-validated)
- **Line:** Forward selection (greedy)
- **Annotations:** Which timepoints added at each step
- **Shows:** Diminishing returns of adding more timepoints

**Plot 30: Prediction Probability Heatmap**
- **X-axis:** Prediction time t_i (30-70 hpf)
- **Y-axis:** Individual embryos (sorted by outcome)
- **Color:** P(penetrant at t_j=100 hpf) from logistic model
- **Right sidebar:** True outcome (penetrant=1, non-penetrant=0)
- **Shows:** Which embryos are consistently predicted across times

**Plot 31: Threshold ROC Comparison**
- Overlay 3 ROC curves per genotype
- **Curve 1:** Simple threshold (Part 2 method)
- **Curve 2:** Single-timepoint logistic regression
- **Curve 3:** Multi-timepoint logistic regression
- **Shows:** Does probabilistic model outperform threshold?

---

## **Part 7: Cross-Method Comparison**

### Plots

**Plot 32: Five-Method Agreement**
- **X-axis:** Prediction time (30-70 hpf)
- **Y-axis:** Optimal threshold τ*
- **Five lines:**
  1. Separation maximization
  2. ROC (Youden's J)
  3. Temporal consistency
  4. Mutual information
  5. Logistic regression
- **Shaded bands:** Bootstrap confidence intervals
- **Shows:** Do methods agree on optimal τ?

**Plot 33: Method Performance Grid** (Het vs Homo only)
- 2×5 grid (2 genotypes × 5 methods)
- For each: Show penetrance curves with optimal τ* applied
- High group (>τ*) vs Low group (≤τ*)
- Shows: Which method gives best dose-response?

**Plot 34: Early Prediction Accuracy**
- Fix t_i = 46 hpf
- X-axis: Target time t_j (50-130 hpf)
- Y-axis: Prediction metric (Δ_penetrance, AUC, etc.)
- Multiple lines for each method
- Shows: How far ahead can we reliably predict?

---

## **Part 8: Biological Interpretation**

### Plots

**Plot 35: WT-Relative Threshold Positions**
- Histogram: WT metric distribution at t=46 hpf
- Overlay: Optimal τ* from each method (vertical lines)
- Annotate: Percentile of WT where τ* falls
- Multiple genotypes in subplots
- Shows: Is optimal τ near WT median? Or in tail?

**Plot 36: Threshold Stability Across Development**
- **X-axis:** Embryo developmental stage (hpf)
- **Y-axis:** Optimal threshold τ* for predicting t=100 hpf
- **Multiple lines:** One per method
- **Shows:** Can we use same τ throughout, or must it change?

**Plot 37: False Positive/Negative Analysis**
- For optimal τ* at 46→100 hpf
- Left panel: Embryos misclassified (false pos/neg)
- Right panel: Their actual trajectories
- Shows: What biological features distinguish errors?

---

## **Summary Tables**

**Table 1: Optimal Thresholds Summary**
Columns: Genotype | Method | t_i | t_j | τ* | AUC | Δ_penetrance | MI (bits) | Log-likelihood | Sensitivity | Specificity

Methods: Separation, ROC, Temporal_consistency, Mutual_info, Logistic_reg

**Table 2: Bootstrap Statistics**
Columns: Genotype | Method | t_i | t_j | τ*_mean | τ*_std | CI_lower | CI_upper

**Table 3: Time-to-Event Predictions**
Columns: Genotype | Method | τ* at t=46 | Predicted_penetrance_at_t=100 | Observed_penetrance | Error

**Table 4: Model Comparison**
Columns: Genotype | Method | AUC | Accuracy | F1_score | Calibration_error

**Table 5: Feature Importance (Multi-timepoint logistic)**
Columns: Genotype | Feature_time | β_coefficient | SE | P_value | Rank

---

## **Implementation Structure (Updated with DTW)**

```python
# ============================================================================
# STEP 1: 07a_threshold_prediction_core.py (Baseline Methods)
# ============================================================================
# 1. Load data & compute WT envelope (global IQR ±2.0)
# 2. Bin by time (2 hpf windows)

# PART 1: Separation maximization
#   - Optimize thresholds for all (i,j) pairs
#   - Generate horizon plots 1-5
#   - Bootstrap validation

# PART 2: ROC/AUC analysis
#   - Compute ROC curves for key (i,j) pairs
#   - Generate plots 6-9
#   - Store optimal thresholds

# PART 3: Temporal consistency
#   - For each t_i, scan thresholds vs all future t_j
#   - Generate plots 10-12

# PART 4: Information gain / Mutual information
#   - Compute MI for all (τ, t_i, t_j) combinations
#   - Generate plots 19-23
#   - Find τ* maximizing MI

# OUTPUT: 18 plots, Tables 1-2, optimal thresholds for Step 2

# ============================================================================
# STEP 2: 07b_dtw_trajectory_clustering.py (NEW - Discover Groups)
# ============================================================================
# 1. Extract per-embryo trajectories (time × metric)

# PART 5a: Missing Data Validation (Sequential)
#   - Test 4 imputation methods (linear, spline, forward-fill, model-based)
#   - Synthetic holdout validation
#   - Pick best method

# PART 5b: DTW Clustering & Anti-Correlation Test
#   - Compute pairwise DTW distances (Sakoe-Chiba band)
#   - Test k=2,3,4 for optimal clusters
#   - TEST HYPOTHESIS: early-high/late-low vs early-low/late-high (flip-flop)
#   - Permutation test for anti-correlation significance
#   - Generate plots 21-32 (12 core plots + 2 new trend plots)
#   - Save cluster assignments for Step 3

# OUTPUT: 14 plots, Tables 4-7, cluster assignments (embryo_id → cluster)

# ============================================================================
# STEP 3: 07c_bayesian_with_dtw_priors.py (DTW-Informed)
# ============================================================================
# 1. Load cluster assignments from Step 2
# 2. Use clusters to inform priors (outline 4 approaches, pick based on 07b)

# PART 6: Bayesian Threshold Selection
#   - Option 1: Cluster-specific thresholds (simple)
#   - Option 2: Hierarchical model (partial pooling)
#   - Option 3: Mixture model (full integration)
#   - Option 4: Time-varying thresholds (cluster-informed)
#   - Generate plots 31-34+ (TBD based on approach choice)

# OUTPUT: Posterior distributions, Tables 8-9, cluster-specific τ*

# ============================================================================
# STEP 4: 07d_logistic_regression.py (Optional)
# ============================================================================
# - Single-timepoint logistic models
# - Multi-timepoint forward selection
# - Generate plots 35-40

# ============================================================================
# STEP 5: 07e_final_comparison.py (Synthesis)
# ============================================================================
# - Compare all methods (threshold + clusters + Bayesian + logistic)
# - Generate plots 41-43
# - Final recommendations
```

---

## **Key Parameters**

- **Time bins:** 2 hpf (consistent with 06b)
- **Prediction times:** 30, 40, 46, 50, 60 hpf (early development)
- **Target times:** All times > prediction time (60-130 hpf)
- **WT envelope:** Global IQR ±2.0σ (from 06b analysis)
- **Bootstrap:** 50 iterations, 20% embryo holdout
- **Threshold candidates:** 10th, 25th, 50th, 75th, 90th percentiles at t_i

---

## **Expected Outputs**

- **36 figures** (comprehensive visualization suite across 7 parts)
  - Part 1: 5 plots (separation maximization + horizon plots)
  - Part 2: 4 plots (ROC/AUC analysis)
  - Part 3: 3 plots (temporal consistency)
  - Part 4: 5 plots (mutual information)
  - Part 5: 7 plots (logistic regression)
  - Part 6: 3 plots (cross-method comparison)
  - Part 7: 3 plots (biological interpretation)
  - Plus 6 additional specialized visualizations

- **5 summary tables** (CSV format)
  - Optimal thresholds across methods
  - Bootstrap statistics
  - Time-to-event predictions
  - Model comparison metrics
  - Feature importance (logistic regression)

- **Per-genotype analysis** (WT/Het/Homo separate)
- **Interpretation guide** in console output
- **Method recommendations** based on performance

---

## **Key Questions Addressed**

This comprehensive design answers:

1. **Which early timepoint best predicts future penetrance?**
2. **What threshold value maximizes predictive power?**
3. **Do different methods agree on optimal thresholds?**
4. **How far ahead can we reliably predict?**
5. **Which embryos are misclassified and why?**
6. **Is prediction better with single or multiple timepoints?**
7. **How does information decay over developmental time?**

This design provides multiple complementary perspectives on the central question: **"What early threshold best predicts future penetrance?"**

---

## **Analysis Organization & Grouping**

### **Script 1: `07a_threshold_prediction_core.py`**
**Core frequentist methods - Fast, interpretable**

**Methods:**
- Part 1: Penetrance Separation Maximization
- Part 2: ROC/AUC Analysis
- Part 3: Temporal Consistency
- Part 4: Mutual Information

**Why grouped:**
- All use similar data structures (time-binned penetrance)
- Can share optimization loops
- Comparable outputs (horizon plots, threshold matrices)
- Run in ~30-60 minutes

**Outputs:**
- 18 plots (1-18)
- Tables 1-2 (optimal thresholds, bootstrap stats)

---

### **Script 2: `07b_threshold_prediction_probabilistic.py`**
**Probabilistic models - Uncertainty quantification**

**Methods:**
- Part 5: Bayesian Threshold Selection
- Part 6: Logistic Regression (single & multi-timepoint)

**Why grouped:**
- Both provide probabilistic interpretations
- Both quantify uncertainty (credible intervals, SEs)
- Require different dependencies (PyMC3/Stan for Bayesian, sklearn for logistic)
- More computationally intensive (~1-2 hours)

**Outputs:**
- 13 plots (19-31)
- Tables 3-5 (predictions, model comparison, feature importance)

---

### **Script 3: `07c_threshold_prediction_comparison.py`**
**Cross-method synthesis - Final recommendations**

**Methods:**
- Part 7: Cross-Method Comparison (all 5 methods)
- Part 8: Biological Interpretation

**Why separate:**
- Requires results from Scripts 1 & 2
- Focus on synthesis and interpretation
- Produces final recommendations
- Lightweight (~10-20 minutes)

**Outputs:**
- 6 plots (32-37)
- Final recommendation report

---

## **Recommended Workflow**

### **Phase 1: Core Analysis (Start Here)**
```bash
# Run the core frequentist methods first
python 07a_threshold_prediction_core.py
```
**Deliverables:**
- Horizon plots showing optimal τ* across time pairs
- ROC curves for key prediction scenarios
- Mutual information quantifying predictive power

**Decision point:** Do we see clear optimal thresholds? Is prediction feasible?

---

### **Phase 2: Probabilistic Models (If needed)**
```bash
# Add uncertainty quantification
python 07b_threshold_prediction_probabilistic.py
```
**Deliverables:**
- Bayesian posterior distributions (how certain are we?)
- Logistic regression models (probabilistic predictions)
- Multi-timepoint feature importance

**Decision point:** Does Bayesian analysis reveal identifiability issues? Which timepoints matter?

---

### **Phase 3: Synthesis (Final step)**
```bash
# Compare all methods and generate recommendations
python 07c_threshold_prediction_comparison.py
```
**Deliverables:**
- Method agreement analysis
- Performance comparison
- Final threshold recommendations
- Biological interpretation

**Outcome:** Publication-ready figures + recommended threshold for experiments

---

## **Method Comparison Matrix**

| Method | Speed | Uncertainty | Interpretability | Use Case |
|--------|-------|-------------|------------------|----------|
| **Separation Max** | Fast | Bootstrap CI | High | Quick screening |
| **ROC/AUC** | Fast | Bootstrap CI | Very High | Clinical threshold |
| **Temporal Consistency** | Medium | Bootstrap CI | High | Robust predictor |
| **Mutual Information** | Medium | Bootstrap CI | Medium | Information theory |
| **Bayesian** | Slow | Full posterior | Medium | Small samples, hierarchical |
| **Logistic Regression** | Medium | SE/CI | Very High | Prediction models |

---

## **Decision Tree: Which Methods to Use?**

```
START
  │
  ├─ Need quick threshold? → Separation Max (Part 1)
  │
  ├─ Need clinical decision threshold? → ROC/AUC (Part 2)
  │
  ├─ Small sample size / pooling needed? → Bayesian (Part 5)
  │
  ├─ Need probabilistic predictions? → Logistic Regression (Part 6)
  │
  ├─ Want information-theoretic view? → Mutual Information (Part 4)
  │
  └─ Want comprehensive comparison? → Run all, use Script 3 for synthesis
```

---

## **Key Design Principles**

1. **Modularity:** Each script is self-contained, can run independently
2. **Shared infrastructure:** All use same data loading from `load_data.py`
3. **Consistent WT envelope:** Global IQR ±2.0σ (from 06b analysis)
4. **Bootstrap validation:** All methods use 50 iterations, 20% holdout
5. **Comparable metrics:** AUC, Δ_penetrance, sensitivity/specificity across all
6. **Progressive complexity:** Start simple (Script 1), add sophistication as needed

---

## **Dependencies (Simplified - No Heavy Packages)**

### Script 1 (`07a_threshold_prediction_core.py`)
```python
numpy, pandas, scipy, matplotlib, seaborn
sklearn.metrics (ROC, roc_auc_score, silhouette_score)
```

### Script 2 (`07b_dtw_trajectory_clustering.py`) - NEW
```python
# Core
numpy, pandas, scipy, matplotlib, seaborn

# DTW & Clustering
scipy.spatial.distance (cdist for custom metrics)
sklearn.cluster (KMeans with precomputed distances)
sklearn.decomposition (PCA for embedding, replaces UMAP)
scipy.interpolate (spline interpolation for missing data)
scipy.signal (derivatives for elbow detection)

# No need for:
# - tslearn (too heavy, use scipy + custom DTW instead)
# - kneed (use simple second derivative method for elbow)
# - umap-learn (use PCA instead)
```

### Script 3 (`07c_bayesian_with_dtw_priors.py`) - Step 3
```python
# Same as Script 1, plus:
sklearn.linear_model (LogisticRegression)
sklearn.model_selection (cross-validation)
scipy.stats (for Bayesian grid approximation)
# pymc3 optional if doing MCMC (but grid approximation sufficient for initial analysis)
```

### Script 4 (`07d_logistic_regression.py`) - Optional
```python
# Same as Script 3
```

### Script 5 (`07e_final_comparison.py`)
```python
# Same as Script 1 (reads saved results from 1-3)
```

---

## **Success Criteria**

**Minimal success:**
- Script 1 completes, shows optimal τ* exists at early times (e.g., 46 hpf)
- AUC > 0.7 for predicting 100 hpf from 46 hpf

**Good success:**
- Multiple methods agree on optimal τ* (±10% variation)
- AUC > 0.8, clear dose-response (WT < Het < Homo)

**Excellent success:**
- All 5 methods converge (τ* within 5%)
- AUC > 0.9, tight credible intervals
- Multi-timepoint logistic regression shows 1-2 key timepoints dominate
- Biological interpretation aligns with known CEP290 phenotypes

---

This organization allows you to:
1. **Start fast:** Run Script 1 first for quick insights
2. **Add depth:** Run Script 2 if you need probabilistic models
3. **Synthesize:** Run Script 3 to compare everything
4. **Publish:** Use Script 3 outputs as main figures, Scripts 1-2 as supplementary
