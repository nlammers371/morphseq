# Trajectory Clustering Pipeline: From DTW to Functional Equations

## Overview

This pipeline takes time-series curvature data from embryo development and produces stable trajectory clusters with functional equations describing each group’s dynamics. The approach emphasizes stability, uncertainty quantification, and biological interpretability.

Evbaluate scripts in src/analyze/dtw_time_trend_analysis/README_dtw_time_trend_analysis_extraction.md

**Input**: Time-series curvature measurements for N embryos with potentially misaligned timepoints  
**Output**: Stable clusters with per-group functional equations, confidence bands, and individual random effects

-----

## Step 0: Precompute DTW Distance Matrix and Alignment Paths

### What to do

- Compute full DTW distance matrix `D ∈ ℝ^(N×N)` between all embryo pairs
- **Critical**: Save both distances AND alignment paths to disk
- Standardize preprocessing (smoothing, normalization) across all embryos

### Why

- **Computational efficiency**: DTW is expensive (O(T²) per pair). Computing once and reusing saves hours of redundant calculation
- **Consistency**: Using the same distance matrix ensures all downstream analyses are comparable
- **Alignment paths essential**: Required if computing cluster templates via DTW Barycenter Averaging (DBA) - a powerful method for finding representative average trajectories when timepoints are misaligned

### Implementation notes

```python
# Save as HDF5 or pickle for efficient loading
# Structure: {'distances': D, 'paths': alignment_paths}
# Paths are needed for DBA template computation in Step 5
```

-----

## Step 1: Baseline Clustering

### What to do

- Apply clustering algorithm (k-medoids/PAM recommended) to full distance matrix D
- Start with K=3 based on prior analysis, but test K∈{2,3,4,5}
- Store these labels as reference partition

### Why

- **Reference standard**: Need baseline labels to measure stability against in bootstrap analysis
- **K-medoids advantage**: Medoids are actual embryos from your data (interpretable as exemplars), works directly with precomputed distances without needing to compute means in DTW space
- **Prior knowledge**: Starting at K=3 leverages domain expertise while remaining open to data-driven alternatives

-----

## Step 2: Bootstrap Stability Analysis

### What to do

For B=100 iterations:

1. Sample 80% of embryos without replacement
1. Extract submatrix `D_s` from precomputed D (instant operation)
1. Re-cluster using same algorithm and K
1. Record:

- Adjusted Rand Index (ARI) vs reference labels
- Co-association frequencies for all pairs

### Why

- **Robustness testing**: Real biological clusters should persist across subsamples; unstable clusters suggest overfitting or arbitrary boundaries
- **No re-DTW needed**: Submatrix extraction is instant, making 100+ bootstraps computationally feasible
- **Co-association matrix**: Reveals which embryos consistently cluster together (stable core) vs borderline cases that switch clusters

### Outputs

- Mean/median ARI across bootstraps (higher = more stable structure)
- Co-association matrix `C ∈ [0,1]^(N×N)` where C_ij = fraction of times embryos i,j cluster together
- **Adaptive threshold**: If bootstrap variance is high, lower core membership threshold from 80% to 70%

-----

## Step 3: Determine Optimal K

### What to do

Apply multiple validation metrics:

1. **Stability curve**: Plot mean ARI vs K, identify elbow where stability gains plateau
1. **Consensus clustering**: Apply hierarchical clustering to co-association matrix C, look for natural blocks
1. **Silhouette score**: Compute on D, prefer K with highest average (measures cluster separation)
1. **Gap statistic**: Statistical test comparing within-cluster dispersion to null distribution
1. **Eigengap** (if using spectral): Check eigenvalue gaps in graph Laplacian for natural cluster count

### Why

- **No single truth**: Each metric captures different aspects of cluster quality (stability, separation, compactness)
- **Convergent evidence**: When 3+ independent metrics agree, we have strong statistical support
- **Biological prior**: Weight K=3 slightly higher given domain knowledge, but allow data to override if evidence is strong
- **Consensus blocks**: Visual inspection of block structure provides intuitive validation

### Decision rule

Choose K where ≥3 metrics agree AND consensus matrix shows clean block diagonal structure. If tied, prefer simpler (smaller K) for interpretability.

-----

## Step 4: Define Core vs Uncertain Members

### What to do

For each cluster k:

- **Core members**:
  - Median co-association with cluster-mates ≥ adaptive threshold (70-80% based on bootstrap variance)
  - Silhouette score ≥ 0.2 (positive value indicates better fit to own cluster than neighbors)
- **Uncertain members**: Assigned to cluster but below thresholds
- **Outliers**: Low total co-association with any group (<30% average)

### Why

- **Statistical honesty**: Acknowledging uncertainty prevents overconfident biological conclusions
- **Robust modeling**: Core members provide clean signal for mean curve estimation; uncertain members add noise
- **Biological insight**: Outliers may represent:
  - Distinct developmental programs worth investigating
  - Measurement artifacts or damaged specimens
  - Transitional states between major patterns

-----

## Step 5: Fit Functional Mixed-Effects Models

### What to do

For each cluster k, using CORE members only:

1. **Compute mean trajectory** using TWO methods for cross-validation:

- **Method A**: Fit penalized splines (P-splines) to get smooth `m_k(t)`
- **Method B**: Use DBA to compute DTW-aware average trajectory
- Compare results; if similar, high confidence in mean shape

1. **Estimate random effects** per embryo:

- `b_e,0`: random intercept (vertical shift from mean)
- `b_e,1`: random slope (rate difference)

1. **Estimate variance components**:

- Var(b_e,0), Var(b_e,1), Cov(b_e,0, b_e,1), residual variance σ²_k

### Why

- **Core-only initialization**: Prevents uncertain/outlier members from distorting group patterns
- **P-splines advantages**:
  - Built-in smoothness penalty prevents overfitting
  - More stable than cubic splines with irregular sampling
  - Automatic knot placement
- **DBA validation**: Provides alignment-aware average, confirms spline isn’t missing temporal shifts
- **Mixed-effects framework**: Cleanly separates group-level patterns from individual variation
- **Time centering**: Centers time at mean hpf so intercept b_e,0 represents biologically meaningful midpoint

### Model equation

```
y_e(t) = m_k(t) + b_e,0 + b_e,1·t + ε_e(t)
```

Where:

- `m_k(t)`: Group mean trajectory (from splines or DBA)
- `b_e,0 + b_e,1·t`: Linear individual deviations
- `ε_e(t)`: Residual noise

-----

## Step 6: Optional EM Refinement (Maximum 2-3 iterations)

### What to do

- **E-step**: Compute likelihood of each embryo under each cluster’s mixed-effects model → soft assignments
- **M-step**: Refit curves weighted by soft assignments (keep cores at 2x weight of uncertain members)
- **Stop early** if clusters begin merging or if log-likelihood improvement < 1%

### Why

- **Soft boundaries**: Biology rarely has hard cutoffs; some embryos genuinely intermediate
- **Limited iterations**: Prevents over-smoothing that would erase biologically distinct patterns
- **Core weighting**: Maintains cluster identity while allowing gentle boundary refinement
- **Early stopping**: Biological distinctness more important than mathematical optimality

### Caution

Monitor cluster separation at each iteration. If distinct phenotypes start blending, revert to pre-EM results.

-----

## Step 7: Validation with Held-Out Data

### What to do

1. Hold out 15-20% of embryos BEFORE entire pipeline
1. Run full pipeline on training set
1. Assign holdouts to clusters using two methods:

- **DTW-based**: Nearest medoid in DTW space
- **Model-based**: Maximum likelihood under fitted mixed-effects models

1. Compare assignments and compute prediction RMSE

### Why

- **Generalization test**: Ensures clusters aren’t overfitted artifacts of specific dataset
- **Model validation**: Tests whether functional equations capture generalizable patterns
- **Method agreement**: DTW (shape-based) and model (parametric) assignment should largely agree if clusters are real
- **Prediction accuracy**: RMSE on holdouts tests practical utility of models

### Success criteria

- Assignment agreement > 85% between methods
- Holdout RMSE within 20% of training RMSE

-----

## Step 8: Optional Advanced Analysis - Functional PCA

### What to do

If residuals show systematic patterns, apply functional PCA to capture non-linear variation:

1. Compute residuals: `r_e(t) = y_e(t) - m_k(t) - b_e,0 - b_e,1·t`
1. Apply fPCA to find principal component functions φ_j(t)
1. Examine if first 2-3 components explain >80% of variance

### Why

- **Richer variation model**: Linear random effects may miss important biological variation
- **Example patterns fPCA might find**:
  - Earlier/later peak timing (phase shifts)
  - Sharper/broader peaks (shape changes)
  - Secondary oscillations
- **Biological interpretation**: PC functions often correspond to meaningful developmental variation

### If significant PCs found

Extend model to: `y_e(t) = m_k(t) + b_e,0 + b_e,1·t + Σ_j c_e,j·φ_j(t)`

Where c_e,j are PC scores for embryo e.

-----

## Step 9: Generate Final Deliverables

### Core outputs

#### 1. Cluster Statistics

- Final labels with confidence scores (probability of assignment)
- Consensus matrix heatmap with dendrogram
- Stability metrics: ARI distribution boxplot, mean silhouette by cluster
- Table of core/uncertain/outlier counts with classification criteria

#### 2. Per-Cluster Equations

For each cluster k:

- **Mean trajectory**:
  - P-spline: basis coefficients + knot locations
  - Explicit equation: `m_k(t) = Σ_j β_kj·B_j(t)`
  - Plot with 95% confidence bands
- **Variance components**:
  
  ```
  Random effects: (b_0, b_1) ~ N(0, Σ_k)
  Σ_k = [[var_b0, cov_b0b1], [cov_b0b1, var_b1]]
  Residual SD: σ_k
  ```
- **Goodness of fit**: R², AIC, BIC

#### 3. Per-Embryo Results

- Cluster assignment with soft probability
- Random effects estimates (b̂_e,0, b̂_e,1) with 95% CI
- Residual diagnostics: Q-Q plot, autocorrelation
- Flag if residual SD > 2σ_k (potential misfit)

#### 4. Dynamic Equations (for mechanistic insight)

Per cluster, fit and report:

- **Discrete AR(1)**: `y_t = α_k + φ_k·y_{t-1} + η_t`
  - Stability: |φ_k| < 1 implies convergence
- **Continuous ODE**: `dy/dt = a_k + b_k·y`
  - Stability: b_k < 0 implies convergence to -a_k/b_k

Include parameter CIs and phase plane visualization if applicable.

### Why these outputs

- **Reproducibility**: Complete parameter specification enables exact reconstruction
- **Hypothesis generation**: Equations suggest testable mechanistic models
- **Quality control**: Diagnostics identify specimens needing review
- **Future predictions**: Dynamic models enable trajectory forecasting

-----

## Acceptance Criteria

✓ **Stability**: Bootstrap mean ARI > 0.7 for chosen K  
✓ **Consensus**: Clear block diagonal structure (off-diagonal < 0.3)  
✓ **Coverage**: ≥80% embryos classified as core members  
✓ **Fit quality**: R² > 0.8 for core members in each cluster  
✓ **Validation**: Holdout assignment accuracy > 85%  
✓ **Interpretability**: Cluster means visually distinct, differences > 2σ  
✓ **Convergence**: If EM used, log-likelihood converged

-----

## Key Decision Points and Troubleshooting

### If consensus suggests K≠3

- Investigate thoroughly - biology may differ from expectations
- Check if K=4 splits one cluster or reveals new pattern
- Consider hierarchical structure (subclusters within main groups)

### If >30% uncertain/outliers

- Check data quality and preprocessing
- Consider increasing K or using soft clustering throughout
- May indicate continuous rather than discrete phenotypes

### If validation fails (<70% accuracy)

- Clusters may be unstable - increase bootstrap samples
- Try different clustering algorithms (spectral, DBSCAN)
- Consider that trajectories may not cluster cleanly

### If curves too wiggly/overfitted

- Increase P-spline penalty parameter λ
- Reduce number of knots
- Try polynomial basis instead of splines

### If DBA and spline means differ substantially

- Indicates temporal misalignment issues
- Trust DBA for shape, investigate phase variation
- Consider time-warping before mixed-effects fitting

-----

## Software Stack

### Essential packages

- **DTW**: `dtaidistance` (fast C backend) or `tslearn` (more features)
- **Clustering**: `scikit-learn.cluster.KMedoids`, `scipy.cluster.hierarchy`
- **Mixed-effects**: `statsmodels.MixedLM` or `pymer4`
- **Splines**: `scipy.interpolate.UnivariateSpline` with smoothing
- **Validation**: `scikit-learn.metrics` for ARI, silhouette

### Optional advanced packages

- **DBA**: `tslearn.barycenters.dtw_barycenter_averaging`
- **fPCA**: `scikit-fda` or `fpca` package
- **Visualization**: `seaborn` for heatmaps, `plotly` for interactive trajectories

-----

## Final Notes

This pipeline balances statistical rigor with biological interpretability. Each step has clear rationale and validation. The modular structure

Let’s create simple plan/scriots to accomplish this , note I already have a working dtw algo to use.