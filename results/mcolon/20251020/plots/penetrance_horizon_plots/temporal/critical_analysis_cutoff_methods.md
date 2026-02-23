# Critical Analysis: Cutoff Determination Methods

**Date**: 2025-01-20
**Question**: How do we determine a valid cutoff, and how do we measure penetrance using it?

---

## The Fundamental Problem

We want to classify **homozygous mutant embryos** as either:
- **Penetrant**: Shows mutant phenotype
- **Non-penetrant**: Looks wildtype-like

But we have **two continuous variables**, not ground truth labels:
1. **Distance from WT** (morphological deviation)
2. **Predicted mutant probability** (classifier output)

**Challenge**: Neither is a perfect "truth" - both are proxy measurements!

---

## Method 1: Regression-Based Cutoff

### How It Works
From regression: `prob = β₀ + β₁ × distance`

Solve for distance when prob = 0.5:
```
d* = (0.5 - β₀) / β₁
```

**Classify**:
- Penetrant if `distance > d*`
- Non-penetrant if `distance ≤ d*`

### Critical Issues

#### Issue 1: Why 0.5?
- **Arbitrary threshold**
- Assumes 50% probability is meaningful boundary
- But classifier may be calibrated differently
- What if classifier is biased (e.g., β₀ = 0.44 for CEP290)?

#### Issue 2: Assumes Linear Relationship
- Regression only valid if relationship is linear
- Breaks down when R² is low (<0.3)
- CEP290 early bins: R² = 0.001 → regression is meaningless!

#### Issue 3: Sensitivity to Outliers
- Single outlier can drastically shift β₀ and β₁
- Especially problematic with small sample sizes (n=10-20 per bin)

#### Issue 4: Time-Dependent Cutoffs Are Noisy
- d* ranges from -5.6 to 1.6 for CEP290
- Negative cutoffs are biologically nonsensical
- High variance → unstable classification

#### Issue 5: Circular Logic Problem
**We're using classifier probability to define cutoff, then using distance to classify embryos, but:**
- Distance and probability are **correlated by design** (that's what we tested!)
- We're essentially saying: "If classifier says 50% mutant, how far from WT should you be?"
- Then using that distance threshold to declare penetrance
- **But the classifier itself is uncertain** (especially for CEP290)

---

## Method 2: ROC-Based Cutoff (Youden Index)

### How It Works
Treat WT vs Homozygous as **binary truth**:
- Positive class: Homozygous mutants
- Negative class: WT

Use **distance** as discriminator:
```python
from sklearn.metrics import roc_curve

# Create binary labels
y_true = (genotype == 'homozygous').astype(int)
scores = distances

# Get ROC curve
fpr, tpr, thresholds = roc_curve(y_true, scores)

# Find Youden index: max(TPR - FPR)
d* = thresholds[argmax(tpr - fpr)]
```

**Classify homozygous**:
- Penetrant if `distance > d*`
- Non-penetrant if `distance ≤ d*`

### Critical Issues

#### Issue 1: Assumes All Homozygous Should Be "Positive"
- **Incorrect assumption** for incomplete penetrance!
- We're trying to measure penetrance, not assume 100% penetrance
- ROC assumes homozygous = mutant phenotype (which is what we're questioning!)

#### Issue 2: WT Contamination in "Negative" Class
- Some WT embryos may be far from WT mean (natural variation)
- Especially at early timepoints
- Biases cutoff

#### Issue 3: No Biological Interpretation
- Maximizes TPR - FPR, but what does that mean biologically?
- Optimizes statistical separation, not biological relevance

#### Issue 4: Ignores Classifier Information
- Completely discards probability predictions
- Only uses distance

---

## Method 3: Mixture Model / Clustering

### How It Works
Assume homozygous population contains **two subpopulations**:
1. **Non-penetrant** (WT-like): Low distance, low probability
2. **Penetrant** (mutant-like): High distance, high probability

Fit Gaussian Mixture Model (GMM):
```python
from sklearn.mixture import GaussianMixture

# Use both distance and probability
X = np.column_stack([distances, probabilities])

# Fit 2-component GMM
gmm = GaussianMixture(n_components=2, random_state=42)
labels = gmm.fit_predict(X)

# Identify which component is "penetrant" (higher mean distance)
penetrant_cluster = np.argmax(gmm.means_[:, 0])
```

### Advantages
- **Data-driven** cutoff
- Uses both distance AND probability
- Accounts for natural bimodality
- No arbitrary thresholds

### Critical Issues

#### Issue 1: Assumes Bimodality
- What if penetrance is continuous spectrum, not binary?
- May force artificial separation

#### Issue 2: Sample Size Dependent
- GMM unstable with small n
- Per-time-bin analysis would be unreliable

#### Issue 3: Cutoff Is Implicit, Not Explicit
- Can't easily define d* for interpretation
- Classification is probabilistic

---

## Method 4: Quantile-Based Cutoff

### How It Works
Use **WT distribution** to define "normal range":

```python
# Compute WT distance distribution
wt_distances = distances[genotype == 'wildtype']

# Define cutoff as WT 95th percentile
d* = np.percentile(wt_distances, 95)

# Classify homozygous
penetrant = homozygous_distances > d*
```

**Interpretation**: Penetrant = embryo is outside WT normal range

### Advantages
- **Biologically interpretable**: "Abnormally far from WT"
- Independent of classifier
- Based on empirical WT distribution
- Time-specific if computed per bin

### Critical Issues

#### Issue 1: WT May Not Be "Clean"
- WT sample may include:
  - Misgenotyped embryos
  - Heterozygous with strong phenotype
  - Natural outliers

#### Issue 2: Arbitrary Percentile Choice
- Why 95%? Why not 90% or 99%?
- Different choices → very different penetrance estimates

#### Issue 3: Ignores Classifier Information
- Doesn't use probability predictions at all
- Classifier spent all that effort learning - now we ignore it?

#### Issue 4: Time-Dependent WT Distribution Changes
- WT variance may increase with developmental time
- Cutoff shifts even if mutant phenotype doesn't

---

## Method 5: Probability-Based Cutoff

### How It Works
Use **predicted probability directly**:

```python
# Classify based on probability threshold
penetrant = probabilities > 0.5  # or 0.6, 0.7, etc.
```

**Interpretation**: Penetrant = classifier is confident it's mutant

### Advantages
- **Simple**
- Uses classifier output directly
- No need to involve distance
- Threshold is interpretable (probability scale)

### Critical Issues

#### Issue 1: Classifier Calibration
- Is 50% probability actually meaningful?
- CEP290: mean prob for homozygous = 0.52 (barely above 50%!)
- Classifier may be uncalibrated (probabilities don't reflect true frequencies)

#### Issue 2: Ignores Morphological Information
- Distance provides independent evidence
- Throwing away valuable information

#### Issue 3: Classifier Uncertainty
- For CEP290, classifier is uncertain (probabilities 0.35-0.78)
- Not confident enough to trust binary cutoff

---

## Method 6: Combined Score (Distance + Probability)

### How It Works
Create composite score:

```python
# Z-score normalize both
distance_z = (distance - distance_mean) / distance_std
prob_z = (prob - prob_mean) / prob_std

# Combined score
penetrance_score = 0.5 * distance_z + 0.5 * prob_z

# Apply cutoff on combined score
penetrant = penetrance_score > threshold
```

Or use **agreement**:
```python
# Penetrant if BOTH criteria met
penetrant = (distance > d_threshold) & (prob > p_threshold)
```

### Advantages
- Uses both sources of information
- Can weight them based on reliability
- More robust than single measure

### Critical Issues

#### Issue 1: How to Weight?
- Equal weights (0.5, 0.5)?
- Weight by reliability (e.g., by R²)?
- Arbitrary choice

#### Issue 2: Cutoffs on Each Component
- Still need to define d_threshold and p_threshold
- Circular problem remains

---

## The Deeper Conceptual Problem

### What Is "Penetrance" Really?

**Option A: Binary State**
- Embryo either has phenotype or doesn't
- Penetrance = % with phenotype
- **Problem**: Phenotype is continuous, not binary!

**Option B: Continuous Spectrum**
- Phenotype severity ranges from WT-like to severe
- Penetrance = distribution of phenotype severity
- **Problem**: How to summarize as single %?

**Option C: Probability of Expression**
- Each embryo has probability of expressing phenotype (varies by genetic background, environment, stochasticity)
- Penetrance = average probability
- **Problem**: We only observe each embryo once!

### The Measurement Problem

We're trying to measure penetrance, but our measurements are:

1. **Distance**: Morphological deviation
   - Pros: Objective, continuous, time-resolved
   - Cons: No ground truth for "how far is too far"

2. **Probability**: Classifier confidence
   - Pros: Trained on WT vs mutant, integrates multiple features
   - Cons: Uncertain, especially for intermediate embryos

3. **Neither is ground truth!**
   - We don't have manual phenotype labels
   - We don't know which embryos "truly" have penetrant phenotype

**Fundamental issue**: We're defining penetrance using the same measurements we're trying to validate!

---

## Recommended Approach: Multi-Method Validation

Since no single method is perfect, use **convergent evidence**:

### Step 1: Filter to Valid Time Range
- **CEP290**: ≥30 hpf only (R² > 0.2)
- **TMEM67**: ≥30 hpf (strong signal throughout)

### Step 2: Apply Multiple Classification Methods

```python
# Method 1: Regression cutoff (aggregate, not per-bin)
d_reg = (0.5 - beta0_aggregate) / beta1_aggregate
penetrant_reg = distance > d_reg

# Method 2: WT 95th percentile
d_wt95 = np.percentile(wt_distances, 95)
penetrant_wt95 = distance > d_wt95

# Method 3: Probability threshold
penetrant_prob = probability > 0.5

# Method 4: Agreement (conservative)
penetrant_conservative = (distance > d_reg) & (probability > 0.5)

# Method 5: Either criterion (liberal)
penetrant_liberal = (distance > d_reg) | (probability > 0.5)
```

### Step 3: Report Range of Estimates

```
Penetrance estimates for CEP290 (≥30 hpf):
- Regression cutoff (d>0.87): 45% penetrant
- WT 95th percentile (d>1.2): 32% penetrant
- Probability >0.5: 52% penetrant
- Conservative (both): 38% penetrant
- Liberal (either): 59% penetrant

Range: 32-59% (depending on method)
Best estimate: ~40-50% penetrant
```

### Step 4: Sensitivity Analysis

- How does penetrance % change with threshold?
- Plot penetrance vs threshold (for each method)
- Identify "stable range" where estimate doesn't change much

### Step 5: Biological Validation

- Do "penetrant" embryos actually look different?
- Examine images of penetrant vs non-penetrant
- Check if penetrance correlates with other phenotypes (if available)
- Are non-penetrant embryos healthy? Viable?

---

## Measuring "How Many Above/Below Cutoff"

Once cutoff is defined, measuring penetrance seems simple:

```python
penetrant_count = np.sum(distances > d_cutoff)
total_count = len(distances)
penetrance = penetrant_count / total_count
```

### But Critical Issues Remain:

#### 1. **Unit of Measurement**: Embryo or Timepoint?

**Option A: Per-Embryo**
```python
# Aggregate per embryo (e.g., mean distance)
embryo_distances = df.groupby('embryo_id')['distance'].mean()
penetrance = np.mean(embryo_distances > d_cutoff)
```
- Treats each embryo equally
- But loses temporal information

**Option B: Per-Timepoint**
```python
# Each timepoint is independent observation
penetrance = np.mean(all_distances > d_cutoff)
```
- More data points
- But pseudo-replication (same embryo measured multiple times)

**Option C: Majority Vote Per Embryo**
```python
# Embryo is penetrant if >50% of its timepoints exceed cutoff
embryo_penetrant = df.groupby('embryo_id')['distance'].apply(
    lambda x: np.mean(x > d_cutoff) > 0.5
)
penetrance = np.mean(embryo_penetrant)
```
- Robust to single outlier timepoints
- Requires substantial data per embryo

#### 2. **Confidence Intervals**

Don't just report point estimate!

```python
from scipy import stats

# Bootstrap CI
bootstrap_penetrance = []
for _ in range(1000):
    resample = np.random.choice(embryo_distances, size=len(embryo_distances), replace=True)
    bootstrap_penetrance.append(np.mean(resample > d_cutoff))

ci_lower = np.percentile(bootstrap_penetrance, 2.5)
ci_upper = np.percentile(bootstrap_penetrance, 97.5)

print(f"Penetrance: {penetrance:.1%} (95% CI: {ci_lower:.1%}-{ci_upper:.1%})")
```

#### 3. **Sample Size Considerations**

With n=32 embryos (CEP290):
- If true penetrance = 50%, we can estimate ± ~17% (95% CI)
- Narrow CI requires n > 100

**Report uncertainty**:
```
CEP290 penetrance: 45% (95% CI: 28-62%)
TMEM67 penetrance: 85% (95% CI: 71-95%)
```

#### 4. **Time-Stratified Penetrance**

Penetrance may change with developmental time:

```python
# Compute penetrance per time bin
for time_bin in time_bins:
    bin_data = df[df['time_bin'] == time_bin]
    bin_penetrance = np.mean(bin_data['distance'] > d_cutoff)
    print(f"{time_bin} hpf: {bin_penetrance:.1%}")
```

**Question**: Is penetrance increasing over time, or just detectability?

---

## Final Recommendations

### For CEP290 (Weak Signal):

1. **Use aggregate cutoff** from filtered data (≥30 hpf)
   - Regression: d* = 0.87
   - WT 95%: d* ≈ 1.2
   - Report range: "32-52% penetrant depending on method"

2. **Per-embryo classification** (avoid pseudo-replication)
   - Mean distance per embryo >30 hpf
   - Apply cutoff

3. **Multiple methods** + sensitivity analysis

4. **Report as range with CI**:
   ```
   CEP290 incomplete penetrance (≥30 hpf):
   - Regression method: 45% (95% CI: 28-62%)
   - WT percentile method: 32% (95% CI: 17-50%)
   - Consensus estimate: ~35-45%
   ```

5. **Acknowledge limitations**:
   - Low R² (0.26) indicates high uncertainty
   - Classifier not very confident
   - Results should be validated with orthogonal assays

### For TMEM67 (Strong Signal):

1. **Can use either** aggregate or time-specific cutoffs
   - Aggregate: d* = 0.74
   - Per-bin: d* ≈ 0.6-0.8 (stable)

2. **High confidence** due to high R² (0.65)

3. **Expected result**:
   ```
   TMEM67 penetrance (≥30 hpf):
   - ~80-90% penetrant
   - High consistency across methods
   - 95% CI: 70-95%
   ```

---

## The Honest Answer

**We cannot determine a single "correct" cutoff** because:
1. No ground truth exists
2. Phenotype is continuous, not binary
3. Methods make different assumptions
4. Sample sizes are modest

**Best practice**:
- Use multiple methods
- Report range of estimates
- Quantify uncertainty (CIs)
- Validate with independent data
- Be transparent about assumptions

**For publication**:
```
"We estimated incomplete penetrance using multiple complementary methods.
Based on regression-derived cutoffs, WT reference distributions, and
classifier probabilities, we estimate CEP290 penetrance at 35-50%
(n=32 embryos, developmental stage >30 hpf). Results should be validated
with orthogonal phenotyping assays."
```

---

## Questions for Experimental Design

1. **Do you have manual phenotype annotations** for any subset of embryos?
   - If yes: use as validation set
   - If no: consider annotating subset blind to genotype

2. **Are there other phenotypes** you can measure?
   - Cilia length/number
   - Kidney morphology
   - Survival/viability
   - These could validate penetrance calls

3. **Can you measure heterozygous**?
   - Expected to be intermediate
   - Would help calibrate cutoff

4. **What is the biological question**?
   - Comparing penetrance across genes? (TMEM67 vs CEP290)
   - Identifying modifiers? (why do some escape?)
   - Developmental timing? (when does penetrance emerge?)
   - Different questions → different optimal methods

---

**Summary**: Cutoff determination is inherently subjective. Use multiple methods, report ranges, quantify uncertainty, and validate biologically. The goal is robust, reproducible findings, not false precision.
