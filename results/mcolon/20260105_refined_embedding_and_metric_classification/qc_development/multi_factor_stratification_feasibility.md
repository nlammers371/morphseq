# Multi-Factor Stratified Null Distribution - Feasibility Analysis

**Date:** 2026-01-06
**Status:** Reference document for future consideration
**Decision:** Defer implementation; focus on current analyses

---

## Problem Statement

The current null distribution generation in `compare_groups()` uses **time-stratified permutations** to control for within-bin age confounding. However, additional confounders may create spurious classification signals:

### Potential Confounders

1. **n_obs (observation count per embryo)**
   - Embryos that are "more messed up" may move less → captured at more timepoints
   - Classifier could learn this proxy signal rather than true biological differences
   - Class imbalance: penetrant phenotypes may systematically have higher/lower n_obs

2. **experiment_id (batch effects)**
   - Different experiments may have systematic differences in imaging, handling, etc.
   - If group composition differs across experiments, batch effects can confound signals
   - Proportional representation may vary between genotypes/phenotypes

3. **Class imbalance**
   - Already partially addressed by `class_weight='balanced'` in LogisticRegression
   - But stratification could further help

---

## Current Implementation

**File:** `src/analyze/difference_detection/comparison.py:473-502`

### Time Stratification (Existing)

```python
# Stratified permutation within time strata
if within_bin_time_stratification:
    # Compute time strata (0.5 hr bins within 2 hr time bins)
    time_strata = np.floor(
        (embryo_mean_times - float(t)) / float(within_bin_time_strata_width)
    ).astype(int)

    # Shuffle labels within each time stratum
    for stratum_id in np.unique(time_strata):
        stratum_mask = (time_strata == stratum_id)
        if np.sum(stratum_mask) > 1:
            y_perm[stratum_mask] = local_rng.permutation(y[stratum_mask])
```

**What this controls:** Within-bin age differences (e.g., 12.0 vs 12.5 vs 13.0 hpf)

**What this doesn't control:** n_obs, experiment_id, or other technical confounders

---

## Evidence from QC Scripts

### validate_12hpf_signal.py

**Lines 387, 455-464:**
- Computes `n_obs` per embryo as **count of timepoint observations** in that time bin
- Tests if `n_obs` alone can classify penetrant vs control
- Uses stratified permutation within time strata
- **Key insight:** Reveals if n_obs is driving the classification signal

```python
# Per-embryo summary (line 382-392)
per_embryo = sub.groupby([embryo_id_col, "group"]).agg(
    mean_time=(time_col, "mean"),
    n_obs=(time_col, "size"),  # Number of timepoints per embryo
    curvature_mean=(curvature_col, "mean"),
    ...
)

# Test n_obs-only classification (line 455-464)
nobs_perm = _perm_pval(
    X=per_embryo["n_obs"].values,
    y=y,
    strata=time_strata,  # Still stratify by time
    ...
)
```

---

## Feasibility Analysis

### Challenge: Multi-Factor Stratification

**The Problem:**
Stratifying on multiple factors simultaneously creates **combinatorial explosion** of strata.

**Example:**
- Time strata: 4-8 bins (0.5 hr width in 2-4 hr bins)
- n_obs bins: 5-10 bins
- experiment_id: 3-10 experiments
- **Combined:** 4 × 5 × 5 = **100+ strata**

**Consequences:**
- Many strata will have only 1-2 embryos → **cannot permute**
- Unbalanced class distribution within strata → **no valid permutations**
- Sparse sampling → **unreliable null distribution**

---

## Alternative Approaches

### Option 1: Sequential (Hierarchical) Stratification

**Concept:** Stratify on most important factor first, then sub-stratify within those groups.

**Priority ordering:**
1. **Time** (highest priority - biological confounder)
2. **Experiment_id** (batch effects)
3. **n_obs** (technical artifact)

**Pseudocode:**
```python
def hierarchical_stratified_permutation(y, time_strata, experiment_ids, rng):
    y_perm = y.copy()

    # Level 1: Stratify by time
    for time_s in unique(time_strata):
        time_mask = (time_strata == time_s)

        # Level 2: Sub-stratify by experiment_id within this time stratum
        for exp_id in unique(experiment_ids[time_mask]):
            exp_mask = time_mask & (experiment_ids == exp_id)

            # Permute within time × experiment stratum
            if np.sum(exp_mask) > 1:
                y_perm[exp_mask] = rng.permutation(y[exp_mask])

    return y_perm
```

**Pros:**
- Preserves primary confound control (time)
- Can layer additional factors hierarchically
- Fewer empty strata than full cross-product

**Cons:**
- Order matters - which factor is "most important"?
- Still can have sparse sub-strata
- Doesn't fully control all factors simultaneously

---

### Option 2: Matched Permutation (Propensity Score-like)

**Concept:** Create matched pairs/groups based on confounders, permute within matches.

**Pseudocode:**
```python
# Bin continuous confounders
time_bins = pd.qcut(embryo_times, q=4)
nobs_bins = pd.qcut(n_obs, q=3)

# Create matching key
match_key = time_bins.astype(str) + "_" + nobs_bins.astype(str) + "_" + experiment_ids

# Permute within each match group
for match_group in unique_matches:
    mask = (match_key == match_group)
    if np.sum(mask) > 1 and has_both_classes(y[mask]):
        y_perm[mask] = rng.permutation(y[mask])
```

**Pros:**
- Explicitly controls multiple factors
- Conceptually similar to propensity score matching
- Can adaptively bin to ensure non-empty strata

**Cons:**
- Requires choosing binning strategy (arbitrary)
- Can still have sparse strata if factors are correlated
- May exclude many embryos if no good matches exist

---

### Option 3: Covariate-Adjusted Null (Regression Residuals)

**Concept:** Include confounders as covariates in the model, test residual signal.

**Pseudocode:**
```python
# Model: predict group from features + confounders
X_with_confounders = np.column_stack([X, time, n_obs, experiment_dummies])

# Observed AUROC
clf.fit(X_with_confounders, y)
probs = clf.predict_proba(X_with_confounders)[:, 1]
auroc_obs = roc_auc_score(y, probs)

# Null: permute labels, retrain with same confounders
for perm in range(n_permutations):
    y_perm = rng.permutation(y)
    clf_perm.fit(X_with_confounders, y_perm)
    probs_perm = clf_perm.predict_proba(X_with_confounders)[:, 1]
    auroc_null = roc_auc_score(y_perm, probs_perm)
```

**Pros:**
- Regression framework naturally handles multiple confounders
- No stratification sparsity issues
- Standard statistical approach

**Cons:**
- **Changes the question:** Now testing "does X add signal *beyond* confounders?"
- More complex to implement
- Harder to interpret - is the null distribution still meaningful?
- Could over-control and remove real biological signal if confounders are correlated with phenotype

---

### Option 4: Diagnostic-Only Approach ⭐ **RECOMMENDED**

**Concept:** Don't change null generation; **quantify confounder signals separately** as QC metrics.

**What validate_12hpf_signal.py already does:**
```python
# Test each potential confounder independently
time_only_auroc = classify_on(time_only)
nobs_only_auroc = classify_on(n_obs_only)
experiment_only_auroc = classify_on(experiment_dummies)

# Then test main signal
feature_auroc = classify_on(features)

# Interpretation:
# If nobs_only_auroc > feature_auroc → n_obs is driving the signal!
# If experiment_only_auroc > 0.65 → batch effects are problematic
```

**Pros:**
- **Simple, transparent**
- Reveals which confounders are problematic
- Doesn't require changing core pipeline
- Easy to interpret
- **Well-established** in existing QC scripts

**Cons:**
- Doesn't generate a "confounder-free" null
- Requires manual interpretation
- Post-hoc rather than built-in

---

## Recommended Approach

### Short Term: **Expand Diagnostic Scripts** (Option 4)

**Rationale:**
- Multi-factor stratification is **technically very tricky** with real risk of sparse strata
- Diagnostic approach is **well-established** in validation scripts
- Can **immediately identify** if confounders are driving signals
- Defers complex implementation until we know it's necessary

**Implementation:**

#### 1. Add Confounder Diagnostics to Classification Output

Modify `compare_groups()` to compute and save:
- `n_obs_mean_group1`, `n_obs_mean_group2` (per-embryo observation counts)
- `n_obs_auroc` (classification performance using n_obs only)
- `experiment_composition_divergence` (measure of batch imbalance)

#### 2. Create Standardized Diagnostic Function

```python
def diagnose_confounders(df_binned, group_col, confounders=['n_obs', 'experiment_id']):
    """Test if confounders alone can classify groups.

    Returns dict with:
        - {confounder}_auroc
        - {confounder}_mean_group1
        - {confounder}_mean_group2
    """
    results = {}
    y = (df_binned[group_col] == group1).astype(int)

    for conf in confounders:
        if conf == 'n_obs':
            X = df_binned.groupby('embryo_id').size().values
        elif conf == 'experiment_id':
            X = pd.get_dummies(df_binned['experiment_id']).values

        auroc = cv_classify(X, y)
        results[f'{conf}_auroc'] = auroc

    return results
```

#### 3. Flag Suspicious Signals Automatically

**Thresholds:**
- `n_obs_auroc > 0.60` → **WARNING:** n_obs may be driving signal
- `experiment_id_auroc > 0.65` → **WARNING:** batch effects detected
- Include warnings in summary output CSV

**Example output:**
```
time_bin, auroc_observed, auroc_p_value, n_obs_auroc, experiment_auroc, WARNING
12,       0.72,           0.01,          0.55,         0.58,             None
16,       0.78,           0.001,         0.68,         0.62,             n_obs_confound
20,       0.82,           0.0001,        0.54,         0.71,             batch_effect
```

---

### Medium Term: **Hierarchical Stratification** (Option 1)

**When to implement:**
- If diagnostics reveal **systematic** confounder issues across multiple comparisons
- If confounder AUROCs consistently > 0.65
- If users need automated confounder control

**Implementation:**
- Add `hierarchical_stratification` parameter to `compare_groups()`
- Add `stratification_factors` parameter (ordered list: ['time', 'experiment_id'])
- Implement nested permutation loop

**Usage:**
```python
compare_groups(
    ...,
    hierarchical_stratification=True,
    stratification_factors=['time', 'experiment_id']
)
```

---

### Long Term: **Covariate Adjustment** (Option 3)

**When to consider:**
- If confounders are unavoidable and pervasive
- If stratification approaches consistently fail due to sparsity
- If scientific question is explicitly "signal beyond confounders"

**Caution:**
- Changes the inferential question
- Requires careful interpretation
- Risk of over-controlling

---

## Key Decisions for Future

### 1. Which confounders are most concerning?
- **n_obs?** (technical, non-biological)
- **experiment_id?** (batch effects)
- **Both?**

### 2. Threshold for "problematic" confounder signal?
- When `nobs_auroc > 0.60`? 0.65? 0.70?
- When `experiment_auroc > 0.65`? 0.70?

### 3. Should we modify existing analyses or just add diagnostics?
- **Safe option:** Diagnostics only for now (recommended)
- **Aggressive option:** Re-run all analyses with hierarchical stratification

---

## Implementation Files

**Current stratification:**
- `src/analyze/difference_detection/comparison.py:473-502`

**Diagnostic reference:**
- `qc_development/validate_12hpf_signal.py:455-464`

**Future modifications (if needed):**
- `src/analyze/difference_detection/comparison.py` - Add diagnostic/hierarchical functions
- Analysis scripts - Add confounder diagnostic outputs

---

## Conclusion

**Current recommendation:** Use **diagnostic-only approach** (Option 4)
- Document confounder signals as QC metrics
- Flag suspicious cases automatically
- Defer complex stratification until proven necessary
- Focus on current analyses; revisit if confounder issues are pervasive

**Future work:** If diagnostics reveal systematic issues, implement hierarchical stratification (Option 1) with priority: time > experiment_id > n_obs
