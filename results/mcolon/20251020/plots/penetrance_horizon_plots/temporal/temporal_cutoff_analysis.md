# Temporal Cutoff Analysis

## How Cutoffs Are Computed

For each time bin, we fit a linear regression:
```
predicted_probability = β₀ + β₁ × distance
```

The **penetrance cutoff** `d*` is the distance where predicted probability = 0.5:
```
0.5 = β₀ + β₁ × d*
d* = (0.5 - β₀) / β₁
```

### When Cutoffs Are Valid

A cutoff `d*` is valid only when:
1. **β₁ > 0** (positive slope - distance increases probability)
2. **β₁ is statistically significant** (p < 0.05)
3. **R² is reasonable** (model explains variance)
4. **d* is positive** (biologically meaningful)

---

## CEP290 Temporal Cutoffs - Problems

Looking at the data, we see many **problematic time bins**:

### Invalid Cutoffs (β₁ ≤ 0)

| Time Bin | β₀ | β₁ | d* | R² | Issue |
|----------|-----|-----|-----|-----|-------|
| 22 hpf | 0.499 | **-0.006** | -0.22 | 0.001 | Negative slope! |
| 24 hpf | 0.546 | **-0.068** | 0.68 | 0.151 | Negative slope! |
| 26 hpf | 0.477 | **-0.012** | -1.89 | 0.013 | Negative slope! |
| 28 hpf | 0.488 | **-0.003** | -3.81 | 0.000 | Negative slope! |
| 50 hpf | 0.511 | **-0.016** | 0.68 | 0.009 | Negative slope! |

**Interpretation**: At these early timepoints (<30 hpf), there's **NO relationship** or even a **negative relationship** between distance and mutant probability. This confirms your hypothesis that early data has weak/absent phenotype!

### Questionable Cutoffs (positive β₁ but very weak model)

| Time Bin | β₀ | β₁ | d* | R² | Issue |
|----------|-----|-----|-----|-----|-------|
| 42 hpf | 0.467 | 0.036 | 0.91 | **0.035** | R² = 3.5% (essentially no predictive power) |
| 44 hpf | 0.500 | 0.032 | -0.002 | **0.024** | R² = 2.4% |
| 48 hpf | 0.529 | 0.005 | -5.57 | **0.001** | R² = 0.1% |

**Interpretation**: Even when slope is positive, R² near zero means the model is useless for prediction.

---

## TMEM67 Temporal Cutoffs - Much Better!

TMEM67 shows **consistently valid cutoffs**:

| Time Bin | β₀ | β₁ | d* | R² | Status |
|----------|-----|-----|-----|-----|--------|
| 30 hpf | 0.400 | 0.146 | 0.68 | **0.785** | ✓ Excellent |
| 32 hpf | 0.393 | 0.135 | 0.80 | **0.687** | ✓ Good |
| 36 hpf | 0.355 | 0.164 | 0.88 | **0.729** | ✓ Excellent |
| 40 hpf | 0.383 | 0.147 | 0.80 | **0.749** | ✓ Excellent |
| 60 hpf | 0.420 | 0.123 | 0.65 | **0.659** | ✓ Good |

**Key observations**:
1. **Consistently positive slopes** (β₁ = 0.05-0.19)
2. **High R²** (most >0.5, many >0.7)
3. **Stable cutoffs** (d* ranges 0.4-0.9, mostly 0.6-0.9)
4. **Biologically interpretable**

### Only 2 questionable bins:

| Time Bin | β₀ | β₁ | d* | R² | Issue |
|----------|-----|-----|-----|-----|-------|
| 50 hpf | 0.472 | 0.050 | 0.55 | **0.093** | Low R² (9.3%) |
| 102 hpf | 0.497 | **-0.006** | -0.48 | **0.002** | Negative slope, low R² |

These are isolated bins - likely due to low sample size at extremes.

---

## Recommended Cutoff Strategy

Based on temporal analysis:

### For CEP290:
1. **DO NOT use cutoffs from bins with R² < 0.2**
2. **Onset detected at ~30 hpf** (R² jumps from 0.03 to 0.26)
3. **Use post-onset data only** (≥30 hpf)
4. **Valid cutoff range**: d* ≈ 0.6-1.3 for bins with R² > 0.3

### For TMEM67:
1. **Strong signal across most bins** (≥30 hpf)
2. **Consistent cutoff**: d* ≈ 0.6-0.8 (most bins)
3. **Can use aggregate cutoff** or time-specific cutoffs
4. **Exclude only problematic bins** (50, 102 hpf)

---

## Biological Interpretation

### CEP290 Penetrance Emerges Late
- **Before 30 hpf**: Morphology is NOT predictive of genotype
  - Homozygous embryos look WT-like
  - Classifier cannot distinguish
  - Distance-probability relationship is **absent or inverted**

- **After 30 hpf**: Phenotype gradually emerges
  - Some homozygous embryos start showing mutant morphology
  - R² increases (but still moderate: 0.2-0.6)
  - Cutoffs become meaningful

**Conclusion**: CEP290 shows **delayed penetrance onset** around 30 hpf, with incomplete penetrance even after onset.

### TMEM67 Penetrance Is Early and Strong
- **By 30 hpf**: Strong phenotype already present
  - R² = 0.78 (excellent predictive power)
  - Clear separation of mutant vs WT morphology

- **Remains strong** throughout development
  - Consistent R² > 0.5 across most bins
  - Stable cutoffs

**Conclusion**: TMEM67 shows **early onset** (<30 hpf) with **high penetrance** and strong phenotype.

---

## Statistical Validity Criteria

For a cutoff to be **biologically and statistically valid**, we require:

1. **β₁ > 0.05** (meaningful effect size)
2. **R² > 0.2** (at least 20% variance explained)
3. **p-value < 0.05** for slope
4. **n ≥ 15 samples** per bin
5. **d* > 0** (positive cutoff)

### Valid Bins Summary:

**CEP290**: 16/28 bins meet criteria (57%)
- Valid bins: 30-40, 52-72 hpf

**TMEM67**: 31/33 bins meet criteria (94%)
- Valid bins: 30-100 hpf (except 50, 102)

---

## Recommended Actions

1. **Filter temporal cutoffs** before using them
   ```python
   valid_cutoffs = temporal_cutoffs[
       (temporal_cutoffs['r_squared'] > 0.2) &
       (temporal_cutoffs['beta1'] > 0.05) &
       (temporal_cutoffs['d_star'] > 0)
   ]
   ```

2. **Report NaN for invalid bins** instead of nonsensical negative values

3. **Use aggregate cutoff for CEP290** (from Step 2):
   - d* = 0.87 (from filtered data ≥30 hpf)
   - More stable than per-bin cutoffs

4. **TMEM67 can use either**:
   - Aggregate: d* = 0.74
   - Per-bin: Most bins are valid

5. **Update workflow document** to reflect onset times and valid ranges

---

## Code Fix Needed

The `compute_temporal_cutoffs()` function should **filter out invalid cutoffs**:

```python
def compute_temporal_cutoffs(
    temporal_results: pd.DataFrame,
    prob_threshold: float = 0.5,
    min_r_squared: float = 0.2,
    min_slope: float = 0.05
) -> pd.DataFrame:
    """Extract valid time-dependent cutoffs."""

    # Compute cutoffs
    cutoffs = temporal_results.copy()
    cutoffs['d_star'] = (prob_threshold - cutoffs['beta0']) / cutoffs['beta1']

    # Mark invalid cutoffs as NaN
    invalid_mask = (
        (cutoffs['r_squared'] < min_r_squared) |
        (cutoffs['beta1'] <= min_slope) |
        (cutoffs['d_star'] <= 0)
    )

    cutoffs.loc[invalid_mask, 'd_star'] = np.nan

    return cutoffs[['time_bin', 'd_star', 'beta0', 'beta1', 'r_squared', 'n_samples']]
```

---

## Summary

- **CEP290**: Most early bins (<30 hpf) have invalid cutoffs due to absent phenotype → confirms hypothesis!
- **TMEM67**: Almost all bins valid → strong, early-onset phenotype
- **Action**: Set invalid cutoffs to NaN, focus on post-onset data
- **Interpretation**: Temporal analysis reveals **when** penetrance becomes detectable, not just **whether** it exists
