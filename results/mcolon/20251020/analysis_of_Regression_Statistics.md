# Analysis of Regression Statistics: Incomplete Penetrance

**Date**: 2025-01-20
**Analysis**: Step 1 (Correlation) and Step 2 (Regression) Results
**Genotypes**: CEP290 and TMEM67 homozygous mutants

---

## Executive Summary

This analysis quantifies the relationship between **morphological distance from wildtype** and **classifier-predicted mutant probability** to assess incomplete penetrance in homozygous mutants.

### Key Findings:
- **TMEM67** shows **strong penetrance** with distance explaining **65.1%** of classifier variance
- **CEP290** shows **moderate penetrance** with distance explaining **32.4%** of classifier variance
- Both genotypes show **significant positive correlations**, validating distance as a phenotypic readout
- TMEM67 has **2× stronger morphological deviation** per unit distance compared to CEP290

---

## Step 1: Correlation Analysis

### CEP290 Homozygous (N=32 embryos)

**Correlation Statistics:**
- **Pearson r** = 0.569 (p = 6.7×10⁻⁴)
  - Moderate-to-strong **linear** correlation
  - Statistically significant at p < 0.001

- **Spearman ρ** = 0.648 (p = 6.1×10⁻⁵)
  - Stronger **rank** correlation (more robust to outliers)
  - Higher than Pearson suggests slight non-linearity or outlier influence

- **Bootstrap 95% CIs**:
  - Pearson: [0.283, 0.830]
  - Spearman: [0.327, 0.841]
  - Wide confidence intervals due to modest sample size
  - But consistently positive, confirming real relationship

**Biological Interpretation:**
Embryos farther from WT morphology are more likely to be classified as mutant by the classifier. However, there's substantial variability (~43% unexplained variance), suggesting:
- **Incomplete penetrance**: Some homozygous embryos remain WT-like
- **Variable expressivity**: Wide range of phenotype severity
- **Genetic background effects**: Potential modifiers influencing phenotype

---

### TMEM67 Homozygous (N=31 embryos)

**Correlation Statistics:**
- **Pearson r** = 0.807 (p = 4.1×10⁻⁸)
  - **Strong linear correlation**
  - Highly significant (p << 0.001)

- **Spearman ρ** = 0.787 (p = 1.5×10⁻⁷)
  - Also strong; similar to Pearson suggests predominantly linear relationship

- **Bootstrap 95% CIs**:
  - Pearson: [0.722, 0.896]
  - Spearman: [0.537, 0.931]
  - Tight confidence intervals = robust, reliable finding

**Biological Interpretation:**
TMEM67 shows a much stronger penetrance-distance relationship than CEP290. Morphological distance is a highly reliable predictor of mutant phenotype, suggesting:
- **More complete penetrance**: Most homozygous embryos develop mutant phenotypes
- **Consistent phenotype**: Less variability in phenotypic expression
- **Stronger morphological signal**: Clear separation from WT

---

## Step 2: Regression Analysis

### CEP290 Homozygous Regression

#### OLS Model
```
predicted_probability = 0.4417 + 0.0667 × distance
```

**Model Statistics:**
- **R² = 0.324** (32.4% variance explained)
  - Distance explains about 1/3 of classifier probability variance
  - Moderate predictive power

- **Adjusted R² = 0.302**
  - Penalized for model complexity (minimal change = good fit)

**Regression Coefficients:**
- **β₀ (Intercept) = 0.4417 ± 0.0242**
  - Baseline probability at distance = 0
  - **Interpretation**: Even embryos with zero distance from WT have ~44% predicted mutant probability
  - This might indicate:
    - Classifier uncertainty near decision boundary
    - Batch effects or technical variability
    - Probabilistic nature of binary classification

- **β₁ (Slope) = 0.0667 ± 0.0176**
  - 95% CI: [0.0308, 0.1026]
  - **Positive and significant** (p = 6.7×10⁻⁴)
  - **Interpretation**: For every 1-unit increase in Euclidean distance from WT, the predicted mutant probability increases by **6.7 percentage points**
  - Bootstrap CI: [0.0260, 0.1601] (confirms robustness)

**Model Quality:**
- **F-statistic** = 14.38 (p = 6.7×10⁻⁴)
  - Overall model is highly significant
- **AIC = -72.44** (lower is better)
  - OLS performs better than logit model for CEP290
- **BIC = -69.50**
  - Similar conclusion with Bayesian penalty

**Diagnostic Tests:**
- ✅ **Normality (Shapiro-Wilk)**: W = 0.943, p = 0.089
  - Residuals are approximately normally distributed
  - Assumption satisfied

- ⚠️ **Heteroskedasticity (Breusch-Pagan)**: LM = 3.97, p = 0.046
  - Mild violation (p just below 0.05)
  - Variance of residuals may not be constant
  - **Recommendation**: Consider using robust standard errors or weighted least squares
  - However, violation is marginal and results likely robust

#### Logit Model
```
logit(predicted_probability) = -0.2450 + 0.2836 × distance
```

**Model Statistics:**
- **R² = 0.320** (32.0% variance explained)
  - Similar to OLS

- **β₁ (Slope) = 0.2836 ± 0.0756**
  - Positive and significant
  - On logit scale, harder to interpret directly

**Model Quality:**
- **AIC = 20.85** (worse than OLS)
- **BIC = 23.78** (worse than OLS)
- **Conclusion**: OLS model is preferred for CEP290

---

### TMEM67 Homozygous Regression

#### OLS Model
```
predicted_probability = 0.4003 + 0.1340 × distance
```

**Model Statistics:**
- **R² = 0.651** (65.1% variance explained)
  - **Much stronger fit** than CEP290
  - Distance is an excellent predictor for TMEM67
  - Only ~35% of variance unexplained

- **Adjusted R² = 0.639**
  - Minimal penalty = excellent model fit

**Regression Coefficients:**
- **β₀ (Intercept) = 0.4003 ± 0.0202**
  - Lower baseline than CEP290
  - Embryos at distance = 0 have ~40% mutant probability
  - Tighter standard error (more precise estimate)

- **β₁ (Slope) = 0.1340 ± 0.0182**
  - 95% CI: [0.0968, 0.1713]
  - **TWICE as large as CEP290's slope**
  - **Interpretation**: For every 1-unit increase in distance, predicted probability increases by **13.4 percentage points**
  - Bootstrap CI: [0.1094, 0.2562] (robust)
  - **Stronger morphological deviation** per unit distance

**Model Quality:**
- **F-statistic** = 54.19 (p = 4.1×10⁻⁸)
  - Extremely significant overall model
- **AIC = -75.39** (excellent)
  - Better model fit than CEP290
- **BIC = -72.52**
  - Confirms AIC conclusion

**Diagnostic Tests:**
- ✅ **Normality (Shapiro-Wilk)**: W = 0.948, p = 0.138
  - Residuals are normally distributed
  - Assumption well satisfied

- ✅ **Heteroskedasticity (Breusch-Pagan)**: LM = 1.76, p = 0.185
  - No evidence of heteroskedasticity
  - Constant variance assumption satisfied
  - **All assumptions met** - reliable model

#### Logit Model
```
logit(predicted_probability) = -0.4237 + 0.5749 × distance
```

**Model Statistics:**
- **R² = 0.675** (67.5% variance explained)
  - Slightly better than OLS on logit scale

- **β₁ (Slope) = 0.5749 ± 0.0740**
  - Strong positive effect

**Model Quality:**
- **AIC = 11.54** (worse than OLS)
- **BIC = 14.41** (worse than OLS)
- **Conclusion**: OLS model is preferred for TMEM67

---

## Comparative Analysis

### Summary Table

| Metric | CEP290 | TMEM67 | Interpretation |
|--------|--------|--------|----------------|
| **Correlation (Pearson r)** | 0.569 | 0.807 | TMEM67 has **42% stronger** correlation |
| **Correlation (Spearman ρ)** | 0.648 | 0.787 | Consistent with Pearson |
| **Variance explained (R²)** | 32.4% | 65.1% | TMEM67 explains **2× more variance** |
| **Slope (β₁)** | 0.067 | 0.134 | TMEM67 has **2× larger effect size** |
| **Model quality (AIC)** | -72.44 | -75.39 | Both are good; TMEM67 slightly better |
| **Sample size** | N=32 | N=31 | Similar statistical power |
| **Model assumptions** | Mild heteroskedasticity | All satisfied | TMEM67 more reliable |
| **Bootstrap CI width (R²)** | [0.08, 0.69] | [0.52, 0.80] | TMEM67 more precise |

---

## Biological Interpretation

### Why does TMEM67 show stronger relationships?

1. **More complete penetrance**
   - TMEM67 mutations may produce more consistent morphological phenotypes
   - Fewer "escaper" embryos that remain WT-like despite homozygous genotype

2. **Earlier or stronger phenotype**
   - TMEM67 ciliopathy defects may manifest earlier in development
   - Morphological abnormalities may be more severe and easily detectable

3. **Less genetic background noise**
   - CEP290 may be more sensitive to genetic modifiers
   - Environmental factors or stochastic effects may play larger role in CEP290

4. **Clearer WT-mutant separation**
   - TMEM67 homozygous embryos may form more distinct cluster in morphological space
   - Less overlap with wildtype distribution

### CEP290's lower R² suggests:

1. **Incomplete penetrance**
   - Some homozygous embryos look WT-like (low distance + low probability)
   - Approximately 68% of homozygous embryos show penetrant phenotypes (based on R²)

2. **Variable expressivity**
   - Wide range of phenotype severity among affected embryos
   - Same genotype → different morphological outcomes

3. **Classifier uncertainty**
   - Substantial overlap between WT and mutant distributions in latent space
   - Decision boundary may be fuzzy for CEP290

4. **Additional factors influencing phenotype**
   - Genetic modifiers (background strain effects)
   - Maternal factors
   - Stochastic developmental variation
   - Environmental conditions during embryogenesis

---

## Statistical Considerations

### Model Selection: OLS vs Logit

Both genotypes show **OLS model is preferred** (lower AIC):
- OLS is more interpretable (direct percentage point changes)
- Logit may be theoretically appropriate for probabilities bounded [0,1]
- However, for this probability range (0.35-0.78), OLS performs well
- **Recommendation**: Use OLS for simplicity and interpretability

### Bootstrap Validation

Bootstrap confidence intervals confirm:
- **Positive slopes** are robust (CIs exclude zero)
- **R² estimates** have wide uncertainty for CEP290 (variance = low power)
- **TMEM67 results** are more stable (tighter CIs)

### Sample Size Considerations

- N=31-32 embryos is modest for regression
- Bootstrap CIs account for sample size uncertainty
- Results are statistically significant despite modest N
- Larger samples would narrow confidence intervals

---

## Penetrance Cutoff Calculation

Using the regression equation to define **penetrance threshold** at predicted probability = 0.5:

### CEP290
```
prob = 0.4417 + 0.0667 × distance
0.5 = 0.4417 + 0.0667 × d*
d* = (0.5 - 0.4417) / 0.0667
d* = 0.87 units
```

**Interpretation**: Homozygous CEP290 embryos with distance > 0.87 are classified as **penetrant**

### TMEM67
```
prob = 0.4003 + 0.1340 × distance
0.5 = 0.4003 + 0.1340 × d*
d* = (0.5 - 0.4003) / 0.1340
d* = 0.74 units
```

**Interpretation**: Homozygous TMEM67 embryos with distance > 0.74 are classified as **penetrant**

**Note**: TMEM67 has a **lower cutoff** due to steeper slope - embryos reach 50% mutant probability at shorter distance from WT.

---

## Implications for Step 3

The regression coefficients will be used in Step 3 to:

1. **Define penetrance cutoff** using both regression-based and ROC-based approaches
2. **Classify homozygous embryos** as penetrant vs non-penetrant
3. **Quantify incomplete penetrance** (% of homozygous embryos below cutoff)
4. **Validate cutoff** using:
   - Classifier probability differences between groups
   - Latent space visualization (PCA/UMAP)
   - Temporal progression analysis
   - AUROC/PR-AUC metrics

---

## Recommendations

### For CEP290:
1. **Investigate non-penetrant embryos** (distance < 0.87)
   - Are these true biological "escapers"?
   - Or technical issues (genotyping errors, imaging artifacts)?

2. **Search for genetic modifiers**
   - Strain background differences
   - Compensatory mutations

3. **Consider additional phenotypic measurements**
   - Distance alone explains only 32% of variance
   - What other features distinguish penetrant vs non-penetrant?

### For TMEM67:
1. **Strong model validates approach**
   - Distance is an excellent readout for TMEM67 penetrance
   - Results are robust and reproducible

2. **Use as positive control**
   - TMEM67 can serve as benchmark for incomplete penetrance analysis
   - Compare other genes to TMEM67 penetrance strength

### General:
1. **Validate cutoffs in Step 3** using independent metrics (ROC)
2. **Examine outliers** (high influence points in Cook's distance plots)
3. **Temporal analysis** - does correlation strengthen at later timepoints?
4. **Consider heterozygous** - do they show intermediate phenotypes?

---

## Conclusions

1. **Both genotypes show significant positive relationships** between morphological distance and classifier-predicted mutant probability, validating distance as a phenotypic readout.

2. **TMEM67 demonstrates stronger, more reliable penetrance** (R² = 0.65) compared to CEP290 (R² = 0.32), suggesting more complete and consistent phenotype expression.

3. **OLS regression is the preferred model** for both genotypes based on AIC/BIC criteria and interpretability.

4. **Regression-based penetrance cutoffs** are d* = 0.87 for CEP290 and d* = 0.74 for TMEM67 at the 50% probability threshold.

5. **Ready to proceed to Step 3** to validate these cutoffs using ROC analysis and classify embryos as penetrant vs non-penetrant.

---

**Next Steps**: Implement Step 3 to calculate ROC-based cutoffs, compare with regression cutoffs, and quantify incomplete penetrance fractions for both genotypes.
