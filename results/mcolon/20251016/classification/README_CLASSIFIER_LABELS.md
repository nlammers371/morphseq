# Classifier Label Convention

## Problem

sklearn's `LogisticRegression` orders classes **alphabetically**, which causes confusion when interpreting `predict_proba()` output.

### Example Issue

For genotypes: `['cep290_homozygous', 'cep290_wildtype']`

**Before fix:**
```python
model.classes_ = ['cep290_homozygous', 'cep290_wildtype']  # Alphabetical order
positive_prob = proba[:, 1]  # Probability of class[1] = cep290_wildtype
# BUG: This returns P(wildtype), NOT P(mutant)!
```

This led to **negative correlations** between distance and probability, because:
- Higher distance from WT → embryo looks mutant-like
- Higher pred_proba → classifier thinks it's WT (wrong!)

## Solution

**File:** `classification/predictive_test.py`

The `predictive_signal_test()` function now **explicitly identifies** which class is mutant vs WT:

```python
# Identify WT class (contains 'wildtype', 'wik', or 'ab')
wt_classes = [c for c in class_order if 'wildtype' in str(c).lower() or ...]
mutant_classes = [c for c in class_order if c not in wt_classes]

# Get index of mutant class
mutant_idx = np.where(class_order == mutant_class)[0][0]

# pred_proba is now ALWAYS P(mutant), regardless of alphabetical order
positive_prob = proba[:, mutant_idx]
```

## Key Changes

### Output Columns

The `df_embryo_probs` DataFrame now includes:
- **`pred_proba`**: Probability of **MUTANT** class (not alphabetically second!)
- **`mutant_class`**: Which genotype is treated as mutant
- **`wt_class`**: Which genotype is treated as WT
- `true_label`: Actual genotype label
- `predicted_label`: Predicted genotype
- `support_true`: Probability assigned to true class
- `signed_margin`: Signed distance from decision boundary

### Interpretation

**After fix:**
```python
# For cep290_homozygous embryos
mean_pred_proba = 0.75  # ✓ High probability = correctly classified as mutant

# For cep290_wildtype embryos
mean_pred_proba = 0.25  # ✓ Low probability = correctly classified as WT
```

## Expected Behavior

### Correlation Analysis

**Distance vs Classifier Probability should be POSITIVE:**
- ✓ Embryos **farther from WT** → **higher** mutant probability
- ✓ Positive correlation (r > 0.5) confirms distance is valid phenotypic readout
- ✗ Negative correlation means labels are still flipped somewhere

### Sanity Checks

1. **Homozygous mutants** should have **higher** mean pred_proba than WT
2. **Distance** for homozygous should be **higher** than WT
3. **Pearson r** between distance and probability should be **positive**

## Migration Guide

### Old Code (Before Fix)

```python
# Manually flip probability if needed
if genotype_comes_before_wt_alphabetically:
    pred_prob_mutant = 1 - df_embryo_probs['pred_proba']
else:
    pred_prob_mutant = df_embryo_probs['pred_proba']
```

### New Code (After Fix)

```python
# No manual flip needed! pred_proba is already P(mutant)
pred_prob_mutant = df_embryo_probs['pred_proba']
```

## Testing

Run diagnostic validation:
```bash
python validate_labels.py
```

Expected output:
```
✓ CORRECT: Homozygous embryos are farther from WT
✓ CORRECT: Homozygous embryos have higher mutant probability
✓ POSITIVE strong correlation (r > 0.5)
```

## Future Improvements

Consider adding explicit `mutant_class` parameter to `predictive_signal_test()`:

```python
predictive_signal_test(
    df_binned,
    mutant_class='cep290_homozygous',  # Explicitly specify
    wt_class='cep290_wildtype'
)
```

This would remove ambiguity from pattern matching.
