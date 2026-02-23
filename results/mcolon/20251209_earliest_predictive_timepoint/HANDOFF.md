# Earliest Predictive Timepoint Analysis - Handoff Document

**Experiment**: 20250512 (cep290 mutants + wildtype)
**Date**: 2024-12-09
**Status**: Working baseline model implemented, genotype-specific R² visualization in progress

---

## What Works ✅

### Core Analysis Pipeline
- **Data Loading**: Loading df03 build06 output with embeddings + curvature
- **Binning**: Time binning embeddings (2.0h bins) from 12-74 hpf
- **Training**: Ridge regression trained on wildtype + heterozygous together (49 embryos, 1009 binned samples)
- **Predictions**: 496 time-pair combinations (32 time bins × forward pairs)
- **Cross-validation**: GroupKFold by embryo_id (5 folds)

### Results Generated
- **CSV Files**:
  - `results_all_pairs.csv` - Long-format results (496 rows)
  - `r2_matrix_overall.csv` - Overall R² matrix (32×32)
  - `mse_matrix_cep290_wildtype.csv` - MSE for wildtype
  - `mse_matrix_cep290_heterozygous.csv` - MSE for heterozygous

- **Figures**:
  - `r2_horizon_overall.png` - Overall R² horizon plot (clipped 5-95 percentiles)
  - `mse_horizon_by_genotype.png` - MSE comparison by genotype

### Key Findings (Baseline Model)
- **Overall R²**: Mean = -2.038 ± 4.280 (range: -37.3 to 0.313)
- **Best R²**: 0.313 (28h → 30h prediction)
- **Genotype-Specific MSE**:
  - Wildtype: 0.000678 (lower = more predictable)
  - Heterozygous: 0.000889 (higher = less predictable)

---

## What's In Progress 🔄

### Genotype-Specific R² Horizon Plots
**Goal**: Create separate horizon plots showing R² for each genotype (wildtype vs heterozygous)

**Current Issue**: R² values are being computed per-genotype in test folds but NOT being saved to results CSV
- The `fit_single_time_pair()` function computes `r2_cep290_wildtype` and `r2_cep290_heterozygous`
- They should be saved as columns in the results DataFrame
- But column names in CSV show only MSE columns, not R² per-genotype

**What was attempted**:
```python
# In fit_single_time_pair(), for each fold:
geno_r2 = r2_score(y_test[geno_mask], y_pred[geno_mask])
genotype_mses[f'r2_{genotype}'] = [geno_r2, ...]

# In result aggregation:
if key.startswith('r2_'):
    result[key] = np.mean(value_list)  # Should save as r2_cep290_wildtype, etc.
```

**Next steps**:
1. Debug why R² columns aren't appearing in CSV (likely DataFrame column reordering issue)
2. Build R² matrices per genotype: `build_r2_matrix(results_df, metric='r2_cep290_wildtype')`
3. Generate R² horizon plots by genotype for easy visual comparison

---

## File Structure

```
results/mcolon/20251209_earliest_predictive_timepoint/
├── horizon_prediction.py          # Main script (440+ lines)
├── HANDOFF.md                     # This file
├── output/
│   ├── results_all_pairs.csv      # Long-format results
│   ├── r2_matrix_overall.csv
│   ├── mse_matrix_cep290_wildtype.csv
│   ├── mse_matrix_cep290_heterozygous.csv
│   └── figures/
│       ├── r2_horizon_overall.png
│       └── mse_horizon_by_genotype.png
```

---

## How to Run

```bash
cd results/mcolon/20251209_earliest_predictive_timepoint/
python horizon_prediction.py
```

**Runtime**: ~15-20 minutes for full 496 time pairs
**Output**: CSV + PNG files in `output/` folder

---

## Code Structure

### Main Functions in `horizon_prediction.py`

**Data Preparation**:
- `bin_embryos_by_time()` - Bin VAE embeddings by time (local copy to avoid import issues)
- `bin_data_for_prediction()` - Wraps binning, computes binned curvature
- `get_forward_time_pairs()` - Generate (start, target) time pair combinations

**Regression**:
- `fit_single_time_pair()` - Single (start, target) Ridge regression + CV
  - Returns: R², MAE, RMSE, MSE, n_embryos, alpha, plus per-genotype MSE
  - **Issue**: Per-genotype R² not being saved properly
- `run_all_time_pairs()` - Loop over all 496 time pairs, collect results

**Visualization**:
- `build_r2_matrix()` - Pivot long-form results to 2D matrix
- `plot_r2_horizons()` - Use horizon_plots.plot_horizon_grid()

**Main**:
- Orchestrates: load → bin → run → save → plot

---

## Known Issues

1. **R² Per-Genotype Not Saving**
   - Computed in `fit_single_time_pair()` but not making it to CSV
   - Likely DataFrame column merging issue in `run_all_time_pairs()`
   - Workaround: Manually compute from fold-level metrics if needed

2. **sklearn Warnings** (harmless)
   - "R² score is not well-defined with less than two samples"
   - Occurs when some genotypes have <2 test samples in a fold
   - Already handled with NaN masking

3. **Negative R² Values**
   - Model performs worse than baseline (predicting mean)
   - Expected for long-range predictions (40h+ gaps)
   - Solution: Focus on shorter time gaps (Δt < 20h)

---

## Next Steps

### Priority 1: Fix Genotype-Specific R² Export
- Debug DataFrame column handling in `run_all_time_pairs()`
- Verify `r2_cep290_wildtype` and `r2_cep290_heterozygous` appear in CSV
- Generate separate R² horizon plots by genotype

### Priority 2: Add Homozygous Genotype
- Once baseline (WT+Het) works, add homozygous embryos (14 embryos)
- Re-run with `train_genotypes = ['cep290_wildtype', 'cep290_heterozygous', 'cep290_homozygous']`
- Compare predictability: Is homozygous more/less predictable?

### Priority 3: Analyze Time Windows
- Focus on Δt = 2-20h (realistic prediction windows)
- Plot R² vs time gap separately for each genotype
- Identify "sweet spot" times (18-30 hpf) for best prediction

### Priority 4: Migration to src/
- Move stable code to `src/analyze/difference_detection/classification/horizon_prediction.py`
- Add to imports/utilities for reuse in other analyses

---

## Key References

**Data Loading**: `src/analyze/trajectory_analysis/data_loading.py`
**Binning**: `src/analyze/utils/binning.py`
**Regression Pattern**: `src/analyze/difference_detection/classification/curvature_regression.py`
**Visualization**: `src/analyze/difference_detection/horizon_plots.py`

---

## Questions for Continuation

1. **Interpretation**: How to interpret negative R² for this phenotype?
2. **Threshold**: What's a "predictable" R² value? (0.3, 0.5, 0.7?)
3. **Homozygous**: Include in training or evaluate separately?
4. **Time Windows**: Focus on early development (12-30 hpf) or full range?

