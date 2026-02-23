# B9D2 Phenotype Comparison Analysis

**Date:** 2025-12-28

## Goal

Compare b9d2 phenotypes (CE, HTA, BA-rescue) using difference detection to test the hypothesis that HTA and BA-rescue are the same underlying phenotype that diverges after 60 hpf.

## Phenotypes

| Phenotype | File | N embryos | Description |
|-----------|------|-----------|-------------|
| CE | `b9d2-CE-phenotype.txt` | 53 | Convergent Extension defect |
| HTA | `b9d2-HTA-embryos.txt` | 25 | Head-Trunk Angle (appears after 60 hpf) |
| BA-rescue | `b9d2-curved-rescue.txt` | 7 | Starts curved, then straightens |
| Wildtype | genotype = `b9d2_wildtype` | 35 | Control group |

## Comparisons

1. **CE vs Wildtype** - Detect when CE phenotype becomes predictable
2. **HTA vs Wildtype** - Detect when HTA becomes predictable
3. **BA-rescue vs Wildtype** - Detect when BA-rescue becomes predictable
4. **HTA vs BA-rescue** - Test if they're distinguishable (key hypothesis test)
5. **CE vs HTA** - Compare the two distinct phenotype classes

## Methods

### Data
- Experiments: 20251121, 20251125
- VAE embeddings: Only `z_mu_b` columns (80 biological features, NOT `z_mu_n` non-biological)
- Time binning: Configurable (2 or 4 hour bins)

### Difference Detection
- AUROC-based classification with logistic regression
- Permutation testing for p-values (label shuffling)
- Cross-validation (stratified k-fold)

### Visualization
- **Panel A**: AUROC over time with significance coloring
  - Green filled circles = p < 0.05 (significant)
  - Gray open circles = p >= 0.05 (not significant)
- **Panel B**: Morphological divergence (phenotype-specific metric)
  - CE: `total_length_um`
  - HTA/BA-rescue: `baseline_deviation_normalized`
- **Panel C**: Individual trajectories with group means

## Key Implementation Details

### No Extrapolation Fix
The interpolation only uses values within each embryo's actual time range to prevent edge-value contamination of group means.

### Speed Optimizations (configurable)
- `N_SPLITS`: CV folds (default 5, can reduce to 3)
- `N_PERM`: Permutations (default 100, can reduce to 25-50)
- `BIN_WIDTH`: Time bin size (default 2 hours, can increase to 4)

### P-value Interpretation
With N_PERM=50, minimum p-value = 1/51 ≈ 0.020
With N_PERM=100, minimum p-value = 1/101 ≈ 0.010

## Output Structure

```
output/
├── classification_results/
│   ├── CE_vs_wildtype.csv
│   ├── HTA_vs_wildtype.csv
│   ├── BA_rescue_vs_wildtype.csv
│   ├── HTA_vs_BA_rescue.csv
│   └── CE_vs_HTA.csv
└── figures/
    ├── CE_vs_wildtype_comprehensive.png
    ├── HTA_vs_wildtype_comprehensive.png
    ├── BA_rescue_vs_wildtype_comprehensive.png
    ├── HTA_vs_BA_rescue_comprehensive.png
    └── CE_vs_HTA_comprehensive.png
```

## Usage

```bash
cd results/mcolon/20251228_b9d2_phenotype_comparisons
python b9d2_phenotype_comparison.py
```

## Files

- `b9d2_phenotype_comparison.py` - Main analysis script (5 original comparisons)
- `b9d2_phenotype_comparison_extended.py` - Extended analysis script (13 additional comparisons)
- `README.md` - This file

---

## 🎯 Current Status & Handoff

### What's Complete

✅ **Core infrastructure built:**
- Main script (`b9d2_phenotype_comparison.py`) with:
  - Phenotype file parsing (CE, HTA, BA-rescue)
  - Data loading (VAE embeddings z_mu_b only, no extrapolation)
  - AUROC-based difference detection with permutation testing
  - 3-panel figures with **error bars** + **significance coloring** (green=p<0.05, gray=p≥0.05)
  - Progress tracking for each time bin

✅ **Extended script created** (`b9d2_phenotype_comparison_extended.py`):
- **Set 1**: Within-pair CE validation (pair_7, pair_8)
  - CE vs Het (non-penetrant)
  - CE vs Wildtype
  - Het vs Wildtype
- **Set 2**: Negative controls (should show NO difference)
  - pair_2 non-penetrant hets vs pair_8 non-penetrant hets
  - pair_2 WT vs pair_8 WT
  - WT split-in-half (within 20251125 experiment)
- **Set 3**: HTA/BA-rescue analysis (pooled across pairs)
  - HTA vs non-penetrant Het
  - HTA vs non-penetrant WT
  - BA-rescue vs non-penetrant Het
  - BA-rescue vs non-penetrant WT
  - *(HTA vs BA-rescue skipped - already in main script)*

✅ **Naming clarified** to prevent confusion:
- All control groups labeled `_nonpen_ctrl` or `_nonpen` to indicate "excluded from phenotype lists"
- Filenames clearly show what's being compared
- Comments in code explain filtering logic

### What's Ready to Run

The extended script is fully functional and ready to execute. It will:
1. Load phenotypes and raw data from experiments 20251121 & 20251125
2. Run all 13 comparisons
3. Generate CSV results and 3-panel figures
4. Save to `output_extended/` directory

### Key Design Decisions

| Aspect | Choice | Reason |
|--------|--------|--------|
| Embeddings | z_mu_b only (80 dims) | Biological features, exclude z_mu_n (non-biological) |
| Interpolation | No extrapolation | Prevents edge-value contamination of group means |
| P-values | Permutation test | Label shuffling tests if discrimination is due to signal, not chance |
| AUROC display | Green circles + error bars | Shows significance AND uncertainty simultaneously |
| Controls | Non-penetrant embryos | Excluded from phenotype lists, not biologically affected |

### Common Pitfalls (Avoided)

- ❌ Mixed CE pair_8 embryos with non-penetrant controls → ✅ Filtered by `all_phenotype_ids` set
- ❌ Used all 100 VAE dims (z_mu_n + z_mu_b) → ✅ Only z_mu_b (80 dims)
- ❌ Extrapolated outside embryo's time range → ✅ Only use actual measurements per embryo
- ❌ Confusing naming of controls → ✅ All labeled `_nonpen_ctrl` or `_nonpen`

---

## References

- Phenotype lists from: `results/mcolon/20251219_b9d2_phenotype_extraction/phenotype_lists/`
- Similar analysis pattern from: `results/mcolon/20251210_b9d2_earliest_prediction_analysis/`
- Difference detection module: `src/analyze/difference_detection/`

---

## Cross-Experiment Validation Analysis (NEW - 2025-12-29)

✅ **Cross-Experiment Generalization Test**: Models trained on one experiment predict on the other

**Key Finding**: Strong generalization across experiments - phenotype signals are NOT experiment-specific artifacts

**Results Summary**:
- **CE vs Wildtype**: Excellent generalization
  - Within-exp AUROC: 0.94-1.00 (avg ~0.98)
  - Cross-exp AUROC: 0.92-1.00 (avg ~0.97)
  - Generalization gap: **Only ~0.01** → models learn real biological signal

- **HTA vs Wildtype**: Moderate within-exp, variable cross-exp
  - Within-exp AUROC: 0.59-0.93 (more variable than CE)
  - Cross-exp AUROC: 0.51-0.92 (some experiments diverge at late timepoints)
  - Generalization gap: **~0.05-0.20** at some timepoints → experiment-specific variation in HTA phenotype

- **BA-rescue vs Wildtype**: Insufficient data
  - Only 7 embryos total (2 in exp1, 5 in exp2)
  - Skipped due to sample size requirements (min 5 per group per exp)

**Script**: `b9d2_cross_experiment_validation.py`
- Uses k-fold CV ensemble: trains K=5 models on experiment A, predicts on experiment B
- Permutation testing for p-values (N_PERM=100)
- Time-resolved: separate analysis per 4-hour developmental bin

**Output**:
- `cross_exp_output/classification_results/`: CSV files with AUROC, p-values, sample counts
- `cross_exp_output/figures/`: 2-panel plots (AUROC over time + generalization gap)

**Interpretation**:
- **CE phenotype**: Robust across experiments - this is a reliable marker
- **HTA phenotype**: More experiment-dependent variation - suggests biological or experimental factors influence penetrance
- **Next step**: Examine pair-level breakdown to understand HTA variation

---

## Set 4: Pair-Specific Het Phenotype Validation (CRITICAL)

**Purpose**: Prove that non-penetrant hets have a detectable phenotype, but wildtype controls do NOT.

**Why This Matters**: Without showing WTs don't have the het phenotype, reviewers won't believe the signal is real.

**Comparisons**:
- **(e) pair_2 het vs pair_2 WT** - Het phenotype detection in pair_2 (NEW - added 2026-01-05)
- **(c) pair_8 het vs pair_8 WT** - Het phenotype detection in pair_8
- **(d) pair_2 WT vs pair_8 WT** - Negative control (should show NO difference)

**Expected Results**:
- (e) and (c): AUROC > 0.6, p < 0.05 → hets have phenotype
- (d): AUROC ~ 0.5, p > 0.05 → WTs are good controls (no phenotype)

**Run Script**:
```bash
python run_set4_only.py
```

**Outputs**:
- `output_extended/classification_results/pair2_het_nonpen_vs_WT.csv` (NEW)
- `output_extended/figures/pair2_het_nonpen_vs_WT_comprehensive.png` (NEW)
- `output_extended/classification_results/pair8_het_nonpen_vs_WT.csv`
- `output_extended/figures/pair8_het_nonpen_vs_WT_comprehensive.png`
- `output_extended/classification_results/NEGATIVE_pair2_WT_vs_pair8_WT.csv`
- `output_extended/figures/NEGATIVE_pair2_WT_vs_pair8_WT_comprehensive.png`

---

## Next Steps

To continue this analysis:

1. **Run Set 4 validation** (PRIORITY):
   ```bash
   python run_set4_only.py
   ```
   This generates the critical het vs WT validation plots.

2. **Faceted plots by pair and genotype**:
   ```bash
   python b9d2_phenotype_distribution_by_pair.py
   ```
   - Shows how embryos distribute across pairs for each phenotype
   - Separates wildtype (grey) from mutant genotypes
   - Helps identify pair-specific biases

3. **Run extended script**:
   ```bash
   python b9d2_phenotype_comparison_extended.py
   ```
   Outputs will go to `output_extended/classification_results/` and `output_extended/figures/`

4. **Interpret results**:
   - Negative controls (Set 2 & comparison d) should show **low AUROC** (~0.5) and **high p-values** (>0.05)
   - Phenotype comparisons (Sets 1, 3, & 4) should show **high AUROC** and **low p-values** (<0.05)
   - **Green circles** = signal is real; **Gray circles** = likely noise

5. **Validate hypothesis**:
   - If HTA vs BA-rescue shows **high AUROC**, they're different phenotypes
   - If HTA vs BA-rescue shows **low AUROC**, they may be the same phenotype diverging post-60hpf
