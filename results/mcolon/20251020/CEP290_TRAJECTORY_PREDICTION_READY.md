# CEP290 Cross-Genotype Trajectory Prediction Analysis

## ✅ IMPLEMENTATION COMPLETE - Ready to Run

All code has been implemented for CEP290 cross-genotype trajectory prediction using leave-one-embryo-out cross-validation.

---

## What This Analysis Does

**Goal**: Test if morphological embeddings at time `i` contain information to predict Euclidean distance from WT at future time `i+k`.

**Key Question**: How much temporal trajectory information is encoded in the embeddings?

**Approach**:
- Train separate models for each genotype (WT, Het, Homo)
- Test each model on all genotypes (cross-genotype generalization)
- Use leave-one-embryo-out (LOEO) cross-validation for rigor

---

## Training & Testing Strategy

### Models Trained (up to 3)
1. **CEP290_WT model** - learns "normal" developmental trajectory
2. **CEP290_Het model** - learns heterozygous trajectory (if data exists)
3. **CEP290_Homo model** - learns "mutant" developmental trajectory

### Testing Matrix (up to 3×3 = 9 combinations)

```
              Test on →
Model ↓       WT       Het      Homo
─────────────────────────────────────
WT          LOEO      Full     Full
Het         Full      LOEO     Full
Homo        Full      Full     LOEO
```

**Diagonal** (e.g., WT model on WT data): LOEO cross-validation
**Off-diagonal** (e.g., WT model on Homo data): Full model prediction

---

## Key Features

### Data Handling
✅ Loads raw experiments using existing `utils/data_loading.py`
✅ Bins embeddings using `utils/binning.py` (2 hpf bins)
✅ **Uses ALL available data** (no arbitrary time filtering)
✅ Merges with precomputed Euclidean distances
✅ Drops rows with missing distances (avoids NaN errors)
✅ **Gracefully skips genotypes with no data** (e.g., Het if no distances available)

### Heatmap Structure

**Y-axis**: FROM time (starting time `i`)
**X-axis**: TO time (target time `i+k`)
**Triangular**: Only upper triangle (can't predict backwards in time)
**Aggregation**: Mean error across all embryos

```
Example heatmap:
              30   32   34   36   38   40   hpf (TO)
FROM 30 hpf   ·   0.1  0.2  0.3  0.5  0.7
FROM 32 hpf        ·   0.1  0.2  0.4  0.6
FROM 34 hpf             ·   0.1  0.3  0.5
FROM 36 hpf                  ·   0.2  0.4
```

### Outputs Per Model-Test Combination

**7 plots each:**
1. Aggregated heatmap (absolute error)
2. Aggregated heatmap (relative error)
3. Aggregated heatmap (R²)
4. Per-embryo grid (small heatmaps for each embryo)
5. Error vs prediction horizon (Δt)
6. Temporal breakdown (error distribution by FROM time)
7. Per-embryo error distribution

### Additional Outputs

- **3×3 model comparison plot** - overview of all combinations
- **Penetrance classification** - WT model vs Homo model on Homo embryos
- **Performance summary CSV** - metrics for all combinations

---

## Penetrance Classification

Compares WT model vs Homo model performance on **Homo embryos only**:

```python
For each Homo embryo:
    error_ratio = mean_error_wt_model / mean_error_homo_model

    If ratio > 1.5:
        → PENETRANT
        (WT model fails, Homo model succeeds → follows mutant trajectory)

    If ratio < 0.67:
        → NON-PENETRANT
        (WT model succeeds, Homo model fails → follows WT-like trajectory)

    Else:
        → INTERMEDIATE
```

**This gives individual-level penetrance classification based on trajectory dynamics!**

---

## How to Run

```bash
# Navigate to directory
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251020

# Run analysis
python3 run_trajectory_loeo_cep290.py
```

**Runtime**: ~10-30 minutes depending on data size and number of genotypes with data

---

## Expected Outputs

### Data Files
```
data/penetrance/trajectory_loeo/cep290/
├── wt_model/
│   ├── predictions_on_wt.csv
│   ├── predictions_on_het.csv      (if het has data)
│   └── predictions_on_homo.csv
│
├── het_model/                       (if het has data)
│   └── ...
│
├── homo_model/
│   ├── predictions_on_wt.csv
│   ├── predictions_on_het.csv      (if het has data)
│   └── predictions_on_homo.csv
│
├── penetrance_classification.csv   ← PENETRANCE RESULTS
└── model_performance_summary.csv
```

### Plots
```
plots/penetrance/trajectory_loeo/cep290/
├── wt_model/
│   ├── tested_on_wt/
│   │   ├── aggregated_heatmap_absolute_error.png
│   │   ├── aggregated_heatmap_relative_error.png
│   │   ├── aggregated_heatmap_r2.png
│   │   ├── per_embryo_grid.png
│   │   ├── error_vs_horizon.png
│   │   ├── temporal_breakdown.png
│   │   └── per_embryo_error_distribution.png
│   │
│   ├── tested_on_het/  (if het has data)
│   └── tested_on_homo/
│
├── homo_model/
│   └── ... (same structure)
│
├── model_comparison_3x3.png        ← OVERVIEW OF ALL COMBINATIONS
└── penetrance_classification.png   ← PENETRANCE VISUALIZATION
```

---

## Interpretation Guide

### 1. Diagonal Performance (Within-Genotype LOEO)

**Low error** → Embeddings encode rich temporal dynamics
**High error** → Limited temporal information OR high individual variability

**Compare WT vs Homo:**
- WT should have consistent, low error (stereotyped normal development)
- Homo may have higher error if incomplete penetrance causes variability

### 2. WT Model on Homo Embryos (Key for Penetrance!)

**Individual embryo heatmaps reveal penetrance:**

- **Non-penetrant embryos**: Low error (WT model predicts well)
  - These embryos follow WT-like developmental trajectories

- **Penetrant embryos**: High error (WT model fails)
  - These embryos deviate from WT trajectory

**Per-embryo error distribution plot** shows bimodal pattern if incomplete penetrance exists!

### 3. Error vs Prediction Horizon

Shows how prediction quality degrades with time:

- **Flat curve**: Long-term trajectory encoded
- **Sharp increase**: Limited "memory horizon"
- **Jumps at specific Δt**: Developmental transitions

### 4. Temporal Breakdown

Shows which developmental stages are most/least predictable:

- **Low variance**: Stereotyped developmental stage
- **High variance**: Variable/transitional stage

---

## Key Insights This Will Reveal

### Embedding Quality
✅ Do embeddings encode temporal dynamics?
✅ How far ahead can we predict (memory horizon)?
✅ Are there critical developmental transitions?

### Genotype Differences
✅ Is WT development more stereotyped than mutant?
✅ Do Het embryos follow WT or Homo trajectories?
✅ How variable is Homo development?

### Penetrance Detection
✅ Can we identify non-penetrant Homo embryos?
✅ Is penetrance binary or continuous?
✅ When does penetrance become detectable?

---

## Files Created

### Core Modules
1. **`penetrance_analysis/trajectory_loeo.py`** (617 lines)
   - `create_trajectory_pairs()` - Create (i → i+k) prediction examples
   - `train_loeo_and_full_model()` - LOEO training + full model
   - `test_model_on_genotype()` - Cross-genotype testing
   - `compute_overall_metrics()` - Performance metrics
   - `compute_per_embryo_metrics()` - Individual embryo metrics
   - `compute_error_vs_horizon()` - Error vs Δt analysis
   - `classify_penetrance_dual_model()` - WT vs Homo comparison

2. **`penetrance_analysis/trajectory_viz_loeo.py`** (548 lines)
   - `create_aggregated_heatmap()` - FROM×TO error matrix
   - `create_per_embryo_heatmaps()` - Individual embryo matrices
   - `compute_r2_per_cell()` - R² for each (FROM, TO) cell
   - `plot_aggregated_heatmap()` - Main heatmap visualization
   - `plot_per_embryo_grid()` - Grid of individual heatmaps
   - `plot_error_vs_horizon()` - Error vs Δt curves
   - `plot_temporal_breakdown()` - Error by FROM time
   - `plot_per_embryo_error_distribution()` - Individual variability
   - `plot_model_comparison_3x3()` - 3×3 overview grid
   - `plot_penetrance_classification()` - Classification visualization

3. **`run_trajectory_loeo_cep290.py`** (456 lines)
   - Main orchestration script
   - Loads data using existing utilities
   - Coordinates all training, testing, and visualization

### Updated Files
4. **`penetrance_analysis/__init__.py`**
   - Exported all new functions

---

## Fixes Applied

### Issue 1: Missing Embedding Columns
**Problem**: Distances CSV doesn't contain embeddings
**Solution**: Load raw experiments → bin embeddings → merge with distances

### Issue 2: Arbitrary Time Filtering
**Problem**: Originally filtered to ≥30 hpf
**Solution**: Use ALL available data (no minimum time)

### Issue 3: NaN Distances
**Problem**: sklearn error "Input y contains NaN"
**Solution**: Drop rows with missing distances after merging

### Issue 4: Missing Genotypes
**Problem**: Script crashes if Het has no distance data
**Solution**: Gracefully skip genotypes with no trajectory pairs

---

## Next Steps

1. **Run the analysis**: `python3 run_trajectory_loeo_cep290.py`

2. **Review outputs**:
   - Check model performance summary
   - Examine 3×3 comparison plot
   - Look at penetrance classification results

3. **Dive into specific combinations**:
   - WT model on Homo: Non-penetrant detection
   - Homo model on Homo: Baseline performance
   - Per-embryo heatmaps: Individual variation

4. **If results look good, extend to TMEM67**:
   - Copy script and change `GENE = 'tmem67'`
   - TMEM67 should show higher penetrance (less variability)

---

## Contact

Implementation completed from context-limited conversation.

All code ready to run - just execute the main script!

**Key output to examine first**:
- `penetrance_classification.csv` - Individual embryo classifications
- `model_comparison_3x3.png` - Overall performance overview
- `wt_model/tested_on_homo/per_embryo_grid.png` - Individual Homo embryo trajectories
