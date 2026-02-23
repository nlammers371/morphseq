# Trajectory Prediction Implementation - Ready to Run

## Status: ✅ COMPLETE - Ready for Testing

All code for the dual-model trajectory prediction framework has been implemented and is ready to run.

## What Was Implemented

### 1. Core Modules

**`penetrance_analysis/trajectory_prediction.py`**
- Data preparation: Convert binned embeddings to prediction examples (at time i, predict distance at i+k)
- Model training: Random Forest regression on genotype-specific trajectories
- Cross-validation: Leave-one-embryo-out to prevent overfitting
- Dual-model prediction: Predict each homozygous embryo with both models
- Classification: Compare error ratios to identify penetrant vs non-penetrant

**`penetrance_analysis/visualization.py`** (appended)
- `plot_dual_prediction_heatmaps()`: 3-panel comparison of homo/WT errors and ratio
- `plot_trajectory_examples()`: Gallery of actual vs predicted trajectories
- `plot_prediction_error_scatter()`: Homo error vs WT error scatter plot
- `plot_penetrance_distribution()`: Classification summary histograms

**`penetrance_analysis/__init__.py`** (updated)
- Exported all trajectory prediction functions
- Exported all trajectory visualization functions

### 2. Main Script

**`run_penetrance_prediction.py`**
- Orchestrates full workflow for CEP290 and TMEM67
- Loads binned data for homozygous and WT embryos
- Trains Model 2a (homozygous trajectory) and Model 2b (WT trajectory)
- Predicts each homozygous embryo with both models
- Classifies penetrance based on error ratio
- Generates 4 visualizations per genotype
- Saves all results to CSV

**Fixed Issues:**
- ✅ Column name: Script now correctly uses `pred_prob_mutant` instead of `predicted_probability`
- ✅ Import paths: All modules properly exported in `__init__.py`

## How to Run

```bash
# Activate your conda environment
conda activate segmentation_grounded_sam

# Navigate to the directory
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251020

# Run the script
python3 run_penetrance_prediction.py
```

## Expected Outputs

### Data Files (saved to `data/penetrance/trajectory/`)
- `{gene}_homo_training_data.csv` - Training examples for homozygous model
- `{gene}_wt_training_data.csv` - Training examples for WT model
- `{gene}_homo_model_predictions.csv` - Predictions using homozygous model
- `{gene}_wt_model_predictions.csv` - Predictions using WT model
- `{gene}_penetrance_classification.csv` - **Final penetrance classification**

### Plots (saved to `plots/penetrance/trajectory/`)
- `{gene}_dual_heatmaps.png` - 3-panel heatmap comparing model errors
- `{gene}_trajectory_examples.png` - Example trajectories (penetrant/non-penetrant/intermediate)
- `{gene}_error_scatter.png` - Scatter plot of prediction errors
- `{gene}_penetrance_distribution.png` - Classification distribution

## Classification Logic

```
error_ratio = mean_error_wt / mean_error_homo

If error_ratio > 1.5:
    → "penetrant" (follows mutant trajectory, WT model fails)

If error_ratio < 0.67:
    → "non-penetrant" (follows WT trajectory, homo model fails)

Otherwise:
    → "intermediate" (ambiguous)
```

## Key Parameters

```python
PREDICTION_HORIZONS = [2, 4, 6, 8]  # hpf ahead to predict
MIN_TIME = 30.0                      # Filter to ≥30 hpf (post-onset)
MODEL_TYPE = 'random_forest'         # Random Forest regression
ERROR_RATIO_THRESHOLD = 1.5          # Penetrant cutoff
```

## Next Steps After Running

1. **Review classification distributions**: What % of embryos are penetrant vs non-penetrant?
2. **Inspect heatmaps**: Are there temporal patterns in prediction errors?
3. **Examine trajectory examples**: Do penetrant embryos visually follow mutant trajectory?
4. **Compare CEP290 vs TMEM67**: Does TMEM67 show higher penetrance (as expected from R²)?
5. **Validate against manual labels** (if available): How accurate are the classifications?

## Advantages Over Static Cutoffs

✅ **Data-driven**: No arbitrary probability thresholds
✅ **Temporal dynamics**: Considers developmental trajectories
✅ **Genotype-specific**: Learns what "normal" and "mutant" look like
✅ **Individual-level**: Classifies each embryo based on its own trajectory
✅ **Interpretable**: Error ratio directly shows which model fits better

## Troubleshooting

If you encounter errors, check:
1. ✅ Data files exist in `data/penetrance/` (cep290_distances.csv, etc.)
2. ✅ Conda environment is activated
3. ✅ Column names match (script expects `pred_prob_mutant`)
4. ✅ Sufficient data for each genotype (need WT + homozygous)

---

**Implementation completed from conversation session that ran out of context.**

All code is ready - just activate your environment and run the script!
Fixed: Removed invalid 'sort_by' parameter from plot_dual_prediction_heatmaps() call
