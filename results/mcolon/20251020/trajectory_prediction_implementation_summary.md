# Trajectory Prediction Implementation Summary

**Status**: Module created, visualizations added, ready for main script

---

## What Was Implemented

### 1. Core Module: `trajectory_prediction.py`

**Functions created**:
- `prepare_trajectory_data()`: Converts binned data into training examples for prediction
- `train_trajectory_model()`: Trains Random Forest/Linear/GB model on genotype-specific data
- `predict_with_model()`: Makes predictions using trained model
- `cross_validate_trajectory_model()`: Leave-one-embryo-out CV
- `predict_all_trajectories()`: Batch prediction for all embryos
- `compute_per_embryo_prediction_metrics()`: Aggregate errors per embryo
- `classify_penetrance_by_dual_models()`: Compare homo vs WT model performance
- `detect_penetrance_onset()`: Find when error ratio crosses threshold

### 2. Visualization Functions (added to `visualization.py`)

- `plot_dual_prediction_heatmaps()`: 3-panel heatmap (homo error, WT error, ratio)
- `plot_trajectory_examples()`: Gallery of example embryo trajectories with predictions
- `plot_prediction_error_scatter()`: Homo error vs WT error scatter plot
- `plot_penetrance_distribution()`: Bar chart + histogram of classifications

---

## Next Steps

### Create Main Script: `run_penetrance_prediction.py`

The script needs to:

1. **Load Data**
   - Load binned embryo data (from Step 1 or earlier)
   - Filter to appropriate time range (≥30 hpf)

2. **Prepare Training Data**
   - For WT embryos: create prediction examples
   - For homozygous embryos: create prediction examples
   - Prediction horizons: [2, 4, 6, 8] hpf

3. **Train Models**
   - Model 2a: Train on homozygous data
   - Model 2b: Train on WT data
   - Cross-validate each

4. **Make Predictions**
   - Predict each homozygous embryo with BOTH models
   - Compute prediction errors

5. **Classify Penetrance**
   - Compare homo vs WT model errors
   - Error ratio > 1.5 → penetrant
   - Error ratio < 0.67 → non-penetrant

6. **Generate Outputs**
   - CSV: predictions, classification, metrics
   - Plots: heatmaps, trajectories, scatter, distribution

---

## Expected Outputs

### Data Files
```
data/penetrance/trajectory/
├── cep290_wt_training_data.csv
├── cep290_homo_training_data.csv
├── cep290_wt_model_predictions.csv
├── cep290_homo_model_predictions.csv
├── cep290_penetrance_classification.csv
├── tmem67_wt_training_data.csv
├── tmem67_homo_training_data.csv
├── tmem67_wt_model_predictions.csv
├── tmem67_homo_model_predictions.csv
└── tmem67_penetrance_classification.csv
```

### Plots
```
plots/penetrance/trajectory/
├── cep290_dual_heatmaps.png
├── cep290_trajectory_examples.png
├── cep290_error_scatter.png
├── cep290_penetrance_distribution.png
├── tmem67_dual_heatmaps.png
├── tmem67_trajectory_examples.png
├── tmem67_error_scatter.png
└── tmem67_penetrance_distribution.png
```

---

## Testing Plan

1. **Start with CEP290** (harder case)
   - Should see ~40-50% penetrant
   - Should match temporal analysis results

2. **Then TMEM67** (easier case)
   - Should see ~80-90% penetrant
   - Strong signal throughout

3. **Validate**:
   - Do penetrance estimates match other methods?
   - Are trajectory plots interpretable?
   - Does CV show reasonable performance?

---

## Implementation Status

- ✅ Core prediction module created
- ✅ Visualization functions added
- ⏳ Main script (next step)
- ⏳ __init__.py update (next step)
- ⏳ Testing (after script created)

---

Ready to create the main script when you are!
