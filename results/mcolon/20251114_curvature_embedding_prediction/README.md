# Curvature-Embedding Prediction Analysis

## Goal 1: How much can we predict curvature from embeddings?

This analysis quantifies the relationship between morphological embeddings (from build06) and curvature metrics using ridge regression with leave-one-embryo-out (LOEO) cross-validation.

## Research Questions

1. **Primary**: How much of the curvature variation can be explained by morphological embeddings?
2. **Secondary**: Which embedding dimensions are most predictive of curvature?
3. **Extended**: Can we predict future curvature early (e.g., at 16hpf)?

## Analysis Structure

### Utils Package (Reusable Components)

The `utils/` directory contains modular functions designed to be migrated to `src/analyze/difference_detection/classification/` after validation:

#### `regression.py`
- Core regression models (Ridge, GradientBoosting, Linear, Lasso, RandomForest)
- `train_regression_model_loeo()`: Train with leave-one-embryo-out CV
- `predict_with_trained_model()`: Make predictions with trained model

#### `evaluation.py`
- `compute_regression_metrics()`: R², RMSE, MAE, MAPE
- `get_feature_importance()`: Extract feature importance from models
- `compare_multiple_models()`: Side-by-side model comparison
- `compute_prediction_error_by_group()`: Error breakdown by genotype

#### `data_prep.py`
- `prepare_features_and_target()`: Extract and validate features/targets
- `validate_data_completeness()`: Check for NaN, data types, etc.
- `filter_by_genotype()`: Filter to specific genotypes
- `rename_embeddings_to_standard()`: Standardize embedding column names

#### `plotting.py`
- `plot_predictions_vs_actual()`: Scatter + residual panels
- `plot_residuals()`: Diagnostic residual plots (histogram, Q-Q, etc.)
- `plot_feature_importance()`: Bar plot of top features
- `plot_model_comparison()`: Multi-panel model performance comparison
- `plot_metrics_table()`: Text table of metrics

### Main Scripts

#### `predict_curvature_from_embeddings.py`
Main analysis script. Runs:
1. Load build06 df03 data and curvature metrics
2. Train ridge + gradient boosting models for each curvature metric
3. Perform LOEO cross-validation for honest evaluation
4. Extract feature importance
5. Generate visualizations and summary tables

#### `config.py`
Configuration file with:
- Data paths
- Target curvature metrics
- Embedding pattern
- Model hyperparameters
- Analysis settings

## Running the Analysis

```bash
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251114_curvature_embedding_prediction

python predict_curvature_from_embeddings.py
```

## Outputs

### Tables (`outputs/tables/`)
- `model_comparison.csv` - Overall and per-genotype metrics for all models
- `feature_importance_*.csv` - Top features for each metric+model combination

### Figures (`outputs/figures/`)
- `predictions_vs_actual_*.png` - Predicted vs actual scatter plots (by model)
- `residuals_*.png` - Residual diagnostics (histogram, Q-Q, temporal patterns)
- `feature_importance_*.png` - Top 15 features by importance
- `model_comparison.png` - R², MAE, RMSE comparison across models

## Key Findings

(To be filled in after running analysis)

### Ridge Regression (Primary Model)
- **arc_length_ratio**: R² = ?, MAE = ?
- **normalized_baseline_deviation**: R² = ?, MAE = ?

### Gradient Boosting (Secondary Model)
- **arc_length_ratio**: R² = ?, MAE = ?
- **normalized_baseline_deviation**: R² = ?, MAE = ?

### Most Predictive Embedding Dimensions
(Ranked by feature importance)

## Future Directions

### Goal 2: Curvature vs Distance in Morphology Space
Quantify how much curvature explains variation in distance from WT in embedding space.

### Goal 3: Early Prediction of Future Curvature
Train temporal models to predict future curvature from early timepoints (e.g., predict curvature at 45hpf from embedding at 16hpf).

## Moving Utils to Classification Module

Once validated, functions from `utils/` can be migrated to:
```
src/analyze/difference_detection/classification/
```

This will provide reusable regression utilities for the broader pipeline.

## Dependencies

- pandas, numpy
- scikit-learn (regression models, metrics)
- matplotlib, seaborn (visualization)
- joblib (parallelization)

## Related Analyses

- `results/mcolon/20251020/`: Penetrance analysis (identified ridge regression as effective)
- `results/mcolon/20251029_curvature_temporal_analysis/`: Curvature metrics and embeddings
