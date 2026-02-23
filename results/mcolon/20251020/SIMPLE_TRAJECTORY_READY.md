# Simple Trajectory Prediction - Model Comparison

## ✅ IMPLEMENTATION COMPLETE - Ready to Run

Simplified trajectory prediction analysis that focuses on finding the best model for predicting developmental trajectories from a single starting timepoint.

---

## What This Analysis Does

**Core Question:** "Can we predict an embryo's future developmental trajectory from its morphological embedding at 30 hpf?"

### The Experiment

**Training set:** CEP290 homozygous embryos only
**Starting point:** 30 hpf (single fixed timepoint)
**Target:** Predict Euclidean distance from WT at all future times (32, 34, 36... 120+ hpf)
**Validation:** Leave-one-embryo-out cross-validation
**Models tested:** 8 different machine learning approaches

---

## Key Simplifications from Previous Plan

✅ **Single genotype** (Homo only - our end goal for penetrance)
✅ **Single starting time** (30 hpf - no complex FROM×TO matrix)
✅ **Focus on model selection** (find what works best)
✅ **Clear visualizations** (trajectory curves, not dense heatmaps)
✅ **Manageable scope** (can run and interpret in one session)

---

## Models Being Compared

### 1. **Linear Regression** (baseline)
- Simplest possible model
- Assumes linear relationship between embedding and future distance
- Fast, interpretable

### 2. **Ridge Regression** (regularized linear)
- L2 regularization to prevent overfitting
- Good when embedding dimensions are correlated

### 3. **Lasso Regression** (sparse linear)
- L1 regularization - performs feature selection
- Identifies which embedding dimensions matter most

### 4. **Random Forest** (non-linear ensemble)
- Current default in complex plan
- Captures non-linear relationships
- Robust to outliers

### 5. **Gradient Boosting** (non-linear sequential)
- Often better than Random Forest
- Builds trees sequentially, each correcting previous errors

### 6. **XGBoost** (optimized gradient boosting)
- State-of-the-art tree-based method
- Usually best performance for tabular data

### 7. **Support Vector Regression (SVR)** (kernel method)
- Can capture complex patterns via kernel trick
- Good with high-dimensional data

### 8. **Multi-Layer Perceptron (MLP)** (neural network)
- Simple feedforward neural network
- Can learn arbitrary functions
- May be overkill for this problem

---

## How It Works

### Data Structure

For each Homo embryo:
```
At 30 hpf:
  embedding = [dim_0, dim_1, ..., dim_79]  (80 features)

Future timepoints (all available):
  32 hpf → distance = 0.85
  34 hpf → distance = 1.20
  36 hpf → distance = 1.45
  38 hpf → distance = 1.67
  ...
  120 hpf → distance = 3.21

Training examples created:
  (embedding_30hpf) → (distance_32hpf)
  (embedding_30hpf) → (distance_34hpf)
  (embedding_30hpf) → (distance_36hpf)
  ...
```

**Result:** ~40 embryos × ~20 future times = ~800 training examples

### Leave-One-Embryo-Out Validation

```python
For each model type:
  For each embryo E:
    1. Train on all OTHER embryos
    2. Predict embryo E's full trajectory from its 30 hpf embedding
    3. Compute error at each future timepoint
    4. Store predictions

  Aggregate:
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - R² score
    - Error vs time curve
```

**Why LOEO?**
- Tests generalization across individuals
- Prevents overfitting to specific embryo quirks
- More rigorous than random shuffle

---

## Outputs

### 5 Key Visualizations

**1. Model Comparison Curves**
```
X-axis: Future time (32-120 hpf)
Y-axis: Prediction error
Lines: One per model type (8 lines total)

Shows: Which model maintains accuracy longest as we predict further ahead
```

**2. Trajectory Examples (Top 3 Models)**
```
Grid of 6 example embryos:
- Black line: Actual trajectory (ground truth)
- Colored lines: Predicted trajectories from top 3 models
- Red vertical line: Starting point (30 hpf)

Shows: Visual check - can we actually predict the developmental arc?
```

**3. Model Performance Heatmap**
```
Rows: 8 model types
Columns: Future time bins
Color: RED=bad prediction, BLUE=good prediction

Shows: Quick visual comparison of which models work where
```

**4. Error Distribution Boxplots**
```
X-axis: 8 model types
Y-axis: Prediction error
Boxplots showing distribution

Shows: Variability and outliers per model
```

**5. Model Ranking Table**
```
Styled table with:
- Rank (1-8)
- Model name
- MAE, RMSE, R²
- Number of predictions
- Gold/Silver/Bronze highlighting for top 3

Shows: Clear summary of which model wins
```

### Data Files

```
data/penetrance/simple_trajectory/
├── homo_training_pairs.csv              # All (embedding, distance) pairs
├── model_comparison_metrics.csv         # Summary metrics for all models
└── predictions/
    ├── linear_predictions.csv           # LOEO predictions per model
    ├── ridge_predictions.csv
    ├── random_forest_predictions.csv
    ├── xgboost_predictions.csv
    └── ...
```

### Plot Files

```
plots/penetrance/simple_trajectory/
├── model_comparison_over_time.png       # Main result!
├── trajectory_examples_top3.png         # Visual validation
├── model_performance_heatmap.png        # Model × Time heatmap
├── error_distributions.png              # Boxplots
└── model_ranking_table.png              # Summary table
```

---

## How to Run

```bash
# Navigate to directory
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251020

# Run analysis
python3 run_simple_trajectory_homo.py
```

**Runtime:** ~5-15 minutes (depends on number of models and embryos)

---

## What to Look For in Results

### Model Comparison Curves

**Good signs:**
- ✅ Error increases gradually with time (expected degradation)
- ✅ Clear separation between models (some work better)
- ✅ Top model maintains low error longer

**Red flags:**
- ❌ All models equally bad → embeddings don't encode temporal info
- ❌ Random/flat error → no predictive signal
- ❌ Perfect prediction → likely data leakage bug

### Trajectory Examples

**Good signs:**
- ✅ Predicted curves roughly follow actual trajectories
- ✅ Predictions better for near-term (32-40 hpf) than far (80+ hpf)
- ✅ Individual variation captured (some embryos easier to predict)

**Red flags:**
- ❌ Predictions completely miss actual trajectory
- ❌ All predictions converge to same value
- ❌ Negative distances or other impossible values

### Model Ranking

**Expected outcomes:**
- Tree-based models (Random Forest, Gradient Boosting, XGBoost) likely in top 3
- Linear models likely worse (relationship probably non-linear)
- SVR and MLP may be competitive but slower

**Use this to:**
- ✅ Identify best model for future penetrance work
- ✅ Understand if simple or complex model needed
- ✅ Decide if we need hyperparameter tuning

---

## Next Steps After This Analysis

### If a Model Works Well (Low Error, High R²):

**1. Use best model for penetrance detection**
```python
# Train WT model using same best model type
# Compare WT vs Homo model performance on Homo embryos
# Classify penetrant vs non-penetrant based on trajectory fit
```

**2. Extend to multiple starting times**
```python
# Try 40 hpf, 50 hpf, etc.
# See if later start times predict better (more phenotype info)
```

**3. Extend to TMEM67**
```python
# Same analysis on TMEM67 homozygous
# Compare gene-specific trajectory predictability
```

### If No Model Works Well (High Error):

**Possible reasons:**
- Embeddings don't encode much temporal information
- 30 hpf too early (phenotype not emerged yet)
- High individual variability makes prediction hard
- Need different features (not just embeddings)

**What to try:**
- Use later starting time (40-50 hpf)
- Include additional features (time, genotype indicators)
- Try predicting embedding changes instead of distances
- Check if specific embedding dimensions are more predictive

---

## Files Created

### Core Modules

**1. `penetrance_analysis/simple_trajectory.py`** (280 lines)
- `create_start_time_pairs()` - Create (embedding_30, distance_future) pairs
- `get_model()` - Get sklearn model instance by name
- `train_model_loeo()` - Train with leave-one-embryo-out
- `compute_error_by_time()` - Error metrics per target time
- `compare_models()` - Rank all models
- `get_top_models()` - Get top N performers

**2. `penetrance_analysis/simple_viz.py`** (340 lines)
- `plot_model_comparison_curves()` - Error/R² vs time for all models
- `plot_trajectory_examples()` - Example predictions from top models
- `plot_model_performance_heatmap()` - Model × Time heatmap (RED/BLUE)
- `plot_error_distributions()` - Boxplots per model
- `plot_model_ranking_table()` - Styled summary table

**3. `run_simple_trajectory_homo.py`** (265 lines)
- Main orchestration script
- Loads data, trains all models, generates all plots
- Self-contained and easy to modify

### Updated Files

**4. `penetrance_analysis/__init__.py`**
- Exported all new functions

---

## Key Advantages of This Approach

✅ **Simple and focused** - one clear question
✅ **Easy to interpret** - trajectory curves are intuitive
✅ **Systematic** - compares 8 models rigorously
✅ **Fast to run** - ~10 minutes vs hours for complex analysis
✅ **Actionable** - identifies best model for next steps
✅ **No arbitrary choices** - just testing standard ML models

---

## Summary

This is a **clean, focused experiment** that answers:
> "Which machine learning model best predicts future developmental trajectories from morphological embeddings?"

Once we have the answer, we can:
1. Use the best model for penetrance classification
2. Extend to more complex comparisons (WT vs Homo)
3. Explore temporal dynamics in detail

**But first:** Find out what works!

---

**Ready to run!** Just execute:
```bash
python3 run_simple_trajectory_homo.py
```

The script will train 8 models, generate 5 plots, and tell you which model wins.
