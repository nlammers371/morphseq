# Curvature Temporal Analysis - Implementation Complete

**Date**: October 29, 2025
**Status**: All analysis scripts created and ready to run
**Dataset**: 20201017_combined (93 embryos, 3,373 timepoints)

## Overview

Complete implementation of curvature temporal analysis pipeline comparing homozygous, heterozygous, and wildtype embryos across four complementary analyses.

## Files Created

### Core Data Module
- **`load_data.py`** - Single source of truth for data loading
  - Loads curvature summary + embedding metadata
  - Merges on snip_id
  - Normalizes baseline deviation by embryo length
  - Auto-detects embedding columns
  - Exports cleaned dataframe + metadata dict

### Analysis Scripts

1. **`01_individual_trajectories.py`** - Temporal trajectories
   - Plots curvature over development (arc_length_ratio, normalized_baseline_deviation)
   - Individual embryo trajectories (one plot per embryo)
   - Aggregate genotype comparison (all three genotypes side-by-side)
   - Statistical tests (ANOVA + pairwise t-tests)
   - Output: Individual PDFs + comparison figures + summary tables

2. **`02_horizon_plots.py`** - Temporal correlation heatmaps
   - Computes timepoint × timepoint correlation matrices
   - Shows how curvature at one time relates to other times
   - Lagged autocorrelation analysis
   - Aggregate correlation matrices per genotype
   - Output: Heatmap figures + correlation CSV matrices

3. **`03_embedding_distance.py`** - Morphology space analysis
   - Computes pairwise embedding distances in latent space
   - Computes pairwise curvature differences
   - Correlates the two (does morphology distance match curvature difference?)
   - Creates trajectory plots in embedding space colored by curvature
   - Genotype-specific analysis
   - Output: Scatter plots + trajectory plots + correlation statistics

4. **`04_predictive_models.py`** - ML prediction models
   - **Embeddings → Curvature**: Can we predict curvature from morphology?
   - **Curvature → Embeddings**: Can we reconstruct morphology from curvature?
   - Uses leave-one-embryo-out (LOEO) cross-validation
   - Random Forest regressors with feature importance
   - Output: Performance metrics + residual plots + feature importance tables

## Data Flow

```
load_data.py (load_analysis_dataframe)
    ↓
    [Returns merged + normalized dataframe + metadata dict]
    ↓
    ├→ 01_individual_trajectories.py
    ├→ 02_horizon_plots.py
    ├→ 03_embedding_distance.py
    └→ 04_predictive_models.py
```

## Key Features

### Robust Data Handling
- Uses `predicted_stage_hpf` for temporal alignment (accounts for developmental rate variation)
- Normalizes baseline_deviation by embryo length for cross-embryo comparison
- Automatic detection of embedding dimensions
- Proper handling of NaN values throughout

### Three Genotype Comparison
- Wildtype (WT): Control group
- Heterozygous (Het): Intermediate phenotype
- Homozygous (Homo): Severe phenotype

Color scheme maintained throughout (WT=blue, Het=orange, Homo=red)

### Statistical Rigor
- Leave-one-embryo-out cross-validation (avoids data leakage)
- Accounts for repeated measures (multiple timepoints per embryo)
- Spearman correlation for non-linear relationships
- Multiple comparisons where appropriate

### Utility Library Integration
Scripts are designed to leverage:
- `src/analyze/difference_detection/horizon_plots.py` (when implemented)
- `src/analyze/difference_detection/time_matrix.py` (when implemented)

Currently uses direct implementation but imports commented out, ready to switch.

## Output Organization

```
outputs/
├── figures/
│   ├── trajectory_*.png                    # Individual embryo plots (01)
│   ├── aggregate_trajectories_*.png        # Genotype comparisons (01)
│   ├── horizon_comparison_*.png            # Correlation heatmaps (02)
│   ├── embedding_vs_curvature_*.png        # Scatter plots (03)
│   ├── embedding_trajectories_*.png        # Embedding space paths (03)
│   ├── model_performance.png               # Performance comparison (04)
│   └── residuals.png                       # Prediction errors (04)
├── tables/
│   ├── summary_statistics.csv              # Summary stats (01)
│   ├── statistical_tests.csv               # ANOVA results (01)
│   ├── correlation_matrix_*.csv            # Correlation matrices (02)
│   ├── embedding_curvature_correlations.csv# Embedding-curve correlations (03)
│   ├── feature_importance_*.csv            # Feature importance (04)
│   ├── model_performance_*.csv             # Model metrics (04)
│   └── ...
└── models/
    └── [Trained model objects for future use]
```

## How to Run

### Run all analyses in sequence:
```bash
cd results/mcolon/20251029_curvature_temporal_analysis/

python load_data.py              # Test data loading
python 01_individual_trajectories.py
python 02_horizon_plots.py
python 03_embedding_distance.py
python 04_predictive_models.py
```

### Or run individually as needed:
```bash
python 01_individual_trajectories.py   # Just see trajectories
python 04_predictive_models.py          # Just check model performance
```

Each script imports from `load_data.py` and produces independent outputs.

## Metrics Analyzed

### Primary Metrics
- **arc_length_ratio**: Normalized curvature (arc length / chord length)
  - Size-independent, always ≥ 1.0
  - Preferred for cross-embryo comparison

- **normalized_baseline_deviation**: Baseline deviation / embryo length
  - Makes baseline deviation comparable across different embryo sizes
  - Raw values stored in curvature summary

### Secondary Metrics (available in code)
- mean_curvature_per_um
- max_baseline_deviation_um
- keypoint_deviation_mid_um
- total_length_um

## Expected Findings

### From Temporal Trajectories (01)
- Quantify how curvature changes across development
- Identify genotype-specific developmental patterns
- Statistical significance of genotype differences

### From Horizon Plots (02)
- Temporal correlation structure in curvature
- Whether early curvature predicts later curvature
- Genotype-specific temporal correlation patterns

### From Embedding Analysis (03)
- Relationship between morphology changes and curvature changes
- Whether curvature is a good predictor of morphological state
- Genotype clustering in embedding space

### From Predictive Models (04)
- How much of curvature is encoded in morphology embeddings
- Which embedding dimensions are most predictive
- Predictive power of simple curvature metrics

## Next Steps / Extensions

1. **Utility Library Population** (when src/analyze modules ready)
   - Replace direct plotting with `plot_horizon_grid()` from horizon_plots.py
   - Use `build_metric_matrices()` and `load_time_matrix_results()`

2. **Additional Analyses**
   - Bin data by time windows (some/no binning comparison)
   - Extract `plot_hypotenuse_over_stage()` utility from results/mcolon/20250612/
   - Interactive Plotly visualizations for exploration

3. **Model Refinement**
   - Try different regressors (XGBoost, neural networks)
   - Hyperparameter optimization
   - Cross-validation by genotype

4. **Statistical Validation**
   - Bootstrap confidence intervals
   - Effect sizes (Cohen's d)
   - Bayesian model comparison

## Technical Notes

### Data Locations
- Curvature: `morphseq_playground/metadata/body_axis/summary/curvature_metrics_summary_20251017_combined.csv`
- Metadata: `morphseq_playground/metadata/build06_output/df03_final_output_with_latents_20251017_combined.csv`
- Bodies: `morphseq_playground/metadata/body_axis/arrays/curvature_arrays_20251017_combined.csv` (optional, for detailed analysis)

### Dependencies
- pandas, numpy
- matplotlib, seaborn
- scipy (stats, spatial)
- scikit-learn (preprocessing, ensemble, metrics)
- Optional: plotly (for interactive plots)

### Key Design Decisions
1. **load_data.py as foundation** - All downstream scripts depend on this single module
2. **LOEO-CV for models** - Leave-one-embryo-out to avoid overfitting to specific embryos
3. **Normalization** - Baseline deviation normalized by length; embeddings standardized
4. **Reusability** - Scripts use helper functions from load_data.py (get_genotype_short_name, get_genotype_color)
5. **Modular structure** - Each analysis can run independently, uses same data loading

## Summary Statistics

**Dataset: 20201017_combined**
- Total samples: 3,373 timepoints
- Unique embryos: 93
- Genotype distribution:
  - Homozygous: 1,763 (52.3%)
  - Wildtype: 502 (14.9%)
  - Heterozygous: 501 (14.8%)
  - Unknown: 607 (18.0%)
- Mean frames per embryo: 36.3
- Median frames per embryo: 34

**Embedding space**
- Auto-detected embedding dimensions: [varies, check output]
- Assumed latent_* column naming pattern

## Contact / Questions

All scripts follow consistent conventions:
- Genotype color mapping via `get_genotype_color()`
- Short labels via `get_genotype_short_name()`
- Shared output directory structure
- Consistent figure DPI (300) and format (PNG)
