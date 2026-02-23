# Curvature Temporal Analysis - Handoff Document

**Last Updated**: October 29, 2025
**Status**: All analysis scripts created and tested. Ready for production runs.

## Quick Start

All scripts are in `/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251029_curvature_temporal_analysis/`

Run them in order:
```bash
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251029_curvature_temporal_analysis/

# 1. Load and explore data
python load_data.py

# 2. Individual embryo trajectories over development
python 01_individual_trajectories.py

# 3. Temporal correlation heatmaps
python 02_horizon_plots.py

# 4. Embedding space analysis
python 03_embedding_distance.py

# 5. Predictive models
python 04_predictive_models.py
```

## Project Overview

**Goal**: Analyze curvature temporal patterns for embryos from 20201017 combined dataset.
- Compare 3 genotypes: wildtype (WT), heterozygous (Het), homozygous (Homo)
- 93 embryos, 2,766 merged timepoints (after filtering)
- 100 embedding dimensions (z_mu_* columns)

**Key Metrics**:
- `arc_length_ratio`: Normalized curvature (arc length / chord length)
- `normalized_baseline_deviation`: Baseline deviation / embryo length

## File Organization

```
├── load_data.py                      # Data loading foundation (SINGLE SOURCE OF TRUTH)
├── 01_individual_trajectories.py     # Per-embryo + aggregate plots
├── 02_horizon_plots.py               # Temporal correlation heatmaps
├── 03_embedding_distance.py          # Embedding space vs curvature
├── 04_predictive_models.py           # ML: embeddings ↔ curvature
│
├── outputs/
│   ├── figures/
│   │   ├── 01_individual_trajectories/
│   │   ├── 02_horizon_plots/
│   │   ├── 03_embedding_distance/
│   │   └── 04_predictive_models/
│   ├── tables/
│   │   ├── 01_individual_trajectories/
│   │   ├── 02_horizon_plots/
│   │   ├── 03_embedding_distance/
│   │   └── 04_predictive_models/
│   └── models/
│       └── 04_predictive_models/
│
└── README.md, IMPLEMENTATION_COMPLETE.md, this file
```

## Critical Details

### Data Characteristics
- **Combined dataset issue**: 20201017 is combined from part1 and part2
- Frame indices (0-57) appear twice (once per part), interleaved in file
- **Solution**: Always sort by `predicted_stage_hpf` (developmental stage), NOT `frame_index`
- All scripts have been fixed to use `predicted_stage_hpf`

### Data Sources
- **Curvature**: `morphseq_playground/metadata/body_axis/summary/curvature_metrics_summary_20251017_combined.csv`
- **Metadata + Embeddings**: `morphseq_playground/metadata/build06_output/df03_final_output_with_latents_20251017_combined.csv`
- **Embedding columns**: Look for `z_mu_*` columns (100 total), NOT `latent_*`

### load_data.py - The Foundation

This is the single source of truth. All downstream scripts import from here:
```python
from load_data import get_analysis_dataframe, get_genotype_short_name, get_genotype_color

df, metadata = get_analysis_dataframe()
# Returns:
# - df: merged dataframe with curvature + embeddings + normalized metrics
# - metadata: dict with embedding_cols, genotypes, genotype_labels, counts
```

**Important functions**:
- `get_analysis_dataframe()` - Main entry point, loads + merges + normalizes data
- `get_genotype_short_name()` - Returns 'WT', 'Het', 'Homo'
- `get_genotype_color()` - Returns RGB color for consistent plotting

**Key preprocessing**:
- Filters to 3 genotypes (drops 'unknown' genotypes)
- Normalizes `baseline_deviation_um` by `total_length_um`
- Auto-detects `z_mu_*` embedding columns
- Merges on `snip_id`

### Known Issues & Solutions

1. **Frame index jumbling in combined dataset**
   - ✅ FIXED: All scripts now sort by `predicted_stage_hpf`
   - Check any new code follows this pattern

2. **Different embryo lengths**
   - ✅ HANDLED: Aggregate functions (e.g., correlation matrices) handle variable-length embryos
   - Example: `aggregate_correlations_by_genotype()` computes lag correlations individually

3. **Embedding detection**
   - ✅ FIXED: `get_embedding_columns()` looks for `z_mu_*` first, then `latent_*`
   - Check file for actual column names if this fails in future

## Analysis Details

### 01: Individual Trajectories
**Output**: Individual embryo plots + aggregate comparisons
- Shows curvature evolution over development
- Includes ANOVA and pairwise t-tests by genotype

### 02: Horizon Plots
**Output**: Timepoint × timepoint correlation heatmaps
- Shows temporal autocorrelation structure
- Default: lag correlations (0-10 frame lags)
- Averaged across embryos per genotype

### 03: Embedding Distance
**Output**: Scatter plots + 3D trajectories in embedding space
- Analyzes relationship between morphology changes (embedding distance) and curvature changes
- Computes Spearman correlations per genotype

### 04: Predictive Models
**Output**: ML model performance + feature importance
- **Embeddings → Curvature**: Which embedding dims encode curvature?
- **Curvature → Embeddings**: Can curvature metrics predict morphology?
- Uses leave-one-embryo-out (LOEO) cross-validation

## Next Steps / TODO

### Short Term
- [ ] Review generated plots for biological sanity
- [ ] Check if embeddings actually encode curvature well
- [ ] Verify statistical tests are appropriate

### Medium Term
- [ ] Integrate with `src/analyze/difference_detection/horizon_plots.py` once fully implemented
- [ ] Extract `plot_hypotenuse_over_stage()` utility from `results/mcolon/20250612/` and reuse
- [ ] Consider binning by time windows (alternative to lag-based analysis)

### Long Term
- [ ] Build interactive visualizations (Plotly)
- [ ] Test different ML models (XGBoost, neural networks)
- [ ] Statistical refinements (bootstrap CI, Bayesian comparisons)
- [ ] Cross-dataset validation with other embryo datasets

## Troubleshooting

**Issue**: "Found 0 embedding dimensions"
- **Check**: Metadata file has `z_mu_*` columns (look for z_mu_n_00, z_mu_b_20, etc.)
- **Fix**: Update `get_embedding_columns()` if column naming changed

**Issue**: Trajectories still look jumbled
- **Check**: Script is sorting by `predicted_stage_hpf`, not `frame_index`
- **Fix**: Add `sort_values('predicted_stage_hpf')` before plotting

**Issue**: Memory errors with large datasets
- **Fix**: Consider downsampling or processing by genotype separately

**Issue**: Correlation computation too slow
- **Optimize**: Reduce `max_lag` parameter, process subset of embryos

## Environment

- **Python**: 3.10+ (tested with 3.10)
- **Key packages**: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn
- **Environment**: `/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/`

## Key Design Decisions

1. **Single data loading module** - Prevents version mismatches between scripts
2. **Sort by developmental stage, not frame index** - Handles combined dataset properly
3. **Lag-based correlation aggregation** - Works with variable-length embryos
4. **LOEO cross-validation** - Avoids data leakage in predictive models
5. **Organized output structure** - Easy to navigate and compare analyses

## Questions for Next AI

Before you modify anything:
1. Has the dataset changed? Check `snip_id` prefix in data.
2. Are embeddings still in `z_mu_*` columns?
3. Are you comparing same 3 genotypes?
4. Do trajectories look reasonable when plotted?

## Contact Info

If you need to understand historical decisions, check:
- `IMPLEMENTATION_COMPLETE.md` - Full technical details
- `README.md` - Original objectives
- Git history for this commit

---

**Good luck! The foundation is solid, building on it should be straightforward.** 🚀
