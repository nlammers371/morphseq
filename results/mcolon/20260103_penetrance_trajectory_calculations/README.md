# Trajectory-Specific Penetrance Analysis

## Overview

This analysis calculates penetrance rates over developmental time for CEP290 embryos grouped by DTW-derived trajectory clusters. The key insight is that CEP290 embryos don't show purely stochastic phenotypes—they follow distinct trajectories. Therefore, penetrance should be analyzed per-trajectory to understand phenotype expression patterns.

## Analysis Date
January 3, 2026

## Data Source
- **Location**: `results/mcolon/20251229_cep290_phenotype_extraction/final_data/`
- **Files**: `embryo_data_with_labels.csv` (45,767 frames, 635 embryos)
- **Trajectories**: 6 clusters identified via multivariate Dynamic Time Warping (DTW)

## Methodology

### Hybrid Threshold Approach

We use a **hybrid IQR threshold** strategy to account for temporal dynamics in curvature:

1. **Early development (<30 hpf)**: Time-binned IQR
   - Separate threshold per 2-hour bin
   - Captures dynamic curvature changes during early morphogenesis
   - Threshold: `Q1 - k×IQR` to `Q3 + k×IQR` per bin (k=2.0)

2. **Late development (≥30 hpf)**: Global IQR
   - Single threshold pooled from all WT data ≥30 hpf
   - Exploits stable curvature after ~30 hpf
   - Avoids sparse bin issues in later development

**Rationale**: Curvature is highly dynamic before 30 hpf but stabilizes afterward. Time-binned thresholds provide temporal sensitivity early on, while global thresholds provide robustness later.

### Penetrance Calculation

1. **Mark frames**: Classify each frame as penetrant (1) if curvature falls outside WT IQR bounds, otherwise non-penetrant (0)
2. **Embryo-level aggregation**: If ANY frame in a time bin shows penetrance, the embryo counts as penetrant in that bin
3. **Penetrance rate**: `n_penetrant_embryos / n_total_embryos` per trajectory per time bin
4. **Standard error**: `SE = sqrt(p × (1-p) / n)` for confidence intervals

### Metric
- **`baseline_deviation_normalized`**: Maximum deviation of embryo centerline from straight line, normalized by body length
- Provides size-independent measure of curvature

## Two-Phase Analysis

### Phase A: Broad Categories (4 groups)
- **Low_to_High**: Clusters 0 + 2 (progressive phenotype worsening)
- **High_to_Low**: Clusters 1 + 4 (potential phenotype recovery)
- **Intermediate**: Cluster 3 (moderate curvature, large spread in outcomes)
- **Not Penetrant**: Cluster 5 (WT-like)

**Script**: `01a_penetrance_by_broad_category.py`

### Phase B: Subcategories (6 groups)
- **Low_to_High_A**: Cluster 0 (75 embryos, 65.7% homozygous)
- **Low_to_High_B**: Cluster 2 (26 embryos, 86.3% homozygous)
- **High_to_Low_A**: Cluster 1 (52 embryos, 87.3% homozygous)
- **High_to_Low_B**: Cluster 4 (89 embryos, 84.8% homozygous)
- **Intermediate**: Cluster 3 (40 embryos, 85.2% homozygous)
- **Not Penetrant**: Cluster 5 (289 embryos, 47.8% heterozygous, 33.4% WT)

**Script**: `01b_penetrance_by_subcategory.py`

## Scripts

| Script | Purpose | Outputs |
|--------|---------|---------|
| `config.py` | Configuration constants, colors, paths | - |
| `data_loading.py` | Data loading utilities | - |
| `01a_penetrance_by_broad_category.py` | Phase A: 4 broad categories | CSV tables, penetrance curves, heatmap |
| `01b_penetrance_by_subcategory.py` | Phase B: 6 subcategories | CSV tables, penetrance curves, heatmap |
| `02_visualize_combined.py` | Combined/comparative visualizations | Diagnostic plots, comparisons |

## Outputs

### Tables (`outputs/tables/`)
- `wt_threshold_summary.csv`: WT IQR bounds per time bin (with method: time-binned vs global)
- `category_penetrance_by_time.csv`: Penetrance by broad category and time
- `subcategory_penetrance_by_time.csv`: Penetrance by subcategory and time

### Figures (`outputs/figures/`)
- `penetrance_curves_by_category.png`: 4-line plot (Phase A)
- `penetrance_curves_by_subcategory.png`: 6-line plot (Phase B)
- `penetrance_heatmap_category.png`: Heatmap for broad categories
- `penetrance_heatmap_subcategory.png`: Heatmap for subcategories
- `threshold_diagnostic.png`: WT bounds visualization with 30 hpf cutoff
- Additional comparative plots from `02_visualize_combined.py`

## Completed Work

### Phase A: Broad Categories ✓
**Status**: Complete (Jan 3, 2026)

Analyzed penetrance by 4 broad trajectory categories:
- **Not Penetrant**: mean=9.0%, max=57.1%
- **Intermediate**: mean=66.4%, max=100.0%
- **High_to_Low**: mean=63.9%, max=100.0%
- **Low_to_High**: mean=65.9%, max=94.2%

**Outputs**:
- `category_penetrance_by_time.csv` - Penetrance rates across development
- `penetrance_curves_by_category.png` - 4-line temporal plot with SE bands
- `penetrance_heatmap_category.png` - Heatmap visualization

### Phase B: Subcategories ✓
**Status**: Complete (Jan 3, 2026)

Analyzed penetrance by 6 trajectory subcategories:
- **Not Penetrant**: mean=9.0%, max=57.1%
- **Intermediate**: mean=66.4%, max=100.0%
- **High_to_Low_A**: mean=77.5%, max=100.0%
- **High_to_Low_B**: mean=53.1%, max=100.0%
- **Low_to_High_A**: mean=60.1%, max=92.1%
- **Low_to_High_B**: mean=82.0%, max=100.0%

**Outputs**:
- `subcategory_penetrance_by_time.csv` - Fine-grained penetrance rates
- `penetrance_curves_by_subcategory.png` - 6-line temporal plot
- `penetrance_heatmap_subcategory.png` - Heatmap visualization
- `wt_threshold_summary.csv` - Hybrid threshold bounds per time bin

### Key Findings
1. **Hybrid threshold successfully implemented**: Time-binned IQR for <30 hpf (9 bins), global IQR for ≥30 hpf (56 bins, pooled from 5,576 WT samples)
2. **Clear trajectory separation**: Phenotypic trajectories show distinct penetrance patterns
3. **Notable spike at ~31 hpf**: "Not Penetrant" shows 38% penetrance at 31 hpf bin (likely threshold transition artifact)
4. **Late-stage anomalies**: Unusual patterns observed >80 hpf - needs visualization diagnostics

## Running the Analysis

```bash
# Phase A: Broad categories (COMPLETED)
python 01a_penetrance_by_broad_category.py

# Phase B: Subcategories (COMPLETED)
python 01b_penetrance_by_subcategory.py

# Combined visualizations (NEXT STEP)
python 02_visualize_combined.py
```

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `IQR_K` | 2.0 | IQR multiplier (~95% coverage) |
| `EARLY_CUTOFF_HPF` | 30.0 | Threshold for hybrid method switch |
| `TIME_BIN_WIDTH` | 2.0 | Time bin width in hours |
| `METRIC_NAME` | `baseline_deviation_normalized` | Curvature metric |

## Interpretation

- **Not Penetrant**: Should show low penetrance (~0-10%) throughout development
- **Intermediate**: Moderate penetrance with large variance (unstable phenotype)
- **High_to_Low**: Start with high penetrance, potentially decrease over time
- **Low_to_High**: Start with low penetrance, increase progressively

### Expected Patterns
Given genotype enrichment:
- **Low_to_High** clusters (0, 2): Progressive increase in penetrance (homozygous-enriched)
- **High_to_Low** clusters (1, 4): High early penetrance, potential stabilization (homozygous-enriched)
- **Intermediate**: Moderate penetrance with high variability (homozygous-enriched but unstable)
- **Not Penetrant**: Consistently low penetrance (heterozygous + WT enriched)

## Next Steps

### Immediate Priority: WT Penetrance Diagnostic Visualizations
Following the pattern from October scripts (`06b_penetrance_threshold_calibration.py`, `06e_global_iqr_calibration.py`), we need diagnostic plots to understand threshold behavior:

1. **WT Penetrance Validation Plot**
   - Plot WT penetrance across time to validate thresholds
   - Expected: WT penetrance should be low (<5-10%) if thresholds are well-calibrated
   - Current concern: "Not Penetrant" shows 38% penetrance spike at 31 hpf
   - Purpose: Identify if this is real biology or threshold artifact

2. **Threshold Bounds Diagnostic** (like `plot_threshold_bounds_diagnostic` from 06b)
   - Top panel: WT threshold bounds (low, median, high) over time
   - Highlight 30 hpf cutoff with vertical line
   - Bottom panel: Sample counts per bin (identify sparse bins)
   - Purpose: Visualize where threshold method switches and if bins are well-sampled

3. **Scatter Plot with Threshold Overlay** (like 06e scatter plot)
   - Raw curvature data points colored by genotype
   - WT threshold bounds overlaid as horizontal bands (time-binned <30 hpf, global ≥30 hpf)
   - Highlight penetrant vs non-penetrant points
   - Purpose: See which points are being classified as penetrant and why

4. **Late-Stage Diagnostic (>80 hpf)**
   - Focused view of 80-140 hpf region
   - Check for: sparse data, threshold appropriateness, biological vs technical variation
   - Purpose: Understand unusual patterns observed in late development

**Why this matters**: The 31 hpf spike and >80 hpf anomalies suggest potential threshold calibration issues. Without visualizing WT penetrance and thresholds directly, we can't distinguish between:
- Real biological variation in WT embryos
- Threshold artifacts from hybrid method transition
- Sparse data effects in late development

### Script to Create: `03_wt_penetrance_diagnostics.py`
Should generate:
- `wt_penetrance_by_time.png` - WT penetrance curve (should be flat and low)
- `threshold_bounds_diagnostic.png` - Bounds + sample counts over time
- `scatter_with_thresholds.png` - Raw data with threshold overlay
- `late_stage_diagnostic.png` - Focused 80-140 hpf analysis

### Future Work

- Export validated utilities to `src/analyze/penetrance/trajectory.py`
- Compare penetrance across different IQR multipliers (k=1.5, 2.0, 3.0)
- Analyze penetrance by genotype within each trajectory
- Identify critical developmental windows where penetrance changes most rapidly
- Investigate state transitions (when embryos become/cease being penetrant)
