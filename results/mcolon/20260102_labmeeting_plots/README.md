# CEP290 Cluster Plotting - Lab Meeting 2026-01-02

## Overview

Created visualization scripts for CEP290 homozygous phenotype clustering analysis. The goal is to visualize distinct phenotypic clusters identified through k-medoids clustering and verify they are reproducible across experiments (not batch effects).

## Data Sources

**Input data:**
- `/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251229_cep290_phenotype_extraction/data/clustering_data__early_homo.pkl`
  - Contains trajectory data for all genotypes
  - Key: `df_cep290_earyltimepoints` - main DataFrame

- `/net/trapnell/vol1/home/mdcolon/proj/morphseq/results/mcolon/20251229_cep290_phenotype_extraction/kmedoids_k_selection_early_timepoints_cep290_data/k_results.pkl`
  - K-medoids clustering results (k=5)
  - Access: `k_results_kmedoids['clustering_by_k'][5]['assignments']['cluster_labels']`

## Cluster Definitions (k=5)

```python
cluster_names_k5 = {
    0: 'outlier',      # n=195
    1: 'bumby',        # n=857
    2: 'low_to_high',  # n=1234 (combined with cluster 3)
    3: 'low_to_high',  # n=1234 (combined with cluster 2)
    4: 'high_to_low',  # n=754
}
```

## Generated Plots

**Script:** `plot_cep290_clusters.py`

### Plot 1: `cep290_clusters_faceted.png`
- **Layout:** Single row, 4 columns (one per cluster)
- **Purpose:** Overview of each cluster phenotype pooled across experiments
- **Each facet shows:**
  - Very faint WT individual trajectories (green, alpha=0.05)
  - Very faint Het individual trajectories (orange, alpha=0.05)
  - Cluster individual trajectories (cluster color, alpha=0.3)
  - WT trend line (green, linewidth=2.0, alpha=0.6)
  - Het trend line (orange, linewidth=2.0, alpha=0.6)
  - **Cluster trend line (THICK, linewidth=4.0)**

### Plot 2: `cep290_experiments_and_clusters.png`
- **Layout:** 2 rows
  - Row 1: By experiment (2 columns: Exp 20250512, Exp 20251212)
  - Row 2: By cluster (4 columns: bumby, high_to_low, low_to_high, outlier)
- **Purpose:** Compare overall genotype patterns by experiment, then show clusters
- **Note:** Row 1 shows all homozygous clusters together in red

### Plot 3: `cep290_clusters_by_experiment_batch_check.png` ⭐
- **Layout:** 2 rows × 4 columns
  - Row 1: Each cluster from Experiment 1 (20250512) only
  - Row 2: Each cluster from Experiment 2 (20251212) only
- **Purpose:** **Batch effect validation** - verify each cluster looks similar across experiments
- **Each facet shows:**
  - Very faint WT/Het backgrounds from that experiment (alpha=0.03)
  - Cluster embryos from that experiment only
  - WT/Het trend lines (faint, linewidth=1.5, alpha=0.4)
  - Cluster trend line (THICK, linewidth=4.0)
  - **n count annotation** (sample size per cluster per experiment)
- **How to interpret:** Compare vertical pairs (same cluster across experiments)
  - Does "bumby" from Exp 1 look like "bumby" from Exp 2?
  - If yes → phenotype is reproducible, not a batch effect

## Technical Details

### Color Scheme
```python
COLORS = {
    'cep290_wildtype': '#2E7D32',      # Green
    'cep290_heterozygous': '#FFA500',  # Orange
    'outlier': '#D32F2F',              # Red
    'bumby': '#9467BD',                # Purple
    'low_to_high': '#17BECF',          # Cyan
    'high_to_low': '#E377C2',          # Pink
}
```

### Trend Line Computation
- **Method:** Binned median with Gaussian smoothing
- **Parameters:**
  - `bin_width=0.5` hpf
  - `smooth_sigma=1.5` (Gaussian kernel)
- **Function:** `compute_trend_line()` in script

### Alpha Values (Transparency)
- Individual trajectories:
  - WT/Het background: 0.03-0.05 (very faint)
  - Cluster trajectories: 0.3 (visible but not overwhelming)
- Trend lines:
  - WT/Het reference: 0.4-0.6 (subdued)
  - Cluster: 1.0 (fully opaque and thick)

## Running the Script

```bash
cd /net/trapnell/vol1/home/mdcolon/proj/morphseq
python results/mcolon/20260102_labmeeting_plots/plot_cep290_clusters.py
```

**Output:** All 3 plots saved to `results/mcolon/20260102_labmeeting_plots/`

## Future Enhancements

### Proposed: Reference Line Overlay Functionality

**Goal:** Extend `facetted_plotting.py` to support reference genotype overlays

**Use case:** When plotting cluster phenotypes, automatically overlay WT and Het as faint reference lines without manually coding each plot.

**Proposed API:**
```python
plot_trajectories_faceted(
    df,
    x_col='predicted_stage_hpf',
    y_col='baseline_deviation_normalized',
    line_by='embryo_id',
    col_by='cluster',
    color_by_grouping='cluster',
    # NEW PARAMETERS:
    reference_overlay='genotype',  # Column to use for reference
    reference_values=['cep290_wildtype', 'cep290_heterozygous'],  # Which values to overlay
    reference_alpha=0.05,  # Transparency for reference individual traces
    reference_trend_alpha=0.6,  # Transparency for reference trend lines
    reference_split_by='experiment_id',  # Optional: split references by this column
)
```

**Implementation considerations:**
- Need to handle both individual traces and trend lines for references
- Should references be from same facet group or global?
- How to handle reference_split_by when faceting by multiple variables?
- Color assignment: use standard genotype colors or custom palette?

**Alternative approach:** Create a separate overlay layer system
```python
fig = plot_trajectories_faceted(df, col_by='cluster', ...)
add_reference_overlay(fig, df, genotypes=['wt', 'het'], alpha=0.05)
```

**Status:** Deferred for later consideration. Current custom scripts work well for specific use cases.

## Notes

- Gaussian smoothing already applied (sigma=1.5) - don't need to add more
- Sample sizes vary significantly across clusters (195 to 1234 embryos)
- Two clusters (2 and 3) were merged into "low_to_high" category
- All plots share y-axis limits for fair comparison
- GridSpec used for flexible subplot layouts (avoids empty facets)

## Questions for Analysis

1. Are cluster phenotypes reproducible across experiments?
2. Do batch effects explain any of the variation?
3. Which clusters show strongest deviation from WT/Het?
4. Are sample sizes sufficient for each cluster in each experiment?

---

# Statistical Analysis - 2026-01-02 Session

## Achievements

### 1. Core Statistical Analysis Script
**File:** `cep290_statistical_analysis.py`

Implemented comprehensive statistical testing using:
- **Per-cluster classification**: Each phenotype cluster (bumpy, high_to_low, low_to_high) vs WT
- **Pooled classification**: All Homozygous vs WT, Homo vs Het, Het vs WT
- **Time-resolved analysis**: 2-hour time bins to track when differences emerge
- **Permutation testing**: Bootstrap p-values for all AUROC comparisons

### 2. Key Findings (Quick Run, 5 permutations)

**Per-Cluster AUROC vs WT:**
- `low_to_high`: **Max AUROC = 0.998** at 50 hpf (nearly perfect classification!)
- `high_to_low`: Max AUROC = 0.972 at 36 hpf (peaks earlier)
- `bumpy`: Max AUROC = 0.889 at 56 hpf

**Pooled Comparisons:**
- All Homo vs WT: Max AUROC = 0.931 at 56 hpf
- All Homo vs Het: Max AUROC = 0.938 at 56 hpf

**Baseline:**
- Het vs WT: Max AUROC = 0.623 (near chance, confirming Hets similar to WT)

### 3. Critical Observation: Per-Cluster AUROC > Pooled AUROC

**Finding:** Splitting homozygous mutants by cluster *increases* classification performance compared to pooling all homozygous together.

**Interpretation:**
- Each cluster represents a distinct, coherent phenotype
- Pooling mixes different (possibly opposite) trajectories (low→high vs high→low vs bumpy)
- Classifier has cleaner decision boundary when phenotype is homogeneous
- **This validates clustering**: The k-medoids clusters capture real, biologically distinct phenotypes

### 4. Generated Outputs

**Plots:**
- `cep290_cluster_vs_wt_auroc.png` - Per-cluster AUROC vs WT with Het baseline overlay
- `cep290_classification_auroc.png` - Pooled comparisons with p-value annotations
- `cep290_temporal_emergence_clusters.png` - When each cluster becomes distinguishable
- `cep290_temporal_emergence_pooled.png` - Temporal emergence for pooled groups

**Data:**
- `classification_results.csv` - All AUROC and p-values per time bin

---

## Proposed Follow-Up Analyses

### Priority 1: Validate Per-Cluster AUROC Increase

**Hypothesis:** Higher per-cluster AUROC is due to phenotypic coherence, not just smaller sample sizes.

**Test - Random Split Control:**
1. Randomly split all homozygous embryos into 3 groups (matched to cluster sizes)
2. Run same classification (random group vs WT)
3. Compare mean AUROC of random splits to real cluster AUROC
4. **Expected:** Random splits should have lower AUROC than real clusters

**Implementation:** New script `cep290_cluster_validation.py`

---

### Priority 2: Time Resolution Sensitivity Analysis

**Motivation:** Ensure findings aren't artifacts of 2-hour time binning that pools velocity information.

**Test:**
- Re-run all analyses with **1-hour time bins**
- Compare AUROC trajectories and emergence times
- Check if finer resolution changes conclusions

**Implementation:** Add `bin_width` parameter sweep to existing script

---

### Priority 3: Alternative Feature Sets

**Current:** Uses only VAE latent features (`z_mu_b`)

**Tests:**
1. **Morphological features only**: `mean_curvature_per_um`, `baseline_deviation_normalized`
2. **Combined features**: VAE + morphology
3. **Curvature-specific**: Use curvature metrics to classify phenotype types

**Question:** Can we classify clusters using interpretable morphological features instead of latent embeddings?

**Implementation:** Modify `compare_groups()` calls to use different feature sets

---

### Priority 4: Increase Permutations for Robust P-values

**Current limitation:** 5 permutations → minimum p-value = 0.167 (no significance markers)

**Solution:**
- Increase to 50-100 permutations for meaningful p-values
- Run full analysis with proper statistical power
- Generate plots with visible significance annotations

---

## File Organization Recommendations

### Keep in `cep290_statistical_analysis.py`:
- Core per-cluster and pooled classification
- Standard time-binned AUROC analysis
- Main visualization plots

### Create `cep290_cluster_validation.py`:
- Random split control analysis
- Sample size sensitivity tests
- Cluster coherence metrics

### Create `cep290_feature_comparison.py`:
- Alternative feature set comparisons (VAE vs morphology vs combined)
- Feature importance analysis
- Interpretability analysis

### Create `cep290_temporal_resolution.py`:
- Time bin width sweep (0.5h, 1h, 2h, 4h)
- Velocity vs position information analysis
- Optimal temporal resolution determination

---

## Next Steps

1. **Immediate:** Run full analysis with 100 permutations
2. **Validation:** Implement random split control
3. **Robustness:** Test 1-hour time bins
4. **Interpretability:** Try morphological features for classification
5. **Documentation:** Update with final results and statistical interpretations
