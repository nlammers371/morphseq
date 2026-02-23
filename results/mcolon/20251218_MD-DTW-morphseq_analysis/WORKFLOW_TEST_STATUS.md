# MD-DTW Workflow Test - STATUS

**Date:** December 18, 2025
**Status:** ✅ **PHASE 1 COMPLETE - Real Data Analysis Successful**

---

## Phase 1: Real b9d2 Data Analysis ✅ COMPLETE

### Analysis Run: 2025-12-18 18:07:29

Successfully completed full MD-DTW pipeline on real b9d2 dataset with **IQR 4.0× outlier removal**.

**Dataset:** Experiment 20251121 (b9d2 mutants)

### Execution
```bash
python run_analysis.py --experiment 20251121 --k 2 3 4 5
```

**Output:** `output/20251121_20251218_180729/`

---

## Results Summary

### Step 1: Data Loading ✅
- **Total embryos loaded:** 96 (b9d2 genotypes)
- **After QC filtering:** 94 embryos
- **Time range:** 18-48 hpf
- **Final rows:** 3,591 (after time filtering)
- **Timepoints per embryo:** 46 (25.0-47.5 hpf, step=0.5)

### Step 2: Multivariate Array Preparation ✅
- **Metrics:** `baseline_deviation_normalized`, `total_length_um`
- **Array shape:** (94, 46, 2)
- **Normalization:** Z-score (mean=0, std=1 per metric)
- **Interpolation:** Common time grid using existing `interpolate_to_common_grid_multi_df()`

### Step 3: MD-DTW Distance Matrix ✅
- **Method:** Pure Python/NumPy with Sakoe-Chiba constraint (radius=3)
- **Computation:** 4,465 pairwise distances
- **Distance range:** [5.84, 307.73]
- **Matrix properties:** Symmetric ✓, Zero diagonal ✓

### Step 4: Outlier Detection & Removal ✅ **[NEW]**

**Method:** IQR 4.0× (extreme outlier definition)
- **Q1 (25th %ile):** 16.875
- **Q3 (75th %ile):** 26.217
- **IQR:** 9.342
- **Threshold:** Q3 + 4.0×IQR = **63.586**

**Outliers Removed (8 embryos):**
1. `20251121_A05_e01` - median_dist = 270.622 (extreme!)
2. `20251121_A07_e01` - median_dist = 192.808 (extreme!)
3. `20251121_C11_e01` - median_dist = 67.592
4. `20251121_D10_e01` - median_dist = 68.052
5. `20251121_D12_e01` - median_dist = 79.577
6. `20251121_E03_e01` - median_dist = 89.425
7. `20251121_F09_e01` - median_dist = 118.708
8. `20251121_F11_e01` - median_dist = 79.063

**Clean dataset:** 86 embryos (down from 94)

**Why IQR 4.0×?**
- Statistically principled: "extreme outlier" by boxplot standards
- Tested alternatives: 95th percentile (5 outliers), Log-MAD 5.0× (15 outliers), IQR 3.0× (11 outliers)
- Best balance: Removes clear outliers while preserving biological variation

### Step 5: Bootstrap Hierarchical Clustering ✅
- **Method:** Average linkage (UPGMA) with bootstrap (100 iterations)
- **K values tested:** 2, 3, 4, 5
- **Cluster assignments:** All embryos classified

**Clustering Results (After Outlier Removal):**

| k | Cluster Sizes | Notes |
|---|---------------|-------|
| **k=2** | {0: 81, 1: 5} | **Main cluster (81) vs small group (5)** ← Potential HTA/CE split |
| **k=3** | {0: 79, 1: 5, 2: 2} | Main splits slightly, 2 small groups persist |
| **k=4** | {0: 78, 1: 5, 2: 2, 3: 1} | 1 singleton appears |
| **k=5** | {0: 5, 1: 74, 2: 2, 3: 1, 4: 4} | Main cluster (81) splits into 74 vs 5 |

**Comparison to BEFORE outlier removal:**
- **Before:** k=2: {0: 93, 1: 1} ← Singleton inflation problem
- **After:** k=2: {0: 81, 1: 5} ← Clean split! ✅

### Step 6: Visualizations Generated ✅

**Outputs:**
- `dendrogram_md_dtw.png` - Shows cluster structure with k cutoffs
- `multimetric_trajectories_k2.png` - 2 metrics × 2 clusters
- `multimetric_trajectories_k3.png` - 2 metrics × 3 clusters
- `multimetric_trajectories_k4.png` - 2 metrics × 4 clusters
- `multimetric_trajectories_k5.png` - 2 metrics × 5 clusters
- `distance_matrix_heatmap.png` - Sorted by cluster (95th %ile clipped)
- `clustering_summary.png` - Silhouette scores + cluster sizes

**Saved Data:**
- `cluster_assignments.csv` - Cluster labels for all k values
- `distance_matrix_cleaned.npy` - After outlier removal (86×86)
- `distance_matrix_original.npy` - Before outlier removal (94×94)
- `embryo_ids_cleaned.txt` - 86 inliers
- `embryo_ids_original.txt` - 94 original
- `outliers_removed.txt` - Outlier detection summary

---

## Key Findings

### 1. Outlier Removal is Critical ✅
- **Before:** Singleton clusters inflate k, masking biological structure
- **After:** Clean clustering reveals meaningful splits at k=2 and k=5

### 2. k=2 Shows Major Split
- **81 vs 5 embryos** - Potential HTA vs CE distinction
- Next: Validate by coloring multimetric plots by genotype and pair

### 3. Small Persistent Groups (5, 2 embryos)
- Could be:
  - True biological sub-phenotypes
  - Non-penetrant vs penetrant mutants
  - Borderline outliers (could test IQR 3.5× if needed)

### 4. k=5 Reveals Main Cluster Split
- The 81-embryo main cluster splits into **74 vs 5**
- Suggests internal heterogeneity worth investigating

---

## Outlier Detection Methods Tested

We systematically tested 4 principled methods (vs arbitrary percentile):

| Method | Parameter | N Outliers | Threshold | Notes |
|--------|-----------|------------|-----------|-------|
| **IQR (chosen)** | **4.0×** | **8** | **63.6** | ✅ Standard "extreme outlier" definition |
| IQR | 3.5× | 10 | 58.9 | More aggressive |
| IQR | 3.0× | 11 | 54.2 | |
| Log-MAD | 5.0× | 15 | 46.9 | Good for log-normal data, but too aggressive here |
| Log-MAD | 6.0× | 11 | 55.8 | |
| Percentile | 95th | 5 | 79.2 | Simple but arbitrary |
| Percentile | 92nd | 8 | 64.3 | Very close to IQR 4.0× |
| Isolation Forest | cont=0.05 | 5 | N/A | ML approach, black box |

**Consensus outliers (all methods agree):** 4 embryos (A05, A07, E03, F09)

**Strong outliers (≥3/4 methods):** 6 embryos

**Choice:** IQR 4.0× balances statistical rigor with biological preservation.

---

## Implementation Status

| Component | Status | Location |
|-----------|--------|----------|
| `prepare_multivariate_array()` | ✅ Production | md_dtw_prototype.py:40 |
| `compute_md_dtw_distance_matrix()` | ✅ Production | md_dtw_prototype.py:268 |
| `identify_outliers()` | ✅ Production | md_dtw_prototype.py:340 |
| `remove_outliers_from_distance_matrix()` | ✅ Production | md_dtw_prototype.py:462 |
| `plot_dendrogram()` | ✅ Production | md_dtw_prototype.py:518 |
| `run_analysis.py` pipeline | ✅ Production | run_analysis.py |
| Multimetric plotting (by cluster) | ✅ Working | faceted_plotting.py |
| Multimetric plotting (by genotype) | ⏳ **NEXT** | Need to implement |
| Multimetric plotting (by pair) | ⏳ **NEXT** | Need to implement |

---

## PHASE 2: Next Steps 🎯

### Step 1: Multi-Experiment Analysis Strategy

**Experiments to analyze (from Initial_notes-on-b9d2-phenotype-pair-trends.md):**

**Phase 2a: Baseline Distribution (b9d2_spawn - mixed population)**
- `20250501` ⏳ FIRST - Understand distribution without known pair labels
- `20250519` ⏳ FIRST - Understand distribution without known pair labels

**Phase 2b: Pair Validation (breeding pairs with known phenotypes)**
- `20251104` 🎯 PRIMARY FOCUS
- `20251119` 🎯 PRIMARY FOCUS
- `20251121` ✅ (already analyzed with k=2: 81 vs 5)
- `20251125` 🎯 PRIMARY FOCUS (also has pairs)

**Strategy:**
1. **First:** Run on spawn data (20250501, 20250519) to understand phenotypic distribution in mixed population
2. **Then:** Focus on pair experiments (20251104, 20251119, 20251121) where HTA vs CE phenotypes are known
3. **Validate:** Color by pair to confirm clusters match HTA (pair_5) vs CE (pairs_6,7,8)

**Combined Analysis:**
```python
# Load all experiments and combine
experiment_ids = ['20250501', '20250519', '20251104', '20251119', '20251121', '20251125']
# Spawn (baseline): 20250501, 20250519
# Pairs (validation): 20251104, 20251119, 20251121, 20251125
```

**Individual experiment runs:**
```bash
# Phase 2a: Spawn analysis (baseline)
python run_analysis.py --experiment 20250501 --k 2 3 4 5
python run_analysis.py --experiment 20250519 --k 2 3 4 5

# Phase 2b: Pair analysis (validation)
python run_analysis.py --experiment 20251104 --k 2 3 4 5
python run_analysis.py --experiment 20251119 --k 2 3 4 5
python run_analysis.py --experiment 20251121 --k 2 3 4 5  # already done
python run_analysis.py --experiment 20251125 --k 2 3 4 5

# Phase 2c: Combined analysis (all 6 experiments)
python run_analysis.py --combined-experiments 20250501,20250519,20251104,20251119,20251121,20251125 --k 2 3 4 5
```

### Step 2: Add Genotype/Pair Coloring to Plots (Immediate TODO)

We need to generate **3 versions of multimetric plots** for each k value:

1. **✅ Color by cluster** (already working)
   - Shows cluster identity and consistency
   - Files: `multimetric_trajectories_k2.png`, etc.

2. **⏳ Color by genotype** (NEXT)
   - Validate that clusters separate genotypes (b9d2_homozygous vs b9d2_heterozygous vs b9d2_wildtype)
   - Check if clusters correspond to expected genotype-phenotype relationships

3. **⏳ Color by pair** (NEXT)
   - Validate that clusters separate known pairs (b9d2_pair_5 = HTA, b9d2_pair_6/7/8 = CE)
   - This is the **critical validation** for HTA vs CE distinction

### Implementation Plan:

**Update `run_analysis.py` to generate 3 plot versions per k:**

```python
# For each k value:
for k in k_values:
    # Version 1: Color by cluster (already done)
    plot_multimetric_trajectories(..., color_by='md_dtw_cluster', ...)

    # Version 2: Color by genotype (NEW)
    plot_multimetric_trajectories(..., color_by='genotype', ...)

    # Version 3: Color by pair (NEW)
    plot_multimetric_trajectories(..., color_by='pair', ...)
```

**Expected outputs per k:**
- `multimetric_trajectories_k2_by_cluster.png`
- `multimetric_trajectories_k2_by_genotype.png`
- `multimetric_trajectories_k2_by_pair.png`
- (Same for k=3, 4, 5)

### Success Criteria:

**For single experiment (20251121):**
- [ ] **Genotype coloring:** Do clusters correspond to genotype groups?
- [ ] **Pair coloring:** Does cluster 0 vs 1 match HTA (pair_5) vs CE (pairs 6,7,8)?
- [ ] **Visual validation:** Do trajectories show expected divergence patterns?
  - HTA: High curvature, normal length
  - CE: High curvature → severe shortening after ~32 hpf

**For combined experiments (all 6):**
- [ ] Load and merge experiments: 20250501, 20250519, 20251104, 20251119, 20251121, 20251125
- [ ] Run full MD-DTW pipeline on combined dataset (~86-96 embryos × 46 timepoints)
- [ ] Generate multimetric plots (cluster, genotype, pair coloring)
- [ ] Compare clustering across experiments
- [ ] Validate if phenotype clusters are consistent across datasets
- [ ] **Critical validation:** Color by pair - confirm HTA (pair_5) vs CE (pairs_6,7,8) separation

---

## Validation Checklist

### Phase 1: Core Pipeline ✅
- [x] Multivariate array preparation handles multiple metrics
- [x] MD-DTW distance computation is symmetric and well-formed
- [x] Outlier detection methods tested and validated
- [x] IQR 4.0× outlier removal implemented
- [x] Dendrogram correctly identifies cluster structure
- [x] K-selection visualization shows cutoff heights
- [x] Cluster assignments extracted for each K
- [x] Multimetric trajectory plots by cluster working
- [x] Output files generated successfully
- [x] Real b9d2 data analysis complete

### Phase 2: Biological Validation ⏳
- [ ] Multimetric plots colored by genotype
- [ ] Multimetric plots colored by pair
- [ ] Cluster-genotype correspondence analyzed
- [ ] Cluster-pair correspondence analyzed
- [ ] HTA vs CE phenotype distinction validated
- [ ] Divergence timepoint identified (~32 hpf expected)

---

**Status:** Ready for Phase 2 - Generate genotype and pair-colored multimetric plots to validate biological interpretation of clusters.

**Next command:**
```bash
# After implementing genotype/pair coloring:
python run_analysis.py --experiment 20251121 --k 2 3 4 5
```
