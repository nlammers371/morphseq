# Implementation Handoff: Cryptic Phenotype Detection Framework

**Status:** Ready for implementation
**Full Plan:** See [PLAN.md](./PLAN.md) for complete details
**Output Directory:** `results/mcolon/20260105_refined_embedding_and_metric_classification`

---

## Quick Summary

Build a framework to detect "cryptic phenotypes" - morphological differences detectable in VAE embedding space BEFORE they appear in standard morphometric measurements (curvature, length).

**Key Question:** When do embeddings detect phenotypes vs when do metrics detect them?

---

## Design Principles

1. **REUSE existing code** - NO reimplementation of classification/divergence
   - Use `src/analyze/difference_detection/comparison.compare_groups()`
   - Use `src/analyze/difference_detection/comparison.compute_metric_divergence()`

2. **Thin wrappers only** - Keep `utils/` minimal
   - `data_prep.py` - wraps `add_group_column()` + file loading
   - `divergence.py` - loops `compute_metric_divergence()` for multiple metrics
   - `cryptic_window.py` - NEW: detects embedding-before-metric signal gap
   - `plotting.py` - NEW: creates 3-panel figures

3. **Explicit class convention** - ALWAYS:
   - `group1` = phenotype (POSITIVE class)
   - `group2` = reference (NEGATIVE class: WT or non-penetrant hets)
   - Label plots: `AUROC (positive=CE, negative=WT)`

4. **Time bin clarity (avoid “12 hpf” misreads)**
   - With coarse bins (e.g. `bin_width=4.0`), `time_bin=12` means **12–16 hpf**, not “exactly 12 hpf”.
   - Prefer plotting at bin centers (e.g. 14 hpf for the 12–16 bin) and/or include bin width in titles.
   - When debugging onset claims, retry with smaller bins (e.g. 2h) to localize which sub-window drives signal.

---

## What to Build

### Phase 1: Utils (Day 1)

**File:** `utils/data_prep.py`
- `load_ids_from_file(filepath)` - load embryo IDs from text file
- `prepare_comparison_data()` - wraps `add_group_column()` with optional filtering

**File:** `utils/divergence.py`
- `compute_multi_metric_divergence()` - loops `compute_metric_divergence()` for multiple metrics
- `zscore_divergence()` - add Z-score normalized column for multi-metric comparison

**File:** `utils/cryptic_window.py`
- `detect_cryptic_window()` - finds time gap between embedding signal and metric divergence
- `summarize_cryptic_windows()` - creates summary table across comparisons

### Phase 2: Plotting (Day 1-2)

**File:** `utils/plotting.py`
- `create_comparison_figure()` - 3-panel layout:
  - Panel A: AUROC over time (metric + optional embedding overlay)
  - Panel B: Z-scored metric divergence
  - Panel C: Individual trajectories with group means

### Phase 3: Analysis Scripts (Day 2-3)

**File:** `cep290_analysis.py`
- Load: `results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv`
- Groups: Penetrant vs Not Penetrant
- Generate 2 plots (with/without embedding overlay)
- Test: Embedding detects at ~18hpf, curvature at >24hpf

**File:** `b9d2_hta_analysis.py`
- Load: `results/mcolon/20251219_b9d2_phenotype_extraction/data/b9d2_labeled_data.csv`
- Groups: HTA vs WT, HTA vs non-pen hets, non-pen hets vs WT
- Generate 8 plots
- Test: HTA appears >60hpf, no cryptic het phenotype for HTA

**File:** `b9d2_ce_analysis.py`
- Same data as HTA
- Groups: CE vs WT, CE vs non-pen hets, non-pen hets vs WT
- Generate 8 plots
- Test: Embedding at ~10hpf (vs WT) / ~15hpf (vs hets), metrics at ~20hpf

---

## Critical Data Contract

### Input DataFrame (after `prepare_comparison_data()`)
```python
df_prep = pd.DataFrame({
    'embryo_id': [...],
    'predicted_stage_hpf': [...],
    'group': ['CE', 'WT', ...],           # Added by add_group_column()
    'z_mu_b_0': [...],                     # Embedding features
    'z_mu_b_1': [...],
    'baseline_deviation_normalized': [...], # Metrics
    'total_length_um': [...],
})
```

### Call Pattern
```python
# 1. Prepare data
from src.analyze.difference_detection.comparison import add_group_column
df_prep = add_group_column(df, groups={'CE': ce_ids, 'WT': wt_ids})

# 2. Run classification (embedding)
from src.analyze.difference_detection.comparison import compare_groups
results = compare_groups(
    df_prep,
    group_col='group',
    group1='CE',        # POSITIVE class (phenotype)
    group2='WT',        # NEGATIVE class (reference)
    features='z_mu_b',  # Auto-detects embedding columns
    bin_width=4.0,
    n_permutations=100,
)
embedding_auroc = results['classification']

# 3. Run classification (metrics)
metric_results = compare_groups(
    df_prep,
    group_col='group',
    group1='CE',
    group2='WT',
    features=['baseline_deviation_normalized', 'total_length_um'],
    bin_width=4.0,
)
metric_auroc = metric_results['classification']

# 4. Compute multi-metric divergence
from utils.divergence import compute_multi_metric_divergence, zscore_divergence
divergence = compute_multi_metric_divergence(
    df_prep,
    group_col='group',
    group1_label='CE',
    group2_label='WT',
    metric_cols=['baseline_deviation_normalized', 'total_length_um'],
)
divergence = zscore_divergence(divergence)

# 5. Detect cryptic window
from utils.cryptic_window import detect_cryptic_window
cryptic = detect_cryptic_window(embedding_auroc, divergence)

# 6. Plot
from utils.plotting import create_comparison_figure
fig = create_comparison_figure(
    auroc_df=metric_auroc,
    divergence_df=divergence,
    df_trajectories=df_prep,
    group1_label='CE',      # POSITIVE
    group2_label='WT',      # NEGATIVE
    metric_cols=['baseline_deviation_normalized', 'total_length_um'],
    embedding_auroc_df=embedding_auroc,
    time_landmarks={24.0: '24 hpf'},
    save_path=Path('output.png'),
)
```

---

## Expected Outputs

### 18 Total Plots

**CEP290 (2 plots):**
1. `cep290_metric_auroc_only.png` - curvature AUROC only
2. `cep290_embedding_vs_metric.png` - embedding + curvature AUROC overlay

**B9D2 HTA (8 plots):**
3-5. HTA vs WT, HTA vs hets, overlay (metric AUROC)
6-8. Same with embedding overlay
9-10. Non-pen hets vs WT (metric only, with embeddings)

**B9D2 CE (8 plots):**
11-13. CE vs WT, CE vs hets, overlay (metric AUROC)
14-16. Same with embedding overlay
17-18. Non-pen hets vs WT (metric only, with embeddings)

### CSV Outputs
- `{comparison}_metric_auroc.csv` - per-time-bin metric AUROC
- `{comparison}_embedding_auroc.csv` - per-time-bin embedding AUROC
- `{comparison}_divergence.csv` - metric divergence over time
- `cryptic_window_summary.csv` - summary of all cryptic windows detected

---

## Validation Checklist

### Scientific
- [ ] CEP290: Embedding significant at ~18hpf, curvature >24hpf (cryptic window)
- [ ] B9D2 CE: Embedding at ~10hpf (vs WT), ~15hpf (vs hets), metrics at ~20hpf
- [ ] B9D2 HTA: No cryptic window (late-onset phenotype)
- [ ] Non-pen hets vs WT: Embedding shows difference, metrics minimal (cryptic het phenotype)

### Technical
- [ ] All utils/ functions are thin wrappers (no reimplementation)
- [ ] Positive/negative class labels explicit in all plots
- [ ] Z-score normalization allows multi-metric comparison
- [ ] All 18 plots generated successfully

---

## Common Pitfalls to Avoid

1. ❌ DON'T reimplement AUROC/CV/permutation logic → use `compare_groups()`
2. ❌ DON'T use private APIs like `_compute_divergence` → use public `compute_metric_divergence()`
3. ❌ DON'T flip positive/negative classes → ALWAYS `group1=phenotype`, `group2=reference`
4. ❌ DON'T forget to Z-score normalize divergence when plotting multiple metrics
5. ❌ DON'T extrapolate trajectories → existing APIs handle this correctly

---

## Next Steps

1. Implement `utils/` modules (Phase 1)
2. Implement `utils/plotting.py` (Phase 2)
3. Implement `cep290_analysis.py` (Phase 3)
4. Implement `b9d2_hta_analysis.py` and `b9d2_ce_analysis.py` (Phase 3)
5. Run all analyses and validate outputs

## Debugging Early CEP290 Signal (Pre-20 hpf)

Two helper scripts exist:
- `results/mcolon/20260105_refined_embedding_and_metric_classification/cep290_validation_analysis.py` (pandas/sklearn): bin-width + data-source + within-bin confound audit
- `results/mcolon/20260105_refined_embedding_and_metric_classification/cep290_minimal_confound_check.py` (no pandas): quick within-bin time/sampling confound check from the refined CSV

**Reference:** See [PLAN.md](./PLAN.md) for complete implementation details, code snippets, and scientific context.
