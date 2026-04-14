# TFAP2 Followup Analysis Summary

**Directory:** `results/mcolon/20260413_tfap2_followup/`
**Date:** 2026-04-13
**Data source:** `results/mcolon/20260324_tfap2_crispant_first_pass/results/tfap2_combined_20260213_20260223_20260224_20260319_20260320.parquet`
— 597 embryos, 16 genotypes (14 crispants + inj_ctrl + non_inj_ctrl), 5 experiments

---

## Scripts

### `scripts/common.py`
Shared constants and data loaders used by all three scripts.
- `EXPERIMENT_IDS`, `BIN_WIDTH = 2.0`, `MIN_SUPPORT = 3`
- `load_aggregate_dataframe()` — reads the first-pass aggregated parquet
- `load_supported_window(results_dir)` — reads `results/supported_window.json`

### `scripts/01_support_table.py`
Determines where all 16 genotypes have adequate embryo coverage.
- Bins embryos into 2 hpf windows, counts unique embryos per (genotype, bin)
- Finds the contiguous window where every genotype has ≥3 embryos
- **Result: 16–32 hpf** (tfap2b_crispant was the tightest constraint)
- Outputs: `results/support_table.csv`, `results/supported_window.json`, `figures/support_heatmap.png`

### `scripts/02_run_all_pairs_classification.py`
Runs all-pairs binary classification across all 16 genotypes on the full time range.
- 120 pairwise comparisons × 2 feature sets (curvature, embedding) × 500 permutations, n_jobs=24
- Saves to `results/all_pairs_classification/`:
  - `scores.parquet` — AUROC + p-values per (comparison, feature_set, time_bin)
  - `raw_contrast_scores_long.parquet` — **131,490 rows** of per-embryo classifier margin scores; this is the key artifact consumed by condensation
  - `classifier_directions.parquet` + `classifier_directions_vectors.npz`
- Renders `figures/emergence_explorer.html` — interactive AUROC emergence timeline for all 16 genotypes, sorted by embryo count

### `scripts/03_run_condensation.py`
Loads `raw_contrast_scores_long.parquet` and runs trajectory condensation.
- Runs **4 combinations**: {curvature, embedding} × {supported_window (16–32 hpf), all_timepoints}
- For each: pivots scores wide (rows = embryo × time_bin, cols = comparison_id margins), builds CondensationData, runs 500-iter condensation
- Outputs per combination:
  - `results/condensation/{feature}/{subset}/run/condensed_positions.npz`
  - `results/condensation/{feature}/{subset}/run/metrics.csv`
  - `figures/condensation/{feature}/{subset}/viz_genotype/time_slice.html`
- `--smoke` flag: 50 iters, 5 embryos/genotype for fast end-to-end testing

---

## Key results

- **Strongest hits (AUROC = 1.0):** `tfap2c_tfap2e_crispant`, `tfap2c_crispant`, `tfap2b_tfap2d_crispant`, `tfap2b_crispant`, `tfap2a_tfap2c_crispant`, `tfap2a_tfap2b_crispant` (embedding); `tfap2c_tfap2e_crispant`, `tfap2a_tfap2c_crispant`, `tfap2b_tfap2d_crispant`, `tfap2a_crispant` (curvature)
- **Supported window:** 16–32 hpf

---

## Next step: 20260320 singles analysis

Experiment 20260320 is the **late-timepoint singles experiment** (34–88 hpf), containing 86 embryos across 7 genotypes: `inj_ctrl`, `tfap2b_crispant`, `tfap2c_crispant`, `tfap2d_crispant`, `tfap2e_crispant`, `tfap2a_tfap2b_crispant`, `tfap2a_tfap2d_crispant`.

**The full pipeline has already been run.** All contrast scores are in `raw_contrast_scores_long.parquet`. No re-classification is needed.

### What the agent needs to do

The only task is to run **new condensation runs** that subset the existing scores to the 20260320 embryos (or to just the single-crispant genotypes). Everything else reuses existing artifacts.

**Concretely:**

1. Load `results/all_pairs_classification/raw_contrast_scores_long.parquet`
2. Filter to embryos from experiment 20260320: `scores[scores["embryo_id"].str.startswith("20260320")]`
   - This gives genotypes: `inj_ctrl`, `tfap2b_crispant`, `tfap2c_crispant`, `tfap2d_crispant`, `tfap2e_crispant`, `tfap2a_tfap2b_crispant`, `tfap2a_tfap2d_crispant`
   - Alternatively filter to just `genotype.isin(["inj_ctrl", "tfap2b_crispant", "tfap2c_crispant", "tfap2d_crispant", "tfap2e_crispant"])` for singles only
3. Re-run condensation with that subset — no need to touch classification at all
4. Output to `results/condensation/20260320_singles/` and `figures/condensation/20260320_singles/`

**No new script needed.** Add a `--subset` CLI argument to `03_run_condensation.py` (e.g. `--subset 20260320` or `--subset singles`) that filters `scores_long` after loading. The condensation config, output paths (namespaced by subset_key), and all helper functions are already in place — it's a small addition to `_parse_args()` and the filter block in `main()` before the existing subset loop.

### Key file locations
| Artifact | Path |
|---|---|
| Contrast scores (pre-computed) | `results/all_pairs_classification/raw_contrast_scores_long.parquet` |
| Reference condensation script | `scripts/03_run_condensation.py` |
| Condensation API | `src/analyze/trajectory_condensation/condensation/__init__.py` |
| UMAP init | `src/analyze/trajectory_condensation/init_embedding.py` |
| Time-slice viewer | `src/analyze/trajectory_condensation/viz/condensed_time_slice_viewer.py` |
| Color lookup | `src/analyze/viz/styling/color_utils.py` → `build_genotype_color_lookup` |
