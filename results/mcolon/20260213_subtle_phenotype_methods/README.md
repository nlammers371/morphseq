# Subtle Phenotype Methods (CEP290 + B9D2)

This directory provides a runnable starting point for the subtle phenotype
methods plan using real validation datasets.

## Default Data Source

Use standardized local inputs first:
- `input_data/experiments/{dataset_id}/input_core.csv`

Reference docs:
- `input_data/README.md`
- `input_data/MODEL_USAGE.md`

## Scripts

- `load_validation_data.py`
  - Resolves CEP290 and B9D2 labeled CSVs from either:
    - current repo `results/...` paths, or
    - fallback `morphseq_CORRUPT_OLD/results/...` paths.
  - Standardizes `phenotype_label` and writes summary TSV/JSON artifacts.

- `phase1_resampling_smoke.py`
  - Loads both datasets, computes embryo-level AUC on
    `baseline_deviation_normalized`, and runs:
    - bootstrap CI on mean difference
    - permutation p-value
  - Uses the shared utility at `src/analyze/utils/resampling`.

- `run_embryo_first_persistence.py`
  - Main discovery runner.
  - Loads `input_core.csv`, applies model/QC filters, bins time with
    `src/analyze/utils/binning`, builds soft-mutual persistence matrices
    (`persistence_mean.npy`, `persistence_topq.npy`), scans cluster resolution
    with bootstrap ARI stability using `src/analyze/utils/resampling`,
    assigns embryo clusters, and writes drift/taxonomy summaries.

- `validate_persistence_clusters.py`
  - Post-hoc validation.
  - Uses `plot_feature_over_time` to generate faceted trajectory plots
    (`color_by='cluster'` and inverse views) and runs
    `run_classification_test` for cluster separability vs null permutations.

- `cluster_enrichment_analysis.py`
  - Enrichment + biological positive-control checks.
  - Runs Fisher tests for `cluster x genotype` and `cluster x phenotype`,
    applies FDR correction, and writes `plot_proportions` views, including
    cluster-colored inverse composition plots.

- `run_full_persistence_bundle.py`
  - Optional orchestrator for discovery -> validation -> enrichment in one run.

- `plot_persistence_diagnostics.py`
  - Plot bundle for manual review from one run directory.
  - Includes:
    - `resolution_scan_metrics.png`
    - `resolution_scan_over_time_heatmap.png` (+ `resolution_scan_over_time.tsv`)
    - `cluster_presence_over_time.png`
    - `persistence_matrix_sorted.png`
    - `classification_ovr_auroc_over_time.png` (if validation file exists)

## Existing Utility To Reuse (Time Binning)

For embryo-first persistence, use the existing binning helpers in:
- `src/analyze/utils/binning.py`

Functions:
- `add_time_bins(df, time_col='predicted_stage_hpf', bin_width=2.0, bin_col='time_bin')`
  - Adds row-level `time_bin` labels without aggregation.
- `bin_embryos_by_time(df, time_col='predicted_stage_hpf', z_cols=None, bin_width=2.0, suffix='_binned')`
  - Aggregates latent features per `embryo_id x time_bin`.
- `filter_binned_data(df_binned, min_time_bins=3, min_embryos=5)`
  - Drops embryos with insufficient temporal coverage.

Import path:
```python
from src.analyze.utils.binning import add_time_bins, bin_embryos_by_time, filter_binned_data
```

This should be the default route for 2 hpf window construction (instead of re-implementing custom binning logic).

## Validation/Enrichment Visualization Utility

Use `plot_proportions` for cluster enrichment visualization after stable clusters are found:

```python
from src.analyze.viz.plotting import plot_proportions
```

Typical views:
- proportions by `cluster` faceted by `genotype`
- proportions by `cluster` faceted by `phenotype`
- inverse composition view with `color_by_grouping='cluster'` to show how cluster mix differs across groups

## Run

```bash
PYTHON=/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python
"$PYTHON" results/mcolon/20260213_subtle_phenotype_methods/load_validation_data.py
"$PYTHON" results/mcolon/20260213_subtle_phenotype_methods/phase1_resampling_smoke.py

# Discovery (within-experiment, auto feature mode, multi-CPU bootstrap)
"$PYTHON" results/mcolon/20260213_subtle_phenotype_methods/run_embryo_first_persistence.py \
  --scope-mode within_experiment \
  --feature-mode auto \
  --n-bootstrap 100 \
  --n-jobs 16

# Validation and enrichment on latest run dir
"$PYTHON" results/mcolon/20260213_subtle_phenotype_methods/validate_persistence_clusters.py
"$PYTHON" results/mcolon/20260213_subtle_phenotype_methods/cluster_enrichment_analysis.py

# End-to-end convenience wrapper
"$PYTHON" results/mcolon/20260213_subtle_phenotype_methods/run_full_persistence_bundle.py \
  --scope-mode within_experiment \
  --feature-mode auto \
  --n-bootstrap 100 \
  --n-permutations 200 \
  --n-jobs 16

# Manual diagnostics plot bundle for a completed run
"$PYTHON" results/mcolon/20260213_subtle_phenotype_methods/plot_persistence_diagnostics.py \
  --run-dir results/mcolon/20260213_subtle_phenotype_methods/output/embryo_first_persistence/<run_id>
```

If you hit OpenMP shared-memory errors in restricted environments, run with:
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` and retry.

Outputs are written to `results/mcolon/20260213_subtle_phenotype_methods/output/`.
