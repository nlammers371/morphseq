# Embryo-First Persistence Plan (v2)

## Goal
Implement a reproducible embryo-first persistence workflow that uses simplified `input_data` tables and validates biological signal with clustering stability, faceted trajectory visualization, classification tests, and enrichment tests.

## Data Source (Simplified)
Default input should come from:
- `input_data/experiments/{dataset_id}/input_core.csv`

Why:
- avoids loader friction,
- consistent schema,
- model-ready filtering guidance already documented in `input_data/README.md` and `input_data/MODEL_USAGE.md`.

## Analysis Scope Strategy
Run in this order:

1. Within-experiment runs
- Build cohorts per `experiment_id` first (lowest batch/confound risk).

2. Within-dataset pooled runs
- Pool experiments within `cep290` and within `b9d2`.
- Evaluate whether stable cohort structure is conserved across experiments.

3. Cross-dataset comparison run
- Combine cep290 + b9d2 for benchmarking phenotype separability.
- Treat this as a benchmark/diagnostic stage, not primary discovery.

## Feature Benchmark Requirement
Benchmark both feature modes in every scope above:
- `zmu_b` (biological latents, default)
- `pca95` (PCs explaining 95% variance)

Primary benchmark outputs:
- runtime,
- memory proxy (n_features and matrix shapes),
- cluster stability metrics,
- downstream validation agreement.

## Script Plan

### 1) `run_embryo_first_persistence.py`
Main cohort discovery script.

Workflow:
1. Load `input_core.csv` tables.
2. Apply model/QC filtering from `MODEL_USAGE.md` plus explicit run-time filters.
3. Build time bins using existing utility:
   - `add_time_bins`
   - `bin_embryos_by_time`
   - `filter_binned_data`
4. Build per-window soft-mutual kNN graphs.
5. Aggregate persistence matrices (`P_mean`, `P_topq`).
6. Scan resolution with bootstrap consensus.
7. Select stable resolution and assign cohorts.
8. Compute drift/gain/loss and embryo taxonomy.
9. Write outputs.

### 2) `validate_persistence_clusters.py`
Post-hoc cluster reality checks.

Workflow:
1. Load cohort assignments + source data.
2. Generate faceted feature-over-time plots with `plot_feature_over_time`:
   - explicitly set `color_by='cluster'` for trajectory separation checks.
   - required views:
     - `color_by='cluster'`, `facet_row='genotype'`, `facet_col='experiment_id'`
     - `color_by='cluster'`, `facet_row='phenotype'`, `facet_col='experiment_id'`
     - `color_by='genotype'`, `facet_col='cluster'` (inverse view)
3. Validate separability with `run_classification_test`:
   - classify `cluster` from selected features across time bins,
   - report AUROC/null p-values.
4. Add distribution-difference checks by cluster:
   - per-cluster feature distributions (e.g., violin/box summaries) by genotype/phenotype.
   - report whether clusters have distinct internal composition profiles.
5. Summarize whether clusters are robust vs null.

### 3) `cluster_enrichment_analysis.py`
Biology validation using positive controls.

Workflow:
1. Build contingency tables: `cluster x genotype`, `cluster x phenotype`.
2. Run enrichment tests (Fisher exact for 2x2; chi-square/permutation otherwise).
3. Correct p-values (FDR).
4. Report odds ratios/effect sizes and highlight expected enrichments:
   - cep290 and penetrant mutant phenotypes in expected trajectory classes
   - e.g., high-to-low / low-to-high expectations.
5. Generate proportion visualizations using `plot_proportions`:
   - `row_by='genotype'`, `col_by='cluster'`, `color_by_grouping='phenotype'`
   - `row_by='phenotype'`, `col_by='cluster'`, `color_by_grouping='genotype'`
   - optional inverse:
     - `row_by='genotype'`, `col_by='phenotype'`, `color_by_grouping='cluster'`
     - `row_by='experiment_id'`, `col_by='cluster'`, `color_by_grouping='genotype'`
   - use these to inspect which stable clusters are enriched for expected controls.

### 4) `run_full_persistence_bundle.py` (optional orchestrator)
Runs discovery + validation + enrichment end-to-end for chosen scope/mode.

## Function Signatures (Planned)

```python
def run_persistence(
    input_paths: list[str],
    feature_mode: str,              # "zmu_b" | "pca95"
    scope_mode: str,                # "within_experiment" | "within_dataset" | "cross_dataset"
    window_hpf: float = 2.0,
    k: int = 15,
    topq: float = 0.25,
    resolutions: list[float] | None = None,
    n_bootstrap: int = 100,
    bootstrap_frac: float = 0.8,
    seed: int = 42,
    output_dir: str | None = None,
) -> dict:
    ...
```

```python
def validate_clusters(
    data_path: str,
    assignments_path: str,
    feature_cols: list[str],
    time_col: str = "predicted_stage_hpf",
    embryo_id_col: str = "embryo_id",
    output_dir: str | None = None,
) -> dict:
    ...
```

```python
def run_enrichment(
    assignments_path: str,
    metadata_path: str,
    cluster_col: str = "cluster",
    genotype_col: str = "genotype",
    phenotype_col: str = "phenotype",
    make_proportion_plots: bool = True,
    output_dir: str | None = None,
) -> dict:
    ...
```

## Standard Outputs
Per run directory:
`results/mcolon/20260213_subtle_phenotype_methods/output/embryo_first_persistence/{run_id}/`

Required artifacts:
- `config.json`
- `cohort_assignments.tsv`
- `persistence_mean.npy`
- `persistence_topq.npy`
- `resolution_scan.tsv`
- `drift_per_window.tsv`
- `drift_summary.tsv`
- `feature_benchmark.tsv`
- `cluster_validation_summary.tsv`
- `classification_validation.tsv`
- `enrichment_genotype.tsv`
- `enrichment_phenotype.tsv`
- `cluster_proportions_by_genotype.png`
- `cluster_proportions_by_phenotype.png`

## Decision Criteria (What “Working” Means)
A run is considered biologically credible when:
- bootstrap consensus indicates stable cohorts,
- faceted trajectories show coherent cohort behavior,
- `run_classification_test` shows above-null separation,
- enrichment recovers expected positive controls for genotype/phenotype.

## Immediate Build Order
1. Implement `run_embryo_first_persistence.py`.
2. Implement `validate_persistence_clusters.py` with faceted plotting + classification test.
3. Implement `cluster_enrichment_analysis.py`.
4. Run first benchmark matrix:
   - scope: within-experiment,
   - feature mode: `zmu_b` vs `pca95`,
   - dataset: cep290 then b9d2.
