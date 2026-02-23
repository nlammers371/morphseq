# Stream D Reference Embryo Analysis (Migrated)

This directory contains the ad hoc Stream D reference-embryo workflow migrated out of `src/` and into `results/`.

## Layout

- `pipeline/`: stepwise scripts (`01` to `06`)
- `output/`: generated manifests, plots, tables, and cached exports
- `notes/`: run notes and decisions

## Run Order

1. `pipeline/01_build_cohort_manifest.py`
2. `pipeline/02_run_batch_ot_export.py`
3. `pipeline/03_build_reference_fields.py`
4. `pipeline/04_compute_deviations.py`
5. `pipeline/05_pca_raw_vector_fields.py`
6. `pipeline/06_difference_classification_clustering.py`

Defaults are configured so these scripts read/write from this analysis package (`output/`) while importing reusable code from `src/`.
