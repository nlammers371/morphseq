# Model Input Guide

Use `results/mcolon/20260213_subtle_phenotype_methods/input_data/experiments/{dataset_id}/input_core.csv` as the default model table.

## Minimal modeling contract

- Row unit: one snip/frame observation
- Required filtering before training:
  - `use_embryo_flag == True`
  - project decision for `dead_flag2`
- Default numeric feature block:
  - `total_length_um`
  - `baseline_deviation_um`
  - `mean_curvature_per_um`
  - `std_curvature_per_um`
  - `max_curvature_per_um`
  - `surface_area_um`
  - `area_um2`

## Typical targets

- phenotype classification: `phenotype`
- genotype classification: `genotype`
- perturbation classification: `short_pert_name`
- developmental timing regression: `predicted_stage_hpf`

## Suggested split keys

To reduce leakage, split by embryo or well when possible:

- group key candidates: `embryo_id`, `well_id`, `video_id`

## Notes

- `schema_core.csv` is the canonical column definition.
- `datasets_manifest.csv` records source/output mapping and refresh metadata.
- `phenotype` in `input_core.csv` is the derived phenotype label
  (priority: `cluster_categories > phenotype_label > phenotype`).
- Unknown/unlabeled genotype or phenotype rows are removed at build time.
