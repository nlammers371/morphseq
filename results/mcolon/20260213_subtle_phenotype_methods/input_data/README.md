# Input Data (Simple, Model-Friendly)

This folder exists so people can load one standardized CSV per experiment without tracing multiple utilities.

## What goes here

- `./experiments/{dataset_id}/input_core.csv`
  - One lean CSV per experiment
  - Same core schema across experiments
  - Intended as the default table for analysis/modeling
  - Built with explicit label policy:
    - `phenotype` is resolved from derived labels with priority
      `cluster_categories > phenotype_label > phenotype`
    - rows with unknown/unlabeled genotype or phenotype are dropped during build

## Core schema purpose

The core CSV keeps only high-value columns for modeling and filtering:

1. IDs and provenance
- `snip_id`, `image_id`, `embryo_id`, `video_id`, `experiment_id`, `well_id`, `experiment_date`

2. Time context
- `frame_index`, `time_int`, `raw_time_s`, `relative_time_s`, `predicted_stage_hpf`, `start_age_hpf`

3. Labels / perturbation context
- `genotype`, `chem_perturbation`, `phenotype`, `short_pert_name`, `control_flag`

4. QC / include flags
- `use_embryo_flag`, `dead_flag2`, `well_qc_flag`, `sam2_qc_flag`, `focus_flag`, `bubble_flag`, `no_yolk_flag`, `sa_outlier_flag`

5. Morphology features
- `total_length_um`, `baseline_deviation_um`, `mean_curvature_per_um`, `std_curvature_per_um`, `max_curvature_per_um`, `surface_area_um`, `area_um2`

Full per-column metadata is in `schema_core.csv`.

## For modelers: recommended usage

1. Start from `input_core.csv`
2. Filter rows with:
- `use_embryo_flag == True`
- `dead_flag2 == False` (or your project-specific choice)
- optional: drop rows with obvious QC failures (`focus_flag`, `bubble_flag`, etc.)
3. Use morphology features as default numeric inputs:
- `total_length_um`
- `baseline_deviation_um`
- `mean_curvature_per_um`
- `std_curvature_per_um`
- `max_curvature_per_um`
- `surface_area_um`
- `area_um2`
4. Use labels based on task:
- classification: `genotype`, `phenotype`, `short_pert_name`
- regression/time modeling: `predicted_stage_hpf`, `relative_time_s`

## Build / refresh standardized CSVs

```bash
PYTHON=/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python
"$PYTHON" results/mcolon/20260213_subtle_phenotype_methods/input_data/build_standardized_inputs.py
```

`datasets_manifest.csv` records how many rows were dropped due to
unknown/unlabeled labels (`dropped_unknown_or_unlabeled`).

## Quick load example

```python
import pandas as pd

df = pd.read_csv(
    "results/mcolon/20260213_subtle_phenotype_methods/input_data/experiments/cep290_20251229/input_core.csv"
)

df_model = df[
    (df["use_embryo_flag"] == True)
    & (df["dead_flag2"] == False)
].copy()

feature_cols = [
    "total_length_um",
    "baseline_deviation_um",
    "mean_curvature_per_um",
    "std_curvature_per_um",
    "max_curvature_per_um",
    "surface_area_um",
    "area_um2",
]
X = df_model[feature_cols]
y = df_model["phenotype"]
```

## Important

Generated data files are local working artifacts and should not be committed by default.
