# Runbook

From repository root:

```bash
PYTHON=/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python

"$PYTHON" results/mcolon/20260213_stream_d_reference_embryo/pipeline/01_build_cohort_manifest.py
"$PYTHON" results/mcolon/20260213_stream_d_reference_embryo/pipeline/02_run_batch_ot_export.py
"$PYTHON" results/mcolon/20260213_stream_d_reference_embryo/pipeline/03_build_reference_fields.py
"$PYTHON" results/mcolon/20260213_stream_d_reference_embryo/pipeline/04_compute_deviations.py
"$PYTHON" results/mcolon/20260213_stream_d_reference_embryo/pipeline/05_pca_raw_vector_fields.py
"$PYTHON" results/mcolon/20260213_stream_d_reference_embryo/pipeline/06_difference_classification_clustering.py
```

All scripts accept CLI overrides for input/output paths; these commands use the migrated defaults.
