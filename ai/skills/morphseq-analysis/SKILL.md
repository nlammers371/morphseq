# morphseq-analysis — General Analysis Reference

## Project Structure

```
morphseq/
  src/analyze/
    viz/                    # Visualization (plotting, hpf_coverage)
    trajectory_analysis/    # DTW, clustering, bootstrap projection
    classification/         # AUROC classification, MulticlassOVRResults
    utils/                  # Data loading, PCA, binning, splitting
    difference_detection/   # Deprecated shim → classification
    tutorials/              # Jupyter notebooks
  results/mcolon/           # Analysis output directories
  morphseq_playground/
    metadata/build06_output/  # Build06 experiment CSVs
```

## Python Environment

```bash
PYTHON=/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python
# Tests: PYTHONPATH=src:$PYTHONPATH "$PYTHON" -m pytest ...
```

## Script Boilerplate

```python
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

project_root = Path(__file__).resolve().parents[N]
sys.path.insert(0, str(project_root / "src"))
```

## Output Layout

```
results/mcolon/{YYYYMMDD}_{description}/
  figures/        # PNG, HTML plots
  results/        # CSVs, summary text files
```

## Key Data Columns

| Column | Type | Description |
|---|---|---|
| `embryo_id` | str | Unique embryo ID within experiment |
| `genotype` | str | Genotype label (normalize before use) |
| `predicted_stage_hpf` | float | Developmental stage (hours post fertilization) |
| `total_length_um` | float | Body length (micrometers) |
| `baseline_deviation_normalized` | float | Curvature metric |
| `z_mu_b_0` ... `z_mu_b_N` | float | VAE latent dimensions |
| `use_embryo_flag` | bool | Quality filter |
| `experiment_id` | str | Experiment date ID |

## Submodule Reference

### Visualization (`analyze.viz`)
- `plot_feature_over_time()` — time-series feature plots with faceting
- `plot_proportions()` — category proportion bar charts
- `plot_3d_scatter()` — 3D scatter with trajectory lines
- `plot_experiment_time_coverage()` / `plot_hpf_overlap_quick()` — HPF coverage
- See: `ai/skills/analyze-viz/SKILL.md`

### Trajectory Analysis (`analyze.trajectory_analysis`)
- `compute_trajectory_distances()` — pairwise DTW distances
- `run_k_selection_with_plots()` — optimal k selection
- `run_bootstrap_projection_with_plots()` — bootstrap projection onto reference
- `get_color_for_genotype()` / `sort_genotypes_by_suffix()` — genotype styling
- See: `ai/skills/analyze-trajectory/SKILL.md`

### Classification (`analyze.classification`)
- `run_classification_test()` — time-binned AUROC with permutation testing
- `MulticlassOVRResults` — dict-like results container
- `plot_feature_comparison_grid()` — side-by-side AUROC panels
- `plot_multiple_aurocs()` — overlay AUROC curves
- See: `ai/skills/analyze-classification/SKILL.md`

### Utilities (`analyze.utils`)
- `load_experiments()` / `load_experiment()` — build06 data loading
- `fit_transform_pca()` — PCA with auto-detection of z_mu_b columns
- `compute_wt_reference_by_time()` / `subtract_wt_reference()` — WT deviation
- `bin_embryos_by_time()` — time-binned embedding averaging
- `train_test_split_by_group()` — group-level train/test split
- See: `ai/skills/analyze-utils/SKILL.md`

## Workflow Skills

- **genotype-trends:** Standardized genotype trend analysis script generation
- **cep290-analysis:** CEP290-specific data locations and full pipeline
