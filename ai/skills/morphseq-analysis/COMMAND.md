You are a morphseq analysis expert. This is the entry point for any analysis task in the morphseq project. Use this to orient yourself, then defer to the specific skill commands for detailed function signatures.

## Python Environment

**From CLAUDE.md (must follow):**
```bash
PYTHON=/net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python
```

Never use bare `python`, `python3`, `pip`, or `conda activate`.

## Script Setup Pattern

Every analysis script in `results/mcolon/` follows this pattern:

```python
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # required for headless execution
import matplotlib.pyplot as plt
import pandas as pd

project_root = Path(__file__).resolve().parents[N]  # N depends on depth
sys.path.insert(0, str(project_root / "src"))

# Then import from analyze.*
```

**PYTHONPATH for tests:** `PYTHONPATH=src:$PYTHONPATH`

## Script Output Layout

```
results/mcolon/{YYYYMMDD}_{description}/
  figures/           # plots (PNG, HTML)
  results/           # CSVs, summary text
  scripts/           # if multiple scripts needed
```

## Submodule Map

| Module | Import Path | Skill |
|---|---|---|
| Visualization | `analyze.viz.plotting` | `/analyze-viz` |
| HPF coverage | `analyze.viz.hpf_coverage` | `/analyze-viz` |
| Trajectory DTW | `analyze.trajectory_analysis.utilities.dtw_utils` | `/analyze-trajectory` |
| Clustering | `analyze.trajectory_analysis.clustering` | `/analyze-trajectory` |
| Genotype styling | `analyze.trajectory_analysis.viz.styling` | `/analyze-trajectory` |
| Classification | `analyze.classification` | `/analyze-classification` |
| Classification viz | `analyze.classification.viz.classification` | `/analyze-classification` |
| Data loading | `analyze.utils.data_loading` | `/analyze-utils` |
| PCA | `analyze.utils.pca` | `/analyze-utils` |
| Binning | `analyze.utils.binning` | `/analyze-utils` |
| Splitting | `analyze.utils.splitting` | `/analyze-utils` |

**Deprecated shim:** `analyze.difference_detection` re-exports from `analyze.classification`. Both work.

## Workflow Skills

| Skill | Use When |
|---|---|
| `/genotype-trends` | Generate a complete genotype trend analysis script |
| `/cep290-analysis` | CEP290-specific data locations, pipeline, and reference clusters |

## Data Path

Build06 data lives at:
```
morphseq_playground/metadata/build06_output/df03_final_output_with_latents_{experiment_id}.csv
```

## Common Columns

| Column | Description |
|---|---|
| `embryo_id` | Unique embryo identifier within an experiment |
| `genotype` | Genotype label (normalize before use) |
| `predicted_stage_hpf` | Predicted developmental stage in hours post fertilization |
| `total_length_um` | Body length in micrometers |
| `baseline_deviation_normalized` | Curvature metric (normalized baseline deviation) |
| `z_mu_b_0` ... `z_mu_b_N` | VAE latent embedding dimensions |
| `use_embryo_flag` | Boolean quality filter (True = keep) |
| `experiment_id` | Experiment date identifier |
