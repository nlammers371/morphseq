You are a morphseq data loading and utility expert. When the user asks about loading experiment data, PCA, wildtype reference computation, binning, or train/test splitting, use the `src/analyze/utils/` module. Follow these rules exactly.

## Setup

```python
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[N]  # adjust N
sys.path.insert(0, str(project_root / "src"))

from analyze.utils.data_loading import load_experiments, load_experiment
from analyze.utils.pca import fit_transform_pca, compute_wt_reference_by_time, subtract_wt_reference
from analyze.utils.binning import bin_embryos_by_time
from analyze.utils.splitting import train_test_split_by_group
```

## Data Loading

### `load_experiments(experiment_ids, build_dir, verbose=True) -> pd.DataFrame`

Loads multiple experiment CSVs from build06 output and concatenates. Adds `source_experiment` column.

```python
build_dir = project_root / "morphseq_playground" / "metadata" / "build06_output"
df = load_experiments(["20260208", "20260210"], build_dir)
```

**Data path pattern:** `{build_dir}/df03_final_output_with_latents_{experiment_id}.csv`

### `load_experiment(experiment_id, build_dir) -> pd.DataFrame`

Load a single experiment.

## PCA

### `fit_transform_pca(df, z_mu_cols=None, n_components=3, scale=True, prefix='PCA') -> (df, pca, scaler, pca_cols)`

Auto-detects `z_mu_b_*` columns if `z_mu_cols=None`. Returns modified DataFrame with PCA columns appended.

```python
df, pca, scaler, pca_cols = fit_transform_pca(df, n_components=3)
# pca_cols = ['PCA1', 'PCA2', 'PCA3']
```

## Wildtype Deviation Analysis

### `compute_wt_reference_by_time(df, pca_cols, time_col=, wt_genotype_pattern='wildtype', bin_width=2.0)`

Computes time-binned mean PCA values for wildtype embryos. Returns reference DataFrame.

### `subtract_wt_reference(df, wt_reference, pca_cols, time_col=, bin_width=2.0, suffix='_delta')`

Subtracts WT reference from each embryo's PCA values. Creates `PCA1_delta`, `PCA2_delta`, etc.

```python
wt_ref = compute_wt_reference_by_time(df, pca_cols)
df = subtract_wt_reference(df, wt_ref, pca_cols)
```

## Binning

### `bin_embryos_by_time(df, time_col=, z_cols=None, bin_width=2.0, suffix='_binned') -> pd.DataFrame`

Averages embeddings per embryo_id × time_bin. Auto-detects `z_mu_b_*` if `z_cols=None`.

## Train/Test Split

### `train_test_split_by_group(df, group_col='embryo_id', test_fraction=0.20, random_state=42) -> (train_df, test_df)`

Splits by group — all rows for a given embryo go to either train or test, never split. Prevents data leakage in time-series data.
