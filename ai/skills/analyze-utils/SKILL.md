# analyze-utils — Utilities Reference

## Module: `analyze.utils.data_loading`

### `load_experiment`
**File:** `src/analyze/utils/data_loading.py`

```python
def load_experiment(
    experiment_id: str,
    build_dir: Union[str, Path],
) -> pd.DataFrame
```

Loads `{build_dir}/df03_final_output_with_latents_{experiment_id}.csv`.

### `load_experiments`
**File:** `src/analyze/utils/data_loading.py`

```python
def load_experiments(
    experiment_ids: List[str],
    build_dir: Union[str, Path],
    verbose: bool = True,
) -> pd.DataFrame
```

Concatenates multiple experiments. Adds `source_experiment` column.

**Standard build_dir:** `morphseq_playground/metadata/build06_output/`

---

## Module: `analyze.utils.pca`

### `fit_transform_pca`
**File:** `src/analyze/utils/pca.py`

```python
def fit_transform_pca(
    df: pd.DataFrame,
    z_mu_cols: Optional[List[str]] = None,  # auto-detects z_mu_b_* if None
    n_components: int = 3,
    scale: bool = True,
    prefix: str = 'PCA',
) -> Tuple[pd.DataFrame, PCA, Optional[StandardScaler], List[str]]
```

Returns `(df_with_pca, pca_model, scaler, pca_col_names)`. PCA columns are `PCA1, PCA2, ...` (or `{prefix}1, ...`).

### `compute_wt_reference_by_time`
```python
def compute_wt_reference_by_time(
    df: pd.DataFrame,
    pca_cols: List[str],
    time_col: str = 'predicted_stage_hpf',
    wt_embryo_ids: Optional[List[str]] = None,
    embryo_id_col: str = 'embryo_id',
    genotype_col: str = 'genotype',
    wt_genotype_pattern: str = 'wildtype',
    bin_width: float = 2.0,
) -> pd.DataFrame
```

Returns DataFrame with columns: `time_bin`, plus one column per PCA col (mean value for WT embryos in that time bin).

### `subtract_wt_reference`
```python
def subtract_wt_reference(
    df: pd.DataFrame,
    wt_reference: pd.DataFrame,
    pca_cols: List[str],
    time_col: str = 'predicted_stage_hpf',
    bin_width: float = 2.0,
    suffix: str = '_delta',
) -> pd.DataFrame
```

Creates `{col}{suffix}` columns (e.g., `PCA1_delta`).

---

## Module: `analyze.utils.binning`

### `bin_embryos_by_time`
**File:** `src/analyze/utils/binning.py`

```python
def bin_embryos_by_time(
    df: pd.DataFrame,
    time_col: str = "predicted_stage_hpf",
    z_cols: Optional[List[str]] = None,   # auto-detects z_mu_b_* if None
    bin_width: float = 2.0,
    suffix: str = "_binned",
) -> pd.DataFrame
```

Averages embeddings per embryo_id × time_bin. Keeps metadata columns (genotype, etc.).

---

## Module: `analyze.utils.splitting`

### `train_test_split_by_group`
**File:** `src/analyze/utils/splitting.py`

```python
def train_test_split_by_group(
    df: pd.DataFrame,
    group_col: str = 'embryo_id',
    test_fraction: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]
```

Group-level split: all rows for a given group go to train OR test, never both.
