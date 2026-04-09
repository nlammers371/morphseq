# analyze-classification — Misclassification Viz Reference

## Module: `analyze.classification.viz.misclassification`

**File:** `src/analyze/classification/viz/misclassification.py`

---

## `plot_margin_trends`

Per-embryo signed-margin trajectory plots for a binary comparison. One panel per group (negative left, positive right), each embryo is a line colored by mean `truth_signed_margin` on RdBu_r.

### Generating the data

Run classification with `save_predictions=True` (or `save_dir=`), then load `predictions.parquet`.
See `COMMAND.md` for the full pipeline command.

The predictions parquet produced by the pipeline has these columns:

| column | type | notes |
|---|---|---|
| `comparison_id` | str | e.g. `"inj_ctrl__vs__pbx4_crispant"` |
| `positive_label` | str | positive class label (y_true == 1) |
| `negative_label` | str | negative class label (y_true == 0) |
| `feature_set` | str | e.g. `"vae"`, `"shape"` |
| `embryo_id` | str | |
| `time_bin` | int | bin start (hpf) |
| `time_bin_center` | float | bin midpoint (hpf) — default x-axis |
| `bin_width` | float | bin width in hpf |
| `n_positive` | int | positive embryos in this bin |
| `n_negative` | int | negative embryos in this bin |
| `auroc_obs` | float | per-bin AUROC |
| `y_true` | int | 0 or 1 |
| `p_pos` | float | classifier probability of positive class |
| `truth_signed_margin` | float | in [-1, 1]; +1 = most correct, -1 = most wrong |
| `y_pred` | int | predicted class |
| `is_correct` | bool | |

### API

```python
plot_margin_trends(
    df,
    *,
    # Selector — exactly one mode required
    comparison_id: str | None = None,        # pipeline mode
    positive_label: str | None = None,       # explicit mode: right panel
    negative_label: str | None = None,       # explicit mode: left panel

    # Secondary filter (optional)
    feature_col: str = "feature_set",
    feature_id: str | None = None,

    # Column names
    margin_col: str = "truth_signed_margin", # must be in [-1, 1]
    x_col: str = "time_bin_center",          # one value per (embryo, x) pair
    embryo_col: str = "embryo_id",

    # Display
    color_mode: str = "continuous",          # "continuous" | "discrete"
    discrete_class_lookup: dict | None = None,
    discrete_class_colors: dict | None = None,
    max_embryos: int = 30,
    vmin: float = -1.0,
    vmax: float = 1.0,
    time_window: tuple[float, float] | None = None,
    output_path: Path | None = None,
) -> matplotlib.figure.Figure
```

### Validation

The function enforces these in order:
1. Exactly one selector mode (`comparison_id` xor `positive_label`+`negative_label`)
2. Comparison filter applied; error if empty
3. Feature filter applied if `feature_id` given; error if empty
4. `embryo_col`, `x_col`, `margin_col` all present
5. `margin_col` values in [-1, 1]
6. One row per `(embryo_col, x_col)` pair — duplicate (embryo, x) raises immediately
7. If explicit-label mode and df has `positive_label`/`negative_label` columns: verifies the
   passed labels match the data. Will **not** swap silently — raises with a clear error.

### Usage examples

**Pipeline-native (comparison_id from predictions parquet):**
```python
import pandas as pd
from analyze.classification.viz.misclassification import plot_margin_trends

predictions = pd.read_parquet("results/.../predictions.parquet")

# All feature rows for this comparison
fig = plot_margin_trends(
    predictions,
    comparison_id="inj_ctrl__vs__pbx4_crispant",
    output_path="margin_trends.png",
)

# Filter to one feature set
fig = plot_margin_trends(
    predictions,
    comparison_id="inj_ctrl__vs__pbx4_crispant",
    feature_id="vae",
    output_path="margin_trends_vae.png",
)
```

**Explicit-label mode (ad hoc dataframe):**
```python
fig = plot_margin_trends(
    df,
    positive_label="pbx4_crispant",
    negative_label="inj_ctrl",
    feature_id="vae",
    output_path="margin_trends.png",
)
```

**Loop over all comparisons:**
```python
for cid in sorted(predictions["comparison_id"].unique()):
    slug = cid.replace("__vs__", "_vs_")
    plot_margin_trends(
        predictions,
        comparison_id=cid,
        feature_id="vae",
        max_embryos=50,
        output_path=figures_dir / f"margin_{slug}.png",
    )
```

**Custom x-axis (non-time):**
```python
fig = plot_margin_trends(
    df,
    comparison_id="inj_ctrl__vs__pbx4_crispant",
    x_col="pca_1",
    margin_col="truth_signed_margin",
)
```

---

## Validator utilities

**File:** `src/analyze/classification/viz/utils.py`

```python
from analyze.classification.viz.utils import (
    validate_required_columns,   # raises if columns missing
    validate_margin_range,       # raises if values outside [-1, 1]
    validate_unique_embryo_x,    # raises if duplicate (embryo, x) pairs
)
```

These are called internally by `plot_margin_trends` but can be used standalone to validate
a dataframe before passing it to the plotter.
