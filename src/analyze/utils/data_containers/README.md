# Data Containers (`BinObject`) — Lean Guide

This package is intentionally small right now.

## What it does

- Builds a `BinObject` from raw embryo-time data.
- Auto-materializes VAE embedding columns containing `z_mu_b` into binned mean features.
- Lets you add calculated features back onto the object with a strict grain check.

## Levels

- `binned`: grain is `embryo_id × bin_id`
- `cross_bin`: grain is `embryo_id`
- `embryo_meta`: grain is `embryo_id`
- `bin_meta`: grain is `embryo_id × bin_id`
- `raw`: original rows with bin labels attached

## Main API

```python
bo = BinObject.from_raw(raw_df, bin_width=2.0)

bo.add_feature(
    level="binned",
    values=binned_df_or_series,
    key="bin__toy__feature",
    overwrite=False,
)
```

## Contract

- The target `level` must be one of the writable levels (`binned`, `bin_meta`,
  `embryo_meta`, `cross_bin`). `raw` is read-only and cannot be written via
  `add_feature`.
- Incoming grain rows must be a **subset** of the level's existing grain.
  Missing rows get `NaN`; unknown grain rows (e.g. embryo_ids not present in
  the level) raise an error.
- Grain rows in the incoming values must be unique (no duplicate
  `(embryo_id[, bin_id])`).
- Existing keys cannot be replaced unless `overwrite=True`.
- Cross-level key takeover is not allowed.

## Common usage

- Compute anything you want in pandas.
- Hand back a `Series` or `DataFrame` with the right grain.
- Let `BinObject` validate and attach it to the correct level.

## Classification wiring (simple)

`BinObject` is just the validation/container layer. Classification should still
consume a plain pandas DataFrame.

```python
from analyze.utils.data_containers import BinObject
from analyze.classification import run_classification

bo = BinObject.from_raw(raw_df, bin_width=2.0)

# Optional: add your own computed per-bin feature
toy = bo.levels.binned[["embryo_id", "bin_id"]].copy()
toy["bin__toy__score"] = 1.0
bo.add_feature(level="binned", values=toy, key="bin__toy__score")

# Build classification input as a normal DataFrame
clf_df = bo.levels.binned.merge(
  bo.levels.embryo_meta[["embryo_id", "genotype"]],
  on="embryo_id",
  how="left",
)

run_classification(
  clf_df,
  class_col="genotype",
  id_col="embryo_id",
  time_col="bin_center_time",
  features={"vae": ["bin__vae__0__mean", "bin__vae__1__mean", "bin__toy__score"]},
  comparisons="all_pairs",
  bin_width=2.0,
)
```

## Notes

- VAE features are the primary default path and are binned with `mean`.
- More specialized feature helpers can be added later if they become worth it.
- Older reducer-based code has been archived under _archive/reducers/ for
    reference, but it is no longer part of the public API.
