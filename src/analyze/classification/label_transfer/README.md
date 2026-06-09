# label_transfer

Embryo-level label transfer from a labeled reference dataset to unlabeled query embryos.

## Quick start

```python
from analyze.classification.label_transfer import prepare_reference, transfer_labels

# 1. Fit reference model
ref_model = prepare_reference(
    ref_df,
    feature_cols,
    label_col="cluster_categories",
    group_col="embryo_id",
    time_col="predicted_stage_hpf",
    cv_group_col="experiment_id",   # leave-one-experiment-out CV for quality report
    model_type="per_bin",           # recommended: one model per 4-hpf time bin
)

# 2. Transfer to query embryos
result = transfer_labels(ref_model, query_df)
emb = result["embryo_predictions"]  # one row per query embryo
```

## Model types

| `model_type` | Description | When to use |
|---|---|---|
| `"global"` | One model trained on all reference embryos regardless of time | Quick baseline; reference spans a narrow time window |
| `"per_bin"` | One model per `bin_width`-hpf time bin (embryo-mean features); falls back to global for sparse bins | **Recommended.** Query embryos span multiple developmental stages; avoids conflating early/late phenotype expression |

`"per_bin"` is **Mode C** from the LOEO benchmark in
`results/mcolon/20260601_label_transfer_method/` — it outperforms the global
model at later stages and avoids the collapse seen at early timepoints where the
phenotype has not yet emerged.

## Output schema

### `ref_model` dict

| Key | Content |
|---|---|
| `config` | feature_cols, label_col, group_col, time_col, bin_width, model_type, cv_strategy |
| `classes` | sorted list of label strings from reference |
| `final_model` | global sklearn pipeline (always present; per_bin uses it as fallback) |
| `bin_models` | `{bin_center: pipeline}` — empty dict when `model_type="global"` |
| `quality_report` | per-class precision/recall/F1, by-timebin breakdown, confusion matrix, balanced accuracy |
| `label_profile` | class centroids, n_embryos, hpf quartiles |
| `diagnostics` | per-class transferability flags: `"ok"` / `"warn"` / `"skip"` |

### `result` dict from `transfer_labels`

| Key | Content |
|---|---|
| `embryo_predictions` | One row per query embryo: `prob_<CLASS>`, `predicted_label`, `top_probability`, `argmax_margin`, `consistency_score`, `n_images`, `status`, `bin_used` |
| `image_predictions` | One row per query image with per-class probabilities and `bin_used` |
| `skipped_classes` | Classes flagged `"skip"` and excluded from argmax |

`bin_used` (float, hpf bin center) indicates which per-bin model scored each
image/embryo — useful for joining with per-bin p-values from `run_classification`.

## Significance marking with `run_classification`

The label transfer module does not compute p-values. Use `run_classification`
on the reference embryos to get per-bin AUROC and permutation p-values, then
join on `bin_used` to mark which embryos were scored in statistically significant
bins (p ≤ 0.05):

```python
from analyze.classification import run_classification

scores = run_classification(
    emb_ref,
    class_col="cluster_categories", id_col="embryo_id",
    time_col="predicted_stage_hpf",
    positive="Low_to_High", negative="High_to_Low",
    features={"emb": feat},
    bin_width=4.0, n_permutations=200,
).scores
# scores has: time_bin_center, auroc_obs, pval
```

See `results/mcolon/20260605_sci_cilia_qc_first_pass/cep290_homo_low_to_high.py`
for a worked example.

## Architecture

```
label_transfer/
├── README.md          # this file
├── IMPROVEMENTS.md    # deferred improvements and open questions
├── __init__.py        # exports: prepare_reference, transfer_labels, run_label_transfer,
│                      #          plot_reference_quality, plot_transfer_result
└── core.py            # all implementation
```

## Benchmark provenance

The per-bin Mode C architecture was developed and benchmarked in:
`results/mcolon/20260601_label_transfer_method/`

Key files:
- `logistic_label_transfer.py` — original three-mode implementation (A/B/C)
- `run_loeo_logistic_benchmark.py` — LOEO benchmark comparing modes
- `plots/loeo_logistic_macro_f1.png` — Mode C wins at later hpf

The production `core.py` implements Mode C as `model_type="per_bin"`.
