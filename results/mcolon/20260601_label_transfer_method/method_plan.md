# Plan: CEP290 Phenotype Label Transfer Method

## Context
We need to assign phenotype labels to new CEP290 embryo data using a labeled reference dataset. The method uses distance-weighted K-nearest neighbors on any numeric feature columns to transfer labels from a reference set to a query set, aggregating image-level evidence up to embryo-level predictions with confidence scoring. This is a new module — no existing label transfer code exists.

All development goes in: `results/mcolon/20260601_label_transfer_method/`

---

## Reference Data
- **File**: `results/mcolon/20251229_cep290_phenotype_extraction/final_data/embryo_data_with_labels.csv`
- **Key columns**: `snip_id` (globally unique per image), `embryo_id`, `experiment_id`, `predicted_stage_hpf`, `cluster_categories`, `z_mu_b_*`, `z_mu_n_*`, morphology features
- **Labels**: `Low_to_High`, `High_to_Low`, `Intermediate`, `Not Penetrant` (NaN rows excluded from reference)
- **Experiments**: 20250512, 20251017_combined, 20251106, 20251112, 20251113, 20251205, 20251212

---

## Implementation Plan: Two Passes

### Pass 1: MVP Label Transfer (`label_transfer_core.py`)

Goal: produce `embryo_pred_label_dict` from reference + query DataFrames.

**Steps**:

1. **Input validation / missing-feature handling**
   - Verify all `feature_cols` exist in both DataFrames.
   - Drop query rows where any `feature_cols` value is NaN. Track those embryos → `status=not_evaluated`, `status_reason=missing_features`.

2. **Time-window filter**
   - Keep rows where `min_hpf ≤ predicted_stage_hpf ≤ max_hpf` (default 30–48).
   - Apply to both reference (also drop NaN labels) and query.
   - Track query embryos with no remaining images → `status=not_evaluated`, `status_reason=outside_time_window`.

3. **KNN search**
   - Fit `sklearn.neighbors.NearestNeighbors(n_neighbors=k, metric=metric)` on reference `feature_cols`.
   - Transform query rows to get distances + indices.
   - `feature_cols` has no default — always specified by caller.

4. **Neighbor long table**
   - One row per (query snip × neighbor).
   - Columns: `query_embryo_id`, `query_snip_id`, `query_hpf`, `neighbor_rank`, `ref_snip_id`, `ref_embryo_id`, `ref_label`, `ref_hpf`, `distance`, `weight = 1/(distance + epsilon)`.
   - Optional: `ref_experiment_id`, `query_experiment_id` if `experiment_col` provided.
   - QC checkpoint: inspect distance distributions; check for suspiciously zero distances.

5. **Image-level label probabilities**
   - Group by `query_snip_id` (globally unique); keep `query_embryo_id` as metadata.
   - Weighted vote per label: `P(label) = sum(weights for that label) / sum(all weights)`.
   - Output: long format, one row per (snip × label), probabilities sum to 1 per snip.
   - QC checkpoint: confirm probabilities sum to 1; spot-check a few images.

6. **Image prediction summary**
   - `image_pred_label = argmax(probs)`, `image_neighbor_agreement = max(probs)`.

7. **Embryo-level aggregation**
   - Mean `image_label_probability` across images per embryo per label.
   - `embryo_pred_label = argmax`, `top_label_probability = max`.
   - Output long-format `embryo_label_probabilities`.

8. **Return dict (Pass 1 minimum)**:
   ```python
   {
       "neighbor_long_table": ...,
       "image_label_probabilities": ...,
       "image_prediction_summary": ...,
       "embryo_label_probabilities": ...,
       "embryo_label_transfer_summary": ...,
       "embryo_pred_label_dict": ...,
   }
   ```

---

### Pass 2: Confidence Scoring + Status + LOEO Validation

**Confidence components** (added to `label_transfer_core.py`):

- **`mean_image_neighbor_agreement`**: mean of `image_neighbor_agreement` across images per embryo.
- **`embryo_distance_score`**:
  - Fit NearestNeighbors on reference with `n_neighbors=k+1`. Query reference itself. Drop self-match (rank 0, distance≈0). Mean distance to remaining `k` neighbors per reference image → reference density distribution.
  - For each query image: compute `mean_knn_distance`, percentile-rank against reference distribution → `distance_in_distribution_score = 1 - (percentile/100)`. Average per embryo.
- **`embryo_consistency_score`**: fraction of images where `image_pred_label == embryo_pred_label`. Set to 1.0 for single-image embryos (do not over-interpret; inspect `n_images`).
- **`embryo_confidence = mean_image_neighbor_agreement × embryo_distance_score × embryo_consistency_score`**
- QC checkpoint: inspect distributions of each component independently before computing aggregate.

**Diagnostic boolean flags** (preserved alongside compact status):
- `is_low_density`, `is_low_agreement`, `is_low_consistency`, `is_low_top_probability`

**Status assignment** (priority order):
```
not_evaluated  → no images in window, missing features, or other failure
low_density    → is_low_density
ambiguous      → is_low_agreement OR is_low_consistency OR is_low_top_probability
assigned       → otherwise
```

**Default thresholds** (development placeholders — tune after LOEO):
| Threshold | Default | Notes |
|-----------|---------|-------|
| `agreement_threshold` | 0.5 | |
| `consistency_threshold` | 0.6 | |
| `distance_score_threshold` | 0.05 | Conservative start — 0.25 would be aggressive |
| `top_probability_threshold` | 0.4 | |

**Full embryo summary columns**:
`query_embryo_id`, `n_images`, `min_query_hpf`, `max_query_hpf`, `query_hpf_range`, `predicted_label`, `top_label_probability`, `mean_image_neighbor_agreement`, `embryo_distance_score`, `embryo_consistency_score`, `embryo_confidence`, `is_low_density`, `is_low_agreement`, `is_low_consistency`, `is_low_top_probability`, `status`, `status_reason`

---

### File 2: `run_loeo_validation.py`

1. Load `embryo_data_with_labels.csv`. Caller selects `feature_cols` (e.g. `z_mu_b_*`).
2. Drop rows with NaN `cluster_categories`.
3. **Before joining ground truth**: assert true labels are constant per embryo (warn or raise if any embryo has multiple labels in the data).
4. For each of 7 experiments: hold it out as query (labels withheld from `run_label_transfer()`), rest as reference. Run `run_label_transfer()`. Join true labels back after prediction returns.
5. Save: `leave_one_experiment_out_predictions.csv`, `leave_one_experiment_out_summary.csv`.
6. Print per-fold QC: status breakdown, `accuracy_all_evaluated`, `accuracy_assigned_only`.

---

### File 3: `plot_loeo_results.py`
Matplotlib only.
- Per-experiment accuracy (all evaluated vs assigned only)
- Confusion matrix (all evaluated; assigned only)
- `embryo_confidence` distribution by correctness
- Accuracy by confidence bin
- Accuracy by status
- `n_images` distribution

---

## Helper Function
```python
def add_label_transfer_predictions(query_df, embryo_summary_df, embryo_col,
    label_col_out="predicted_label",
    confidence_col_out="label_transfer_confidence",
    status_col_out="label_transfer_status") -> pd.DataFrame
```

---

## Dependencies
- `sklearn.neighbors.NearestNeighbors`
- `pandas`, `numpy`, `scipy.stats`
- `matplotlib`

---

## QC Checkpoints (Exploratory)
1. Neighbor table: distance distributions, any zero distances?
2. Image probabilities: sum to 1, any single-label-dominated images?
3. Confidence components: spread vs clumped?
4. LOEO: does `accuracy_assigned_only ≥ accuracy_all_evaluated`? If not, thresholds need revision.
5. Confusion matrix: which pairs are most confused?
6. Low-density embryos: are their distances visibly higher than reference distribution median?
