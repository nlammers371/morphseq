# label_transfer — deferred improvements

## 1. P-value integration for global model

**Status:** deferred  
**Date noted:** 2026-06-08

When `model_type="global"`, the single model is applied uniformly across all time bins.
`run_classification` can still be run on the reference to produce per-bin AUROC and
permutation p-values, but the interpretation is less clean: the global model was not
trained on per-bin data, so its discriminability at (say) 18 hpf reflects the whole
reference distribution, not just embryos at that stage.

Options considered:
- Use `run_classification` per-bin p-values as-is with a caveat label on the plot
- Pseudo-time the global model: filter reference to each bin window and re-run
  `run_classification` locally — gives a fair per-bin AUROC but is expensive
- Leave p-value marking as per_bin-only and skip it for the global path

**Decision needed:** which of the above to implement. Currently p-value marking in
`cep290_homo_low_to_high.py` calls `run_classification` on the full reference regardless
of model_type — this is a reasonable approximation but not fully calibrated for the
global model.

---

## 2. Brier score calibration

**Status:** deferred  
**Date noted:** 2026-06-08

The Brier score (mean squared error of predicted probabilities vs true labels) was
suggested as a calibration diagnostic. It would quantify how well the predicted
probability vectors match ground-truth frequencies, complementing the AUROC/pval signal.

Would be most useful as:
- A per-bin Brier score curve in the quality report
- A calibration plot (predicted p vs observed frequency) for reference CV predictions

Not implementing now because: (a) no labeled query embryos to score against, and
(b) the CV quality report already covers per-bin precision/recall adequately for
current needs.

---

## 3. `bin_used` → p-value join in embryo_predictions

**Status:** deferred  
**Date noted:** 2026-06-08

`embryo_predictions` now carries a `bin_used` column (the hpf bin center of the
per-bin model used). The join to per-bin p-values from `run_classification` is
currently done in the calling script (`cep290_homo_low_to_high.py`).

A cleaner API would be to optionally accept a `pval_scores` DataFrame in
`transfer_labels` and stamp `pval` directly onto `embryo_predictions`, so every
downstream consumer gets significance for free.

Blocked on: deciding whether `transfer_labels` should depend on `run_classification`
output (a heavier dependency) or leave that join to the caller.

---

## 4. Per-bin CV quality in `model_type="per_bin"`

**Status:** deferred  
**Date noted:** 2026-06-08

The current quality report (precision/recall/F1 by time bin) uses the global CV
predictions for all modes. When `model_type="per_bin"`, the quality metrics should
ideally come from per-bin LOEO CV — i.e., for each bin, hold out one experiment,
fit the bin model on the rest, predict the held-out bin embryos.

This is the LOEO benchmark from `results/mcolon/20260601_label_transfer_method/`
but wired into `prepare_reference` rather than a standalone script.

Cost: significantly more compute (one CV fit per bin per fold).
