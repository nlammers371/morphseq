# Handoff: Label Transfer Architecture + CEP290 Spectrum Plots

**Date:** 2026-06-08  
**Owner:** mdcolon  
**Context:** This session made architectural changes to `label_transfer/core.py` and extended the CEP290 homozygous phenotype analysis plots. A future model picking this up needs to understand both the internal change and the current plot state.

---

## 1. Critical Change to `label_transfer/core.py`

### What changed
`prepare_reference()` now uses **within-bin embryo aggregation** for the per-bin models instead of across-all-time embryo medians.

**Before:** The entire reference was collapsed to one row per embryo (mean features, median hpf across the embryo's full imaging lifetime). Per-bin models then sliced this table by each embryo's median hpf. An embryo imaged from 20–80 hpf had median ~50 hpf and only contributed to the ~50 hpf bin, even though it had real data at 20 hpf.

**After:** For each `bin_width`-hpf window, raw image rows are filtered to that window and collapsed to embryo means using `_aggregate_binned` from `analyze.classification.engine.data_prep`. An embryo spanning 20–80 hpf now contributes independently to every bin it has images in, with features computed only from images in that window. This is exactly what `run_classification` does internally.

### Why this matters
The old approach was statistically unprincipled for embryos imaged across wide time ranges — averaging features across 60 hours of development and then calling that "the 50 hpf embryo" loses all temporal structure. The within-bin approach is the correct unit of measurement for a time-binned model.

### Shared engine — no duplicated logic
The change imports two functions from the classification engine:
```python
from ..engine.data_prep import _aggregate_binned
from ...utils.binning import add_time_bins
```
These are the same functions `run_classification` uses, so the binning logic is not duplicated.

### Observable effect
Per-bin models fitted jumped from **15 → 33** (the reference now spans 12–140 hpf at the image level, as expected). The homozygous sequenced embryo split changed from `High_to_Low=60 / Low_to_High=20` to `High_to_Low=33 / Low_to_High=47` — the old model was dominated by late-timepoint embryo-mean features.

### Future improvement noted in `IMPROVEMENTS.md`
`run_classification` should expose its fitted per-bin pipelines so `label_transfer` can reuse them directly instead of refitting. Currently the pipelines are trained inside the loop and discarded.

---

## 2. `_ref_cv_probs()` — also fixed

`_ref_cv_probs` in `cep290_homo_low_to_high.py` generates LOEO CV predictions for reference embryos, used in the bottom row of the comparison plots. It had the same bug: it called `_embryo_table` (lifetime median hpf) and so produced zero reference embryos near 18/24 hpf.

**After fix:** Uses `_aggregate_binned` on raw image rows, producing one row per `(embryo, time_bin)`. An embryo spanning multiple bins now contributes a prediction to each bin — same logic as the per-bin model training. The time column in `ref_cv` is now `time_bin_center` (float, hpf bin center), not `predicted_stage_hpf`.

---

## 3. Current Plot State

All plots save to:
```
results/mcolon/20260605_sci_cilia_qc_first_pass/plots/sequenced_focus/cep290/homozygous_focus/
```

### `cep290_homo_low_to_high_homo_only_probability_spectrum_sequenced.png`
The main spectrum plot. Three rows of sequenced genotype groups × four stage columns (18/24/30/48 hpf). X-axis is P(Low_to_High). Bins with permutation p≤0.05 get a dark border and `*` in the title. Uses per-bin model predictions.

**Known issue / open question:** The AB→wildtype and cep290_wildtype rows are present but arguably noisy — they're not the primary target of a homo-only model. Consider dropping them or moving to a separate plot.

### `cep290_homo_low_to_high_spectrum_bottom_row_options.png`
A **design comparison figure** — 4 rows × 4 columns. Use this to decide which bottom row format to adopt for the final spectrum plot. The user has not yet made a final choice.

- **Row 0 (Query):** Sequenced homozygous embryos, P(Low_to_High) strip, probability-colored. Same as the main spectrum.
- **Row 1 (Option A):** Reference LOEO CV embryos split by true class (High_to_Low strip at y=0, Low_to_High at y=1). Colored by P(Low_to_High) via the same colormap. **Alpha=0.35** so overlapping dots accumulate visually, revealing density. A dense correct cluster becomes a saturated blob; isolated wrong calls stay faint.
- **Row 2 (Option B):** Stacked accuracy bars per true class (green=correct, red=wrong at 0.5 threshold). Shows classification accuracy but loses probability distribution.
- **Row 3 (Option C):** Violin plots of P(Low_to_High) per true class, clipped to [0,1]. Shows full distribution shape per class per stage. **User preferred this over the previous calibration scatter.**

**Next decision:** Pick one of A/B/C for the final spectrum plot and merge it into `_plot_probability_spectrum`.

### `cep290_homo_low_to_high_global_vs_perbin_model_comparison.png`
2×4 grid (rows = query/reference, columns = 18/24/30/48 hpf). X-axis = global model P(High_to_Low), Y-axis = per-bin model P(High_to_Low). Shows where the two models disagree. Top row is sequenced homozygous query embryos; bottom row is reference LOEO CV embryos.

**At 18/24 hpf:** Query top row shows strong disagreement — global model calls nearly everything High_to_Low (x≈1), per-bin model calls most Low_to_High (y≈0). Reference bottom row is still sparse at 18/24 hpf because the reference experiments don't have many embryos at those early stages (the reference is mostly 30–107 hpf).

### `cep290_homo_low_to_high_perbin_model_reference_quality_timebin.png`
Precision and recall by time bin for the per-bin model from LOEO CV. Shows where in developmental time the model is reliable vs unreliable.

---

## 4. Key Constants and Conventions

- `PHENO_ORDER = ["High_to_Low", "Low_to_High"]` — index 0 = High_to_Low, index 1 = Low_to_High
- X-axis convention: `prob_High_to_Low` as the axis variable (0 = Low_to_High side, 1 = High_to_Low side) in the comparison plot; `prob_Low_to_High` in the spectrum plot
- `STAGE_GRID = [18, 24, 30, 48]` — the four target hpf stages for cep290
- Permutation p-values come from `run_classification` called on reference embryo means; p≤0.05 → significant bin → dark border in spectrum plot
- `seq_ids` = embryo IDs confirmed by sequencing (from the `sequenced` sheet bypass, value > 0)

---

## 5. Files Modified This Session

| File | Change |
|---|---|
| `src/analyze/classification/label_transfer/core.py` | Per-bin model training now uses within-bin `_aggregate_binned`; imports `add_time_bins` + `_aggregate_binned` from engine |
| `src/analyze/classification/label_transfer/IMPROVEMENTS.md` | Added note: `run_classification` should expose fitted per-bin pipelines |
| `results/mcolon/20260605_sci_cilia_qc_first_pass/cep290_homo_low_to_high.py` | Fixed `_ref_cv_probs` (within-bin aggregation); added `_plot_spectrum_with_accuracy` (design comparison); updated Option A alpha, Option C to violin; fixed `time_bin_center` column name in comparison plot |
