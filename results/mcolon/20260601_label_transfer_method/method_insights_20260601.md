# Label Transfer Method: Insights, Limitations, and Open Questions
**Timestamp:** 2026-06-01
**Status:** Post-first-LOEO-run. Method is running but not yet trustworthy for deployment.

---

## What we found

### 1. `Low_to_High` is catastrophically confused with `Not Penetrant` in the 30–48 hpf window

Global LOEO confusion (all folds combined):
- 98 true `Low_to_High` embryos
- Only 6 predicted correctly as `Low_to_High`
- 82 predicted as `Not Penetrant`

The 30–48 hpf window is likely too early to see the `Low_to_High` phenotype manifest.
Correctly-predicted `Low_to_High` embryos have more images (mean 36 vs 23) and wider hpf
range (13.8 vs 11.0 hrs) than misclassified ones — suggesting that embryos with images
closer to 48 hpf get more signal.

**Current limitation:** Without timepoints beyond 48 hpf, `Low_to_High` is
indistinguishable from `Not Penetrant` by embedding distance alone.

**Proposed fix to try:** Extend max_hpf to 56 or 60 where data supports it and measure
whether `Low_to_High` recall improves. This is the highest-priority experiment.

---

### 2. The `low_density` accuracy inversion is a label composition artifact

In folds 20251106 and 20251113, `low_density` embryos have *higher* accuracy than
`assigned` embryos — which looks like the status flag is broken. It is not.

What is actually happening:
- `low_density` in these folds is almost entirely `High_to_Low` embryos (e.g. 40/41 in
  20251113), which are easy to classify regardless.
- `assigned` in the same folds contains many `Low_to_High` embryos, which are
  systematically mispredicted as `Not Penetrant` (see point 1).

The `low_density` flag is doing its job geometrically (catching embryos far from the
reference distribution). The accuracy comparison is misleading because it conflates label
composition with flag quality.

**Implication:** Scalar accuracy is not an informative summary metric for this label set.
A trivial classifier that always predicts `Not Penetrant` gets ~50–60% accuracy in
some folds.

---

### 3. The distance score threshold is experiment-dependent

Mean `embryo_distance_score` by fold:
- 20251113: 0.097  (very far from reference — 41/92 flagged low_density)
- 20251106: 0.166  (far)
- 20250512: 0.189
- 20251017: 0.279
- 20251212: 0.454
- 20251112: 0.499
- 20251205: 0.540

Two experiments (20251113, 20251106) sit systematically far from the reference
distribution. This could be batch effects, imaging differences, or developmental timing
differences. It is not necessarily a failure of the method — but it means the current
`distance_score_threshold=0.05` catches a lot of real embryos in those experiments.

---

## Open methodological questions

### On accuracy as a metric

**Precision/recall is the right diagnostic tool** — it tells us per-label recall
(fraction of true `Low_to_High` recovered) rather than collapsing everything into a
single number dominated by `Not Penetrant`.

**But:** precision/recall requires knowing the true label. That is only available during
validation. The method is designed for the case where we *don't* have labels — that's
why we need label transfer in the first place.

So precision/recall is a calibration/validation tool only, not something that ships as
part of the method output.

### On what "working" means without ground truth

The method always returns the closest label. The question for an unlabeled query embryo
is: *how do we know if the prediction is reliable without a ground truth to compare against?*

The confidence scores (neighbor agreement, distance score, consistency) are the current
answer — but they do not tell us whether we are doing better than chance.

**The core problem:** For any given prediction, we want to say "this embryo is more
likely `Low_to_High` than would be expected from the reference label distribution alone."
But without knowing the prior, we don't know if a high `Low_to_High` probability is
signal or just a reflection of `Low_to_High` being common in the reference.

### On the null distribution question

What we would really want is:
- For each query embryo, compare the label probability distribution to what you would
  get from a random draw of reference neighbors (i.e., the reference label distribution).
- If the KNN label distribution for a query embryo is not meaningfully different from
  the reference marginal, the prediction is uninformative.

This would require building a null (permutation or marginal) distribution — either:
1. **Marginal null:** Compare the predicted label distribution to the reference label
   frequency distribution. If they match, the prediction adds no information.
2. **Permutation null:** Shuffle reference labels, run KNN, repeat many times. Compare
   real prediction probability to the permuted distribution.

The marginal null is cheap and interpretable. The permutation null is expensive but
principled.

We do not want to build this yet. But it may be necessary if the method is going to
be used on experiments where we genuinely have no validation set.

**The simpler framing:** What we really want from the method is not "is this prediction
correct?" (unknowable without labels) but "is this embryo clearly different from the
reference distribution in a way that points toward one label?" The distance score
gets at this partially, but not in label-space — only in feature-space.

---

## Proposed next experiments (priority order)

1. **Extend time window to max_hpf=56 or 60** — test whether `Low_to_High` recall
   improves. This is the cheapest fix and most likely to matter.
2. **Add per-label precision/recall to LOEO validation summary** — as a diagnostic tool,
   not a method output.
3. **Add per-label accuracy breakdown by status** to the diagnostic output — the scalar
   comparison of `assigned` vs `low_density` accuracy is misleading.
4. **Investigate 20251106 and 20251113 batch distance** — are these experiments
   genuinely different from the others in feature space? PCA or UMAP of all experiments
   together would help.
5. **Consider a marginal null comparison** — for each query embryo, compute KL divergence
   between the predicted label distribution and the reference marginal. Flag embryos
   where this is below a threshold as "not more informative than the reference prior."
