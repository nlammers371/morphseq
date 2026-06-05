# CEP290 Label Transfer — Final Analysis

**Date:** 2026-06-04  
**Author:** mdcolon  
**Status:** Closed. Production path decided.

---

## TL;DR

The morphological embedding encodes the discriminative signal for CEP290 phenotype
labels, but that signal is **global** — spread across the full embedding space — not
local. KNN reads local Euclidean density, which is dominated by `Not Penetrant`
everywhere and cannot recover the rare mixed classes. Multiclass logistic regression
reads the full embedding via a linear decision boundary and finds the discriminative
directions even in mixed, high-density-overlap regions, more than doubling argmax
accuracy on `Intermediate` (13% → 27%) and net-rescuing 613 `Low_to_High` images that
KNN misclassified. Conformal prediction sets add no useful precision on top of that —
the base classifier accuracy is the ceiling, and no uncertainty wrapper can raise it.
The production path is **multiclass logistic argmax + margin-based abstention (τ sweep)**.

---

## What We Set Out to Do

Transfer phenotype labels (`Low_to_High`, `High_to_Low`, `Intermediate`, `Not Penetrant`)
from a labeled CEP290 reference set to new embryos, using morphological embeddings
(`z_mu_b_*` latent features). Validated with leave-one-experiment-out (LOEO) across 7
experiments, ~45k images per method.

The candidate pipeline was:
1. Generate per-image class scores `q` (KNN vote or multiclass logistic)
2. Wrap with APS conformal prediction sets for uncertainty quantification
3. Roll up to embryo-level predictions

---

## Finding 1: The Embedding Has the Signal — But It's Global, Not Local

**The morphological embedding encodes all the information needed to discriminate CEP290
phenotypes, but the discriminative directions are distributed across the full embedding
space. Local Euclidean density is the wrong lens.**

KNN reads local neighborhood majority. In this embedding, `Not Penetrant` embryos (~47%
of the dataset) dominate the local neighborhood of nearly every query point, regardless
of the query's true label. The trajectory labels — especially `Intermediate` and
`Low_to_High` — are defined by dynamic patterns that overlap with NP morphology in
Euclidean distance but are separable via global linear projections.

Key geometry numbers from `q_diagnostic_neighbor_geometry_summary.csv`:

| True label | First same-label neighbor rank | Same-label fraction (top-15) | NP fraction (top-15) |
|---|---:|---:|---:|
| Not Penetrant | 1.7 | 0.82 | 0.82 |
| High_to_Low | 4.2 | 0.69 | 0.12 |
| Low_to_High | 11.9 | 0.22 | 0.57 |
| Intermediate | 22.3 | 0.12 | 0.45 |

For `Intermediate`, the first same-label neighbor on average doesn't appear until rank
22. A KNN-15 window almost never contains a same-label example. KNN is not failing
because the embedding lacks signal — it is failing because it is reading the wrong
property of the embedding (density) rather than the discriminative structure.

**Multiclass logistic regression uses the full embedding via a linear decision boundary
and recovers the discriminative signal even where neighborhoods are mixed.** Because it
optimizes a global objective over all embedding dimensions simultaneously, it can hone in
on the directions that separate `Intermediate` and `Low_to_High` from `Not Penetrant`
without requiring those classes to dominate local density.

Evidence:

| Metric | KNN-q | Multiclass-q |
|---|---:|---:|
| Argmax accuracy | 0.695 | 0.663 |
| Balanced accuracy | 0.483 | **0.529** |
| Macro F1 | 0.477 | **0.519** |
| LtH→NP collapse | 68% | **41%** |
| Intermediate argmax accuracy (all HPF) | ~13% | **~27%** |

KNN's higher raw accuracy is entirely explained by NP collapse — it predicts NP for
almost everything and NP is the majority class. On balanced accuracy, macro F1, and
rare-class recovery, multiclass logistic is strictly better.

Rescue group analysis (`q_diagnostic_rescue_groups.csv`):

| True label | KNN wrong / logistic right | KNN right / logistic wrong | Net rescued |
|---|---:|---:|---:|
| Low_to_High | 613 | 185 | **+428** |
| Intermediate | 161 | 60 | **+101** |
| High_to_Low | 209 | 274 | -65 |

Logistic is not just swapping errors with KNN — it is net-recovering the rare mixed
classes at the cost of some High_to_Low accuracy, which is the right tradeoff given
the analysis goals.

---

## Finding 2: Conformal Prediction Cannot Rescue a Weak Predictor

We implemented APS (Adaptive Prediction Sets) conformal prediction as an uncertainty
wrapper around both the KNN and multiclass `q` scores, calibrated at α=0.10
(target 90% marginal coverage), validated with LOEO.

**The conformal sets did not help. Argmax was unambiguously better.**

The root cause is that conformal prediction sets are a *coverage guarantee*, not a
confidence filter. When the base classifier has only 57% argmax accuracy on a 4-class
problem, achieving 90% coverage requires sets that include 2–3 classes for nearly every
image. Specifically:

- The APS score of the true label is the cumulative softmax probability down to and
  including the true class in sorted order. With 43% of calibration images having the
  true class at rank 2-4, the 90th percentile of true-label APS scores lands at 0.9999.
- qhat calibrates to ~0.999, which means a class is only excluded from the set if the
  cumulative probability of all higher-ranked classes already exceeds 0.999.
- Result: 92% of images get set_size ≥ 2. Singletons (the only "useful" conformal
  output for a point prediction) appear in 7.6% of multiclass-q images and 0% of KNN-q
  images.

**Singleton F1 vs argmax F1** (mean across LOEO folds, multiclass-q):

| Class | Argmax F1 | Singleton F1 (pooled) | Singleton F1 (Mondrian) |
|---|---:|---:|---:|
| Low_to_High | 0.35 | 0.05 | 0.02 |
| High_to_Low | 0.51 | 0.13 | — |
| Intermediate | 0.18 | 0.01 | — |
| Not Penetrant | 0.69 | 0.10 | — |
| **Macro** | **0.44** | **0.07** | **0.02** |

Argmax wins on every class by a large margin. Singleton F1 is near zero because the
singleton rate is near zero — most true-class images are abstained on.

### Why Mondrian Doesn't Fix It

Mondrian conformal (per-class qhat) correctly diagnoses and partially fixes the
subsidization problem: `Not Penetrant` (47% of data) was overcovering and eating the
marginal budget that should go to `Intermediate`. CovGap fell from 0.12 to 0.06 with
Mondrian, and `Intermediate` coverage improved. But the cost was larger sets and a
drop in singleton rate from 7.6% to 0.2%. The model simply doesn't have enough
discriminative information about `Intermediate` to produce tight sets even on its own
calibration budget.

### What Conformal Is Actually Good For Here

Conformal is not useless — it is just answering the wrong question if used as a
predictor. The set size IS a real per-image ambiguity signal:

- `set_size = 1`: the model is strongly committed (high argmax probability margin)
- `set_size = 2-3`: the model sees genuine uncertainty between classes

The right use is flagging, not prediction: tag `set_size > 1` images as low-confidence
for downstream QC, rather than using the set as a prediction.

**Conformal is better suited for problems with larger, well-separated class neighborhoods
or class-balanced datasets. For morphological embeddings with strong class imbalance and
overlapping trajectories, it cannot manufacture signal that the embedding doesn't have.**

---

## Finding 3: The Prior Doesn't Help Either

An earlier investigation attempted to augment the likelihood `P(q | label)` with a
local-region prior `P(label | local region)` in a Bayesian product:

> score ∝ P(q | label) × P(label | local region)

Every non-uniform prior variant — raw local, prevalence-corrected, ring or full
neighborhood — *hurt*. All winning configurations used the uniform prior (i.e., the
prior term was irrelevant).

The reason is structural: the prior was estimated from the same embedding neighborhood
as the likelihood. In a product `likelihood × prior` you need two independent views.
When both are derived from the same NP-dominated Euclidean neighborhood, the prior
amplifies the same contamination that already hurts the likelihood, rather than
correcting it.

---

## Decision: Production Path

**Use multiclass logistic argmax with a margin-based abstention rule (τ threshold).**

This is the standard "selective classification" approach:

- **Predict** the argmax class when `p_top1 - p_top2 ≥ τ`
- **Abstain** (mark as ambiguous) when the margin is below τ

The margin `p_top1 - p_top2` is more robust than raw `p_top1` on miscalibrated models
because it is a relative comparison between the top two competing classes.

τ is calibrated by sweeping the precision-vs-coverage curve and picking the threshold
that achieves the desired precision floor per class. This is the next immediate step.

### What's Not Being Pursued

| Approach | Why not |
|---|---|
| KNN-based label transfer | Geometry doesn't support it — NP dominates all neighborhoods |
| APS conformal sets as predictions | Singleton rate too low to be useful; argmax dominates on F1 |
| Mondrian conformal | Improves CovGap but further reduces singleton rate; same root problem |
| Local-region prior | Dependent on same embedding as likelihood; amplifies NP contamination |
| Time-windowed KNN | Improves local purity but the fundamental geometry issue persists |

---

## Next Steps

### Step 1 — τ sweep: calibrate the margin abstention threshold

Implement the precision-vs-coverage curve sweep over `argmax_margin` (= `p_top1 -
p_top2`). For each candidate τ:
- committed images: those with `argmax_margin ≥ τ`
- per-class precision and recall on committed images only
- coverage = fraction of images committed (1 − abstention rate)

Pick τ at the point where per-class precision hits a target floor (e.g. 85%) across all
classes. This directly answers "when the model commits, how often is it right?"

Code target: `new_files/selective_prediction.py` + `run_tau_sweep.py`  
Key output: `plots/tau_sweep_precision_coverage_per_class.png`

### Step 2 — Embryo-level rollup

Aggregate image-level committed predictions to embryo-level labels. Standard options:
- **Majority vote** over committed images only
- **Soft vote** (mean `q` vector) over committed images, then argmax
- **Coverage gate**: require a minimum fraction of committed images per embryo before
  issuing an embryo label; flag the rest as ambiguous

The τ from Step 1 feeds directly into which images are included in the rollup.

### Step 3 — Integration into label_transfer_core.py

Once τ is chosen, wire it into `label_transfer_core.py` as the production interface:
input reference + query dataframes, output per-embryo labels with confidence flags.
The conformal machinery (`conformal_sets.py`, `coverage_diagnostics.py`) can be retained
as an optional diagnostic module but is not on the prediction path.

---

## File Index

| File | Purpose |
|---|---|
| `run_q_conformal_benchmark.py` | Main LOEO benchmark (KNN-q and multiclass-q, APS conformal) |
| `run_coverage_diagnostics.py` | CovGap + Mondrian re-calibration comparison |
| `new_files/conformal_sets.py` | APS conformal machinery (pure functions) |
| `new_files/coverage_diagnostics.py` | CovGap with Wilson CIs, Mondrian qhat, Mondrian set builder |
| `new_files/CONFORMAL_UPGRADES.md` | Design notes for conformal extensions (Mondrian, TACP, ERT, abstain) |
| `run_q_failure_diagnostics.py` | KNN geometry diagnostics (neighbor rank, rescue groups) |
| `calibration_core.py` | Earlier likelihood × prior calibration experiments |
| `make_report_figure_0[2-6]_*.py` | Figure scripts for each analysis panel |
| `plots/` | All generated figures |
| `q_conformal_benchmark_full_time_image_predictions.csv` | Full image-level predictions with q-scores, set membership |
| `coverage_diagnostics_covgap.csv` | Per-class CovGap table |
| `coverage_diagnostics_mondrian_summary.csv` | Pooled vs Mondrian per-fold comparison |
