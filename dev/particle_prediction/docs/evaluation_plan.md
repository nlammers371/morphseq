# MorphSeq Beta Predictor: Evaluation Plan

## 1. Goal

Evaluation for the beta model should be:

- efficient,
- interpretable,
- visually grounded,
- and aligned with the actual use case.

The point is to answer:

> Does the model predict future morphology better than trivial baselines, and do its forecast clouds look sensible?

This is not the stage to optimize formal probabilistic scores.

---

## 2. Evaluation philosophy

The predictive core operates in arc-length steps, so the primary evaluation units are **morphological-progress steps**, not minutes or hpf.

The evaluation stack should therefore prioritize:

- next-step geometric accuracy,
- rollout accuracy versus horizon,
- comparison of forecast clouds to observed futures,
- and visual diagnosis of when/why the model fails.

---

## 3. Evaluation task types

## 3.1 One-step next-state prediction

Given a query ending at resampled state \(z_i\), predict \(z_{i+1}\).

Run in both:

- snapshot mode
- history-conditioned mode

This is the cheapest and most sensitive sanity check.

## 3.2 Multi-step rollout prediction

Given a query ending at \(z_i\), roll forward for horizons:

\[
1, 2, 4, 8, \dots
\]

or any small fixed set appropriate to the resampled trajectory lengths.

This tests compounding error and branch preservation.

## 3.3 Matching ablations

At minimum compare:

- no history reranking
- default ordered-segment history reranking
- optional fast summary matching

This will tell you whether the history machinery is actually helping.

## 3.4 Baseline comparisons

At minimum include:

1. persistence baseline
2. linear extrapolation baseline
3. local empirical transition model without history
4. full local empirical transition model with history
5. optional fast summary-matching variant

---

## 4. Primary metrics

## 4.1 One-step Euclidean error

For one-step predictions:

\[
e_1 = \left\lVert \hat z_{i+1} - z_{i+1}^{\text{true}} \right\rVert
\]

If using particles, compute both:

- mean-prediction error
- nearest-sample-to-truth distance if desired

The mean-prediction error should be the default reported scalar.

## 4.2 Endpoint error by rollout horizon

For horizon \(h\):

\[
e_h = \left\lVert \hat z_{i+h} - z_{i+h}^{\text{true}} \right\rVert
\]

Aggregate by mean, median, and a few quantiles across queries.

This is the main rollout metric.

## 4.3 Average displacement error (optional but useful)

For a rollout of length \(H\):

\[
\text{ADE} =
\frac{1}{H}\sum_{h=1}^{H}
\left\lVert \hat z_{i+h} - z_{i+h}^{\text{true}} \right\rVert
\]

This captures whole-rollout quality, not just the final endpoint.

## 4.4 Truth-in-cloud distance (sample-based)

If forward samples / particles are available, compute:

\[
d_{\text{cloud}} =
\min_{p \in \text{particles}}
\left\lVert p - z_{i+h}^{\text{true}} \right\rVert
\]

This helps when the predictive distribution is genuinely multimodal and the mean is misleading.

This is optional for v1 but recommended.

## 4.5 Support diagnostics

For each prediction also store:

- candidate count
- search radius or neighborhood size
- effective sample size / weight entropy
- history mismatch of selected transitions

These are not metrics of accuracy, but they are important explanatory covariates.

---

## 5. Reporting views

## 5.1 Summary tables

For each model / ablation report:

- one-step mean error
- endpoint error by horizon
- ADE if used
- support statistics

## 5.2 Horizon curves

Plot mean and median endpoint error versus horizon for all baselines.

This should be one of the main tracking plots.

## 5.3 Snapshot vs history-conditioned comparison

Plot side-by-side or overlaid curves showing the effect of adding history matching.

## 5.4 Error versus support

Plot prediction error against:

- candidate count
- history mismatch
- search radius

This will tell you whether errors are mostly support-limited.

---

## 6. Visual evaluation

Visual evaluation is not secondary here. It is required.

## 6.1 One-step local prediction plots

For selected queries, show:

- query context
- local candidate states
- candidate next-step increment cloud
- sampled next-step predictions
- observed true next point

## 6.2 Rollout fan plots

For selected embryos and horizons, show:

- observed context
- forward particles / samples
- predictive mean
- observed future trajectory

These should be generated for both good and bad cases.

## 6.3 Matching diagnostic plots

For selected queries, show:

- candidate ordering by position-only search
- reordering after history matching
- comparison of default versus fast history mode

---

## 7. Data splits for the beta

Do not overcomplicate this at first.

Recommended initial splits:

### Split A — within-dataset held-out embryos
Hold out random embryos while keeping classes represented.

Purpose:
- basic interpolation sanity check

### Split B — class-restricted / class-prior ablation
Evaluate with different class-prior settings when relevant.

Purpose:
- see whether class-aware weighting materially helps

### Split C — longer-horizon stress cases
Use the same held-out embryos but emphasize longer rollouts.

Purpose:
- understand graceful degradation

A later stage can add more ambitious perturbation generalization tests.

---

## 8. Notebook outputs

The evaluation notebook should always include:

- a summary table
- horizon-error plots
- at least a few rollout fan examples
- a few failure-case diagnostics

If a metric improves but the fans obviously look worse, trust the visuals and inspect further.

---

## 9. What not to optimize yet

For v1, do not center evaluation around:

- Gaussian NLL
- calibration curves
- formal likelihood ratios
- complex distributional metrics

Those may come later. Right now the system should answer a simpler question:

> Does this predictor put the embryo in the right place, and does the forecast spread look biologically plausible?
