# CONFORMAL_UPGRADES.md

Companion to `conformal_sets.py`. That file is the lean, defensible MVP:
**split-conformal APS prediction sets on a KNN vote, with a density gate.**
This document is the map of everything we deliberately left *out* of it --
why, when you'd add it, and where it hooks in. Treat it as the "extra sauce"
drawer: open a section only when the MVP's own diagnostics tell you to.

The governing principle throughout: **measure first, escalate only on
evidence.** Every upgrade below trades simplicity (and often a clean
coverage guarantee, or variance) for a fix to a *specific* failure. Do not
pre-build them. Build the MVP, read its coverage report, and add the one
upgrade the report justifies.

---

## 0. The MVP, in one paragraph

Three data roles (all supplied by the caller): **reference** (KNN neighbor
pool), **calibration** (sets the threshold `qhat`), **query** (gets sets).
`calibrate_conformal` builds the KNN on the reference pool, scores each
calibration point's *true label* with APS, and takes the
`ceil((n+1)(1-alpha))`-th order statistic as `qhat`. `predict_conformal`
runs each query through: feature filter -> density gate -> KNN vote -> APS
scores -> `{y : s_y <= qhat}`. Marginal coverage `1-alpha` is guaranteed
under exchangeability of calibration and query points. Per-class coverage is
**not** guaranteed -- that's where most of the upgrades below come in.

---

## 1. Why APS (and what it does / doesn't protect)

APS = Adaptive Prediction Sets. The nonconformity score of a label is the
cumulative probability mass from the top of the ranking down to that label.
Calibrate a threshold on true-label scores; include every label below it.

- **What APS protects against:** overconfidence at *ambiguous points*. The
  set grows where the probability vector is flat (mushy region -> several
  labels accumulate before crossing `qhat`) and shrinks to a singleton where
  it's peaked. Set size becomes an honest, per-point ambiguity signal.
- **What APS does NOT touch:** class *imbalance*. APS adapts to per-point
  uncertainty (shape of `q`), not to class frequency. A single pooled `qhat`
  dominated by an abundant easy class can sit too low for a rare hard class,
  which then undercovers while marginal coverage still reads fine. The fix
  for that is in sections 3-5, not in the score.

These are orthogonal jobs: APS handles "this point is ambiguous"; the
imbalance tools handle "this class is rare."

### 1a. The APS-pins-near-1.0 effect (seen in the demo)

When the KNN is *highly confident* (well-separated classes -> all k
neighbors agree -> top class probability ~1.0), the top (true) label's own
APS score is ~1.0, so the whole true-label score distribution clusters near
1.0 and `qhat` -> 1.0 (degenerate full sets). This is **not a bug** -- it's
APS being conservative on near-deterministic data, where singletons are
correct anyway. It only looks degenerate; with any real overlap the scores
spread out and `qhat` becomes informative. Two practical notes:
- `knn_probabilities` applies a tiny Laplace `smoothing` (default 1e-3) so a
  class with *zero* neighbor votes doesn't get a hard 1.0 score purely from
  being unreachable. Set `smoothing=0` to disable.
- If you genuinely operate in a high-confidence regime and still want
  non-trivial sets, that's a sign you want **randomized APS** (below) for
  exact (non-conservative) coverage, or simply that singletons are right.

### 1b. include_last_label

The MVP uses `include_last_label=True`: keep the label that tips the
cumulative sum over `qhat`, never emit an empty set. Coverage is then
slightly *conservative* (>= 1-alpha), deterministic, seed-free, easy to
defend. The alternatives:
- `False`: exclude the tipping label -> can produce empty sets.
- `"randomized"`: coin-flip on the tipping label based on how far past
  `qhat` you landed -> gives *exact* 1-alpha coverage, at the cost of
  seed-dependence. Adopt only if a reviewer wants exactness or if 1a's
  conservativeness is genuinely hurting set sizes.

---

## 2. Calibration data efficiency: split vs cross-conformal vs LOO

The MVP uses **split conformal** (one fixed reference/calibration split).
Cleanest exact guarantee, leanest code, but it "spends" data: only the
calibration slice informs `qhat`, so a small calibration set gives a
high-variance threshold.

Upgrade path if that variance bites:
- **Cross-conformal / k-fold:** fold the reference data, calibrate each fold
  against the rest, pool the scores. Uses all data for calibration; lower
  variance. Can in principle lose exact validity (in practice usually fine).
- **Jackknife+ / LOO:** the k=n extreme. Most data-efficient.

**Why this is unusually cheap for us:** the usual cost of cross-conformal is
*model retraining per fold*. KNN has no training -- "refitting" is just
changing the neighbor pool -- so cross-conformal/LOO cost almost nothing
here. LOO is also natural: it's exactly the "exclude the point from its own
neighbor set" discipline, applied to every reference point.

**Trigger to upgrade:** a *stability check*. Reshuffle the
reference/calibration split a few times, recompute `qhat`, look at its
spread. Tight -> split is fine. Wide (or any per-class calibration n is
tiny) -> go to cross-conformal. Standard split sizes: ~70% reference,
~15% calibration, ~15% test, stratified on the label.

---

## 3. Diagnosing class imbalance: per-class coverage, CovGap, ERT

Before any imbalance *fix*, you need the *diagnosis*. The MVP already emits
the cheap version; ERT is the powerful version for small classes.

### 3a. CovGap (in the MVP, as `per_class_coverage`)
For each class c, coverage_c = fraction of true-c test points whose set
contains c. `gap_c = coverage_c - (1-alpha)`; negative = undercoverage
(subsidized), positive = overcoverage (subsidizing). `CovGap` = mean
absolute gap across classes. Marginal coverage can look healthy while a rare
class craters -- always read the per-class numbers, never just the marginal.

**Caveat:** per-class coverage on a tiny class (n~25) is a noisy Bernoulli
average. Put a **Wilson confidence interval** on each coverage_c and only act
on a class whose interval's upper end is still below target. Otherwise you'll
"fix" sampling noise.

### 3b. ERT (Excess Risk of the Target coverage) -- the powerful diagnostic
A data-driven generalization of CovGap. Define the per-point coverage
residual `e_i = 1[y_i in C(x_i)] - (1-alpha)`. If conditional coverage held
everywhere, `e_i` would be mean-zero over *every* subpopulation. So: fit a
model `g(x)` to predict `e_i` from features; if it beats the constant
predictor (the excess risk is positive), there's a region where coverage is
broken, and inspecting `g` shows *where*.

- It's a diagnostic/auditor, **not** a set modifier. It never touches `C(x)`.
  It requires true labels (computes `e_i`), so it's offline evaluation, never
  a per-query gate.
- Why it beats CovGap at small n: CovGap strands each class's estimate inside
  its own tiny bucket; ERT pools information across *all* test points to find
  the violated region, borrowing strength from the geometry. The advantage is
  largest exactly when the violated class is small relative to the whole.
- **Discipline:** fit `g` on one split, evaluate the excess risk on another
  (honest splitting), or `g` will hallucinate violations.
- We already do classification on this platform, so the residual-prediction
  model is cheap infrastructure-wise. The cleverness is the *target* (the
  coverage residual), not a fancier learner.
- An open-source ERT package exists -- prefer it over reimplementing.

**When to use which:** always run CovGap (it's free). Reach for ERT only when
a decision-relevant gap is too close to call by eye -- ERT's power matters at
the margin. If a gap is stark (0.65 vs 0.90), CovGap + Wilson already settles
it.

---

## 4. Fixing class imbalance: Mondrian, clustered, TACP

All of these change **how `qhat` is computed** (in `calibrate_conformal`),
not the score and not the prediction step. Pick based on the diagnosis.

### 4a. Mondrian (class-conditional) conformal
Calibrate a *separate* `qhat[c]` from each class's own true-label scores;
a label is judged against its class's threshold. Upgrades the guarantee from
marginal to ~per-class. **Cost:** each class calibrates on a fraction of the
data; rare classes get high-variance thresholds. Only use when undercovered
classes have enough calibration n (warn below, say, 50-100).

### 4b. Clustered conformal (Ding et al. 2023)
Middle path: cluster classes by the shape of their score distributions, take
one genuine quantile per *cluster*, a class inherits its cluster's threshold.
Borrows strength across similar classes; keeps a real quantile (guarantee
intact, cluster-conditional exactly, class-conditional approximately). Built
for the *many-classes* regime (100-1000). At K=4 it mostly collapses toward
pooled or full-Mondrian depending on per-class n -- i.e. it auto-makes the
pooled-vs-Mondrian decision -- but the clustering does little real work.
Available in `TorchCP` (`predictors.cluster`).

### 4c. TACP / sTACP (Liu, Huang & Ong 2025) -- the right fit for a rare tail
Tail-Aware Conformal Prediction. Adds a penalty to the score for *head-class*
labels that are *low-ranked* at a given point:

    s_TACP(x, y) = s(x, y) + lambda * 1[y in head] * (rank_x(y) - k_r)^+

This pushes implausible head labels out of sets, *lowering head coverage*;
to keep marginal coverage at target the threshold relaxes, which *raises tail
coverage*. There's a fixed marginal-coverage budget; TACP taxes the head's
overspend and the freed budget flows to the tail. Crucially it's still an
ordinary score fed to an ordinary quantile, so the **marginal guarantee is
preserved** -- the fix lives entirely in the score, not in a (guarantee-
breaking) threshold hack. `sTACP` replaces the hard head/tail indicator with
a smooth frequency-graded penalty, balancing *all* classes, not just
head-vs-tail.

- **This is the natural fix for our setup** (one rare hard class, e.g. Int)
  -- better suited than clustered conformal, which targets many classes.
- **Scale caveat:** TACP is built/tested on 100-1000 class long-tail
  benchmarks; gains are real but modest, and the penalty acts on label
  *ranks*, which are coarse at K=4 (`k_r=2` only ever touches ranks 3-4).
  For four classes, implement the *idea* directly -- a small frequency-graded
  additive penalty on the APS score, then one global quantile -- rather than
  importing the full ranked machinery. ~15 lines.

**Hook in code:** the penalty is added inside `aps_scores` (marked
`# UPGRADE: TACP`), and the resulting scores flow through the existing
`aps_quantile` unchanged.

### 4d. The escalation policy (collapse the decision tree into one rule)
```
run pooled APS (the MVP)
compute per-class coverage + Wilson CIs + CovGap   (per_class_coverage)
if no class undercovers beyond tolerance:        -> ship pooled.
elif undercovered class has enough calibration n:-> add TACP/sTACP (4c)
                                                    (or Mondrian 4a if you
                                                     want strict per-class)
else (undercovers AND tiny n):                   -> NOT a threshold problem;
                                                    flag "needs more data for
                                                    class c". Do not paper
                                                    over it.
```
The honest branch is the last one: sometimes the rare class is data-starved
and *no* recalibration fixes it -- you need more reference embryos, not a
fancier `qhat`.

---

## 5. The density gate: global percentile vs local density ratio

The MVP gate (`density_flags`) is heuristic and **global**: a query is
unsupported if its mean kNN distance exceeds the 95th percentile of the
reference set's own LOO mean neighbor distances. No coverage guarantee --
it's a support filter that runs *before* labeling, separating *epistemic
poverty* (far from reference) from *aleatoric mixture* (close but mixed). It
catches what APS structurally can't: APS reads the *shape* of `q`, never the
absolute distances, so an unsupported point with a coincidentally peaked `q`
would otherwise get a confident (wrong) singleton. The gate is what stops
that.

**Upgrade (local density ratio):** a single global distance cutoff is too
strict in legitimately sparse regions and too loose in dense ones. Replace it
with a *local* density estimate -- compare the query's kNN density to the
reference density *of its own neighborhood* (a kNN density ratio), so "far"
is judged relative to how far reference points in that region normally sit.
Same instinct as Mondrian (judge locally, not on a global average), applied
to support instead of coverage. Add only if the global gate misfires at
sparse-but-valid edges. Hook: `# UPGRADE` marker in `density_flags`.

---

## 6. The abstain rule (a different objective from conformal)

Not in the MVP, and worth being clear it's a *different question*, not a
variant. Conformal never abstains -- it widens. The abstain rule does the
opposite: commit to the top label only if `max_y q_y >= tau`, else emit
`low_confidence`. It optimizes singleton *precision* at the cost of throwing
away ambiguous points -- "if you're not certain, drop it." Calibrate `tau` to
a precision target (e.g. top-1 accuracy >= 95% among non-abstained), not by
hand. The metric that keeps it honest: *coverage of the true label among
abstained points* -- how much real signal you're discarding. Use this when
you'd rather have clean singletons + an explicit "don't know" than guaranteed
coverage with wide sets. It does **not** handle support (that's still the
density gate's job -- a confidently-voted unsupported point would pass the
`tau` test; gate it first).

---

## 7. Testing strategy

Not implemented yet; this is what each test should *check*. Generate small
synthetic data where the answer is known (Gaussian blobs with controllable
overlap), plus a few hand-built vectors.

- `knn_probabilities`: rows sum to 1; shape (n, K); a point inside one
  cluster gets ~all mass on that class; smoothing removes hard zeros.
- `density_flags`: a point planted far from all reference is flagged True;
  a point inside the cloud is False.
- `aps_scores`: equals the cumulative sorted mass (check vs a hand-computed
  vector); top label's score == its own probability; scores monotone in rank.
- `aps_quantile`: returns the `ceil((n+1)(1-alpha))`-th order statistic --
  hand-check the rank on a tiny array (this is the off-by-one guard, the
  single most common coverage bug).
- `build_sets`: returns exactly the labels with score <= qhat; never empty;
  singleton when one label dominates.
- `per_class_coverage`: on data with controlled truth, recovers the right
  per-class rate; on a calibrated Gaussian-blob set, marginal coverage lands
  near 1-alpha (the end-to-end correctness check).

Two correctness traps that cause *silent* coverage failure (test for both):
1. **Neighbor leakage** -- a calibration/query point must never be in its own
   neighbor pool. Enforced structurally here (reference is a separate arg),
   but assert it if you ever merge the paths.
2. **Rank off-by-one** -- coverage consistently a few points low usually means
   `n` instead of `n+1`, or `floor` instead of `ceil`, in `aps_quantile`.

Optional oracle: run the MVP's APS against MAPIE's APS
(`method="aps"` / `"cumulated_score"`) on identical toy data and assert the
sets match. Use MAPIE as a *cross-check*, not a dependency -- its API is
mid-migration across versions, so pin a version if you do this.

---

## 8. Quick reference: what changes where

| Upgrade                | Changes which step           | Guarantee impact                |
|------------------------|------------------------------|---------------------------------|
| Randomized APS (1b)    | `build_sets`                 | exact instead of conservative   |
| Cross-conformal (2)    | `calibrate_conformal`        | ~valid (usually); lower variance|
| CovGap / ERT (3)       | none (diagnostic only)       | none -- it only measures        |
| Mondrian (4a)          | `calibrate_conformal` (qhat) | marginal -> ~per-class          |
| Clustered (4b)         | `calibrate_conformal` (qhat) | cluster-cond. exact             |
| TACP / sTACP (4c)      | `aps_scores` (+penalty)      | marginal preserved; tail fairer |
| Density ratio (5)      | `density_flags`              | none (gate is heuristic anyway) |
| Abstain (6)            | replaces `build_sets` logic  | precision target, not coverage  |

Keep `conformal_sets.py` pure. Every row above is a localized change to one
named function -- which is the whole reason the helpers are kept lean and
single-purpose.
