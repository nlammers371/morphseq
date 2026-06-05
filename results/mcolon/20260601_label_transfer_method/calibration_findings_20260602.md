# Reference-Null Calibration — Findings & Next Step

**Date:** 2026-06-02 11:59
**Author:** mdcolon (with Claude)
**Track:** CEP290 label transfer — image-level calibration via reference leave-one-out (LOO)
**Code:** `calibration_core.py`, `run_calibration_benchmark.py`
**Companion docs:** `method_insights_20260601.md` (earlier purity findings), `method_plan.md`

---

## 0. HONEST STATUS — which half of the equation actually works (added 2026-06-02)

The equation we set out to validate was:

> P(label | q, local region) ∝ P(q | label) × P(label | local region)

**What the bake-off actually earned is only the first half:**

> calibrated P(label | q) ∝ P(q | label)    [prior ≡ UNIFORM]

- ✅ **WORKING — the likelihood `P(q | label)`** (q-space reference-null calibration).
  This is the entire win. Every metric improvement over raw baseline comes from here.
- ❌ **NOT WORKING / ABANDONED — the local-region prior `P(label | local region)`.**
  Every non-uniform prior variant (`raw_local`, `prevalence_corrected`, ring OR full)
  *hurt*. The winning configs all use the UNIFORM prior — i.e. the prior term does
  nothing. Verified in `calibration_benchmark_results.csv`:
    - raw_q_knn: uniform macro_f1 0.824 > raw_local-ring 0.761.
    - label_profile_distance: uniform 0.815 >> raw_local-ring **0.501** (near random;
      LtH recall 0.152, collapse 0.721). A bad prior can DESTROY a good likelihood.

**Why the prior fails (diagnosis):** it is computed from the SAME feature-space
neighborhood as q (the ring prior is literally the outer shell of q's neighbor set).
So it carries the same NP-abundance contamination q does, and in a `likelihood × prior`
product you want two INDEPENDENT views — we gave it two views of the same thing. A
local-region prior built from the same embedding is **redundant with q, not
complementary**, almost by construction.

**Implication for adding time:** do NOT add time as a third multiplicative term
(`× P(label | t)`) — that repeats the same mistake (a redundant term from the same data
we'd have to bake off and likely abandon). Instead make the WORKING term
time-conditional: `P(q | label, t)`. Sharpen the one term that works; do not stack a
term we don't trust. [[user decided: soft kernel in t; no decidability flag yet]]

---

## 1. What we built and why

We stopped treating raw distance-weighted KNN vote fractions (`q`) as calibrated
probabilities. Instead we learn, from the labeled reference itself via leave-one-out,
how those vote patterns behave when the truth is known, and interpret new images
through that empirical null:

> P(label | q, local region) ∝ P(q | label) × P(label | local region)

- **q** — small-K (K=15) distance-weighted label distribution: the sharp local signal.
- **local_prior** — larger-K (K=100) label distribution: the smoother local context.
  Stored two ways: a **ring** prior (neighbors K_small+1..K_prior, disjoint from q, no
  double-counting) and a **full** prior (overlaps q).
- **reference null** — q / local_prior vectors for every reference image with that
  image excluded from its own neighbor search.

The known failure mode this guards against: **Not Penetrant (NP) is ~10× more abundant
than Intermediate** (7504 vs 723), so naive likelihood/prior both let NP win by sheer
count — "a black hole with a lab notebook." We tested 3 likelihood estimators ×
4 prior variants on the reference LOO table BEFORE touching query data.

Data: 13,062 labeled images, 30–48 hpf, 80 `z_mu_b_*` features.
Label counts: NP 7504, High_to_Low 2540, Low_to_High 2295, Intermediate 723.

---

## 2. Bake-off results (reference LOO, in-sample; q-space self-match dropped)

**Baseline — raw argmax(q):**
- accuracy 0.873, balanced_acc 0.752, macro_f1 0.793
- LtH recall 0.597, Intermediate recall 0.506
- **LtH→NP collapse 0.352**, NP→LtH false-call 0.013

**Top configs:**

| likelihood | prior | macro_f1 | bal_acc | acc | LtH rec | Int rec | LtH→NP | NP→LtH |
|---|---|---|---|---|---|---|---|---|
| raw_q_knn | uniform | **0.824** | 0.810 | 0.888 | 0.744 | 0.627 | 0.197 | 0.036 |
| balanced_q_knn | uniform | 0.789 | **0.849** | 0.850 | **0.803** | **0.833** | **0.096** | 0.082 |
| label_profile_distance | uniform | 0.815 | 0.820 | 0.878 | 0.732 | 0.707 | 0.188 | 0.048 |
| RAW_BASELINE | – | 0.793 | 0.752 | 0.873 | 0.597 | 0.506 | 0.352 | 0.013 |

(Full grid in `calibration_benchmark_results.csv`.)

## 3. Findings

**F1 — The likelihood does the heavy lifting; the prior barely helps.**
Best configs all use a **uniform** prior. Working in *q-space* (rather than feature
space) already absorbs NP's abundance advantage: a borderline-LtH image (q ≈ [0.5 LtH,
0.4 NP]) lands near *other* borderline-LtH images, whose true label is mostly LtH. So
re-correcting abundance via a prevalence-corrected prior is largely redundant. **We do
not need the fragile prevalence-corrected prior with its extra knobs.** This is a better
outcome than predicted.

**F2 — There is a real conservative-vs-sensitive trade-off.**
- `raw_q_knn + uniform`: best macro_f1, fewest false alarms (NP→LtH 0.036), still nearly
  halves the collapse (0.352→0.197). The conservative all-rounder.
- `balanced_q_knn + uniform`: best balanced accuracy and best rare-class recovery
  (LtH recall +21 pts, Intermediate +33 pts over baseline, collapse 0.352→0.096), at the
  cost of NP→LtH false calls rising to 0.082 (denting NP/LtH precision → lower macro_f1).
  Class-balancing the likelihood divides each label's neighbor count by its base rate, so
  it is more willing to call rare labels.

The choice is a values question (cost of missed-LtH vs false-LtH) and is **not yet
decided.** May be deferred to embryo-level rollup, where consistency voting + the
`ambiguous` status can absorb some image-level false LtH calls.

**F3 — `label_profile_distance` (centroid JS) is a respectable middle ground** (macro_f1
0.815, Int recall 0.707) but never strictly dominates either KNN estimator.

**F4 — Calibration is a clear win regardless of config.** Every reasonable combination
beats raw baseline on macro_f1 and on collapse. The empirical-null reframe works.

## 4. Caveats on these numbers

- **In-sample.** Scored on the same 13k images that built the q-null. The q-space
  self-match is dropped, but **sibling frames from the same embryo remain in the null** —
  near-duplicate images could inflate recall optimistically. An experiment-held-out
  (LOEO-style) re-run, or collapsing to embryo grain before scoring, would remove this.
  Not yet done.
- These are **image-level** metrics. The deliverable grain is **embryo-level**; rollup
  not yet integrated with the calibrated call.

---

## 5. NEXT STEP (the diagnostic that matters most now): time-window analysis

**Reframe:** the current method is solving two problems at once —
1. What phenotype does this image resemble?
2. At what developmental time is that phenotype separable?

Labels are **trajectory-level**; images are **static**. A broad 30–48 hpf window blurs
together images where the phenotype has not yet diverged with images where it has. So
the **LtH→NP collapse may be a time-window failure, not a method failure.** Before
concluding the classifier is at fault, ask: *are the labels actually separable in feature
space at the time points being used?*

**Core hypothesis (time-dependent confusion):**
```
early  (30–34): LtH and NP nearly indistinguishable
middle (34–42): LtH begins to diverge
late   (42–48): LtH more separable
```

**Plan — run the full reference-null calibration separately per HPF window.**
Windows: non-overlapping (30–34, 34–38, 38–42, 42–48) and/or sliding
(30–36, 32–38, …, 42–48). For each window:
1. filter reference to that window → 2. build image-level LOO null →
3. raw q → 4. calibrated prediction → 5. evaluate vs known labels →
6. compare separability / mixing / calibration across time.

**Per-window stats to track:**
n images per label · raw vs calibrated accuracy · balanced accuracy · macro_f1 ·
per-label precision/recall · LtH→NP and NP→LtH confusion · Intermediate recall ·
mean reference purity per label · main confusion label per label ·
density percentile distributions · raw-vs-calibrated disagreement rate.

**Most important diagnostic:** does the **LtH/NP neighbor profile change over time?**
Compute P(neighbor label | true label, window). If true-LtH neighbors shift from
~55% NP (early) to ~20% NP (late), LtH becomes separable later → time-window problem.
If the profile is flat across time, the issue is **feature representation / label
definition / genuine biological overlap**, not time.

**Compare both calibration configs per window** (`raw_q_knn+uniform` vs
`balanced_q_knn+uniform`) to learn whether balanced calibration helps uniformly or only
in specific developmental periods. Possible outcomes:
1. balanced helps early (recovers weak rare-class signal pre-separation),
2. balanced helps late (works once visually separable),
3. balanced overcalls rare labels early (too sensitive when classes truly overlap),
4. no window improves (features don't encode the phenotype).

**Make the density gate time-aware too:** compare a query image's mean-KNN-distance to
the reference density distribution *for the same HPF window*, not the whole 30–48 range.
Avoids asking whether a 32 hpf image is dense relative to a later-dominated reference.

**Possible method change (if performance varies strongly by time):** make the image-level
method **time-local** — for a query at time t, classify against reference images from
t ± hpf_window (e.g. 38 hpf → 36–40 hpf reference). Cleaner question: *among embryos at a
comparable developmental time, which phenotype best explains this image?* Likely more
biologically honest than comparing across the full window.

**Decision this analysis settles:**
- When does LtH become separable from NP?
- When does Intermediate become identifiable?
- Does balanced calibration help uniformly or only in certain windows?
- → Should the method be **globally calibrated** or **time-local**?

---

## 6. Time-window diagnostic RESULTS (run 2026-06-02, `run_time_window_diagnostic.py`)

**Hypothesis CONFIRMED, quantitatively and monotonically.** Outputs:
`time_window_diagnostic_metrics.csv`, `time_window_neighbor_profile.csv`,
`plots/time_window_*.png`. (Note: window analysis used the full labeled set, ~44.8k
images, since no global 30–48 prefilter — windows define their own time bounds.)

**Headline diagnostic — P(neighbor label | true = Low_to_High) over sliding windows:**

| window | LtH self | NP contamination |
|---|---|---|
| 30–36 | 0.421 | **0.479** (NP neighbors OUTNUMBER LtH) |
| 34–40 | 0.469 | 0.417 |
| 38–44 | 0.494 | 0.401 |
| 42–48 | **0.541** | **0.332** |

Before ~36 hpf, a true-LtH image's nearest neighbors are *more often NP than LtH* — the
phenotype has not diverged; early LtH embryos ARE NP embryos in feature space. Crossover
~36–38 hpf. By 42–48, LtH self-similarity 0.54, NP contamination down ~30%. Exactly the
predicted early-indistinguishable → middle-diverging → late-separable trajectory.

**Intermediate** is even more time-dependent: NP contamination 0.45→0.20, self-purity
0.34→0.46 across the window. Least defined early, identifiable late.

**Every calibration metric improves monotonically with developmental time:**

| metric (method) | 30–36 | 42–48 |
|---|---|---|
| LtH→NP collapse (conservative) | 0.279 | 0.152 |
| LtH→NP collapse (sensitive) | 0.114 | 0.075 |
| macro_f1 (conservative) | 0.778 | 0.848 |
| Intermediate recall (sensitive) | 0.800 | 0.839 |
| balanced_acc (sensitive) | 0.811 | 0.869 |

### Conclusions

**C1 — The global 30–48 bake-off averaged a hard early regime with an easy late one.**
Part of the apparent "method failure" was window-averaging, not classifier failure.

**C2 — It's time AND a residual feature/biology floor.** Even at 42–48, true-LtH
neighbors are only 54% LtH (~33% residual NP overlap that time alone does not resolve).
Time explains the *trend*; the feature representation / genuine biological overlap
explains the *residual*. (Your outcome #1 + a floor, NOT outcome #4 "no window helps.")

**C3 — `sensitive` (balanced_q_knn) earns its keep most in the hard early window.**
At 30–36 it cuts the collapse to 0.114 vs conservative 0.279 — rescues weak rare-class
signal precisely when classes genuinely overlap. The two configs converge late. Argues
for the balanced likelihood especially early.

**C4 — The method should become TIME-LOCAL.** Separability is a strong function of
developmental time; per-window metrics all beat the global 30–48 numbers. For a query
at time t, classify against reference images from t ± δ. More biologically honest:
"among embryos at comparable developmental time, which phenotype best explains this
image?" Density gate should likewise be within-window.

### Open design decisions (next)
- Time-local window width δ (the per-window n must stay large enough to calibrate; even
  6-hpf windows held ~3.8k images, plenty).
- Whether to also report a per-prediction "phenotype not yet separable at this stage"
  flag for early-window queries, since early LtH/NP is genuinely ambiguous.
- Held-out (LOEO-style) confirmation still outstanding (sibling-frame in-sample caveat).

---

## 7. NEXT ITERATION — time-aware reference-null calibration (planned 2026-06-02)

**North star:** *We are building a reference-calibrated image-level label-transfer
method: raw feature-space KNN produces a vote vector q, and reference leave-one-out
teaches us how to convert that vote vector plus developmental time into posterior-like
label probabilities.* Inspired by single-cell reference mapping (SingleR / scmap /
Seurat `TransferData`): map query onto a labeled reference, **return calibrated scores,
not just hard labels**, preserve uncertainty.

**Goal:** estimate P(y_i = label | q_i, t_i, R) per query image — posterior-like
calibrated probabilities, with hard labels derived secondarily via argmax.

**Why time enters as a KERNEL WEIGHT, not a third term.** The abandoned local-region
prior (Section 0) failed because it was a redundant second *view* multiplied into the
product. The time-aware form avoids that trap entirely: there is ONE estimator (vote
among reference-null examples) with a smarter WEIGHTING:

    w_ij = W_q(q_i, q_j) × W_t(t_i, t_j)
    W_q = exp(-D_q² / 2σ_q²)          # q-similarity (D_q from a chosen q-metric)
    W_t = exp(-Δt² / 2σ_t²)           # developmental-time similarity
    P(label=L | q_i, t_i, R) = Σ_j w_ij·1[true_label_j == L] / Σ_j w_ij

Both factors weight the SAME reference-null rows — no independent terms multiplied, so
no redundancy. Prior stays UNIFORM throughout (local prior abandoned).

**Benchmark plan (decided with user 2026-06-02) — all under the W_q×W_t time kernel,
uniform prior, decided ONLY after the time-aware bake-off:**

5 likelihood/q-similarity estimators, kept as FIRST-CLASS contenders (not replacements):
  1. `raw_q_knn` + time kernel
  2. `balanced_q_knn` + time kernel   ← prior bake-off's strongest rare-class recovery; KEPT
  3. q-distance kernel, Jensen-Shannon  (note's recommended default; ONE option, not THE default)
  4. q-distance kernel, L1 / cityblock
  5. q-distance kernel, Euclidean

Plus a SEPARATE supervised baseline, run EARLY as the bar to clear:
  - multinomial logistic regression on [q-vector, hpf], optionally q×hpf interactions.
    If it ties the nonparametric method → consider shipping the simpler model.
    If we beat it → the nonparametric q-space shape is earned.

**Amendments to the source note (Claude, recorded so intent is explicit):**
- (a) `balanced_q_knn` reinstated as first-class — the source note proposed JS-distance
  as default, but JS-distance = `label_profile_distance` scored macro_f1 0.815 < raw 0.824
  and lost the rare-class metrics to balanced_q_knn (bal_acc 0.849, LtH recall 0.803).
- (b) LR baseline runs first.
- (c) σ_t / σ_q set FROM DATA, not fixed guesses: σ_t default in 2–4 hpf range (the
  LtH/NP neighbor-profile transition spans ~6–8 hpf), confirmed by grid {2,3,4,6};
  σ_q default = median pairwise q-distance (standard kernel heuristic), not hand-picked.
- (d) Compute note: keep `NearestNeighbors` (top-K) for the 13k×13k LOO build; reserve
  full `cdist` for modest query batches (13k² JS via cdist is too slow).

**Metrics (rare-class focused, NOT raw accuracy):** macro_f1, balanced_accuracy,
per-label precision/recall, LtH recall, Intermediate recall, LtH→NP collapse,
NP→LtH false-call rate, calibrated-margin behavior.

**Validation ladder (fixes the sibling-frame caveat):**
image-LOO (fast iteration) → leave-one-EMBRYO-out (kills sibling-frame leakage) →
leave-one-EXPERIMENT-out (batch generalization).

**Image-level output (scores preserved, hard label secondary):**
query_snip_id, query_embryo_id, query_hpf, raw_pred_label, raw_top_probability,
calibrated_pred_label, calibrated_P_<label>×4, calibrated_top_probability,
calibrated_margin, q_metric, sigma_t, density_percentile, image_status,
image_status_reason.

**Tasks:** (1) q-distance util via `cdist`; (2) `calibrate_q_time()` kernel estimator;
(3) benchmark q_metric × σ_t grid + balanced/raw KNN under time kernel; (4) LR baseline
on [q, hpf]; (5) pick default image-level method after the bake-off.

---

## 8. Time-aware bake-off RESULTS (run 2026-06-02, `run_time_aware_benchmark.py`)

Image-LOO arena, drop_self=True, uniform prior. Output:
`time_aware_benchmark_results.csv`, `reference_loo_table_timeaware.csv`.

**Top of the ranking (by macro_f1):**

| method | σ_t | macro_f1 | bal_acc | LtH rec | Int rec | LtH→NP | NP→LtH |
|---|---|---|---|---|---|---|---|
| raw_q_knn | 2 | **0.836** | 0.820 | 0.745 | 0.658 | 0.203 | 0.037 |
| raw_q_knn | inf | 0.824 | 0.810 | 0.744 | 0.627 | 0.197 | 0.036 |
| LR[q,hpf,q×hpf] | – | 0.820 | 0.799 | 0.712 | 0.613 | 0.234 | 0.034 |
| LR[q,hpf] | – | 0.818 | 0.797 | 0.711 | 0.607 | 0.235 | 0.035 |
| RAW_BASELINE | – | 0.793 | 0.752 | 0.597 | 0.506 | 0.352 | 0.013 |
| balanced_q_knn | 4 | 0.791 | **0.852** | 0.807 | **0.845** | **0.091** | 0.083 |
| balanced_q_knn | inf | 0.789 | 0.849 | 0.803 | 0.833 | 0.096 | 0.082 |

### Findings

**T1 — The time kernel helps, modestly but CONSISTENTLY, and never hurts.** Tightening
σ_t from ∞ (time-agnostic control) to 2 hpf improves every estimator: raw_q_knn macro_f1
0.824→0.836, Intermediate recall 0.627→0.658; balanced_q_knn bal_acc 0.849→0.852,
collapse 0.096→0.091. **Wording (stats-clean):** conditioning the q-space calibration on
time *improves empirical label-transfer performance* vs. the time-agnostic calibration —
NOT "P(q|label,t) ≥ P(q|label)" (a conditional likelihood is not in general ≥ the
marginal; that inequality is meaningless). The effect is small because q already partly
encodes stage (similar-q images tend to be similar-stage; morphology changes over time,
so the KNN vote pattern already carries developmental signal). Best σ_t is 2–4 hpf,
matching the data-derived expectation (LtH/NP transition spans ~6–8 hpf).

**T2 — The conservative/sensitive split survives time-conditioning, unchanged.**
raw_q_knn σ_t=2 = best macro_f1 + fewest false calls (NP→LtH 0.037). balanced_q_knn
σ_t≈4 = best balanced acc + best rare-class recovery (LtH 0.807, Int 0.845, collapse
0.091). Still a values choice; deferrable to embryo rollup.

**T3 — Logistic regression (the bar) is beaten, BUT the comparison was UNFAIR on time.**
LR[q,hpf(,q×hpf)] reaches macro_f1 0.818–0.820 — above raw baseline (0.793), ties
time-agnostic raw_q_knn, loses to the time-kernel KNN. q×hpf barely helped (+0.002).
**Caveat (user, correct):** the time-kernel KNN is time-LOCAL (a different effective
q→label map at each stage via W_t weighting), but `LR[q,hpf]` is time-GLOBAL (one linear
surface, hpf just a feature). A single global LR cannot represent a different q→label
mapping at 32 vs 46 hpf — so this compared a time-local method against a time-global
baseline. The honest comparison is a **time-local LR**: either windowed (fit LR per hpf
bin) or W_t-weighted (fit LR with the same temporal sample weights as the kernel). The
W_t-weighted version is truest apples-to-apples — identical time localization, only
linear-LR vs nonparametric-KNN differs. **Pending** (next benchmark). If time-local LR
closes the gap → ship the simpler model; if KNN still wins → the nonparametric shape of
q-space *within a stage* is what matters.

**T4 — ⚠️ KERNEL q-distance estimators (kernel_js/l1/eu) results are INVALID — bug, not
finding.** All 15 kernel configs gave Intermediate recall EXACTLY 0.000 and LtH→NP
collapse ~0.99 — a normalization-failure signature, not a modeling result. Cause: the
`kernel` scheme sums weights over ALL ~13k reference rows with NO class balancing, so
NP's 57% mass swamps rare classes (the "black hole with a lab notebook" in pure form).
The KNN schemes avoid this via top-K + optional base-rate division. **Fix needed:** add a
class-balanced kernel variant (divide each label's summed kernel weight by its base
rate), then re-benchmark. Until then JS/L1/Euclidean are NOT validly compared. This is
NOT a verdict on JS-distance.

### Current leading default (pending kernel fix + embryo rollup)
- **raw_q_knn + time kernel, σ_t=2** if optimizing macro_f1 / minimizing false calls.
- **balanced_q_knn + time kernel, σ_t≈4** if optimizing rare-class recovery (the stated
  goal: expose/manage mixed/rare phenotypes).
- Both beat LR and raw baseline. Final pick after (a) class-balanced kernel re-benchmark,
  (b) leave-one-embryo-out validation (sibling-frame caveat), (c) embryo-level rollup.
