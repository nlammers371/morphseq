# Pairwise Classification Probe : Rationale, Math, and Current Design

## Why this exists

We wanted a representation for MorphSeq trajectories that keeps the sensitivity of pairwise class comparisons without forcing ourselves into a full all-pairs-by-all-pairs object as the final geometry.

The original tension was:

- **Multiclass space** is clean and shared across all embryos, but may wash out subtle pair-specific contrasts.
- **Pairwise classifiers** are sensitive to specific phenotypic differences, but their raw outputs are awkward to use as a common representation because many evaluations are out-of-support and can reflect generic geometry instead of biologically meaningful contrast.

The goal became:

> Build a **shared, label-agnostic fingerprint space** from pairwise probes, while explicitly controlling for the main weakness of out-of-support pairwise evaluations.

---

## How we got here

### 1. Why not all-pairs as the final object?

For `G` classes, the naive all-pairs representation gives `G choose 2` unordered comparisons, or `G(G-1)` ordered comparisons.

That is rich, but it has two problems:

1. **Redundancy**
   - many pairwise axes are correlated
   - ordered pairs contain structural antisymmetry
2. **Out-of-support comparisons**
   - a classifier trained only on classes `f` and `g` may be evaluated on an embryo from class `c`
   - the resulting output is numerically defined, but not automatically biologically interpretable

This made the naive all-pairs representation feel mathematically dirty as a final geometry.

### 2. Why not just switch to multiclass?

Multiclass is the cleanest shared representation if the only goal is a coherent class-evidence geometry.

But pairwise probes may still carry useful information because they measure specific contrast axes that a joint softmax can average over.

So instead of choosing one or the other immediately, we reframed pairwise classifiers as:

> **phenotypic probes**, not just label assigners.

### 3. The key weakness: out-of-support comparisons

The biggest conceptual problem was this:

- a real `f`-vs-`g` probe may be valid on its own training support
- but a class `c` embryo may project onto that probe for reasons that have nothing to do with the biological `f`-vs-`g` distinction
- stable projection over time does **not** rule this out, because systematic geometric confounds can also be stable

This led to the need for explicit controls.

---

## The three-control framework we landed on

### 1. Temporal consistency

A probe response that is stable over time is less likely to be random noise.

What it tells us:
- rejects pure noise

What it does **not** tell us:
- whether the axis is biologically specific rather than a stable geometric confound

### 2. Label-permutation null

This became the main control.

For each pairwise probe `(f, g)`, retrain null probes on label-permuted `f/g` data.
Then compare the real off-support signal of class `c` on the real probe versus the null probes.

This answers:

> Is class `c`'s placement on the `(f, g)` axis more structured than what generic geometry would produce under a scrambled version of the same probe?

This is a **validation** step, not a correction step.

### 3. Antisymmetry / consistency diagnostic

For ordered probes, the ideal structure is antisymmetric.
In probability form, `h_fg + h_gf ≈ 1`.
In margin/logit form, `m_fg ≈ -m_gf`.

This gives a system-level QC diagnostic:
- strong antisymmetry supports coherent probe behavior
- symmetric residue flags calibration mismatch or generic geometry

---

## The important conceptual refinement

We initially thought validity might have to be **probe × class** dependent.
That is still true scientifically, but from a representation-design perspective we decided:

> **Do not use the true class label of the embryo at inference time to choose the shrinkage weight.**

Why?
Because then two embryos with identical feature vectors but different labels would get different fingerprints.
That would reintroduce label dependence into a representation we want to keep as class-comparison agnostic as possible.

So instead, the weight adjustment should be:

- the **same for all embryos** for a given probe
- derived from probe-level or globally aggregated validity diagnostics
- not looked up using `c(x)` at inference time

This keeps the fingerprint label-agnostic.

---

## The current mathematical object

Let there be `G` classes.
Train one binary classifier for each unordered pair `(i, j)`, with `i < j`.
There are

```text
M = G choose 2
```

such probes.

For each probe, define a signed score for embryo/timepoint `x`:

```math
m_{ij}(x) = 2 h_{ij}(x) - 1 \in [-1, 1]
```

Interpretation:
- `+1` = strongly `i`-like vs `j`
- `0` = neutral
- `-1` = strongly `j`-like vs `i`

For each embryo/timepoint, the **raw pairwise fingerprint** is:

```math
F_{raw}(x) = (m_{12}(x), m_{13}(x), \dots, m_{(G-1)G}(x)) \in \mathbb{R}^{M}
```

So for `G = 5`, this is a 10-dimensional fingerprint.

---

## How we deal with the out-of-support weakness

The key move is **null-validated shrinkage toward neutral**.

For each probe `(i, j)`:

1. Train the real probe on the true `i/j` labels.
2. Train many permuted-label null probes for the same pair.
3. Compare the real probe signal to the null distribution.
4. Convert this into a **probe-level specificity weight**

```math
w_{ij} \in [0,1]
```

where:
- `w_ij ≈ 1` means the probe carries contrast-specific signal
- `w_ij ≈ 0` means the probe behaves like generic geometry / nonspecific structure

Then shrink the probe coordinate toward the neutral value:

```math
m^*_{ij}(x) = w_{ij} \, m_{ij}(x)
```

Because we work in signed-margin space, the neutral value is already `0`.

So the **final validity-adjusted fingerprint** is:

```math
F(x) = (m^*_{12}(x), m^*_{13}(x), \dots, m^*_{(G-1)G}(x))
```

This means:
- reliable pairwise axes stay active
- unreliable pairwise axes are collapsed toward neutral
- the fingerprint remains the same shape for every embryo
- no true class label is needed at inference time for the shrinkage step

---

## Why this is better than hard masking

We considered masking invalid coordinates entirely, but shrinkage is better because:

- it preserves a fixed-dimensional representation
- it avoids missing-value logic in downstream clustering / embedding
- it allows continuous confidence rather than binary keep/drop
- it cleanly expresses “uninformative” as “neutral”

In probability space this would mean shrinking toward `0.5`.
In signed-margin space it means shrinking toward `0`.

Signed-margin space is cleaner for this reason.

---

## What biological information this gives us

This is the main payoff.

The method does **not** claim that every coordinate is biologically meaningful.
Instead, it gives two coupled objects:

1. **A shared pairwise phenotypic fingerprint**
2. **A validity profile for those probe coordinates**

That means we can ask:

- Which pairwise contrasts are biologically resolved in this feature space?
- At what developmental times do specific genotype pairs become distinguishable?
- Which contrasts are robust across batches, and which are batch-sensitive?
- Which probes are carrying genuine contrast-specific information versus generic geometry?

So the weakness of out-of-support pairwise comparisons becomes a source of information rather than only a nuisance:

> the reliability pattern of the fingerprint tells us where the phenotypic space has resolution and where it does not.

---

## How this fits the cosmological trajectory framework

Once each embryo/timepoint has a validity-adjusted fingerprint `F(x)`, those fingerprints can be used as the input feature vectors for the existing trajectory pipeline.

For each embryo over time:

```math
F_e(t_1), F_e(t_2), \dots, F_e(t_n)
```

These become the time-varying state vectors that are then:

1. embedded
2. linked via temporal coherence
3. refined / condensed by the cosmological trajectory machinery

So the pairwise probe fingerprint is upstream of the trajectory geometry.
It replaces raw morphology or simple classification space with a richer, validity-aware phenotypic state representation.

---

## What this gives us conceptually

This is more than just a representation choice.

It gives MorphSeq a framework that:

- uses pairwise probe sensitivity without blindly trusting all out-of-support readouts
- remains shared and label-agnostic across embryos
- diagnoses its own weak coordinates
- turns null-failing axes into neutralized coordinates rather than arbitrary noise
- provides a biologically interpretable map of where phenotypic distinctions emerge over developmental time

A good one-line summary is:

> We represent each embryo by its response profile across all pairwise phenotypic probes, then use permutation-validated shrinkage to neutralize nonspecific probe axes. The result is a shared fingerprint space that preserves contrast-sensitive morphology while explicitly measuring where the phenotypic representation is and is not reliable.

---

## Current design decisions to lock down

### Keep
- one probe per unordered class pair
- signed-margin representation
- permutation null as the main validity control
- shrinkage toward neutral
- use the same shrinkage rule for all embryos for a given probe
- store validity information as a first-class artifact alongside the fingerprints

### Avoid
- using the embryo's true class label at inference time to choose probe weights
- interpreting temporal consistency as proof of semantic specificity
- treating null failure as something to “correct” rather than something to validate and neutralize

---

## Open questions

1. Should probe specificity weights be:
   - hard-thresholded
   - or continuous from effect size / p-value?

2. Should antisymmetry be used only as a QC diagnostic,
   - or also to derive a principled low-rank reduction of the pairwise probe family?

3. Should validity be stored:
   - probe-level only
   - batch-specific
   - timepoint-specific
   - or all three?

4. Once the fingerprint is built, what is the best geometry downstream:
   - direct trajectory condensation
   - principal graph / tree fitting
   - or both?

---

## Bottom line

We wanted a way to keep the pair-specific sensitivity of binary classifiers without accepting the full weakness of raw out-of-support pairwise evaluations.

The solution we arrived at is:

1. train pairwise probes,
2. evaluate every embryo on all probes,
3. use permutation nulls to estimate which probe axes are genuinely contrast-specific,
4. shrink null-failing axes toward neutral,
5. feed the resulting shared fingerprint trajectories into the cosmological trajectory framework.

This gives us a representation that is richer than direct multiclass alone, but still disciplined enough to defend mathematically and biologically.


=======================================================================
               MORPHSEQ CONTRAST PIPELINE ARCHITECTURE
=======================================================================

HIGH-LEVEL CONCEPTUAL SUMMARY
Raw pairwise classifiers are highly sensitive to subtle phenotypes, but 
they generate artifacts when evaluated on out-of-support data. 

This architecture solves that by reframing classifiers as active 
"Classification Contrast Probes." Instead of trusting every measurement, 
it applies a permutation-based specificity weight (w_ij) to validate 
the probe's reliability. Uninformative or nonspecific probe readouts 
are shrunk toward 0 (neutral). 

The result is a shared, continuous, and validity-aware coordinate 
space that seamlessly feeds into downstream trajectory condensation.

=======================================================================
                           DATA FLOW
=======================================================================

                   [ Raw Embryo Morphology Data ]
                                 |
                                 v
+=====================================================================+
|                    CLASSIFICATION CONTRAST MAP                      |
|                  (The Orchestrator Engine / Layer 2)                |
|                                                                     |
|  Receives raw embryo data and maps it to the new contrast space.    |
|                                                                     |
|      +-------------------------------------------------------+      |
|      | LOOP: FOR EACH UNORDERED PAIR (i, j)                  |      |
|      |                                                       |      |
|      |   +-----------------------------------------------+   |      |
|      |   |         ClassificationContrastProbe           |   |      |
|      |   |             (The Atom / Layer 1)              |   |      |
|      |   |                                               |   |      |
|      |   |  • Contrast Tuple: ("pbx1b", "wik_ab")        |   |      |
|      |   |  • Trained ML Classifier                      |   |      |
|      |   |  • Permutation Null Distribution              |   |      |
|      |   |  • Probe Specificity Weight (w_ij)            |   |      |
|      |   |                                               |   |      |
|      |   |  Action:                                      |   |      |
|      |   |  1. Calculate raw signed margin m_ij(x)       |   |      |
|      |   |  2. Apply shrinkage: m*_ij(x) = w_ij * m_ij   |   |      |
|      |   +-----------------------------------------------+   |      |
|      +-------------------------------------------------------+      |
|                                                                     |
|  Assembles all individual probe measurements into one vector.       |
+=====================================================================+
                                 |
                                 | (Outputs)
                                 v
+=====================================================================+
|                    CLASSIFICATION COORDINATES                       |
|                    (The Output Vector / Layer 3)                    |
|                                                                     |
|  • F_raw:         [ 0.82, -0.15,  0.94, ... ]                       |
|  • F_shrunk:      [ 0.82,  0.00,  0.94, ... ]                       |
|  • Residual D(x): [ 0.00, -0.15,  0.00, ... ] (F_raw - F_shrunk)    |
|  • Probe Index:   Index 0 -> ("pbx1b", "wik_ab")                    |
+=====================================================================+
                                 |
                                 | (Passed downstream)
                                 v
            [ Cosmological Trajectory Condensation Pipeline ]

=======================================================================
               FUTURE EXTENSIONS / OPEN QUESTIONS
=======================================================================
* Diagnostic Probe/Class Validity (w_ij,c): While shrinkage is strictly 
  label-agnostic at inference (global w_ij), class-specific validity 
  weights can be calculated during the null-permutation phase and stored 
  as a diagnostic matrix. This allows post-hoc analysis of which probes 
  generate systematic artifacts for specific off-support classes.

* Per-Embryo Support Weighting: For future applications involving 
  unknown perturbation screening, consider extending the shrinkage 
  function to m*_ij(x) = w_ij * c_ij(x) * m_ij(x), where c_ij(x) is 
  a local density estimate of the embryo's distance to the support 
  manifold of classes i and j.