# Spec: Classifier-Assisted Morphological Geometry Extraction

## Purpose

This document specifies a new analysis module for extracting **geometrically interpretable, individual-level morphology values** from pairwise condition comparisons in a shared morphology feature space.

The key shift is that we are moving away from treating pairwise classifiers as tools that produce only **pairwise summary values** such as AUROC, balanced accuracy, or confusion-derived quantities, and toward treating them as tools that define **morphological directions in feature space**.

These directions give us richer outputs:

- an interpretable **axis of phenotypic separation** for each comparison
- an **individual-level signed coordinate** for each embryo along that axis
- a notion of **directionality**
- a notion of **distance from the separating boundary**
- a decomposition of each embryo’s morphology into:
  - the component aligned with the comparison axis
  - the residual morphology orthogonal to that axis

This module is intended to be implemented as a **separate subsystem** from ordinary classification evaluation.

---

# 1. Motivation and conceptual shift

## 1.1 Previous framing: pairwise scalar values

The prior framing treated pairwise comparison as a source of scalar outputs such as:

- AUROC
- balanced accuracy
- misclassification rate
- calibrated probability
- logit-like score

These values are useful for answering:

- Can two conditions be distinguished?
- How separable are they at the population level?
- Which pairs are easy or hard to classify?

However, these values do **not** fully exploit the fact that all embryos live in the same morphology feature space.

In particular, pairwise summary statistics do **not** directly tell us:

- what morphological direction distinguishes the conditions
- where an individual embryo lies along that direction
- whether an embryo is shifted toward one condition or the other
- how much of the embryo’s morphology is explained by that comparison axis

---

## 1.2 New framing: pairwise classifiers as geometry extractors

If all comparisons are made in a common morphology space, and if the classifier is linear in that space, then each fitted pairwise model defines a **direction vector** in morphology space.

That vector can be interpreted as a **phenotypic separation axis**.

This changes the role of the classifier:

- before: classifier as a source of prediction and summary discrimination metrics
- now: classifier as a source of a **geometry-defining direction**

The key idea is:

> A pairwise comparison does not just tell us whether two conditions are separable.  
> It tells us the morphological direction along which they are separated.

From that direction, we can compute individual-level coordinates and residual structure.

---

## 1.3 Why this is better

This geometry-based approach is more descriptive because it preserves more of the structure present in the shared morphology space.

Instead of collapsing a comparison to a single scalar summary, it yields:

- a vector-valued axis
- a signed coordinate for each embryo
- an interpretable center
- an aligned vs orthogonal decomposition

This is especially useful when the goal is not only to score separation, but to understand:

- the morphology of partial rescue
- intermediate phenotypes
- combinatorial perturbation behavior
- whether embryos vary primarily along a biologically meaningful comparison axis
- how pairwise comparison geometry changes over developmental time

---

# 2. Scope

This spec covers a first-pass module for:

- pairwise binary comparisons
- linear classifiers in a shared morphology feature space
- k-fold cross-validation
- extraction of fold-level and consensus geometry
- per-embryo out-of-fold geometry values

This spec does **not** yet cover:

- nonlinear classifiers
- multiclass geometry extraction
- temporal smoothing across time bins
- dynamic graph construction from geometry outputs
- causal interpretation of axes

---

# 3. Mathematical formulation

## 3.1 Data

Let each embryo be represented by a feature vector:

\[
x \in \mathbb{R}^p
\]

where \(p\) is the number of morphology features.

For a pairwise comparison between conditions \(+\) and \(-\), we fit a linear classifier in this feature space.

---

## 3.2 Linear classifier

For a fitted linear model, we obtain:

\[
w \in \mathbb{R}^p, \qquad b \in \mathbb{R}
\]

where:

- \(w\) is the coefficient vector
- \(b\) is the intercept

The classifier decision function is:

\[
f(x) = w^\top x + b
\]

This defines a separating hyperplane:

\[
w^\top x + b = 0
\]

The vector \(w\) is normal to that hyperplane and therefore defines the primary direction of separation.

---

## 3.3 Unit direction vector

We define the normalized comparison axis:

\[
u = \frac{w}{\|w\|}
\]

This is the unit vector pointing along the learned phenotypic separation direction.

Interpretation:

- positive direction = movement toward the positive class
- negative direction = movement toward the negative class

The sign convention must be kept consistent with label ordering.

---

## 3.4 Choice of center

To get an interpretable coordinate, we must choose a center \(c\) in feature space.

This module will support multiple centering strategies.

### Midpoint centering
Let \(\mu_+\) and \(\mu_-\) be the class centroids in feature space.

Then define:

\[
c = \frac{\mu_+ + \mu_-}{2}
\]

This gives an axis centered at the midpoint between the two conditions.

### Other possible centers
Other supported centers may include:

- origin
- positive class centroid
- negative class centroid
- boundary anchor
- custom center

The center affects the interpretation of the resulting coordinate and must be recorded explicitly.

---

## 3.5 Signed projection coordinate

For an embryo \(x\), define its signed coordinate along the comparison axis as:

\[
\alpha(x) = u^\top (x - c)
\]

Interpretation:

- \(\alpha(x) > 0\): embryo lies in the positive direction
- \(\alpha(x) < 0\): embryo lies in the negative direction
- \(|\alpha(x)|\): magnitude of displacement along the comparison axis

This is the main **individual-level morphology geometry value**.

---

## 3.6 Signed distance to decision boundary

A separate but related quantity is the signed distance to the classifier boundary:

\[
d(x) = \frac{w^\top x + b}{\|w\|}
\]

Interpretation:

- positive = embryo is on the positive side of the boundary
- negative = embryo is on the negative side
- magnitude = perpendicular distance to the decision hyperplane

This is classifier-centered, whereas \(\alpha(x)\) is center-choice-centered.

Both are useful and should be retained.

---

## 3.7 Along-axis and residual decomposition

Let:

\[
v(x) = x - c
\]

be the centered embryo vector.

Then the component aligned with the comparison axis is:

\[
a(x) = \big(u^\top (x-c)\big)u = \alpha(x)u
\]

The orthogonal residual is:

\[
r(x) = (x-c) - a(x)
\]

So:

\[
x - c = a(x) + r(x)
\]

Interpretation:

- \(a(x)\): morphology explained by the comparison axis
- \(r(x)\): morphology not explained by that axis

This decomposition is important because not all variation in an embryo should be expected to lie on the pairwise axis.

---

# 4. Conceptual outputs

For each pairwise comparison, the geometry module should return:

## 4.1 Comparison-level outputs

- raw classifier coefficient vector \(w\)
- normalized direction vector \(u\)
- chosen center \(c\)
- fold-wise directions
- consensus direction across folds
- centroid information
- metadata about space and centering

## 4.2 Embryo-level outputs

For each embryo:

- signed projection coordinate \(\alpha(x)\)
- signed boundary distance \(d(x)\)
- decision function value \(f(x)\)
- optional predicted probability
- along-axis component norm
- residual norm
- fold assignment
- true label

---

# 5. Why this should be a separate subsystem

This should **not** be embedded directly into the normal classification loop as an incidental side effect.

The ordinary classification pipeline is primarily designed to answer:

- how accurate is the model?
- how separable are conditions?
- what are the summary performance metrics?

The geometry extraction pipeline is designed to answer:

- what is the pairwise morphological direction?
- where does each embryo lie on that axis?
- how should the axis be centered?
- how stable is the direction across folds?
- how much of each embryo lies along the comparison geometry?

These are different contracts and produce different primary outputs.

Therefore this should be implemented as a distinct module with its own result objects and API.

---

# 6. Design principles

## 6.1 Explicit feature space contract

All geometry results must explicitly declare the feature space in which they were computed.

This includes:

- feature columns used
- whether features were standardized
- whether coefficients are expressed in transformed space or raw space
- whether centroids and centers were computed in transformed space or raw space

This must be explicit to avoid later ambiguity.

---

## 6.2 Direction and centering are modular

Direction extraction and centering must be separate modules.

This is necessary because we want to compare alternatives such as:

- direction from classifier coefficients
- direction from mean difference
- midpoint centering
- boundary-centered anchoring

These should be swappable without changing the rest of the pipeline.

---

## 6.3 Use held-out embryos for embryo-level geometry values

Per-embryo geometry values used downstream should preferably be computed from **out-of-fold** predictions / projections.

This ensures that embryo-level coordinates are not trivially overfit to the same data used to train the comparison axis.

---

## 6.4 Fold sign alignment is required

A direction vector \(u\) and \(-u\) represent the same axis with opposite orientation.

Therefore, before aggregating fold-wise directions, signs must be aligned consistently.

Otherwise fold averaging may collapse directions toward zero.

---

# 7. Proposed module decomposition

## 7.1 Top-level orchestration

### `run_morphology_geometry(...)`

Main entry point for a single pairwise comparison.

Responsibilities:

- subset data for the requested pair
- run cross-validation
- fit the model in each fold
- extract fold-level geometry
- compute out-of-fold embryo geometry values
- aggregate fold directions
- return a structured result object

---

## 7.2 Comparison spec

### `PairwiseGeometrySpec`

Defines the comparison and geometry options.

Suggested fields:

- `positive_label`
- `negative_label`
- `feature_cols`
- `group_col`
- `time_col`
- `direction_mode`
- `center_mode`
- `standardize`
- `estimator_config`

This keeps the configuration declarative and reviewable.

---

## 7.3 Fold fitting

### `fit_pairwise_linear_model(...)`

Responsibilities:

- receive train/test split
- fit a linear estimator
- return coefficients, intercept, raw scores, and transformed test data

This function should remain small and standard.

---

## 7.4 Direction extraction

### `compute_geometry_direction(...)`

Responsibilities:

- compute the comparison direction vector
- support multiple strategies

Supported modes in v1:

- `"coef"`: use fitted classifier coefficients
- `"mean_diff"`: use difference in class centroids

Returns:

- raw direction vector
- normalized direction vector

---

## 7.5 Center computation

### `compute_geometry_center(...)`

Responsibilities:

- compute the centering reference point

Supported modes in v1:

- `"midpoint"`
- `"origin"`
- `"positive_centroid"`
- `"negative_centroid"`

Returns:

- center vector \(c\)

---

## 7.6 Geometry extraction

### `extract_geometry_from_linear_model(...)`

Responsibilities:

Given \(X\), \(y\), \(w\), \(b\), and a center \(c\), compute:

- unit direction vector
- signed projection coordinate
- signed boundary distance
- along-axis vector
- residual vector
- norms of aligned and residual components

This is the mathematical heart of the module.

---

## 7.7 Fold aggregation

### `aggregate_geometry_across_folds(...)`

Responsibilities:

- align signs of fold directions
- compute consensus direction
- summarize direction stability
- aggregate metadata

---

## 7.8 Collection-level orchestration

### `run_all_pairwise_morphology_geometry(...)`

Responsibilities:

- iterate over multiple pairwise specs
- collect `PairwiseGeometryResult` objects
- return a collection object for downstream analysis

---

# 8. Proposed result objects

## 8.1 `FoldGeometryResult`

Stores geometry for one fitted fold.

Suggested fields:

- `fold`
- `coef`
- `intercept`
- `direction`
- `unit_direction`
- `center`
- `sample_table`
- `metadata`

---

## 8.2 `PairwiseGeometryResult`

Stores geometry for one pairwise comparison across folds.

Suggested fields:

- `comparison_id`
- `positive_label`
- `negative_label`
- `feature_cols`
- `fold_results`
- `consensus_direction`
- `consensus_center`
- `sample_table`
- `direction_summary`
- `metadata`

---

## 8.3 `MorphologyGeometryCollection`

Stores multiple pairwise comparison results.

Suggested methods:

- `get(comparison_id)`
- `sample_table()`
- `direction_table()`
- `comparison_ids()`

---

# 9. Suggested v1 API sketch

```python
results = run_morphology_geometry(
    X=df[feature_cols],
    y=df[label_col],
    groups=df[embryo_id_col],
    spec=PairwiseGeometrySpec(
        positive_label="cond_a",
        negative_label="cond_b",
        feature_cols=feature_cols,
        direction_mode="coef",
        center_mode="midpoint",
        standardize=True,
    ),
    estimator=LogisticRegression(...),
    cv=cv,
)