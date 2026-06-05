"""
conformal_sets.py
=================================================================
Split-conformal prediction sets on top of a KNN label vote, with a
density gate for out-of-support points.

This file is PURE CONFORMAL MACHINERY. It receives arrays and returns
prediction sets. It knows nothing about where the data came from:
time-windowing, NaN handling, train/cal/query splitting, and any other
upstream/downstream concern is the CALLER's job. Keep it that way --
it is what makes every function here independently testable.

For everything we deliberately left OUT (TACP for class imbalance,
Mondrian / clustered conformal, the ERT coverage diagnostic, the
density-ratio gate, cross-conformal / LOO calibration, the abstain
rule, and the testing strategy), see CONFORMAL_UPGRADES.md. Hooks in
the code below are marked  # UPGRADE: ...  where an extension slots in.

-----------------------------------------------------------------
NOTATION  (every symbol used below; the math is consistent throughout)
-----------------------------------------------------------------
  K            number of classes
  d            feature dimension
  x            a single point's feature vector, shape (d,)
  X_*          a stack of feature vectors, shape (n_*, d)
  y_*          integer labels in {0, ..., K-1}, shape (n_*,)

  THREE DATA ROLES this file understands (all supplied by the caller):
    reference    the neighbor POOL the KNN searches against
    calibration  labeled points whose true-label scores set the threshold
    query        unlabeled points we want prediction sets for
  reference and calibration MUST be disjoint, otherwise a calibration
  point can be its own neighbor and the threshold is biased optimistic.
  This file enforces that structurally: they are separate arguments.

  q(x)         length-K KNN probability vector, sums to 1          [Stage 2]
                 q_y(x) = (sum of 1/dist over neighbors of class y)
                          / (sum of 1/dist over all k neighbors)
  s_y(x)       APS nonconformity score for label y                 [Stage 4]
                 sort labels by q descending; s_y = cumulative q
                 from the top DOWN TO AND INCLUDING y.
                 (top label -> small score; buried label -> ~1)
  alpha        target miss rate; marginal coverage target = 1 - alpha
  qhat         calibrated APS threshold                            [Stage 5]
                 the ceil((n_cal + 1)(1 - alpha)) / n_cal empirical
                 quantile of true-label APS scores on calibration data.
  C(x)         prediction set = { y : s_y(x) <= qhat }             [Stage 6]
                 with include_last_label=True (never empty).

  STATUS taxonomy (assigned by the orchestrator, not the helpers):
    "not_evaluated"  bad/missing features (caller may also pre-filter)
    "low_density"    query is outside reference support (density gate)
    "assigned"       prediction set has exactly one label
    "ambiguous"      prediction set has two or more labels
-----------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


# =================================================================
# HELPERS -- each is ONE mathematical object from the notation block.
# Pure: no state, no I/O, individually testable.
# =================================================================

def knn_probabilities(
    X_reference: np.ndarray,
    y_reference: np.ndarray,
    X: np.ndarray,
    k: int,
    n_classes: int,
    smoothing: float = 1e-3,
) -> np.ndarray:
    """Stage 2 -- the local KNN vote q(x).

    Distance-weighted KNN probability for every row of X, computed
    against the reference pool. Returns shape (len(X), n_classes),
    each row summing to 1. The neighbor search is the only nontrivial
    work and is delegated to sklearn.

    `smoothing` adds a small mass to every class before normalizing
    (Laplace-style). This matters for APS: a hard zero on a class means
    that class can only ever be reached at cumulative mass 1.0, so a
    true label with zero vote gets APS score exactly 1.0. With many
    classes and a finite k, zeros are common and would pin qhat at 1.0
    (degenerate full sets). A tiny floor removes the pathology without
    materially changing the vote. Set smoothing=0 to disable.

    NOTE on disjointness: X must not contain reference points (a point
    would be its own nearest neighbor). The caller guarantees this by
    passing calibration/query rows here, never reference rows.
    """
    clf = KNeighborsClassifier(n_neighbors=k, weights="distance")
    clf.fit(X_reference, y_reference)
    proba = clf.predict_proba(X)  # (len(X), n_classes_seen)

    # sklearn only emits columns for classes present in y_reference.
    # Re-expand to the full K-column layout so column index == class id.
    q = np.zeros((X.shape[0], n_classes), dtype=float)
    q[:, clf.classes_] = proba

    if smoothing > 0:
        q = q + smoothing
        q = q / q.sum(axis=1, keepdims=True)
    return q


def density_flags(
    X_reference: np.ndarray,
    X: np.ndarray,
    k: int,
    percentile: float,
) -> np.ndarray:
    """Stage 3 -- support gate (distance-based, heuristic).

    A query is 'low density' if its mean distance to its k nearest
    reference points exceeds the `percentile`-th percentile of the
    reference set's own leave-one-out mean neighbor distances. I.e.
    'is this query farther from the reference than reference points
    typically are from each other?'

    Returns a boolean mask over X: True == low density (UNSUPPORTED).

    This gate carries NO coverage guarantee; it only decides whether
    the reference is applicable at all. It runs BEFORE labels.
    # UPGRADE: replace the single global percentile with a local
    #          kNN density ratio. See CONFORMAL_UPGRADES.md.
    """
    nn = KNeighborsClassifier(n_neighbors=k)  # only used for kneighbors()
    nn.fit(X_reference, np.zeros(len(X_reference)))

    # Reference leave-one-out: ask for k+1 neighbors and drop self (col 0).
    ref_dist, _ = nn.kneighbors(X_reference, n_neighbors=k + 1)
    ref_loo_mean = ref_dist[:, 1:].mean(axis=1)
    gate = np.percentile(ref_loo_mean, percentile)

    # Query: mean distance to its k reference neighbors.
    q_dist, _ = nn.kneighbors(X, n_neighbors=k)
    q_mean = q_dist.mean(axis=1)

    return q_mean > gate


def aps_scores(q: np.ndarray) -> np.ndarray:
    """Stage 4 -- APS nonconformity score s_y(x) for every label.

    For each row, sort labels by probability descending and take the
    cumulative sum down to (and including) each label, then unsort back
    to the original class-index layout.

    Input  q: (n, K) probability rows.
    Output s: (n, K) scores; s[i, y] is the APS score of label y at i.
    Matches MAPIE's cumulated-score definition (Romano et al. 2020).
    # UPGRADE: TACP adds a frequency-aware penalty to s here. See md.
    """
    order = np.argsort(-q, axis=1)              # indices, most prob first
    q_sorted = np.take_along_axis(q, order, axis=1)
    cum_sorted = np.cumsum(q_sorted, axis=1)    # score in SORTED order
    s = np.empty_like(cum_sorted)
    np.put_along_axis(s, order, cum_sorted, axis=1)  # back to class order
    return s


def aps_quantile(s_true_calibration: np.ndarray, alpha: float) -> float:
    """Stage 5 -- the calibrated threshold qhat.

    Finite-sample-corrected empirical quantile of the calibration
    points' TRUE-LABEL APS scores. The (n+1) and the ceiling are what
    make coverage exact for a genuinely new point; do not 'simplify'
    them away.

    s_true_calibration: shape (n_cal,), each entry s_{y_i}(x_i).
    """
    n = len(s_true_calibration)
    if n == 0:
        raise ValueError("empty calibration set")
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    rank = min(rank, n)                         # clip: alpha too small for n
    return float(np.sort(s_true_calibration)[rank - 1])


def build_sets(s: np.ndarray, qhat: float) -> np.ndarray:
    """Stage 6 -- prediction sets C(x) = { y : s_y(x) <= qhat }.

    include_last_label=True semantics: keep every label at or below
    qhat; if that yields an empty set (can happen only when the single
    most-probable label already exceeds qhat), keep that top label so
    sets are NEVER empty. Deterministic -- no randomized tie-breaking.

    Output: boolean membership matrix, shape (n, K).
    """
    sets = s <= qhat
    empty = ~sets.any(axis=1)
    if empty.any():
        top = np.argmax(-s[empty], axis=1)      # smallest-score label
        sets[np.where(empty)[0], top] = True
    return sets


def per_class_coverage(
    sets: np.ndarray,
    y_true: np.ndarray,
    n_classes: int,
    alpha: float,
) -> dict:
    """Stage 7 -- coverage REPORT (not an action).

    Marginal coverage, per-class coverage, per-class n, and CovGap.
    Read this to decide whether class imbalance bit you. The decision
    of what to DO about it (Mondrian / TACP / collect more data) lives
    in CONFORMAL_UPGRADES.md, not here.

    sets: (n, K) membership. y_true: (n,) true labels.
    """
    covered = sets[np.arange(len(y_true)), y_true]
    target = 1 - alpha

    per_class_cov, per_class_n = {}, {}
    for c in range(n_classes):
        mask = y_true == c
        per_class_n[c] = int(mask.sum())
        per_class_cov[c] = float(covered[mask].mean()) if mask.any() else None

    gaps = [abs(per_class_cov[c] - target)
            for c in range(n_classes) if per_class_cov[c] is not None]
    cov_gap = float(np.mean(gaps)) if gaps else None

    return {
        "marginal_coverage": float(covered.mean()),
        "target": target,
        "per_class_coverage": per_class_cov,
        "per_class_n": per_class_n,
        "cov_gap": cov_gap,
        "mean_set_size": float(sets.sum(axis=1).mean()),
    }


# =================================================================
# ORCHESTRATOR -- two functions that walk the stages in order.
# calibrate_conformal runs once; predict_conformal runs per query batch.
# The Calibrated handoff carries all fitted state, so there is no
# hidden global state and the two phases cannot get out of sync.
# =================================================================

@dataclass
class Calibrated:
    """Everything predict_conformal needs from the calibration phase."""
    X_reference: np.ndarray
    y_reference: np.ndarray
    k: int
    n_classes: int
    alpha: float
    qhat: float
    density_percentile: float
    label_names: Optional[list]  # length K, or None


def calibrate_conformal(
    X_reference: np.ndarray,
    y_reference: np.ndarray,
    X_calibration: np.ndarray,
    y_calibration: np.ndarray,
    k: int,
    alpha: float,
    n_classes: int,
    density_percentile: float = 95.0,
    label_names: Optional[list] = None,
) -> Calibrated:
    """Calibration phase. Builds the KNN neighbor pool and computes qhat.

    reference and calibration are SEPARATE arguments and must be
    disjoint -- this is how neighbor leakage is prevented structurally.
    """
    # Stage 2 (calibration): KNN vote for each calibration point,
    #                        scored against the reference pool only.
    q_cal = knn_probabilities(X_reference, y_reference,
                              X_calibration, k, n_classes)

    # Stage 4 (calibration): APS score of each calibration point's TRUE label.
    s_cal = aps_scores(q_cal)
    s_true = s_cal[np.arange(len(y_calibration)), y_calibration]

    # Stage 5: calibrate the threshold.
    qhat = aps_quantile(s_true, alpha)
    # UPGRADE: Mondrian / clustered / TACP all change THIS step (a
    #          per-class or reweighted qhat). See CONFORMAL_UPGRADES.md.

    return Calibrated(
        X_reference=X_reference, y_reference=y_reference, k=k,
        n_classes=n_classes, alpha=alpha, qhat=qhat,
        density_percentile=density_percentile, label_names=label_names,
    )


def predict_conformal(cal: Calibrated, X_query: np.ndarray) -> list[dict]:
    """Prediction phase. One result dict per query point.

    Walks the stages in order: feature filter -> density gate -> KNN
    vote -> APS -> set -> status. Points failing a gate get the
    corresponding status and an empty set.

    NOTE: this file's feature filter only catches NaN/inf. Any richer
    'not_evaluated' logic (missing timepoint, no embedding) is the
    caller's job -- such rows can simply be withheld from X_query.
    """
    n = len(X_query)
    results: list[dict] = [None] * n

    # Stage 1 -- feature validity (structural; pre-KNN).
    bad = ~np.isfinite(X_query).all(axis=1)

    # Stage 3 -- density gate, only for feature-valid rows.
    ok = np.where(~bad)[0]
    low_density = np.zeros(n, dtype=bool)
    if len(ok):
        flags = density_flags(cal.X_reference, X_query[ok],
                              cal.k, cal.density_percentile)
        low_density[ok] = flags

    # Rows that survive both gates get labeled.
    live = np.where(~bad & ~low_density)[0]
    if len(live):
        # Stage 2 -- KNN vote.
        q = knn_probabilities(cal.X_reference, cal.y_reference,
                             X_query[live], cal.k, cal.n_classes)
        # Stage 4 -- APS scores.
        s = aps_scores(q)
        # Stage 6 -- build sets at the calibrated threshold.
        sets = build_sets(s, cal.qhat)

    def names(mask_row):
        idx = np.where(mask_row)[0].tolist()
        if cal.label_names is None:
            return idx
        return [cal.label_names[i] for i in idx]

    # Assemble results with status (Stage 6 labeling).
    live_pos = {int(i): j for j, i in enumerate(live)}
    for i in range(n):
        if bad[i]:
            results[i] = {"index": i, "status": "not_evaluated",
                          "prediction_set": [], "set_size": 0, "q": None}
        elif low_density[i]:
            results[i] = {"index": i, "status": "low_density",
                          "prediction_set": [], "set_size": 0, "q": None}
        else:
            j = live_pos[i]
            size = int(sets[j].sum())
            results[i] = {
                "index": i,
                "status": "assigned" if size == 1 else "ambiguous",
                "prediction_set": names(sets[j]),
                "set_size": size,
                "q": q[j].tolist(),
            }
    return results


# =================================================================
# RUNNING-EXAMPLE DEMO -- reproduces the worked example from the design
# discussion and doubles as a smoke test. Run:  python conformal_sets.py
# =================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    LABELS = ["NP", "LtH", "Int", "HtL"]
    K = len(LABELS)

    # Four Gaussian blobs in 2D with modest class imbalance (NP is the
    # head). Noise is high relative to center spacing, so neighborhoods
    # genuinely mix: clean cores still give singletons, but seams give
    # ambiguous multi-label sets and APS scores spread well below 1.0.
    centers = np.array([[0, 0], [2.6, 0], [1.3, 2.4], [3.9, 2.4]], float)
    sizes = [1600, 600, 240, 360]             # NP head, Int rare tail
    X = np.vstack([rng.normal(centers[c], 1.3, (sizes[c], 2))
                   for c in range(K)])
    y = np.concatenate([np.full(sizes[c], c) for c in range(K)])

    # Caller-side split into reference / calibration / test (stratified-ish).
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    n_ref = int(0.6 * len(X)); n_cal = int(0.2 * len(X))
    Xr, yr = X[:n_ref], y[:n_ref]
    Xc, yc = X[n_ref:n_ref + n_cal], y[n_ref:n_ref + n_cal]
    Xt, yt = X[n_ref + n_cal:], y[n_ref + n_cal:]

    cal = calibrate_conformal(Xr, yr, Xc, yc, k=15, alpha=0.10,
                              n_classes=K, label_names=LABELS)
    print(f"qhat = {cal.qhat:.3f}")

    # The single worked-example query: in the NP/LtH seam -- supported
    # (passes the density gate) yet genuinely ambiguous.
    x_demo = np.array([[1.6, 0.0]])
    q_demo = knn_probabilities(Xr, yr, x_demo, k=15, n_classes=K)[0]
    s_demo = aps_scores(q_demo[None, :])[0]
    print("q(x) =", {LABELS[i]: round(q_demo[i], 2) for i in range(K)})
    print("s(x) =", {LABELS[i]: round(s_demo[i], 2) for i in range(K)})
    print("set  =", predict_conformal(cal, x_demo)[0]["prediction_set"])

    # Coverage report on the held-out test split.
    test_sets_dicts = predict_conformal(cal, Xt)
    membership = np.zeros((len(Xt), K), dtype=bool)
    for j, r in enumerate(test_sets_dicts):
        for nm in r["prediction_set"]:
            membership[j, LABELS.index(nm)] = True
    report = per_class_coverage(membership, yt, n_classes=K, alpha=0.10)
    print("\ncoverage report:")
    print(f"  marginal = {report['marginal_coverage']:.3f} "
          f"(target {report['target']})")
    print(f"  CovGap   = {report['cov_gap']:.3f}")
    for c in range(K):
        print(f"  {LABELS[c]:>4}: cov={report['per_class_coverage'][c]:.3f}"
              f"  n={report['per_class_n'][c]}")
