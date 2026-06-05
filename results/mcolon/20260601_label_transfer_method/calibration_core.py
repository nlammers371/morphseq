"""
Reference-null calibration for CEP290 label transfer.

The core statistical move: stop treating raw KNN vote fractions (q) as calibrated
probabilities. Instead, use reference leave-one-out (LOO) to learn how those vote
patterns behave when the truth is known, then interpret a new q-vector through that
empirical null.

    P(label | q, local region)  ∝  P(q | label) × P(label | local region)

where:
    q             = distance-weighted KNN label distribution, small K (sharp evidence)
    local_prior   = label distribution, larger K (smoother local context)
    reference_null= q_ref / local_prior_ref vectors for every reference image, with
                    that image left out of its own neighbor search.

This module provides the *building blocks* — likelihood estimators, priors, and the
posterior combiner — so they can be compared empirically on the reference LOO table
(`run_calibration_benchmark.py`) BEFORE any are applied to query data.

Design notes (the known failure mode this guards against):
    Not Penetrant (NP) is ~10x more abundant than Intermediate. A naive likelihood
    ("what labels are common among nearby q-vectors?") and a naive prior ("raw local
    counts") both let NP win by sheer abundance — "a black hole with a lab notebook."
    The class-balanced likelihood and prevalence-corrected prior each correct one side.

Nothing here decides which estimator/prior is best. That is an empirical question
answered on the reference LOO table.

Usage:
    conda run -n segmentation_grounded_sam --no-capture-output python ...
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


MAIN_LABELS = ["Low_to_High", "High_to_Low", "Intermediate", "Not Penetrant"]


# ===========================================================================
# Stage 1: Build the reference leave-one-out (LOO) null
# ===========================================================================

def build_reference_loo_table(
    ref_X: np.ndarray,
    ref_labels: np.ndarray,
    labels: list = MAIN_LABELS,
    k_small: int = 15,
    k_prior: int = 100,
    metric: str = "euclidean",
    epsilon: float = 1e-8,
    ref_hpf: np.ndarray = None,
) -> pd.DataFrame:
    """
    For every reference image, compute its q-vector (small-K, distance-weighted
    label distribution) and local_prior vector (large-K), each with the image
    itself excluded from the neighbor search.

    This is the empirical null: many examples of the (q, local_prior) vectors that
    each TRUE label produces. Downstream estimators read only this table.

    Parameters
    ----------
    ref_X       : (n_ref, n_features) reference feature matrix.
    ref_labels  : (n_ref,) true labels for each reference image.
    labels      : ordered label universe (defines column order of q / prior vectors).
    k_small     : neighbors for the sharp q-vector.
    k_prior     : neighbors for the broader local-context prior. Must be > k_small.
    metric      : distance metric.
    epsilon     : added to distances for the 1/d weighting.

    Returns
    -------
    DataFrame, one row per reference image:
        ref_index, true_label,
        q_<label>            for each label (distance-weighted, small K, self excluded)
        local_prior_<label>  for each label (distance-weighted, K_small+1..K_prior ring,
                              i.e. the "ring prior" that excludes the inner q neighbors)
        local_prior_full_<label> (distance-weighted, full 1..K_prior, self excluded)
        mean_knn_distance    (mean distance to the small-K neighbors, self excluded)
        raw_pred_label       (argmax of q)
        raw_top_probability  (max of q)

    Notes
    -----
    Two prior flavors are stored so the benchmark can compare them without recompute:
      - local_prior_*      : RING prior (excludes inner k_small) -> disjoint from q,
                             cleaner empirical-Bayes (no double-counting of evidence).
      - local_prior_full_* : full large-K prior (overlaps q). The classic but
                             double-counting version.
    Prevalence correction is applied LATER (in the prior functions), not here, so the
    raw enrichment information is preserved.
    """
    if k_prior <= k_small:
        raise ValueError(f"k_prior ({k_prior}) must exceed k_small ({k_small}).")

    n_ref = len(ref_X)
    k_query = k_prior + 1  # +1 to drop self
    if k_query > n_ref:
        raise ValueError(
            f"k_prior+1 ({k_query}) exceeds reference size ({n_ref}); "
            f"reduce k_prior."
        )

    nn = NearestNeighbors(n_neighbors=k_query, metric=metric)
    nn.fit(ref_X)
    dists, idxs = nn.kneighbors(ref_X)

    # Drop self (rank 0)
    dists = dists[:, 1:]          # (n_ref, k_prior)
    idxs = idxs[:, 1:]            # (n_ref, k_prior)
    neighbor_labels = ref_labels[idxs]  # (n_ref, k_prior)
    weights = 1.0 / (dists + epsilon)   # (n_ref, k_prior)

    records = []
    for i in range(n_ref):
        rec = {"ref_index": i, "true_label": ref_labels[i]}

        # --- q: small-K distance-weighted label distribution ---
        q_lbl = neighbor_labels[i, :k_small]
        q_w = weights[i, :k_small]
        q_vec = _weighted_label_dist(q_lbl, q_w, labels)
        for lbl in labels:
            rec[f"q_{lbl}"] = q_vec[lbl]

        # --- ring prior: neighbors k_small..k_prior (disjoint from q) ---
        ring_lbl = neighbor_labels[i, k_small:k_prior]
        ring_w = weights[i, k_small:k_prior]
        ring_vec = _weighted_label_dist(ring_lbl, ring_w, labels)
        for lbl in labels:
            rec[f"local_prior_{lbl}"] = ring_vec[lbl]

        # --- full prior: neighbors 0..k_prior (overlaps q) ---
        full_lbl = neighbor_labels[i, :k_prior]
        full_w = weights[i, :k_prior]
        full_vec = _weighted_label_dist(full_lbl, full_w, labels)
        for lbl in labels:
            rec[f"local_prior_full_{lbl}"] = full_vec[lbl]

        rec["mean_knn_distance"] = float(dists[i, :k_small].mean())
        rec["raw_pred_label"] = max(q_vec, key=q_vec.get)
        rec["raw_top_probability"] = max(q_vec.values())
        if ref_hpf is not None:
            rec["ref_hpf"] = float(ref_hpf[i])
        records.append(rec)

    return pd.DataFrame(records)


def _weighted_label_dist(neighbor_labels: np.ndarray, weights: np.ndarray,
                         labels: list) -> dict:
    """Distance-weighted fraction of weight assigned to each label. Sums to 1."""
    total = weights.sum()
    if total <= 0:
        return {lbl: 0.0 for lbl in labels}
    out = {}
    for lbl in labels:
        out[lbl] = float(weights[neighbor_labels == lbl].sum() / total)
    return out


def compute_query_vectors(
    query_X: np.ndarray,
    ref_X: np.ndarray,
    ref_labels: np.ndarray,
    labels: list = MAIN_LABELS,
    k_small: int = 15,
    k_prior: int = 100,
    metric: str = "euclidean",
    epsilon: float = 1e-8,
) -> pd.DataFrame:
    """
    Same q / local_prior computation as the reference LOO table, but for query
    images against the (full) reference set. No self-exclusion (query != reference).

    Returns one row per query image with the same q_*, local_prior_*,
    local_prior_full_*, mean_knn_distance, raw_pred_label, raw_top_probability cols.
    """
    if k_prior <= k_small:
        raise ValueError(f"k_prior ({k_prior}) must exceed k_small ({k_small}).")
    if k_prior > len(ref_X):
        raise ValueError(f"k_prior ({k_prior}) exceeds reference size ({len(ref_X)}).")

    nn = NearestNeighbors(n_neighbors=k_prior, metric=metric)
    nn.fit(ref_X)
    dists, idxs = nn.kneighbors(query_X)
    neighbor_labels = ref_labels[idxs]
    weights = 1.0 / (dists + epsilon)

    records = []
    for i in range(len(query_X)):
        rec = {"query_index": i}
        q_vec = _weighted_label_dist(neighbor_labels[i, :k_small],
                                     weights[i, :k_small], labels)
        for lbl in labels:
            rec[f"q_{lbl}"] = q_vec[lbl]
        ring_vec = _weighted_label_dist(neighbor_labels[i, k_small:k_prior],
                                        weights[i, k_small:k_prior], labels)
        for lbl in labels:
            rec[f"local_prior_{lbl}"] = ring_vec[lbl]
        full_vec = _weighted_label_dist(neighbor_labels[i, :k_prior],
                                        weights[i, :k_prior], labels)
        for lbl in labels:
            rec[f"local_prior_full_{lbl}"] = full_vec[lbl]
        rec["mean_knn_distance"] = float(dists[i, :k_small].mean())
        rec["raw_pred_label"] = max(q_vec, key=q_vec.get)
        rec["raw_top_probability"] = max(q_vec.values())
        records.append(rec)
    return pd.DataFrame(records)


# ===========================================================================
# Stage 2: Density null (the outlier gate)
# ===========================================================================

def density_percentile(query_mean_knn_dist: np.ndarray,
                       ref_mean_knn_dist: np.ndarray) -> np.ndarray:
    """
    Percentile rank of each query image's mean-KNN-distance against the reference
    LOO mean-KNN-distance distribution. High percentile = far from reference =
    candidate outlier. This is a distribution test, NOT an average-distance cutoff.
    """
    ref_sorted = np.sort(ref_mean_knn_dist)
    # 'right' so a query equal to the max ref distance -> 100th percentile
    ranks = np.searchsorted(ref_sorted, query_mean_knn_dist, side="right")
    return 100.0 * ranks / len(ref_sorted)


# ===========================================================================
# Stage 4: Likelihood estimators  P(q | label)
# ===========================================================================
# Each estimator takes:
#   q_query   : (n_query, n_labels) array of query q-vectors
#   q_ref     : (n_ref, n_labels) array of reference-null q-vectors
#   ref_true  : (n_ref,) true labels of the reference-null rows
#   labels    : ordered label list (columns of q arrays)
# and returns:
#   likelihood : (n_query, n_labels) array, P(q | label) up to a per-row constant.
# Estimators do NOT need to be normalized across labels (the posterior step
# normalizes); but class-balancing is each estimator's own responsibility.


def _q_space_neighbors(q_query, q_ref, k, metric, drop_self):
    """
    Find k reference-null neighbors in q-space for each query q-vector.

    When q_query IS q_ref (evaluating on the null itself), each point's nearest
    q-space neighbor is itself (distance 0), which would leak the true label. Set
    drop_self=True to query k+1 and discard rank 0. Auto-detected by the estimators
    via object identity.
    """
    n_query = (k + 1) if drop_self else k
    n_query = min(n_query, len(q_ref))
    nn = NearestNeighbors(n_neighbors=n_query, metric=metric)
    nn.fit(q_ref)
    _, idxs = nn.kneighbors(q_query)
    if drop_self:
        idxs = idxs[:, 1:]  # drop self-match
    return idxs


def likelihood_raw_q_knn(q_query, q_ref, ref_true, labels,
                         k: int = 50, metric: str = "euclidean") -> np.ndarray:
    """
    Option A (raw): for each query q-vector, find its k nearest reference-null
    q-vectors and use the *fraction* of those neighbors carrying each label as the
    likelihood. This is the simple, abundance-sensitive baseline: common labels win
    in q-space simply because there are more of them.
    """
    labels = list(labels)
    drop_self = q_query is q_ref
    idxs = _q_space_neighbors(q_query, q_ref, k, metric, drop_self)
    neigh_labels = ref_true[idxs]  # (n_query, k)
    out = np.zeros((len(q_query), len(labels)))
    for j, lbl in enumerate(labels):
        out[:, j] = (neigh_labels == lbl).mean(axis=1)
    return out


def likelihood_balanced_q_knn(q_query, q_ref, ref_true, labels,
                              k: int = 50, metric: str = "euclidean") -> np.ndarray:
    """
    Option B (class-balanced): same neighbor search as raw, but normalize each
    label's neighbor count by that label's TOTAL prevalence in the reference null.
    This converts "how many nearby q-vectors are label L" into "how ENRICHED label L
    is near this q-vector, relative to its base rate" — a proper class-conditional
    likelihood that does not let abundant NP dominate.

        score(L) = (count of L among k neighbors) / (total count of L in q_ref)
    """
    labels = list(labels)
    drop_self = q_query is q_ref
    idxs = _q_space_neighbors(q_query, q_ref, k, metric, drop_self)
    neigh_labels = ref_true[idxs]  # (n_query, k)

    # Base rates (avoid divide-by-zero for absent labels)
    base_counts = {lbl: max((ref_true == lbl).sum(), 1) for lbl in labels}

    out = np.zeros((len(q_query), len(labels)))
    for j, lbl in enumerate(labels):
        neigh_count = (neigh_labels == lbl).sum(axis=1).astype(float)
        out[:, j] = neigh_count / base_counts[lbl]
    return out


def likelihood_label_profile_distance(q_query, q_ref, ref_true, labels,
                                      method: str = "jensenshannon") -> np.ndarray:
    """
    Option C: compare the query q-vector to each label's MEAN q-vector profile
    (the centroid of that label's reference-null q-vectors), via a distribution
    distance, and convert distance to a likelihood-like score.

    Class-balanced by construction: each label contributes exactly one centroid, so
    abundance does not tilt the comparison. The distance->score conversion uses a
    softmax over negative distances (temperature 1).

        score(L) ∝ exp(-distance(q_query, centroid_L))
    """
    from scipy.spatial.distance import jensenshannon

    labels = list(labels)
    # Per-label centroid q-vector
    centroids = {}
    for lbl in labels:
        mask = ref_true == lbl
        if mask.sum() == 0:
            centroids[lbl] = np.full(len(labels), 1.0 / len(labels))
        else:
            centroids[lbl] = q_ref[mask].mean(axis=0)

    out = np.zeros((len(q_query), len(labels)))
    for i in range(len(q_query)):
        qv = q_query[i]
        dists = np.empty(len(labels))
        for j, lbl in enumerate(labels):
            cv = centroids[lbl]
            if method == "jensenshannon":
                # JS distance needs non-negative vectors summing to 1; q already does
                d = jensenshannon(qv + 1e-12, cv + 1e-12)
                d = 0.0 if np.isnan(d) else d
            elif method == "l1":
                d = np.abs(qv - cv).sum()
            else:  # squared euclidean
                d = np.sum((qv - cv) ** 2)
            dists[j] = d
        # softmax over -distance
        e = np.exp(-(dists - dists.min()))
        out[i] = e / e.sum()
    return out


LIKELIHOOD_ESTIMATORS = {
    "raw_q_knn": likelihood_raw_q_knn,
    "balanced_q_knn": likelihood_balanced_q_knn,
    "label_profile_distance": likelihood_label_profile_distance,
}


# ===========================================================================
# Stage 3/5: Priors  P(label | local region)
# ===========================================================================
# Each prior takes:
#   prior_vectors : (n, n_labels) the local_prior (ring) or local_prior_full vectors
#   ref_true      : (n_ref,) reference-null true labels (for global base rates)
#   labels        : ordered labels
# and returns:
#   prior : (n, n_labels), each row a normalized prior over labels.


def prior_uniform(prior_vectors, ref_true, labels) -> np.ndarray:
    n = len(prior_vectors)
    p = np.full((n, len(labels)), 1.0 / len(labels))
    return p


def prior_raw_local(prior_vectors, ref_true, labels) -> np.ndarray:
    """Use the local label distribution directly (abundance-sensitive)."""
    p = np.asarray(prior_vectors, dtype=float).copy()
    row = p.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    return p / row


def prior_prevalence_corrected(prior_vectors, ref_true, labels) -> np.ndarray:
    """
    Divide the local label distribution by each label's GLOBAL frequency, then
    renormalize. Encodes local ENRICHMENT (over base rate) rather than local
    abundance — so a region merely full of common NP does not get an NP-heavy prior.
    """
    labels = list(labels)
    p = np.asarray(prior_vectors, dtype=float).copy()
    n_ref = len(ref_true)
    base_rate = np.array([
        max((ref_true == lbl).sum(), 1) / n_ref for lbl in labels
    ])
    p = p / base_rate[np.newaxis, :]
    row = p.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    return p / row


PRIOR_ESTIMATORS = {
    "uniform": prior_uniform,
    "raw_local": prior_raw_local,
    "prevalence_corrected": prior_prevalence_corrected,
}


# ===========================================================================
# Stage 5: Combine likelihood × prior -> calibrated posterior
# ===========================================================================

def combine_posterior(likelihood: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """
    posterior(label) ∝ likelihood(label) × prior(label), normalized per row.
    Robust to all-zero rows (falls back to uniform).
    """
    post = likelihood * prior
    row = post.sum(axis=1, keepdims=True)
    bad = (row[:, 0] <= 0) | ~np.isfinite(row[:, 0])
    out = np.where(row > 0, post / np.where(row == 0, 1.0, row), post)
    if bad.any():
        out[bad] = 1.0 / likelihood.shape[1]
    return out


def calibrated_predictions(posterior: np.ndarray, labels: list):
    """Return (argmax_label array, top_probability array) from a posterior matrix."""
    labels = np.asarray(list(labels))
    idx = posterior.argmax(axis=1)
    return labels[idx], posterior[np.arange(len(posterior)), idx]


# ===========================================================================
# Time-aware reference-null calibration  (Section 7 of findings doc)
# ===========================================================================
# Time enters as a KERNEL WEIGHT on reference-null examples, NOT as a third
# multiplicative term (the abandoned local-region prior failed precisely because it
# multiplied in a redundant second view). Here there is ONE estimator — a weighted vote
# among reference-null examples — with the weight combining q-similarity and
# developmental-time similarity:
#
#     w_ij = W_q(q_i, q_j) × W_t(t_i, t_j)
#     P(label=L | q_i, t_i) = Σ_j w_ij·1[true_j == L] / Σ_j w_ij
#
# The q-similarity weight comes from a chosen q-distance metric (jensenshannon / l1 /
# euclidean) passed through a Gaussian kernel; OR from a class-balanced count-based
# scheme (raw_q_knn / balanced_q_knn) where the "weight" is membership in the top-K
# q-neighbors, optionally divided by class base rate.

from scipy.spatial.distance import cdist


def median_pairwise_q_distance(q_ref: np.ndarray, metric: str = "jensenshannon",
                               sample: int = 2000, seed: int = 0) -> float:
    """
    Robust default for σ_q: median pairwise q-distance on a random subsample
    (full 13k×13k is wasteful). Standard kernel-bandwidth heuristic.
    """
    rng = np.random.default_rng(seed)
    n = len(q_ref)
    idx = rng.choice(n, size=min(sample, n), replace=False)
    D = cdist(q_ref[idx], q_ref[idx], metric=metric)
    iu = np.triu_indices_from(D, k=1)
    med = float(np.median(D[iu]))
    return med if med > 0 else 1.0


def calibrate_q_time(
    query_q: np.ndarray,
    query_hpf: np.ndarray,
    ref_q: np.ndarray,
    ref_hpf: np.ndarray,
    ref_labels: np.ndarray,
    labels: list = MAIN_LABELS,
    likelihood: str = "balanced_q_knn",
    q_metric: str = "jensenshannon",
    k_q: int = 50,
    sigma_q: float = None,
    sigma_t: float = 3.0,
    drop_self: bool = False,
) -> np.ndarray:
    """
    Time-aware reference-null calibration. Returns (n_query, n_labels) posterior-like
    probabilities P(label | q_i, t_i, R), uniform prior.

    likelihood / q-similarity scheme (W_q):
        'raw_q_knn'        : top-k_q q-neighbors vote, each weight = W_t only.
        'balanced_q_knn'   : same, but each neighbor's vote divided by its class base
                             rate (breaks NP abundance dominance).
        'kernel'           : ALL reference rows weighted by exp(-D_q²/2σ_q²)·W_t, where
                             D_q uses q_metric (jensenshannon / cityblock / euclidean).
        'kernel_balanced'  : as 'kernel', but each label's summed weight is divided by
                             its class base rate (breaks NP abundance dominance — without
                             this the global kernel sum is swamped by the majority class).

    W_t = exp(-Δt² / 2σ_t²), Δt = |hpf_i - hpf_j|.

    drop_self=True drops each query's own nearest q-neighbor (use when query IS the
    reference-null, i.e. evaluating on the LOO table itself).
    """
    labels = list(labels)
    n_q, n_lbl = len(query_q), len(labels)
    out = np.zeros((n_q, n_lbl))

    # Precompute base rates for balancing
    base_counts = np.array([max((ref_labels == l).sum(), 1) for l in labels], dtype=float)
    label_to_col = {l: j for j, l in enumerate(labels)}
    ref_label_col = np.array([label_to_col[l] for l in ref_labels])

    if likelihood in ("raw_q_knn", "balanced_q_knn"):
        # Top-k_q neighbors in q-space; weight by time kernel only.
        idxs = _q_space_neighbors(query_q, ref_q, k_q, "euclidean", drop_self)
        for i in range(n_q):
            nb = idxs[i]
            dt = np.abs(query_hpf[i] - ref_hpf[nb])
            w_t = np.exp(-(dt ** 2) / (2 * sigma_t ** 2))
            cols = ref_label_col[nb]
            acc = np.zeros(n_lbl)
            np.add.at(acc, cols, w_t)
            if likelihood == "balanced_q_knn":
                acc = acc / base_counts
            s = acc.sum()
            out[i] = acc / s if s > 0 else 1.0 / n_lbl

    elif likelihood in ("kernel", "kernel_balanced"):
        balance = likelihood == "kernel_balanced"
        if sigma_q is None:
            sigma_q = median_pairwise_q_distance(ref_q, metric=q_metric)
        # Process in row chunks to bound memory (cdist of full query batch is fine,
        # but ref is large -> chunk query rows).
        chunk = 512
        for start in range(0, n_q, chunk):
            qsl = slice(start, min(start + chunk, n_q))
            D_q = cdist(query_q[qsl], ref_q, metric=q_metric)  # (c, n_ref)
            W_q = np.exp(-(D_q ** 2) / (2 * sigma_q ** 2))
            dt = np.abs(query_hpf[qsl][:, None] - ref_hpf[None, :])
            W_t = np.exp(-(dt ** 2) / (2 * sigma_t ** 2))
            W = W_q * W_t  # (c, n_ref)
            if drop_self:
                # zero each row's self-match (the minimum-distance, here exact 0 dist)
                self_idx = np.arange(start, min(start + chunk, n_q))
                for r, gi in enumerate(self_idx):
                    if gi < len(ref_q):
                        W[r, gi] = 0.0
            # accumulate weight per label
            for j, l in enumerate(labels):
                mask = ref_label_col == j
                out[qsl, j] = W[:, mask].sum(axis=1)
            if balance:
                # divide each label's summed kernel weight by its base rate, so the
                # majority class cannot win on sheer mass.
                out[qsl] = out[qsl] / base_counts[None, :]
            row = out[qsl].sum(axis=1, keepdims=True)
            row[row == 0] = 1.0
            out[qsl] = out[qsl] / row
    else:
        raise ValueError(f"unknown likelihood scheme: {likelihood}")

    return out


def calibrated_margin(posterior: np.ndarray) -> np.ndarray:
    """Top probability minus second-highest, per row. A confidence/uncertainty measure."""
    s = np.sort(posterior, axis=1)
    return s[:, -1] - s[:, -2]
