"""
coverage_diagnostics.py
=======================
CovGap diagnostic with Wilson confidence intervals, and Mondrian (class-
conditional) conformal calibration.

Two independent jobs:

    covgap_report(membership, y_true, alpha)
        -> per-class coverage, Wilson CIs, gap vs target, subsidization flag.
        Pure diagnostic: touches no sets, changes nothing.

    mondrian_qhat(aps_scores_cal, y_cal, alpha, n_classes)
        -> dict mapping class index -> qhat_c.
        Drop-in replacement for a single aps_quantile call.

    build_sets_mondrian(s, qhat_per_class)
        -> bool membership matrix, same shape contract as build_sets().

The rest of conformal_sets.py is unchanged: APS scores, knn_probabilities,
and the rest of the pipeline are reused as-is.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Wilson confidence interval (one proportion)
# ─────────────────────────────────────────────────────────────────────────────

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Two-sided Wilson CI for a proportion k/n.

    Returns (lo, hi). Safe at k=0 and k=n.
    z=1.96 -> 95% CI.
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ─────────────────────────────────────────────────────────────────────────────
# CovGap diagnostic
# ─────────────────────────────────────────────────────────────────────────────

def covgap_report(
    membership: np.ndarray,
    y_true: np.ndarray,
    alpha: float,
    label_names: Optional[list[str]] = None,
    wilson_z: float = 1.96,
) -> dict:
    """Per-class coverage diagnostic with Wilson confidence intervals.

    Parameters
    ----------
    membership : (n, K) bool array — True if class k is in the prediction set.
    y_true     : (n,) int array — true class indices in [0, K).
    alpha      : target miss rate; target coverage = 1 - alpha.
    label_names: optional list of K names for human-readable output.
    wilson_z   : z-score for the Wilson CI (default 1.96 → 95%).

    Returns
    -------
    dict with keys:
        marginal_coverage  float
        target             float  (= 1 - alpha)
        cov_gap            float  (mean |gap_c| across classes with n_c > 0)
        classes            list of per-class dicts, each:
            label          str or int
            n              int
            covered        int
            coverage       float
            wilson_lo      float
            wilson_hi      float
            gap            float  (coverage - target; negative = undercoverage)
            undercovered   bool   (True if wilson_hi < target — statistically
                                   below target, not just noise)
            overcovered    bool   (True if wilson_lo > target)
            subsidizes     bool   (overcovered — this class is eating budget)
    """
    membership = np.asarray(membership, dtype=bool)
    y_true = np.asarray(y_true, dtype=int)
    n, K = membership.shape
    target = 1.0 - alpha

    covered_global = membership[np.arange(n), y_true]
    marginal_coverage = float(covered_global.mean())

    class_records = []
    gaps = []
    for c in range(K):
        mask = y_true == c
        n_c = int(mask.sum())
        if n_c == 0:
            class_records.append({
                "label": label_names[c] if label_names else c,
                "n": 0, "covered": 0,
                "coverage": None, "wilson_lo": None, "wilson_hi": None,
                "gap": None, "undercovered": None, "overcovered": None,
                "subsidizes": None,
            })
            continue
        cov_c = int(covered_global[mask].sum())
        rate_c = cov_c / n_c
        lo, hi = wilson_ci(cov_c, n_c, z=wilson_z)
        gap = rate_c - target
        gaps.append(abs(gap))
        class_records.append({
            "label": label_names[c] if label_names else c,
            "n": n_c,
            "covered": cov_c,
            "coverage": round(rate_c, 6),
            "wilson_lo": round(lo, 6),
            "wilson_hi": round(hi, 6),
            "gap": round(gap, 6),
            "undercovered": bool(hi < target),   # upper CI end still below target
            "overcovered": bool(lo > target),
            "subsidizes": bool(lo > target),
        })

    cov_gap = float(np.mean(gaps)) if gaps else None

    return {
        "marginal_coverage": round(marginal_coverage, 6),
        "target": target,
        "cov_gap": round(cov_gap, 6) if cov_gap is not None else None,
        "classes": class_records,
    }


def print_covgap_report(report: dict) -> None:
    """Pretty-print a covgap_report dict."""
    print(f"  marginal coverage : {report['marginal_coverage']:.4f}  "
          f"(target {report['target']:.4f})")
    print(f"  CovGap            : {report['cov_gap']:.4f}")
    print()
    header = f"  {'class':<22} {'n':>6}  {'cov':>6}  {'95% CI':>14}  {'gap':>7}  {'flag'}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for c in report["classes"]:
        if c["n"] == 0:
            print(f"  {str(c['label']):<22} {'0':>6}  {'—':>6}  {'—':>14}")
            continue
        flag = ""
        if c["undercovered"]:
            flag = "← UNDERCOVERS (subsidized)"
        elif c["overcovered"]:
            flag = "→ overcovering (subsidizes)"
        print(
            f"  {str(c['label']):<22} {c['n']:>6}  {c['coverage']:>6.4f}"
            f"  [{c['wilson_lo']:.4f}, {c['wilson_hi']:.4f}]"
            f"  {c['gap']:>+7.4f}  {flag}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Mondrian calibration
# ─────────────────────────────────────────────────────────────────────────────

def mondrian_qhat(
    s_cal: np.ndarray,
    y_cal: np.ndarray,
    alpha: float,
    n_classes: int,
    min_n_warn: int = 30,
) -> dict[int, float]:
    """Compute a separate qhat for each class (Mondrian conformal).

    Parameters
    ----------
    s_cal      : (n_cal, K) APS score matrix on calibration points.
    y_cal      : (n_cal,) int array of true class indices.
    alpha      : target miss rate per class.
    n_classes  : K.
    min_n_warn : warn if any class has fewer calibration points than this.

    Returns
    -------
    dict mapping class index -> qhat_c (float).
    If a class has zero calibration points, returns 1.0 for that class
    (conservative: include everything).
    """
    from conformal_sets import aps_quantile

    y_cal = np.asarray(y_cal, dtype=int)
    qhats: dict[int, float] = {}
    for c in range(n_classes):
        mask = y_cal == c
        n_c = int(mask.sum())
        if n_c == 0:
            warnings.warn(
                f"Mondrian: class {c} has 0 calibration points — using qhat=1.0 (full sets).",
                UserWarning, stacklevel=2,
            )
            qhats[c] = 1.0
            continue
        if n_c < min_n_warn:
            warnings.warn(
                f"Mondrian: class {c} has only {n_c} calibration points — "
                f"qhat_c will be high-variance.",
                UserWarning, stacklevel=2,
            )
        true_scores_c = s_cal[mask, c]
        qhats[c] = aps_quantile(true_scores_c, alpha)
    return qhats


def build_sets_mondrian(
    s: np.ndarray,
    qhat_per_class: dict[int, float],
) -> np.ndarray:
    """Build prediction sets using per-class Mondrian thresholds.

    Each label c is included if s[i, c] <= qhat_per_class[c].
    include_last_label=True: never emit an empty set.

    Parameters
    ----------
    s               : (n, K) APS score matrix.
    qhat_per_class  : dict mapping class index -> qhat_c.

    Returns
    -------
    (n, K) bool membership matrix.
    """
    n, K = s.shape
    thresholds = np.array([qhat_per_class[c] for c in range(K)], dtype=float)
    sets = s <= thresholds[None, :]          # broadcast: (n, K)

    # never-empty: if a row is all-False, include the top label
    empty = ~sets.any(axis=1)
    if empty.any():
        top = np.argmin(s[empty], axis=1)    # lowest APS = most probable label
        sets[np.where(empty)[0], top] = True

    return sets
