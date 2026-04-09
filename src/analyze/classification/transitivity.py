"""
transitivity.py
---------------
Helpers for tri-state pairwise separability analysis and transitivity/ultrametric
diagnostics for phenotype emergence.

## Tri-state edge model

Each (pair, time_bin) is classified as one of:
  - separated:     pval < p_sep  AND auroc >= auroc_sep
  - not_separated: pval > p_ns   (no AUROC ceiling required)
  - ambiguous:     everything else

## Onset rule (L-consecutive)

For pair (i, j), onset = first time bin t where the pair is `separated`
in that bin AND the next L-1 consecutive bins.

## Transitivity violations

At each time bin, a violation is a triple (A, B, C) where:
  - two edges are confidently of one state
  - the third is confidently incompatible
  - ambiguous edges are excluded from violation counting

## Ultrametric gap diagnostic

For each triple with all three finite onsets (τ1 ≤ τ2 ≤ τ3):
  ultrametric_gap = τ3 - τ2

gap = 0 means tree-consistent onset timing for that triple.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class TransitivityParams:
    """Parameter set for tri-state classification and onset detection.

    Onset rule
    ----------
    A pair is considered durably separated starting at bin t if:
      - pval[t] < p_sep  (the initial bin is significant)
      - among all bins from t onward, >= subsequent_frac have pval < p_sep

    Edge state (for transitivity / Panel B)
    ----------------------------------------
    - separated:     pval < p_sep
    - not_separated: pval > p_ns
    - ambiguous:     p_sep <= pval <= p_ns
    """
    # Separated: pval < p_sep (p-value only; no AUROC gate)
    p_sep: float = 0.05
    # Not-separated: pval > p_ns
    p_ns: float = 0.10
    # Onset: fraction of subsequent bins (t onward) that must have pval < p_sep
    subsequent_frac: float = 0.75
    # Legacy field kept for API compatibility — unused in onset computation
    auroc_sep: float = 0.0
    L: int = 3


# ---------------------------------------------------------------------------
# Tri-state classification
# ---------------------------------------------------------------------------

SEPARATED     = "separated"
NOT_SEPARATED = "not_separated"
AMBIGUOUS     = "ambiguous"


def classify_pair_state(
    pval: float,
    auroc: float,
    params: TransitivityParams,
) -> str:
    """Classify a single (pval, auroc) observation into tri-state.

    Separation requires pval < p_sep AND auroc >= auroc_sep.
    auroc_sep=0.0 (default) effectively disables the AUROC gate.
    """
    if pval < params.p_sep and auroc >= params.auroc_sep:
        return SEPARATED
    if pval > params.p_ns:
        return NOT_SEPARATED
    return AMBIGUOUS


def classify_pair_state_over_time(
    scores_df: pd.DataFrame,
    params: TransitivityParams,
    *,
    time_col: str = "time_bin_center",
    class_i_col: str = "positive_label",
    class_j_col: str = "negative_label",
    pval_col: str = "pval",
    auroc_col: str = "auroc_obs",
) -> pd.DataFrame:
    """
    Assign tri-state edge classification to every (pair, time_bin) row.

    Returns a copy of scores_df with an added column `edge_state`
    taking values: 'separated' | 'not_separated' | 'ambiguous'.

    Also adds `pair_key` column (canonical sorted pair string) for grouping.
    """
    df = scores_df.copy()
    states = []
    keys = []
    for _, row in df.iterrows():
        s = classify_pair_state(row[pval_col], row[auroc_col], params)
        states.append(s)
        a, b = row[class_i_col], row[class_j_col]
        keys.append(f"{min(a,b)}__{max(a,b)}")
    df["edge_state"] = states
    df["pair_key"] = keys
    return df


# ---------------------------------------------------------------------------
# Onset detection (L-consecutive rule)
# ---------------------------------------------------------------------------

def compute_pair_onsets(
    classified_df: pd.DataFrame,
    params: TransitivityParams,
    *,
    time_col: str = "time_bin_center",
    class_i_col: str = "positive_label",
    class_j_col: str = "negative_label",
) -> pd.DataFrame:
    """
    For each unique pair, compute the durable onset time using the L-consecutive rule.

    Onset = first time bin t where the pair is `separated` in t and the
    next L-1 consecutive bins.

    Parameters
    ----------
    classified_df : output of classify_pair_state_over_time()

    Returns
    -------
    DataFrame with columns:
        class_i, class_j, pair_key, onset_hpf (NaN if never separated durably),
        n_separated_bins, n_total_bins, first_separated_bin
    """
    rows = []
    pairs = classified_df["pair_key"].unique()

    for pk in sorted(pairs):
        sub = (
            classified_df[classified_df["pair_key"] == pk]
            .sort_values(time_col)
            .reset_index(drop=True)
        )
        ci = sub[class_i_col].iloc[0]
        cj = sub[class_j_col].iloc[0]
        # canonical order
        if ci > cj:
            ci, cj = cj, ci

        times  = sub[time_col].values
        states = sub["edge_state"].values
        n      = len(times)

        n_sep = int((states == SEPARATED).sum())
        first_sep = float(times[states == SEPARATED][0]) if n_sep > 0 else float("nan")

        # Onset rule: first bin t where pval[t] < p_sep AND
        # >= subsequent_frac of all bins from t onward have pval < p_sep.
        pvals = sub["pval"].values if "pval" in sub.columns else None
        onset = float("nan")
        for i in range(n):
            if states[i] != SEPARATED:
                continue
            # All bins from i onward (inclusive)
            remaining = states[i:]
            if len(remaining) == 0:
                continue
            frac_sep = float(np.sum(remaining == SEPARATED)) / len(remaining)
            if frac_sep >= params.subsequent_frac:
                onset = float(times[i])
                break

        rows.append({
            "class_i": ci,
            "class_j": cj,
            "pair_key": pk,
            "onset_hpf": onset,
            "n_separated_bins": n_sep,
            "n_total_bins": n,
            "first_separated_bin": first_sep,
        })

    return pd.DataFrame(rows)


def build_onset_matrix(onset_df: pd.DataFrame, all_classes: list[str]) -> pd.DataFrame:
    """
    Build symmetric (classes x classes) onset matrix from pair onset DataFrame.
    Diagonal is NaN. Never-separated pairs are NaN.
    """
    mat = pd.DataFrame(index=all_classes, columns=all_classes, dtype=float)
    for _, row in onset_df.iterrows():
        ci, cj, t = row["class_i"], row["class_j"], row["onset_hpf"]
        mat.loc[ci, cj] = t
        mat.loc[cj, ci] = t
    return mat


# ---------------------------------------------------------------------------
# Transitivity violations (tri-state aware)
# ---------------------------------------------------------------------------

@dataclass
class TransitivityViolation:
    """A single non-transitive triple at a time bin."""
    time_bin: float
    a: str
    b: str
    c: str
    # State of each edge
    state_ab: str
    state_bc: str
    state_ac: str
    # Which configuration makes this a violation
    violation_type: str   # e.g. "sep_sep_notsep": A sep B, B sep C, A not-sep C
    auroc_ab: float
    auroc_bc: float
    auroc_ac: float
    pval_ab: float
    pval_bc: float
    pval_ac: float


def _edge_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _lookup(classified_df: pd.DataFrame, a: str, b: str, t: float,
            time_col: str, pval_col: str, auroc_col: str
            ) -> tuple[str, float, float]:
    """Return (state, auroc, pval) for pair (a,b) at time t. Returns ambiguous if missing."""
    pk = f"{min(a,b)}__{max(a,b)}"
    mask = (classified_df["pair_key"] == pk) & np.isclose(classified_df[time_col], t, atol=2.5)
    sub = classified_df[mask]
    if sub.empty:
        return AMBIGUOUS, float("nan"), float("nan")
    row = sub.iloc[0]
    return row["edge_state"], row[auroc_col], row[pval_col]


def detect_timebin_transitivity_violations(
    classified_df: pd.DataFrame,
    params: TransitivityParams,
    *,
    time_col: str = "time_bin_center",
    class_i_col: str = "positive_label",
    class_j_col: str = "negative_label",
    pval_col: str = "pval",
    auroc_col: str = "auroc_obs",
    time_bins: Iterable[float] | None = None,
) -> list[TransitivityViolation]:
    """
    Detect non-transitive triples at each time bin using the tri-state model.

    A violation requires two confidently-classified edges that are inconsistent
    with a third confidently-classified edge. Ambiguous edges are skipped.

    Violation patterns detected (for unordered triple {A,B,C}):
      sep_sep_notsep : A sep B, B sep C, A not-sep C   (separation not transitive)
      ns_ns_sep      : A not-sep B, B not-sep C, A sep C (grouping not transitive)

    Both are biologically interesting but for different reasons.
    """
    if time_bins is None:
        time_bins = sorted(classified_df[time_col].unique())

    all_classes = sorted(
        set(classified_df[class_i_col].unique()) |
        set(classified_df[class_j_col].unique())
    )

    violations: list[TransitivityViolation] = []

    for t in time_bins:
        for a, b, c in combinations(all_classes, 3):
            s_ab, auroc_ab, pval_ab = _lookup(classified_df, a, b, t, time_col, pval_col, auroc_col)
            s_bc, auroc_bc, pval_bc = _lookup(classified_df, b, c, t, time_col, pval_col, auroc_col)
            s_ac, auroc_ac, pval_ac = _lookup(classified_df, a, c, t, time_col, pval_col, auroc_col)

            states = {s_ab, s_bc, s_ac}

            # Skip if any edge is ambiguous — we cannot call a violation
            if AMBIGUOUS in states:
                continue

            # Pattern 1: two separated, one not-separated → separation non-transitive
            if (s_ab == SEPARATED and s_bc == SEPARATED and s_ac == NOT_SEPARATED):
                violations.append(TransitivityViolation(
                    time_bin=t, a=a, b=b, c=c,
                    state_ab=s_ab, state_bc=s_bc, state_ac=s_ac,
                    violation_type="sep_sep_notsep",
                    auroc_ab=auroc_ab, auroc_bc=auroc_bc, auroc_ac=auroc_ac,
                    pval_ab=pval_ab, pval_bc=pval_bc, pval_ac=pval_ac,
                ))
            if (s_ab == SEPARATED and s_ac == SEPARATED and s_bc == NOT_SEPARATED):
                violations.append(TransitivityViolation(
                    time_bin=t, a=a, b=b, c=c,
                    state_ab=s_ab, state_bc=s_bc, state_ac=s_ac,
                    violation_type="sep_sep_notsep",
                    auroc_ab=auroc_ab, auroc_bc=auroc_bc, auroc_ac=auroc_ac,
                    pval_ab=pval_ab, pval_bc=pval_bc, pval_ac=pval_ac,
                ))
            if (s_bc == SEPARATED and s_ac == SEPARATED and s_ab == NOT_SEPARATED):
                violations.append(TransitivityViolation(
                    time_bin=t, a=a, b=b, c=c,
                    state_ab=s_ab, state_bc=s_bc, state_ac=s_ac,
                    violation_type="sep_sep_notsep",
                    auroc_ab=auroc_ab, auroc_bc=auroc_bc, auroc_ac=auroc_ac,
                    pval_ab=pval_ab, pval_bc=pval_bc, pval_ac=pval_ac,
                ))

            # Pattern 2: two not-separated, one separated → grouping non-transitive
            if (s_ab == NOT_SEPARATED and s_bc == NOT_SEPARATED and s_ac == SEPARATED):
                violations.append(TransitivityViolation(
                    time_bin=t, a=a, b=b, c=c,
                    state_ab=s_ab, state_bc=s_bc, state_ac=s_ac,
                    violation_type="ns_ns_sep",
                    auroc_ab=auroc_ab, auroc_bc=auroc_bc, auroc_ac=auroc_ac,
                    pval_ab=pval_ab, pval_bc=pval_bc, pval_ac=pval_ac,
                ))
            if (s_ab == NOT_SEPARATED and s_ac == NOT_SEPARATED and s_bc == SEPARATED):
                violations.append(TransitivityViolation(
                    time_bin=t, a=a, b=b, c=c,
                    state_ab=s_ab, state_bc=s_bc, state_ac=s_ac,
                    violation_type="ns_ns_sep",
                    auroc_ab=auroc_ab, auroc_bc=auroc_bc, auroc_ac=auroc_ac,
                    pval_ab=pval_ab, pval_bc=pval_bc, pval_ac=pval_ac,
                ))
            if (s_bc == NOT_SEPARATED and s_ac == NOT_SEPARATED and s_ab == SEPARATED):
                violations.append(TransitivityViolation(
                    time_bin=t, a=a, b=b, c=c,
                    state_ab=s_ab, state_bc=s_bc, state_ac=s_ac,
                    violation_type="ns_ns_sep",
                    auroc_ab=auroc_ab, auroc_bc=auroc_bc, auroc_ac=auroc_ac,
                    pval_ab=pval_ab, pval_bc=pval_bc, pval_ac=pval_ac,
                ))

    return violations


def summarize_timebin_violation_rate(
    violations: list[TransitivityViolation],
    classified_df: pd.DataFrame,
    all_time_bins: list[float],
    *,
    time_col: str = "time_bin_center",
    class_i_col: str = "positive_label",
    class_j_col: str = "negative_label",
) -> pd.DataFrame:
    """
    Per-time-bin summary of violation counts and evaluable triple counts.

    Returns DataFrame with columns:
        time_bin, n_triples_evaluable, n_violations, n_sep_sep_notsep,
        n_ns_ns_sep, violation_rate
    """
    all_classes = sorted(
        set(classified_df[class_i_col].unique()) |
        set(classified_df[class_j_col].unique())
    )
    n_triples_total = len(list(combinations(all_classes, 3)))

    rows = []
    for t in all_time_bins:
        t_viols = [v for v in violations if np.isclose(v.time_bin, t, atol=0.5)]

        # Count evaluable triples at t (no ambiguous edges)
        n_eval = 0
        for a, b, c in combinations(all_classes, 3):
            s_ab, _, _ = _lookup(classified_df, a, b, t, time_col, "pval", "auroc_obs")
            s_bc, _, _ = _lookup(classified_df, b, c, t, time_col, "pval", "auroc_obs")
            s_ac, _, _ = _lookup(classified_df, a, c, t, time_col, "pval", "auroc_obs")
            if AMBIGUOUS not in {s_ab, s_bc, s_ac}:
                n_eval += 1

        n_v   = len(t_viols)
        n_ssn = sum(1 for v in t_viols if v.violation_type == "sep_sep_notsep")
        n_nns = sum(1 for v in t_viols if v.violation_type == "ns_ns_sep")
        rate  = n_v / n_eval if n_eval > 0 else float("nan")

        rows.append({
            "time_bin": t,
            "n_triples_total": n_triples_total,
            "n_triples_evaluable": n_eval,
            "n_violations": n_v,
            "n_sep_sep_notsep": n_ssn,
            "n_ns_ns_sep": n_nns,
            "violation_rate": rate,
        })

    return pd.DataFrame(rows)


def violations_to_df(violations: list[TransitivityViolation]) -> pd.DataFrame:
    """Convert violation list to tidy DataFrame."""
    if not violations:
        return pd.DataFrame(columns=[
            "time_bin","a","b","c","state_ab","state_bc","state_ac",
            "violation_type","auroc_ab","auroc_bc","auroc_ac",
            "pval_ab","pval_bc","pval_ac",
        ])
    return pd.DataFrame([v.__dict__ for v in violations])


# ---------------------------------------------------------------------------
# Ultrametric gap diagnostic
# ---------------------------------------------------------------------------

@dataclass
class OnsetConsistencySummary:
    """Summary statistics for ultrametric gap diagnostic."""
    n_triples_total: int
    n_triples_evaluable: int   # all three onsets are finite
    n_gap_zero: int
    frac_gap_zero: float
    mean_gap: float
    median_gap: float
    max_gap: float


def check_onset_ultrametric_consistency(
    onset_matrix: pd.DataFrame,
) -> tuple[OnsetConsistencySummary, pd.DataFrame]:
    """
    For each triple (A, B, C) with all three finite onset times,
    compute the ultrametric gap = τ3 - τ2 (sorted ascending).

    A gap of 0 means the two latest onsets coincide → tree-compatible.
    Larger gaps indicate deviation from a clean tree-like emergence structure.

    Parameters
    ----------
    onset_matrix : symmetric (classes x classes) DataFrame, values = onset_hpf

    Returns
    -------
    summary : OnsetConsistencySummary
    triple_df : per-triple DataFrame with columns:
        a, b, c, tau1, tau2, tau3, ultrametric_gap,
        class_tau1, class_tau2, class_tau3
    """
    classes = list(onset_matrix.index)
    n_total = len(list(combinations(classes, 3)))

    rows = []
    for a, b, c in combinations(classes, 3):
        t_ab = onset_matrix.loc[a, b]
        t_bc = onset_matrix.loc[b, c]
        t_ac = onset_matrix.loc[a, c]

        if any(pd.isna(v) for v in [t_ab, t_bc, t_ac]):
            continue

        # Sort
        pairs_and_times = sorted(
            [("ab", float(t_ab)), ("bc", float(t_bc)), ("ac", float(t_ac))],
            key=lambda x: x[1],
        )
        labels_sorted = [p[0] for p in pairs_and_times]
        times_sorted  = [p[1] for p in pairs_and_times]
        tau1, tau2, tau3 = times_sorted
        gap = tau3 - tau2

        rows.append({
            "a": a, "b": b, "c": c,
            "tau1": tau1, "tau2": tau2, "tau3": tau3,
            "pair_tau1": labels_sorted[0],
            "pair_tau2": labels_sorted[1],
            "pair_tau3": labels_sorted[2],
            "ultrametric_gap": gap,
        })

    triple_df = pd.DataFrame(rows)
    n_eval = len(triple_df)

    if n_eval == 0:
        summary = OnsetConsistencySummary(
            n_triples_total=n_total, n_triples_evaluable=0,
            n_gap_zero=0, frac_gap_zero=float("nan"),
            mean_gap=float("nan"), median_gap=float("nan"), max_gap=float("nan"),
        )
        return summary, triple_df

    gaps = triple_df["ultrametric_gap"].values
    n_zero = int((gaps == 0).sum())

    summary = OnsetConsistencySummary(
        n_triples_total=n_total,
        n_triples_evaluable=n_eval,
        n_gap_zero=n_zero,
        frac_gap_zero=n_zero / n_eval,
        mean_gap=float(np.mean(gaps)),
        median_gap=float(np.median(gaps)),
        max_gap=float(np.max(gaps)),
    )
    return summary, triple_df


# ---------------------------------------------------------------------------
# Top-level report builder
# ---------------------------------------------------------------------------

@dataclass
class TransitivityReport:
    params: TransitivityParams
    classified_df: pd.DataFrame            # all (pair, time) rows with edge_state
    onset_df: pd.DataFrame                 # per-pair onset times
    onset_matrix: pd.DataFrame             # symmetric matrix
    timebin_summary: pd.DataFrame          # per-time-bin violation summary
    triple_violations: pd.DataFrame        # all violation rows
    onset_summary: OnsetConsistencySummary
    onset_triple_df: pd.DataFrame          # per-triple ultrametric gaps


def build_transitivity_report(
    scores_df: pd.DataFrame,
    params: TransitivityParams | None = None,
    *,
    time_col: str = "time_bin_center",
    class_i_col: str = "positive_label",
    class_j_col: str = "negative_label",
    pval_col: str = "pval",
    auroc_col: str = "auroc_obs",
) -> TransitivityReport:
    """
    Run the full transitivity pipeline and return a structured report.

    Steps:
      1. Classify all (pair, time_bin) into tri-state
      2. Compute L-consecutive durable onsets
      3. Build onset matrix
      4. Detect time-bin transitivity violations
      5. Summarize per time bin
      6. Compute ultrametric gap diagnostic
    """
    if params is None:
        params = TransitivityParams()

    all_time_bins = sorted(scores_df[time_col].unique())
    all_classes   = sorted(
        set(scores_df[class_i_col].unique()) |
        set(scores_df[class_j_col].unique())
    )

    classified_df = classify_pair_state_over_time(
        scores_df, params,
        time_col=time_col, class_i_col=class_i_col,
        class_j_col=class_j_col, pval_col=pval_col, auroc_col=auroc_col,
    )

    onset_df = compute_pair_onsets(
        classified_df, params,
        time_col=time_col, class_i_col=class_i_col, class_j_col=class_j_col,
    )

    onset_matrix = build_onset_matrix(onset_df, all_classes)

    violations = detect_timebin_transitivity_violations(
        classified_df, params,
        time_col=time_col, class_i_col=class_i_col,
        class_j_col=class_j_col, pval_col=pval_col, auroc_col=auroc_col,
        time_bins=all_time_bins,
    )

    timebin_summary = summarize_timebin_violation_rate(
        violations, classified_df, all_time_bins,
        time_col=time_col, class_i_col=class_i_col, class_j_col=class_j_col,
    )

    triple_violations = violations_to_df(violations)

    onset_summary, onset_triple_df = check_onset_ultrametric_consistency(onset_matrix)

    return TransitivityReport(
        params=params,
        classified_df=classified_df,
        onset_df=onset_df,
        onset_matrix=onset_matrix,
        timebin_summary=timebin_summary,
        triple_violations=triple_violations,
        onset_summary=onset_summary,
        onset_triple_df=onset_triple_df,
    )
