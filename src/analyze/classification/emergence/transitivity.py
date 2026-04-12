"""Transitivity and ultrametric diagnostics for emergence analysis."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd

from .onset import (
    AMBIGUOUS,
    NOT_SEPARATED,
    SEPARATED,
    OnsetParams,
    TransitivityParams,
    build_onset_matrix,
    classify_pair_state,
    classify_pair_state_over_time,
    compute_pair_onsets,
)
from .types import OnsetConsistencySummary, TransitivityReport, TransitivityViolation


def _lookup(
    classified_df: pd.DataFrame,
    a: str,
    b: str,
    t: float,
    time_col: str,
    pval_col: str,
    auroc_col: str,
) -> tuple[str, float, float]:
    pk = f"{min(a, b)}__{max(a, b)}"
    mask = (classified_df["pair_key"] == pk) & np.isclose(classified_df[time_col], t, atol=2.5)
    sub = classified_df[mask]
    if sub.empty:
        return AMBIGUOUS, float("nan"), float("nan")
    row = sub.iloc[0]
    return row["edge_state"], row[auroc_col], row[pval_col]


def detect_timebin_transitivity_violations(
    classified_df: pd.DataFrame,
    params: OnsetParams,
    *,
    time_col: str = "time_bin_center",
    class_i_col: str = "positive_label",
    class_j_col: str = "negative_label",
    pval_col: str = "pval",
    auroc_col: str = "auroc_obs",
    time_bins: Iterable[float] | None = None,
) -> list[TransitivityViolation]:
    """Detect non-transitive triples at each time bin using the tri-state model."""

    if time_bins is None:
        time_bins = sorted(classified_df[time_col].unique())
    all_classes = sorted(set(classified_df[class_i_col].unique()) | set(classified_df[class_j_col].unique()))
    violations: list[TransitivityViolation] = []

    for t in time_bins:
        for a, b, c in combinations(all_classes, 3):
            s_ab, auroc_ab, pval_ab = _lookup(classified_df, a, b, t, time_col, pval_col, auroc_col)
            s_bc, auroc_bc, pval_bc = _lookup(classified_df, b, c, t, time_col, pval_col, auroc_col)
            s_ac, auroc_ac, pval_ac = _lookup(classified_df, a, c, t, time_col, pval_col, auroc_col)
            states = {s_ab, s_bc, s_ac}
            if AMBIGUOUS in states:
                continue

            if (s_ab == SEPARATED and s_bc == SEPARATED and s_ac == NOT_SEPARATED) or (
                s_ab == SEPARATED and s_ac == SEPARATED and s_bc == NOT_SEPARATED
            ) or (s_bc == SEPARATED and s_ac == SEPARATED and s_ab == NOT_SEPARATED):
                violations.append(
                    TransitivityViolation(
                        time_bin=t,
                        a=a,
                        b=b,
                        c=c,
                        state_ab=s_ab,
                        state_bc=s_bc,
                        state_ac=s_ac,
                        violation_type="sep_sep_notsep",
                        auroc_ab=auroc_ab,
                        auroc_bc=auroc_bc,
                        auroc_ac=auroc_ac,
                        pval_ab=pval_ab,
                        pval_bc=pval_bc,
                        pval_ac=pval_ac,
                    )
                )

            if (s_ab == NOT_SEPARATED and s_bc == NOT_SEPARATED and s_ac == SEPARATED) or (
                s_ab == NOT_SEPARATED and s_ac == NOT_SEPARATED and s_bc == SEPARATED
            ) or (s_bc == NOT_SEPARATED and s_ac == NOT_SEPARATED and s_ab == SEPARATED):
                violations.append(
                    TransitivityViolation(
                        time_bin=t,
                        a=a,
                        b=b,
                        c=c,
                        state_ab=s_ab,
                        state_bc=s_bc,
                        state_ac=s_ac,
                        violation_type="ns_ns_sep",
                        auroc_ab=auroc_ab,
                        auroc_bc=auroc_bc,
                        auroc_ac=auroc_ac,
                        pval_ab=pval_ab,
                        pval_bc=pval_bc,
                        pval_ac=pval_ac,
                    )
                )

    return violations


def summarize_timebin_violation_rate(
    violations: list[TransitivityViolation],
    classified_df: pd.DataFrame,
    all_time_bins: list[float],
    *,
    time_col: str = "time_bin_center",
    class_i_col: str = "positive_label",
    class_j_col: str = "negative_label",
    pval_col: str = "pval",
    auroc_col: str = "auroc_obs",
) -> pd.DataFrame:
    """Per-time-bin summary of violation counts and evaluable triple counts."""

    all_classes = sorted(set(classified_df[class_i_col].unique()) | set(classified_df[class_j_col].unique()))
    n_triples_total = len(list(combinations(all_classes, 3)))
    rows = []
    for t in all_time_bins:
        t_viols = [v for v in violations if np.isclose(v.time_bin, t, atol=0.5)]
        n_eval = 0
        for a, b, c in combinations(all_classes, 3):
            s_ab, _, _ = _lookup(classified_df, a, b, t, time_col, pval_col, auroc_col)
            s_bc, _, _ = _lookup(classified_df, b, c, t, time_col, pval_col, auroc_col)
            s_ac, _, _ = _lookup(classified_df, a, c, t, time_col, pval_col, auroc_col)
            if AMBIGUOUS not in {s_ab, s_bc, s_ac}:
                n_eval += 1

        n_v = len(t_viols)
        n_ssn = sum(1 for v in t_viols if v.violation_type == "sep_sep_notsep")
        n_nns = sum(1 for v in t_viols if v.violation_type == "ns_ns_sep")
        rate = n_v / n_eval if n_eval > 0 else float("nan")
        rows.append(
            {
                "time_bin": t,
                "n_triples_total": n_triples_total,
                "n_triples_evaluable": n_eval,
                "n_violations": n_v,
                "n_sep_sep_notsep": n_ssn,
                "n_ns_ns_sep": n_nns,
                "violation_rate": rate,
            }
        )
    return pd.DataFrame(rows)


def violations_to_df(violations: list[TransitivityViolation]) -> pd.DataFrame:
    """Convert violation list to tidy DataFrame."""

    if not violations:
        return pd.DataFrame(
            columns=[
                "time_bin", "a", "b", "c", "state_ab", "state_bc", "state_ac", "violation_type",
                "auroc_ab", "auroc_bc", "auroc_ac", "pval_ab", "pval_bc", "pval_ac",
            ]
        )
    return pd.DataFrame([v.__dict__ for v in violations])


def check_onset_ultrametric_consistency(
    onset_matrix: pd.DataFrame,
) -> tuple[OnsetConsistencySummary, pd.DataFrame]:
    """Compute ultrametric-gap diagnostics over onset triples."""

    classes = list(onset_matrix.index)
    n_total = len(list(combinations(classes, 3)))
    rows = []
    for a, b, c in combinations(classes, 3):
        t_ab = onset_matrix.loc[a, b]
        t_bc = onset_matrix.loc[b, c]
        t_ac = onset_matrix.loc[a, c]
        if any(pd.isna(v) for v in [t_ab, t_bc, t_ac]):
            continue

        pairs_and_times = sorted([("ab", float(t_ab)), ("bc", float(t_bc)), ("ac", float(t_ac))], key=lambda x: x[1])
        labels_sorted = [p[0] for p in pairs_and_times]
        tau1, tau2, tau3 = [p[1] for p in pairs_and_times]
        rows.append(
            {
                "a": a,
                "b": b,
                "c": c,
                "tau1": tau1,
                "tau2": tau2,
                "tau3": tau3,
                "pair_tau1": labels_sorted[0],
                "pair_tau2": labels_sorted[1],
                "pair_tau3": labels_sorted[2],
                "ultrametric_gap": tau3 - tau2,
            }
        )

    triple_df = pd.DataFrame(rows)
    n_eval = len(triple_df)
    if n_eval == 0:
        return (
            OnsetConsistencySummary(
                n_triples_total=n_total,
                n_triples_evaluable=0,
                n_gap_zero=0,
                frac_gap_zero=float("nan"),
                mean_gap=float("nan"),
                median_gap=float("nan"),
                max_gap=float("nan"),
            ),
            triple_df,
        )

    gaps = triple_df["ultrametric_gap"].values
    n_zero = int((gaps == 0).sum())
    return (
        OnsetConsistencySummary(
            n_triples_total=n_total,
            n_triples_evaluable=n_eval,
            n_gap_zero=n_zero,
            frac_gap_zero=n_zero / n_eval,
            mean_gap=float(np.mean(gaps)),
            median_gap=float(np.median(gaps)),
            max_gap=float(np.max(gaps)),
        ),
        triple_df,
    )


def build_transitivity_report(
    scores_df: pd.DataFrame,
    params: OnsetParams | None = None,
    *,
    time_col: str = "time_bin_center",
    class_i_col: str = "positive_label",
    class_j_col: str = "negative_label",
    pval_col: str = "pval",
    auroc_col: str = "auroc_obs",
) -> TransitivityReport:
    """Run the full transitivity pipeline and return a structured report."""

    if params is None:
        params = OnsetParams()

    all_time_bins = sorted(scores_df[time_col].unique())
    all_classes = sorted(set(scores_df[class_i_col].unique()) | set(scores_df[class_j_col].unique()))

    classified_df = classify_pair_state_over_time(
        scores_df,
        params,
        time_col=time_col,
        class_i_col=class_i_col,
        class_j_col=class_j_col,
        pval_col=pval_col,
        auroc_col=auroc_col,
    )
    onset_df = compute_pair_onsets(
        classified_df,
        params,
        time_col=time_col,
        class_i_col=class_i_col,
        class_j_col=class_j_col,
    )
    onset_matrix = build_onset_matrix(onset_df, all_classes)
    violations = detect_timebin_transitivity_violations(
        classified_df,
        params,
        time_col=time_col,
        class_i_col=class_i_col,
        class_j_col=class_j_col,
        pval_col=pval_col,
        auroc_col=auroc_col,
        time_bins=all_time_bins,
    )
    timebin_summary = summarize_timebin_violation_rate(
        violations,
        classified_df,
        all_time_bins,
        time_col=time_col,
        class_i_col=class_i_col,
        class_j_col=class_j_col,
        pval_col=pval_col,
        auroc_col=auroc_col,
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
