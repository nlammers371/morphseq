"""Onset semantics and onset-matrix construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SEPARATED = "separated"
NOT_SEPARATED = "not_separated"
AMBIGUOUS = "ambiguous"


@dataclass
class OnsetParams:
    """Parameter set for tri-state classification and durable onset detection."""

    p_sep: float = 0.05
    p_ns: float = 0.10
    subsequent_frac: float = 0.75
    auroc_sep: float = 0.0


TransitivityParams = OnsetParams


def classify_pair_state(
    pval: float,
    auroc: float,
    params: OnsetParams,
) -> str:
    """Classify a single (pval, auroc) observation into tri-state."""

    if pval < params.p_sep and auroc >= params.auroc_sep:
        return SEPARATED
    if pval > params.p_ns:
        return NOT_SEPARATED
    return AMBIGUOUS


def classify_pair_state_over_time(
    scores_df: pd.DataFrame,
    params: OnsetParams,
    *,
    time_col: str = "time_bin_center",
    class_i_col: str = "positive_label",
    class_j_col: str = "negative_label",
    pval_col: str = "pval",
    auroc_col: str = "auroc_obs",
) -> pd.DataFrame:
    """Assign tri-state edge classification to every (pair, time_bin) row."""

    df = scores_df.copy()
    states = []
    keys = []
    for _, row in df.iterrows():
        states.append(classify_pair_state(row[pval_col], row[auroc_col], params))
        a, b = row[class_i_col], row[class_j_col]
        keys.append(f"{min(a, b)}__{max(a, b)}")
    df["edge_state"] = states
    df["pair_key"] = keys
    return df


def compute_pair_onsets(
    classified_df: pd.DataFrame,
    params: OnsetParams,
    *,
    time_col: str = "time_bin_center",
    class_i_col: str = "positive_label",
    class_j_col: str = "negative_label",
) -> pd.DataFrame:
    """For each unique pair, compute the durable onset time."""

    rows = []
    for pk in sorted(classified_df["pair_key"].unique()):
        sub = classified_df[classified_df["pair_key"] == pk].sort_values(time_col).reset_index(drop=True)
        ci = sub[class_i_col].iloc[0]
        cj = sub[class_j_col].iloc[0]
        if ci > cj:
            ci, cj = cj, ci

        times = sub[time_col].values
        states = sub["edge_state"].values
        n_sep = int((states == SEPARATED).sum())
        first_sep = float(times[states == SEPARATED][0]) if n_sep > 0 else float("nan")

        onset = float("nan")
        for i in range(len(times)):
            if not states[i] == SEPARATED:
                continue
            remaining = states[i:]
            if len(remaining) == 0:
                continue
            frac_sep = float(np.sum(remaining == SEPARATED)) / len(remaining)
            if frac_sep >= params.subsequent_frac:
                onset = float(times[i])
                break

        rows.append(
            {
                "class_i": ci,
                "class_j": cj,
                "pair_key": pk,
                "onset_hpf": onset,
                "n_separated_bins": n_sep,
                "n_total_bins": len(times),
                "first_separated_bin": first_sep,
            }
        )

    return pd.DataFrame(rows)


def build_onset_matrix(onset_df: pd.DataFrame, all_classes: list[str]) -> pd.DataFrame:
    """Build symmetric onset matrix from a pair onset table."""

    mat = pd.DataFrame(index=all_classes, columns=all_classes, dtype=float)
    for _, row in onset_df.iterrows():
        ci, cj, t = row["class_i"], row["class_j"], row["onset_hpf"]
        mat.loc[ci, cj] = t
        mat.loc[cj, ci] = t
    return mat
