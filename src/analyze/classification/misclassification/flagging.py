from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    try:
        from scipy.stats import false_discovery_control as _bh

        return _bh(pvals)
    except Exception:
        try:
            from statsmodels.stats.multitest import fdrcorrection

            return fdrcorrection(pvals, alpha=0.05, method="indep")[1]
        except Exception as exc:
            raise ImportError("Need SciPy>=1.11 or statsmodels for BH-FDR") from exc


def flag_consistently_misclassified(
    per_embryo_metrics: pd.DataFrame,
    *,
    q_val_threshold: float = 0.05,
    wrong_rate_z_threshold: float = 2.0,
    wrong_rate_delta_threshold: float = 0.20,
    top_confused_frac_threshold: float = 0.80,
    require_n_windows_min: int = 3,
    require_n_wrong_min: int = 3,
) -> pd.DataFrame:
    df = per_embryo_metrics.copy()

    df["too_few_windows"] = df["n_windows"] < require_n_windows_min
    df["too_few_wrong_for_confusion_test"] = df["n_wrong"] < require_n_wrong_min

    code_to_label = {
        "A": f"qval_wrong_rate<{q_val_threshold}",
        "B": f"qval_streak<{q_val_threshold}",
        "C": f"wrong_rate_z>{wrong_rate_z_threshold}",
        "D": f"wrong_rate_delta>{wrong_rate_delta_threshold}",
        "E": (
            f"top_confused_frac>{top_confused_frac_threshold}"
            f"&qval_top_confused<{q_val_threshold}"
        ),
    }

    is_flagged = []
    codes_list = []
    labels_list = []

    for _, row in df.iterrows():
        codes: list[str] = []

        if row.get("qval_wrong_rate", 1.0) < q_val_threshold:
            codes.append("A")
        if row.get("qval_streak", 1.0) < q_val_threshold:
            codes.append("B")
        if (not bool(row["too_few_windows"])) and row.get("wrong_rate_z", -1e9) > wrong_rate_z_threshold:
            codes.append("C")
        if (not bool(row["too_few_windows"])) and (
            float(row["wrong_rate"]) - float(row["expected_wrong_rate"])
        ) > wrong_rate_delta_threshold:
            codes.append("D")

        if (
            float(row.get("top_confused_frac", 0.0)) > top_confused_frac_threshold
            and float(row.get("qval_top_confused_frac", 1.0)) < q_val_threshold
            and (not bool(row["too_few_wrong_for_confusion_test"]))
            and (not bool(row.get("top_confused_test_skipped", False)))
        ):
            codes.append("E")

        is_flagged.append(len(codes) > 0)
        codes_list.append(codes)
        labels_list.append([code_to_label[c] for c in codes])

    df["is_flagged"] = is_flagged
    df["flag_reason_codes"] = codes_list
    df["flag_reason_labels"] = labels_list
    return df


def compute_confusion_enrichment(
    embryo_predictions: pd.DataFrame,
    flagged_embryos: pd.DataFrame,
) -> pd.DataFrame:
    """Compute flagged-vs-unflagged enrichment for each true->confused pair."""
    pred = embryo_predictions.copy()
    pred["is_wrong"] = (pred["pred_class"] != pred["true_class"]).astype(bool)
    wrong = pred[pred["is_wrong"]].copy()

    flagged_ids = set(flagged_embryos["embryo_id"].astype(str).tolist())
    wrong["is_flagged_embryo"] = wrong["embryo_id"].astype(str).isin(flagged_ids)

    rows: list[dict[str, Any]] = []
    for true_class, g_true in wrong.groupby("true_class"):
        total_flagged = int(g_true["is_flagged_embryo"].sum())
        total_unflagged = int((~g_true["is_flagged_embryo"]).sum())
        if (total_flagged + total_unflagged) == 0:
            continue

        for confused_as in sorted(g_true["pred_class"].astype(str).unique()):
            is_pair = g_true["pred_class"].astype(str) == confused_as
            a = int((g_true["is_flagged_embryo"] & is_pair).sum())
            b = int((g_true["is_flagged_embryo"] & ~is_pair).sum())
            c = int((~g_true["is_flagged_embryo"] & is_pair).sum())
            d = int((~g_true["is_flagged_embryo"] & ~is_pair).sum())

            observed_frac = float(a / (a + b)) if (a + b) > 0 else np.nan
            expected_frac = float(c / (c + d)) if (c + d) > 0 else np.nan

            try:
                from scipy.stats import chi2_contingency

                chi2, pval, _, _ = chi2_contingency([[a, b], [c, d]])
            except Exception:
                chi2, pval = np.nan, np.nan

            rows.append(
                {
                    "true_class": true_class,
                    "confused_as": confused_as,
                    "n_flagged": a,
                    "n_unflagged": c,
                    "expected_frac": expected_frac,
                    "observed_frac": observed_frac,
                    "chi2": chi2,
                    "pval": pval,
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty and out["pval"].notna().any():
        mask = out["pval"].notna()
        q = np.full(len(out), np.nan)
        q[mask.to_numpy()] = _bh_fdr(out.loc[mask, "pval"].to_numpy(dtype=float))
        out["qval"] = q
    else:
        out["qval"] = np.nan

    return out


def compute_time_localization(
    embryo_predictions: pd.DataFrame,
    flagged_embryos: pd.DataFrame,
    *,
    rolling_window_bins: int = 3,
    rolling_threshold: float = 0.60,
) -> pd.DataFrame:
    pred = embryo_predictions.copy()
    pred["is_wrong"] = (pred["pred_class"] != pred["true_class"]).astype(int)

    flagged_ids = set(flagged_embryos["embryo_id"].astype(str).tolist())
    work = pred[pred["embryo_id"].astype(str).isin(flagged_ids)].copy()

    rows: list[dict[str, Any]] = []
    for embryo_id, g in work.sort_values(["embryo_id", "time_bin"]).groupby("embryo_id"):
        g = g.copy()
        g["rolling_wrong_rate"] = (
            g["is_wrong"].rolling(window=rolling_window_bins, min_periods=rolling_window_bins).mean()
        )

        above = g["rolling_wrong_rate"] > rolling_threshold
        if above.any():
            onset_time_bin = int(g.loc[above, "time_bin"].iloc[0])
            offset_time_bin = int(g.loc[above, "time_bin"].iloc[-1])
            duration = int(offset_time_bin - onset_time_bin + 1)
        else:
            onset_time_bin = np.nan
            offset_time_bin = np.nan
            duration = 0

        bins = g["time_bin"].to_numpy(dtype=float)
        if len(bins) == 0 or np.isnan(onset_time_bin):
            failure_phase = "none"
        else:
            q1, q2, q3 = np.quantile(bins, [0.25, 0.5, 0.75])
            if onset_time_bin <= q1:
                failure_phase = "early"
            elif onset_time_bin <= q2:
                failure_phase = "mid"
            elif onset_time_bin <= q3:
                failure_phase = "late"
            else:
                failure_phase = "sustained"

        rows.append(
            {
                "embryo_id": str(embryo_id),
                "onset_time_bin": onset_time_bin,
                "offset_time_bin": offset_time_bin,
                "failure_duration_bins": duration,
                "failure_phase": failure_phase,
                "rolling_window_bins": int(rolling_window_bins),
                "rolling_threshold": float(rolling_threshold),
            }
        )

    return pd.DataFrame(rows)
