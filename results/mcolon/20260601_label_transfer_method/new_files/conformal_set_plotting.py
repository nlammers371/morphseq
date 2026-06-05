"""
Reusable plotting/data helpers for conformal label-transfer reports.

These functions operate on the image-level benchmark prediction table produced by
`run_q_conformal_benchmark.py`. They are intentionally light wrappers around tidy
tables so report scripts can share the same definitions for confusion, coverage, set
composition, and time-resolved hard-call performance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


METHOD_ORDER = ["knn_q", "multiclass_q"]
METHOD_LABELS = {"knn_q": "KNN-q", "multiclass_q": "Multiclass-q"}
LABEL_ORDER = ["Low_to_High", "High_to_Low", "Intermediate", "Not Penetrant"]
LABEL_DISPLAY = {
    "Low_to_High": "Low-to-High",
    "High_to_Low": "High-to-Low",
    "Intermediate": "Intermediate",
    "Not Penetrant": "Not Penetrant",
}
LABEL_SHORT = {
    "Low_to_High": "LtH",
    "High_to_Low": "HtL",
    "Intermediate": "Int",
    "Not Penetrant": "NP",
}
METHOD_COLORS = {"knn_q": "#4C78A8", "multiclass_q": "#F58518"}
SET_COLORS = {"singleton": "#4C78A8", "two_label": "#72B7B2", "three_plus": "#F58518"}


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "font.family": "DejaVu Sans",
    })


def set_display(prediction_set: str) -> str:
    labels = str(prediction_set).split("|") if pd.notna(prediction_set) else []
    return "{" + ", ".join(LABEL_SHORT.get(label, label) for label in labels) + "}"


def add_conformal_columns(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    out["covered"] = [bool(row[f"in_set_{row['true_label']}"]) for _, row in out.iterrows()]
    out["singleton"] = out["set_size"] == 1
    out["set_kind"] = np.select(
        [out["set_size"] == 1, out["set_size"] == 2, out["set_size"] >= 3],
        ["singleton", "two_label", "three_plus"],
        default="unknown",
    )
    return out


def build_confusion_tables(pred: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pred = add_conformal_columns(pred)
    tables = {}

    rows = []
    for method, sub in pred.groupby("method"):
        for true_label in LABEL_ORDER:
            s = sub[sub["true_label"] == true_label]
            denom = len(s)
            if denom == 0:
                continue
            for included_label in LABEL_ORDER:
                rows.append({
                    "method": method,
                    "view": "set_membership",
                    "true_label": true_label,
                    "predicted_label": included_label,
                    "count": int(s[f"in_set_{included_label}"].sum()),
                    "denominator": int(denom),
                    "proportion": float(s[f"in_set_{included_label}"].mean()),
                })
    tables["set_membership"] = pd.DataFrame(rows)

    rows = []
    for method, sub in pred.groupby("method"):
        single = sub[sub["set_size"] == 1].copy()
        single["singleton_label"] = single["prediction_set"]
        for true_label in LABEL_ORDER:
            s = single[single["true_label"] == true_label]
            denom = len(s)
            for pred_label in LABEL_ORDER:
                count = int((s["singleton_label"] == pred_label).sum()) if denom else 0
                rows.append({
                    "method": method,
                    "view": "singleton_only",
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "count": count,
                    "denominator": int(denom),
                    "proportion": float(count / denom) if denom else np.nan,
                })
    tables["singleton_only"] = pd.DataFrame(rows)

    tables["argmax"] = build_argmax_confusion(pred)
    tables["set_pattern"] = build_set_pattern_table(pred)
    return tables


def build_argmax_confusion(pred: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    group_cols = list(group_cols or [])
    rows = []
    for keys, sub in pred.groupby(group_cols, dropna=False) if group_cols else [((), pred)]:
        if not isinstance(keys, tuple):
            keys = (keys,)
        group_meta = dict(zip(group_cols, keys))
        for method, msub in sub.groupby("method"):
            for true_label in LABEL_ORDER:
                s = msub[msub["true_label"] == true_label]
                denom = len(s)
                for pred_label in LABEL_ORDER:
                    count = int((s["argmax_label"] == pred_label).sum()) if denom else 0
                    rows.append({
                        **group_meta,
                        "method": method,
                        "view": "argmax",
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "count": count,
                        "denominator": int(denom),
                        "proportion": float(count / denom) if denom else np.nan,
                    })
    return pd.DataFrame(rows)


def build_set_pattern_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, sub in pred.groupby("method"):
        for true_label in LABEL_ORDER:
            s = sub[sub["true_label"] == true_label]
            denom = len(s)
            counts = s["prediction_set"].value_counts()
            for pattern, count in counts.items():
                rows.append({
                    "method": method,
                    "view": "set_pattern",
                    "true_label": true_label,
                    "prediction_set": pattern,
                    "prediction_set_display": set_display(pattern),
                    "count": int(count),
                    "denominator": int(denom),
                    "proportion": float(count / denom) if denom else np.nan,
                })
    pattern = pd.DataFrame(rows)
    if not pattern.empty:
        pattern["rank_within_method_label"] = (
            pattern.sort_values(["method", "true_label", "proportion"], ascending=[True, True, False])
            .groupby(["method", "true_label"])
            .cumcount()
            + 1
        )
    return pattern


def matrix_from_confusion(table: pd.DataFrame, method: str) -> np.ndarray:
    sub = table[table["method"] == method]
    mat = np.full((len(LABEL_ORDER), len(LABEL_ORDER)), np.nan)
    for i, true_label in enumerate(LABEL_ORDER):
        for j, pred_label in enumerate(LABEL_ORDER):
            vals = sub[(sub["true_label"] == true_label) & (sub["predicted_label"] == pred_label)]["proportion"]
            if len(vals):
                mat[i, j] = vals.iloc[0]
    return mat


def draw_confusion_heatmap(
    ax,
    mat: np.ndarray,
    title: str,
    cmap: str = "Blues",
    vmax: float = 1.0,
    xlabel: str = "Predicted / included label",
) -> None:
    ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(LABEL_ORDER)))
    ax.set_yticks(np.arange(len(LABEL_ORDER)))
    ax.set_xticklabels([LABEL_SHORT[l] for l in LABEL_ORDER], rotation=30, ha="right")
    ax.set_yticklabels([LABEL_SHORT[l] for l in LABEL_ORDER])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("True label")


def save_figure(fig, path_png: Path, path_svg: Path, dpi: int = 180) -> None:
    path_png.parent.mkdir(exist_ok=True)
    fig.savefig(path_png, bbox_inches="tight", dpi=dpi)
    fig.savefig(path_svg, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path_png}")
    print(f"Saved {path_svg}")
