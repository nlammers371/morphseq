"""
Make Figure 4: conformal confusion views.

Outputs four conformal/argmax confusion summaries for each q source:
    1. balanced set-membership confusion
       rows=true label, columns=label included in conformal set.
       Row-normalized; rows may sum >1 because sets can include multiple labels.
    2. singleton-only confusion
       standard row-normalized confusion among conformal singleton calls only.
    3. argmax confusion
       standard row-normalized confusion from argmax(q).
    4. set-pattern composition
       rows=true label, columns=full conformal set pattern.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


HERE = Path(__file__).resolve().parent
PLOT_DIR = HERE / "plots"
PREDICTIONS = HERE / "q_conformal_benchmark_image_predictions.csv"

METHOD_ORDER = ["knn_q", "multiclass_q"]
METHOD_LABELS = {"knn_q": "KNN-q", "multiclass_q": "Multiclass-q"}
LABEL_ORDER = ["Low_to_High", "High_to_Low", "Intermediate", "Not Penetrant"]
LABEL_SHORT = {
    "Low_to_High": "LtH",
    "High_to_Low": "HtL",
    "Intermediate": "Int",
    "Not Penetrant": "NP",
}


def _set_display(prediction_set: str) -> str:
    labels = str(prediction_set).split("|") if pd.notna(prediction_set) else []
    return "{" + ", ".join(LABEL_SHORT.get(label, label) for label in labels) + "}"


def build_confusion_tables(pred: pd.DataFrame) -> dict[str, pd.DataFrame]:
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

    rows = []
    for method, sub in pred.groupby("method"):
        for true_label in LABEL_ORDER:
            s = sub[sub["true_label"] == true_label]
            denom = len(s)
            for pred_label in LABEL_ORDER:
                count = int((s["argmax_label"] == pred_label).sum()) if denom else 0
                rows.append({
                    "method": method,
                    "view": "argmax",
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "count": count,
                    "denominator": int(denom),
                    "proportion": float(count / denom) if denom else np.nan,
                })
    tables["argmax"] = pd.DataFrame(rows)

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
                    "prediction_set_display": _set_display(pattern),
                    "count": int(count),
                    "denominator": int(denom),
                    "proportion": float(count / denom) if denom else np.nan,
                })
    pattern = pd.DataFrame(rows)
    pattern["rank_within_method_label"] = (
        pattern.sort_values(["method", "true_label", "proportion"], ascending=[True, True, False])
        .groupby(["method", "true_label"])
        .cumcount()
        + 1
    )
    tables["set_pattern"] = pattern
    return tables


def _matrix(table: pd.DataFrame, method: str) -> np.ndarray:
    sub = table[table["method"] == method]
    mat = np.full((len(LABEL_ORDER), len(LABEL_ORDER)), np.nan)
    for i, true_label in enumerate(LABEL_ORDER):
        for j, pred_label in enumerate(LABEL_ORDER):
            vals = sub[(sub["true_label"] == true_label) & (sub["predicted_label"] == pred_label)]["proportion"]
            if len(vals):
                mat[i, j] = vals.iloc[0]
    return mat


def _heatmap(ax, mat: np.ndarray, title: str, cmap: str = "Blues", vmax: float = 1.0) -> None:
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
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
    ax.set_xlabel("Predicted / included label")
    ax.set_ylabel("True label")
    return im


def _pattern_panel(ax, pattern: pd.DataFrame, method: str) -> None:
    sub = pattern[(pattern["method"] == method) & (pattern["rank_within_method_label"] <= 4)].copy()
    top_patterns = (
        sub.groupby("prediction_set_display")["count"].sum().sort_values(ascending=False).head(8).index.tolist()
    )
    mat = np.zeros((len(LABEL_ORDER), len(top_patterns)))
    for i, true_label in enumerate(LABEL_ORDER):
        s = sub[sub["true_label"] == true_label]
        for j, pat in enumerate(top_patterns):
            vals = s[s["prediction_set_display"] == pat]["proportion"]
            mat[i, j] = vals.iloc[0] if len(vals) else 0.0
    ax.imshow(mat, cmap="Oranges", vmin=0, vmax=max(0.5, float(np.nanmax(mat))), aspect="auto")
    ax.set_xticks(np.arange(len(top_patterns)))
    ax.set_yticks(np.arange(len(LABEL_ORDER)))
    ax.set_xticklabels(top_patterns, rotation=35, ha="right")
    ax.set_yticklabels([LABEL_SHORT[l] for l in LABEL_ORDER])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] > 0.015:
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(f"Set patterns: {METHOD_LABELS[method]}")
    ax.set_xlabel("Conformal set")
    ax.set_ylabel("True label")


def make_figure(tables: dict[str, pd.DataFrame]) -> None:
    PLOT_DIR.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "font.family": "DejaVu Sans",
    })

    fig, axes = plt.subplots(4, 2, figsize=(10.8, 14.2), constrained_layout=True)
    for col, method in enumerate(METHOD_ORDER):
        _heatmap(
            axes[0, col],
            _matrix(tables["set_membership"], method),
            f"Set-membership confusion: {METHOD_LABELS[method]}",
            cmap="Blues",
            vmax=1.0,
        )
        _heatmap(
            axes[1, col],
            _matrix(tables["singleton_only"], method),
            f"Singleton-only confusion: {METHOD_LABELS[method]}",
            cmap="Greens",
            vmax=1.0,
        )
        _heatmap(
            axes[2, col],
            _matrix(tables["argmax"], method),
            f"Argmax confusion: {METHOD_LABELS[method]}",
            cmap="Purples",
            vmax=1.0,
        )
        _pattern_panel(axes[3, col], tables["set_pattern"], method)

    fig.suptitle("Conformal confusion views: row-normalized by true label", y=1.01, fontsize=13)
    png = PLOT_DIR / "report_figure_04_conformal_confusion.png"
    svg = PLOT_DIR / "report_figure_04_conformal_confusion.svg"
    fig.savefig(png, bbox_inches="tight", dpi=180)
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png}")
    print(f"Saved {svg}")


def main() -> None:
    pred = pd.read_csv(PREDICTIONS)
    tables = build_confusion_tables(pred)
    long = pd.concat(tables.values(), ignore_index=True, sort=False)
    out = HERE / "figure_04_conformal_confusion_plot_data.csv"
    long.to_csv(out, index=False)
    print(f"Saved {out}")
    make_figure(tables)


if __name__ == "__main__":
    main()
