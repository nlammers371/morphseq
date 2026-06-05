"""
Make Figure 6: argmax versus conformal-singleton F1 over time.

This answers: what if we had just used argmax?

For each q source, true class, and 4-hpf bin, compare:
    - argmax one-vs-rest F1 on all images in the bin
    - conformal singleton one-vs-rest F1 on all images in the bin

Singleton F1 is coverage-aware: non-singleton conformal sets are treated as abstentions.
For a true class c, an abstention on a true-c image contributes a false negative. This
keeps the comparison honest: singleton calls can have high precision but low F1 if the
method almost never commits.
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
PREDICTIONS = HERE / "q_conformal_benchmark_full_time_image_predictions.csv"

METHOD_ORDER = ["knn_q", "multiclass_q"]
METHOD_LABELS = {"knn_q": "KNN-q", "multiclass_q": "Multiclass-q"}
LABEL_ORDER = ["Low_to_High", "High_to_Low", "Intermediate", "Not Penetrant"]
LABEL_DISPLAY = {
    "Low_to_High": "Low-to-High",
    "High_to_Low": "High-to-Low",
    "Intermediate": "Intermediate",
    "Not Penetrant": "Not Penetrant",
}
MODE_COLORS = {"argmax": "#4C78A8", "singleton": "#F58518"}


def _prf_for_class(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    true_pos = y_true == label
    pred_pos = y_pred == label
    tp = int((true_pos & pred_pos).sum())
    fp = int((~true_pos & pred_pos).sum())
    fn = int((true_pos & ~pred_pos).sum())
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) else np.nan
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def build_plot_data(pred: pd.DataFrame) -> pd.DataFrame:
    pred = pred.copy()
    pred["hpf_bin"] = pred["_hpf_bin"].astype(float)
    pred["hpf_bin_center"] = pred["hpf_bin"] + 2.0
    pred["singleton_pred_label"] = np.where(pred["set_size"] == 1, pred["prediction_set"], "ABSTAIN")

    rows = []
    group_cols = ["method", "heldout_experiment_id", "hpf_bin", "hpf_bin_center"]
    for keys, sub in pred.groupby(group_cols, dropna=False):
        method, exp_id, hpf_bin, hpf_bin_center = keys
        y_true = sub["true_label"].astype(str).to_numpy()
        for mode, pred_col in [("argmax", "argmax_label"), ("singleton", "singleton_pred_label")]:
            y_pred = sub[pred_col].astype(str).to_numpy()
            assigned_rate = float((y_pred != "ABSTAIN").mean()) if mode == "singleton" else 1.0
            for label in LABEL_ORDER:
                metrics = _prf_for_class(y_true, y_pred, label)
                rows.append({
                    "method": method,
                    "method_display": METHOD_LABELS[method],
                    "heldout_experiment_id": exp_id,
                    "hpf_bin": hpf_bin,
                    "hpf_bin_center": hpf_bin_center,
                    "true_label": label,
                    "label_display": LABEL_DISPLAY[label],
                    "prediction_mode": mode,
                    "n_bin": int(len(sub)),
                    "n_true_label": int((y_true == label).sum()),
                    "assigned_rate": assigned_rate,
                    **metrics,
                })

    fold_df = pd.DataFrame(rows)

    pooled_rows = []
    for keys, sub in pred.groupby(["method", "hpf_bin", "hpf_bin_center"], dropna=False):
        method, hpf_bin, hpf_bin_center = keys
        y_true = sub["true_label"].astype(str).to_numpy()
        for mode, pred_col in [("argmax", "argmax_label"), ("singleton", "singleton_pred_label")]:
            y_pred = sub[pred_col].astype(str).to_numpy()
            assigned_rate = float((y_pred != "ABSTAIN").mean()) if mode == "singleton" else 1.0
            for label in LABEL_ORDER:
                metrics = _prf_for_class(y_true, y_pred, label)
                pooled_rows.append({
                    "method": method,
                    "method_display": METHOD_LABELS[method],
                    "heldout_experiment_id": "POOLED",
                    "hpf_bin": hpf_bin,
                    "hpf_bin_center": hpf_bin_center,
                    "true_label": label,
                    "label_display": LABEL_DISPLAY[label],
                    "prediction_mode": mode,
                    "n_bin": int(len(sub)),
                    "n_true_label": int((y_true == label).sum()),
                    "assigned_rate": assigned_rate,
                    **metrics,
                })
    return pd.concat([fold_df, pd.DataFrame(pooled_rows)], ignore_index=True, sort=False)


def make_figure(plot_df: pd.DataFrame) -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "font.family": "DejaVu Sans",
    })
    PLOT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(len(LABEL_ORDER), len(METHOD_ORDER), figsize=(11.6, 10.2), sharex=True, sharey=True, constrained_layout=True)

    for col, method in enumerate(METHOD_ORDER):
        for row, label in enumerate(LABEL_ORDER):
            ax = axes[row, col]
            sub = plot_df[(plot_df["method"] == method) & (plot_df["true_label"] == label)]
            for mode in ["argmax", "singleton"]:
                fold = sub[(sub["prediction_mode"] == mode) & (sub["heldout_experiment_id"] != "POOLED")]
                for _, exp in fold.groupby("heldout_experiment_id"):
                    exp = exp.sort_values("hpf_bin_center")
                    ax.plot(exp["hpf_bin_center"], exp["f1"], color=MODE_COLORS[mode], alpha=0.14, linewidth=0.9)
                pooled = sub[(sub["prediction_mode"] == mode) & (sub["heldout_experiment_id"] == "POOLED")].sort_values("hpf_bin_center")
                ax.plot(
                    pooled["hpf_bin_center"],
                    pooled["f1"],
                    color=MODE_COLORS[mode],
                    marker="o",
                    linewidth=2.2,
                    label={"argmax": "Argmax", "singleton": "Conformal singleton"}[mode],
                )

            if row == 0:
                ax.set_title(METHOD_LABELS[method])
            if col == 0:
                ax.set_ylabel(f"{LABEL_DISPLAY[label]}\nF1")
            if row == len(LABEL_ORDER) - 1:
                ax.set_xlabel("HPF bin center (4-hpf windows)")
            ax.set_ylim(0, 1.05)
            ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("If we just used argmax: per-class F1 over time versus conformal singleton calls", y=1.045, fontsize=13)
    png = PLOT_DIR / "report_figure_06_argmax_vs_singleton_f1_over_time.png"
    svg = PLOT_DIR / "report_figure_06_argmax_vs_singleton_f1_over_time.svg"
    fig.savefig(png, bbox_inches="tight", dpi=180)
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png}")
    print(f"Saved {svg}")


def main() -> None:
    pred = pd.read_csv(PREDICTIONS, low_memory=False)
    plot_df = build_plot_data(pred)
    out = HERE / "figure_06_argmax_vs_singleton_f1_over_time_plot_data.csv"
    plot_df.to_csv(out, index=False)
    print(f"Saved {out}")
    make_figure(plot_df)


if __name__ == "__main__":
    main()
