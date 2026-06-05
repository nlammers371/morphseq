"""
Make Figure 5: argmax confusion/recall over time, per experiment.

The existing benchmark used 4-hpf q windows. This figure uses the full 30-48 hpf
prediction table and summarizes argmax confusion by held-out experiment and HPF bin.

Main view:
    per-class argmax recall over time, with faint lines for each held-out experiment
    and a bold pooled curve for each q source.

Companion output:
    row-normalized argmax confusion by method, held-out experiment, hpf bin, true label,
    and predicted label.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PLOT_DIR = HERE / "plots"
PREDICTIONS = HERE / "q_conformal_benchmark_image_predictions.csv"
sys.path.insert(0, str(HERE / "new_files"))

from conformal_set_plotting import (  # noqa: E402
    LABEL_DISPLAY,
    LABEL_ORDER,
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_ORDER,
    build_argmax_confusion,
    configure_matplotlib,
    save_figure,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def build_time_confusion(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = pred.copy()
    pred["hpf_bin"] = pred["_hpf_bin"].astype(float)
    pred["hpf_bin_center"] = pred["hpf_bin"] + 2.0
    confusion = build_argmax_confusion(
        pred,
        group_cols=["heldout_experiment_id", "hpf_bin", "hpf_bin_center"],
    )
    confusion["is_correct_cell"] = confusion["true_label"] == confusion["predicted_label"]

    recall = confusion[confusion["is_correct_cell"]].copy()
    recall = recall.rename(columns={"proportion": "recall"})
    pooled = build_argmax_confusion(pred, group_cols=["hpf_bin", "hpf_bin_center"])
    pooled = pooled[pooled["true_label"] == pooled["predicted_label"]].copy()
    pooled = pooled.rename(columns={"proportion": "recall"})
    pooled["heldout_experiment_id"] = "POOLED"
    return confusion, pd.concat([recall, pooled], ignore_index=True, sort=False)


def make_figure(recall: pd.DataFrame) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2), sharex=True, sharey=True, constrained_layout=True)
    axes = axes.ravel()

    for ax, true_label in zip(axes, LABEL_ORDER):
        sub = recall[recall["true_label"] == true_label]
        for method in METHOD_ORDER:
            msub = sub[(sub["method"] == method) & (sub["heldout_experiment_id"] != "POOLED")]
            for _, exp in msub.groupby("heldout_experiment_id"):
                exp = exp.sort_values("hpf_bin_center")
                ax.plot(
                    exp["hpf_bin_center"],
                    exp["recall"],
                    color=METHOD_COLORS[method],
                    alpha=0.18,
                    linewidth=1.0,
                )
            pooled = sub[(sub["method"] == method) & (sub["heldout_experiment_id"] == "POOLED")].sort_values("hpf_bin_center")
            ax.plot(
                pooled["hpf_bin_center"],
                pooled["recall"],
                color=METHOD_COLORS[method],
                marker="o",
                linewidth=2.4,
                label=METHOD_LABELS[method],
            )

        ax.set_title(LABEL_DISPLAY[true_label])
        ax.set_ylim(0, 1.05)
        ax.set_xlim(31, 47)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel("Argmax recall")
        ax.set_xlabel("HPF bin center (4-hpf q windows)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Argmax class recall over developmental time, per held-out experiment", y=1.045, fontsize=13)
    save_figure(
        fig,
        PLOT_DIR / "report_figure_05_argmax_recall_over_time.png",
        PLOT_DIR / "report_figure_05_argmax_recall_over_time.svg",
    )


def main() -> None:
    pred = pd.read_csv(PREDICTIONS)
    confusion, recall = build_time_confusion(pred)
    confusion_path = HERE / "figure_05_argmax_confusion_over_time_plot_data.csv"
    recall_path = HERE / "figure_05_argmax_recall_over_time_plot_data.csv"
    confusion.to_csv(confusion_path, index=False)
    recall.to_csv(recall_path, index=False)
    print(f"Saved {confusion_path}")
    print(f"Saved {recall_path}")
    make_figure(recall)


if __name__ == "__main__":
    main()
