"""
Make Figure 3: singleton set rate over time by true label.

This checks whether the apparent conformal usefulness of multiclass-q is just a time
effect. The benchmark generated q in 4-hpf windows; this figure uses the same `_hpf_bin`
column and plots, for each true label and method:
    - singleton rate over HPF bin
    - coverage over HPF bin

Coverage is shown beside singleton rate so a high singleton rate is only interpreted as
useful when coverage remains acceptable.
"""

from __future__ import annotations

import argparse
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
LABEL_DISPLAY = {
    "Low_to_High": "Low-to-High",
    "High_to_Low": "High-to-Low",
    "Intermediate": "Intermediate",
    "Not Penetrant": "Not Penetrant",
}
COLORS = {"knn_q": "#4C78A8", "multiclass_q": "#F58518"}


def build_plot_data(pred: pd.DataFrame) -> pd.DataFrame:
    pred = pred.copy()
    pred["covered"] = [bool(row[f"in_set_{row['true_label']}"]) for _, row in pred.iterrows()]
    pred["singleton"] = pred["set_size"] == 1
    pred["hpf_bin_center"] = pred["_hpf_bin"].astype(float) + 2.0

    summary = (
        pred.groupby(["method", "true_label", "_hpf_bin", "hpf_bin_center"], as_index=False)
        .agg(
            n=("snip_id", "size"),
            singleton_rate=("singleton", "mean"),
            coverage=("covered", "mean"),
            mean_set_size=("set_size", "mean"),
        )
    )

    rows = []
    for _, row in summary.iterrows():
        for metric in ["singleton_rate", "coverage", "mean_set_size"]:
            rows.append(
                {
                    "method": row["method"],
                    "method_display": METHOD_LABELS[row["method"]],
                    "true_label": row["true_label"],
                    "label_display": LABEL_DISPLAY[row["true_label"]],
                    "hpf_bin": row["_hpf_bin"],
                    "hpf_bin_center": row["hpf_bin_center"],
                    "metric": metric,
                    "value": row[metric],
                    "n": row["n"],
                }
            )
    return pd.DataFrame(rows)


def _metric_series(plot_df: pd.DataFrame, method: str, label: str, metric: str) -> pd.DataFrame:
    return plot_df[
        (plot_df["method"] == method)
        & (plot_df["true_label"] == label)
        & (plot_df["metric"] == metric)
    ].sort_values("hpf_bin_center")


def make_figure(plot_df: pd.DataFrame, output_prefix: str) -> None:
    PLOT_DIR.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "font.family": "DejaVu Sans",
    })

    fig, axes = plt.subplots(len(LABEL_ORDER), 2, figsize=(10.8, 10.4), sharex=True, constrained_layout=True)
    for row_i, label in enumerate(LABEL_ORDER):
        ax_single = axes[row_i, 0]
        ax_cov = axes[row_i, 1]

        for method in METHOD_ORDER:
            s = _metric_series(plot_df, method, label, "singleton_rate")
            c = _metric_series(plot_df, method, label, "coverage")
            ax_single.plot(
                s["hpf_bin_center"],
                s["value"],
                marker="o",
                linewidth=1.8,
                color=COLORS[method],
                label=METHOD_LABELS[method],
            )
            ax_cov.plot(
                c["hpf_bin_center"],
                c["value"],
                marker="o",
                linewidth=1.8,
                color=COLORS[method],
                label=METHOD_LABELS[method],
            )
            for _, point in s.iterrows():
                if point["n"] < 25:
                    ax_single.text(point["hpf_bin_center"], point["value"] + 0.02, f"n={int(point['n'])}", fontsize=6, ha="center")

        ax_single.set_ylabel(LABEL_DISPLAY[label])
        ax_single.set_ylim(0, 0.55)
        ax_single.grid(axis="y", color="#DDDDDD", linewidth=0.8)
        ax_single.spines["top"].set_visible(False)
        ax_single.spines["right"].set_visible(False)

        ax_cov.axhline(0.90, color="#333333", linewidth=1.0, linestyle="--")
        ax_cov.set_ylim(0, 1.05)
        ax_cov.grid(axis="y", color="#DDDDDD", linewidth=0.8)
        ax_cov.spines["top"].set_visible(False)
        ax_cov.spines["right"].set_visible(False)

        if row_i == 0:
            ax_single.set_title("Singleton rate")
            ax_cov.set_title("Coverage")
        if row_i == len(LABEL_ORDER) - 1:
            ax_single.set_xlabel("HPF bin center (4-hpf q windows)")
            ax_cov.set_xlabel("HPF bin center (4-hpf q windows)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Singleton conformal calls over developmental time by true class", y=1.045, fontsize=13)

    png = PLOT_DIR / f"{output_prefix}.png"
    svg = PLOT_DIR / f"{output_prefix}.svg"
    fig.savefig(png, bbox_inches="tight", dpi=180)
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png}")
    print(f"Saved {svg}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, default=PREDICTIONS)
    p.add_argument("--output-prefix", default="report_figure_03_singletons_over_time")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pred = pd.read_csv(args.predictions)
    plot_df = build_plot_data(pred)
    stem = args.output_prefix
    if stem.startswith("report_"):
        stem = stem[len("report_"):]
    out = HERE / f"{stem}_plot_data.csv"
    plot_df.to_csv(out, index=False)
    print(f"Saved {out}")
    make_figure(plot_df, args.output_prefix)


if __name__ == "__main__":
    main()
