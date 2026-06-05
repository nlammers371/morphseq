"""
Make Figure 2: class-specific conformal sets and KNN-neighborhood support.

This expands the Figure 1 conformal panel by asking, for each true class:
    - how often is the true label covered?
    - how often does conformal return a singleton, 2-label, or 3+-label set?
    - for mixed labels, do singleton/covered calls concentrate where raw KNN
      neighborhoods have more same-label support?

The benchmark used 4-hpf local windows when generating q, via the `_hpf_bin` column.
The density/support panels below use raw Euclidean top-15 neighbor composition from
the LOEO diagnostic, so they are explicitly a support diagnostic rather than the model
score itself.
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
NEIGHBOR_GEOMETRY = HERE / "q_diagnostic_neighbor_geometry.csv"

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
SET_COLORS = {"singleton": "#4C78A8", "two_label": "#72B7B2", "three_plus": "#F58518"}


def build_plot_data(pred: pd.DataFrame, geom: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = pred.copy()
    pred["set_kind"] = np.select(
        [pred["set_size"] == 1, pred["set_size"] == 2, pred["set_size"] >= 3],
        ["singleton", "two_label", "three_plus"],
        default="unknown",
    )
    pred["covered"] = [bool(row[f"in_set_{row['true_label']}"]) for _, row in pred.iterrows()]

    class_summary = (
        pred.groupby(["method", "true_label"], as_index=False)
        .agg(
            n=("snip_id", "size"),
            coverage=("covered", "mean"),
            singleton_rate=("set_size", lambda x: (x == 1).mean()),
            mean_set_size=("set_size", "mean"),
        )
    )

    set_kind = pred.groupby(["method", "true_label", "set_kind"], as_index=False).agg(n=("snip_id", "size"))
    set_kind["frac"] = set_kind["n"] / set_kind.groupby(["method", "true_label"])["n"].transform("sum")

    set_comp = pred.groupby(["method", "true_label", "prediction_set"], as_index=False).agg(n=("snip_id", "size"))
    set_comp["frac"] = set_comp["n"] / set_comp.groupby(["method", "true_label"])["n"].transform("sum")
    set_comp = set_comp.sort_values(["method", "true_label", "frac"], ascending=[True, True, False])
    set_comp["rank_within_class"] = set_comp.groupby(["method", "true_label"]).cumcount() + 1

    merged = pred.merge(
        geom[
            [
                "snip_id",
                "frac_true_label_neighbors_top15",
                "frac_np_neighbors_top15",
                "rank_first_true_label_neighbor",
            ]
        ],
        on="snip_id",
        how="left",
    )
    merged["same_label_top15_bin"] = pd.cut(
        merged["frac_true_label_neighbors_top15"],
        bins=[-0.001, 0.0, 0.1, 0.25, 0.5, 1.0],
        labels=["0", "0-0.10", "0.10-0.25", "0.25-0.50", "0.50-1.00"],
    )

    rows = []
    for _, row in class_summary.iterrows():
        for metric in ["coverage", "singleton_rate", "mean_set_size"]:
            rows.append(
                {
                    "table": "class_summary",
                    "method": row["method"],
                    "true_label": row["true_label"],
                    "metric": metric,
                    "value": row[metric],
                    "n": row["n"],
                }
            )
    for _, row in set_kind.iterrows():
        rows.append(
            {
                "table": "set_kind",
                "method": row["method"],
                "true_label": row["true_label"],
                "set_kind": row["set_kind"],
                "metric": "frac",
                "value": row["frac"],
                "n": row["n"],
            }
        )
    for _, row in set_comp[set_comp["rank_within_class"] <= 5].iterrows():
        rows.append(
            {
                "table": "set_composition_top5",
                "method": row["method"],
                "true_label": row["true_label"],
                "prediction_set": row["prediction_set"],
                "rank_within_class": row["rank_within_class"],
                "metric": "frac",
                "value": row["frac"],
                "n": row["n"],
            }
        )

    density_summary = (
        merged.groupby(["method", "true_label", "same_label_top15_bin"], observed=True, as_index=False)
        .agg(
            n=("snip_id", "size"),
            coverage=("covered", "mean"),
            singleton_rate=("set_size", lambda x: (x == 1).mean()),
            mean_set_size=("set_size", "mean"),
            mean_np_top15=("frac_np_neighbors_top15", "mean"),
            mean_first_same_rank=("rank_first_true_label_neighbor", "mean"),
        )
    )
    for _, row in density_summary.iterrows():
        for metric in ["coverage", "singleton_rate", "mean_set_size", "mean_np_top15", "mean_first_same_rank"]:
            rows.append(
                {
                    "table": "density_summary",
                    "method": row["method"],
                    "true_label": row["true_label"],
                    "same_label_top15_bin": row["same_label_top15_bin"],
                    "metric": metric,
                    "value": row[metric],
                    "n": row["n"],
                }
            )
    return pd.DataFrame(rows), merged


def _class_metric(ax, data: pd.DataFrame, metric: str, ylabel: str, ylim: tuple[float, float]) -> None:
    width = 0.34
    x = np.arange(len(LABEL_ORDER))
    offsets = {"knn_q": -width / 2, "multiclass_q": width / 2}
    for method in METHOD_ORDER:
        vals = []
        for label in LABEL_ORDER:
            mask = (
                (data["table"] == "class_summary")
                & (data["method"] == method)
                & (data["true_label"] == label)
                & (data["metric"] == metric)
            )
            vals.append(float(data.loc[mask, "value"].iloc[0]))
        ax.bar(x + offsets[method], vals, width=width, color=COLORS[method], edgecolor="white", linewidth=0.8, label=METHOD_LABELS[method])
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL_DISPLAY[l] for l in LABEL_ORDER], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _set_kind_panel(ax, data: pd.DataFrame, method: str) -> None:
    x = np.arange(len(LABEL_ORDER))
    bottom = np.zeros(len(LABEL_ORDER))
    for kind in ["singleton", "two_label", "three_plus"]:
        vals = []
        for label in LABEL_ORDER:
            mask = (
                (data["table"] == "set_kind")
                & (data["method"] == method)
                & (data["true_label"] == label)
                & (data["set_kind"] == kind)
            )
            vals.append(float(data.loc[mask, "value"].iloc[0]) if mask.any() else 0.0)
        ax.bar(x, vals, bottom=bottom, color=SET_COLORS[kind], edgecolor="white", linewidth=0.8, label={"singleton": "Singleton", "two_label": "2 labels", "three_plus": "3-4 labels"}[kind])
        bottom += np.asarray(vals)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL_DISPLAY[l] for l in LABEL_ORDER], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of conformal sets")
    ax.set_title(f"Set sizes for {METHOD_LABELS[method]}")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _density_panel(ax, merged: pd.DataFrame, method: str, label: str) -> None:
    sub = merged[(merged["method"] == method) & (merged["true_label"] == label)].copy()
    summary = (
        sub.groupby("same_label_top15_bin", observed=True)
        .agg(
            n=("snip_id", "size"),
            singleton_rate=("set_size", lambda x: (x == 1).mean()),
            coverage=("covered", "mean"),
            mean_set_size=("set_size", "mean"),
        )
        .reset_index()
    )
    x = np.arange(len(summary))
    ax.bar(x - 0.16, summary["singleton_rate"], width=0.32, color="#4C78A8", edgecolor="white", linewidth=0.8, label="Singleton rate")
    ax.bar(x + 0.16, summary["coverage"], width=0.32, color="#F58518", edgecolor="white", linewidth=0.8, label="Coverage")
    for i, row in summary.iterrows():
        ax.text(i, 0.03, f"n={int(row['n'])}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.axhline(0.90, color="#333333", linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["same_label_top15_bin"].astype(str), rotation=20, ha="right")
    ax.set_xlabel("Same-label fraction among top-15 Euclidean neighbors")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{METHOD_LABELS[method]} on true {LABEL_DISPLAY[label]}")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_figure(data: pd.DataFrame, merged: pd.DataFrame) -> None:
    PLOT_DIR.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9, "legend.fontsize": 8, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.0), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.ravel()
    _class_metric(ax_a, data, "coverage", "Conformal coverage", (0, 1.05))
    ax_a.axhline(0.90, color="#333333", linewidth=1.0, linestyle="--")
    ax_a.set_title("A. Coverage by true class")
    _class_metric(ax_b, data, "singleton_rate", "Singleton rate", (0, 0.35))
    ax_b.set_title("B. One-label set rate by true class")
    _set_kind_panel(ax_c, data, "knn_q")
    _set_kind_panel(ax_d, data, "multiclass_q")
    _density_panel(ax_e, merged, "multiclass_q", "Intermediate")
    _density_panel(ax_f, merged, "multiclass_q", "Low_to_High")
    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.31, 1.02))
    ax_c.legend(loc="upper right", frameon=False)
    ax_e.legend(loc="lower right", frameon=False)
    fig.suptitle("Class-specific conformal sets and KNN-neighborhood support (4-hpf q windows)", y=1.055, fontsize=13)
    png = PLOT_DIR / "report_figure_02_conformal_sets_by_class.png"
    svg = PLOT_DIR / "report_figure_02_conformal_sets_by_class.svg"
    fig.savefig(png, bbox_inches="tight", dpi=180)
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png}")
    print(f"Saved {svg}")


def main() -> None:
    pred = pd.read_csv(PREDICTIONS)
    geom = pd.read_csv(NEIGHBOR_GEOMETRY)
    data, merged = build_plot_data(pred, geom)
    out = HERE / "figure_02_conformal_sets_by_class_plot_data.csv"
    data.to_csv(out, index=False)
    print(f"Saved {out}")
    make_figure(data, merged)


if __name__ == "__main__":
    main()
