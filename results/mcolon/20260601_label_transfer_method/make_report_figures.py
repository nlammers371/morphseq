"""
Make report figures for the CEP290 label-transfer conformal benchmark.

Figure 1 shows the core two-axis comparison:
    q generator: KNN-q vs multiclass-q
    output mode: argmax vs conformal

Outputs:
    figure_01_main_benchmark_plot_data.csv
    plots/report_figure_01_main_benchmark.png
    plots/report_figure_01_main_benchmark.svg

Usage:
    /net/trapnell/vol1/home/mdcolon/software/miniconda3/envs/segmentation_grounded_sam/bin/python \
        results/mcolon/20260601_label_transfer_method/make_report_figures.py
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
GLOBAL_SUMMARY = HERE / "q_conformal_benchmark_global_summary.csv"
FOLD_SUMMARY = HERE / "q_conformal_benchmark_summary.csv"

METHOD_ORDER = ["knn_q", "multiclass_q"]
METHOD_LABELS = {
    "knn_q": "KNN-q",
    "multiclass_q": "Multiclass-q",
}
LABEL_ORDER = ["Low_to_High", "High_to_Low", "Intermediate", "Not Penetrant"]
LABEL_DISPLAY = {
    "Low_to_High": "Low-to-High",
    "High_to_Low": "High-to-Low",
    "Intermediate": "Intermediate",
    "Not Penetrant": "Not Penetrant",
}
COLORS = {
    "knn_q": "#4C78A8",
    "multiclass_q": "#F58518",
}


def _metric_row(
    row: pd.Series,
    metric: str,
    value: float,
    metric_family: str,
    label: str | None = None,
    source: str = "global",
) -> dict:
    return {
        "source": source,
        "method": row["method"],
        "q_source": row["q_source"],
        "output_type": row["output_type"],
        "heldout_experiment_id": row["heldout_experiment_id"],
        "metric": metric,
        "metric_family": metric_family,
        "label": label,
        "value": value,
    }


def build_plot_data(global_df: pd.DataFrame, fold_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, df in [("global", global_df), ("fold", fold_df)]:
        for _, row in df.iterrows():
            if row["output_type"] == "argmax":
                for metric in ["accuracy", "balanced_accuracy", "macro_f1"]:
                    if metric in row and pd.notna(row[metric]):
                        rows.append(_metric_row(row, metric, row[metric], "argmax", source=source))
                for label in ["Low_to_High", "Intermediate"]:
                    metric = f"recall[{label}]"
                    if metric in row and pd.notna(row[metric]):
                        rows.append(_metric_row(row, metric, row[metric], "rare_label_success", label=label, source=source))
                if "LtH->NP_collapse" in row and pd.notna(row["LtH->NP_collapse"]):
                    rows.append(_metric_row(row, "LtH_not_called_NP", 1.0 - row["LtH->NP_collapse"], "rare_label_success", label="Low_to_High", source=source))
                if "NP->LtH_falsecall" in row and pd.notna(row["NP->LtH_falsecall"]):
                    rows.append(_metric_row(row, "NP_not_called_LtH", 1.0 - row["NP->LtH_falsecall"], "rare_label_success", label="Not Penetrant", source=source))
                for metric in ["LtH->NP_collapse", "NP->LtH_falsecall"]:
                    if metric in row and pd.notna(row[metric]):
                        rows.append(_metric_row(row, metric, row[metric], "rare_failure", source=source))

            if row["output_type"] == "conformal":
                for metric in ["marginal_coverage", "mean_set_size", "singleton_rate"]:
                    if metric in row and pd.notna(row[metric]):
                        rows.append(_metric_row(row, metric, row[metric], "conformal", source=source))
                for label in LABEL_ORDER:
                    col = f"coverage[{label}]"
                    if col in row and pd.notna(row[col]):
                        rows.append(_metric_row(row, "coverage", row[col], "per_class_coverage", label=label, source=source))
    out = pd.DataFrame(rows)
    out["method_display"] = out["method"].map(METHOD_LABELS)
    out["label_display"] = out["label"].map(LABEL_DISPLAY)
    return out


def _global_value(plot_df: pd.DataFrame, method: str, metric: str, family: str, label: str | None = None) -> float:
    mask = (
        (plot_df["source"] == "global")
        & (plot_df["method"] == method)
        & (plot_df["metric"] == metric)
        & (plot_df["metric_family"] == family)
    )
    if label is not None:
        mask &= plot_df["label"] == label
    vals = plot_df.loc[mask, "value"].to_numpy()
    return float(vals[0]) if len(vals) else np.nan


def _fold_values(plot_df: pd.DataFrame, method: str, metric: str, family: str, label: str | None = None) -> np.ndarray:
    mask = (
        (plot_df["source"] == "fold")
        & (plot_df["method"] == method)
        & (plot_df["metric"] == metric)
        & (plot_df["metric_family"] == family)
    )
    if label is not None:
        mask &= plot_df["label"] == label
    return plot_df.loc[mask, "value"].dropna().to_numpy(dtype=float)


def _bar_with_fold_points(
    ax,
    plot_df: pd.DataFrame,
    metrics: list[str],
    metric_family: str,
    metric_labels: dict[str, str],
    ylabel: str,
    ylim: tuple[float, float] | None = (0, 1),
    target_line: float | None = None,
) -> None:
    width = 0.34
    x = np.arange(len(metrics))
    offsets = {"knn_q": -width / 2, "multiclass_q": width / 2}

    for method in METHOD_ORDER:
        vals = [_global_value(plot_df, method, metric, metric_family) for metric in metrics]
        xpos = x + offsets[method]
        ax.bar(
            xpos,
            vals,
            width=width,
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.8,
            label=METHOD_LABELS[method],
            zorder=2,
        )
        for i, metric in enumerate(metrics):
            fold_vals = _fold_values(plot_df, method, metric, metric_family)
            if len(fold_vals):
                jitter = np.linspace(-0.07, 0.07, len(fold_vals))
                ax.scatter(
                    np.full(len(fold_vals), xpos[i]) + jitter,
                    fold_vals,
                    s=18,
                    color="black",
                    alpha=0.45,
                    linewidth=0,
                    zorder=3,
                )

    if target_line is not None:
        ax.axhline(target_line, color="#333333", linewidth=1.0, linestyle="--", zorder=1)
        ax.text(len(metrics) - 0.45, target_line + 0.015, f"target {target_line:.2f}", fontsize=8, ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels([metric_labels[m] for m in metrics], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _per_class_coverage_panel(ax, plot_df: pd.DataFrame) -> None:
    width = 0.34
    x = np.arange(len(LABEL_ORDER))
    offsets = {"knn_q": -width / 2, "multiclass_q": width / 2}

    for method in METHOD_ORDER:
        vals = [_global_value(plot_df, method, "coverage", "per_class_coverage", label) for label in LABEL_ORDER]
        xpos = x + offsets[method]
        ax.bar(
            xpos,
            vals,
            width=width,
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.8,
            label=METHOD_LABELS[method],
            zorder=2,
        )
        for i, label in enumerate(LABEL_ORDER):
            fold_vals = _fold_values(plot_df, method, "coverage", "per_class_coverage", label)
            if len(fold_vals):
                jitter = np.linspace(-0.07, 0.07, len(fold_vals))
                ax.scatter(
                    np.full(len(fold_vals), xpos[i]) + jitter,
                    fold_vals,
                    s=16,
                    color="black",
                    alpha=0.42,
                    linewidth=0,
                    zorder=3,
                )

    ax.axhline(0.90, color="#333333", linewidth=1.0, linestyle="--", zorder=1)
    ax.text(len(LABEL_ORDER) - 0.45, 0.915, "target 0.90", fontsize=8, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL_DISPLAY[l] for l in LABEL_ORDER], rotation=20, ha="right")
    ax.set_ylabel("Conformal coverage")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_figure_01(plot_df: pd.DataFrame) -> None:
    PLOT_DIR.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": 180,
        "font.family": "DejaVu Sans",
    })

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    _bar_with_fold_points(
        ax_a,
        plot_df,
        metrics=["accuracy", "balanced_accuracy", "macro_f1"],
        metric_family="argmax",
        metric_labels={
            "accuracy": "Accuracy",
            "balanced_accuracy": "Balanced acc.",
            "macro_f1": "Macro F1",
        },
        ylabel="Argmax score",
    )
    ax_a.set_title("A. Hard-label performance")

    _bar_with_fold_points(
        ax_b,
        plot_df,
        metrics=["recall[Low_to_High]", "recall[Intermediate]", "LtH_not_called_NP"],
        metric_family="rare_label_success",
        metric_labels={
            "recall[Low_to_High]": "LtH recall",
            "recall[Intermediate]": "Intermediate recall",
            "LtH_not_called_NP": "LtH not called NP",
        },
        ylabel="Rare-label success rate",
    )
    ax_b.set_title("B. Rare-label hard-call success")

    _bar_with_fold_points(
        ax_c,
        plot_df,
        metrics=["marginal_coverage", "singleton_rate"],
        metric_family="conformal",
        metric_labels={
            "marginal_coverage": "Coverage",
            "singleton_rate": "Singleton rate",
        },
        ylabel="Conformal rate",
        target_line=0.90,
    )
    ax_c2 = ax_c.twinx()
    width = 0.34
    x = np.array([2.0])
    for method, offset in {"knn_q": -width / 2, "multiclass_q": width / 2}.items():
        val = _global_value(plot_df, method, "mean_set_size", "conformal")
        ax_c2.bar(
            x + offset,
            [val],
            width=width,
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.45,
            zorder=2,
        )
    ax_c.set_xlim(-0.6, 2.6)
    ax_c.set_xticks([0, 1, 2])
    ax_c.set_xticklabels(["Coverage", "Singleton rate", "Mean set size"], rotation=20, ha="right")
    ax_c2.set_ylim(0, 4)
    ax_c2.set_ylabel("Mean set size")
    ax_c2.spines["top"].set_visible(False)
    ax_c.set_title("C. Conformal behavior")

    _per_class_coverage_panel(ax_d, plot_df)
    ax_d.set_title("D. Per-class conformal coverage")

    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("KNN-q vs multiclass-q under argmax and conformal prediction", y=1.055, fontsize=13)

    out_png = PLOT_DIR / "report_figure_01_main_benchmark.png"
    out_svg = PLOT_DIR / "report_figure_01_main_benchmark.svg"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")
    print(f"Saved {out_svg}")


def main() -> None:
    global_df = pd.read_csv(GLOBAL_SUMMARY)
    fold_df = pd.read_csv(FOLD_SUMMARY)
    plot_df = build_plot_data(global_df, fold_df)
    out_data = HERE / "figure_01_main_benchmark_plot_data.csv"
    plot_df.to_csv(out_data, index=False)
    print(f"Saved {out_data}")
    make_figure_01(plot_df)


if __name__ == "__main__":
    main()
