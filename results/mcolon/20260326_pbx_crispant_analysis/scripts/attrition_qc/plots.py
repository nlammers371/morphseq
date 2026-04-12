from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phenotypic_positioning.data import short_name
from phenotypic_positioning.plots import build_color_palette


REASON_COLORS = {
    "excluded_non_death_qc": "#fdb863",
    "excluded_death_involved": "#b2182b",
    "excluded_qc_only": "#fdb863",
    "excluded_dead_and_qc": "#d6604d",
    "excluded_dead_only": "#67001f",
}

FLAG_COLORS = {
    "dead_flag": "#6a3d9a",
    "dead_flag2": "#b15928",
    "sa_outlier_flag": "#1b9e77",
    "sam2_qc_flag": "#7570b3",
    "frame_flag": "#e6ab02",
    "no_yolk_flag": "#d95f02",
}

FLAG_LABELS = {
    "dead_flag": "Dead manual",
    "dead_flag2": "Dead inferred",
    "sa_outlier_flag": "Shape outlier",
    "sam2_qc_flag": "SAM2 QC",
    "frame_flag": "Frame QC",
    "no_yolk_flag": "No yolk",
}


def _make_grid(n_panels: int, *, n_cols: int = 3, figsize_scale: tuple[float, float] = (5.0, 3.6)):
    n_cols = min(n_cols, max(1, n_panels))
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_scale[0] * n_cols, figsize_scale[1] * n_rows), squeeze=False, sharex=True)
    flat = axes.flatten()
    for ax in flat[n_panels:]:
        ax.set_visible(False)
    return fig, flat


def plot_embryo_presence_over_time(summary_df: pd.DataFrame, *, figures_dir: Path, color_palette: dict[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for genotype, grp in summary_df.groupby("genotype"):
        grp = grp.sort_values("time_bin_center")
        ax.plot(
            grp["time_bin_center"],
            grp["embryos_present"],
            marker="o",
            markersize=4,
            linewidth=2,
            color=color_palette.get(genotype, "#888888"),
            label=short_name(genotype),
        )
    ax.set_title("Embryos present over time", fontweight="bold")
    ax.set_xlabel("Time (hpf)")
    ax.set_ylabel("Embryos present")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=min(5, summary_df["genotype"].nunique()), frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(figures_dir / "embryo_presence_over_time.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_included_vs_excluded_over_time(summary_df: pd.DataFrame, *, figures_dir: Path, color_palette: dict[str, str]) -> None:
    genotypes = list(summary_df["genotype"].drop_duplicates())
    fig, axes = _make_grid(len(genotypes))
    handles = None
    labels = None
    for ax, genotype in zip(axes, genotypes):
        grp = summary_df[summary_df["genotype"] == genotype].sort_values("time_bin_center")
        color = color_palette.get(genotype, "#888888")
        ax.plot(grp["time_bin_center"], grp["embryos_present"], color="#444444", linestyle="--", linewidth=1.8, label="present")
        ax.plot(grp["time_bin_center"], grp["embryos_included"], color=color, linewidth=2.0, marker="o", markersize=3.5, label="included")
        ax.plot(grp["time_bin_center"], grp["embryos_excluded"], color="#c23b22", linewidth=1.8, marker="o", markersize=3.0, label="excluded")
        ax.set_title(short_name(genotype), fontweight="bold")
        ax.set_xlabel("Time (hpf)")
        ax.set_ylabel("Embryo-bin count")
        handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False)
    fig.suptitle("Included vs excluded embryo-bins over time", fontweight="bold", y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(figures_dir / "embryo_included_vs_excluded_over_time.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_excluded_reasons_over_time(
    summary_df: pd.DataFrame,
    *,
    figures_dir: Path,
    detailed: bool = False,
) -> None:
    if detailed:
        cols = ["excluded_qc_only", "excluded_dead_and_qc", "excluded_dead_only"]
        labels = ["QC only", "death + QC", "death only"]
        out_name = "excluded_reason_detail_over_time.png"
        title = "Excluded embryo-bin reasons over time"
    else:
        cols = ["excluded_non_death_qc", "excluded_death_involved"]
        labels = ["non-death QC", "death involved"]
        out_name = "excluded_reason_over_time.png"
        title = "Excluded embryo-bin reasons over time"
    genotypes = list(summary_df["genotype"].drop_duplicates())
    fig, axes = _make_grid(len(genotypes))
    for ax, genotype in zip(axes, genotypes):
        grp = summary_df[summary_df["genotype"] == genotype].sort_values("time_bin_center")
        xs = grp["time_bin_center"].to_numpy(dtype=float)
        ys = [grp[col].to_numpy(dtype=float) for col in cols]
        ax.stackplot(xs, ys, labels=labels, colors=[REASON_COLORS[col] for col in cols], alpha=0.9)
        ax.set_title(short_name(genotype), fontweight="bold")
        ax.set_xlabel("Time (hpf)")
        ax.set_ylabel("Excluded embryo-bins")
    fig.legend(labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=len(labels), frameon=False)
    fig.suptitle(title, fontweight="bold", y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(figures_dir / out_name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_alive_only_use_embryo_rate(alive_summary_df: pd.DataFrame, *, figures_dir: Path, color_palette: dict[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for genotype, grp in alive_summary_df.groupby("genotype"):
        grp = grp.sort_values("time_bin_center")
        ax.plot(
            grp["time_bin_center"],
            100.0 * grp["alive_use_embryo_pass_rate"],
            marker="o",
            markersize=4,
            linewidth=2,
            color=color_palette.get(genotype, "#888888"),
            label=short_name(genotype),
        )
    ax.set_title("Use embryo pass rate over time among alive embryo-bins", fontweight="bold")
    ax.set_xlabel("Time (hpf)")
    ax.set_ylabel("Pass rate among alive (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=min(5, alive_summary_df["genotype"].nunique()), frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(figures_dir / "alive_only_use_embryo_rate_over_time.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_alive_only_qc_rates(alive_summary_df: pd.DataFrame, *, figures_dir: Path, color_palette: dict[str, str]) -> None:
    metrics = [
        ("alive_sam2_qc_flag_rate", "SAM2 QC"),
        ("alive_sa_outlier_flag_rate", "Shape outlier"),
        ("alive_frame_flag_rate", "Frame QC"),
        ("alive_no_yolk_flag_rate", "No yolk"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, (col, title) in zip(axes.flatten(), metrics):
        for genotype, grp in alive_summary_df.groupby("genotype"):
            grp = grp.sort_values("time_bin_center")
            ax.plot(
                grp["time_bin_center"],
                100.0 * grp[col],
                marker="o",
                markersize=3.5,
                linewidth=1.8,
                color=color_palette.get(genotype, "#888888"),
                label=short_name(genotype),
            )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Time (hpf)")
        ax.set_ylabel("Rate among alive (%)")
        ax.set_ylim(0, 100)
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=min(5, len(labels)), frameon=False)
    fig.suptitle("Canonical QC rates among alive embryo-bins", fontweight="bold", y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(figures_dir / "alive_only_canonical_qc_rates_over_time.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_included_by_experiment(summary_df: pd.DataFrame, *, figures_dir: Path, color_palette: dict[str, str]) -> None:
    experiments = list(summary_df["experiment_date"].drop_duplicates())
    fig, axes = _make_grid(len(experiments), n_cols=2, figsize_scale=(6.0, 4.2))
    for ax, experiment in zip(axes, experiments):
        sub = summary_df[summary_df["experiment_date"] == experiment]
        for genotype, grp in sub.groupby("genotype"):
            grp = grp.sort_values("time_bin_center")
            color = color_palette.get(genotype, "#888888")
            ax.plot(grp["time_bin_center"], grp["embryos_present"], linestyle="--", linewidth=1.2, color=color, alpha=0.45)
            ax.plot(grp["time_bin_center"], grp["embryos_included"], linewidth=2.0, marker="o", markersize=3.5, color=color, label=short_name(genotype))
        ax.set_title(str(experiment), fontweight="bold")
        ax.set_xlabel("Time (hpf)")
        ax.set_ylabel("Included embryo-bins")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=min(5, len(labels)), frameon=False)
    fig.suptitle("Included embryo-bins by experiment", fontweight="bold", y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(figures_dir / "embryo_included_by_experiment.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_alive_only_use_embryo_rate_by_experiment(
    alive_summary_df: pd.DataFrame,
    *,
    figures_dir: Path,
    color_palette: dict[str, str],
) -> None:
    experiments = list(alive_summary_df["experiment_date"].drop_duplicates())
    fig, axes = _make_grid(len(experiments), n_cols=2, figsize_scale=(6.0, 4.2))
    for ax, experiment in zip(axes, experiments):
        sub = alive_summary_df[alive_summary_df["experiment_date"] == experiment]
        for genotype, grp in sub.groupby("genotype"):
            grp = grp.sort_values("time_bin_center")
            ax.plot(
                grp["time_bin_center"],
                100.0 * grp["alive_use_embryo_pass_rate"],
                linewidth=2.0,
                marker="o",
                markersize=3.5,
                color=color_palette.get(genotype, "#888888"),
                label=short_name(genotype),
            )
        ax.set_title(str(experiment), fontweight="bold")
        ax.set_xlabel("Time (hpf)")
        ax.set_ylabel("Alive pass rate (%)")
        ax.set_ylim(0, 100)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=min(5, len(labels)), frameon=False)
    fig.suptitle("Alive-only use embryo pass rate by experiment", fontweight="bold", y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(figures_dir / "alive_only_use_embryo_rate_by_experiment.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_exclusionary_flag_fractions(
    summary_df: pd.DataFrame,
    *,
    figures_dir: Path,
    denominator: str,
) -> None:
    if denominator not in {"present", "excluded"}:
        raise ValueError(f"Unsupported denominator: {denominator}")
    if denominator == "present":
        suffix = "fraction_present"
        out_name = "exclusionary_flag_fraction_of_present_over_time.png"
        title = "Exclusionary flag fraction over time"
        y_label = "Fraction of embryo-bins present"
    else:
        suffix = "fraction_excluded"
        out_name = "exclusionary_flag_fraction_of_excluded_over_time.png"
        title = "Exclusionary flag fraction among excluded embryo-bins over time"
        y_label = "Fraction of excluded embryo-bins"

    flag_cols = [
        "dead_flag",
        "dead_flag2",
        "sa_outlier_flag",
        "sam2_qc_flag",
        "frame_flag",
        "no_yolk_flag",
    ]
    genotypes = list(summary_df["genotype"].drop_duplicates())
    fig, axes = _make_grid(len(genotypes))
    for ax, genotype in zip(axes, genotypes):
        grp = summary_df[summary_df["genotype"] == genotype].sort_values("time_bin_center")
        for flag in flag_cols:
            col = f"{flag}_{suffix}"
            if col not in grp.columns:
                continue
            ax.plot(
                grp["time_bin_center"],
                100.0 * grp[col].fillna(0.0),
                marker="o",
                markersize=3.0,
                linewidth=1.8,
                color=FLAG_COLORS[flag],
                label=FLAG_LABELS[flag],
            )
        ax.set_title(short_name(genotype), fontweight="bold")
        ax.set_xlabel("Time (hpf)")
        ax.set_ylabel(y_label + " (%)")
        ax.set_ylim(0, 100)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False)
    fig.suptitle(title, fontweight="bold", y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(figures_dir / out_name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_flag_focus_over_time(
    summary_df: pd.DataFrame,
    *,
    figures_dir: Path,
    denominator: str,
) -> None:
    if denominator not in {"present", "excluded"}:
        raise ValueError(f"Unsupported denominator: {denominator}")
    if denominator == "present":
        suffix = "fraction_present"
        out_name = "flag_focus_no_yolk_frame_death_of_present_over_time.png"
        title = "No yolk vs frame vs death over time"
        y_label = "Fraction of embryo-bins present"
    else:
        suffix = "fraction_excluded"
        out_name = "flag_focus_no_yolk_frame_death_of_excluded_over_time.png"
        title = "No yolk vs frame vs death among excluded embryo-bins"
        y_label = "Fraction of excluded embryo-bins"
    focus_flags = ["no_yolk_flag", "frame_flag", "dead_flag2", "dead_flag"]
    genotypes = list(summary_df["genotype"].drop_duplicates())
    fig, axes = _make_grid(len(genotypes))
    for ax, genotype in zip(axes, genotypes):
        grp = summary_df[summary_df["genotype"] == genotype].sort_values("time_bin_center")
        for flag in focus_flags:
            col = f"{flag}_{suffix}"
            if col not in grp.columns:
                continue
            ax.plot(
                grp["time_bin_center"],
                100.0 * grp[col].fillna(0.0),
                marker="o",
                markersize=3.8,
                linewidth=2.2,
                color=FLAG_COLORS[flag],
                label=FLAG_LABELS[flag],
            )
        ax.set_title(short_name(genotype), fontweight="bold")
        ax.set_xlabel("Time (hpf)")
        ax.set_ylabel(y_label + " (%)")
        ax.set_ylim(0, 100)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, frameon=False)
    fig.suptitle(title, fontweight="bold", y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(figures_dir / out_name, dpi=200, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "build_color_palette",
    "plot_alive_only_qc_rates",
    "plot_alive_only_use_embryo_rate",
    "plot_alive_only_use_embryo_rate_by_experiment",
    "plot_embryo_presence_over_time",
    "plot_exclusionary_flag_fractions",
    "plot_excluded_reasons_over_time",
    "plot_flag_focus_over_time",
    "plot_included_by_experiment",
    "plot_included_vs_excluded_over_time",
]
