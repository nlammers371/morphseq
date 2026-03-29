from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from common import BUILD06_DIR, REPO_ROOT, resolve_embedding_roots
EXPERIMENT_IDS = ["20260304", "20260306"]
sys.path.insert(0, str(REPO_ROOT / "src"))

from analyze.viz.plotting.feature_over_time import plot_feature_over_time
from analyze.viz.plotting.proportions import plot_proportions
from analyze.viz.plotting.faceting_engine import FacetSpec


RESULTS_DIR, FIGURES_DIR = resolve_embedding_roots()

DEFAULT_GROUP1 = "pbx4_crispant"
DEFAULT_GROUP2 = "pbx1b_pbx4_crispant"
DEFAULT_REFERENCE = "inj_ctrl"
DEFAULT_PREDICTIONS = RESULTS_DIR / "embryo_predictions_pbx4_crispant_vs_pbx1b_pbx4_crispant.csv"

METRIC_LABELS = {
    "baseline_deviation_normalized": "Curvature",
    "total_length_um": "Length (um)",
    "surface_area_um": "Surface area (um^2)",
}
WINDOW_LABELS = {
    "full": "Full",
    "pre74": "<74 hpf",
    "post74": ">=74 hpf",
}
PHENOTYPE_COLORS = {
    "inj_ctrl": "#E0A100",
}
TRUE_LABEL_LINESTYLES = {
    "pbx4_crispant": "-",
    "pbx1b_pbx4_crispant": "--",
}
EXPERIMENT_LINESTYLES = {
    "20260304": "-",
    "20260306": "--",
}
EXPERIMENT_OFFSETS = {
    "20260304": -0.12,
    "20260306": 0.12,
}


def _normalize_genotype(genotype: str) -> str:
    g = str(genotype).strip().lower().replace(" ", "_")
    while "__" in g:
        g = g.replace("__", "_")
    if g in {
        "ab_inj_ctrl",
        "wik-ab_inj_ctrl",
        "wik-ab_ctrl_inj",
        "wik_ab_inj_ctrl",
        "wik_ab_ctrl_inj",
    }:
        return "inj_ctrl"
    return g.replace("wik-ab", "wik_ab")


def _pretty_label(label: str) -> str:
    return str(label).replace("_", " ")


def _display_name(label: str) -> str:
    norm = str(label).strip().lower()
    if norm == "pbx4_crispant":
        return "pbx4 crispant"
    if norm == "pbx1b_pbx4_crispant":
        return "pbx1b+4 crispant"
    if norm == "inj_ctrl":
        return "inj_ctrl"
    return _pretty_label(label)


def _short_name(label: str) -> str:
    norm = str(label).strip().lower()
    if norm == "pbx4_crispant":
        return "pbx4"
    if norm == "pbx1b_pbx4_crispant":
        return "pbx1b+4"
    if norm == "inj_ctrl":
        return "inj_ctrl"
    return _pretty_label(label)


def _safe_name(group1: str, group2: str) -> str:
    return f"{group1}_vs_{group2}_bidirectional"


def _rank_group_color(rank_group: str, group1: str, group2: str) -> str:
    color_lookup = {
        "inj_ctrl": PHENOTYPE_COLORS["inj_ctrl"],
        f"{group1}_like_true_{group1}": "#D94E4E",
        f"{group2}_like_true_{group1}": "#F2A7B5",
        f"{group1}_like_true_{group2}": "#A1D99B",
        f"{group2}_like_true_{group2}": "#238B45",
    }
    return color_lookup.get(rank_group, "#666666")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bidirectional pairwise PBX morphology follow-up.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--group1", default=DEFAULT_GROUP1)
    parser.add_argument("--group2", default=DEFAULT_GROUP2)
    parser.add_argument("--reference-label", default=DEFAULT_REFERENCE)
    parser.add_argument("--bin-width", type=float, default=2.0)
    parser.add_argument("--late-threshold-hpf", type=float, default=74.0)
    parser.add_argument("--min-time-bins", type=int, default=8)
    return parser.parse_args()


def load_predictions(path: Path, group1: str, group2: str, late_threshold_hpf: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"embryo_id", "true_label", "time_bin_center", "signed_margin", "is_correct"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required prediction columns: {missing}")

    df = df[df["true_label"].isin([group1, group2])].copy()
    if df.empty:
        raise ValueError(f"No rows found for requested labels in {path}")

    df["window"] = np.where(df["time_bin_center"] < late_threshold_hpf, "pre74", "post74")
    return df


def _subclass_for_row(true_label: str, mean_signed_margin: float, group1: str, group2: str) -> tuple[str, str, str]:
    is_self_like = float(mean_signed_margin) >= 0.0
    if true_label == group1:
        if is_self_like:
            return (
                f"{group1}_like_true_{group1}",
                f"{_short_name(group1)}-like / true {_short_name(group1)}",
                "pbx4_like",
            )
        return (
            f"{group2}_like_true_{group1}",
            f"{_short_name(group2)}-like / true {_short_name(group1)}",
            "pbx1b_pbx4_like",
        )
    if is_self_like:
        return (
            f"{group2}_like_true_{group2}",
            f"{_short_name(group2)}-like / true {_short_name(group2)}",
            "pbx1b_pbx4_like",
        )
    return (
        f"{group1}_like_true_{group2}",
        f"{_short_name(group1)}-like / true {_short_name(group2)}",
        "pbx4_like",
    )


def rank_pair_embryos(pred_df: pd.DataFrame, group1: str, group2: str, min_time_bins: int) -> tuple[pd.DataFrame, float]:
    ranked = (
        pred_df.groupby(["embryo_id", "true_label"], as_index=False)
        .agg(
            mean_signed_margin=("signed_margin", "mean"),
            frac_correct=("is_correct", "mean"),
            n_time_bins=("time_bin_center", "nunique"),
            first_hpf=("time_bin_center", "min"),
            last_hpf=("time_bin_center", "max"),
        )
        .sort_values(["true_label", "mean_signed_margin"])
        .reset_index(drop=True)
    )

    pre = (
        pred_df[pred_df["window"] == "pre74"]
        .groupby(["embryo_id", "true_label"])["signed_margin"]
        .mean()
        .rename("mean_signed_margin_pre74")
        .reset_index()
    )
    post = (
        pred_df[pred_df["window"] == "post74"]
        .groupby(["embryo_id", "true_label"])["signed_margin"]
        .mean()
        .rename("mean_signed_margin_post74")
        .reset_index()
    )
    ranked = ranked.merge(pre, on=["embryo_id", "true_label"], how="left").merge(post, on=["embryo_id", "true_label"], how="left")
    ranked["eligible_for_grouping"] = ranked["n_time_bins"] >= min_time_bins

    subclass_rows = ranked.apply(
        lambda row: _subclass_for_row(str(row["true_label"]), float(row["mean_signed_margin"]), group1, group2),
        axis=1,
        result_type="expand",
    )
    subclass_rows.columns = ["rank_group", "plot_group", "phenotype_family"]
    ranked = pd.concat([ranked, subclass_rows], axis=1)
    return ranked, 0.0


def load_morphology(bin_width: float, genotype_labels: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for experiment_id in EXPERIMENT_IDS:
        path = BUILD06_DIR / f"df03_final_output_with_latents_{experiment_id}.csv"
        usecols = [
            "embryo_id",
            "genotype",
            "predicted_stage_hpf",
            "experiment_id",
            "baseline_deviation_normalized",
            "total_length_um",
            "surface_area_um",
            "use_embryo_flag",
        ]
        frames.append(pd.read_csv(path, usecols=usecols, low_memory=False))

    df = pd.concat(frames, ignore_index=True)
    if "use_embryo_flag" in df.columns:
        df = df[df["use_embryo_flag"] == True].copy()
    df = df[df["embryo_id"].notna()].copy()
    df["genotype"] = df["genotype"].fillna("unknown").map(_normalize_genotype)
    df = df[df["genotype"].isin(genotype_labels)].copy()
    df["time_bin"] = (np.floor(df["predicted_stage_hpf"] / bin_width) * bin_width).astype(int)
    df["time_bin_center"] = df["time_bin"] + bin_width / 2.0
    return df


def attach_experiment_id(ranked_df: pd.DataFrame, morph_df: pd.DataFrame) -> pd.DataFrame:
    exp_map = (
        morph_df[["embryo_id", "experiment_id"]]
        .drop_duplicates()
        .groupby("embryo_id")["experiment_id"]
        .agg(lambda s: ",".join(sorted({str(v) for v in s if pd.notna(v)})))
        .rename("experiment_id")
        .reset_index()
    )
    return ranked_df.merge(exp_map, on="embryo_id", how="left")


def summarize_morphology(
    morph_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
    reference_label: str,
    late_threshold_hpf: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_group_map = ranked_df.loc[:, ["embryo_id", "rank_group", "plot_group", "phenotype_family", "true_label"]].copy()
    ref_map = (
        morph_df.loc[morph_df["genotype"] == reference_label, ["embryo_id"]]
        .drop_duplicates()
        .assign(
            rank_group=reference_label,
            plot_group=reference_label,
            phenotype_family=reference_label,
            true_label=reference_label,
        )
    )
    group_map = pd.concat([pair_group_map, ref_map], ignore_index=True)

    merged = morph_df.merge(group_map, on="embryo_id", how="inner")
    if merged.empty:
        raise ValueError("No morphology rows matched the requested embryo groups.")

    embryo_time = (
        merged.groupby(
            ["embryo_id", "experiment_id", "rank_group", "plot_group", "phenotype_family", "true_label", "time_bin", "time_bin_center"],
            as_index=False,
        )
        .agg(
            baseline_deviation_normalized=("baseline_deviation_normalized", "mean"),
            total_length_um=("total_length_um", "mean"),
            surface_area_um=("surface_area_um", "mean"),
        )
    )

    trajectory_rows: list[pd.DataFrame] = []
    for metric in METRIC_LABELS:
        part = (
            embryo_time.groupby(["rank_group", "plot_group", "phenotype_family", "true_label", "time_bin", "time_bin_center"], as_index=False)
            .agg(
                mean=(metric, "mean"),
                sd=(metric, "std"),
                n_embryos=(metric, lambda s: int(s.notna().sum())),
            )
        )
        part["sem"] = part["sd"] / np.sqrt(part["n_embryos"].clip(lower=1))
        part["metric"] = metric
        trajectory_rows.append(part)
    trajectory_df = pd.concat(trajectory_rows, ignore_index=True)

    per_embryo_rows: list[pd.DataFrame] = []
    for metric in METRIC_LABELS:
        for window_name, mask in {
            "full": np.full(len(embryo_time), True, dtype=bool),
            "pre74": embryo_time["time_bin_center"] < late_threshold_hpf,
            "post74": embryo_time["time_bin_center"] >= late_threshold_hpf,
        }.items():
            part = embryo_time.loc[
                mask,
                ["embryo_id", "experiment_id", "rank_group", "plot_group", "phenotype_family", "true_label", metric],
            ].copy()
            if part.empty:
                continue
            part = (
                part.groupby(
                    ["embryo_id", "experiment_id", "rank_group", "plot_group", "phenotype_family", "true_label"],
                    as_index=False,
                )[metric]
                .mean()
                .rename(columns={metric: "metric_value"})
            )
            part["metric"] = metric
            part["window"] = window_name
            per_embryo_rows.append(part)
    per_embryo_summary = pd.concat(per_embryo_rows, ignore_index=True)

    group_summary = (
        per_embryo_summary.groupby(["metric", "window", "rank_group", "plot_group", "phenotype_family", "true_label"], as_index=False)
        .agg(
            n_embryos=("metric_value", "count"),
            mean=("metric_value", "mean"),
            median=("metric_value", "median"),
            sd=("metric_value", "std"),
        )
    )

    return trajectory_df, per_embryo_summary, group_summary, embryo_time


def summarize_experiment_counts(ranked_df: pd.DataFrame) -> pd.DataFrame:
    return (
        ranked_df.groupby(["rank_group", "plot_group", "true_label", "experiment_id"], as_index=False)
        .agg(n_embryos=("embryo_id", "nunique"))
        .sort_values(["true_label", "rank_group", "experiment_id"])
        .reset_index(drop=True)
    )


def summarize_reference_counts(morph_df: pd.DataFrame, reference_label: str) -> pd.DataFrame:
    return (
        morph_df[morph_df["genotype"] == reference_label][["embryo_id", "experiment_id"]]
        .drop_duplicates()
        .assign(rank_group=reference_label, plot_group=reference_label, true_label=reference_label)
        .groupby(["rank_group", "plot_group", "true_label", "experiment_id"], as_index=False)
        .agg(n_embryos=("embryo_id", "nunique"))
        .sort_values(["rank_group", "experiment_id"])
        .reset_index(drop=True)
    )


def build_embryo_like_labels(ranked_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "embryo_id",
        "true_label",
        "experiment_id",
        "rank_group",
        "plot_group",
        "phenotype_family",
        "mean_signed_margin",
        "mean_signed_margin_pre74",
        "mean_signed_margin_post74",
        "frac_correct",
        "n_time_bins",
        "eligible_for_grouping",
    ]
    return ranked_df.loc[:, cols].sort_values(["true_label", "mean_signed_margin"]).reset_index(drop=True)


def _prepare_proportion_df(labels_df: pd.DataFrame, group1: str, group2: str) -> pd.DataFrame:
    plot_group_order = [
        f"{_short_name(group1)}-like / true {_short_name(group1)}",
        f"{_short_name(group2)}-like / true {_short_name(group1)}",
        f"{_short_name(group1)}-like / true {_short_name(group2)}",
        f"{_short_name(group2)}-like / true {_short_name(group2)}",
    ]
    phenotype_family_order = [f"{_short_name(group1)}-like", f"{_short_name(group2)}-like"]
    true_label_order = [_display_name(group1), _display_name(group2)]

    pair_only = labels_df[labels_df["true_label"].isin([group1, group2])].copy()
    pair_only["experiment_id"] = pair_only["experiment_id"].astype(str)
    pair_only["true_label_display"] = pair_only["true_label"].map(
        {
            group1: _display_name(group1),
            group2: _display_name(group2),
        }
    )
    pair_only["phenotype_family_display"] = np.where(
        pair_only["phenotype_family"] == "pbx4_like",
        f"{_short_name(group1)}-like",
        f"{_short_name(group2)}-like",
    )

    overall = pair_only.copy()
    overall["experiment_panel"] = "all"
    by_experiment = pair_only.copy()
    by_experiment["experiment_panel"] = by_experiment["experiment_id"]
    plot_df = pd.concat([overall, by_experiment], ignore_index=True)

    plot_df["plot_group"] = pd.Categorical(plot_df["plot_group"], categories=plot_group_order, ordered=True)
    plot_df["phenotype_family_display"] = pd.Categorical(
        plot_df["phenotype_family_display"],
        categories=phenotype_family_order,
        ordered=True,
    )
    plot_df["true_label_display"] = pd.Categorical(
        plot_df["true_label_display"],
        categories=true_label_order,
        ordered=True,
    )
    plot_df["experiment_panel"] = pd.Categorical(
        plot_df["experiment_panel"],
        categories=["all", *EXPERIMENT_IDS],
        ordered=True,
    )
    return plot_df


def plot_subclass_proportions(labels_df: pd.DataFrame, group1: str, group2: str, path: Path) -> None:
    plot_df = _prepare_proportion_df(labels_df, group1, group2)
    color_order = [
        f"{_short_name(group1)}-like / true {_short_name(group1)}",
        f"{_short_name(group2)}-like / true {_short_name(group1)}",
        f"{_short_name(group1)}-like / true {_short_name(group2)}",
        f"{_short_name(group2)}-like / true {_short_name(group2)}",
    ]
    color_palette = {
        label: _rank_group_color(rank_group, group1, group2)
        for label, rank_group in {
            f"{_short_name(group1)}-like / true {_short_name(group1)}": f"{group1}_like_true_{group1}",
            f"{_short_name(group2)}-like / true {_short_name(group1)}": f"{group2}_like_true_{group1}",
            f"{_short_name(group1)}-like / true {_short_name(group2)}": f"{group1}_like_true_{group2}",
            f"{_short_name(group2)}-like / true {_short_name(group2)}": f"{group2}_like_true_{group2}",
        }.items()
    }
    fig = plot_proportions(
        plot_df,
        color_by_grouping="plot_group",
        row_by="experiment_panel",
        col_by="true_label_display",
        count_by="embryo_id",
        facet_order={
            "row_by": ["all", *EXPERIMENT_IDS],
            "col_by": [_display_name(group1), _display_name(group2)],
        },
        color_order=color_order,
        color_palette=color_palette,
        normalize=True,
        bar_mode="stacked",
        title=f"{_short_name(group1)} vs {_short_name(group2)} subclass proportions",
        show_counts=True,
        legend_loc="outside",
        output_path=path,
    )
    plt.close(fig)


def plot_true_label_overlap_proportions(labels_df: pd.DataFrame, group1: str, group2: str, path: Path) -> None:
    plot_df = _prepare_proportion_df(labels_df, group1, group2)
    true_label_colors = {
        _display_name(group1): "#D94E4E",
        _display_name(group2): "#238B45",
    }
    fig = plot_proportions(
        plot_df,
        color_by_grouping="true_label_display",
        row_by="experiment_panel",
        col_by="phenotype_family_display",
        count_by="embryo_id",
        facet_order={
            "row_by": ["all", *EXPERIMENT_IDS],
            "col_by": [f"{_short_name(group1)}-like", f"{_short_name(group2)}-like"],
        },
        color_order=[_display_name(group1), _display_name(group2)],
        color_palette=true_label_colors,
        normalize=True,
        bar_mode="stacked",
        title=f"{_short_name(group1)} vs {_short_name(group2)} true-label overlap proportions",
        show_counts=True,
        legend_loc="outside",
        output_path=path,
    )
    plt.close(fig)


def plot_ranking(ranked_df: pd.DataFrame, split_threshold: float, group1: str, group2: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, true_label in zip(axes, [group1, group2]):
        part = ranked_df[ranked_df["true_label"] == true_label].sort_values("mean_signed_margin").reset_index(drop=True)
        ax.axhline(split_threshold, color="#666666", lw=1.6, ls="--", alpha=0.8)

        for phenotype_family in ["pbx4_like", "pbx1b_pbx4_like"]:
            phen = part[part["phenotype_family"] == phenotype_family]
            if phen.empty:
                continue
            eligible = phen[phen["eligible_for_grouping"]]
            ineligible = phen[~phen["eligible_for_grouping"]]
            if not eligible.empty:
                color = _rank_group_color(str(eligible["rank_group"].iloc[0]), group1, group2)
                ax.scatter(
                    eligible.index,
                    eligible["mean_signed_margin"],
                    s=62,
                    color=color,
                    edgecolors="#333333",
                    linewidths=0.8,
                    alpha=0.95,
                    label=eligible["plot_group"].iloc[0],
                )
            if not ineligible.empty:
                color = _rank_group_color(str(ineligible["rank_group"].iloc[0]), group1, group2)
                ax.scatter(
                    ineligible.index,
                    ineligible["mean_signed_margin"],
                    s=62,
                    color=color,
                    edgecolors="#333333",
                    linewidths=0.8,
                    alpha=0.35,
                    label=ineligible["plot_group"].iloc[0],
                )

        ax.set_title(f"{_display_name(true_label)} (n={len(part)})", fontsize=13, fontweight="bold")
        ax.set_xlabel("Embryos (sorted)")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Mean signed margin")
    handles = []
    labels = []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    seen: set[str] = set()
    keep_handles = []
    keep_labels = []
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        keep_handles.append(handle)
        keep_labels.append(label)
    fig.legend(keep_handles, keep_labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
    fig.suptitle(
        f"{_short_name(group1)} vs {_short_name(group2)} ranked by mean signed margin",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories(
    trajectory_df: pd.DataFrame,
    embryo_time_df: pd.DataFrame,
    late_threshold_hpf: float,
    group1: str,
    group2: str,
    path: Path,
) -> None:
    plot_order = [
        "inj_ctrl",
        f"{group1}_like_true_{group1}",
        f"{group2}_like_true_{group1}",
        f"{group1}_like_true_{group2}",
        f"{group2}_like_true_{group2}",
    ]
    plot_group_lookup = {
        "inj_ctrl": "inj_ctrl",
        f"{group1}_like_true_{group1}": f"{_short_name(group1)}-like / true {_short_name(group1)}",
        f"{group2}_like_true_{group1}": f"{_short_name(group2)}-like / true {_short_name(group1)}",
        f"{group1}_like_true_{group2}": f"{_short_name(group1)}-like / true {_short_name(group2)}",
        f"{group2}_like_true_{group2}": f"{_short_name(group2)}-like / true {_short_name(group2)}",
    }
    color_lookup = {
        plot_group_lookup[rank_group]: _rank_group_color(rank_group, group1, group2)
        for rank_group in plot_order
    }
    features_over_time_df = embryo_time_df.copy()
    features_over_time_df["plot_group"] = features_over_time_df["rank_group"].map(plot_group_lookup)

    fig = plot_feature_over_time(
        features_over_time_df,
        features=list(METRIC_LABELS.keys()),
        time_col="time_bin_center",
        id_col="embryo_id",
        color_by="plot_group",
        color_lookup=color_lookup,
        backend="matplotlib",
        title=f"{_short_name(group1)} vs {_short_name(group2)} morphology features over time",
        legend_loc="outside",
    )
    for ax in fig.axes:
        ax.axvline(late_threshold_hpf, color="#777777", lw=1.0, ls="--", alpha=0.75)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_like_family_columns(
    embryo_time_df: pd.DataFrame,
    late_threshold_hpf: float,
    group1: str,
    group2: str,
    path: Path,
) -> None:
    pair_only = embryo_time_df[embryo_time_df["rank_group"] != "inj_ctrl"].copy()
    pair_only["phenotype_family_display"] = np.where(
        pair_only["phenotype_family"] == "pbx4_like",
        f"{_short_name(group1)}-like",
        f"{_short_name(group2)}-like",
    )
    color_lookup = {
        f"{_short_name(group1)}-like / true {_short_name(group1)}": _rank_group_color(f"{group1}_like_true_{group1}", group1, group2),
        f"{_short_name(group2)}-like / true {_short_name(group1)}": _rank_group_color(f"{group2}_like_true_{group1}", group1, group2),
        f"{_short_name(group1)}-like / true {_short_name(group2)}": _rank_group_color(f"{group1}_like_true_{group2}", group1, group2),
        f"{_short_name(group2)}-like / true {_short_name(group2)}": _rank_group_color(f"{group2}_like_true_{group2}", group1, group2),
    }
    fig = plot_feature_over_time(
        pair_only,
        features=list(METRIC_LABELS.keys()),
        time_col="time_bin_center",
        id_col="embryo_id",
        color_by="plot_group",
        color_lookup=color_lookup,
        facet_col="phenotype_family_display",
        layout=FacetSpec(col_order=[f"{_short_name(group1)}-like", f"{_short_name(group2)}-like"]),
        backend="matplotlib",
        title=f"{_short_name(group1)} vs {_short_name(group2)} split by *-like family",
        legend_loc="outside",
    )
    for ax in fig.axes:
        ax.axvline(late_threshold_hpf, color="#777777", lw=1.0, ls="--", alpha=0.75)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_total_length_all_groups(
    embryo_time_df: pd.DataFrame,
    late_threshold_hpf: float,
    group1: str,
    group2: str,
    path: Path,
) -> None:
    plot_order = [
        "inj_ctrl",
        f"{group1}_like_true_{group1}",
        f"{group2}_like_true_{group1}",
        f"{group1}_like_true_{group2}",
        f"{group2}_like_true_{group2}",
    ]
    plot_group_lookup = {
        "inj_ctrl": "inj_ctrl",
        f"{group1}_like_true_{group1}": f"{_short_name(group1)}-like / true {_short_name(group1)}",
        f"{group2}_like_true_{group1}": f"{_short_name(group2)}-like / true {_short_name(group1)}",
        f"{group1}_like_true_{group2}": f"{_short_name(group1)}-like / true {_short_name(group2)}",
        f"{group2}_like_true_{group2}": f"{_short_name(group2)}-like / true {_short_name(group2)}",
    }
    color_lookup = {
        plot_group_lookup[rank_group]: _rank_group_color(rank_group, group1, group2)
        for rank_group in plot_order
    }
    features_df = embryo_time_df.copy()
    features_df["plot_group"] = features_df["rank_group"].map(plot_group_lookup)
    fig = plot_feature_over_time(
        features_df,
        features=["total_length_um"],
        time_col="time_bin_center",
        id_col="embryo_id",
        color_by="plot_group",
        color_lookup=color_lookup,
        backend="matplotlib",
        title=f"{_short_name(group1)} vs {_short_name(group2)} total length over time",
        legend_loc="outside",
    )
    for ax in fig.axes:
        ax.axvline(late_threshold_hpf, color="#777777", lw=1.0, ls="--", alpha=0.75)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_true_label_overlap(
    embryo_time_df: pd.DataFrame,
    late_threshold_hpf: float,
    group1: str,
    group2: str,
    path: Path,
) -> None:
    pair_only = embryo_time_df[embryo_time_df["rank_group"] != "inj_ctrl"].copy()
    pair_only["phenotype_family_display"] = np.where(
        pair_only["phenotype_family"] == "pbx4_like",
        f"{_short_name(group1)}-like",
        f"{_short_name(group2)}-like",
    )
    pair_only["true_label_display"] = pair_only["true_label"].map(
        {
            group1: _display_name(group1),
            group2: _display_name(group2),
        }
    )
    true_label_colors = {
        _display_name(group1): "#D94E4E",
        _display_name(group2): "#238B45",
    }
    fig = plot_feature_over_time(
        pair_only,
        features=list(METRIC_LABELS.keys()),
        time_col="time_bin_center",
        id_col="embryo_id",
        color_by="true_label_display",
        color_lookup=true_label_colors,
        facet_col="phenotype_family_display",
        layout=FacetSpec(col_order=[f"{_short_name(group1)}-like", f"{_short_name(group2)}-like"]),
        backend="matplotlib",
        title=f"{_short_name(group1)} vs {_short_name(group2)} true-label overlap within *-like families",
        legend_loc="outside",
    )
    for ax in fig.axes:
        ax.axvline(late_threshold_hpf, color="#777777", lw=1.0, ls="--", alpha=0.75)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_summary(
    per_embryo_summary: pd.DataFrame,
    group1: str,
    group2: str,
    path: Path,
) -> None:
    per_embryo_summary = per_embryo_summary.copy()
    per_embryo_summary["experiment_id"] = per_embryo_summary["experiment_id"].astype(str)
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=False)

    rank_group_order = [
        "inj_ctrl",
        f"{group1}_like_true_{group1}",
        f"{group2}_like_true_{group1}",
        f"{group1}_like_true_{group2}",
        f"{group2}_like_true_{group2}",
    ]
    xtick_positions = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]
    xtick_labels = [
        "inj_ctrl",
        f"{_short_name(group1)}-like\ntrue {_short_name(group1)}",
        f"{_short_name(group2)}-like\ntrue {_short_name(group1)}",
        f"{_short_name(group1)}-like\ntrue {_short_name(group2)}",
        f"{_short_name(group2)}-like\ntrue {_short_name(group2)}",
    ] * 2

    for ax, metric in zip(axes, METRIC_LABELS):
        metric_panel = per_embryo_summary[per_embryo_summary["metric"] == metric].copy()
        for positions, window in [([1, 2, 3, 4, 5], "pre74"), ([7, 8, 9, 10, 11], "post74")]:
            panel = metric_panel[metric_panel["window"] == window]
            for xpos, rank_group in zip(positions, rank_group_order):
                group_panel = panel[panel["rank_group"] == rank_group]
                if group_panel.empty:
                    continue

                color = _rank_group_color(rank_group, group1, group2)
                for experiment_id in EXPERIMENT_IDS:
                    exp_part = group_panel[group_panel["experiment_id"] == experiment_id]["metric_value"].dropna().to_numpy()
                    if len(exp_part) == 0:
                        continue
                    box = ax.boxplot(
                        [exp_part],
                        positions=[xpos + EXPERIMENT_OFFSETS[experiment_id]],
                        widths=0.22,
                        patch_artist=True,
                        boxprops=dict(facecolor=color, alpha=0.22, edgecolor=color, linewidth=1.4),
                        medianprops=dict(color=color, linewidth=1.6),
                        whiskerprops=dict(color=color, linewidth=1.2),
                        capprops=dict(color=color, linewidth=1.2),
                        flierprops=dict(
                            marker="o",
                            markerfacecolor="white",
                            markeredgecolor=color,
                            markersize=4.5,
                            alpha=0.9,
                        ),
                    )
                    for artist in box["boxes"]:
                        artist.set_linestyle(EXPERIMENT_LINESTYLES[experiment_id])
                    for artist in box["whiskers"] + box["caps"] + box["medians"]:
                        artist.set_linestyle(EXPERIMENT_LINESTYLES[experiment_id])
                    if experiment_id == "20260306":
                        for artist in box["whiskers"] + box["caps"] + box["medians"]:
                            artist.set_dashes((3, 2))

        ax.axvline(6.0, color="#bbbbbb", lw=1.0, ls="--", zorder=0)
        ax.text(3.0, 1.02, WINDOW_LABELS["pre74"], transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=10)
        ax.text(9.0, 1.02, WINDOW_LABELS["post74"], transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=10)
        ax.set_xlim(0.4, 11.6)
        ax.set_xticks(xtick_positions, xtick_labels, rotation=15, ha="right")
        ax.set_title(METRIC_LABELS[metric])
        ax.set_ylabel("Per-embryo mean")

    legend_handles = [
        Line2D([0], [0], color="#444444", linewidth=1.6, linestyle=EXPERIMENT_LINESTYLES[experiment_id], label=experiment_id)
        for experiment_id in EXPERIMENT_IDS
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False, title="Experiment")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_delta_summary(group_summary: pd.DataFrame, group1: str, group2: str) -> pd.DataFrame:
    wide = (
        group_summary.pivot(index=["metric", "window"], columns="rank_group", values="mean")
        .sort_index()
        .reset_index()
    )
    wide["delta_cross_minus_self_true_group1"] = wide.get(f"{group2}_like_true_{group1}") - wide.get(f"{group1}_like_true_{group1}")
    wide["delta_cross_minus_self_true_group2"] = wide.get(f"{group1}_like_true_{group2}") - wide.get(f"{group2}_like_true_{group2}")
    return wide


def write_summary_text(
    ranked_df: pd.DataFrame,
    pair_counts: pd.DataFrame,
    reference_counts: pd.DataFrame,
    delta_summary: pd.DataFrame,
    split_threshold: float,
    group1: str,
    group2: str,
    path: Path,
) -> None:
    lines = [
        f"Bidirectional morphology driver summary: {group1} vs {group2}",
        "",
        f"n_embryos_{group1}: {int((ranked_df['true_label'] == group1).sum())}",
        f"n_embryos_{group2}: {int((ranked_df['true_label'] == group2).sum())}",
        f"n_below_min_time_bins: {int((~ranked_df['eligible_for_grouping']).sum())}",
        f"signed_margin_split_threshold: {split_threshold:.6f}",
        "",
        "rank_group_counts:",
    ]
    for group_name, count in ranked_df["rank_group"].value_counts().sort_index().items():
        lines.append(f"- {group_name}: {int(count)}")

    lines.extend(["", "pair_group_counts_by_experiment:"])
    for _, row in pair_counts.iterrows():
        lines.append(f"- {row['rank_group']} / {row['experiment_id']}: {int(row['n_embryos'])}")

    lines.extend(["", "inj_ctrl_counts_by_experiment:"])
    for _, row in reference_counts.iterrows():
        lines.append(f"- {row['rank_group']} / {row['experiment_id']}: {int(row['n_embryos'])}")

    lines.extend(["", "morphology_cross_minus_self_mean_deltas:"])
    for _, row in delta_summary.iterrows():
        d1 = row.get("delta_cross_minus_self_true_group1")
        d2 = row.get("delta_cross_minus_self_true_group2")
        d1_str = "nan" if pd.isna(d1) else f"{float(d1):.6f}"
        d2_str = "nan" if pd.isna(d2) else f"{float(d2):.6f}"
        lines.append(
            f"- {row['metric']} / {row['window']}: "
            f"{group1}={d1_str}, {group2}={d2_str}"
        )

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    pred_df = load_predictions(args.predictions, args.group1, args.group2, args.late_threshold_hpf)
    ranked_df, split_threshold = rank_pair_embryos(pred_df, args.group1, args.group2, args.min_time_bins)

    morph_df = load_morphology(args.bin_width, [args.group1, args.group2, args.reference_label])
    ranked_df = attach_experiment_id(ranked_df, morph_df)
    trajectory_df, per_embryo_summary, group_summary, embryo_time_df = summarize_morphology(
        morph_df=morph_df,
        ranked_df=ranked_df,
        reference_label=args.reference_label,
        late_threshold_hpf=args.late_threshold_hpf,
    )

    pair_counts = summarize_experiment_counts(ranked_df)
    reference_counts = summarize_reference_counts(morph_df, args.reference_label)
    all_counts = pd.concat([pair_counts, reference_counts], ignore_index=True)
    delta_summary = build_delta_summary(group_summary, args.group1, args.group2)

    safe_name = _safe_name(args.group1, args.group2)
    ranking_csv = args.results_dir / f"ranking_{safe_name}.csv"
    group_counts_csv = args.results_dir / f"group_counts_{safe_name}.csv"
    trajectory_csv = args.results_dir / f"morphology_trajectories_{safe_name}.csv"
    per_embryo_summary_csv = args.results_dir / f"morphology_per_embryo_summary_{safe_name}.csv"
    summary_csv = args.results_dir / f"morphology_summary_{safe_name}.csv"
    summary_txt = args.results_dir / f"summary_{safe_name}.txt"
    labels_csv = args.results_dir / f"embryo_like_labels_{safe_name}.csv"

    ranking_fig = args.figures_dir / f"ranking_{safe_name}.png"
    trajectory_fig = args.figures_dir / f"morphology_trajectories_{safe_name}.png"
    summary_fig = args.figures_dir / f"morphology_summary_{safe_name}.png"
    family_split_fig = args.figures_dir / f"morphology_trajectories_{safe_name}_by_like_family.png"
    true_label_overlap_fig = args.figures_dir / f"morphology_trajectories_{safe_name}_true_label_overlap.png"
    total_length_fig = args.figures_dir / f"total_length_trajectories_{safe_name}.png"
    subclass_proportions_fig = args.figures_dir / f"subclass_proportions_{safe_name}.png"
    true_label_overlap_proportions_fig = args.figures_dir / f"true_label_overlap_proportions_{safe_name}.png"

    labels_df = build_embryo_like_labels(ranked_df)

    ranked_df.to_csv(ranking_csv, index=False)
    all_counts.to_csv(group_counts_csv, index=False)
    trajectory_df.to_csv(trajectory_csv, index=False)
    per_embryo_summary.to_csv(per_embryo_summary_csv, index=False)
    group_summary.to_csv(summary_csv, index=False)
    labels_df.to_csv(labels_csv, index=False)

    plot_ranking(ranked_df, split_threshold, args.group1, args.group2, ranking_fig)
    plot_trajectories(trajectory_df, embryo_time_df, args.late_threshold_hpf, args.group1, args.group2, trajectory_fig)
    plot_like_family_columns(embryo_time_df, args.late_threshold_hpf, args.group1, args.group2, family_split_fig)
    plot_true_label_overlap(embryo_time_df, args.late_threshold_hpf, args.group1, args.group2, true_label_overlap_fig)
    plot_total_length_all_groups(embryo_time_df, args.late_threshold_hpf, args.group1, args.group2, total_length_fig)
    plot_subclass_proportions(labels_df, args.group1, args.group2, subclass_proportions_fig)
    plot_true_label_overlap_proportions(labels_df, args.group1, args.group2, true_label_overlap_proportions_fig)
    plot_summary(per_embryo_summary, args.group1, args.group2, summary_fig)
    write_summary_text(
        ranked_df,
        pair_counts,
        reference_counts,
        delta_summary,
        split_threshold,
        args.group1,
        args.group2,
        summary_txt,
    )

    print(f"Wrote ranking: {ranking_csv}")
    print(f"Wrote group counts: {group_counts_csv}")
    print(f"Wrote trajectories: {trajectory_csv}")
    print(f"Wrote per-embryo summary: {per_embryo_summary_csv}")
    print(f"Wrote summary: {summary_csv}")
    print(f"Wrote embryo labels: {labels_csv}")
    print(f"Wrote figure: {ranking_fig}")
    print(f"Wrote figure: {trajectory_fig}")
    print(f"Wrote figure: {family_split_fig}")
    print(f"Wrote figure: {true_label_overlap_fig}")
    print(f"Wrote figure: {total_length_fig}")
    print(f"Wrote figure: {subclass_proportions_fig}")
    print(f"Wrote figure: {true_label_overlap_proportions_fig}")
    print(f"Wrote figure: {summary_fig}")


if __name__ == "__main__":
    main()
