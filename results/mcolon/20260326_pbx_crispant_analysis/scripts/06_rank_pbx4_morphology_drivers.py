from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


EXPERIMENT_IDS = ["20260304", "20260306"]
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

from analyze.viz.plotting.feature_over_time import plot_feature_over_time


RESULTS_DIR = REPO_ROOT / "results" / "mcolon" / "20260326_pbx_crispant_analysis" / "results" / "misclassification" / "embedding"
FIGURES_DIR = REPO_ROOT / "results" / "mcolon" / "20260326_pbx_crispant_analysis" / "figures" / "misclassification" / "embedding"
BUILD06_DIR = REPO_ROOT / "morphseq_playground" / "metadata" / "build06_output"

PAIRWISE_PREDICTIONS = RESULTS_DIR / "embryo_predictions_inj_ctrl_vs_pbx4_crispant.csv"
RANKING_CSV = RESULTS_DIR / "pbx4_wildtype_like_ranking_vs_inj_ctrl.csv"
GROUP_COUNTS_CSV = RESULTS_DIR / "pbx4_wildtype_like_group_counts_by_experiment.csv"
TRAJECTORY_CSV = RESULTS_DIR / "pbx4_wildtype_like_morphology_trajectories.csv"
SUMMARY_CSV = RESULTS_DIR / "pbx4_wildtype_like_morphology_summary.csv"
PER_EMBRYO_SUMMARY_CSV = RESULTS_DIR / "pbx4_wildtype_like_morphology_per_embryo_summary.csv"
SUMMARY_TXT = RESULTS_DIR / "pbx4_wildtype_like_morphology_summary.txt"

RANKING_FIG = FIGURES_DIR / "pbx4_wildtype_like_ranked_mean_signed_margin_vs_inj_ctrl.png"
TRAJECTORY_FIG = FIGURES_DIR / "pbx4_wildtype_like_morphology_trajectories.png"
SUMMARY_FIG = FIGURES_DIR / "pbx4_wildtype_like_morphology_summary_pre74_post74.png"

CONTROL_LABEL = "inj_ctrl"
PBX4_GROUP_ORDER = ["wildtype_like_pbx4", "pbx4_like_pbx4"]
GROUP_ORDER = [CONTROL_LABEL, *PBX4_GROUP_ORDER]
GROUP_LABELS = {
    CONTROL_LABEL: "inj_ctrl",
    "wildtype_like_pbx4": "wildtype-like-pbx4",
    "pbx4_like_pbx4": "pbx4-like-pbx4",
}
GROUP_COLORS = {
    CONTROL_LABEL: "#E0A100",
    "wildtype_like_pbx4": "#2C7FB8",
    "pbx4_like_pbx4": "#D94E4E",
}
EXPERIMENT_COLORS = {
    "20260304": "#2A9D8F",
    "20260306": "#F58518",
}
EXPERIMENT_MARKERS = {
    "20260304": "o",
    "20260306": "s",
}
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
        return CONTROL_LABEL
    return g.replace("wik-ab", "wik_ab")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank PBX4 embryos by signed margin and test morphology drivers.")
    parser.add_argument("--predictions", type=Path, default=PAIRWISE_PREDICTIONS)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--bin-width", type=float, default=2.0)
    parser.add_argument("--late-threshold-hpf", type=float, default=74.0)
    parser.add_argument("--min-time-bins", type=int, default=8)
    parser.add_argument("--pbx4-label", default="pbx4_crispant")
    parser.add_argument("--control-label", default=CONTROL_LABEL)
    return parser.parse_args()


def load_predictions(path: Path, pbx4_label: str, late_threshold_hpf: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"embryo_id", "true_label", "time_bin_center", "signed_margin", "is_correct"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required prediction columns: {missing}")

    df = df[df["true_label"] == pbx4_label].copy()
    if df.empty:
        raise ValueError(f"No rows found for true_label={pbx4_label!r} in {path}")

    df["window"] = np.where(df["time_bin_center"] < late_threshold_hpf, "pre74", "post74")
    return df


def rank_pbx4_embryos(pred_df: pd.DataFrame, min_time_bins: int) -> tuple[pd.DataFrame, float]:
    ranked = (
        pred_df.groupby("embryo_id", as_index=False)
        .agg(
            mean_signed_margin=("signed_margin", "mean"),
            frac_correct=("is_correct", "mean"),
            n_time_bins=("time_bin_center", "nunique"),
            first_hpf=("time_bin_center", "min"),
            last_hpf=("time_bin_center", "max"),
        )
        .sort_values("mean_signed_margin")
        .reset_index(drop=True)
    )

    pre = (
        pred_df[pred_df["window"] == "pre74"]
        .groupby("embryo_id")["signed_margin"]
        .mean()
        .rename("mean_signed_margin_pre74")
    )
    post = (
        pred_df[pred_df["window"] == "post74"]
        .groupby("embryo_id")["signed_margin"]
        .mean()
        .rename("mean_signed_margin_post74")
    )
    ranked = ranked.merge(pre, on="embryo_id", how="left").merge(post, on="embryo_id", how="left")
    ranked["eligible_for_grouping"] = ranked["n_time_bins"] >= min_time_bins
    ranked["rank_group"] = np.where(
        ranked["mean_signed_margin"] < 0.0,
        "wildtype_like_pbx4",
        "pbx4_like_pbx4",
    )
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
    control_label: str,
    late_threshold_hpf: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pbx4_group_map = ranked_df.loc[:, ["embryo_id", "rank_group"]].copy()
    control_map = (
        morph_df.loc[morph_df["genotype"] == control_label, ["embryo_id"]]
        .drop_duplicates()
        .assign(rank_group=control_label)
    )
    group_map = pd.concat([pbx4_group_map, control_map], ignore_index=True)

    merged = morph_df.merge(group_map, on="embryo_id", how="inner")
    if merged.empty:
        raise ValueError("No morphology rows matched the requested embryo groups.")

    embryo_time = (
        merged.groupby(["embryo_id", "experiment_id", "rank_group", "time_bin", "time_bin_center"], as_index=False)
        .agg(
            baseline_deviation_normalized=("baseline_deviation_normalized", "mean"),
            total_length_um=("total_length_um", "mean"),
            surface_area_um=("surface_area_um", "mean"),
        )
    )

    trajectory_rows: list[pd.DataFrame] = []
    for metric in METRIC_LABELS:
        part = (
            embryo_time.groupby(["rank_group", "time_bin", "time_bin_center"], as_index=False)
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

    features_over_time_df = embryo_time.copy()
    features_over_time_df["plot_group"] = features_over_time_df["rank_group"].map(GROUP_LABELS)
    features_over_time_df["predicted_stage_hpf"] = features_over_time_df["time_bin_center"]

    per_embryo_rows: list[pd.DataFrame] = []
    for metric in METRIC_LABELS:
        for window_name, mask in {
            "full": np.full(len(embryo_time), True, dtype=bool),
            "pre74": embryo_time["time_bin_center"] < late_threshold_hpf,
            "post74": embryo_time["time_bin_center"] >= late_threshold_hpf,
        }.items():
            part = embryo_time.loc[mask, ["embryo_id", "experiment_id", "rank_group", metric]].copy()
            if part.empty:
                continue
            part = (
                part.groupby(["embryo_id", "experiment_id", "rank_group"], as_index=False)[metric]
                .mean()
                .rename(columns={metric: "metric_value"})
            )
            part["metric"] = metric
            part["window"] = window_name
            per_embryo_rows.append(part)
    per_embryo_summary = pd.concat(per_embryo_rows, ignore_index=True)

    group_summary = (
        per_embryo_summary.groupby(["metric", "window", "rank_group"], as_index=False)
        .agg(
            n_embryos=("metric_value", "count"),
            mean=("metric_value", "mean"),
            median=("metric_value", "median"),
            sd=("metric_value", "std"),
        )
    )

    wide = (
        group_summary.pivot(index=["metric", "window"], columns="rank_group", values=["mean", "median", "n_embryos", "sd"])
        .sort_index()
    )
    wide.columns = ["_".join(col).strip() for col in wide.columns.to_flat_index()]
    wide = wide.reset_index()
    wide["mean_delta_wt_like_minus_pbx4_like"] = wide.get("mean_wildtype_like_pbx4") - wide.get("mean_pbx4_like_pbx4")
    return trajectory_df, per_embryo_summary, wide, features_over_time_df


def summarize_experiment_counts(ranked_df: pd.DataFrame) -> pd.DataFrame:
    return (
        ranked_df.groupby(["rank_group", "experiment_id"], as_index=False)
        .agg(n_embryos=("embryo_id", "nunique"))
        .sort_values(["rank_group", "experiment_id"])
        .reset_index(drop=True)
    )


def summarize_control_experiment_counts(morph_df: pd.DataFrame, control_label: str) -> pd.DataFrame:
    return (
        morph_df[morph_df["genotype"] == control_label][["embryo_id", "experiment_id"]]
        .drop_duplicates()
        .assign(rank_group=control_label)
        .groupby(["rank_group", "experiment_id"], as_index=False)
        .agg(n_embryos=("embryo_id", "nunique"))
        .sort_values(["rank_group", "experiment_id"])
        .reset_index(drop=True)
    )


def plot_ranking(ranked_df: pd.DataFrame, split_threshold: float, path: Path) -> None:
    ranked_plot = ranked_df.sort_values("mean_signed_margin").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axhline(split_threshold, color=GROUP_COLORS["wildtype_like_pbx4"], lw=2.8, ls="--", alpha=0.95)

    for experiment_id, marker in EXPERIMENT_MARKERS.items():
        part_exp = ranked_plot[ranked_plot["experiment_id"] == experiment_id]
        if part_exp.empty:
            continue
        for group_name in PBX4_GROUP_ORDER:
            part = part_exp[part_exp["rank_group"] == group_name]
            if part.empty:
                continue
            ax.scatter(
                part.index,
                part["mean_signed_margin"],
                s=62,
                marker=marker,
                color=GROUP_COLORS[group_name],
                edgecolors=EXPERIMENT_COLORS.get(experiment_id, "#333333"),
                linewidths=1.4,
                alpha=0.95,
                label=f"{experiment_id} / {GROUP_LABELS[group_name]}",
            )

    ax.set_title("PBX4 embryos ranked by mean signed margin vs inj_ctrl")
    ax.set_xlabel("PBX4 embryos (sorted)")
    ax.set_ylabel("Mean signed margin")

    handles, labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    keep_handles = []
    keep_labels = []
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        keep_handles.append(handle)
        keep_labels.append(label)
    ax.legend(keep_handles, keep_labels, frameon=False, fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories(features_over_time_df: pd.DataFrame, late_threshold_hpf: float, path: Path) -> None:
    fig = plot_feature_over_time(
        features_over_time_df,
        features=list(METRIC_LABELS.keys()),
        time_col="predicted_stage_hpf",
        id_col="embryo_id",
        color_by="plot_group",
        color_lookup={GROUP_LABELS[key]: GROUP_COLORS[key] for key in GROUP_ORDER},
        backend="matplotlib",
        title="PBX4 morphology features over time vs inj_ctrl",
    )
    for ax in fig.axes:
        ax.axvline(late_threshold_hpf, color="#666666", lw=1.0, ls="--", alpha=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_summary(per_embryo_summary: pd.DataFrame, path: Path) -> None:
    per_embryo_summary = per_embryo_summary.copy()
    per_embryo_summary["experiment_id"] = per_embryo_summary["experiment_id"].astype(str)
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=False)
    panel_order = [
        "baseline_deviation_normalized",
        "total_length_um",
        "surface_area_um",
    ]
    experiment_offsets = {
        "20260304": -0.12,
        "20260306": 0.12,
    }
    experiment_linestyles = {
        "20260304": "-",
        "20260306": "--",
    }
    window_positions = {
        "pre74": [1, 2, 3],
        "post74": [5, 6, 7],
    }
    xtick_positions = window_positions["pre74"] + window_positions["post74"]
    xtick_labels = [GROUP_LABELS[g] for g in GROUP_ORDER] + [GROUP_LABELS[g] for g in GROUP_ORDER]

    for ax, metric in zip(axes, panel_order):
        metric_panel = per_embryo_summary[per_embryo_summary["metric"] == metric].copy()
        for window, positions in window_positions.items():
            panel = metric_panel[metric_panel["window"] == window]
            for xpos, group_name in zip(positions, GROUP_ORDER):
                group_panel = panel[panel["rank_group"] == group_name]
                for experiment_id in EXPERIMENT_IDS:
                    exp_part = group_panel[group_panel["experiment_id"] == experiment_id]["metric_value"].dropna().to_numpy()
                    if len(exp_part) == 0:
                        continue
                    box = ax.boxplot(
                        [exp_part],
                        positions=[xpos + experiment_offsets[experiment_id]],
                        widths=0.22,
                        patch_artist=True,
                        boxprops=dict(
                            facecolor=GROUP_COLORS[group_name],
                            alpha=0.22,
                            edgecolor=GROUP_COLORS[group_name],
                            linewidth=1.4,
                        ),
                        medianprops=dict(
                            color=GROUP_COLORS[group_name],
                            linewidth=1.6,
                        ),
                        whiskerprops=dict(
                            color=GROUP_COLORS[group_name],
                            linewidth=1.2,
                        ),
                        capprops=dict(
                            color=GROUP_COLORS[group_name],
                            linewidth=1.2,
                        ),
                        flierprops=dict(
                            marker="o",
                            markerfacecolor="white",
                            markeredgecolor=GROUP_COLORS[group_name],
                            markersize=4.5,
                            alpha=0.9,
                        ),
                    )
                    for artist in box["boxes"]:
                        artist.set_linestyle(experiment_linestyles[experiment_id])
                    for artist in box["whiskers"] + box["caps"] + box["medians"]:
                        artist.set_linestyle(experiment_linestyles[experiment_id])
                    if experiment_id == "20260306":
                        for artist in box["whiskers"] + box["caps"] + box["medians"]:
                            artist.set_dashes((3, 2))
        ax.axvline(4.0, color="#bbbbbb", lw=1.0, ls="--", zorder=0)
        ax.text(2.0, 1.02, WINDOW_LABELS["pre74"], transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=10)
        ax.text(6.0, 1.02, WINDOW_LABELS["post74"], transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=10)
        ax.set_xlim(0.4, 7.6)
        ax.set_xticks(xtick_positions, xtick_labels, rotation=15, ha="right")
        ax.set_title(METRIC_LABELS[metric])
        ax.set_ylabel("Per-embryo mean")
    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#444444",
            linewidth=1.6,
            linestyle=experiment_linestyles[experiment_id],
            label=experiment_id,
        )
        for experiment_id in EXPERIMENT_IDS
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False, title="Experiment")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary_text(
    ranked_df: pd.DataFrame,
    pbx4_group_counts: pd.DataFrame,
    control_counts: pd.DataFrame,
    group_summary: pd.DataFrame,
    split_threshold: float,
    path: Path,
) -> None:
    lines = [
        "PBX4 morphology driver summary",
        "",
        f"n_pbx4_embryos_total: {len(ranked_df)}",
        f"n_below_min_time_bins: {int((~ranked_df['eligible_for_grouping']).sum())}",
        f"signed_margin_split_threshold: {split_threshold:.6f}",
        "",
        "rank_group_counts:",
    ]
    for group_name, count in ranked_df["rank_group"].value_counts().sort_index().items():
        lines.append(f"- {group_name}: {int(count)}")

    lines.extend(["", "pbx4_group_counts_by_experiment:"])
    for _, row in pbx4_group_counts.iterrows():
        lines.append(f"- {row['rank_group']} / {row['experiment_id']}: {int(row['n_embryos'])}")

    lines.extend(["", "inj_ctrl_counts_by_experiment:"])
    for _, row in control_counts.iterrows():
        lines.append(f"- {row['rank_group']} / {row['experiment_id']}: {int(row['n_embryos'])}")

    lines.extend(["", "morphology_mean_deltas:"])
    for _, row in group_summary.iterrows():
        delta = row.get("mean_delta_wt_like_minus_pbx4_like")
        delta_str = "nan" if pd.isna(delta) else f"{float(delta):.6f}"
        lines.append(f"- {row['metric']} / {row['window']}: {delta_str}")

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    pred_df = load_predictions(args.predictions, args.pbx4_label, args.late_threshold_hpf)
    ranked_df, split_threshold = rank_pbx4_embryos(pred_df, args.min_time_bins)

    morph_df = load_morphology(args.bin_width, [args.pbx4_label, args.control_label])
    ranked_df = attach_experiment_id(ranked_df, morph_df)
    trajectory_df, per_embryo_summary, group_summary, features_over_time_df = summarize_morphology(
        morph_df=morph_df,
        ranked_df=ranked_df,
        control_label=args.control_label,
        late_threshold_hpf=args.late_threshold_hpf,
    )

    pbx4_group_counts = summarize_experiment_counts(ranked_df)
    control_counts = summarize_control_experiment_counts(morph_df, args.control_label)
    all_counts = pd.concat([pbx4_group_counts, control_counts], ignore_index=True)

    ranked_df.to_csv(RANKING_CSV, index=False)
    all_counts.to_csv(GROUP_COUNTS_CSV, index=False)
    trajectory_df.to_csv(TRAJECTORY_CSV, index=False)
    per_embryo_summary.to_csv(PER_EMBRYO_SUMMARY_CSV, index=False)
    group_summary.to_csv(SUMMARY_CSV, index=False)

    plot_ranking(ranked_df, split_threshold, RANKING_FIG)
    plot_trajectories(features_over_time_df, args.late_threshold_hpf, TRAJECTORY_FIG)
    plot_summary(per_embryo_summary, SUMMARY_FIG)
    write_summary_text(ranked_df, pbx4_group_counts, control_counts, group_summary, split_threshold, SUMMARY_TXT)

    print(f"Wrote ranking: {RANKING_CSV}")
    print(f"Wrote group counts: {GROUP_COUNTS_CSV}")
    print(f"Wrote trajectories: {TRAJECTORY_CSV}")
    print(f"Wrote summary: {SUMMARY_CSV}")
    print(f"Wrote figure: {RANKING_FIG}")
    print(f"Wrote figure: {TRAJECTORY_FIG}")
    print(f"Wrote figure: {SUMMARY_FIG}")


if __name__ == "__main__":
    main()
