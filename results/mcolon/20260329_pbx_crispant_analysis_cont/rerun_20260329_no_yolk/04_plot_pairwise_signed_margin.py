from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from common import BUILD06_DIR, REPO_ROOT, resolve_embedding_roots
EXPERIMENT_IDS = ["20260304", "20260306"]
EXPERIMENT_LABEL = "20260304_20260306"
DEFAULT_GROUP1 = "pbx4_crispant"
DEFAULT_GROUP2 = "pbx1b_pbx4_crispant"
DEFAULT_BIN_WIDTH = 2.0
DEFAULT_TIME_COL = "predicted_stage_hpf"
DEFAULT_EMBEDDING_PREFIX = "z_mu_b"
DISCRETE_CLASS_COLORS = {
    "inj_ctrl": "#E0A100",
    "wildtype_like_pbx4": "#2C7FB8",
    "pbx4_like_pbx4": "#D94E4E",
    "pbx4_crispant_like_true_pbx4_crispant": "#D94E4E",
    "pbx1b_pbx4_crispant_like_true_pbx4_crispant": "#F2A7B5",
    "pbx4_crispant_like_true_pbx1b_pbx4_crispant": "#A1D99B",
    "pbx1b_pbx4_crispant_like_true_pbx1b_pbx4_crispant": "#238B45",
}
DISCRETE_CLASS_LABELS = {
    "inj_ctrl": "inj_ctrl",
    "wildtype_like_pbx4": "wildtype-like-pbx4",
    "pbx4_like_pbx4": "pbx4-like-pbx4",
    "pbx4_crispant_like_true_pbx4_crispant": "pbx4-like / true pbx4",
    "pbx1b_pbx4_crispant_like_true_pbx4_crispant": "pbx1b+4-like / true pbx4",
    "pbx4_crispant_like_true_pbx1b_pbx4_crispant": "pbx4-like / true pbx1b+4",
    "pbx1b_pbx4_crispant_like_true_pbx1b_pbx4_crispant": "pbx1b+4-like / true pbx1b+4",
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


def _load_pbx_dataframe(project_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for exp_id in EXPERIMENT_IDS:
        data_path = BUILD06_DIR / f"df03_final_output_with_latents_{exp_id}.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Missing data file: {data_path}")

        part = pd.read_csv(data_path, low_memory=False)
        if "experiment_id" in part.columns:
            part = part[part["experiment_id"].astype(str) == exp_id].copy()
        else:
            part["experiment_id"] = exp_id
        frames.append(part)

    df = pd.concat(frames, ignore_index=True)
    if "use_embryo_flag" in df.columns:
        df = df[df["use_embryo_flag"]].copy()

    df = df[df["embryo_id"].notna()].copy()
    df["genotype"] = df["genotype"].fillna("unknown").map(_normalize_genotype)
    return df


def _run_engine_pairwise_classification(
    df: pd.DataFrame,
    *,
    group1: str,
    group2: str,
    time_col: str,
    feature_cols: list[str],
    bin_width: float,
    n_splits: int,
    n_permutations: int,
    n_jobs: int,
    random_state: int,
):
    from analyze.classification import run_classification

    analysis = run_classification(
        df=df.copy(),
        class_col="genotype",
        id_col="embryo_id",
        time_col=time_col,
        positive=group2,
        negative=group1,
        features={"embedding": feature_cols},
        bin_width=float(bin_width),
        n_permutations=int(n_permutations),
        n_splits=int(n_splits),
        min_samples_per_group=2,
        min_samples_per_member=2,
        n_jobs=int(n_jobs),
        random_state=int(random_state),
        verbose=True,
        save_predictions=True,
    )

    score_df = analysis.scores.copy()
    pred_df = analysis.layers["predictions"].copy()

    if score_df.empty or pred_df.empty:
        raise ValueError(f"No engine outputs produced for pair {group1} vs {group2}")

    score_df = score_df.sort_values("time_bin_center").reset_index(drop=True)
    score_df["time_bin"] = np.floor(score_df["time_bin_center"] - float(bin_width) / 2.0 + 1e-6).astype(int)
    score_df["n_samples"] = score_df["n_positive"] + score_df["n_negative"]
    score_df["n_group1"] = score_df["n_negative"]
    score_df["n_group2"] = score_df["n_positive"]
    auc_df = score_df[
        ["time_bin", "time_bin_center", "auroc_obs", "pval", "n_samples", "n_group1", "n_group2"]
    ].copy()

    pred_df = pred_df.sort_values(["embryo_id", "time_bin_center"]).reset_index(drop=True)
    pred_df["time_bin"] = np.floor(pred_df["time_bin_center"] - float(bin_width) / 2.0 + 1e-6).astype(int)
    pred_df["true_label"] = np.where(pred_df["y_true"].astype(int) == 1, group2, group1)
    pred_df["predicted_label"] = np.where(pred_df["y_pred"].astype(int) == 1, group2, group1)
    pred_df["pred_prob_group2"] = pred_df["p_pos"].astype(float)
    pred_df["positive_class"] = group2
    pred_df["support_true"] = np.where(
        pred_df["y_true"].astype(int) == 1,
        pred_df["pred_prob_group2"],
        1.0 - pred_df["pred_prob_group2"],
    )
    pred_df["confidence"] = np.abs(pred_df["pred_prob_group2"] - 0.5)
    pred_df["signed_margin"] = np.where(
        pred_df["y_true"].astype(int) == 1,
        pred_df["pred_prob_group2"] - 0.5,
        0.5 - pred_df["pred_prob_group2"],
    )

    embryo_df = pred_df[
        [
            "embryo_id",
            "time_bin",
            "time_bin_center",
            "true_label",
            "pred_prob_group2",
            "positive_class",
            "predicted_label",
            "support_true",
            "confidence",
            "signed_margin",
            "is_correct",
        ]
    ].copy()
    embryo_df = embryo_df.sort_values(["true_label", "embryo_id", "time_bin"]).reset_index(drop=True)
    return auc_df, embryo_df


def _compute_embryo_penetrance(df_embryo_probs: pd.DataFrame) -> pd.DataFrame:
    if df_embryo_probs.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for embryo_id, grp in df_embryo_probs.groupby("embryo_id"):
        grp = grp.sort_values("time_bin")
        rows.append(
            {
                "embryo_id": str(embryo_id),
                "true_label": str(grp["true_label"].iloc[0]),
                "n_time_bins": int(len(grp)),
                "mean_confidence": float(grp["confidence"].mean()),
                "mean_support_true": float(grp["support_true"].mean()),
                "mean_signed_margin": float(grp["signed_margin"].mean()),
                "abs_mean_signed_margin": float(np.abs(grp["signed_margin"].mean())),
                "min_signed_margin": float(grp["signed_margin"].min()),
                "temporal_consistency": float(grp["is_correct"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["true_label", "abs_mean_signed_margin", "mean_signed_margin"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def _load_discrete_class_lookup(project_root: Path, *, group1: str, group2: str) -> dict[str, str]:
    default_results_dir, _ = resolve_embedding_roots()
    if {group1, group2} != {"inj_ctrl", "pbx4_crispant"}:
        if {group1, group2} != {"pbx4_crispant", "pbx1b_pbx4_crispant"}:
            return {}

        ranking_path = default_results_dir / "ranking_pbx4_crispant_vs_pbx1b_pbx4_crispant_bidirectional.csv"
        if not ranking_path.exists():
            return {}

        ranking_df = pd.read_csv(ranking_path, usecols=["embryo_id", "rank_group"])
        return {
            str(row["embryo_id"]): str(row["rank_group"])
            for _, row in ranking_df.iterrows()
        }

    ranking_path = default_results_dir / "pbx4_wildtype_like_ranking_vs_inj_ctrl.csv"
    if not ranking_path.exists():
        return {}

    ranking_df = pd.read_csv(ranking_path, usecols=["embryo_id", "rank_group"])
    return {
        str(row["embryo_id"]): str(row["rank_group"])
        for _, row in ranking_df.iterrows()
    }


def _signed_margin_rgba(mean_margin: float, alpha: float) -> tuple[float, float, float, float]:
    cmap = plt.cm.RdBu_r
    norm = Normalize(vmin=-0.5, vmax=0.5)
    base = cmap(norm(mean_margin))
    return (base[0], base[1], base[2], alpha)


def _draw_genotype_panel(
    ax,
    df_embryo_probs: pd.DataFrame,
    pen: pd.DataFrame,
    *,
    genotype: str,
    max_embryos: int,
    color_mode: str,
    discrete_class_lookup: dict[str, str] | None = None,
) -> list[str]:
    pen = pen.assign(abs_margin=np.abs(pen["mean_signed_margin"]))
    pen = pen.sort_values(["abs_margin", "mean_signed_margin"], ascending=[False, False]).head(max_embryos)
    top_embryos = pen["embryo_id"].astype(str).tolist()
    pen_lookup = pen.set_index("embryo_id")
    alphas = np.linspace(0.35, 0.9, len(top_embryos)) if top_embryos else []
    highlight_id = top_embryos[0] if top_embryos else None

    for alpha, embryo_id in zip(alphas, top_embryos):
        curve = df_embryo_probs[df_embryo_probs["embryo_id"].astype(str) == embryo_id].sort_values("time_bin")
        if curve.empty:
            continue

        mean_margin = float(pen_lookup.at[embryo_id, "mean_signed_margin"])
        if color_mode == "discrete":
            discrete_class = None
            if discrete_class_lookup:
                if genotype == "inj_ctrl":
                    discrete_class = "inj_ctrl"
                else:
                    discrete_class = discrete_class_lookup.get(embryo_id)
            base_color = DISCRETE_CLASS_COLORS.get(discrete_class, "#B0B0B0")
            color_rgba = matplotlib.colors.to_rgba(base_color, alpha=alpha)
        else:
            color_rgba = _signed_margin_rgba(mean_margin, alpha)

        linewidth = 1.6
        marker_size = 3.0
        if embryo_id == highlight_id:
            if color_mode == "discrete":
                color_rgba = matplotlib.colors.to_rgba(base_color, alpha=0.98)
            else:
                color_rgba = _signed_margin_rgba(mean_margin, 0.98)
            linewidth = 2.8
            marker_size = 4.0

        ax.plot(
            curve["time_bin"].to_numpy(dtype=float),
            curve["signed_margin"].to_numpy(dtype=float),
            color=color_rgba,
            linewidth=linewidth,
            marker="o",
            markersize=marker_size,
            alpha=0.95,
        )

    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.3, alpha=0.7, label="Decision boundary")
    ax.set_xlabel("Time (hpf)", fontsize=12)
    ax.set_ylabel("Signed Margin vs 0.5", fontsize=12)
    ax.set_ylim([-0.5, 0.5])
    ax.grid(alpha=0.3)
    ax.set_title(f"{_pretty_label(genotype)} (n={len(top_embryos)})", fontsize=14, fontweight="bold")
    if top_embryos:
        ax.legend(loc="upper left", fontsize=9)
    return top_embryos


def _plot_signed_margin_trajectories(
    df_embryo_probs: pd.DataFrame,
    df_penetrance: pd.DataFrame,
    *,
    group1: str,
    group2: str,
    max_embryos: int,
    output_path: Path,
    discrete_class_lookup: dict[str, str] | None = None,
) -> None:
    if df_embryo_probs.empty or df_penetrance.empty:
        raise ValueError("No embryo-level predictions available for plotting")

    genotypes = [g for g in [group1, group2] if g in df_penetrance["true_label"].astype(str).unique()]
    if not genotypes:
        raise ValueError("Requested groups are absent from penetrance table")

    fig, axes = plt.subplots(1, len(genotypes), figsize=(8 * len(genotypes), 6), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, genotype in zip(axes, genotypes):
        pen = df_penetrance[df_penetrance["true_label"].astype(str) == genotype].copy()
        _draw_genotype_panel(
            ax,
            df_embryo_probs,
            pen,
            genotype=genotype,
            max_embryos=max_embryos,
            color_mode="continuous",
        )

    legend_handles: list[Line2D] = []
    if discrete_class_lookup:
        if {group1, group2} == {"pbx4_crispant", "pbx1b_pbx4_crispant"}:
            legend_order = [
                "pbx4_crispant_like_true_pbx4_crispant",
                "pbx1b_pbx4_crispant_like_true_pbx4_crispant",
                "pbx4_crispant_like_true_pbx1b_pbx4_crispant",
                "pbx1b_pbx4_crispant_like_true_pbx1b_pbx4_crispant",
            ]
            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    color=DISCRETE_CLASS_COLORS[key],
                    lw=2.6,
                    marker="o",
                    markersize=5,
                    label=DISCRETE_CLASS_LABELS[key],
                )
                for key in legend_order
            ]

    fig.suptitle(
        f"Embryo Signed Margin Trajectories: {_pretty_label(group1)} vs {_pretty_label(group2)}",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=2,
            frameon=False,
            fontsize=10,
        )
        fig.tight_layout(rect=[0, 0.07, 1, 1])
    else:
        fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_pbx4_split_comparison(
    df_embryo_probs: pd.DataFrame,
    df_penetrance: pd.DataFrame,
    *,
    group1: str,
    group2: str,
    max_embryos: int,
    output_path: Path,
    discrete_class_lookup: dict[str, str],
) -> None:
    if {group1, group2} == {"pbx4_crispant", "pbx1b_pbx4_crispant"}:
        genotypes = [g for g in [group1, group2] if g in df_penetrance["true_label"].astype(str).unique()]
        if not genotypes:
            return

        fig, axes = plt.subplots(2, len(genotypes), figsize=(8 * len(genotypes), 11), sharey=True)
        axes = np.atleast_2d(axes)
        top_embryos_by_genotype: dict[str, list[str]] = {}

        for col_idx, genotype in enumerate(genotypes):
            pen = df_penetrance[df_penetrance["true_label"].astype(str) == genotype].copy()
            top_ax = axes[0, col_idx]
            bottom_ax = axes[1, col_idx]

            top_embryos = _draw_genotype_panel(
                top_ax,
                df_embryo_probs,
                pen,
                genotype=genotype,
                max_embryos=max_embryos,
                color_mode="continuous",
            )
            top_embryos_by_genotype[genotype] = top_embryos
            top_ax.set_title(f"{_pretty_label(genotype)}\ncolored by signed margin", fontsize=13, fontweight="bold")

            bottom_pen = pen[pen["embryo_id"].astype(str).isin(top_embryos)].copy()
            _draw_genotype_panel(
                bottom_ax,
                df_embryo_probs,
                bottom_pen,
                genotype=genotype,
                max_embryos=max_embryos,
                color_mode="discrete",
                discrete_class_lookup=discrete_class_lookup,
            )
            bottom_ax.set_title(f"{_pretty_label(genotype)}\nsplit into discrete classes", fontsize=13, fontweight="bold")

        legend_order = [
            "pbx4_crispant_like_true_pbx4_crispant",
            "pbx1b_pbx4_crispant_like_true_pbx4_crispant",
            "pbx4_crispant_like_true_pbx1b_pbx4_crispant",
            "pbx1b_pbx4_crispant_like_true_pbx1b_pbx4_crispant",
        ]
        legend_handles = [
            Line2D(
                [0],
                [0],
                color=DISCRETE_CLASS_COLORS[key],
                lw=2.4,
                marker="o",
                markersize=5,
                label=DISCRETE_CLASS_LABELS[key],
            )
            for key in legend_order
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.01),
            ncol=2,
            frameon=False,
            fontsize=10,
        )
        fig.suptitle(
            f"Embryo Signed Margin Trajectories: {_pretty_label(group1)} vs {_pretty_label(group2)}",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )
        fig.tight_layout(rect=[0, 0.06, 1, 0.95])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    pbx4_pen = df_penetrance[df_penetrance["true_label"].astype(str) == group2].copy()
    if pbx4_pen.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    left_ax, right_ax = axes

    left_embryos = _draw_genotype_panel(
        left_ax,
        df_embryo_probs,
        pbx4_pen,
        genotype=group2,
        max_embryos=max_embryos,
        color_mode="continuous",
    )
    left_ax.set_title(f"{_pretty_label(group2)} vs {_pretty_label(group1)}\ncolored by signed margin", fontsize=13, fontweight="bold")

    right_pen = pbx4_pen[pbx4_pen["embryo_id"].astype(str).isin(left_embryos)].copy()
    _draw_genotype_panel(
        right_ax,
        df_embryo_probs,
        right_pen,
        genotype=group2,
        max_embryos=max_embryos,
        color_mode="discrete",
        discrete_class_lookup=discrete_class_lookup,
    )
    right_ax.set_title(f"{_pretty_label(group2)} split into *-like classes", fontsize=13, fontweight="bold")

    legend_handles = [
        Line2D([0], [0], color=DISCRETE_CLASS_COLORS["wildtype_like_pbx4"], lw=2.4, marker="o", markersize=5, label=DISCRETE_CLASS_LABELS["wildtype_like_pbx4"]),
        Line2D([0], [0], color=DISCRETE_CLASS_COLORS["pbx4_like_pbx4"], lw=2.4, marker="o", markersize=5, label=DISCRETE_CLASS_LABELS["pbx4_like_pbx4"]),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.74, 1.01),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        f"Embryo Signed Margin Trajectories: {_pretty_label(group1)} vs {_pretty_label(group2)}",
        fontsize=16,
        fontweight="bold",
        y=1.05,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_summary(
    *,
    auc_df: pd.DataFrame,
    penetrance_df: pd.DataFrame,
    output_path: Path,
    group1: str,
    group2: str,
) -> None:
    lines = [
        f"pair: {group1} vs {group2}",
        "classifier: analyze.classification.run_classification (binary path, class_weight='balanced')",
        f"n_auc_bins: {len(auc_df)}",
        "",
        "top_embryos_by_abs_mean_signed_margin:",
    ]

    for _, row in penetrance_df.head(20).iterrows():
        lines.append(
            f"- {row['embryo_id']} [{row['true_label']}]: "
            f"mean_signed_margin={float(row['mean_signed_margin']):.3f}, "
            f"temporal_consistency={float(row['temporal_consistency']):.3f}, "
            f"n_time_bins={int(row['n_time_bins'])}"
        )

    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot pairwise PBX signed-margin embryo trajectories.")
    parser.add_argument("--group1", default=DEFAULT_GROUP1)
    parser.add_argument("--group2", default=DEFAULT_GROUP2)
    default_results_dir, default_figures_dir = resolve_embedding_roots()
    parser.add_argument("--results-dir", type=Path, default=default_results_dir)
    parser.add_argument("--figures-dir", type=Path, default=default_figures_dir)
    parser.add_argument("--bin-width", type=float, default=DEFAULT_BIN_WIDTH)
    parser.add_argument("--time-col", default=DEFAULT_TIME_COL)
    parser.add_argument("--feature-prefix", default=DEFAULT_EMBEDDING_PREFIX)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-permutations", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-embryos", type=int, default=30)
    args = parser.parse_args()

    project_root = REPO_ROOT
    sys.path.insert(0, str(project_root / "src"))

    df = _load_pbx_dataframe(project_root)
    df = df[df["genotype"].isin([args.group1, args.group2])].copy()
    if df.empty:
        raise ValueError(f"No rows found for pair {args.group1} vs {args.group2}")

    feature_cols = [c for c in df.columns if c.startswith(args.feature_prefix)]
    if not feature_cols:
        raise ValueError(f"No feature columns found for prefix {args.feature_prefix}")

    df = df.dropna(subset=feature_cols + [args.time_col]).reset_index(drop=True)

    auc_df, embryo_df = _run_engine_pairwise_classification(
        df,
        group1=str(args.group1),
        group2=str(args.group2),
        time_col=str(args.time_col),
        feature_cols=feature_cols,
        bin_width=float(args.bin_width),
        n_splits=int(args.n_splits),
        n_permutations=int(args.n_permutations),
        n_jobs=int(args.n_jobs),
        random_state=int(args.random_state),
    )
    pen_df = _compute_embryo_penetrance(embryo_df)
    discrete_class_lookup = _load_discrete_class_lookup(
        project_root,
        group1=str(args.group1),
        group2=str(args.group2),
    )

    figures_dir = args.figures_dir
    results_dir = args.results_dir
    safe_name = f"{args.group1}_vs_{args.group2}"

    figure_path = figures_dir / f"embryo_trajectories_signed_margin_{safe_name}.png"
    split_compare_path = figures_dir / f"embryo_trajectories_signed_margin_{safe_name}_pbx4_split_compare.png"
    auc_path = results_dir / f"classification_auroc_{safe_name}.csv"
    embryo_path = results_dir / f"embryo_predictions_{safe_name}.csv"
    pen_path = results_dir / f"embryo_penetrance_{safe_name}.csv"
    summary_path = results_dir / f"summary_{safe_name}.txt"

    results_dir.mkdir(parents=True, exist_ok=True)
    auc_df.to_csv(auc_path, index=False)
    embryo_df.to_csv(embryo_path, index=False)
    pen_df.to_csv(pen_path, index=False)
    _write_summary(
        auc_df=auc_df,
        penetrance_df=pen_df,
        output_path=summary_path,
        group1=str(args.group1),
        group2=str(args.group2),
    )
    _plot_signed_margin_trajectories(
        embryo_df,
        pen_df,
        group1=str(args.group1),
        group2=str(args.group2),
        max_embryos=int(args.max_embryos),
        output_path=figure_path,
        discrete_class_lookup=discrete_class_lookup,
    )
    if discrete_class_lookup:
        _plot_pbx4_split_comparison(
            embryo_df,
            pen_df,
            group1=str(args.group1),
            group2=str(args.group2),
            max_embryos=int(args.max_embryos),
            output_path=split_compare_path,
            discrete_class_lookup=discrete_class_lookup,
        )

    print(f"Saved figure: {figure_path}")
    if discrete_class_lookup:
        print(f"Saved figure: {split_compare_path}")
    print(f"Saved embryo predictions: {embryo_path}")
    print(f"Saved embryo penetrance: {pen_path}")
    print(f"Saved AUROC summary: {auc_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
