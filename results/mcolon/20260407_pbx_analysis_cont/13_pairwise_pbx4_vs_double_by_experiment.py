from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from analyze.classification import run_classification
from analyze.classification.viz.auroc_over_time import plot_aurocs_over_time

from common import (
    ALL_EXPERIMENT_IDS,
    BRIDGE_EXPERIMENT_ID,
    CURRENT_EXPERIMENT_IDS,
    RESULTS_ROOT,
    load_combined_pbx_dataframe,
)

GROUP1 = "pbx4_crispant"
GROUP2 = "pbx1b_pbx4_crispant"
DEFAULT_TIME_COL = "stage_hpf"
DEFAULT_FEATURE_PREFIX = "z_mu_b"

SCOPE_MAP = {
    BRIDGE_EXPERIMENT_ID: [BRIDGE_EXPERIMENT_ID],
    CURRENT_EXPERIMENT_IDS[0]: [CURRENT_EXPERIMENT_IDS[0]],
    CURRENT_EXPERIMENT_IDS[1]: [CURRENT_EXPERIMENT_IDS[1]],
    "combined_all": list(ALL_EXPERIMENT_IDS),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pbx4_crispant vs pbx1b_pbx4_crispant classification per experiment and combined in the 20260407 PBX analysis folder."
    )
    parser.add_argument("--bin-width", type=float, default=2.0)
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-embryos", type=int, default=-1, help="Maximum embryos per genotype to draw; use -1 for all.")
    parser.add_argument(
        "--results-subdir",
        default="pbx4_vs_double_by_experiment_perm500",
        help="Subdirectory under 20260407 results/classification and figures/classification.",
    )
    parser.add_argument(
        "--scopes",
        nargs="+",
        default=list(SCOPE_MAP.keys()),
        choices=list(SCOPE_MAP.keys()),
        help="Which experiment scopes to run.",
    )
    return parser.parse_args()


def _pretty(label: str) -> str:
    return str(label).replace("_", " ")


def _load_scope_dataframe(experiment_ids: list[str]) -> pd.DataFrame:
    return load_combined_pbx_dataframe(
        experiment_ids=experiment_ids,
        genotypes=[GROUP1, GROUP2],
    )


def _feature_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [c for c in df.columns if str(c).startswith(prefix)]
    if not cols:
        raise ValueError(f"No feature columns found with prefix {prefix!r}.")
    return cols


def _run_pairwise(
    df: pd.DataFrame,
    *,
    bin_width: float,
    n_permutations: int,
    n_jobs: int,
    n_splits: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_cols = _feature_columns(df, DEFAULT_FEATURE_PREFIX)
    analysis = run_classification(
        df=df.copy(),
        class_col="genotype",
        id_col="embryo_id",
        time_col=DEFAULT_TIME_COL,
        positive=GROUP2,
        negative=GROUP1,
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
        raise ValueError("No classification outputs were produced.")

    score_df = score_df.sort_values("time_bin_center").reset_index(drop=True)
    score_df["time_bin"] = np.floor(score_df["time_bin_center"] - float(bin_width) / 2.0 + 1e-6).astype(int)
    score_df["n_samples"] = score_df["n_positive"] + score_df["n_negative"]
    score_df["n_group1"] = score_df["n_negative"]
    score_df["n_group2"] = score_df["n_positive"]
    auc_df = score_df[["time_bin", "time_bin_center", "auroc_obs", "pval", "n_samples", "n_group1", "n_group2"]].copy()

    pred_df = pred_df.sort_values(["embryo_id", "time_bin_center"]).reset_index(drop=True)
    pred_df["time_bin"] = np.floor(pred_df["time_bin_center"] - float(bin_width) / 2.0 + 1e-6).astype(int)
    pred_df["true_label"] = np.where(pred_df["y_true"].astype(int) == 1, GROUP2, GROUP1)
    pred_df["predicted_label"] = np.where(pred_df["y_pred"].astype(int) == 1, GROUP2, GROUP1)
    pred_df["pred_prob_group2"] = pred_df["p_pos"].astype(float)
    pred_df["positive_class"] = GROUP2
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

    embryo_df = pred_df[[
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
    ]].copy()
    embryo_df = embryo_df.sort_values(["true_label", "embryo_id", "time_bin"]).reset_index(drop=True)
    return score_df, auc_df, embryo_df


def _compute_penetrance(df_embryo: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for embryo_id, grp in df_embryo.groupby("embryo_id"):
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


def _rgba(mean_margin: float, alpha: float) -> tuple[float, float, float, float]:
    cmap = plt.cm.RdBu_r
    norm = Normalize(vmin=-0.5, vmax=0.5)
    rgba = cmap(norm(float(mean_margin)))
    return (rgba[0], rgba[1], rgba[2], alpha)


def _plot_multiline(
    embryo_df: pd.DataFrame,
    pen_df: pd.DataFrame,
    *,
    max_embryos: int,
    title: str,
    output_path: Path,
) -> None:
    genotypes = [g for g in [GROUP1, GROUP2] if g in pen_df["true_label"].astype(str).unique()]
    if not genotypes:
        return

    fig, axes = plt.subplots(1, len(genotypes), figsize=(7 * len(genotypes), 6), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, genotype in zip(axes, genotypes):
        ranked = pen_df[pen_df["true_label"].astype(str) == genotype].sort_values(
            ["abs_mean_signed_margin", "mean_signed_margin"],
            ascending=[False, False],
        )
        if int(max_embryos) > 0:
            ranked = ranked.head(int(max_embryos))
        selected = ranked["embryo_id"].astype(str).tolist()
        ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.axhline(0.5, color="#cccccc", linestyle=":", linewidth=1.0, alpha=0.8)
        ax.axhline(-0.5, color="#cccccc", linestyle=":", linewidth=1.0, alpha=0.8)

        for embryo_id in selected:
            traj = embryo_df[embryo_df["embryo_id"].astype(str) == embryo_id].sort_values("time_bin_center")
            if traj.empty:
                continue
            mean_margin = float(ranked.loc[ranked["embryo_id"].astype(str) == embryo_id, "mean_signed_margin"].iloc[0])
            alpha = 0.95 if abs(mean_margin) >= 0.1 else 0.6
            color = _rgba(mean_margin, alpha)
            ax.plot(
                traj["time_bin_center"].to_numpy(dtype=float),
                traj["signed_margin"].to_numpy(dtype=float),
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=3.5,
            )

        ax.set_title(f"{_pretty(genotype)}\n{len(selected)} embryos", fontsize=12, fontweight="bold")
        ax.set_xlabel("Hours Post Fertilization (hpf)")
        ax.grid(True, alpha=0.2)
        ax.set_ylim(-0.55, 0.55)

    axes[0].set_ylabel("Signed margin")
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_summary(
    *,
    scope_name: str,
    experiment_ids: list[str],
    auc_df: pd.DataFrame,
    pen_df: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        f"scope: {scope_name}",
        f"experiments: {','.join(experiment_ids)}",
        f"pair: {GROUP1} vs {GROUP2}",
        "classifier: analyze.classification.run_classification",
        f"n_auc_bins: {len(auc_df)}",
        "",
        "top_embryos_by_abs_mean_signed_margin:",
    ]
    for _, row in pen_df.head(20).iterrows():
        lines.append(
            f"- {row['embryo_id']} [{row['true_label']}]: "
            f"mean_signed_margin={float(row['mean_signed_margin']):.3f}, "
            f"temporal_consistency={float(row['temporal_consistency']):.3f}, "
            f"n_time_bins={int(row['n_time_bins'])}"
        )
    output_path.write_text("\n".join(lines) + "\n")


def _write_manifest(output_path: Path, payload: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    results_dir = RESULTS_ROOT / "results" / "classification" / args.results_subdir
    figures_dir = RESULTS_ROOT / "figures" / "classification" / args.results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    manifest_payload = {
        "analysis": "pairwise_pbx4_vs_double_by_experiment",
        "pair": [GROUP1, GROUP2],
        "scopes": {scope: SCOPE_MAP[scope] for scope in args.scopes},
        "bin_width_hpf": float(args.bin_width),
        "n_permutations": int(args.n_permutations),
        "n_jobs": int(args.n_jobs),
        "n_splits": int(args.n_splits),
        "max_embryos": int(args.max_embryos),
        "time_col": DEFAULT_TIME_COL,
        "feature_prefix": DEFAULT_FEATURE_PREFIX,
    }

    for scope in args.scopes:
        experiment_ids = list(SCOPE_MAP[scope])
        df = _load_scope_dataframe(experiment_ids)
        score_df, auc_df, embryo_df = _run_pairwise(
            df,
            bin_width=float(args.bin_width),
            n_permutations=int(args.n_permutations),
            n_jobs=int(args.n_jobs),
            n_splits=int(args.n_splits),
            random_state=int(args.random_state),
        )
        pen_df = _compute_penetrance(embryo_df)

        scoped_name = f"{scope}_{GROUP1}_vs_{GROUP2}"
        embryo_df.to_csv(results_dir / f"embryo_predictions_{scoped_name}.csv", index=False)
        pen_df.to_csv(results_dir / f"embryo_penetrance_{scoped_name}.csv", index=False)
        auc_df.to_csv(results_dir / f"classification_auroc_{scoped_name}.csv", index=False)

        _write_summary(
            scope_name=scope,
            experiment_ids=experiment_ids,
            auc_df=auc_df,
            pen_df=pen_df,
            output_path=results_dir / f"summary_{scoped_name}.txt",
        )

        _plot_multiline(
            embryo_df,
            pen_df,
            max_embryos=int(args.max_embryos),
            title=f"Embryo signed-margin trajectories: {_pretty(GROUP1)} vs {_pretty(GROUP2)} ({scope})",
            output_path=figures_dir / f"embryo_trajectories_signed_margin_{scoped_name}.png",
        )

        plot_aurocs_over_time(
            score_df.assign(positive_label=GROUP2),
            curve_col="positive_label",
            backend="both",
            show_null_band=True,
            show_significance=True,
            sig_threshold=0.01,
            title=f"Classification over time: {_pretty(GROUP1)} vs {_pretty(GROUP2)} ({scope})",
            output_path=figures_dir / f"classification_over_time_{scoped_name}",
        )

        print(scope)
        print(results_dir / f"classification_auroc_{scoped_name}.csv")
        print(figures_dir / f"embryo_trajectories_signed_margin_{scoped_name}.png")
        print(figures_dir / f"classification_over_time_{scoped_name}.png")

    _write_manifest(results_dir / "pairwise_pbx4_vs_double_manifest.json", manifest_payload)
    print(results_dir)
    print(figures_dir)


if __name__ == "__main__":
    main()
