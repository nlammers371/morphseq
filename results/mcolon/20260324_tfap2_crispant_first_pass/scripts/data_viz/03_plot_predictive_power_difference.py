"""Plot one-vs-all minus each-vs-inj_ctrl predictive-power differences."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FEATURE_SETS = ["curvature", "length", "embedding"]
FEATURE_LABELS = {
    "curvature": "Curvature",
    "length": "Length",
    "embedding": "Embedding",
}


def _min_or_nan(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return float("nan")
    return float(clean.min())


def _plot_delta_heatmap(ax, df: pd.DataFrame, title: str) -> None:
    if df.empty:
        ax.text(0.5, 0.5, "No overlapping data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    pivot = df.pivot_table(index="positive_label", columns="time_bin_center", values="delta_auroc")
    if pivot.empty:
        ax.text(0.5, 0.5, "No overlapping data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    row_order = (
        df.groupby("positive_label", observed=True)
        .agg(max_one_vs_all=("auroc_one_vs_all", "max"), max_delta=("delta_auroc", "max"))
        .sort_values(["max_one_vs_all", "max_delta"], ascending=[False, False])
        .index
    )
    pivot = pivot.loc[row_order]

    max_abs = float(np.nanmax(np.abs(pivot.to_numpy(dtype=float)))) if pivot.size else 0.0
    max_abs = max(max_abs, 0.05)
    image = ax.imshow(pivot.values, cmap="coolwarm", vmin=-max_abs, vmax=max_abs, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{col:.0f}" for col in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Time (hpf)")
    ax.set_ylabel("Genotype")
    ax.set_title(title)
    plt.colorbar(image, ax=ax, label="Delta AUROC")


def main() -> None:
    run_dir = Path(__file__).resolve().parent.parent.parent
    results_dir = run_dir / "results"
    figures_dir = run_dir / "figures" / "classification"
    classification_dir = results_dir / "classification"
    figures_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parents[5]
    sys.path.insert(0, str(run_dir))
    sys.path.insert(0, str(project_root / "src"))

    from scripts.common import EXPERIMENT_LABEL

    merged_frames: list[pd.DataFrame] = []

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for idx, feature_set in enumerate(FEATURE_SETS):
        one_path = classification_dir / f"{EXPERIMENT_LABEL}_one_vs_all_{feature_set}_comparisons.csv"
        ctrl_path = classification_dir / f"{EXPERIMENT_LABEL}_each_vs_inj_ctrl_{feature_set}_comparisons.csv"

        if not one_path.exists() or not ctrl_path.exists():
            _plot_delta_heatmap(axes[idx], pd.DataFrame(), FEATURE_LABELS[feature_set])
            continue

        one_df = pd.read_csv(one_path)
        ctrl_df = pd.read_csv(ctrl_path)

        one_df = one_df[one_df["positive_label"] != "inj_ctrl"].copy()
        ctrl_df = ctrl_df[ctrl_df["positive_label"] != "inj_ctrl"].copy()

        merged = one_df.merge(
            ctrl_df,
            on=["feature_set", "positive_label", "time_bin", "time_bin_center", "bin_width"],
            how="inner",
            suffixes=("_one_vs_all", "_each_vs_inj_ctrl"),
        )
        merged["delta_auroc"] = merged["auroc_obs_one_vs_all"] - merged["auroc_obs_each_vs_inj_ctrl"]
        merged["delta_pval"] = merged["pval_one_vs_all"] - merged["pval_each_vs_inj_ctrl"]
        merged_frames.append(merged)

        panel_df = merged.rename(
            columns={
                "auroc_obs_one_vs_all": "auroc_one_vs_all",
                "auroc_obs_each_vs_inj_ctrl": "auroc_each_vs_inj_ctrl",
            }
        )
        _plot_delta_heatmap(
            axes[idx],
            panel_df[["positive_label", "time_bin_center", "delta_auroc", "auroc_one_vs_all"]],
            FEATURE_LABELS[feature_set],
        )

    fig.suptitle(
        f"{EXPERIMENT_LABEL} Predictive Power Difference\n(one-vs-all minus each-vs-inj_ctrl)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    figure_path = figures_dir / f"{EXPERIMENT_LABEL}_one_vs_all_minus_each_vs_inj_ctrl_heatmap.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {figure_path.name}")

    if merged_frames:
        merged_df = pd.concat(merged_frames, ignore_index=True)
        merged_df.to_csv(
            classification_dir / f"{EXPERIMENT_LABEL}_one_vs_all_minus_each_vs_inj_ctrl_merged.csv",
            index=False,
        )

        summary = (
            merged_df.groupby(["feature_set", "positive_label"], as_index=False, observed=True)
            .agg(
                max_delta_auroc=("delta_auroc", "max"),
                min_delta_auroc=("delta_auroc", "min"),
                max_one_vs_all_auroc=("auroc_obs_one_vs_all", "max"),
                max_each_vs_inj_ctrl_auroc=("auroc_obs_each_vs_inj_ctrl", "max"),
                min_one_vs_all_pval=("pval_one_vs_all", _min_or_nan),
                min_each_vs_inj_ctrl_pval=("pval_each_vs_inj_ctrl", _min_or_nan),
                n_overlap_bins=("time_bin_center", "nunique"),
            )
            .sort_values(["max_one_vs_all_auroc", "max_delta_auroc"], ascending=[False, False])
        )
        summary_path = classification_dir / f"{EXPERIMENT_LABEL}_one_vs_all_minus_each_vs_inj_ctrl_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Saved: {summary_path.name}")

    print(f"Predictive-power difference outputs saved to: {figures_dir}")


if __name__ == "__main__":
    main()
