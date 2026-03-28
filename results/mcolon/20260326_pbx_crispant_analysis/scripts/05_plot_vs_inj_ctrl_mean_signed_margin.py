from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _pretty_label(label: str) -> str:
    return str(label).replace("_", " ")


def _load_condition_summaries(results_dir: Path) -> pd.DataFrame:
    paths = sorted(results_dir.glob("embryo_predictions_inj_ctrl_vs_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No inj_ctrl prediction tables found in {results_dir}")

    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path)
        if df.empty:
            continue

        condition = str(path.stem).removeprefix("embryo_predictions_inj_ctrl_vs_")
        df = df[df["true_label"].astype(str) == condition].copy()
        if df.empty:
            continue

        summary = (
            df.groupby(["time_bin", "time_bin_center"], as_index=False)
            .agg(
                mean_signed_margin=("signed_margin", "mean"),
                std_signed_margin=("signed_margin", "std"),
                n_rows=("signed_margin", "size"),
                n_embryos=("embryo_id", "nunique"),
            )
        )
        summary["condition"] = condition
        summary["se_signed_margin"] = summary["std_signed_margin"] / np.sqrt(summary["n_rows"].clip(lower=1))
        summary["se_signed_margin"] = summary["se_signed_margin"].fillna(0.0)
        frames.append(summary)

    if not frames:
        raise ValueError("No condition summaries could be computed from inj_ctrl prediction tables")

    return pd.concat(frames, ignore_index=True).sort_values(
        ["condition", "time_bin_center"]
    ).reset_index(drop=True)


def _plot_mean_signed_margin(summary_df: pd.DataFrame, output_path: Path) -> None:
    conditions = summary_df["condition"].drop_duplicates().tolist()
    colors = {
        "wik_ab": "#4C78A8",
        "pbx1b_crispant": "#F58518",
        "pbx4_crispant": "#54A24B",
        "pbx1b_pbx4_crispant": "#E45756",
    }

    fig, ax = plt.subplots(figsize=(11, 6.5))

    for condition in conditions:
        sub = summary_df[summary_df["condition"] == condition].copy()
        color = colors.get(condition, None)
        x = sub["time_bin_center"].to_numpy(dtype=float)
        y = sub["mean_signed_margin"].to_numpy(dtype=float)
        se = sub["se_signed_margin"].to_numpy(dtype=float)
        ax.fill_between(
            x,
            y - se,
            y + se,
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        ax.plot(
            x,
            y,
            label=_pretty_label(condition),
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=3.6,
            alpha=0.95,
        )

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_xlabel("Time (hpf)", fontsize=12)
    ax.set_ylabel("Mean Signed Margin\n(condition embryos only)", fontsize=12)
    ax.set_title("Mean Signed Margin Over Time vs Injection Control", fontsize=15, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=10, loc="best")

    y_min = min(-0.05, float(summary_df["mean_signed_margin"].min() - summary_df["se_signed_margin"].max() - 0.03))
    y_max = max(0.5, float(summary_df["mean_signed_margin"].max() + summary_df["se_signed_margin"].max() + 0.03))
    ax.set_ylim(y_min, y_max)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    run_root = Path(__file__).resolve().parent.parent
    results_dir = run_root / "results" / "misclassification" / "embedding"
    figures_dir = run_root / "figures" / "misclassification" / "embedding"

    summary_df = _load_condition_summaries(results_dir)

    summary_path = results_dir / "mean_signed_margin_vs_inj_ctrl_by_condition.csv"
    figure_path = figures_dir / "mean_signed_margin_vs_inj_ctrl_by_condition.png"

    summary_df.to_csv(summary_path, index=False)
    _plot_mean_signed_margin(summary_df, figure_path)

    print(f"Saved summary: {summary_path}")
    print(f"Saved figure: {figure_path}")


if __name__ == "__main__":
    main()
