#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _load_plot_auroc_heatmap():
    helper_path = Path(__file__).with_name("01_plot_classification_heatmaps.py")
    spec = importlib.util.spec_from_file_location("pbx_plot_classification_heatmaps", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load plotting helper from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.plot_auroc_heatmap


EXPERIMENT_LABEL = "20260304_20260306"
MODE_STEM = "all_genotypes_vs_inj_ctrl"
FEATURE_SETS = ["curvature", "length", "embedding"]


def plot_all_vs_inj_ctrl_heatmaps(bin_width: float = 2.0) -> Path:
    plot_auroc_heatmap = _load_plot_auroc_heatmap()
    run_dir = Path("results/mcolon/20260326_pbx_crispant_analysis")
    classification_dir = run_dir / "results" / f"bin_width_{bin_width:.1f}hpf" / "classification"
    figure_dir = run_dir / "figures" / f"bin_width_{bin_width:.1f}hpf" / "classification"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(FEATURE_SETS), 1, figsize=(12, 12))
    if len(FEATURE_SETS) == 1:
        axes = [axes]

    for ax, feature_set in zip(axes, FEATURE_SETS):
        comparisons_file = classification_dir / f"{EXPERIMENT_LABEL}_{MODE_STEM}_{feature_set}_comparisons.csv"
        if comparisons_file.exists():
            comparisons_df = pd.read_csv(comparisons_file)
            if not comparisons_df.empty:
                plot_auroc_heatmap(
                    ax,
                    comparisons_df,
                    f"All Genotypes vs inj_ctrl: {feature_set}",
                    sig_threshold=0.01,
                )
                continue

        ax.text(0.5, 0.5, f"No {feature_set} data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"All Genotypes vs inj_ctrl: {feature_set}")

    fig.tight_layout()
    out_path = figure_dir / f"{EXPERIMENT_LABEL}_{MODE_STEM}_heatmaps.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PBX all-genotypes-vs-inj_ctrl AUROC heatmaps.")
    parser.add_argument("--bin-width", type=float, default=2.0, help="Time bin width in hpf.")
    args = parser.parse_args()
    out_path = plot_all_vs_inj_ctrl_heatmaps(bin_width=args.bin_width)
    print(out_path)


if __name__ == "__main__":
    main()
