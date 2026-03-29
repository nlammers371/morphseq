#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import EXPERIMENT_LABEL, REPO_ROOT, resolve_bin_width_roots

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from analyze.classification.viz import plot_auroc_heatmaps

FEATURE_SETS = ["curvature", "length", "embedding"]
GENOTYPE_ORDER = [
    "inj_ctrl",
    "wik_ab",
    "pbx1b_crispant",
    "pbx4_crispant",
    "pbx1b_pbx4_crispant",
]
CONTROL_ORDER = ["wik_ab", "inj_ctrl"]


def _load_mode_scores():
    helper_path = Path(__file__).with_name("01_plot_classification_heatmaps.py")
    spec = importlib.util.spec_from_file_location("pbx_rerun_heatmaps", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load heatmap helper from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_mode_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PBX rerun control-comparison heatmaps side by side.")
    parser.add_argument("--bin-width", type=float, default=2.0, help="Time bin width in hpf.")
    parser.add_argument("--results-subdir", default=None, help="Relative results subdir under the PBX analysis root.")
    parser.add_argument("--figures-subdir", default=None, help="Relative figures subdir under the PBX analysis root.")
    args = parser.parse_args()

    results_dir, figures_dir = resolve_bin_width_roots(
        bin_width=args.bin_width,
        results_subdir=args.results_subdir,
        figures_subdir=args.figures_subdir,
    )
    classification_dir = results_dir / "classification"
    figure_path = figures_dir / "classification" / f"{EXPERIMENT_LABEL}_wik_ab_vs_inj_ctrl_heatmaps_v2.png"

    load_mode_scores = _load_mode_scores()
    scores = []
    for mode_stem in ("all_crispants_vs_wik_ab", "all_genotypes_vs_inj_ctrl"):
        part = load_mode_scores(classification_dir, mode_stem)
        if not part.empty:
            scores.append(part)
    if not scores:
        raise FileNotFoundError(f"No control comparison tables found in {classification_dir}")

    combined = pd.concat(scores, ignore_index=True)

    fig = plot_auroc_heatmaps(
        combined,
        heatmap_row="positive_label",
        heatmap_col="time_bin_center",
        facet_row="feature_set",
        facet_col="negative_label",
        heatmap_row_order=GENOTYPE_ORDER,
        facet_row_order=FEATURE_SETS,
        facet_col_order=CONTROL_ORDER,
        title="Control Comparisons",
        x_label="Time (hpf)",
        y_label="Genotype",
        colorbar_label="AUROC",
        sig_threshold=0.01,
        backend="matplotlib",
        cmap="BuPu",
        vmin=0.4,
        vmax=1.0,
    )
    # The default faceting-engine heatmap width is too compressed for 2 hpf control
    # comparisons with many time bins; widen the figure so significance borders remain legible.
    fig.set_size_inches(18.0, 12.0, forward=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(figure_path)


if __name__ == "__main__":
    main()
