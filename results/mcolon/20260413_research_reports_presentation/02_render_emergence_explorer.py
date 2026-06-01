"""Render the talk version of the phenotype emergence explorer."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze.classification.viz.emergence import (
    compute_emergence_data,
    render_emergence_html,
)

DATA_DIR = PROJECT_ROOT / "results/mcolon/20260407_pbx_analysis_cont/results/positioning/pairwise/combined_pairwise_5class_bin4_perm500"
OUTPUT_PATH = Path(__file__).resolve().parent / "emergence_explorer.html"

AUROC_LEVELS = {
    "none": 0.0,
    "0.60": 0.60,
    "0.65": 0.65,
    "0.70": 0.70,
}

ALL_CLASSES = [
    "pbx1b_pbx4_crispant",
    "pbx4_crispant",
    "pbx1b_crispant",
    "inj_ctrl",
    "wik_ab",
]

CLASS_LABELS = {
    "inj_ctrl": "Inj. Ctrl",
    "wik_ab": "wild type",
    "pbx1b_crispant": "pbx1b",
    "pbx4_crispant": "pbx4",
    "pbx1b_pbx4_crispant": "pbx1b;pbx4",
}

CLASS_COLORS = {
    "inj_ctrl": "#2166AC",
    "wik_ab": "#6baed6",
    "pbx1b_crispant": "#9467bd",
    "pbx4_crispant": "#F7B267",
    "pbx1b_pbx4_crispant": "#B2182B",
}


def load_scores() -> pd.DataFrame:
    scores = pd.read_parquet(DATA_DIR / "scores.parquet")
    return scores[scores["feature_set"] == "vae"].copy().reset_index(drop=True)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Loading scores...")
    scores = load_scores()

    print("Computing emergence data...")
    data = compute_emergence_data(
        scores,
        ALL_CLASSES,
        auroc_levels=AUROC_LEVELS,
        p_sep=0.05,
        p_ns=0.10,
        subsequent_frac=0.40,
    )

    print(f"Rendering HTML -> {OUTPUT_PATH}")
    render_emergence_html(
        data,
        class_labels=CLASS_LABELS,
        class_colors=CLASS_COLORS,
        bin_width=4.0,
        min_cross_support=0.5,
        heatmap_font_scale=1.2,
        output_path=OUTPUT_PATH,
    )
    print("Done.")


if __name__ == "__main__":
    main()
