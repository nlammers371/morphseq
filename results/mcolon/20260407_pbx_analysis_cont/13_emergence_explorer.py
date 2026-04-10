"""
13_emergence_explorer.py
------------------------
Interactive phenotype emergence explorer — fully standalone HTML + inline D3.

Controls
--------
  Included genotypes    : checklist — which genotypes are in the analysis
  Emergence reference   : checklist — defines the baseline (reference set)
  AUROC threshold       : radio — none | 0.60 | 0.65 | 0.70

Tree rendering
--------------
  Layer 1 — Emergence from reference: each non-reference class is scored by
    median onset to any reference member.  Classes are grouped into emergence
    blocks by time bin (floor/bin_width * bin_width).  Each block is placed at
    the raw median emergence time (not the floored bin key).

  Layer 2 — Within-block resolution: for multi-member blocks, a recursive
    bipartition finds the best split by cross-median onset.  Accepted only when
    >= 50% of cross-partition pairs are finite.  Unresolved blocks shown with
    dashed border.

All tree computation happens client-side so reference switching is instant.
The heatmap and AUROC switch load pre-computed onset matrices.

Output
------
  results/emergence/emergence_explorer.html   (fully standalone)

Run
---
  conda run -n segmentation_grounded_sam --no-capture-output python \\
      results/mcolon/20260407_pbx_analysis_cont/13_emergence_explorer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "src"))

from analyze.classification.viz.emergence import (
    compute_emergence_data,
    render_emergence_html,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "results/positioning/pairwise/combined_pairwise_5class_bin4_perm500"
OUT_DIR  = Path(__file__).parent / "results/emergence"

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
    "inj_ctrl":            "Inj. Ctrl",
    "wik_ab":              "WIK/AB",
    "pbx1b_crispant":      "pbx1b",
    "pbx4_crispant":       "pbx4",
    "pbx1b_pbx4_crispant": "pbx1b;pbx4",
}

CLASS_COLORS = {
    "inj_ctrl":            "#2166AC",
    "wik_ab":              "#6baed6",
    "pbx1b_crispant":      "#9467bd",
    "pbx4_crispant":       "#F7B267",
    "pbx1b_pbx4_crispant": "#B2182B",
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def load_scores() -> pd.DataFrame:
    scores = pd.read_parquet(DATA_DIR / "scores.parquet")
    return scores[scores["feature_set"] == "vae"].copy().reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
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

    out = OUT_DIR / "emergence_explorer.html"
    print(f"Rendering HTML → {out}")
    render_emergence_html(
        data,
        class_labels=CLASS_LABELS,
        class_colors=CLASS_COLORS,
        bin_width=4.0,
        min_cross_support=0.5,
        output_path=out,
    )
    print("Done.")


if __name__ == "__main__":
    main()
