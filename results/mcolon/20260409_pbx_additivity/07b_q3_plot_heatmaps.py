"""
Q3 extension: Per-embryo direction heatmaps (plot only).

Loads saved classification results from disk and generates separability heatmaps.
Requires 07a_q3_run_classification.py to have been run first.

Two feature variants:
  raw_direction : z_i = x_i - mu_non      (direction + magnitude)
  unit_direction: z_i / ||z_i||           (direction only, magnitude removed)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from analyze.classification.engine.analysis import ClassificationAnalysis
from analyze.classification.viz.heatmaps import plot_auroc_heatmaps

INJ_CTRL = "inj_ctrl"
PBX1B    = "pbx1b_crispant"
PBX4     = "pbx4_crispant"
ROW_ORDER = [INJ_CTRL, PBX1B, PBX4]

SIG_THRESHOLD = 0.05

RESULTS_DIR = SCRIPT_DIR / "results" / "q3_embryo_direction"
FIGURES_DIR = SCRIPT_DIR / "figures" / "q3_embryo_direction"


def _add_separability(scores: pd.DataFrame, save_dir: Path) -> pd.DataFrame:
    """
    Replace signed AUROC with separability = max(auroc, 1-auroc) and compute
    symmetric permutation p-values: p = fraction of null |auroc_null - 0.5|
    >= |auroc_obs - 0.5|.
    """
    scores = scores.copy()
    scores["separability"] = np.maximum(scores["auroc_obs"], 1.0 - scores["auroc_obs"])

    null_path = Path(save_dir) / "null_distributions.npz"
    if null_path.exists():
        raw = np.load(null_path, allow_pickle=True)
        feat_sets    = raw["feature_set"]
        comp_ids_arr = raw["comparison_id"]
        tbc_arr      = raw["time_bin_center"]
        null_auc     = raw["null_auc"]
        pvals = []
        for _, row in scores.iterrows():
            mask = (
                (feat_sets == row["feature_set"]) &
                (comp_ids_arr == row["comparison_id"]) &
                (np.abs(tbc_arr - row["time_bin_center"]) < 0.1)
            )
            idx = np.where(mask)[0]
            if len(idx) == 0:
                pvals.append(float("nan"))
                continue
            null_aurocs = null_auc[idx[0]]
            sep_obs  = abs(row["auroc_obs"] - 0.5)
            sep_null = np.abs(null_aurocs - 0.5)
            p = (np.sum(sep_null >= sep_obs) + 1) / (len(sep_null) + 1)
            pvals.append(float(p))
        scores["pval_sym"] = pvals
    else:
        scores["pval_sym"] = np.nan

    return scores


def _mirror_separability(scores: pd.DataFrame) -> pd.DataFrame:
    """Mirror A-vs-B into B-vs-A (separability is symmetric)."""
    flipped = scores.copy()
    flipped["positive_label"] = scores["negative_label"].values
    flipped["negative_label"] = scores["positive_label"].values
    return pd.concat([scores, flipped], ignore_index=True)


def _plot_heatmap(results_subdir: Path, out_path: Path, title: str) -> None:
    res = ClassificationAnalysis.load(results_subdir)
    scores = _add_separability(res.scores, results_subdir)
    mirrored = _mirror_separability(scores)
    row_order = [g for g in ROW_ORDER if g in set(mirrored["positive_label"].unique())]

    pval_col = "pval_sym" if not mirrored["pval_sym"].isna().all() else None

    plot_auroc_heatmaps(
        mirrored,
        heatmap_row="positive_label",
        heatmap_col="time_bin_center",
        facet_col="negative_label",
        heatmap_row_order=row_order or None,
        auroc_col="separability",
        pval_col=pval_col or "pval_sym",
        show_significance=pval_col is not None,
        title=title,
        x_label="Time bin center (hpf)",
        y_label="Positive genotype",
        colorbar_label="Separability  max(AUROC, 1−AUROC)",
        vmin=0.5,
        vmax=1.0,
        sig_threshold=SIG_THRESHOLD,
        output_path=out_path,
        backend="matplotlib",
    )
    print(f"Saved: {out_path}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    _plot_heatmap(
        results_subdir=RESULTS_DIR / "raw_direction",
        out_path=FIGURES_DIR / "q3_sep_raw_direction.png",
        title=(
            "Q3: Separability — raw direction vectors (z_i = x_i − mu_non)\n"
            "max(AUROC, 1−AUROC): orientation-independent, 0.5 = chance\n"
            "direction + magnitude combined | inj_ctrl row should be near 0.5 vs pbx"
        ),
    )
    _plot_heatmap(
        results_subdir=RESULTS_DIR / "unit_direction",
        out_path=FIGURES_DIR / "q3_sep_unit_direction.png",
        title=(
            "Q3: Separability — unit direction vectors (z_i / ||z_i||)\n"
            "max(AUROC, 1−AUROC): orientation-independent, 0.5 = chance\n"
            "direction only (magnitude removed) | raw >> unit → magnitude drives sep | raw ≈ unit → direction sufficient"
        ),
    )


if __name__ == "__main__":
    main()
