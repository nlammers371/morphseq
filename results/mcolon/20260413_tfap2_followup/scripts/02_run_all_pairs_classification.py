"""TFAP2 followup: all-pairs classification across all genotypes, full time range.

Runs all-pairs binary classification across all 16 TFAP2 genotypes with no
time-window restriction. Saves scores, classifier directions, and per-embryo
contrast scores (raw_contrast_scores_long) which are consumed by
03_run_condensation.py.

Also renders an interactive emergence HTML explorer
(figures/emergence_explorer.html) from the embedding scores.

Outputs (in results/all_pairs_classification/):
  scores.parquet                        — AUROC scores per (comparison, feature, time_bin)
  raw_contrast_scores_long.parquet      — per-embryo per-comparison classifier scores
  classifier_directions.parquet         — direction metadata
  classifier_directions_vectors.npz     — unit coefficient vectors

Outputs (in figures/):
  emergence_explorer.html               — interactive emergence timeline

Run:
  conda run -n segmentation_grounded_sam --no-capture-output \\
      python results/mcolon/20260413_tfap2_followup/scripts/02_run_all_pairs_classification.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BIN_WIDTH, load_aggregate_dataframe

from analyze.classification import run_classification
from analyze.classification.viz.emergence import render_emergence_html_from_scores
from analyze.viz.styling.color_utils import build_genotype_color_lookup


def main() -> None:
    run_dir = Path(__file__).resolve().parents[1]
    results_dir = run_dir / "results" / "all_pairs_classification"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Full data — no time window filter
    df = load_aggregate_dataframe()
    df = df[df["genotype"].notna()].copy()
    print(f"Loaded {len(df):,} rows, {df['embryo_id'].nunique():,} embryos, "
          f"{df['genotype'].nunique()} genotypes")

    feature_sets = {
        "curvature": ["baseline_deviation_normalized"],
        "embedding": "z_mu_b",
    }

    print("\nRunning all-pairs classification...")
    result = run_classification(
        df,
        class_col="genotype",
        id_col="embryo_id",
        time_col="predicted_stage_hpf",
        comparisons="all_pairs",
        features=feature_sets,
        bin_width=BIN_WIDTH,
        n_permutations=500,
        min_samples_per_group=3,
        min_samples_per_member=2,
        n_jobs=24,
        save_classifier_directions=True,
        save_contrast_coordinates=True,
        save_dir=results_dir,
        overwrite=True,
        verbose=False,
    )

    print(f"Saved to: {results_dir}")
    print(f"  scores.parquet:                 {len(result.scores):,} rows")
    scores_long = result.layers["raw_contrast_scores_long"]
    print(f"  raw_contrast_scores_long:       {len(scores_long):,} rows")

    summary = (
        result.scores[result.scores["pval"] <= 0.01]
        .groupby(["feature_set", "positive_label"])["auroc_obs"]
        .max()
        .reset_index()
        .sort_values("auroc_obs", ascending=False)
    )
    print(f"\nTop significant hits (p<=0.01):")
    print(summary.head(20).to_string(index=False))

    # -----------------------------------------------------------------------
    # Emergence HTML — embedding scores only, all 16 genotypes
    # -----------------------------------------------------------------------
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Class order: all genotypes sorted by embryo count descending
    embryo_counts = (
        df.groupby("genotype")["embryo_id"].nunique().sort_values(ascending=False)
    )
    class_order = embryo_counts.index.tolist()
    color_map = build_genotype_color_lookup(class_order)

    emb_scores = result.scores[result.scores["feature_set"] == "embedding"].copy()
    print(f"\nRendering emergence HTML ({len(class_order)} genotypes)...")
    render_emergence_html_from_scores(
        emb_scores,
        class_order=class_order,
        class_colors=color_map,
        bin_width=BIN_WIDTH,
        output_path=figures_dir / "emergence_explorer.html",
    )
    print(f"  → {figures_dir / 'emergence_explorer.html'}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
