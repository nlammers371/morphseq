from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
import sys
import warnings

import pandas as pd


def _configure_runtime_env() -> None:
    cache_root = Path("/tmp") / "morphseq_multiclass_bridge_cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))


_configure_runtime_env()
warnings.filterwarnings("ignore", category=FutureWarning, message=".*multi_class.*deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*liblinear.*multiclass classification.*deprecated.*")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "results/mcolon/20260326_pbx_crispant_analysis/scripts"))

from analyze.classification import run_classification
from phenotypic_positioning.multiclass import (
    build_multiclass_logit_vectors,
    build_multiclass_probability_vectors,
    prepare_multiclass_confusion_summary,
    prepare_multiclass_predictions,
    summarize_multiclass_centroids,
    summarize_probability_trajectories,
)
from phenotypic_positioning.plots import (
    build_color_palette,
    plot_multiclass_centroid_distances,
    plot_multiclass_confusion_snapshots,
    plot_multiclass_probability_trajectories,
    run_multiclass_embedding_qc,
)

from common import (
    ALL_EXPERIMENT_IDS,
    BRIDGE_EXPERIMENT_ID,
    BRIDGE_PLUS_WIK_AB_GENOTYPES,
    REPO_ROOT as _COMMON_REPO_ROOT,
    SHARED_GENOTYPES,
    load_bridge_ready_dataframe,
)

assert REPO_ROOT == _COMMON_REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bridge-enabled multiclass PBX positioning with 20251207_pbx.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results" / "mcolon" / "20260329_pbx_crispant_analysis_cont" / "results" / "phenotypic_positioning_multiclass_bridge_bin4_perm500",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=REPO_ROOT / "results" / "mcolon" / "20260329_pbx_crispant_analysis_cont" / "figures" / "phenotypic_positioning_multiclass_bridge_bin4_perm500",
    )
    parser.add_argument("--include-wik-ab", action="store_true", help="Include wik_ab from current experiments only.")
    parser.add_argument("--bin-width", type=float, default=4.0)
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--min-samples-per-group", type=int, default=3)
    parser.add_argument("--snapshot-times", nargs="+", type=float, default=[25.0, 55.0, 79.0])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    requested_genotypes = BRIDGE_PLUS_WIK_AB_GENOTYPES if args.include_wik_ab else SHARED_GENOTYPES
    df = load_bridge_ready_dataframe(genotypes=requested_genotypes)
    feature_cols = sorted(c for c in df.columns if c.startswith("z_mu_b_"))
    embryo_meta = (
        df[["embryo_id", "genotype", "experiment_id"]]
        .drop_duplicates()
        .rename(columns={"genotype": "true_label"})
        .reset_index(drop=True)
    )

    analysis = run_classification(
        df,
        class_col="genotype",
        id_col="embryo_id",
        time_col="stage_hpf_bridge",
        comparisons="all_vs_rest",
        features={"z_mu_b": feature_cols},
        n_jobs=int(args.n_jobs),
        n_permutations=int(args.n_permutations),
        bin_width=float(args.bin_width),
        min_samples_per_group=int(args.min_samples_per_group),
        random_state=42,
        verbose=False,
        save_multiclass_predictions=True,
    )

    class_labels = [label for label in requested_genotypes if f"pred_proba_{label}" in analysis.layers["multiclass_predictions"].columns]
    pred_df, prob_cols = prepare_multiclass_predictions(
        analysis.layers["multiclass_predictions"],
        embryo_meta=embryo_meta,
        class_labels=class_labels,
    )
    probability_vectors_df, vector_cols = build_multiclass_probability_vectors(pred_df, class_labels=class_labels)
    logit_vectors_df, logit_cols = build_multiclass_logit_vectors(pred_df, class_labels=class_labels)
    confusion_df = prepare_multiclass_confusion_summary(
        analysis.layers.get("confusion"),
        class_labels=class_labels,
        bin_width=float(args.bin_width),
    )
    centroids_df, centroid_distances_df = summarize_multiclass_centroids(probability_vectors_df, class_labels=class_labels)
    trajectory_df = summarize_probability_trajectories(probability_vectors_df, class_labels=class_labels)

    analysis.scores.to_csv(args.results_dir / "multiclass_score_summary.csv", index=False)
    pred_df.to_csv(args.results_dir / "multiclass_predictions.csv", index=False)
    probability_vectors_df.to_csv(args.results_dir / "multiclass_probability_vectors.csv", index=False)
    logit_vectors_df.to_csv(args.results_dir / "multiclass_logit_vectors.csv", index=False)
    confusion_df.to_csv(args.results_dir / "multiclass_confusion_summary.csv", index=False)
    centroids_df.to_csv(args.results_dir / "multiclass_centroids.csv", index=False)
    centroid_distances_df.to_csv(args.results_dir / "multiclass_centroid_distances.csv", index=False)
    trajectory_df.to_csv(args.results_dir / "multiclass_probability_trajectory_summary.csv", index=False)

    color_palette = build_color_palette(class_labels)
    run_multiclass_embedding_qc(
        probability_vectors_df,
        vector_cols=vector_cols,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        color_palette=color_palette,
        snapshot_times=args.snapshot_times,
        random_state=42,
    )
    plot_multiclass_probability_trajectories(trajectory_df, figures_dir=args.figures_dir, color_palette=color_palette)
    plot_multiclass_confusion_snapshots(confusion_df, figures_dir=args.figures_dir, class_labels=class_labels, snapshot_times=args.snapshot_times)
    plot_multiclass_centroid_distances(centroid_distances_df, figures_dir=args.figures_dir)

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "experiments": ALL_EXPERIMENT_IDS,
        "bridge_experiment_id": BRIDGE_EXPERIMENT_ID,
        "bridge_stage_rule": "predicted_stage_hpf if present else start_age_hpf + relative_time_s/3600",
        "bridge_window_hpf": [48.0, 80.0],
        "class_labels": class_labels,
        "include_wik_ab": bool(args.include_wik_ab),
        "results_dir": str(args.results_dir),
        "figures_dir": str(args.figures_dir),
        "bin_width": float(args.bin_width),
        "n_permutations": int(args.n_permutations),
        "n_jobs": int(args.n_jobs),
        "probability_columns": prob_cols,
        "logit_columns": logit_cols,
    }
    pd.Series(manifest, dtype="object").to_json(args.results_dir / "multiclass_manifest.json", indent=2)
    print(args.results_dir)
    print(args.figures_dir)


if __name__ == "__main__":
    main()
