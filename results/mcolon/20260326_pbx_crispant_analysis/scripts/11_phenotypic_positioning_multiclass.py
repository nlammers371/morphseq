from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
import sys
import warnings


def _configure_runtime_env() -> None:
    cache_root = Path("/tmp") / "morphseq_multiclass_cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))


_configure_runtime_env()
warnings.filterwarnings(
    "ignore",
    message=".*'multi_class' was deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*Using the 'liblinear' solver for multiclass classification is deprecated.*",
    category=FutureWarning,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from analyze.classification import run_classification  # noqa: E402

from phenotypic_positioning.config import (  # noqa: E402
    DEFAULT_BIN_WIDTH,
    DEFAULT_EMBEDDING_PREFIX,
    DEFAULT_GENOTYPES,
    DEFAULT_MULTICLASS_MIN_SAMPLES_PER_GROUP,
    DEFAULT_MULTICLASS_N_JOBS,
    DEFAULT_MULTICLASS_N_PERMUTATIONS,
    DEFAULT_MULTICLASS_RESULTS_SUBDIR,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SNAPSHOT_TIMES,
    DEFAULT_TIME_COL,
    FIGURES_BASE,
    RESULTS_BASE,
)
from phenotypic_positioning.data import load_dataframe, resolve_feature_columns  # noqa: E402
from phenotypic_positioning.io import save_manifest  # noqa: E402
from phenotypic_positioning.multiclass import (  # noqa: E402
    build_multiclass_logit_vectors,
    build_multiclass_probability_vectors,
    prepare_multiclass_confusion_summary,
    prepare_multiclass_predictions,
    summarize_multiclass_centroids,
    summarize_probability_trajectories,
)
from phenotypic_positioning.plots import (  # noqa: E402
    build_color_palette,
    plot_multiclass_centroid_distances,
    plot_multiclass_confusion_snapshots,
    plot_multiclass_probability_trajectories,
    run_multiclass_embedding_qc,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multiclass phenotypic positioning for PBX crispant analysis.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_BASE / DEFAULT_MULTICLASS_RESULTS_SUBDIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_BASE / DEFAULT_MULTICLASS_RESULTS_SUBDIR)
    parser.add_argument("--bin-width", type=float, default=DEFAULT_BIN_WIDTH)
    parser.add_argument("--time-col", default=DEFAULT_TIME_COL)
    parser.add_argument("--embedding-prefix", default=DEFAULT_EMBEDDING_PREFIX)
    parser.add_argument("--n-permutations", type=int, default=DEFAULT_MULTICLASS_N_PERMUTATIONS)
    parser.add_argument("--min-samples-per-group", type=int, default=DEFAULT_MULTICLASS_MIN_SAMPLES_PER_GROUP)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_MULTICLASS_N_JOBS)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--genotypes", nargs="+", default=DEFAULT_GENOTYPES)
    parser.add_argument("--snapshot-times", nargs="+", type=float, default=DEFAULT_SNAPSHOT_TIMES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading multiclass data ...")
    df, embryo_meta = load_dataframe(args.genotypes)
    feature_cols = resolve_feature_columns(df, args.embedding_prefix)
    print(f"  {len(df)} raw rows")
    print(f"  {len(feature_cols)} embedding features")
    print(f"  {len(args.genotypes)} target genotypes")

    feature_set_name = args.embedding_prefix
    print("Running multiclass classification ...")
    analysis = run_classification(
        df,
        class_col="genotype",
        id_col="embryo_id",
        time_col=args.time_col,
        comparisons="all_vs_rest",
        features={feature_set_name: feature_cols},
        n_jobs=int(args.n_jobs),
        n_permutations=int(args.n_permutations),
        bin_width=float(args.bin_width),
        min_samples_per_group=int(args.min_samples_per_group),
        random_state=int(args.random_state),
        verbose=False,
        save_multiclass_predictions=True,
    )

    available_labels = [label for label in args.genotypes if f"pred_proba_{label}" in analysis.layers["multiclass_predictions"].columns]
    pred_df, prob_cols = prepare_multiclass_predictions(
        analysis.layers["multiclass_predictions"],
        embryo_meta=embryo_meta,
        class_labels=available_labels,
    )
    probability_vectors_df, vector_cols = build_multiclass_probability_vectors(pred_df, class_labels=available_labels)
    logit_vectors_df, logit_cols = build_multiclass_logit_vectors(pred_df, class_labels=available_labels)
    confusion_df = prepare_multiclass_confusion_summary(
        analysis.layers.get("confusion"),
        class_labels=available_labels,
        bin_width=float(args.bin_width),
    )
    centroids_df, centroid_distances_df = summarize_multiclass_centroids(probability_vectors_df, class_labels=available_labels)
    trajectory_df = summarize_probability_trajectories(probability_vectors_df, class_labels=available_labels)

    print("Writing multiclass tables ...")
    analysis.scores.to_csv(args.results_dir / "multiclass_score_summary.csv", index=False)
    pred_df.to_csv(args.results_dir / "multiclass_predictions.csv", index=False)
    probability_vectors_df.to_csv(args.results_dir / "multiclass_probability_vectors.csv", index=False)
    logit_vectors_df.to_csv(args.results_dir / "multiclass_logit_vectors.csv", index=False)
    confusion_df.to_csv(args.results_dir / "multiclass_confusion_summary.csv", index=False)
    centroids_df.to_csv(args.results_dir / "multiclass_centroids.csv", index=False)
    centroid_distances_df.to_csv(args.results_dir / "multiclass_centroid_distances.csv", index=False)
    trajectory_df.to_csv(args.results_dir / "multiclass_probability_trajectory_summary.csv", index=False)

    color_palette = build_color_palette(args.genotypes)
    print("Rendering multiclass figures ...")
    run_multiclass_embedding_qc(
        probability_vectors_df,
        vector_cols=vector_cols,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        color_palette=color_palette,
        snapshot_times=args.snapshot_times,
        random_state=args.random_state,
    )
    plot_multiclass_probability_trajectories(trajectory_df, figures_dir=args.figures_dir, color_palette=color_palette)
    plot_multiclass_confusion_snapshots(
        confusion_df,
        figures_dir=args.figures_dir,
        class_labels=available_labels,
        snapshot_times=args.snapshot_times,
    )
    plot_multiclass_centroid_distances(centroid_distances_df, figures_dir=args.figures_dir)

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "results_dir": str(args.results_dir),
        "figures_dir": str(args.figures_dir),
        "bin_width": float(args.bin_width),
        "time_col": args.time_col,
        "embedding_prefix": args.embedding_prefix,
        "feature_set_name": feature_set_name,
        "feature_cols": feature_cols,
        "n_permutations": int(args.n_permutations),
        "min_samples_per_group": int(args.min_samples_per_group),
        "n_jobs": int(args.n_jobs),
        "random_state": int(args.random_state),
        "genotypes": list(args.genotypes),
        "class_order": available_labels,
        "probability_columns": prob_cols,
        "logit_columns": logit_cols,
        "embedding_inputs": {
            "canonical": "multiclass_probability_vectors.csv",
            "aligned_umap_input_columns": prob_cols,
            "plain_umap_input_columns": prob_cols,
        },
        "outputs": {
            "multiclass_predictions.csv": "OOF multiclass prediction rows from the shared classifier",
            "multiclass_probability_vectors.csv": "Dense one-row-per-embryo-timebin probability vectors",
            "multiclass_logit_vectors.csv": "Logit transform of the probability vectors for diagnostics",
            "multiclass_confusion_summary.csv": "Per-time true-vs-predicted confusion proportions",
            "multiclass_centroids.csv": "Per-time genotype centroids in probability space",
            "multiclass_centroid_distances.csv": "Per-time pairwise distances between genotype centroids",
            "multiclass_probability_trajectory_summary.csv": "Per-time median class probabilities grouped by true genotype",
            "multiclass_aligned_umap_2d_coordinates.csv": "AlignedUMAP coordinates on the probability vectors",
            "multiclass_plain_umap_2d_coordinates.csv": "Plain UMAP coordinates on the probability vectors",
        },
    }
    save_manifest(args.results_dir / "multiclass_manifest.json", manifest)

    print("Done.")
    print(f"  Results: {args.results_dir}")
    print(f"  Figures: {args.figures_dir}")


if __name__ == "__main__":
    main()
