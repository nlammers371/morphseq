from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
import sys


def _configure_runtime_env() -> None:
    cache_root = Path("/tmp") / "morphseq_phase2_cache"
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))


_configure_runtime_env()

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from phenotypic_positioning.config import (  # noqa: E402
    DEFAULT_BIN_WIDTH,
    DEFAULT_EMBEDDING_PREFIX,
    DEFAULT_GENOTYPES,
    DEFAULT_K_NEIGHBORS,
    DEFAULT_N_BOOTSTRAPS,
    DEFAULT_N_SPLITS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_RESULTS_SUBDIR,
    DEFAULT_SNAPSHOT_TIMES,
    DEFAULT_TIME_COL,
    FIGURES_BASE,
    RESULTS_BASE,
)
from phenotypic_positioning.data import (  # noqa: E402
    aggregate_features_by_time,
    load_dataframe,
    pair_id,
    resolve_feature_columns,
)
from phenotypic_positioning.io import save_manifest  # noqa: E402
from phenotypic_positioning.pairwise import run_pairwise_support_analysis  # noqa: E402
from phenotypic_positioning.plots import (  # noqa: E402
    build_color_palette,
    plot_control_control_qc,
    plot_novelty_residual_scatter,
    plot_support_heatmap,
    plot_support_weight_distribution,
    run_embedding_comparison_qc,
)
from phenotypic_positioning.support import summarize_support_components  # noqa: E402
from phenotypic_positioning.vectors import build_vector_table, summarize_pairwise_support  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-two support-aware pairwise phenotypic positioning.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_BASE / DEFAULT_RESULTS_SUBDIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_BASE / DEFAULT_RESULTS_SUBDIR)
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--bin-width", type=float, default=DEFAULT_BIN_WIDTH)
    parser.add_argument("--time-col", default=DEFAULT_TIME_COL)
    parser.add_argument("--embedding-prefix", default=DEFAULT_EMBEDDING_PREFIX)
    parser.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS)
    parser.add_argument("--n-bootstraps", type=int, default=DEFAULT_N_BOOTSTRAPS)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--k-neighbors", type=int, default=DEFAULT_K_NEIGHBORS)
    parser.add_argument("--genotypes", nargs="+", default=DEFAULT_GENOTYPES)
    parser.add_argument("--snapshot-times", nargs="+", type=float, default=DEFAULT_SNAPSHOT_TIMES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models_dir = args.models_dir or (args.results_dir / "models")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and aggregating data ...")
    df, _ = load_dataframe(args.genotypes)
    feature_cols = resolve_feature_columns(df, args.embedding_prefix)
    df_binned = aggregate_features_by_time(
        df,
        feature_cols=feature_cols,
        time_col=args.time_col,
        bin_width=args.bin_width,
    )
    print(f"  {len(df)} raw rows -> {len(df_binned)} embryo-timebin rows")
    print(f"  {len(feature_cols)} embedding features")

    print("Running phase-two pairwise support analysis ...")
    axis_df, score_df, model_index_df, feature_support_df, coefficient_df = run_pairwise_support_analysis(
        df_binned,
        feature_cols=feature_cols,
        genotypes=args.genotypes,
        n_splits=args.n_splits,
        n_bootstraps=args.n_bootstraps,
        random_state=args.random_state,
        k_neighbors=args.k_neighbors,
        models_dir=models_dir,
    )

    axis_path = args.results_dir / "pairwise_axis_scores.csv"
    score_path = args.results_dir / "pairwise_score_summary.csv"
    model_index_path = args.results_dir / "pairwise_model_index.csv"
    feature_support_path = args.results_dir / "feature_support_summary.csv"
    coefficient_path = args.results_dir / "feature_model_coefficients.csv"
    axis_df.to_csv(axis_path, index=False)
    score_df.to_csv(score_path, index=False)
    model_index_df.to_csv(model_index_path, index=False)
    feature_support_df.to_csv(feature_support_path, index=False)
    coefficient_df.to_csv(coefficient_path, index=False)

    print("Building support-aware vectors ...")
    raw_vectors, pair_cols = build_vector_table(axis_df, value_col="position_logit_mean", fill_value=0.0)
    supported_vectors, _ = build_vector_table(axis_df, value_col="supported_position", fill_value=0.0)
    availability = axis_df.copy()
    availability["model_available_numeric"] = availability["model_available"].astype(float)
    availability_vectors, _ = build_vector_table(availability, value_col="model_available_numeric", fill_value=0.0)
    summary_df = summarize_pairwise_support(axis_df)
    component_summary_df = summarize_support_components(axis_df)

    raw_vectors.to_csv(args.results_dir / "raw_position_vectors.csv", index=False)
    supported_vectors.to_csv(args.results_dir / "support_position_vectors.csv", index=False)
    availability_vectors.to_csv(args.results_dir / "vector_availability.csv", index=False)
    summary_df.to_csv(args.results_dir / "support_metrics_summary.csv", index=False)
    component_summary_df.to_csv(args.results_dir / "support_component_summary.csv", index=False)

    color_palette = build_color_palette(args.genotypes)
    control_pair_id = pair_id("inj_ctrl", "wik_ab")
    important_pairs = [
        pair_id("inj_ctrl", "pbx4_crispant"),
        pair_id("pbx4_crispant", "pbx1b_pbx4_crispant"),
    ]

    print("Rendering QC figures ...")
    plot_control_control_qc(axis_df, figures_dir=args.figures_dir, pair_id=control_pair_id, color_palette=color_palette)
    plot_support_heatmap(summary_df, figures_dir=args.figures_dir)
    plot_novelty_residual_scatter(axis_df, figures_dir=args.figures_dir, pair_ids=important_pairs, color_palette=color_palette)
    plot_support_weight_distribution(axis_df, figures_dir=args.figures_dir, color_palette=color_palette)
    run_embedding_comparison_qc(
        raw_vectors,
        supported_vectors,
        pair_cols=pair_cols,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        color_palette=color_palette,
        snapshot_times=args.snapshot_times,
        random_state=args.random_state,
    )

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "results_dir": str(args.results_dir),
        "figures_dir": str(args.figures_dir),
        "models_dir": str(models_dir),
        "bin_width": float(args.bin_width),
        "time_col": args.time_col,
        "embedding_prefix": args.embedding_prefix,
        "feature_cols": feature_cols,
        "n_splits": int(args.n_splits),
        "n_bootstraps": int(args.n_bootstraps),
        "random_state": int(args.random_state),
        "k_neighbors": int(args.k_neighbors),
        "genotypes": list(args.genotypes),
        "pair_ids": pair_cols,
        "vector_fill_value": 0.0,
        "score_roles": {
            "in_pair_oof": "out-of-fold ensemble score for embryos in the comparison pair",
            "out_pair_probe": "full-fit ensemble probe score for embryos outside the comparison pair",
            "unavailable": "pair/time model could not be fit because the bin lacked enough samples",
        },
        "support_metrics": {
            "position_logit_mean": "ensemble-mean logit position on the pairwise axis",
            "position_logit_sd": "ensemble standard deviation of the logit score",
            "axis_projection": "projection onto the standardized centroid-difference axis",
            "axis_residual": "orthogonal distance from that axis",
            "knn_novelty": "mean distance to the pooled pair training cloud",
            "support_weight": "variance_weight * residual_weight * novelty_weight",
            "supported_position": "position_logit_mean multiplied by support_weight",
        },
        "feature_attribution_outputs": {
            "feature_support_summary.csv": "per pair/time/genotype summaries of per-feature residual and novelty contributions",
            "feature_model_coefficients.csv": "per pair/time bootstrap ensemble coefficient summaries for the pairwise logistic models",
            "support_component_summary.csv": "per pair/time/genotype summaries of variance, residual, novelty, and total support weights",
        },
    }
    save_manifest(args.results_dir / "phase2_manifest.json", manifest)

    print("Done.")
    print(f"  Axis scores: {axis_path}")
    print(f"  Score summary: {score_path}")
    print(f"  Model index: {model_index_path}")
    print(f"  Results: {args.results_dir}")
    print(f"  Figures: {args.figures_dir}")


if __name__ == "__main__":
    main()
