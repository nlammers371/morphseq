"""
Comparison of WT / Het / Homo full time-matrix models.

This script now delegates all heavy lifting to the reusable utilities in
``src/analyze/difference_detection`` so that other projects can share the same
codepath.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analyze.difference_detection import (
    build_metric_matrices,
    load_time_matrix_results,
    plot_best_condition_map,
    plot_horizon_grid,
    summarise_bundles,
)
from analyze.difference_detection.time_matrix import align_matrix_times


# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "penetrance"
PLOT_DIR = BASE_DIR / "plots" / "penetrance"

MODEL_DATA_DIRS = {
    "WT": DATA_DIR / "wt_full_time_matrix",
    "Het": DATA_DIR / "het_full_time_matrix",
    "Homo": DATA_DIR / "homo_full_time_matrix",
}

OUTPUT_DATA_DIR = DATA_DIR / "3model_comparison"
OUTPUT_PLOT_DIR = PLOT_DIR / "3model_comparison"

# Consistent ordering for display
MODEL_ORDER = ["WT", "Het", "Homo"]
GENOTYPES = ["cep290_wildtype", "cep290_heterozygous", "cep290_homozygous"]
METRICS = ["mae", "r2", "error_std"]

# Training genotype lookup for LOEO tagging
MODEL_TRAINING_GENOTYPE = {
    "WT": "cep290_wildtype",
    "Het": "cep290_heterozygous",
    "Homo": "cep290_homozygous",
}


# ============================================================================
# Helpers
# ============================================================================


def prepare_nested_matrices(
    bundles,
    *,
    metric: str,
) -> dict:
    """
    Build and align matrices across models for the given metric.
    """

    nested = {model: {} for model in MODEL_ORDER}
    for model, bundle in bundles.items():
        if isinstance(bundle.data, dict):
            matrices = build_metric_matrices(bundle.data, metric=metric)
        else:
            matrices = {"all": build_metric_matrices(bundle.data, metric=metric)}
        nested[model] = matrices

    # Align times for each genotype across models
    for genotype in GENOTYPES:
        to_align = {
            model: matrices[genotype]
            for model, matrices in nested.items()
            if genotype in matrices
        }
        if not to_align:
            continue
        aligned = align_matrix_times(to_align, time_axis="both")
        for model, matrix in aligned.items():
            nested.setdefault(model, {})[genotype] = matrix

    return {model: nested[model] for model in MODEL_ORDER}


def ensure_output_dirs():
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Main analysis
# ============================================================================


def main() -> None:
    ensure_output_dirs()

    print("\nLoading model outputs...")
    bundles = load_time_matrix_results(
        root=".",  # absolute paths provided below
        conditions=MODEL_DATA_DIRS,
        group_col="genotype",
        filename="full_time_matrix_metrics.csv",
    )

    print("Computing summary statistics...")
    summary = summarise_bundles(
        bundles,
        metrics=METRICS,
        group_col="genotype",
        training_lookup=MODEL_TRAINING_GENOTYPE,
    )
    summary_path = OUTPUT_DATA_DIR / "3model_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  Saved summary → {summary_path}")

    print("\nGenerating horizon plots...")
    for metric in METRICS:
        matrices = prepare_nested_matrices(bundles, metric=metric)

        if not any(matrices[model] for model in MODEL_ORDER):
            print(f"  Skipping {metric} (no data)")
            continue

        fig = plot_horizon_grid(
            matrices,
            row_labels=[f"{model} model" for model in MODEL_ORDER],
            col_labels=[genotype.replace("_", " ") for genotype in GENOTYPES],
            metric=metric,
            clip_percentiles=(5, 95),
            title=f"{metric.upper()} comparison",
            save_path=OUTPUT_PLOT_DIR / f"3model_comparison_{metric}.png",
        )
        plt_close(fig)

        best_fig = plot_best_condition_map(
            matrices,
            row_labels=[f"{model} model" for model in MODEL_ORDER],
            col_labels=[genotype.replace("_", " ") for genotype in GENOTYPES],
            metric=metric,
            mode="min" if metric != "r2" else "max",
            title=f"Best performing model per cell ({metric.upper()})",
            save_path=OUTPUT_PLOT_DIR / f"best_model_{metric}.png",
        )
        plt_close(best_fig)

    print("\nComparison complete.")
    print(f"  Summary CSV: {summary_path}")
    print(f"  Plots directory: {OUTPUT_PLOT_DIR}")


def plt_close(fig):
    """
    Safe helper to close matplotlib figures without importing pyplot globally.
    """

    import matplotlib.pyplot as plt  # local import to avoid side effects during CLI import

    plt.close(fig)


if __name__ == "__main__":
    main()

