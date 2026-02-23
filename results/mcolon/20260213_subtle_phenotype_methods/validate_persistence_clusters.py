#!/usr/bin/env python
"""Post-hoc validation for embryo-first persistence clusters.

Consumes one persistence run directory (with `binned_data.tsv` and
`cohort_assignments.tsv`) and produces:
- faceted feature-over-time plots (cluster-colored views)
- cluster separability via run_classification_test
- cluster composition + within-cluster feature distribution checks
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency, kruskal

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analyze.difference_detection.classification_test_multiclass import run_classification_test  # noqa: E402
from analyze.viz.plotting import plot_feature_over_time  # noqa: E402


DEFAULT_PERSISTENCE_ROOT = Path(__file__).resolve().parent / "output" / "embryo_first_persistence"


def _resolve_latest_run(persistence_root: Path) -> Path:
    if not persistence_root.exists():
        raise FileNotFoundError(f"Persistence root does not exist: {persistence_root}")

    run_dirs = sorted([p for p in persistence_root.iterdir() if p.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in: {persistence_root}")
    return run_dirs[-1]


def _load_run_data(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    binned_path = run_dir / "binned_data.tsv"
    assignments_path = run_dir / "cohort_assignments.tsv"
    if not binned_path.exists() or not assignments_path.exists():
        raise FileNotFoundError(
            f"Missing required files in {run_dir}. Expected binned_data.tsv and cohort_assignments.tsv"
        )

    binned = pd.read_csv(binned_path, sep="\t")
    assignments = pd.read_csv(assignments_path, sep="\t")

    if "cluster" not in binned.columns:
        binned = binned.merge(assignments[["embryo_id", "cluster"]], on="embryo_id", how="left")
    return binned, assignments


def _pick_plot_features(binned: pd.DataFrame, explicit_features: Sequence[str] | None, max_features: int) -> List[str]:
    if explicit_features:
        missing = [c for c in explicit_features if c not in binned.columns]
        if missing:
            raise ValueError(f"Requested plot features not in binned_data.tsv: {missing}")
        return list(explicit_features)

    preferred = [
        "baseline_deviation_um_binned",
        "total_length_um_binned",
        "mean_curvature_per_um_binned",
        "std_curvature_per_um_binned",
        "surface_area_um_binned",
    ]
    available_preferred = [c for c in preferred if c in binned.columns]

    if available_preferred:
        return available_preferred[:max_features]

    fallback = [c for c in binned.columns if c.endswith("_binned")]
    if not fallback:
        raise ValueError("No *_binned feature columns found for plotting.")
    return fallback[:max_features]


def _pick_classification_features(binned: pd.DataFrame, max_features: int) -> List[str]:
    candidates = [c for c in binned.columns if c.endswith("_binned")]
    if not candidates:
        raise ValueError("No *_binned feature columns found for classification.")

    if len(candidates) <= max_features:
        return candidates

    # Prefer morphology columns when present, then append PCA/other binned columns.
    preferred_prefixes = [
        "baseline_deviation_um_binned",
        "total_length_um_binned",
        "mean_curvature_per_um_binned",
        "std_curvature_per_um_binned",
        "max_curvature_per_um_binned",
        "surface_area_um_binned",
        "area_um2_binned",
    ]
    selected = [c for c in preferred_prefixes if c in candidates]

    for c in candidates:
        if c not in selected:
            selected.append(c)
        if len(selected) >= max_features:
            break

    return selected[:max_features]


def _render_faceted_plots(binned: pd.DataFrame, plot_features: Sequence[str], output_dir: Path) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []

    plot_specs = [
        {
            "name": "cluster_by_genotype_and_experiment",
            "color_by": "cluster",
            "facet_row": "genotype_group" if "genotype_group" in binned.columns else "genotype",
            "facet_col": "experiment_id",
        },
        {
            "name": "cluster_by_phenotype_and_experiment",
            "color_by": "cluster",
            "facet_row": "phenotype_group" if "phenotype_group" in binned.columns else "phenotype",
            "facet_col": "experiment_id",
        },
        {
            "name": "genotype_by_cluster",
            "color_by": "genotype_group" if "genotype_group" in binned.columns else "genotype",
            "facet_row": None,
            "facet_col": "cluster",
        },
    ]

    for feature in plot_features:
        for spec in plot_specs:
            if spec["color_by"] not in binned.columns:
                continue
            if spec["facet_row"] is not None and spec["facet_row"] not in binned.columns:
                continue
            if spec["facet_col"] is not None and spec["facet_col"] not in binned.columns:
                continue

            out_path = output_dir / f"{feature}__{spec['name']}.png"
            plot_feature_over_time(
                binned,
                feature,
                time_col="time_bin",
                id_col="embryo_id",
                color_by=spec["color_by"],
                facet_row=spec["facet_row"],
                facet_col=spec["facet_col"],
                show_individual=False,
                show_error_band=True,
                trend_statistic="median",
                bin_width=2.0,
                backend="matplotlib",
                output_path=out_path,
                title=f"{feature}: {spec['name']}",
            )
            paths.append(str(out_path))

    return paths


def _run_cluster_classification(
    binned: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    n_permutations: int,
    n_jobs: int,
    bin_width: float,
) -> pd.DataFrame:
    work = binned.dropna(subset=["cluster", "time_bin", "embryo_id", *feature_cols]).copy()
    work = work[work["cluster"].astype(str) != ""]

    if work["cluster"].nunique() < 2:
        raise ValueError("Need at least 2 clusters for classification validation.")

    cls = run_classification_test(
        work,
        groupby="cluster",
        groups="all",
        reference="rest",
        features=list(feature_cols),
        time_col="time_bin",
        embryo_id_col="embryo_id",
        bin_width=bin_width,
        n_splits=3,
        n_permutations=n_permutations,
        n_jobs=n_jobs,
        min_samples_per_class=2,
        within_bin_time_stratification=True,
        within_bin_time_strata_width=0.5,
        max_comparisons=200,
        random_state=42,
        verbose=False,
    )

    return cls.comparisons.copy()


def _plot_classification_results(cls_df: pd.DataFrame, output_dir: Path) -> List[str]:
    if cls_df.empty:
        return []

    time_col = "time_bin_center" if "time_bin_center" in cls_df.columns else "time_bin"
    auroc_col = "auroc_obs" if "auroc_obs" in cls_df.columns else "auroc_observed"
    p_col = "pval" if "pval" in cls_df.columns else "p_value"
    class_col = "positive" if "positive" in cls_df.columns else "positive_class"

    if time_col not in cls_df.columns or auroc_col not in cls_df.columns or class_col not in cls_df.columns:
        return []

    classes = sorted(cls_df[class_col].dropna().astype(str).unique().tolist())
    if not classes:
        return []

    n = len(classes)
    fig, axes = plt.subplots(n, 1, figsize=(10, max(3, 2.6 * n)), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, cls in zip(axes, classes):
        sub = cls_df[cls_df[class_col].astype(str) == cls].copy().sort_values(time_col)
        if sub.empty:
            continue

        x = sub[time_col].to_numpy(dtype=float)
        y = sub[auroc_col].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=1.8, label=f"{cls} vs rest")

        if {"auroc_null_mean", "auroc_null_std"}.issubset(set(sub.columns)):
            null_mean = sub["auroc_null_mean"].to_numpy(dtype=float)
            null_std = sub["auroc_null_std"].to_numpy(dtype=float)
            lower = np.clip(null_mean - null_std, 0.0, 1.0)
            upper = np.clip(null_mean + null_std, 0.0, 1.0)
            ax.plot(x, null_mean, color="gray", linewidth=1.2, alpha=0.9, label="null mean")
            ax.fill_between(x, lower, upper, color="gray", alpha=0.2, label="null ±1 sd")

        if p_col in sub.columns:
            sig = sub[p_col].to_numpy(dtype=float) < 0.05
            if np.any(sig):
                ax.scatter(x[sig], y[sig], color="crimson", s=35, zorder=4, label="p < 0.05")

        ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_ylim(0.0, 1.02)
        ax.set_ylabel("AUROC")
        ax.set_title(f"One-vs-rest classification: {cls}", fontsize=10)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Time bin center (hpf)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=True)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_main = output_dir / "classification_ovr_auroc_over_time.png"
    fig.savefig(out_main, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Aggregate per-class summary for quick reading
    rows = []
    for cls in classes:
        sub = cls_df[cls_df[class_col].astype(str) == cls].copy()
        if sub.empty:
            continue
        rows.append(
            {
                "positive_class": cls,
                "n_time_bins": int(len(sub)),
                "median_auroc": float(sub[auroc_col].median()),
                "max_auroc": float(sub[auroc_col].max()),
                "min_p_value": float(sub[p_col].min()) if p_col in sub.columns else np.nan,
                "n_p_lt_0_05": int((sub[p_col] < 0.05).sum()) if p_col in sub.columns else 0,
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "classification_by_cluster_summary.tsv", sep="\t", index=False)
    return [str(out_main)]


def _composition_tests(assignments: pd.DataFrame) -> pd.DataFrame:
    embryo_df = assignments.drop_duplicates(subset=["embryo_id"]).copy()
    group_cols = [c for c in ["genotype_group", "genotype", "phenotype_group", "phenotype"] if c in embryo_df.columns]

    rows: List[Dict[str, object]] = []
    for group_col in group_cols:
        contingency = pd.crosstab(embryo_df["cluster"], embryo_df[group_col], dropna=False)
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            continue

        chi2, pval, dof, _ = chi2_contingency(contingency)
        rows.append(
            {
                "test_type": "global_composition",
                "group_col": group_col,
                "cluster": "ALL",
                "feature": "NA",
                "statistic": float(chi2),
                "p_value": float(pval),
                "dof": int(dof),
                "n_rows": int(len(embryo_df)),
                "n_levels": int(contingency.shape[1]),
            }
        )

    return pd.DataFrame(rows)


def _within_cluster_distribution_tests(binned: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    group_cols = [c for c in ["genotype_group", "genotype", "phenotype_group", "phenotype"] if c in binned.columns]

    rows: List[Dict[str, object]] = []
    for cluster, cdf in binned.groupby("cluster", dropna=False):
        for group_col in group_cols:
            valid = cdf[[group_col, *feature_cols]].copy()
            for feature in feature_cols:
                levels = []
                for _, sdf in valid.groupby(group_col, dropna=False):
                    arr = pd.to_numeric(sdf[feature], errors="coerce").dropna().to_numpy(dtype=float)
                    if arr.size >= 2:
                        levels.append(arr)

                if len(levels) < 2:
                    continue

                stat, pval = kruskal(*levels)
                rows.append(
                    {
                        "cluster": str(cluster),
                        "group_col": group_col,
                        "feature": feature,
                        "statistic": float(stat),
                        "p_value": float(pval),
                        "n_rows": int(len(valid)),
                        "n_groups_tested": int(len(levels)),
                    }
                )

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None, help="One persistence run directory")
    parser.add_argument("--persistence-root", type=Path, default=DEFAULT_PERSISTENCE_ROOT)
    parser.add_argument(
        "--plot-features",
        type=str,
        default="",
        help="Comma-separated list of features in binned_data.tsv to plot",
    )
    parser.add_argument("--max-plot-features", type=int, default=3)
    parser.add_argument("--max-classification-features", type=int, default=12)
    parser.add_argument("--classification-bin-width", type=float, default=2.0)
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir if args.run_dir else _resolve_latest_run(args.persistence_root)
    run_dir = run_dir.resolve()

    binned, assignments = _load_run_data(run_dir)

    explicit = [x.strip() for x in args.plot_features.split(",") if x.strip()] if args.plot_features else None
    plot_features = _pick_plot_features(binned, explicit_features=explicit, max_features=args.max_plot_features)
    cls_features = _pick_classification_features(binned, max_features=args.max_classification_features)

    output_dir = run_dir / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = _render_faceted_plots(binned, plot_features, output_dir=output_dir / "feature_over_time")

    cls_df = _run_cluster_classification(
        binned,
        cls_features,
        n_permutations=int(args.n_permutations),
        n_jobs=int(args.n_jobs),
        bin_width=float(args.classification_bin_width),
    )
    cls_path = output_dir / "classification_validation.tsv"
    cls_df.to_csv(cls_path, sep="\t", index=False)
    cls_plot_paths = _plot_classification_results(cls_df, output_dir=output_dir)

    composition_df = _composition_tests(assignments)
    comp_path = output_dir / "cluster_composition_tests.tsv"
    composition_df.to_csv(comp_path, sep="\t", index=False)

    dist_df = _within_cluster_distribution_tests(binned, cls_features)
    dist_path = output_dir / "cluster_distribution_tests.tsv"
    dist_df.to_csv(dist_path, sep="\t", index=False)

    auroc_col = "auroc_obs" if "auroc_obs" in cls_df.columns else "auroc_observed"
    p_col = "pval" if "pval" in cls_df.columns else "p_value"

    summary = pd.DataFrame(
        [
            {
                "run_dir": str(run_dir),
                "n_clusters": int(assignments["cluster"].nunique()),
                "n_embryos": int(assignments["embryo_id"].nunique()),
                "n_binned_rows": int(len(binned)),
                "n_plot_features": int(len(plot_features)),
                "n_classification_features": int(len(cls_features)),
                "classification_rows": int(len(cls_df)),
                "classification_median_auroc": float(cls_df[auroc_col].median()) if len(cls_df) else np.nan,
                "classification_max_auroc": float(cls_df[auroc_col].max()) if len(cls_df) else np.nan,
                "classification_min_p": float(cls_df[p_col].min()) if len(cls_df) else np.nan,
                "classification_n_p_lt_0_05": int((cls_df[p_col] < 0.05).sum()) if len(cls_df) else 0,
                "n_composition_tests": int(len(composition_df)),
                "n_distribution_tests": int(len(dist_df)),
                "n_distribution_p_lt_0_05": int((dist_df["p_value"] < 0.05).sum()) if len(dist_df) else 0,
                "n_plots_written": int(len(plot_paths)),
                "n_classification_plots_written": int(len(cls_plot_paths)),
            }
        ]
    )
    summary_path = output_dir / "cluster_validation_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)

    print(f"Run directory: {run_dir}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {cls_path}")
    print(f"Wrote: {comp_path}")
    print(f"Wrote: {dist_path}")
    print(f"Plots: {len(plot_paths)}")


if __name__ == "__main__":
    main()
